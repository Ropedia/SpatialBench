"""
VGGT-Long model adapter.
Chunk-based long-sequence 3D reconstruction method based on VGGT.
Split input images into chunks by chunk_size plus overlap, then run VGGT, 
thenvia SIM(3) aligns withoverlapregion, merge into complete sequence results.

VGGT-Long outputs:
  - depth: (N, H, W) depth map
  - extrinsics: (N, 3, 4) cam2world pose
  - world_points: (N, H, W, 3) world-coordinate points
  - world_points_conf: (N, H, W) confidence

Note: Loop closure is disabled (benchmark scenes are shorter and do not need it).
     chunk parameters are configured via yaml.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.utils.hf_weights import ensure_hf_snapshot
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR
from benchmark.utils.sim3_align import (
    align_overlapping_chunks,
    accumulate_sim3_transforms,
    apply_sim3_to_points,
    apply_sim3_to_c2w,
)


@register_adapter("vggt_long")
class VGGTLongAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        # ---- VGGT-Long specific parameters ----
        self.chunk_size = 60   # frames per chunk
        self.overlap = 30      # overlap frames between adjacent chunks

    def name(self):
        return "VGGT-Long"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from vggt.models.vggt import VGGT

        self.model = VGGT()

        if checkpoint and os.path.isdir(checkpoint):
            self.model = VGGT.from_pretrained(checkpoint)
            print("[VGGTLongAdapter] load model weights from local directory {}".format(checkpoint))
        elif checkpoint and os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
            print("[VGGTLongAdapter] load model weights from {}".format(checkpoint))
        else:
            repo_id = checkpoint or "facebook/VGGT-1B"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            self.model = VGGT.from_pretrained(snapshot_dir)
            print("[VGGTLongAdapter] load model weights from HuggingFace Hub -> {}".format(snapshot_dir))
        self.model = self.model.to(device)
        self.model.eval()
        print(f"[VGGTLongAdapter] model loaded on {device}, "
              f"chunk_size={self.chunk_size}, overlap={self.overlap}")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _run_vggt_chunk(self, images_chunk):
        """Run VGGT forward, on a single chunk and return numpy results.

        Args:
            images_chunk: (chunk_N, 3, H, W) tensor [0, 1]

        Returns:
            dict with keys: world_points (chunk_N, H, W, 3),
                            world_points_conf (chunk_N, H, W),
                            depth (chunk_N, H, W),
                            c2w (chunk_N, 4, 4),
                            w2c (chunk_N, 3, 4),
                            intrinsics (chunk_N, 3, 3)
        """
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        chunk_N, _, H, W = images_chunk.shape
        images_input = images_chunk.unsqueeze(0).to(self.device)  # (1, chunk_N, 3, H, W)

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            outputs = self.model(images_input)

        result = {}

        # depth: (B, S, H, W, 1) -> (chunk_N, H, W)
        depth = outputs['depth'][0, :, :, :, 0].cpu().numpy().astype(np.float32)
        result['depth'] = depth

        # world-coordinate points: (B, S, H, W, 3) -> (chunk_N, H, W, 3)
        world_points = outputs['world_points'][0].cpu().numpy().astype(np.float32)
        result['world_points'] = world_points

        # confidence: prefer world_points_conf, fallback depth_conf
        if 'world_points_conf' in outputs:
            conf = outputs['world_points_conf'][0].cpu().numpy().astype(np.float32)
        elif 'depth_conf' in outputs:
            conf = outputs['depth_conf'][0].cpu().numpy().astype(np.float32)
        else:
            conf = np.ones((chunk_N, H, W), dtype=np.float32)
        result['world_points_conf'] = conf

        # pose: pose_enc -> w2c -> c2w (4x4)
        pose_enc = outputs['pose_enc']  # (B, S, 9)
        w2c, pred_intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W)
        )
        w2c = w2c[0].cpu().numpy().astype(np.float32)  # (chunk_N, 3, 4)
        result['w2c'] = w2c
        result['intrinsics'] = pred_intrinsics[0].cpu().numpy().astype(np.float32)

        # w2c (3x4) -> c2w (4x4)
        c2w_list = []
        for i in range(chunk_N):
            R = w2c[i, :3, :3]
            t = w2c[i, :3, 3]
            R_inv = R.T
            t_inv = -R_inv @ t
            c2w_44 = np.eye(4, dtype=np.float32)
            c2w_44[:3, :3] = R_inv
            c2w_44[:3, 3] = t_inv
            c2w_list.append(c2w_44)
        result['c2w'] = np.stack(c2w_list).astype(np.float32)  # (chunk_N, 4, 4)

        return result

    def _split_into_chunks(self, N):
        """Compute the start/end index list for chunks.

        Returns:
            list of (start, end) tuples
        """
        chunks = []
        start = 0
        while start < N:
            end = min(start + self.chunk_size, N)
            chunks.append((start, end))
            if end >= N:
                break
            start = end - self.overlap
        return chunks

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, scene):
        """Run VGGT-Long inference.

        Split the input sequence into overlapping chunks, run per chunk VGGT, 
        via SIM(3) align the overlapping-region world_points, merge results.
        """
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        chunk_indices = self._split_into_chunks(N)
        num_chunks = len(chunk_indices)
        print(f"[VGGTLongAdapter] N={N}, chunk_size={self.chunk_size}, "
              f"overlap={self.overlap}, num_chunks={num_chunks}")

        # Single chunk: run directly with no alignment
        if num_chunks == 1:
            start, end = chunk_indices[0]
            chunk_result = self._run_vggt_chunk(images_raw[start:end])
            c2w_34 = chunk_result['c2w'][:, :3, :]  # (N, 3, 4)
            result = {
                'pred_depth': chunk_result['depth'],
                'pred_pose': c2w_34.astype(np.float32),
                'w2c_extrinsics': chunk_result['w2c'],
                'pred_intrinsic': chunk_result['intrinsics'],
                'pred_pointcloud': chunk_result['world_points'].reshape(-1, 3).astype(np.float32),
                'pred_confidence': chunk_result['world_points_conf'],
            }
            torch.cuda.empty_cache()
            return result

        # Multiple chunks: infer chunk by chunk
        chunk_results = []
        for ci, (start, end) in enumerate(chunk_indices):
            print(f"  Chunk {ci}/{num_chunks - 1}: frames [{start}, {end})")
            cr = self._run_vggt_chunk(images_raw[start:end])
            chunk_results.append(cr)
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # SIM(3) aligns with: overlapping regions of adjacent chunk pairs
        # ------------------------------------------------------------------
        pairwise_transforms = []  # Each pair (s, R, t): chunk_{i+1} -> chunk_i coordinate system

        for ci in range(num_chunks - 1):
            start_i, end_i = chunk_indices[ci]
            start_j, end_j = chunk_indices[ci + 1]
            overlap_size = end_i - start_j  # overlap frame count

            if overlap_size <= 0:
                print(f"  [WARNING] No overlap between chunk {ci} and {ci+1}, "
                      f"using identity transform")
                pairwise_transforms.append((1.0, np.eye(3), np.zeros(3)))
                continue

            # chunk_i overlapping part in: last overlap_size frames
            cr_i = chunk_results[ci]
            pts_i = cr_i['world_points'][-overlap_size:]    # (overlap, H, W, 3)
            conf_i = cr_i['world_points_conf'][-overlap_size:]  # (overlap, H, W)

            # chunk_j overlapping part in: first overlap_size frames
            cr_j = chunk_results[ci + 1]
            pts_j = cr_j['world_points'][:overlap_size]     # (overlap, H, W, 3)
            conf_j = cr_j['world_points_conf'][:overlap_size]  # (overlap, H, W)

            print(f"  Aligning chunk {ci+1} -> chunk {ci} "
                  f"(overlap={overlap_size} frames)")
            s, R, t = align_overlapping_chunks(pts_i, conf_i, pts_j, conf_j)
            pairwise_transforms.append((s, R, t))

        # Accumulate transforms: get each chunk -> chunk_0 coordinate-frame transform
        cumulative_transforms = accumulate_sim3_transforms(pairwise_transforms)

        # ------------------------------------------------------------------
        # Apply to chunk 1..K Apply SIM(3), then merge
        # ------------------------------------------------------------------
        # final result arrays
        final_depth = np.zeros((N, H, W), dtype=np.float32)
        final_c2w = np.zeros((N, 4, 4), dtype=np.float32)
        final_world_points = np.zeros((N, H, W, 3), dtype=np.float32)
        final_conf = np.zeros((N, H, W), dtype=np.float32)
        final_intrinsics = np.zeros((N, 3, 3), dtype=np.float32)
        frame_filled = np.zeros(N, dtype=bool)

        for ci, (start, end) in enumerate(chunk_indices):
            cr = chunk_results[ci]
            chunk_len = end - start

            # Apply to chunk > 0 Applyaccumulated SIM(3)
            if ci > 0:
                s, R, t = cumulative_transforms[ci - 1]
                cr['world_points'] = apply_sim3_to_points(
                    cr['world_points'], s, R, t
                ).astype(np.float32)
                cr['c2w'] = apply_sim3_to_c2w(
                    cr['c2w'].astype(np.float64), s, R, t
                ).astype(np.float32)
                # depth is scaled by scale
                cr['depth'] = (cr['depth'] * s).astype(np.float32)

            # Fill: for overlapping regions, prefer keeping predictions from the previous chunk
            for k in range(chunk_len):
                global_idx = start + k
                if not frame_filled[global_idx]:
                    final_depth[global_idx] = cr['depth'][k]
                    final_c2w[global_idx] = cr['c2w'][k]
                    final_world_points[global_idx] = cr['world_points'][k]
                    final_conf[global_idx] = cr['world_points_conf'][k]
                    final_intrinsics[global_idx] = cr['intrinsics'][k]
                    frame_filled[global_idx] = True

        # c2w (4x4) -> (N, 3, 4)
        pred_pose = final_c2w[:, :3, :].astype(np.float32)

        # w2c: derived from c2w
        w2c_list = []
        for i in range(N):
            R = final_c2w[i, :3, :3]
            t = final_c2w[i, :3, 3]
            R_inv = R.T
            t_inv = -R_inv @ t
            w2c = np.zeros((3, 4), dtype=np.float32)
            w2c[:3, :3] = R_inv
            w2c[:3, 3] = t_inv
            w2c_list.append(w2c)

        result = {
            'pred_depth': final_depth,
            'pred_pose': pred_pose,
            'w2c_extrinsics': np.stack(w2c_list).astype(np.float32),
            'pred_intrinsic': final_intrinsics,
            'pred_pointcloud': final_world_points.reshape(-1, 3).astype(np.float32),
            'pred_confidence': final_conf,
        }

        torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """VGGT-Long normalization: align to first camera + scale by avg point distance."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """VGGT-Long visualization: use world_points directly buildpoint cloud, expp1 confidence filtering."""
        from benchmark.utils.visualization import (
            save_pointcloud_glb, _collect_colors,
        )

        scene_id = scene["scene_id"]
        images_raw = scene["images_raw"]
        pred_depth = predictions.get("pred_depth")
        pred_conf = predictions.get("pred_confidence")
        if pred_depth is None:
            return

        N, H, W = pred_depth.shape

        pred_poses = predictions.get("pred_pose", scene["extrinsic"])
        pred_intrinsic = predictions.get("pred_intrinsic", scene["intrinsic"])

        pred_valid = (pred_depth > 0) & np.isfinite(pred_depth)
        if z_far is not None:
            pred_valid = pred_valid & (pred_depth < z_far)

        # expp1 confidencepercentile filtering
        if pred_conf is not None and vis_conf_percent > 0:
            conf_valid = pred_conf[pred_valid]
            if len(conf_valid) > 0:
                threshold_val = np.percentile(conf_valid, vis_conf_percent)
                pred_valid = pred_valid & (pred_conf >= threshold_val)
                print(f"    Conf filter (expp1): percentile={vis_conf_percent}%, "
                      f"threshold={threshold_val:.4f}, "
                      f"range=[{conf_valid.min():.4f}, {conf_valid.max():.4f}]")

        from benchmark.evaluation.metrics import unproject_to_pointcloud
        pred_points = unproject_to_pointcloud(
            pred_depth, pred_poses, pred_intrinsic, pred_valid)
        if len(pred_points) == 0:
            return

        pred_colors = _collect_colors(images_raw, pred_valid)
        suffix = "_pred_pred_pose"
        if vis_conf_percent > 0:
            suffix += f"_top{int(100 - vis_conf_percent)}pct"
        pred_glb_path = os.path.join(output_dir, f"{scene_id}{suffix}.glb")
        N_frames, _, H, W = images_raw.shape
        save_pointcloud_glb(pred_points, pred_colors, pred_glb_path,
                            extrinsics=pred_poses, intrinsics=pred_intrinsic,
                            frustum_scale=0.04, image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
              f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")
