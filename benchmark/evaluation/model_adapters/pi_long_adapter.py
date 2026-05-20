"""
Pi-Long model adapter.
Chunk-based long-sequence 3D reconstruction method based on Pi3.
Split input images into chunks by chunk_size plus overlap, then run Pi3, 
thenvia SIM(3) aligns withoverlapregion, merge into complete sequence results.

Pi-Long outputs:
  - depth: (N, H, W) depth map (local_points[..., 2])
  - camera_poses: (N, 3, 4) cam2world pose
  - points: (N, H, W, 3) world-coordinate points
  - conf: (N, H, W) confidence (sigmoid)

Note: Pi3 camera_poses is already in cam2world format.
     conf requires sigmoid activation.
     chunk parameters are written directly into the adapter.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.models.omnivggt.utils.geometry import closed_form_inverse_se3
from benchmark.utils.sim3_align import (
    align_overlapping_chunks,
    accumulate_sim3_transforms,
    apply_sim3_to_points,
    apply_sim3_to_c2w,
)
from pi3.utils.geometry import recover_intrinsic_from_rays_d


@register_adapter("pi_long")
class PiLongAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.chunk_size = 60
        self.overlap = 30

    def name(self):
        return "Pi-Long"

    def load_model(self, checkpoint=None, device="cuda", **kwargs):
        self.device = device
        from pi3.models.pi3 import Pi3 as Pi3Model

        if checkpoint and os.path.isdir(checkpoint):
            self.model = Pi3Model.from_pretrained(checkpoint)
            print(f"[PiLongAdapter] Model loaded from {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = Pi3Model()
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[PiLongAdapter] Model loaded from {checkpoint}")
        else:
            self.model = Pi3Model.from_pretrained("yyfz233/Pi3")
            print("[PiLongAdapter] Model loaded from HuggingFace Hub")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[PiLongAdapter] Model on {device}")

    def _run_pi3_chunk(self, images_chunk):
        """Run Pi3 on a single chunk of images.

        Args:
            images_chunk: (chunk_N, 3, H, W) tensor [0, 1]

        Returns:
            dict with keys: c2w (chunk_N, 4, 4), points (chunk_N, H, W, 3),
                  local_points (chunk_N, H, W, 3), conf (chunk_N, H, W)
        """
        images_input = images_chunk.unsqueeze(0).to(self.device)  # (1, chunk_N, 3, H, W)

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            outputs = self.model(images_input)

        # camera_poses: (B, N, 4, 4) c2w
        c2w = outputs['camera_poses'][0].cpu()  # (chunk_N, 4, 4)
        # points: (B, N, H, W, 3) world coords
        points = outputs['points'][0].cpu()  # (chunk_N, H, W, 3)
        # local_points: (B, N, H, W, 3)
        local_points = outputs['local_points'][0].cpu()  # (chunk_N, H, W, 3)
        # conf: (B, N, H, W, 1) -> sigmoid -> (chunk_N, H, W)
        conf = torch.sigmoid(outputs['conf'][0, :, :, :, 0]).cpu()  # (chunk_N, H, W)

        # Align c2w to first frame: c2w_aligned[i] = inv(c2w[0]) @ c2w[i]
        c2w_np = c2w.numpy().astype(np.float64)
        c2w0_inv = closed_form_inverse_se3(c2w_np[0:1])[0]  # (4, 4)
        c2w_aligned = np.matmul(c2w0_inv, c2w_np)  # (chunk_N, 4, 4)

        # Also transform world points to first-frame-aligned coordinate system
        # points_aligned = c2w0_inv @ points (homogeneous)
        points_np = points.numpy().astype(np.float64)
        original_shape = points_np.shape  # (chunk_N, H, W, 3)
        pts_flat = points_np.reshape(-1, 3)  # (chunk_N*H*W, 3)
        R0_inv = c2w0_inv[:3, :3]
        t0_inv = c2w0_inv[:3, 3]
        pts_aligned = (R0_inv @ pts_flat.T).T + t0_inv
        points_aligned = pts_aligned.reshape(original_shape)

        torch.cuda.empty_cache()

        return {
            'c2w': c2w_aligned,                         # (chunk_N, 4, 4) float64
            'points': points_aligned,                    # (chunk_N, H, W, 3) float64
            'local_points': local_points.numpy(),        # (chunk_N, H, W, 3) float32
            'conf': conf.numpy(),                        # (chunk_N, H, W) float32
        }

    def predict(self, scene):
        """Run Pi-Long chunk-based inference.

        Pi3 input: imgs (B, N, 3, H, W) in [0, 1]
        Pi3 output: camera_poses (B, N, 4, 4) c2w, points (B, N, H, W, 3),
                     local_points (B, N, H, W, 3), conf (B, N, H, W, 1)
        """
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        chunk_size = self.chunk_size
        overlap = self.overlap

        # ---------- Single chunk: run Pi3 directly ----------
        if N <= chunk_size:
            return self._predict_single_chunk(images_raw, N, H, W)

        # ---------- Multi-chunk processing ----------
        # Build chunk indices
        step = chunk_size - overlap
        chunk_indices = []
        start = 0
        while start < N:
            end = min(start + chunk_size, N)
            chunk_indices.append((start, end))
            if end == N:
                break
            start += step
        num_chunks = len(chunk_indices)
        print(f"[PiLongAdapter] {N} frames -> {num_chunks} chunks "
              f"(chunk_size={chunk_size}, overlap={overlap})")

        # Run Pi3 on each chunk
        chunk_results = []
        for ci, (s_idx, e_idx) in enumerate(chunk_indices):
            print(f"[PiLongAdapter] Processing chunk {ci}/{num_chunks-1} "
                  f"[{s_idx}:{e_idx}] ({e_idx - s_idx} frames)")
            chunk_data = self._run_pi3_chunk(images_raw[s_idx:e_idx])
            chunk_results.append(chunk_data)

        # ---------- SIM(3) alignment between consecutive chunks ----------
        pairwise_transforms = []
        for ci in range(num_chunks - 1):
            cr1 = chunk_results[ci]
            cr2 = chunk_results[ci + 1]
            s1, e1 = chunk_indices[ci]
            s2, e2 = chunk_indices[ci + 1]
            actual_overlap = e1 - s2
            if actual_overlap <= 0:
                print(f"[WARNING] No overlap between chunk {ci} and {ci+1}")
                pairwise_transforms.append((1.0, np.eye(3), np.zeros(3)))
                continue

            # Overlap region in chunk1: last `actual_overlap` frames
            # Overlap region in chunk2: first `actual_overlap` frames
            pm1 = cr1['points'][-actual_overlap:]    # (overlap, H, W, 3)
            cf1 = cr1['conf'][-actual_overlap:]      # (overlap, H, W)
            pm2 = cr2['points'][:actual_overlap]     # (overlap, H, W, 3)
            cf2 = cr2['conf'][:actual_overlap]       # (overlap, H, W)

            print(f"[PiLongAdapter] Aligning chunk {ci} <-> {ci+1} "
                  f"(overlap={actual_overlap} frames)")
            s, R, t = align_overlapping_chunks(pm1, cf1, pm2, cf2)
            pairwise_transforms.append((s, R, t))

        # Accumulate transforms: cumulative[i] transforms chunk (i+1) -> chunk 0
        cumulative = accumulate_sim3_transforms(pairwise_transforms)

        # ---------- Apply SIM(3) and combine results ----------
        # First chunk: take all frames
        # Subsequent chunks: take frames after the overlap region (non-overlapping part)
        all_c2w = [chunk_results[0]['c2w']]                      # (chunk_N, 4, 4)
        all_points = [chunk_results[0]['points']]                # (chunk_N, H, W, 3)
        all_local_points = [chunk_results[0]['local_points']]    # (chunk_N, H, W, 3)
        all_conf = [chunk_results[0]['conf']]                    # (chunk_N, H, W)

        for ci in range(1, num_chunks):
            s, R, t_vec = cumulative[ci - 1]
            cr = chunk_results[ci]
            s_idx, e_idx = chunk_indices[ci]
            prev_e = chunk_indices[ci - 1][1]
            actual_overlap = prev_e - s_idx
            # Take only the non-overlapping part from this chunk
            keep_from = actual_overlap

            # Transform c2w
            c2w_transformed = apply_sim3_to_c2w(
                cr['c2w'][keep_from:], s, R, t_vec)
            all_c2w.append(c2w_transformed)

            # Transform world points
            pts_transformed = apply_sim3_to_points(
                cr['points'][keep_from:], s, R, t_vec)
            all_points.append(pts_transformed)

            # local_points and conf don't need SIM3 transform
            all_local_points.append(cr['local_points'][keep_from:])
            all_conf.append(cr['conf'][keep_from:])

        # Concatenate along frame dimension
        combined_c2w = np.concatenate(all_c2w, axis=0)              # (N, 4, 4)
        combined_points = np.concatenate(all_points, axis=0)        # (N, H, W, 3)
        combined_local = np.concatenate(all_local_points, axis=0)   # (N, H, W, 3)
        combined_conf = np.concatenate(all_conf, axis=0)            # (N, H, W)

        assert combined_c2w.shape[0] == N,\
            f"Frame count mismatch: got {combined_c2w.shape[0]}, expected {N}"

        # ---------- Build result dict ----------
        return self._build_result(combined_c2w, combined_points,
                                  combined_local, combined_conf, N, H, W)

    def _predict_single_chunk(self, images_raw, N, H, W):
        """Handle the single-chunk case (N <= chunk_size)."""
        chunk_data = self._run_pi3_chunk(images_raw)
        return self._build_result(chunk_data['c2w'], chunk_data['points'],
                                  chunk_data['local_points'], chunk_data['conf'],
                                  N, H, W)

    def _build_result(self, c2w, points, local_points, conf, N, H, W):
        """Build the result dict from combined outputs.

        Args:
            c2w: (N, 4, 4) cam2world float64
            points: (N, H, W, 3) world points
            local_points: (N, H, W, 3) camera-frame points
            conf: (N, H, W) confidence (already sigmoid)
            N, H, W: frame and spatial dimensions
        """
        result = {}

        # Depth: local_points[..., 2]
        result['pred_depth'] = local_points[:, :, :, 2].astype(np.float32)

        # Pose: c2w (N, 3, 4)
        result['pred_pose'] = c2w[:, :3, :4].astype(np.float32)
        w2c = closed_form_inverse_se3(c2w)  # (N, 4, 4)
        result['w2c_extrinsics'] = w2c[:, :3, :4].astype(np.float32)

        # Point cloud: filter by confidence
        all_pts = []
        conf_np = conf.astype(np.float32)
        points_np = points.astype(np.float32)
        for i in range(N):
            mask = conf_np[i] > 0.05
            dz = local_points[i, :, :, 2]
            mask = mask & (dz > 0) & np.isfinite(dz)
            pts = points_np[i][mask]
            if len(pts) > 0:
                all_pts.append(pts)
        if all_pts:
            result['pred_pointcloud'] = np.concatenate(
                all_pts, axis=0).astype(np.float32)

        # Confidence
        result['pred_confidence'] = conf_np

        # Intrinsic estimation from local_points rays
        local_pts_tensor = torch.from_numpy(local_points.astype(np.float32))
        rays_d = torch.nn.functional.normalize(local_pts_tensor, dim=-1)  # (N, H, W, 3)
        K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)  # (N, 3, 3)
        result['pred_intrinsic'] = K.cpu().numpy().astype(np.float32)
        print(f"[PiLongAdapter] Estimated first frame intrinsic:\n{K[0].cpu().numpy()}")

        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """GT c2w align to the first framethen convert to w2c, consistent with predict processing.

        aligned_c2w[i] = inv(c2w[0]) @ c2w[i], thenConvert to w2c.
        """
        gt_c2w = scene["extrinsic"]  # (N, 3, 4)
        N = gt_c2w.shape[0]
        # Expand to 4x4
        c2w_44 = np.zeros((N, 4, 4), dtype=np.float64)
        c2w_44[:, :3, :] = gt_c2w
        c2w_44[:, 3, 3] = 1.0
        # align to the first frame
        c2w0_inv = closed_form_inverse_se3(c2w_44[0:1])[0]  # (4, 4)
        c2w_aligned = np.matmul(c2w0_inv, c2w_44)  # (N, 4, 4)
        # Convert to w2c
        w2c = closed_form_inverse_se3(c2w_aligned)  # (N, 4, 4)
        return w2c[:, :3, :4].astype(np.float32)

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """Pi-Long visualization: use world points, sigmoid confidence filtering."""
        from benchmark.evaluation.metrics import unproject_to_pointcloud
        from benchmark.utils.visualization import (
            save_pointcloud_glb, _collect_colors,
        )

        scene_id = scene["scene_id"]
        images_raw = scene["images_raw"]
        pred_depth = predictions.get("pred_depth")
        pred_conf = predictions.get("pred_confidence")
        if pred_depth is None:
            return

        pred_poses = predictions.get("pred_pose", scene["extrinsic"])
        intrinsic = predictions.get("pred_intrinsic", scene["intrinsic"])

        pred_valid = (pred_depth > 0) & np.isfinite(pred_depth)
        if z_far is not None:
            pred_valid = pred_valid & (pred_depth < z_far)

        # sigmoid confidencepercentile filtering
        if pred_conf is not None and vis_conf_percent > 0:
            conf_valid = pred_conf[pred_valid]
            if len(conf_valid) > 0:
                threshold_val = np.percentile(conf_valid, vis_conf_percent)
                pred_valid = pred_valid & (pred_conf >= threshold_val)
                print(f"    Conf filter (sigmoid): percentile={vis_conf_percent}%, "
                      f"threshold={threshold_val:.4f}, "
                      f"range=[{conf_valid.min():.4f}, {conf_valid.max():.4f}]")

        pred_points = unproject_to_pointcloud(
            pred_depth, pred_poses, intrinsic, pred_valid)
        if len(pred_points) == 0:
            return

        pred_colors = _collect_colors(images_raw, pred_valid)
        suffix = "_pred_pred_pose"
        if vis_conf_percent > 0:
            suffix += f"_top{int(100 - vis_conf_percent)}pct"
        pred_glb_path = os.path.join(output_dir, f"{scene_id}{suffix}.glb")
        N, _, H, W = images_raw.shape
        save_pointcloud_glb(pred_points, pred_colors, pred_glb_path,
                            extrinsics=pred_poses, intrinsics=intrinsic,
                            image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
              f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")
