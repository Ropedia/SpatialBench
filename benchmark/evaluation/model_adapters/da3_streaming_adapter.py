"""
DA3-Streaming model adapter.
Chunk-based long-sequence 3D reconstruction method based on Depth Anything 3.
Split input images into chunks by chunk_size plus overlap, then run DA3 inference, 
thenvia SIM(3) align the overlapping-regionworld-coordinate points, merge into complete sequence results.

DA3-Streaming outputs:
  - depth: (N, H, W) depth map
  - extrinsics: (N, 3, 4) w2c -> Convert to cam2world
  - conf: (N, H, W) confidence
  - intrinsics: (N, 3, 3) intrinsics

Note: DA3 extrinsics is world-to-camera, predict() converts to cam2world.
     DA3 conf requires subtracting 1.0 (expp1 offset).
     world-coordinate pointsvia depth + intrinsics + extrinsics back-projection.
     chunk parameters are written directly into the adapter.
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


@register_adapter("da3_streaming")
class DA3StreamingAdapter(ModelAdapter):

    # checkpoint repo_id / path keyword -> model spec name
    _SIZE_MAP = {
        "SMALL": "da3-small",
        "BASE": "da3-base",
        "LARGE": "da3-large",
        "GIANT": "da3-giant",
    }

    # DA3-Streaming default chunk parameters (DA3 is lighter, chunk can be larger)
    DEFAULT_CHUNK_SIZE = 120
    DEFAULT_OVERLAP = 60

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.model_name = "da3"
        self.process_res = None
        self.chunk_size = self.DEFAULT_CHUNK_SIZE
        self.overlap = self.DEFAULT_OVERLAP
        self.ref_view_strategy = "saddle_balanced"

    def name(self):
        suffix = self.model_name.split('-')[-1].upper() if '-' in self.model_name else ""
        return f"DA3-Streaming-{suffix}" if suffix else "DA3-Streaming"

    @classmethod
    def _infer_model_name(cls, checkpoint):
        """Infer model spec from checkpoint path / repo_id."""
        if not checkpoint:
            return "da3-giant"
        key = checkpoint.upper()
        for token, name in cls._SIZE_MAP.items():
            if token in key:
                return name
        return "da3"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        self.model_name = self._infer_model_name(checkpoint)

        from depth_anything_3.api import DepthAnything3

        if checkpoint and os.path.isdir(checkpoint):
            self.model = DepthAnything3.from_pretrained(checkpoint)
            print(f"[DA3-Streaming] Model loaded: {self.model_name} from {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            self.model = DepthAnything3.from_pretrained(checkpoint)
            print(f"[DA3-Streaming] Model loaded: {self.model_name} from {checkpoint}")
        else:
            repo_id = checkpoint or "depth-anything/DA3-GIANT-1.1"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            self.model = DepthAnything3.from_pretrained(snapshot_dir)
            print(f"[DA3-Streaming] Model loaded: {self.model_name} from {repo_id} -> {snapshot_dir}")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[DA3-Streaming] {self.model_name} on {device}, "
              f"chunk_size={self.chunk_size}, overlap={self.overlap}")

    def supports_gt_prior(self):
        return {'pose': False, 'depth': False, 'intrinsic': False, 'partial': False}

    def predict(self, scene, gt_config=None):
        """Run DA3-Streaming inference.

        Split input images into chunks (with overlap), run on each chunk DA3 inference, 
        via SIM(3) aligns withoverlapregionworld-coordinate points, merge into complete sequence results.

        Args:
            scene: dict from BenchmarkDataset
            gt_config: optional (DA3-Streaming does not use GT prior)
        """
        from benchmark.utils.sim3_align import (
            align_overlapping_chunks,
            accumulate_sim3_transforms,
            apply_sim3_to_points,
            apply_sim3_to_c2w,
            depth_to_world_points,
        )

        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # DA3 inference() accepts a list of numpy arrays (H, W, 3) uint8
        all_images = []
        for i in range(N):
            img_np = (images_raw[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            all_images.append(img_np)

        # Automatically match the input resolution: take the longest side and round up to a 14 multiple
        PATCH_SIZE = 14
        if self.process_res is not None:
            process_res = self.process_res
        else:
            longest = max(H, W)
            process_res = ((longest + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

        chunk_size = self.chunk_size
        overlap = self.overlap

        # ---- single-chunk fast path ----
        if N <= chunk_size:
            return self._run_single_chunk(all_images, N, H, W, process_res)

        # ---- multi-chunk splitting ----
        chunks = self._build_chunks(N, chunk_size, overlap)
        print(f"[DA3-Streaming] {N} frames -> {len(chunks)} chunks "
              f"(chunk_size={chunk_size}, overlap={overlap})")

        # infer chunk by chunk
        chunk_results = []
        for ci, (start, end) in enumerate(chunks):
            print(f"  [Chunk {ci}] frames [{start}, {end})")
            image_list = all_images[start:end]

            with torch.no_grad():
                prediction = self.model.inference(
                    image=image_list,
                    process_res=process_res,
                    process_res_method="upper_bound_resize",
                    use_ray_pose=True,
                    ref_view_strategy=self.ref_view_strategy,
                    export_dir=None,
                    infer_gs=False,
                    align_to_input_ext_scale=False,
                )

            depth = prediction.depth.astype(np.float32)       # (M, H, W)
            conf = prediction.conf.astype(np.float32)          # (M, H, W)
            conf -= 1.0  # expp1 offset: [1, inf) -> [0, inf)
            w2c = prediction.extrinsics                        # (M, 4, 4)
            intrinsics = prediction.intrinsics.astype(np.float32)  # (M, 3, 3)

            # w2c (M, 3, 4)
            w2c_34 = w2c[:, :3, :].astype(np.float32)

            # Build world-coordinate points
            world_points = depth_to_world_points(depth, intrinsics, w2c_34)  # (M, H, W, 3)

            # w2c -> c2w (M, 4, 4)
            M = depth.shape[0]
            c2w_44 = np.zeros((M, 4, 4), dtype=np.float64)
            for i in range(M):
                R = w2c[i, :3, :3]
                t = w2c[i, :3, 3]
                R_inv = R.T
                t_inv = -R_inv @ t
                c2w_44[i, :3, :3] = R_inv
                c2w_44[i, :3, 3] = t_inv
                c2w_44[i, 3, 3] = 1.0

            chunk_results.append({
                'depth': depth,
                'conf': conf,
                'world_points': world_points,
                'c2w_44': c2w_44,
                'intrinsics': intrinsics,
            })

            torch.cuda.empty_cache()

        # ---- SIM(3) aligns with ----
        pairwise_transforms = []
        for ci in range(len(chunks) - 1):
            start_curr, end_curr = chunks[ci]
            start_next, end_next = chunks[ci + 1]
            overlap_size = end_curr - start_next

            if overlap_size <= 0:
                print(f"  [WARNING] No overlap between chunk {ci} and {ci+1}, "
                      "using identity transform")
                pairwise_transforms.append((1.0, np.eye(3), np.zeros(3)))
                continue

            # chunk_ci last overlap_size frames vs chunk_{ci+1} first overlap_size frames
            wp1 = chunk_results[ci]['world_points'][-overlap_size:]     # (O, H, W, 3)
            c1 = chunk_results[ci]['conf'][-overlap_size:]             # (O, H, W)
            wp2 = chunk_results[ci + 1]['world_points'][:overlap_size]  # (O, H, W, 3)
            c2 = chunk_results[ci + 1]['conf'][:overlap_size]          # (O, H, W)

            print(f"  [Align {ci}->{ci+1}] overlap={overlap_size} frames")
            s, R, t = align_overlapping_chunks(wp1, c1, wp2, c2)
            pairwise_transforms.append((s, R, t))

        # Accumulate transforms (chunk 0 unchanged, subsequent chunk gradually aligned to chunk 0 coordinate system)
        cumulative_transforms = accumulate_sim3_transforms(pairwise_transforms)

        # ---- merge results ----
            # Use chunk 0 directly; for subsequent chunks, drop the overlapping prefix and apply SIM3.
        final_depth = [chunk_results[0]['depth']]
        final_conf = [chunk_results[0]['conf']]
        final_c2w_44 = [chunk_results[0]['c2w_44']]
        final_intrinsics = [chunk_results[0]['intrinsics']]

        for ci in range(1, len(chunks)):
            start_curr, end_curr = chunks[ci]
            start_prev, end_prev = chunks[ci - 1]
            overlap_size = end_prev - start_curr

            # Remove the part overlapping with the previous chunk
            new_start = overlap_size if overlap_size > 0 else 0

            s, R, t_vec = cumulative_transforms[ci - 1]

            # depth: Depth needs scaling after SIM3
            depth_chunk = chunk_results[ci]['depth'][new_start:]
            final_depth.append(depth_chunk * s)

            final_conf.append(chunk_results[ci]['conf'][new_start:])
            final_intrinsics.append(chunk_results[ci]['intrinsics'][new_start:])

            # Align c2w
            c2w_chunk = chunk_results[ci]['c2w_44'][new_start:]
            c2w_aligned = apply_sim3_to_c2w(c2w_chunk, s, R, t_vec)
            final_c2w_44.append(c2w_aligned)

        all_depth = np.concatenate(final_depth, axis=0)        # (N, H, W)
        all_conf = np.concatenate(final_conf, axis=0)          # (N, H, W)
        all_c2w_44 = np.concatenate(final_c2w_44, axis=0)      # (N, 4, 4)
        all_intrinsics = np.concatenate(final_intrinsics, axis=0)  # (N, 3, 3)

        assert all_depth.shape[0] == N,\
            f"Frame count mismatch: got {all_depth.shape[0]}, expected {N}"

        # c2w (N, 4, 4) -> (N, 3, 4)
        pred_pose = all_c2w_44[:, :3, :].astype(np.float32)

        # w2c: derived from c2w
        w2c_list = []
        for i in range(N):
            R = all_c2w_44[i, :3, :3]
            t = all_c2w_44[i, :3, 3]
            R_inv = R.T
            t_inv = -R_inv @ t
            w2c = np.zeros((3, 4), dtype=np.float32)
            w2c[:3, :3] = R_inv
            w2c[:3, 3] = t_inv
            w2c_list.append(w2c)

        result = {
            'pred_depth': all_depth.astype(np.float32),
            'pred_pose': pred_pose,
            'w2c_extrinsics': np.stack(w2c_list).astype(np.float32),
            'pred_confidence': all_conf.astype(np.float32),
        }
        return result

    def _run_single_chunk(self, image_list, N, H, W, process_res):
        """single-chunk fast path: does not need SIM3 alignment."""
        with torch.no_grad():
            prediction = self.model.inference(
                image=image_list,
                process_res=process_res,
                process_res_method="upper_bound_resize",
                use_ray_pose=True,
                ref_view_strategy=self.ref_view_strategy,
                export_dir=None,
                infer_gs=False,
                align_to_input_ext_scale=False,
            )

        result = {}

        # depth
        if prediction.depth is not None:
            pred_depth = prediction.depth
            assert pred_depth.shape[1] == H and pred_depth.shape[2] == W,\
                f"DA3 depth resolution mismatch: pred {pred_depth.shape[1:]}, expected ({H}, {W})"
            result['pred_depth'] = pred_depth.astype(np.float32)

        # pose: w2c -> c2w
        if prediction.extrinsics is not None:
            w2c = prediction.extrinsics  # (N, 4, 4)
            c2w_list = []
            for i in range(N):
                R = w2c[i, :3, :3]
                t = w2c[i, :3, 3]
                R_inv = R.T
                t_inv = -R_inv @ t
                c2w = np.zeros((3, 4), dtype=np.float32)
                c2w[:3, :3] = R_inv
                c2w[:3, 3] = t_inv
                c2w_list.append(c2w)
            result['pred_pose'] = np.stack(c2w_list).astype(np.float32)
            result['w2c_extrinsics'] = w2c[:, :3, :].astype(np.float32)

        # confidence (expp1 offset)
        if prediction.conf is not None:
            pred_conf = prediction.conf.astype(np.float32)
            pred_conf -= 1.0
            if pred_conf.shape[1] != H or pred_conf.shape[2] != W:
                import cv2
                resized = np.stack([
                    cv2.resize(pred_conf[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    for i in range(N)
                ])
                pred_conf = resized
            result['pred_confidence'] = pred_conf

        torch.cuda.empty_cache()
        return result

    @staticmethod
    def _build_chunks(N, chunk_size, overlap):
        """Build the chunk index list [(start, end), ...].

        Args:
            N: total number of frames
            chunk_size: frames per chunk
            overlap: overlap frames between adjacent chunks

        Returns:
            list of (start, end) tuples
        """
        stride = chunk_size - overlap
        chunks = []
        start = 0
        while start < N:
            end = min(start + chunk_size, N)
            chunks.append((start, end))
            if end >= N:
                break
            start += stride
        return chunks

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """DA3-Streaming normalization: consistent with DA3 training."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """DA3-Streaming visualization: depth + pose back-projection."""
        from benchmark.evaluation.metrics import unproject_to_pointcloud
        from benchmark.utils.visualization import (
            save_pointcloud_glb, _collect_colors,
        )

        scene_id = scene["scene_id"]
        images_raw = scene["images_raw"]
        pred_depth = predictions.get("pred_depth")
        if pred_depth is None:
            return

        pred_poses = predictions.get("pred_pose", scene["extrinsic"])
        pred_conf = predictions.get("pred_confidence")
        intrinsic = scene["intrinsic"]

        pred_valid = (pred_depth > 0) & np.isfinite(pred_depth)
        if z_far is not None:
            pred_valid = pred_valid & (pred_depth < z_far)

        if pred_conf is not None and vis_conf_percent > 0:
            conf_valid = pred_conf[pred_valid]
            if len(conf_valid) > 0:
                threshold_val = np.percentile(conf_valid, vis_conf_percent)
                pred_valid = pred_valid & (pred_conf >= threshold_val)
                print(f"    Conf filter: percentile={vis_conf_percent}%, "
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
