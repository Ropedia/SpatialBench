"""
DA3 (Depth Anything 3) model adapter.
Runs inference via DA3's high-level inference() API.

DA3 outputs:
  - depth: (N, H, W) depth maps
  - extrinsics: (N, 4, 4) world-to-camera poses
  - intrinsics: (N, 3, 3) intrinsics
  - conf: (N, H, W) confidence

Note: DA3 extrinsics are world-to-camera, while the benchmark GT is cam2world,
     so predict() converts DA3's output into cam2world format.
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


@register_adapter("da3")
class DA3Adapter(ModelAdapter):

    # checkpoint repo_id / path keyword -> model spec name
    _SIZE_MAP = {
        "SMALL": "da3-small",
        "BASE": "da3-base",
        "LARGE": "da3-large",
        "GIANT": "da3-giant",
    }

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.model_name = "da3"
        self.process_res = None  # None = automatically match input resolution
        self.ref_view_strategy = "first"  # reference-frame selection strategy

    def name(self):
        return f"DA3-{self.model_name.split('-')[-1].upper()}" if '-' in self.model_name else "DA3"

    @classmethod
    def _infer_model_name(cls, checkpoint):
        """Infer the model spec from the checkpoint path / repo_id."""
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
            print(f"[DA3Adapter] Model loaded: {self.model_name} from {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            self.model = DepthAnything3.from_pretrained(checkpoint)
            print(f"[DA3Adapter] Model loaded: {self.model_name} from {checkpoint}")
        else:
            repo_id = checkpoint or "depth-anything/DA3-GIANT-1.1"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            self.model = DepthAnything3.from_pretrained(snapshot_dir)
            print(f"[DA3Adapter] Model loaded: {self.model_name} from {repo_id} -> {snapshot_dir}")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[DA3Adapter] {self.model_name} on {device}")

    def supports_gt_prior(self):
        return {'pose': True, 'depth': False, 'intrinsic': True, 'partial': False}

    def predict(self, scene, gt_config=None):
        """Run DA3 inference.

        Uses DA3's inference() API with a list of numpy images as input.
        Outputs are converted to the benchmark standard format (cam2world).

        Args:
            scene: dict from BenchmarkDataset
            gt_config: optional dict with keys:
                use_pose, use_depth, use_intrinsic (bool)
                gt_frame_indices (list[int])
        """
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # DA3 inference() expects a list of numpy arrays (H, W, 3) uint8
        image_list = []
        for i in range(N):
            img_np = (images_raw[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            image_list.append(img_np)

        # Auto-match input resolution: take the longest side, round up to a multiple of 14
        PATCH_SIZE = 14
        if self.process_res is not None:
            process_res = self.process_res
        else:
            longest = max(H, W)
            process_res = ((longest + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

        # ---- Build GT prior inputs (all-or-nothing) ----
        extrinsics_arg = None
        intrinsics_arg = None

        has_camera_gt = gt_config and (gt_config.get('use_pose') or gt_config.get('use_intrinsic'))
        has_depth_gt = gt_config and gt_config.get('use_depth')
        if has_camera_gt or has_depth_gt:

            if gt_config.get('use_pose'):
                # DA3 expects w2c (N, 4, 4) numpy
                # DA3's inference() internally calls _normalize_extrinsics, so here we only do c2w -> w2c
                gt_c2w = scene['extrinsic']  # (N, 3, 4) c2w
                w2c_44 = np.zeros((N, 4, 4), dtype=np.float64)
                for i in range(N):
                    R, t = gt_c2w[i, :3, :3], gt_c2w[i, :3, 3]
                    w2c_44[i, :3, :3] = R.T
                    w2c_44[i, :3, 3] = -R.T @ t
                    w2c_44[i, 3, 3] = 1.0
                extrinsics_arg = w2c_44
                print(f"    [DA3] GT prior: pose=ALL {N} frames")

            if gt_config.get('use_intrinsic'):
                intrinsics_arg = scene['intrinsic']  # (N, 3, 3) numpy
                print(f"    [DA3] GT prior: intrinsic=ALL {N} frames")

        # Run inference (no file export)
        prediction = self.model.inference(
            image=image_list,
            extrinsics=extrinsics_arg,
            intrinsics=intrinsics_arg,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
            use_ray_pose=True,
            infer_gs=False,
            ref_view_strategy=self.ref_view_strategy,
            align_to_input_ext_scale = False
        )

        result = {}

        # Depth: (N, H, W)
        if prediction.depth is not None:
            pred_depth = prediction.depth  # (N, H_proc, W_proc)
            assert pred_depth.shape[1] == H and pred_depth.shape[2] == W, \
                f"DA3 depth resolution mismatch: pred {pred_depth.shape[1:]}, expected ({H}, {W})"
            result['pred_depth'] = pred_depth.astype(np.float32)

        # Pose: DA3 outputs world-to-camera (N, 4, 4) -> convert to cam2world (N, 3, 4)
        if prediction.extrinsics is not None:
            w2c = prediction.extrinsics  # (N, 4, 4)
            c2w_list = []
            for i in range(N):
                # Inverse: cam2world = inv(world2camera)
                w2c_i = w2c[i]
                R = w2c_i[:3, :3]
                t = w2c_i[:3, 3]
                R_inv = R.T
                t_inv = -R_inv @ t
                c2w = np.zeros((3, 4), dtype=np.float32)
                c2w[:3, :3] = R_inv
                c2w[:3, 3] = t_inv
                c2w_list.append(c2w)
            result['pred_pose'] = np.stack(c2w_list).astype(np.float32)
            result['w2c_extrinsics'] = w2c[:, :3, :]  # (N, 3, 4)

        # Confidence
        if prediction.conf is not None:
            pred_conf = prediction.conf
            if pred_conf.shape[1] != H or pred_conf.shape[2] != W:
                import cv2
                resized = np.stack([
                    cv2.resize(pred_conf[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    for i in range(N)
                ])
                pred_conf = resized
            result['pred_confidence'] = pred_conf.astype(np.float32)

        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """DA3 normalization: matches normalize_camera_extrinsics_and_points_batch used during training."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """DA3 visualization: pure depth + pose unprojection, no direct point cloud output."""
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
        # DA3 does not output intrinsics; use GT intrinsics
        intrinsic = scene["intrinsic"]

        pred_valid = (pred_depth > 0) & np.isfinite(pred_depth)
        if z_far is not None:
            pred_valid = pred_valid & (pred_depth < z_far)

        sky_mask = scene.get("sky_mask")
        if sky_mask is not None:
            pred_valid = pred_valid & ~sky_mask

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
                            frustum_scale=0.02, image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
              f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")
