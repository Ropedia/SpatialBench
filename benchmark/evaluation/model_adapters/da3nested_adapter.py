"""
DA3Nested (Depth Anything 3 Nested) model adapter.
Use the DA3 inference() high-level API for inference, the model is NestedDepthAnything3Net.

DA3Nested architecture: dual-branch (anyview giant + metric large), outputs metric-scale depth.

DA3Nested outputs:
  - depth: (N, H, W) metric scale depth map
  - extrinsics: (N, 4, 4) world-to-camera pose
  - intrinsics: (N, 3, 3) intrinsics
  - conf: (N, H, W) confidence
  - is_metric: 1 (metric depth)
  - scale_factor: float (metric scale factor)

Note: DA3Nested extrinsics is world-to-camera, benchmark GT is cam2world, 
     predict() converts DA3 outputs to cam2world format.
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


@register_adapter("da3nested")
class DA3NestedAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.model_name = "da3nested-giant-large"
        self.process_res = None  # None = Automatically match the input resolution
        self.ref_view_strategy = "first"  # reference-frame selection strategy

    def name(self):
        return "DA3Nested"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from depth_anything_3.api import DepthAnything3

        if checkpoint and os.path.isdir(checkpoint):
            self.model = DepthAnything3.from_pretrained(checkpoint)
            print(f"[DA3NestedAdapter] Model loaded from {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            self.model = DepthAnything3.from_pretrained(checkpoint)
            print(f"[DA3NestedAdapter] Model loaded from {checkpoint}")
        else:
            repo_id = checkpoint or "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            self.model = DepthAnything3.from_pretrained(snapshot_dir)
            print(f"[DA3NestedAdapter] Model loaded from {repo_id} -> {snapshot_dir}")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[DA3NestedAdapter] Model on {device}")

    def predict(self, scene):
        """Run DA3Nested inference.

        Use the DA3 inference() API, input numpy image list.
        Convert outputs to benchmark benchmark standard format (cam2world).
        """
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # DA3 inference() accepts a list of numpy arrays (H, W, 3) uint8
        image_list = []
        for i in range(N):
            img_np = (images_raw[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            image_list.append(img_np)

        # Automatically match the input resolution: take the longest side and round up to a 14 multiple
        PATCH_SIZE = 14
        if self.process_res is not None:
            process_res = self.process_res
        else:
            longest = max(H, W)
            process_res = ((longest + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

        # Run inference
        prediction = self.model.inference(
            image=image_list,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
            use_ray_pose=True,
            infer_gs=False,
            ref_view_strategy=self.ref_view_strategy,
        )

        result = {}

        # depth: (N, H, W)
        if prediction.depth is not None:
            pred_depth = prediction.depth
            assert pred_depth.shape[1] == H and pred_depth.shape[2] == W,\
                f"DA3Nested depth resolution mismatch: pred {pred_depth.shape[1:]}, expected ({H}, {W})"
            result['pred_depth'] = pred_depth.astype(np.float32)

        # pose: DA3 outputs world-to-camera (N, 4, 4) -> Convert to cam2world (N, 3, 4)
        if prediction.extrinsics is not None:
            w2c = prediction.extrinsics  # (N, 4, 4)
            c2w_list = []
            for i in range(N):
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

        # confidence
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
        return True

    def normalize_gt_poses(self, scene):
        """DA3 normalization: consistent with training normalize_camera_extrinsics_and_points_batch consistent."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """DA3Nested visualization: depth + pose back-projection."""
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
