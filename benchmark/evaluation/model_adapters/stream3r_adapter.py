"""
STream3R model adapter.
Use the STream3R forward() API for inference, supporting mode="causal" (streaming) and
mode="window" (sliding-window) two global-attention strategies, both from the upstream README:

    predictions = model(images, mode="causal" | "window" | "full")

Registers two adapters:
  - "stream3r_stream"  : mode="causal"
  - "stream3r_window"  : mode="window" (window_size=5, see
                          stream3r/models/components/aggregator/streamaggregator.py)

also keeps "stream3r" alias -> default causal, for compatibility with old scripts.

STream3R outputs:
  - pose_enc: (B, S, 9) = [T(3), quat(4), fov(2)], world-to-camera
  - depth: (B, S, H, W, 1) depth map
  - depth_conf: (B, S, H, W) depth confidence
  - world_points: (B, S, H, W, 3) world-coordinate points
  - world_points_conf: (B, S, H, W) point confidence

Note: extrinsics in STream3R pose_enc is world-to-camera (camera-from-world), 
     predict() converts to cam2world format.
     STream3R internally performs normalization, input should be [0, 1] range.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter


class _STream3RBase(ModelAdapter):
    # Overridden by subclasses: "causal" | "window" | "full"
    mode: str = "causal"
    # Overridden by subclasses: name shown in reports
    display_name: str = "STream3R"

    def __init__(self):
        self.model = None
        self.device = "cuda"

    def name(self):
        return self.display_name

    def load_model(self, checkpoint=None, device="cuda"):
        self.device = device
        from stream3r.models.stream3r import STream3R

        if checkpoint and os.path.isdir(checkpoint):
            self.model = STream3R.from_pretrained(checkpoint)
            print(f"[{self.display_name}Adapter] Model loaded from {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = STream3R()
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[{self.display_name}Adapter] Model loaded from {checkpoint}")
        else:
            self.model = STream3R.from_pretrained("yslan/STream3R")
            print(f"[{self.display_name}Adapter] Model loaded from HuggingFace Hub")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[{self.display_name}Adapter] Model on {device}, mode={self.mode}")

    def configure(self, **kwargs):
        # Allow YAML to explicitly override mode (causal | window | full)
        if "mode" in kwargs and kwargs["mode"] is not None:
            self.mode = kwargs.pop("mode")
        super().configure(**kwargs)

    def predict(self, scene):
        """Run STream3R inference.

        STream3R input: images (S, 3, H, W) or (B, S, 3, H, W) in [0, 1]
        STream3R outputs: pose_enc (B, S, 9), depth (B, S, H, W, 1), world_points (B, S, H, W, 3)
        """
        from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri

        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # STream3R expects (B, S, 3, H, W) or (S, 3, H, W)
        images_input = images_raw.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        with torch.no_grad():
            outputs = self.model(images_input, mode=self.mode)

        result = {}

        # depth: (B, S, H, W, 1) -> (N, H, W)
        if 'depth' in outputs:
            depth = outputs['depth'][0, :, :, :, 0]  # (S, H, W)
            result['pred_depth'] = depth.cpu().numpy().astype(np.float32)

        # pose: pose_enc (B, S, 9) -> extrinsics (B, S, 3, 4) world-to-camera
        if 'pose_enc' in outputs:
            pose_enc = outputs['pose_enc']  # (B, S, 9)
            w2c, pred_intrinsics = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=(H, W)
            )
            w2c = w2c[0].cpu().numpy()  # (S, 3, 4)

            # Convert to cam2world (benchmark benchmark standard format)
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
            result['w2c_extrinsics'] = w2c
            result['pred_pose'] = np.stack(c2w_list).astype(np.float32)
            result['pred_intrinsic'] = pred_intrinsics[0].cpu().numpy().astype(np.float32)  # (S, 3, 3)

        # World-coordinate point cloud: (B, S, H, W, 3)
        if 'world_points' in outputs:
            world_points = outputs['world_points'][0]  # (S, H, W, 3)
            wp_np = world_points.cpu().numpy().astype(np.float32)

            if 'world_points_conf' in outputs:
                wp_conf = outputs['world_points_conf'][0].cpu().numpy()  # (S, H, W)
            else:
                wp_conf = np.ones((N, H, W), dtype=np.float32)

            all_points = []
            for i in range(N):
                mask = wp_conf[i] > 1.0  # expp1 activation, >1 indicates confidence
                pts = wp_np[i][mask]
                if len(pts) > 0:
                    all_points.append(pts)

            if all_points:
                result['pred_pointcloud'] = np.concatenate(all_points, axis=0).astype(np.float32)

        # confidence: prefer world_points_conf, fallback depth_conf
        if 'world_points_conf' in outputs:
            conf = outputs['world_points_conf'][0].cpu().numpy()  # (S, H, W)
            result['pred_confidence'] = conf.astype(np.float32)
        elif 'depth_conf' in outputs:
            conf = outputs['depth_conf'][0].cpu().numpy()  # (S, H, W)
            result['pred_confidence'] = conf.astype(np.float32)

        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """VGGT normalization: align to first camera + scale by avg point distance.

        consistent with training normalize_camera_extrinsics_and_points_batch consistent.
        """
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """STream3R visualization: use world_points directly buildpoint cloud, expp1 confidence filtering."""
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
                            image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
              f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")


@register_adapter("stream3r_stream")
@register_adapter("stream3r")  # compatibility alias: default causal
class STream3RStreamAdapter(_STream3RBase):
    mode = "causal"
    display_name = "STream3R-Stream"


@register_adapter("stream3r_window")
class STream3RWindowAdapter(_STream3RBase):
    mode = "window"
    display_name = "STream3R-Window"
