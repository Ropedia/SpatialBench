"""
VGGT-Omega model adapter.
Use the VGGT-Omega forward() API for inference.

VGGT-Omega outputs:
  - pose_enc: (B, S, 9) = [T(3), quat(4), fov_h, fov_w], world-to-camera (OpenCV)
  - depth: (B, S, H, W, 1) depth map (exp activation)
  - depth_conf: (B, S, H, W) depth confidence (1 + exp, range [1, +∞))
  - camera_and_register_tokens: (B, S, 1+R, C) camera + register tokens
  - images: (B, S, 3, H, W) actual input resolution used by the model

Note:
  - input images [0, 1] range, the model applies ImageNet normalization
  - input resolution must be divisible by patch_size=16 (via resolution_override ensured)
  - pose_enc decoded extrinsics as world-to-camera, predict() converts to cam2world
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter


@register_adapter("vggt_omega")
class VGGTOmegaAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"

    def name(self):
        return "VGGT-Omega"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from vggt_omega.models import VGGTOmega

        self.model = VGGTOmega()

        if checkpoint and os.path.isfile(checkpoint):
            ckpt_path = checkpoint
        elif checkpoint and os.path.isdir(checkpoint):
            candidate = os.path.join(checkpoint, "vggt_omega_1b_512.pt")
            if not os.path.isfile(candidate):
                raise FileNotFoundError(
                    f"[VGGTOmegaAdapter] expected vggt_omega_1b_512.pt under {checkpoint}"
                )
            ckpt_path = candidate
        else:
            from benchmark.utils.hf_weights import ensure_hf_snapshot
            from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR
            repo_id = checkpoint or "facebook/VGGT-Omega"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            ckpt_path = os.path.join(snapshot_dir, "vggt_omega_1b_512.pt")
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"[VGGTOmegaAdapter] expected vggt_omega_1b_512.pt under {snapshot_dir}"
                )

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict, strict=True)
        print(f"[VGGTOmegaAdapter] load model weights from {ckpt_path}")

        self.model = self.model.to(device).eval()
        print(f"[VGGTOmegaAdapter] model loaded on {device}")

    def predict(self, scene):
        """Run VGGT-Omega inference.

        VGGT-Omega input: images (S, 3, H, W) in [0, 1] (the model applies ImageNet normalization)
        VGGT-Omega outputs: pose_enc (B, S, 9), depth (B, S, H, W, 1), depth_conf (B, S, H, W)
        """
        from vggt_omega.utils.pose_enc import encoding_to_camera

        # VGGT-Omega requires images in the [0, 1] range, does not need ImageNet normalization(the model applies)
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # Validate that the resolution is divisible by patch_size=16(ensured by resolution_override)
        patch_size = 16
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"VGGT-Omega requires H/W to be multiples of {patch_size}; got (H,W)=({H},{W})"
        )

        # VGGT-Omega expects (B, S, 3, H, W)
        images_input = images_raw.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        with torch.inference_mode():
            outputs = self.model(images_input)

        result = {}

        # depth: (B, S, H, W, 1) -> (N, H, W)
        if 'depth' in outputs:
            depth = outputs['depth'][0, :, :, :, 0]  # (S, H, W)
            result['pred_depth'] = depth.float().cpu().numpy().astype(np.float32)

        # pose + intrinsics: encoding_to_camera returns w2c (B, S, 3, 4) and K (B, S, 3, 3)
        if 'pose_enc' in outputs:
            pose_enc = outputs['pose_enc'].float()  # (B, S, 9)
            w2c_tensor, intrinsics_tensor = encoding_to_camera(
                pose_enc, image_size_hw=(H, W)
            )
            w2c = w2c_tensor[0].cpu().numpy()  # (S, 3, 4)

            # Convert to cam2world (benchmark benchmark standard format) - closed-form SE3 inverse
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
            result['w2c_extrinsics'] = w2c.astype(np.float32)
            result['pred_intrinsic'] = intrinsics_tensor[0].cpu().numpy().astype(np.float32)

        # confidence: depth_conf (B, S, H, W) - 1 + exp(...), range [1, +∞)
        if 'depth_conf' in outputs:
            conf = outputs['depth_conf'][0].float().cpu().numpy()  # (S, H, W)
            result['pred_confidence'] = conf.astype(np.float32)

        return result

    def supports_metric_depth(self):
        return False  # VGGT-Omega outputs relative depth

    def normalize_gt_poses(self, scene):
        """VGGT familynormalization: align to first camera + scale by avg point distance."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """VGGT-Omega visualization: depth + pose back-projection and expp1 conf percentile filtering."""
        from benchmark.utils.visualization import (
            save_pointcloud_glb, _collect_colors,
        )
        from benchmark.evaluation.metrics import unproject_to_pointcloud

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

        sky_mask = scene.get("sky_mask")
        if sky_mask is not None:
            pred_valid = pred_valid & ~sky_mask

        if pred_conf is not None and vis_conf_percent > 0:
            conf_valid = pred_conf[pred_valid]
            if len(conf_valid) > 0:
                threshold_val = np.percentile(conf_valid, vis_conf_percent)
                pred_valid = pred_valid & (pred_conf >= threshold_val)
                print(f"    Conf filter (expp1): percentile={vis_conf_percent}%, "
                      f"threshold={threshold_val:.4f}, "
                      f"range=[{conf_valid.min():.4f}, {conf_valid.max():.4f}]")

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
                            frustum_scale=0.02, image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
              f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")
