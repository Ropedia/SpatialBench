"""
OmniVGGT model adapter.
Use the OmniVGGT inference() API for inference.

OmniVGGT outputs:
  - pose_enc: (B, S, 9) = [T(3), quat(4), fov(2)], world-to-camera
  - depth: (B, S, 1, H, W) depth map
  - depth_conf: (B, S, 1, H, W) depth confidence
  - world_points: (B, S, 3, H, W) world-coordinate points
  - world_points_conf: (B, S, 1, H, W) point confidence

Note: extrinsics in OmniVGGT pose_enc is world-to-camera (camera-from-world), 
     predict() converts to cam2world format.
     OmniVGGT internally applies ImageNet normalization, input should be [0, 1] range.
     OmniVGGT supports injecting GT camera/depth auxiliary information during inference (via camera_gt_index / depth_gt_index).
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter


@register_adapter("omnivggt")
class OmniVGGTAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"

    def name(self):
        return "OmniVGGT"

    def load_model(self, checkpoint=None, device="cuda"):
        self.device = device
        from omnivggt.models.omnivggt import OmniVGGT

        if checkpoint and os.path.isdir(checkpoint):
            self.model = OmniVGGT.from_pretrained(checkpoint)
            print(f"[OmniVGGTAdapter] Model loaded from {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = OmniVGGT()
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[OmniVGGTAdapter] Model loaded from {checkpoint}")
        else:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            ckpt_path = hf_hub_download("Livioni/OmniVGGT", filename="OmniVGGT.safetensors")
            state_dict = load_file(ckpt_path)
            self.model = OmniVGGT()
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[OmniVGGTAdapter] Model loaded from HuggingFace Hub ({ckpt_path})")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[OmniVGGTAdapter] Model on {device}")

    def supports_gt_prior(self):
        return {'pose': True, 'depth': True, 'intrinsic': True, 'partial': True}

    def predict(self, scene, gt_config=None):
        """Run OmniVGGT inference.

        OmniVGGT input: images (B, S, 3, H, W) in [0, 1] (the model applies ImageNet normalization)
        OmniVGGT outputs: pose_enc (B, S, 9), depth (B, S, 1, H, W), world_points (B, S, 3, H, W)

        Args:
            scene: dict from BenchmarkDataset
            gt_config: optional dict with keys:
                use_pose, use_depth, use_intrinsic (bool)
                gt_frame_indices (list[int])
        """
        from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

        # OmniVGGT requires images in the [0, 1] range
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # OmniVGGT expects (B, S, 3, H, W)
        images_input = images_raw.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        # ---- Build GT prior input ----
        from omnivggt.utils.geometry import closed_form_inverse_se3

        camera_gt_index = []
        depth_gt_index = []

        if gt_config and any(gt_config.get(k) for k in ('use_pose', 'use_depth', 'use_intrinsic')):
            # OmniVGGT couples pose + intrinsic: camera_gt_index controls both
            if gt_config.get('use_pose') or gt_config.get('use_intrinsic'):
                camera_gt_index = list(gt_config.get('camera_gt_indices',
                                                      gt_config.get('gt_frame_indices', [])))

            if gt_config.get('use_depth'):
                depth_gt_index = list(gt_config.get('depth_gt_indices',
                                                     gt_config.get('gt_frame_indices', [])))

        # Build extrinsics tensor (B, S, 3, 4) w2c
        # OmniVGGT requires world-to-camera format, use closed_form_inverse_se3 consistent with upstream
        # GT frame positions are filled with true w2c, non-GT frames are filled with zeros (the model only index_selects GT frames)
        gt_c2w = scene['extrinsic']  # (N, 3, 4) cam2world
        if camera_gt_index:
            # Expand to (N, 4, 4) for closed_form_inverse_se3
            c2w_44 = np.zeros((N, 4, 4), dtype=np.float32)
            c2w_44[:, :3, :] = gt_c2w
            c2w_44[:, 3, 3] = 1.0
            w2c_44 = closed_form_inverse_se3(c2w_44)  # (N, 4, 4)
            w2c = w2c_44[:, :3, :].astype(np.float32)  # (N, 3, 4)
        else:
            w2c = np.zeros((N, 3, 4), dtype=np.float32)
        extrinsics_input = torch.from_numpy(w2c).unsqueeze(0).to(self.device)  # (1, N, 3, 4)

        # Build intrinsics tensor (B, S, 3, 3)
        # GT frames are filled with true intrinsics, non-GT frames are filled with zeros (the model only index_selects GT frames)
        if camera_gt_index:
            intrinsics_input = torch.from_numpy(
                scene['intrinsic'].astype(np.float32)
            ).unsqueeze(0).to(self.device)  # (1, N, 3, 3)
        else:
            intrinsics_input = torch.zeros(1, N, 3, 3, device=self.device)

        # Build depth tensor (B, S, H, W, 1) and mask (B, S, H, W)
        if depth_gt_index:
            depth_input = torch.from_numpy(
                scene['depth'].astype(np.float32)
            ).unsqueeze(0).unsqueeze(-1).to(self.device)  # (1, N, H, W, 1)
            mask_input = torch.from_numpy(
                scene['valid_mask'].astype(np.float32)
            ).unsqueeze(0).to(self.device)  # (1, N, H, W)
        else:
            depth_input = torch.zeros(1, N, H, W, 1, device=self.device)
            mask_input = torch.zeros(1, N, H, W, device=self.device)

        if camera_gt_index or depth_gt_index:
            print(f"    [OmniVGGT] GT prior: camera_gt_index={camera_gt_index}, "
                  f"depth_gt_index={depth_gt_index}")

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            outputs = self.model.inference(
                images=images_input,
                extrinsics=extrinsics_input,
                intrinsics=intrinsics_input,
                depth=depth_input,
                mask=mask_input,
                depth_gt_index=depth_gt_index,
                camera_gt_index=camera_gt_index,
            )

        result = {}

        # depth: OmniVGGT outputs (B, S, H, W, 1), same as VGGT
        if 'depth' in outputs:
            depth = outputs['depth'][0, :, :, :, 0].cpu().numpy().astype(np.float32)  # (S, H, W)
            result['pred_depth'] = depth

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

        # World-coordinate point cloud: OmniVGGT outputs (B, S, 3, H', W'), convert to (M, 3)
        if 'world_points' in outputs:
            world_points = outputs['world_points'][0]  # (S, 3, H', W')
            if isinstance(world_points, torch.Tensor):
                world_points = world_points.permute(0, 2, 3, 1).cpu().numpy()  # (S, H', W', 3)
            result['pred_pointcloud'] = world_points.reshape(-1, 3).astype(np.float32)

        # confidence: prefer world_points_conf, fallback depth_conf
        # expp1 activation, range [1, +∞), larger values are more reliable
        if 'world_points_conf' in outputs:
            conf = outputs['world_points_conf'][0]
            if isinstance(conf, torch.Tensor):
                conf = conf.cpu().numpy()
            if conf.ndim == 4:
                conf = conf[:, 0]  # (S, 1, H', W') -> (S, H', W')
            result['pred_confidence'] = conf.astype(np.float32)
        elif 'depth_conf' in outputs:
            conf = outputs['depth_conf'][0]
            if isinstance(conf, torch.Tensor):
                conf = conf.cpu().numpy()
            if conf.ndim == 4:
                conf = conf[:, 0]
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
        """OmniVGGT visualization: use world_points directly buildpoint cloud, expp1 confidence filtering."""
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

        sky_mask = scene.get("sky_mask")
        if sky_mask is not None:
            pred_valid = pred_valid & ~sky_mask

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
                            frustum_scale=0.02, image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
              f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")
