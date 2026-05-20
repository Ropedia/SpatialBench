"""
VGGT model adapter.
Runs inference via VGGT's forward() API.

VGGT outputs:
  - pose_enc: (B, S, 9) = [T(3), quat(4), fov(2)], world-to-camera
  - depth: (B, S, H, W, 1) depth maps
  - depth_conf: (B, S, H, W) depth confidence
  - world_points: (B, S, H, W, 3) world-coordinate points

Note: the extrinsics in VGGT's pose_enc are world-to-camera (camera-from-world);
     predict() converts them to cam2world format.
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


@register_adapter("vggt")
class VGGTAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"

    def name(self):
        return "VGGT"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from vggt.models.vggt import VGGT

        self.model = VGGT()

        if checkpoint and os.path.isdir(checkpoint):
            # Load from a local HuggingFace-format directory
            self.model = VGGT.from_pretrained(checkpoint)
            print("[VGGTAdapter] load model weights from local directory {}".format(checkpoint))
        elif checkpoint and os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
            print("[VGGTAdapter] load model weights from {}".format(checkpoint))
        else:
            # Load from HuggingFace Hub: prefer downloading into weights_dir to avoid scattering files under ~/.cache
            repo_id = checkpoint or "facebook/VGGT-1B"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            self.model = VGGT.from_pretrained(snapshot_dir)
            print("[VGGTAdapter] load model weights from HuggingFace Hub -> {}".format(snapshot_dir))
        self.model = self.model.to(device)
        self.model.eval()
        print(f"[VGGTAdapter] model loaded on {device}")

    def predict(self, scene):
        """Run VGGT inference.

        VGGT input: images (S, 3, H, W) in [0, 1] (no ImageNet normalization)
        VGGT output: pose_enc (B, S, 9), depth (B, S, H, W, 1), world_points (B, S, H, W, 3)
        """
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        # VGGT expects images in [0, 1] range and does not need ImageNet normalization
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # VGGT expects (B, S, 3, H, W)
        images_input = images_raw.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            outputs = self.model(images_input)

        result = {}

        # Depth: (B, S, H, W, 1) -> (N, H, W)
        if 'depth' in outputs:
            depth = outputs['depth'][0, :, :, :, 0]  # (S, H, W)
            result['pred_depth'] = depth.cpu().numpy().astype(np.float32)

        # Pose: pose_enc (B, S, 9) -> extrinsics (B, S, 3, 4) world-to-camera
        if 'pose_enc' in outputs:
            pose_enc = outputs['pose_enc']  # (B, S, 9)
            # Convert to extrinsics: (B, S, 3, 4) world-to-camera
            w2c, pred_intrinsics = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=(H, W)
            )
            w2c = w2c[0].cpu().numpy()  # (S, 3, 4)

            # Convert to cam2world (benchmark standard format)
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
            result['w2c_extrinsics'] = w2c
            result['pred_intrinsic'] = pred_intrinsics[0].cpu().numpy().astype(np.float32)  # (S, 3, 3)

        # World-coordinate point cloud: (B, S, H, W, 3) -> (M, 3)
        if 'world_points' in outputs:
            world_points = outputs['world_points'][0]  # (S, H, W, 3)
            if isinstance(world_points, torch.Tensor):
                world_points = world_points.cpu().numpy()
            result['pred_pointcloud'] = world_points.reshape(-1, 3).astype(np.float32)

        # Confidence: prefer world_points_conf (more discriminative for 3D quality), fallback to depth_conf
        # Both use the expp1 activation, range [1, +inf); larger values mean higher confidence
        # During visualization we filter by percentile (matches VGGT demo_viser.py)
        if 'world_points_conf' in outputs:
            conf = outputs['world_points_conf'][0].cpu().numpy()  # (S, H, W)
            result['pred_confidence'] = conf.astype(np.float32)
        elif 'depth_conf' in outputs:
            conf = outputs['depth_conf'][0].cpu().numpy()  # (S, H, W)
            result['pred_confidence'] = conf.astype(np.float32)

        return result

    def supports_metric_depth(self):
        return False  # VGGT outputs relative depth (not metric)

    def normalize_gt_poses(self, scene):
        """VGGT normalization: align to first camera + scale by avg point distance.

        Matches normalize_camera_extrinsics_and_points_batch used during training.
        """
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """VGGT visualization: build point cloud directly from world_points, filtered by expp1 confidence."""
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

        # Use world_points (pred_pointcloud) for visualization; expp1 conf > 1.0 is already filtered in predict
        # Here we use depth + pose unprojection and filter confidence by percentile
        pred_poses = predictions.get("pred_pose", scene["extrinsic"])
        pred_intrinsic = predictions.get("pred_intrinsic", scene["intrinsic"])

        pred_valid = (pred_depth > 0) & np.isfinite(pred_depth)
        if z_far is not None:
            pred_valid = pred_valid & (pred_depth < z_far)

        # expp1 confidence percentile filtering
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
