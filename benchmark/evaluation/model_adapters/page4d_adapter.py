"""
PAGE4D model adapter.
Use the PAGE4D VGGT forward() API for inference.

PAGE4D (4D Pose and Geometry Estimation) outputs:
  - pose_enc: (B, S, 9) = [T(3), quat(4), fov(2)], world-to-camera
  - depth: (B, S, H, W, 1) depth map
  - depth_conf: (B, S, H, W) depth confidence
  - world_points: (B, S, H, W, 3) world-coordinate points
  - world_points_conf: (B, S, H, W) point confidence

Note: PAGE4D is a dynamic-scene extension of VGGT, with a fully compatible API.
     extrinsics in pose_enc is world-to-camera, predict() converts to cam2world.
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


@register_adapter("page4d")
class PAGE4DAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"

    def name(self):
        return "PAGE4D"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from page4d.models.vggt import VGGT as PAGE4D_VGGT

        self.model = PAGE4D_VGGT()

        if checkpoint and os.path.isdir(checkpoint):
            self.model = PAGE4D_VGGT.from_pretrained(checkpoint)
            print(f"[PAGE4DAdapter] Model loaded from directory: {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[PAGE4DAdapter] Model loaded from file: {checkpoint}")
        else:
            repo_id = checkpoint or "zhouk777/PAGE4D"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            # Find the checkpoint file
            ckpt_file = None
            for f in os.listdir(snapshot_dir):
                if f.endswith('.pt') or f.endswith('.pth'):
                    ckpt_file = os.path.join(snapshot_dir, f)
                    break
            if ckpt_file:
                state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']
                elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.model.load_state_dict(state_dict, strict=True)
                print(f"[PAGE4DAdapter] Model loaded from HuggingFace -> {ckpt_file}")
            else:
                self.model = PAGE4D_VGGT.from_pretrained(snapshot_dir)
                print(f"[PAGE4DAdapter] Model loaded from HuggingFace -> {snapshot_dir}")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[PAGE4DAdapter] model loaded on {device}")

    def predict(self, scene):
        """Run PAGE4D inference.

        PAGE4D input: images (S, 3, H, W) in [0, 1]
        PAGE4D outputs: pose_enc (B, S, 9), depth (B, S, H, W, 1), world_points (B, S, H, W, 3)
        """
        from page4d.utils.pose_enc import pose_encoding_to_extri_intri

        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        images_input = images_raw.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        with torch.no_grad():
            outputs = self.model(images_input)

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
            result['pred_intrinsic'] = pred_intrinsics[0].cpu().numpy().astype(np.float32)

        # World-coordinate point cloud
        if 'world_points' in outputs:
            world_points = outputs['world_points'][0]
            if isinstance(world_points, torch.Tensor):
                world_points = world_points.cpu().numpy()
            result['pred_pointcloud'] = world_points.reshape(-1, 3).astype(np.float32)

        # confidence
        if 'world_points_conf' in outputs:
            conf = outputs['world_points_conf'][0].cpu().numpy()
            result['pred_confidence'] = conf.astype(np.float32)
        elif 'depth_conf' in outputs:
            conf = outputs['depth_conf'][0].cpu().numpy()
            result['pred_confidence'] = conf.astype(np.float32)

        torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """PAGE4D normalization: align to first camera + scale by avg point distance."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
