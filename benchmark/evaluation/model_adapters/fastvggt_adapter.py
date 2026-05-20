"""
FastVGGT model adapter.
Adds token merging acceleration (merging parameter controls which block starts merging).

FastVGGT has exactly the same output format as VGGT:
  - pose_enc: (B, S, 9) = [T(3), quat(4), fov(2)], world-to-camera
  - depth: (B, S, H, W, 1) depth map (relative depth)
  - depth_conf: (B, S, H, W) depth confidence
  - world_points: (B, S, H, W, 3) world-coordinate points (if enable_point=True)

Note: must call before inference model.update_patch_dimensions(patch_width, patch_height)
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


@register_adapter("fastvggt")
class FastVGGTAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        # ---- FastVGGT inference parameters ----
        self.merging = 0          # 0=no merging; 4/11/17/23=start token merging from block N
        self.merge_ratio = 0.9    # token merge ratio (0.9=merge 90% redundant tokens)
        self.enable_point = False # whether to output world_points

    def name(self):
        return "FastVGGT"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from fastvggt.models.vggt import VGGT

        if checkpoint and os.path.isdir(checkpoint):
            self.model = VGGT.from_pretrained(checkpoint)
            print(f"[FastVGGTAdapter] loaded from local directory {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            self.model = VGGT(
                enable_camera=True,
                enable_point=self.enable_point,
                enable_depth=True,
                enable_track=False,
                merging=self.merging,
                merge_ratio=self.merge_ratio,
            )
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[FastVGGTAdapter] loaded from {checkpoint}")
        else:
            # Load from HuggingFace Hub and cache in weights_dir
            repo_id = checkpoint or "facebook/VGGT-1B"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            self.model = VGGT(
                enable_camera=True,
                enable_point=self.enable_point,
                enable_depth=True,
                enable_track=False,
                merging=self.merging,
                merge_ratio=self.merge_ratio,
            )
            # Load .pt weight files from the snapshot directory
            ckpt_path = os.path.join(snapshot_dir, "model.pt")
            if not os.path.isfile(ckpt_path):
                import glob
                pts = glob.glob(os.path.join(snapshot_dir, "*.pt"))
                if pts:
                    ckpt_path = pts[0]
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[FastVGGTAdapter] loaded from {ckpt_path}")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[FastVGGTAdapter] model loaded on {device}, "
              f"merging={self.merging}, merge_ratio={self.merge_ratio}")

    def predict(self, scene):
        """Run FastVGGT inference.

        must call before inference update_patch_dimensions, then behaves the same as VGGT.
        """
        from fastvggt.utils.pose_enc import pose_encoding_to_extri_intri

        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # Compute patch dimensions and update the model (FastVGGT specific step)
        patch_width = W // 14
        patch_height = H // 14
        self.model.update_patch_dimensions(patch_width, patch_height)

        # FastVGGT expects (B, S, 3, H, W)
        # the internal aggregator forcibly casts to bfloat16, dpt_head pos_embed returns float32 again
        # Use autocast to handle all precision boundaries uniformly: eligible ops (conv/linear) automatically cast inputs to bfloat16
        images_input = images_raw.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        # Single-frame special case: token merging assumes the first frame is all dst and the remaining frames provide src tokens.
        # N=1 has no src tokens and triggers merge.py:fast_similarity_chunks 
        # "range() arg 3 must not be zero".temporarily set aggregator.merging to None to bypass it.
        orig_merging = self.model.aggregator.merging
        if N == 1:
            self.model.aggregator.merging = None

        try:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(images_input)
        finally:
            self.model.aggregator.merging = orig_merging

        result = {}

        # depth: (B, S, H, W, 1) -> (N, H, W)
        # Note: autocast outputs bfloat16, which numpy does not support; call .float() first.
        if 'depth' in outputs and outputs['depth'] is not None:
            depth = outputs['depth'][0, :, :, :, 0]
            result['pred_depth'] = depth.float().cpu().numpy().astype(np.float32)

        # pose: pose_enc (B, S, 9) -> extrinsics (B, S, 3, 4) world-to-camera
        if 'pose_enc' in outputs and outputs['pose_enc'] is not None:
            pose_enc = outputs['pose_enc']  # (B, S, 9)
            w2c, pred_intrinsics = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=(H, W)
            )
            w2c = w2c[0].float().cpu().numpy()  # (S, 3, 4)

            # w2c -> c2w (closed-form: R^T, -R^T@t)
            c2w_list = []
            for i in range(N):
                R = w2c[i, :3, :3]
                t = w2c[i, :3, 3]
                c2w = np.zeros((3, 4), dtype=np.float32)
                c2w[:3, :3] = R.T
                c2w[:3, 3] = -R.T @ t
                c2w_list.append(c2w)
            result['pred_pose'] = np.stack(c2w_list).astype(np.float32)
            result['w2c_extrinsics'] = w2c.astype(np.float32)
            result['pred_intrinsic'] = pred_intrinsics[0].float().cpu().numpy().astype(np.float32)

        # World-coordinate point cloud (if enable_point=True)
        if 'world_points' in outputs and outputs['world_points'] is not None:
            world_points = outputs['world_points'][0]  # (S, H, W, 3)
            if isinstance(world_points, torch.Tensor):
                world_points = world_points.float().cpu().numpy()
            result['pred_pointcloud'] = world_points.reshape(-1, 3).astype(np.float32)

        # confidence: prefer world_points_conf, fall back to depth_conf
        if 'world_points_conf' in outputs and outputs['world_points_conf'] is not None:
            conf = outputs['world_points_conf'][0].float().cpu().numpy()
            result['pred_confidence'] = conf.astype(np.float32)
        elif 'depth_conf' in outputs and outputs['depth_conf'] is not None:
            conf = outputs['depth_conf'][0].float().cpu().numpy()
            result['pred_confidence'] = conf.astype(np.float32)

        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """FastVGGT normalization: align to first camera + scale by avg point distance."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
