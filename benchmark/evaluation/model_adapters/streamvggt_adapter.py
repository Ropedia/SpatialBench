"""
StreamVGGT model adapter.
Use the StreamVGGT forward() API for inference.

StreamVGGT outputs:
  - pose_enc: (B, S, 9) = [T(3), quat(4), fov(2)], world-to-camera
  - depth: (B, S, H, W, 1) depth map
  - depth_conf: (B, S, H, W) depth confidence
  - world_points: (B, S, H, W, 3) world-coordinate points
  - world_points_conf: (B, S, H, W) point confidence

Note: StreamVGGT is a streaming variant of VGGT, supports view-dict input.
     forward() internally assembles an images tensor from view dicts.
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


@register_adapter("streamvggt")
class StreamVGGTAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"

    def name(self):
        return "StreamVGGT"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from streamvggt.models.streamvggt import StreamVGGT

        self.model = StreamVGGT()

        if checkpoint and os.path.isdir(checkpoint):
            self.model = StreamVGGT.from_pretrained(checkpoint)
            print(f"[StreamVGGTAdapter] Model loaded from directory: {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[StreamVGGTAdapter] Model loaded from file: {checkpoint}")
        else:
            repo_id = checkpoint or "lch01/StreamVGGT"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            # Try from_pretrained (safetensors format)
            try:
                self.model = StreamVGGT.from_pretrained(snapshot_dir)
                print(f"[StreamVGGTAdapter] Model loaded from HuggingFace -> {snapshot_dir}")
            except Exception:
                # fallback: find .pt/.pth file
                for f in os.listdir(snapshot_dir):
                    if f.endswith('.pt') or f.endswith('.pth'):
                        ckpt = os.path.join(snapshot_dir, f)
                        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
                        if isinstance(state_dict, dict) and 'model' in state_dict:
                            state_dict = state_dict['model']
                        self.model.load_state_dict(state_dict, strict=True)
                        print(f"[StreamVGGTAdapter] Model loaded from {ckpt}")
                        break

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[StreamVGGTAdapter] model loaded on {device}")

    def predict(self, scene):
        """Run StreamVGGT inference.

        Use StreamVGGT streaming inference() interface: process frame by frame + KV-cache, 
        move each frame result to CPU immediately and clear the GPU cache, so VRAM does not grow linearly with sequence length.
        (forward() is an offline batch path and OOMs directly at 100+ frames.)
        outputs StreamVGGTOutput, whose ress field is a list of per-frame CPU tensor dicts.
        """
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # StreamVGGT expects list of view dicts
        views = []
        for i in range(N):
            view = {
                'img': images_raw[i:i+1].to(self.device),  # (1, 3, H, W)
            }
            views.append(view)

        with torch.no_grad():
            output = self.model.inference(views)

        result = {}

        # StreamVGGT forward returns StreamVGGTOutput(ress=..., views=...)
        # ress is per-frame dicts: camera_pose(B,9), depth(B,H,W,1), pts3d_in_other_view(B,H,W,3), conf(B,H,W)
        ress = output.ress

        # Collect pose_enc from per-frame camera_pose (B, 9) reassemble as (B, S, 9)
        pose_enc_list = []
        for res in ress:
            pose_enc_list.append(res['camera_pose'])  # (B, 9)
        pose_enc = torch.stack(pose_enc_list, dim=1)  # (B, S, 9)

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

        # depth: from per-frame depth (B, H, W, 1)
        depth_list = []
        for res in ress:
            d = res['depth'][0, :, :, 0]  # (H, W)
            depth_list.append(d.cpu().numpy())
        result['pred_depth'] = np.stack(depth_list).astype(np.float32)

        # World-coordinate point cloud: pts3d_in_other_view (B, H, W, 3)
        all_points = []
        for res in ress:
            pts = res['pts3d_in_other_view'][0].cpu().numpy()  # (H, W, 3)
            all_points.append(pts)
        world_points = np.stack(all_points)  # (N, H, W, 3)
        result['pred_pointcloud'] = world_points.reshape(-1, 3).astype(np.float32)

        # confidence: conf (B, H, W)
        conf_list = []
        for res in ress:
            c = res['conf'][0].cpu().numpy()  # (H, W)
            conf_list.append(c)
        result['pred_confidence'] = np.stack(conf_list).astype(np.float32)

        torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """StreamVGGT normalization: align to first camera + scale by avg point distance."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
