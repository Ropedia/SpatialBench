"""
InfiniteVGGT (StreamVGGT) model adapter.
causal streaming VGGT, supports arbitrarily long sequences via a rolling KV-cache.

StreamVGGT outputs (per-frame, via model.inference()):
  - pts3d_in_other_view: (1, H, W, 3) world-coordinate points
  - conf: (1, H, W) confidence
  - depth: (1, H, W, 1) metric depth
  - depth_conf: (1, H, W) depth confidence
  - camera_pose: (1, 9) pose encoding [T(3), quat(4), fov(2)] world-to-camera

Note: camera_pose is a world-to-camera encoding; decoding gives w2c and must be converted to c2w.
     input resolution H/W must be 14 multiple (patch_size=14).
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


@register_adapter("infinitevggt")
class InfiniteVGGTAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        # ---- InfiniteVGGT inference parameters ----
        self.total_budget = 1200000  # KV-cache token budget (controls maximum context length)

    def name(self):
        return "InfiniteVGGT"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from streamvggt.models.streamvggt import StreamVGGT

        self.model = StreamVGGT(total_budget=self.total_budget)

        if checkpoint and os.path.isfile(checkpoint):
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt, strict=True)
            print(f"[InfiniteVGGTAdapter] loaded from {checkpoint}")
        elif checkpoint and os.path.isdir(checkpoint):
            # Directory: find checkpoints.pth or any .pth/.pt
            import glob
            for fname in ["checkpoints.pth", "model.pt", "model.pth"]:
                p = os.path.join(checkpoint, fname)
                if os.path.isfile(p):
                    ckpt = torch.load(p, map_location="cpu", weights_only=True)
                    self.model.load_state_dict(ckpt, strict=True)
                    print(f"[InfiniteVGGTAdapter] loaded from {p}")
                    break
            else:
                pts = glob.glob(os.path.join(checkpoint, "*.pt")) +\
                      glob.glob(os.path.join(checkpoint, "*.pth"))
                if pts:
                    ckpt = torch.load(pts[0], map_location="cpu", weights_only=True)
                    self.model.load_state_dict(ckpt, strict=True)
                    print(f"[InfiniteVGGTAdapter] loaded from {pts[0]}")
        else:
            # HuggingFace download
            repo_id = checkpoint or "lch01/StreamVGGT"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            import glob
            for fname in ["checkpoints.pth", "model.pt", "model.pth"]:
                p = os.path.join(snapshot_dir, fname)
                if os.path.isfile(p):
                    ckpt = torch.load(p, map_location="cpu", weights_only=True)
                    self.model.load_state_dict(ckpt, strict=True)
                    print(f"[InfiniteVGGTAdapter] loaded from HuggingFace -> {p}")
                    break

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[InfiniteVGGTAdapter] model on {device}, total_budget={self.total_budget}")

    def predict(self, scene):
        """Run InfiniteVGGT streaming inference."""
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # Build the frames list: per-frame {"img": (1, 3, H, W)}
        frames = [{"img": images_raw[i:i+1].to(self.device)} for i in range(N)]

        # streaming inference
        dtype = (torch.bfloat16
                 if self.device != "cpu" and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            output = self.model.inference(frames, cache_results=True)

        # Collect results frame by frame
        all_pose_enc = []
        all_depth = []
        all_pts3d = []
        all_conf = []

        for i in range(N):
            res = output.ress[i]

            # camera_pose: (1, 9) pose encoding
            all_pose_enc.append(res['camera_pose'].squeeze(0).float())  # (9,)

            # depth: (1, H, W, 1) → (H, W)
            d = res['depth'].squeeze(0).squeeze(-1).float()  # (H, W)
            all_depth.append(d)

            # pts3d_in_other_view: (1, H, W, 3) → (H, W, 3)
            pts = res['pts3d_in_other_view'].squeeze(0).float()  # (H, W, 3)
            all_pts3d.append(pts)

            # confidence: prefer conf, fallback depth_conf
            if 'conf' in res and res['conf'] is not None:
                c = res['conf'].squeeze(0).float()  # (H, W)
            elif 'depth_conf' in res and res['depth_conf'] is not None:
                c = res['depth_conf'].squeeze(0).float()  # (H, W)
            else:
                c = torch.ones(H, W, dtype=torch.float32)
            all_conf.append(c)

        # Convert poses: (1, N, 9) -> w2c (N, 3, 4) -> c2w
        pose_enc = torch.stack(all_pose_enc).unsqueeze(0)  # (1, N, 9)
        w2c, pred_intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W)
        )
        w2c = w2c[0].cpu().numpy().astype(np.float32)  # (N, 3, 4)

        # w2c → c2w (closed-form: R^T, -R^T@t)
        c2w_list = []
        for i in range(N):
            R = w2c[i, :3, :3]
            t = w2c[i, :3, 3]
            c2w = np.zeros((3, 4), dtype=np.float32)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            c2w_list.append(c2w)

        result = {}
        result['pred_pose'] = np.stack(c2w_list).astype(np.float32)
        result['w2c_extrinsics'] = w2c
        result['pred_intrinsic'] = pred_intrinsics[0].cpu().numpy().astype(np.float32)

        # depth
        result['pred_depth'] = torch.stack(all_depth).cpu().numpy().astype(np.float32)

        # point cloud
        result['pred_pointcloud'] = (
            torch.stack(all_pts3d).cpu().numpy().reshape(-1, 3).astype(np.float32)
        )

        # confidence
        result['pred_confidence'] = (
            torch.stack(all_conf).cpu().numpy().astype(np.float32)
        )

        torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return False  # StreamVGGT outputs relative depth

    def normalize_gt_poses(self, scene):
        """InfiniteVGGT normalization: align to first camera + scale by avg point distance."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
