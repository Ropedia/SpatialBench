"""
LoGeR model adapter.
Long-Context Geometric Reconstruction with Hybrid Memory (based on Pi3 + TTT/SWA).

LoGeR outputs:
  - camera_poses: (N, 4, 4) cam2world pose (already aligned to the first frame)
  - local_points: (N, H, W, 3) 3D points in the camera coordinate system
  - world_points (points): (N, H, W, 3) 3D points in the world coordinate system
  - confidence: (N, H, W) confidence (already sigmoid)

Note: camera_poses is already c2w format, use directly.
     depth = local_points[..., 2] (Z component).
     input resolution H/W must be 14 multiple (DINOv2 patch size).
"""
import sys
import os

# Inductor subprocess compile workers have known intermittent abort/segfault
# (`Fatal Python error: none_dealloc`), in dense evaluation 900 frameslarge scenes and repeated short scenes
# are especially likely to hit this during recompilation.forcing single-threading uses the in-process path and avoids the issue.
# To fully disable torch.compile: set LOGER_DISABLE_COMPILE=1.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.utils.hf_weights import ensure_hf_snapshot
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR


@register_adapter("loger")
class LoGeRAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.forward_kwargs = {}
        self.device = "cuda"
        # ---- LoGeR inference parameters ----
        self.window_size = -1        # -1=process all frames; >0=chunk window size
        self.overlap_size = 0        # overlap frames between adjacent chunks
        self.se3 = False             # SE3 aligns with (LoGeR_star use)
        self.sim3 = False            # SIM3 aligns with
        self.reset_every = 0         # Reset TTT state every N windows (0=no reset)
        self.turn_off_ttt = False    # disable TTT fast-weight layers
        self.turn_off_swa = False    # disable SWA layers
        self.variant = "LoGeR"       # "LoGeR" or "LoGeR_star"
        self.config_path = None      # model config yaml path (None=auto-discover)

    def name(self):
        return "LoGeR"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        # escape hatch: fully disable torch.compile (loger/models/ttt.py is used in two places).
        # Implementation import loger.* temporarily replace torch.compile with a no-op before, 
        # import then restore it after import - without directly modifying third-party ttt.py.
        if os.environ.get("LOGER_DISABLE_COMPILE", "").lower() in ("1", "true", "yes"):
            _orig_compile = torch.compile
            torch.compile = (lambda fn=None, *a, **kw:
                             fn if fn is not None else (lambda f: f))
            try:
                from loger.pi3_adapter import load_pi3_model
            finally:
                torch.compile = _orig_compile
            print("[LoGeRAdapter] LOGER_DISABLE_COMPILE=1 → torch.compile disabled")
        else:
            from loger.pi3_adapter import load_pi3_model

        # Determine the checkpoint path
        if checkpoint and os.path.isfile(checkpoint):
            ckpt_path = checkpoint
            # auto-discoverin the same directory config yaml
            cfg_path = self.config_path or os.path.join(
                os.path.dirname(checkpoint), "original_config.yaml"
            )
        elif checkpoint and os.path.isdir(checkpoint):
            # Directory: find latest.pt
            ckpt_path = os.path.join(checkpoint, "latest.pt")
            cfg_path = self.config_path or os.path.join(checkpoint, "original_config.yaml")
        else:
            # Default: download from HuggingFace
            repo_id = "Junyi42/LoGeR"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            # Select the variant
            variant_subdir = self.variant  # "LoGeR" or "LoGeR_star"
            ckpt_path = os.path.join(snapshot_dir, variant_subdir, "latest.pt")
            cfg_path = self.config_path or os.path.join(
                snapshot_dir, variant_subdir, "original_config.yaml"
            )

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[LoGeRAdapter] checkpoint not found: {ckpt_path}")

        cfg_path = cfg_path if os.path.isfile(cfg_path) else None
        self.model, self.forward_kwargs = load_pi3_model(
            ckpt_path, config_path=cfg_path, device=torch.device(device)
        )
        self.model.eval()
        print(f"[LoGeRAdapter] loaded from {ckpt_path}, device={device}")
        print(f"[LoGeRAdapter] forward_kwargs={self.forward_kwargs}")

    def predict(self, scene):
        """Run LoGeR inference."""
        from loger.pi3_adapter import run_pi3_inference_on_views, merge_forward_kwargs

        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # Build views list: pi3_adapter accepts {"img": (3, H, W)} in [0, 1]
        views = [{"img": images_raw[i]} for i in range(N)]

        # Merge inference parameters (forward_kwargs from config, can be overridden by self.xxx override)
        overrides = {
            "window_size": self.window_size,
            "overlap_size": self.overlap_size,
            "se3": self.se3,
            "sim3": self.sim3,
            "reset_every": self.reset_every,
            "turn_off_ttt": self.turn_off_ttt,
            "turn_off_swa": self.turn_off_swa,
        }
        fw_kwargs = merge_forward_kwargs(self.forward_kwargs, overrides)

        device = torch.device(self.device)
        _, seq_output = run_pi3_inference_on_views(
            self.model, views, forward_kwargs=fw_kwargs, device=device
        )

        # Extract results (convert all to numpy)
        result = {}

        # pose: (N, 4, 4) c2w -> take (N, 3, 4)
        c2w_44 = seq_output.camera_poses.cpu().float().numpy()   # (N, 4, 4)
        pred_pose = c2w_44[:, :3, :].astype(np.float32)          # (N, 3, 4)
        result['pred_pose'] = pred_pose

        # w2c: closed-form R^T, -R^T@t
        w2c_list = []
        for i in range(N):
            R = pred_pose[i, :3, :3]
            t = pred_pose[i, :3, 3]
            w2c = np.zeros((3, 4), dtype=np.float32)
            w2c[:3, :3] = R.T
            w2c[:3, 3] = -R.T @ t
            w2c_list.append(w2c)
        result['w2c_extrinsics'] = np.stack(w2c_list).astype(np.float32)

        # depth: Z component of local_points (N, H, W, 3)
        local_pts = seq_output.local_points.cpu().float().numpy()  # (N, H, W, 3)
        result['pred_depth'] = local_pts[:, :, :, 2].astype(np.float32)  # (N, H, W)

        # point cloud: world-coordinate points (N, H, W, 3)
        world_pts = seq_output.world_points.cpu().float().numpy()  # (N, H, W, 3)
        result['pred_pointcloud'] = world_pts.reshape(-1, 3).astype(np.float32)

        # confidence: (N, H, W), already sigmoid
        if seq_output.confidence is not None:
            result['pred_confidence'] = seq_output.confidence.cpu().float().numpy().astype(np.float32)

        torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """LoGeR normalization: align to first camera + scale by avg point distance."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
