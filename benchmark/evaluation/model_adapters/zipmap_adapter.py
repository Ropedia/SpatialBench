"""
ZipMap model adapter.

ZipMap outputs:
  - pose_enc: (B, N, 9), converted to world-to-camera extrinsics
  - depth: (B, N, H, W, 1) relative depth
  - depth_conf: (B, N, H, W) depth confidence
  - local_points: (B, N, H, W, 3) camera-coordinate points

The benchmark expects pred_pose in cam2world format. ZipMap's pose encoding
decodes to world-to-camera, so predict() returns both w2c_extrinsics and the
closed-form inverse as pred_pose.
"""
import copy
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

_ZIPMAP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "zipmap")
)
if _ZIPMAP_ROOT not in sys.path:
    sys.path.insert(0, _ZIPMAP_ROOT)

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR


_BASE_TTT_PARAMS = {
    "bias": True,
    "head_dim": 1024,
    "inter_multi": 2,
    "base_lr": 0.01,
    "muon_update_steps": 5,
    "use_gate_fn": True,
}

_MAIN_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": {
        "ttt_mode": True,
        "params": _BASE_TTT_PARAMS,
    },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True,
    },
}

_STREAMING_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": False,
    "enable_camera_mlp": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": {
        "ttt_mode": True,
        "params": _BASE_TTT_PARAMS,
        "window_size": 1,
    },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True,
    },
}


@register_adapter("zipmap")
class ZipMapAdapter(ModelAdapter):
    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.variant = "main"          # main | streaming
        self.affine_invariant = True
        self.align_first_view = True
        self.use_ema = False
        self.disable_compile = False
        self.window_size = None        # streaming-only override

    def name(self):
        return "ZipMap"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        variant = str(self.variant).lower()

        if self.disable_compile:
            orig_compile = torch.compile
            torch.compile = (lambda fn=None, *a, **kw:
                             fn if fn is not None else (lambda f: f))
            try:
                ZipMap, default_filename, model_config = self._import_model(variant)
            finally:
                torch.compile = orig_compile
            print("[ZipMapAdapter] disable_compile=True, torch.compile disabled during import")
        else:
            ZipMap, default_filename, model_config = self._import_model(variant)

        model_config = copy.deepcopy(model_config)
        model_config["other_config"]["affine_invariant"] = bool(self.affine_invariant)
        ckpt_path = self._resolve_checkpoint(
            checkpoint=checkpoint,
            default_filename=default_filename,
            weights_dir=weights_dir,
        )

        self.model = ZipMap(**model_config)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if self.use_ema and isinstance(state, dict) and "ema" in state:
            state_dict = state["ema"]
        elif isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[ZipMapAdapter] missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[ZipMapAdapter] unexpected keys ({len(unexpected)}): {unexpected[:8]}")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[ZipMapAdapter] loaded {variant} checkpoint from {ckpt_path} on {device}")

    def predict(self, scene):
        from zipmap.utils.geometry import homogenize_points
        from zipmap.utils.pose_enc import pose_encoding_to_extri_intri

        images_raw = scene["images_raw"]  # (N, 3, H, W), [0, 1]
        n, _, h, w = images_raw.shape
        if h % 14 != 0 or w % 14 != 0:
            raise ValueError(
                f"ZipMap requires H/W divisible by 14, got H={h}, W={w}. "
                "Set resolution_override: {width: 518, align: 14} in the config."
            )

        images_input = images_raw.to(self.device)
        device_type = torch.device(self.device).type
        if device_type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            amp_context = torch.amp.autocast("cuda", dtype=amp_dtype)
        else:
            amp_context = nullcontext()

        with torch.no_grad(), amp_context:
            if str(self.variant).lower() == "streaming":
                outputs = self.model(images_input, window_size=self.window_size)
            else:
                outputs = self.model(images_input)

        result = {}

        if "depth" in outputs and outputs["depth"] is not None:
            depth = outputs["depth"][0, :, :, :, 0].float()
            depth = self._resize_nhw_tensor(depth, h, w)
            result["pred_depth"] = depth.cpu().numpy().astype(np.float32)

        pred_pose = None
        if "pose_enc" in outputs and outputs["pose_enc"] is not None:
            w2c, pred_intrinsic = pose_encoding_to_extri_intri(
                outputs["pose_enc"], image_size_hw=(h, w)
            )
            w2c = w2c[0].float().cpu().numpy().astype(np.float32)

            if self.align_first_view and n > 0:
                w2c = self._align_w2c_to_first(w2c)

            pred_pose = self._invert_se3(w2c).astype(np.float32)
            result["w2c_extrinsics"] = w2c.astype(np.float32)
            result["pred_pose"] = pred_pose
            result["pred_intrinsic"] = (
                pred_intrinsic[0].float().cpu().numpy().astype(np.float32)
            )

        if "depth_conf" in outputs and outputs["depth_conf"] is not None:
            conf = outputs["depth_conf"][0].float()
            conf = self._resize_nhw_tensor(conf, h, w)
            result["pred_confidence"] = conf.cpu().numpy().astype(np.float32)

        if pred_pose is not None and "local_points" in outputs and outputs["local_points"] is not None:
            local_points = outputs["local_points"][0].float()
            local_points = self._resize_points_tensor(local_points, h, w)
            local_np = local_points.cpu().numpy().astype(np.float32)
            c2w_44 = self._to_44(pred_pose)
            local_h = homogenize_points(local_np)
            world = np.einsum("nij,nhwj->nhwi", c2w_44, local_h)[..., :3]
            result["pred_pointcloud"] = world.reshape(-1, 3).astype(np.float32)

        if device_type == "cuda":
            torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return False

    def requires_intrinsics(self):
        return False

    def normalize_gt_poses(self, scene):
        """ZipMap follows VGGT-style first-frame normalization."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    @staticmethod
    def _import_model(variant):
        if variant == "main":
            from zipmap.models.ZipMap import ZipMap
            return ZipMap, "checkpoint_aff_inv.pt", _MAIN_CONFIG
        if variant == "streaming":
            from zipmap.models.ZipMap_AR import ZipMap
            return ZipMap, "checkpoint_online.pt", _STREAMING_CONFIG
        raise ValueError("ZipMap variant must be 'main' or 'streaming'")

    @staticmethod
    def _resolve_checkpoint(checkpoint, default_filename, weights_dir=None):
        weights_dir = weights_dir or os.path.join(DEFAULT_CHECKPOINTS_DIR, "zipmap")
        default_path = os.path.join(weights_dir, default_filename)

        if checkpoint is None:
            checkpoint = default_path

        checkpoint = os.path.expanduser(str(checkpoint))
        if os.path.isfile(checkpoint):
            return os.path.abspath(checkpoint)
        if os.path.isdir(checkpoint):
            candidate = os.path.join(checkpoint, default_filename)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
            for name in sorted(os.listdir(checkpoint)):
                if name.endswith((".pt", ".pth", ".ckpt")):
                    return os.path.abspath(os.path.join(checkpoint, name))

        if "/" in checkpoint and not checkpoint.endswith((".pt", ".pth", ".ckpt")):
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id=checkpoint,
                filename=default_filename,
                local_dir=weights_dir,
            )

        raise FileNotFoundError(
            f"[ZipMapAdapter] checkpoint not found: {checkpoint}\n"
            f"Download it with:\n"
            f"  hf download coast01/ZipMap {default_filename} --local-dir {weights_dir}\n"
            f"or set checkpoint in the YAML to a local .pt file."
        )

    @staticmethod
    def _resize_nhw_tensor(x, h, w):
        if tuple(x.shape[-2:]) == (h, w):
            return x
        return F.interpolate(
            x.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)

    @staticmethod
    def _resize_points_tensor(points, h, w):
        if tuple(points.shape[1:3]) == (h, w):
            return points
        chw = points.permute(0, 3, 1, 2)
        resized = F.interpolate(chw, size=(h, w), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1)

    @staticmethod
    def _to_44(se3_34):
        n = se3_34.shape[0]
        out = np.zeros((n, 4, 4), dtype=np.float32)
        out[:, :3, :] = se3_34
        out[:, 3, 3] = 1.0
        return out

    @classmethod
    def _align_w2c_to_first(cls, w2c_34):
        w2c_44 = cls._to_44(w2c_34)
        first_c2w = cls._invert_se3(w2c_34[0:1])
        first_c2w_44 = cls._to_44(first_c2w)[0]
        aligned = np.matmul(w2c_44, first_c2w_44)
        return aligned[:, :3, :].astype(np.float32)
