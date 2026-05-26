"""
VGG-TTT model adapter.

VGG-TTT is API-compatible with VGGT at inference time, but its global attention
uses test-time training fast weights. Its infer() API returns:
  - pose: (N, 4, 4) camera-to-world transforms
  - intrinsics: (N, 3, 3)
  - depth: (N, H, W, 1)
  - pts3d: (N, H, W, 3) world-coordinate points
  - conf: (N, H, W)
"""
import os
import sys
from functools import partial

import numpy as np
import torch

_VGG_TTT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "vgg_ttt")
)
if _VGG_TTT_ROOT not in sys.path:
    sys.path.insert(0, _VGG_TTT_ROOT)

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.utils.hf_weights import ensure_hf_snapshot
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR


@register_adapter("vgg_ttt")
class VGGTTTAdapter(ModelAdapter):
    def __init__(self):
        self.model = None
        self.device = "cuda"

        # Forwarded to VGGT.infer().
        self.num_ttt_steps = 1
        self.memory_efficient_inference = True
        self.use_global_pred = False
        self.log_ttt_details = False
        self.offload_to_cpu = False
        self.dtype = None

    def name(self):
        return "VGG-TTT"

    def configure(self, **kwargs):
        super().configure(**kwargs)
        if isinstance(self.dtype, str):
            self.dtype = self._parse_dtype(self.dtype)

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        if torch.device(device).type != "cuda":
            raise RuntimeError(
                "[VGGTTTAdapter] VGG-TTT inference currently requires a CUDA device; "
                "the upstream infer() path uses CUDA autocast and CUDA memory queries."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("[VGGTTTAdapter] CUDA is not available.")

        from vggttt.nets.vggt.models.vggt import VGGT

        ckpt = self._resolve_checkpoint(checkpoint, weights_dir=weights_dir)
        if os.path.isfile(ckpt):
            self.model = self._load_from_state_file(VGGT, ckpt)
        else:
            self.model = VGGT.from_pretrained(ckpt)

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[VGGTTTAdapter] loaded checkpoint from {ckpt} on {device}")

    def predict(self, scene):
        images_raw = scene["images_raw"]  # (N, 3, H, W), [0, 1]
        n, _, h, w = images_raw.shape
        if h % 14 != 0 or w % 14 != 0:
            raise ValueError(
                f"VGG-TTT requires H/W divisible by 14, got H={h}, W={w}. "
                "Set resolution_override: {width: 518, align: 14} in the config."
            )

        images = images_raw.to(self.device, non_blocking=True)
        with torch.inference_mode():
            outputs = self.model.infer(
                images,
                num_ttt_steps=self.num_ttt_steps,
                dtype=self.dtype,
                log_ttt_details=bool(self.log_ttt_details),
                memory_efficient_inference=bool(self.memory_efficient_inference),
                use_global_pred=bool(self.use_global_pred),
                offload_to_cpu=bool(self.offload_to_cpu),
            )

        result = {}

        if "depth" in outputs and outputs["depth"] is not None:
            depth = self._as_numpy(outputs["depth"])
            if depth.ndim == 4 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            result["pred_depth"] = depth.astype(np.float32)

        if "pose" in outputs and outputs["pose"] is not None:
            c2w = self._as_numpy(outputs["pose"]).astype(np.float32)
            if c2w.shape[-2:] == (4, 4):
                c2w = c2w[:, :3, :]
            if c2w.shape != (n, 3, 4):
                raise ValueError(f"Unexpected VGG-TTT pose shape: {c2w.shape}")
            result["pred_pose"] = c2w
            result["w2c_extrinsics"] = self._invert_se3(c2w).astype(np.float32)

        if "intrinsics" in outputs and outputs["intrinsics"] is not None:
            result["pred_intrinsic"] = self._as_numpy(outputs["intrinsics"]).astype(np.float32)

        if "pts3d" in outputs and outputs["pts3d"] is not None:
            pts3d = self._as_numpy(outputs["pts3d"]).astype(np.float32)
            result["pred_pointcloud"] = pts3d.reshape(-1, 3)

        if "conf" in outputs and outputs["conf"] is not None:
            conf = self._as_numpy(outputs["conf"])
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            result["pred_confidence"] = conf.astype(np.float32)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return False

    def requires_intrinsics(self):
        return False

    def normalize_gt_poses(self, scene):
        """VGG-TTT follows VGGT-style first-frame pose normalization."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    @staticmethod
    def _resolve_checkpoint(checkpoint, weights_dir=None):
        weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
        if checkpoint is None:
            return ensure_hf_snapshot("nvidia/vgg-ttt", local_root=weights_dir)

        checkpoint = os.path.expanduser(str(checkpoint))
        if os.path.exists(checkpoint):
            return os.path.abspath(checkpoint)

        if "/" in checkpoint and not checkpoint.endswith((".pt", ".pth", ".ckpt")):
            return ensure_hf_snapshot(checkpoint, local_root=weights_dir)

        raise FileNotFoundError(
            f"[VGGTTTAdapter] checkpoint not found: {checkpoint}\n"
            "Use checkpoint: null to auto-download nvidia/vgg-ttt, set checkpoint "
            "to a local Hugging Face snapshot directory, or provide a local weight file."
        )

    @staticmethod
    def _load_from_state_file(VGGT, ckpt_path):
        from vggttt.nets.ttt_attention import FastWeightAttention

        model = VGGT(
            global_attn_class=partial(
                FastWeightAttention,
                muon_update_steps=5,
                base_lr=0.01,
                mlp_ratio=4,
                short_conv_size_qkv=(0, 0, 3),
                div_lr_by_seq_len=False,
            ),
            gradient_checkpoint=True,
            init_weights=None,
            use_track_head=False,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]

        cleaned = {}
        for key, value in state.items():
            for prefix in ("model.model.", "module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            cleaned[key] = value

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[VGGTTTAdapter] missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[VGGTTTAdapter] unexpected keys ({len(unexpected)}): {unexpected[:8]}")
        return model

    @staticmethod
    def _as_numpy(value):
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy()
        return np.asarray(value)

    @staticmethod
    def _parse_dtype(value):
        value = value.lower()
        if value in ("bf16", "bfloat16", "torch.bfloat16"):
            return torch.bfloat16
        if value in ("fp16", "float16", "half", "torch.float16"):
            return torch.float16
        if value in ("fp32", "float32", "torch.float32"):
            return torch.float32
        raise ValueError("dtype must be one of bf16, fp16, or fp32")
