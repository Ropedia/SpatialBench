"""R3 model adapter.

R3 predicts relative-pose trajectories and dense depth on top of a modified
Depth Anything 3 backbone. The vendored source carries its own
``depth_anything_3`` package, so this adapter activates the R3 source root
before importing model code.
"""
import glob
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.utils.hf_weights import ensure_hf_snapshot
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR


_R3_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "r3")
)
_R3_PREFIXES = ("R3", "depth_anything_3")


def _is_under(path, root):
    try:
        return os.path.commonpath([os.path.abspath(path), root]) == root
    except ValueError:
        return False


def _module_file(module):
    path = getattr(module, "__file__", None)
    return os.path.abspath(path) if path else None


def _activate_r3_imports():
    """Make R3's vendored packages win over benchmark/models/depth_anything_3."""
    if _R3_ROOT in sys.path:
        sys.path.remove(_R3_ROOT)
    sys.path.insert(0, _R3_ROOT)

    for name, module in list(sys.modules.items()):
        if not any(name == p or name.startswith(p + ".") for p in _R3_PREFIXES):
            continue
        module_path = _module_file(module)
        if module_path and _is_under(module_path, _R3_ROOT):
            continue
        del sys.modules[name]


@register_adapter("r3")
class R3Adapter(ModelAdapter):
    _MODE_ALIASES = {
        "short": "local",
        "stride": "strided",
        "sampled": "strided",
        "sparse": "strided",
    }
    _MODE_PRESETS = {
        "test": {
            "online_kv_cache_mode": "all",
            "online_fallback_enabled": False,
            "max_segment_frames": 0,
            "metric_scale_enabled": False,
        },
        "local": {
            "online_kv_cache_mode": "dynamic",
            "online_fallback_enabled": False,
            "max_segment_frames": 0,
            "metric_scale_enabled": False,
            "keyframe_mode": "novelty",
            "keyframe_novelty_threshold": 0.985,
            "keyframe_max_interval": 30,
            "keyframe_max_keyframes": 100,
        },
        "long": {
            "online_kv_cache_mode": "dynamic",
            "online_fallback_enabled": True,
            "max_segment_frames": 300,
            "fallback_drought_threshold_pct": 45.0,
            "metric_scale_enabled": True,
            "metric_bootstrap_frames": 5,
            "keyframe_mode": "novelty",
            "keyframe_novelty_threshold": 0.985,
            "keyframe_max_interval": 30,
            "keyframe_max_keyframes": 100,
        },
        "strided": {
            "online_kv_cache_mode": "all",
            "online_fallback_enabled": True,
            "max_segment_frames": 100,
            "fallback_drought_threshold_pct": 45.0,
            "metric_scale_enabled": True,
            "metric_bootstrap_frames": 5,
        },
    }

    def __init__(self):
        self.model = None
        self.device = "cuda"

        self.config_name = "r3-large"
        self.mode = "default"
        self.wrapper_mode = "online"  # online | offline
        # R3/infer.py default does not override cfg.net.attention_mode.
        self.attention_mode = ""
        self.attention_window_size = 0
        self.patch_size = 14

        self.online_kv_cache_mode = "all"
        self.online_kv_backend = "dense"
        self.flashinfer_page_size = 0
        self.online_recent_frames = 0
        self.bootstrap_full_attention_frames = 0
        self.online_verbose = None
        self.bank_initial_frames = 1
        self.keyframe_mode = "interval"
        self.keyframe_interval = 10
        self.keyframe_novelty_threshold = 0.985
        self.keyframe_max_interval = 30
        self.keyframe_max_keyframes = 100
        self.keyframe_pose_confidence_ratio = 0.0
        self.pose_max_recent = 0
        self.online_finalize_pose_reconstruction = False

        self.rel_pose_reconstruction_method = "greedy"
        self.rel_pose_topn_conf = 999
        self.rel_pose_score_mode = "auto"
        self.pgo_num_iters = 100
        self.pgo_lr = 0.05
        self.pgo_weight_T = 1.0
        self.pgo_weight_R = 0.5
        self.pgo_weight_fl = 0.1
        self.pgo_init_prior_weight = 1e-4
        self.pgo_keyframe_stride = 0
        self.edge_percentile_cutoff = 0.0
        self.pgo_geman_mcclure_c = 0.0
        self.pgo_dcs_phi = 0.0
        self.pgo_max_translation_per_frame = 0.0

        self.online_fallback_enabled = False
        self.fallback_drought_length = 3
        self.fallback_drought_threshold = 1.0
        self.fallback_drought_threshold_pct = 50.0
        self.fallback_drought_warmup_frames = 5
        self.fallback_num_bridge_frames = 5
        self.fallback_min_bridge_baseline_ratio = 0.0
        self.fallback_max_bridge_lookback = 0
        self.evict_low_conf_threshold = 0.0
        self.evict_low_conf_threshold_pct = 0.0
        self.evict_low_conf_warmup_frames = 3
        self.fallback_ref_mode = "bridge"
        self.min_segment_frames = 0
        self.max_segment_frames = 0
        self.fallback_replay_attention = "full"
        self.fallback_skip_confidence_check = False
        self.depth_scale_mode = "ransac"
        self.disable_segment_pgo = False

        self.metric_scale_enabled = False
        self.metric_model_name = "depth-anything/DA3METRIC-LARGE"
        self.metric_min_conf = 1.02
        self.metric_bootstrap_frames = 1
        self.compute_sky_mask = False
        self.sky_mask_threshold = 0.3

        self.use_amp = True
        self.amp_dtype = "bf16"  # bf16 | fp16

    def name(self):
        return "R3"

    def configure(self, **kwargs):
        mode = kwargs.pop("mode", None)
        if mode is not None:
            mode = self._MODE_ALIASES.get(str(mode), str(mode))
            if mode not in self._MODE_PRESETS:
                raise ValueError(f"Unsupported R3 mode: {mode}")
            self.mode = mode
            for key, val in self._MODE_PRESETS[mode].items():
                setattr(self, key, val)

        for key, val in kwargs.items():
            if val is None:
                continue
            if not hasattr(self, key):
                continue
            if key in {"wrapper_mode", "attention_mode", "online_kv_cache_mode",
                       "online_kv_backend", "keyframe_mode",
                       "rel_pose_reconstruction_method", "rel_pose_score_mode",
                       "fallback_ref_mode", "fallback_replay_attention",
                       "depth_scale_mode", "amp_dtype", "config_name"}:
                setattr(self, key, str(val))
            else:
                setattr(self, key, val)

    def _resolve_config_path(self):
        if os.path.exists(self.config_name):
            return self.config_name

        stem = os.path.splitext(os.path.basename(self.config_name))[0]
        candidates = [
            os.path.join(_R3_ROOT, "configs", f"{stem}.yaml"),
            os.path.join(_R3_ROOT, "R3", "configs", f"{stem}.yaml"),
            os.path.join(_R3_ROOT, "depth_anything_3", "configs", f"{stem}.yaml"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"Cannot find R3 config '{self.config_name}'. Looked in: {candidates}"
        )

    def _default_weight_name(self):
        return "r3_long.safetensors" if self.mode in {"long", "strided"} else "r3.safetensors"

    def _find_weight_in_dir(self, directory):
        preferred = os.path.join(directory, self._default_weight_name())
        if os.path.isfile(preferred):
            return preferred

        candidates = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
        if not candidates:
            candidates = sorted(
                glob.glob(os.path.join(directory, "**", "*.safetensors"), recursive=True)
            )
        if not candidates:
            return None

        stem = os.path.splitext(self._default_weight_name())[0]
        for path in candidates:
            if stem in os.path.basename(path):
                return path
        return candidates[0]

    def _resolve_checkpoint_path(self, checkpoint, weights_dir):
        if checkpoint:
            checkpoint = os.path.expanduser(str(checkpoint))
            if os.path.isfile(checkpoint):
                return checkpoint
            if os.path.isdir(checkpoint):
                found = self._find_weight_in_dir(checkpoint)
                if found:
                    return found
                raise FileNotFoundError(f"No .safetensors R3 weight found in {checkpoint}")
            if checkpoint.endswith((".safetensors", ".pt", ".pth", ".ckpt")) or os.path.sep in checkpoint:
                raise FileNotFoundError(f"R3 checkpoint path does not exist: {checkpoint}")

            snapshot_dir = ensure_hf_snapshot(checkpoint, local_root=weights_dir)
            found = self._find_weight_in_dir(snapshot_dir)
            if found:
                return found
            raise FileNotFoundError(f"No R3 .safetensors weight found in HF snapshot {snapshot_dir}")

        local_dir = os.path.join(weights_dir, "R3")
        found = self._find_weight_in_dir(local_dir) if os.path.isdir(local_dir) else None
        if found:
            return found

        snapshot_dir = ensure_hf_snapshot("KevinXu02/R3", local_root=weights_dir)
        found = self._find_weight_in_dir(snapshot_dir)
        if found:
            return found
        raise FileNotFoundError(
            "No R3 checkpoint found. Expected checkpoints/R3/"
            f"{self._default_weight_name()} or pass checkpoint explicitly."
        )

    @staticmethod
    def _load_state_dict(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            checkpoint = load_file(path, device="cpu")
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    @staticmethod
    def _remap_state_dict(state_dict, model_state):
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("net."):
                key = key[len("net."):]
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = "da3." + key[len("model."):]
            if not key.startswith("da3.") and not key.startswith("projections."):
                da3_key = "da3." + key
                if da3_key in model_state:
                    key = da3_key
            remapped[key] = value

        filtered = {}
        skipped_shape = []
        for key, value in remapped.items():
            if key not in model_state:
                continue
            if tuple(model_state[key].shape) != tuple(value.shape):
                skipped_shape.append(key)
                continue
            filtered[key] = value
        return filtered, skipped_shape

    def _rel_pose_kwargs(self):
        return {
            "topn_conf": int(self.rel_pose_topn_conf),
            "score_mode": self.rel_pose_score_mode,
            "pgo_num_iters": int(self.pgo_num_iters),
            "pgo_lr": float(self.pgo_lr),
            "pgo_weight_T": float(self.pgo_weight_T),
            "pgo_weight_R": float(self.pgo_weight_R),
            "pgo_weight_fl": float(self.pgo_weight_fl),
            "pgo_init_prior_weight": float(self.pgo_init_prior_weight),
            "pgo_keyframe_stride": int(self.pgo_keyframe_stride),
            "edge_percentile_cutoff": float(self.edge_percentile_cutoff),
            "geman_mcclure_c": float(self.pgo_geman_mcclure_c),
            "dcs_phi": float(self.pgo_dcs_phi),
            "max_translation_per_frame": float(self.pgo_max_translation_per_frame),
        }

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
        _activate_r3_imports()
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "[R3Adapter] device='cuda' was requested, but torch.cuda.is_available() "
                "is False in the current environment. Run on a GPU node or override "
                "with --device cpu for a slow CPU-only sanity check."
            )

        from omegaconf import OmegaConf
        from R3.models.r3 import R3

        config_path = self._resolve_config_path()
        cfg = OmegaConf.load(config_path)
        if "model" in cfg and "net" in cfg.model and "da3_cfg" in cfg.model.net:
            da3_cfg = cfg.model.net.da3_cfg
        else:
            da3_cfg = cfg
        if self.attention_mode:
            da3_cfg.net.attention_mode = self.attention_mode
        if int(self.attention_window_size) > 0:
            da3_cfg.net.attention_window_size = int(self.attention_window_size)

        self.model = R3(
            da3_cfg=da3_cfg,
            teacher_embed_dim=2048,
            student_embed_dim=2048,
            freeze="none",
            online_mode=(self.wrapper_mode == "online"),
            online_kv_cache_mode=self.online_kv_cache_mode,
            online_kv_backend=self.online_kv_backend,
            flashinfer_page_size=int(self.flashinfer_page_size),
            online_recent_frames=int(self.online_recent_frames),
            online_verbose=(True if self.online_verbose else None),
            bank_initial_frames=int(self.bank_initial_frames),
            keyframe_mode=self.keyframe_mode,
            keyframe_interval=int(self.keyframe_interval),
            keyframe_novelty_threshold=float(self.keyframe_novelty_threshold),
            keyframe_max_interval=int(self.keyframe_max_interval),
            keyframe_max_keyframes=int(self.keyframe_max_keyframes),
            keyframe_pose_confidence_ratio=float(self.keyframe_pose_confidence_ratio),
            online_fallback_enabled=bool(self.online_fallback_enabled),
            drought_length=int(self.fallback_drought_length),
            drought_threshold=float(self.fallback_drought_threshold),
            drought_threshold_pct=float(self.fallback_drought_threshold_pct),
            drought_threshold_warmup_frames=int(self.fallback_drought_warmup_frames),
            num_bridge_frames=int(self.fallback_num_bridge_frames),
            min_bridge_baseline_ratio=float(self.fallback_min_bridge_baseline_ratio),
            max_bridge_lookback=int(self.fallback_max_bridge_lookback),
            evict_low_conf_threshold=float(self.evict_low_conf_threshold),
            evict_low_conf_threshold_pct=float(self.evict_low_conf_threshold_pct),
            evict_low_conf_warmup_frames=int(self.evict_low_conf_warmup_frames),
            fallback_ref_mode=self.fallback_ref_mode,
            min_segment_frames=int(self.min_segment_frames),
            max_segment_frames=int(self.max_segment_frames),
            fallback_replay_attention=self.fallback_replay_attention,
            fallback_skip_confidence_check=bool(self.fallback_skip_confidence_check),
            depth_scale_mode=self.depth_scale_mode,
            disable_segment_pgo=bool(self.disable_segment_pgo),
            metric_scale_enabled=bool(self.metric_scale_enabled),
            metric_model_name=self.metric_model_name,
            metric_min_conf=float(self.metric_min_conf),
            metric_bootstrap_frames=int(self.metric_bootstrap_frames),
            compute_sky_mask=bool(self.compute_sky_mask),
            sky_mask_threshold=float(self.sky_mask_threshold),
        )

        ckpt_path = self._resolve_checkpoint_path(checkpoint, weights_dir)
        state_dict = self._load_state_dict(ckpt_path)
        model_state = self.model.state_dict()
        filtered, skipped_shape = self._remap_state_dict(state_dict, model_state)
        if not filtered:
            raise RuntimeError(f"R3 checkpoint had no matching model keys: {ckpt_path}")
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        print(f"[R3Adapter] config={config_path}")
        print(f"[R3Adapter] checkpoint={ckpt_path}")
        print(
            f"[R3Adapter] loaded {len(filtered)}/{len(model_state)} tensors "
            f"(missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_skipped={len(skipped_shape)})"
        )
        print(f"[R3Adapter] mode={self.mode}, wrapper_mode={self.wrapper_mode}, device={device}")

    def _prepare_images(self, images_raw):
        n, _, h, w = images_raw.shape
        images = images_raw.unsqueeze(0).to(self.device, non_blocking=True).clone()
        align = max(int(self.patch_size), 1)
        h_proc = ((h + align - 1) // align) * align
        w_proc = ((w + align - 1) // align) * align
        if (h_proc, w_proc) == (h, w):
            return images, (h, w), False

        flat = images.flatten(0, 1)
        flat = F.interpolate(flat, size=(h_proc, w_proc), mode="bilinear", align_corners=False)
        return flat.view(1, n, 3, h_proc, w_proc), (h_proc, w_proc), True

    @staticmethod
    def _resize_nhw(array, size_hw, mode="bilinear"):
        tensor = torch.from_numpy(array).unsqueeze(1)
        resized = F.interpolate(tensor, size=size_hw, mode=mode, align_corners=False)
        return resized[:, 0].numpy()

    @staticmethod
    def _scale_intrinsics(intrinsics, src_hw, dst_hw):
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        scaled = intrinsics.copy()
        scaled[:, 0, :] *= float(dst_w) / float(src_w)
        scaled[:, 1, :] *= float(dst_h) / float(src_h)
        return scaled

    def predict(self, scene):
        if self.model is None:
            raise RuntimeError("R3 model is not loaded")

        _activate_r3_imports()
        from R3.utils.pose_enc import pose_encoding_to_extri_intri

        images_raw = scene["images_raw"]
        n, _, h, w = images_raw.shape
        images, proc_hw, resized = self._prepare_images(images_raw)
        h_proc, w_proc = proc_hw

        amp_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        use_cuda_amp = bool(self.use_amp) and torch.device(self.device).type == "cuda"
        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_cuda_amp else nullcontext()
        )

        rel_pose_kwargs = self._rel_pose_kwargs()
        with torch.no_grad():
            with amp_context:
                if hasattr(self.model, "clear_online_state"):
                    self.model.clear_online_state()
                predictions = self.model(
                    images,
                    mode=self.attention_mode or "causal",
                    use_ray_pose=False,
                    pose_max_recent=int(self.pose_max_recent),
                    bootstrap_full_attention_frames=int(self.bootstrap_full_attention_frames),
                    online_finalize_pose_reconstruction=bool(
                        self.online_finalize_pose_reconstruction
                    ),
                    rel_pose_reconstruction_method=self.rel_pose_reconstruction_method,
                    rel_pose_reconstruction_kwargs=rel_pose_kwargs,
                )

        output_frame_ids = predictions.get("output_frame_ids", list(range(n)))
        output_frame_ids = [int(frame_id) for frame_id in output_frame_ids]
        if output_frame_ids != list(range(n)):
            raise ValueError(
                "R3 returned a subset or reordered frames "
                f"(output_frame_ids={output_frame_ids}); disable output eviction for benchmark evaluation."
            )

        result = {}

        depth = predictions.get("depth")
        if isinstance(depth, torch.Tensor):
            pred_depth = depth[0, :, :, :, 0].detach().cpu().float().numpy()
            if resized:
                pred_depth = self._resize_nhw(pred_depth, (h, w))
            result["pred_depth"] = pred_depth.astype(np.float32)

        depth_conf = predictions.get("depth_conf")
        if isinstance(depth_conf, torch.Tensor):
            pred_conf = depth_conf[0].detach().cpu().float().numpy()
            if resized:
                pred_conf = self._resize_nhw(pred_conf, (h, w))
            result["pred_confidence"] = pred_conf.astype(np.float32)

        pose_enc = predictions.get("pose_enc")
        if isinstance(pose_enc, torch.Tensor):
            w2c_t, intrinsic_t = pose_encoding_to_extri_intri(pose_enc, (h_proc, w_proc))
            w2c = w2c_t[0].detach().cpu().float().numpy().astype(np.float32)
            pred_intrinsic = intrinsic_t[0].detach().cpu().float().numpy().astype(np.float32)
            if resized:
                pred_intrinsic = self._scale_intrinsics(pred_intrinsic, (h_proc, w_proc), (h, w))
            result["w2c_extrinsics"] = w2c
            result["pred_pose"] = self._invert_se3(w2c).astype(np.float32)
            result["pred_intrinsic"] = pred_intrinsic.astype(np.float32)

        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return bool(self.metric_scale_enabled)

    def requires_intrinsics(self):
        return False

    def normalize_gt_poses(self, scene):
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
