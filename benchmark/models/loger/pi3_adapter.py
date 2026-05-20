from __future__ import annotations

import os
import re
import yaml
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from loger.models.pi3 import Pi3
from loger.models.pi3x import Pi3X


@dataclass
class Pi3SequenceOutput:
    """Structured results from a Pi3 forward pass for a single sequence."""

    local_points: torch.Tensor  # (T, H, W, 3) in camera coordinates
    world_points: torch.Tensor  # (T, H, W, 3) in world coordinates
    camera_poses: torch.Tensor  # (T, 4, 4) camera to world
    confidence: Optional[torch.Tensor]  # (T, H, W) confidence map (after sigmoid)
    colors: torch.Tensor  # (T, H, W, 3) RGB in [0, 1]
    avg_gate_scale: Optional[float] = None

    def to(self, device: torch.device) -> "Pi3SequenceOutput":
        return Pi3SequenceOutput(
            local_points=self.local_points.to(device),
            world_points=self.world_points.to(device),
            camera_poses=self.camera_poses.to(device),
            confidence=self.confidence.to(device) if self.confidence is not None else None,
            colors=self.colors.to(device),
            avg_gate_scale=self.avg_gate_scale,
        )


def _maybe_parse_sequence(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = yaml.safe_load(stripped)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
    return value


def _coerce_int_value(value: Any, default: Optional[int], *, field_name: str) -> Optional[int]:
    """Convert config values that may be ranges/strings/lists into a single int."""

    if value is None:
        return default

    # If already numeric, cast directly.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(round(value))

    # Handle sequences by selecting the maximum value (largest receptive field).
    if isinstance(value, (list, tuple)):
        numeric_entries = [
            _coerce_int_value(v, None, field_name=field_name)  # type: ignore[arg-type]
            for v in value
        ]
        numeric_entries = [v for v in numeric_entries if v is not None]
        if numeric_entries:
            return max(numeric_entries)
        return default

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"none", "null", "default", ""}:
            return default
        if stripped.lower() in {"auto", "random"}:
            warnings.warn(
                f"Pi3 forward kwarg '{field_name}' set to '{stripped}', coercing to default {default} for inference.",
                RuntimeWarning,
            )
            return default

        # Extract numbers from strings such as "[4,12]" or "4,12".
        matches = re.findall(r"-?\d+", stripped)
        if matches:
            return max(int(m) for m in matches)

        # Fallback attempt to parse as float, then int.
        try:
            return int(round(float(stripped)))
        except ValueError:
            warnings.warn(
                f"Unable to interpret Pi3 forward kwarg '{field_name}' value '{value}', using default {default}.",
                RuntimeWarning,
            )
            return default

    warnings.warn(
        f"Unsupported type {type(value)} for Pi3 forward kwarg '{field_name}', using default {default}.",
        RuntimeWarning,
    )
    return default


def _sanitize_forward_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(kwargs)

    window_default = kwargs.get("window_size", -1)
    overlap_default = kwargs.get("overlap_size", 0)
    iterations_default = kwargs.get("num_iterations", 1)
    reset_every_default = kwargs.get("reset_every", 0)

    sanitized["window_size"] = _coerce_int_value(window_default, -1, field_name="window_size")
    sanitized["overlap_size"] = _coerce_int_value(overlap_default, 0, field_name="overlap_size")
    sanitized["num_iterations"] = _coerce_int_value(iterations_default, 1, field_name="num_iterations")
    sanitized["reset_every"] = _coerce_int_value(reset_every_default, 0, field_name="reset_every")

    return sanitized


def _load_model_config(config_path: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model construction kwargs and default forward kwargs from yaml config."""
    if config_path is None or not os.path.isfile(config_path):
        return {}, {}

    with open(config_path, "r") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    model_section = raw_cfg.get("model", {})
    training_section = raw_cfg.get("training_settings", {})

    valid_model_keys = {
        "ttt_insert_after",
        "ttt_head_dim",
        "ttt_inter_multi",
        "num_muon_update_steps",
        "use_momentum",
        "ttt_update_steps",
        "conf",
        "attn_insert_after",
        "decoder_size",
        "pos_type",
        "ttt_pre_norm",
        "pi3x",
        "pi3x_metric",
    }

    model_kwargs: Dict[str, Any] = {}
    for key in sorted(valid_model_keys):
        if key in model_section:
            value = model_section[key]
            if key in {"ttt_insert_after", "attn_insert_after"}:
                value = _maybe_parse_sequence(value)
            model_kwargs[key] = value

    forward_kwargs: Dict[str, Any] = {
        "causal": training_section.get("causal", False),
        "window_size": training_section.get("window_size", -1),
        "overlap_size": training_section.get("overlap_size", 0),
        "reset_every": training_section.get("reset_every", 0),
        "window_grad": training_section.get("window_grad", False),
        "num_iterations": raw_cfg.get("num_iterations", 1),
        "sim3": raw_cfg.get("sim3", False),
        "sim3_scale_mode": raw_cfg.get("sim3_scale_mode", "median"),
        "se3": raw_cfg.get("se3", False),
        "objective": model_section.get("ttt_objective"),
        "log_ttt_loss": False,
    }
    return model_kwargs, _sanitize_forward_kwargs(forward_kwargs)


def load_pi3_model(
    weights_path: str,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    **kwargs,
) -> Tuple[Pi3, Dict[str, Any]]:
    """Load a Pi3 model and return it together with default forward kwargs."""
    model_kwargs, forward_kwargs = _load_model_config(config_path)
    model_kwargs.update(kwargs)
    model = Pi3(**model_kwargs)

    if weights_path in ["yyfz233/Pi3", "yyfz233/Pi3X"]:
        model = model.from_pretrained(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location="cpu",weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        info = model.load_state_dict(cleaned, strict=strict)
        if info.missing_keys:
            warnings.warn(f"Missing keys in state dict: {info.missing_keys}")
        if info.unexpected_keys:
            warnings.warn(f"Unexpected keys in state dict: {info.unexpected_keys}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, _sanitize_forward_kwargs(forward_kwargs)


def merge_forward_kwargs(
    base_kwargs: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = dict(base_kwargs)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return _sanitize_forward_kwargs(merged)


def _coerce_view_image(view_img: Any) -> torch.Tensor:
    if isinstance(view_img, torch.Tensor):
        tensor = view_img.detach().clone()
    else:
        tensor = torch.as_tensor(view_img)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    max_val = tensor.max().item() if tensor.numel() > 0 else 0.0
    min_val = tensor.min().item() if tensor.numel() > 0 else 0.0

    if max_val > 2.0:  # assume 0-255
        tensor = tensor / 255.0
    elif min_val < -1.01 or max_val > 1.01:
        # unexpected range, clamp to [0,1]
        tensor = tensor.clamp(0.0, 1.0)

    if min_val < 0.0:
        # assume [-1, 1]
        tensor = (tensor + 1.0) * 0.5
    return tensor.clamp(0.0, 1.0)


def _views_to_image_tensor(views: List[Dict[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images: List[torch.Tensor] = []
    colors: List[torch.Tensor] = []
    for view in views:
        img_tensor = _coerce_view_image(view["img"])
        if img_tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor for view image, got shape {img_tensor.shape}")
        images.append(img_tensor)
        colors.append(img_tensor.permute(1, 2, 0))

    images_tensor = torch.stack(images, dim=0).to(device, non_blocking=True)
    colors_tensor = torch.stack(colors, dim=0)  # keep on CPU for downstream
    return images_tensor, colors_tensor


def run_pi3_inference_on_views(
    model: Pi3,
    views: List[Dict[str, Any]],
    forward_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    output_device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], Pi3SequenceOutput]:
    if device is None:
        device = next(model.parameters()).device

    # Keep images on CPU to avoid OOM for large sequences
    images_tensor, colors_tensor = _views_to_image_tensor(views, torch.device("cpu"))
    batch = images_tensor.unsqueeze(0)  # (1, T, 3, H, W)

    enable_autocast = device.type == "cuda"
    if enable_autocast:
        major = torch.cuda.get_device_capability(device)[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.float32

    fw_kwargs = forward_kwargs or {}
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=enable_autocast
    ):
        predictions = model(batch, **fw_kwargs)

    target_device = output_device or torch.device("cpu")

    local_points = predictions["local_points"].squeeze(0).detach().to(target_device)
    camera_poses = predictions["camera_poses"].squeeze(0).detach().to(target_device)
    world_points = predictions["points"].squeeze(0).detach().to(target_device)

    if camera_poses.shape[0] > 0:
        world_dtype = world_points.dtype
        pose_dtype = camera_poses.dtype

        working_world = world_points.to(torch.float32)
        working_poses = camera_poses.to(torch.float32)

        reference_pose = working_poses[0]
        reference_inv = torch.linalg.inv(reference_pose)

        num_frames, height, width, _ = working_world.shape
        flat_points = working_world.reshape(-1, 3)
        ones = torch.ones(flat_points.shape[0], 1, device=flat_points.device, dtype=flat_points.dtype)
        homog_points = torch.cat([flat_points, ones], dim=-1)
        transformed = (homog_points @ reference_inv.T)[..., :3]
        working_world = transformed.view(num_frames, height, width, 3)

        working_poses = torch.matmul(reference_inv.unsqueeze(0), working_poses)

        world_points = working_world.to(world_dtype)
        camera_poses = working_poses.to(pose_dtype)

    conf_tensor = predictions.get("conf")
    confidence: Optional[torch.Tensor]
    if conf_tensor is not None:
        confidence = (
            torch.sigmoid(conf_tensor).squeeze(0).detach().to(target_device).squeeze(-1)
        )
    else:
        confidence = None

    avg_gate_scale = None
    if "avg_gate_scale" in predictions:
        try:
            avg_gate_scale = float(predictions["avg_gate_scale"].detach().cpu().item())
        except Exception:
            avg_gate_scale = None

    preds: List[Dict[str, torch.Tensor]] = []
    rgb_neg1_pos1 = colors_tensor * 2.0 - 1.0
    num_frames = local_points.shape[0]
    for idx in range(num_frames):
        lp = local_points[idx]
        wp = world_points[idx]
        pose = camera_poses[idx]
        rgb = rgb_neg1_pos1[idx]
        conf_map = None
        if confidence is not None:
            conf_map = confidence[idx]
        else:
            conf_map = torch.ones(lp.shape[:-1], dtype=lp.dtype)

        entry: Dict[str, torch.Tensor] = {
            "pts3d_in_self_view": lp.unsqueeze(0),
            "pts3d_in_other_view": wp.unsqueeze(0),
            "pts3d": wp.unsqueeze(0),
            "camera_pose": pose.unsqueeze(0),
            "rgb": rgb.unsqueeze(0).permute(0, 3, 1, 2),
            "conf_self": conf_map.unsqueeze(0).unsqueeze(-1),
            "conf": conf_map.unsqueeze(0),
        }
        preds.append(entry)

    if confidence is None:
        confidence_tensor = torch.ones(
            (num_frames, local_points.shape[1], local_points.shape[2]),
            dtype=local_points.dtype,
            device=target_device,
        )
    else:
        confidence_tensor = confidence

    sequence_output = Pi3SequenceOutput(
        local_points=local_points,
        world_points=world_points,
        camera_poses=camera_poses,
        confidence=confidence_tensor,
        colors=colors_tensor,
        avg_gate_scale=avg_gate_scale,
    )

    outputs = {
        "pred": preds,
        "views": views,
        "pi3_sequence": sequence_output,
    }
    return outputs, sequence_output
