"""Confidence-based fallback and reanchor mechanism for online inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from R3.utils.pose_utils import (
    average_pose_candidates,
    compose_relative_pose,
    hmat_to_pose_encoding,
    pose_encoding_to_hmat,
)
from R3.models.online_utils import summarize_online_frame_feat


@dataclass(frozen=True)
class DroughtStats:
    """Per-segment confidence summary used for fallback thresholds."""

    segment_frame_ids: list[int]
    warmup_frame_ids: list[int]
    current_confidence: float
    max_confidence: float
    mean_confidence: float
    warmup_confidence: float
    absolute_threshold: float
    percent_threshold: float
    effective_threshold: float


class DroughtDetector:
    """Detect consecutive low-confidence frames (drought condition)."""

    def __init__(self, drought_length: int = 3, drought_threshold: float = 1.0):
        self.drought_length = drought_length
        self.drought_threshold = drought_threshold
        self._window: deque[float] = deque(maxlen=drought_length)
        self._cooldown_remaining: int = 0

    def update(self, confidence: float, threshold: float | None = None) -> bool:
        """Append confidence and return True if drought is triggered."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                # Clear window so cooldown frames don't pre-fill detection window.
                self._window.clear()
            return False
        self._window.append(confidence)
        if len(self._window) < self.drought_length:
            return False
        threshold_value = (
            self.drought_threshold if threshold is None else float(threshold)
        )
        return all(c < threshold_value for c in self._window)

    def set_cooldown(self, num_frames: int):
        """Set cooldown period after a fallback procedure completes."""
        self._cooldown_remaining = num_frames
        self._window.clear()

    def reset(self):
        """Clear all state."""
        self._window.clear()
        self._cooldown_remaining = 0


def weighted_median(values: list[float], weights: list[float]) -> float:
    """Compute weighted median of values."""
    if not values:
        return 0.0
    paired = sorted(zip(values, weights))
    cumulative = 0.0
    total = sum(weights)
    for val, w in paired:
        cumulative += w
        if cumulative >= total / 2.0:
            return val
    return paired[-1][0]


def compute_fallback_blend_weights(old_conf, new_conf, eps: float = 1e-6):
    """Normalize old/new confidence into convex blending weights."""
    if isinstance(old_conf, torch.Tensor) or isinstance(new_conf, torch.Tensor):
        ref_tensor = old_conf if isinstance(old_conf, torch.Tensor) else new_conf
        old_tensor = torch.as_tensor(
            old_conf, dtype=ref_tensor.dtype, device=ref_tensor.device
        ).clamp_min(0.0)
        new_tensor = torch.as_tensor(
            new_conf, dtype=ref_tensor.dtype, device=ref_tensor.device
        ).clamp_min(0.0)
        denom = (old_tensor + new_tensor).clamp_min(eps)
        return old_tensor / denom, new_tensor / denom

    old_value = max(float(old_conf), 0.0)
    new_value = max(float(new_conf), 0.0)
    denom = max(old_value + new_value, eps)
    return old_value / denom, new_value / denom


def resolve_temporal_fallback_ref_id(
    new_ref_id: int, bridge_frame_ids: list[int]
) -> int:
    """Keep fallback replay monotonic when the chosen ref already lies inside the bridge."""
    if not bridge_frame_ids:
        raise ValueError(
            "bridge_frame_ids must not be empty when resolving the fallback ref"
        )
    if new_ref_id in bridge_frame_ids:
        return int(bridge_frame_ids[0])
    return int(new_ref_id)


def resolve_replayable_fallback_ref_id(
    ref_id: int,
    bridge_frame_ids: list[int],
    image_buffer,
    keyframe_registry,
) -> int | None:
    """Return a ref id whose image is reachable for fallback replay.

    The caller must pass a bridge already filtered to frames with images in `image_buffer`,
    so demoting to the first entry is guaranteed safe. Returns None when neither the ref
    nor any bridge frame has a usable image (the caller should bail out in that case).
    """

    def has_image(fid: int) -> bool:
        return (
            image_buffer.has_frame(fid) or keyframe_registry.get_image(fid) is not None
        )

    if has_image(ref_id):
        return int(ref_id)
    for fid in bridge_frame_ids:
        if has_image(fid):
            return int(fid)
    return None


def log_online_pose_edges(
    pose_edge_log,
    state,
    current_frame_id: int,
    rel_pose_frame_ids,
    rel_pose_enc: torch.Tensor,
    rel_pose_conf: torch.Tensor,
    rel_pose_conf_t: torch.Tensor | None,
    rel_pose_conf_r: torch.Tensor | None,
    rel_pose_mask: torch.Tensor,
    in_fallback: bool,
):
    """Persist direct relative edges so fallback can reuse predicted poses."""
    if rel_pose_frame_ids is None:
        return

    current_idx = len(rel_pose_frame_ids) - 1
    memory_frame_ids = rel_pose_frame_ids[:-1]
    if not memory_frame_ids:
        return

    edge_pose_enc = rel_pose_enc[:, :current_idx, current_idx]
    edge_mask = rel_pose_mask[:, :current_idx, current_idx].bool()
    valid_edge_mask = edge_mask.any(dim=0)
    if not valid_edge_mask.any():
        return

    edge_conf = F.softplus(rel_pose_conf[:, :current_idx, current_idx].detach().float())
    valid_counts = edge_mask.sum(dim=0).clamp_min(1)
    edge_conf = edge_conf.masked_fill(~edge_mask, 0.0).sum(dim=0) / valid_counts

    edge_conf_t = None
    edge_conf_r = None
    if isinstance(rel_pose_conf_t, torch.Tensor) and isinstance(
        rel_pose_conf_r, torch.Tensor
    ):
        edge_conf_t = F.softplus(
            rel_pose_conf_t[:, :current_idx, current_idx].detach().float()
        )
        edge_conf_t = edge_conf_t.masked_fill(~edge_mask, 0.0).sum(dim=0) / valid_counts
        edge_conf_r = F.softplus(
            rel_pose_conf_r[:, :current_idx, current_idx].detach().float()
        )
        edge_conf_r = edge_conf_r.masked_fill(~edge_mask, 0.0).sum(dim=0) / valid_counts

    keep_indices = valid_edge_mask.nonzero(as_tuple=False).flatten().tolist()
    keep_frame_ids = [memory_frame_ids[idx] for idx in keep_indices]
    pose_edge_log.add_edges_from_step(
        current_frame_id=current_frame_id,
        memory_frame_ids=keep_frame_ids,
        rel_pose_enc=edge_pose_enc[:, keep_indices],
        confidences=edge_conf[keep_indices],
        confidences_t=edge_conf_t[keep_indices] if edge_conf_t is not None else None,
        confidences_r=edge_conf_r[keep_indices] if edge_conf_r is not None else None,
        edge_type=("bridge" if in_fallback else "normal"),
        scale_factor=state.scale_factor,
    )


def get_logged_relative_pose(
    pose_edge_log,
    ref_frame_id: int,
    target_frame_id: int,
    edge_type: str,
    like_pose: torch.Tensor,
):
    """Fetch a stored relative edge and move it onto the active device."""
    edge = pose_edge_log.get_edge(ref_frame_id, target_frame_id, edge_type=edge_type)
    if edge is None:
        return None
    return edge.rel_pose_enc.to(device=like_pose.device, dtype=like_pose.dtype)


def invert_logged_relative_pose(
    rel_pose_enc: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    """Invert a logged relative pose while preserving the new target frame intrinsics."""
    rel_hmat = pose_encoding_to_hmat(rel_pose_enc)
    inv_rel_hmat = torch.linalg.inv(rel_hmat)
    return hmat_to_pose_encoding(inv_rel_hmat, target_pose[..., 7:9])


def identity_relative_pose(target_pose: torch.Tensor) -> torch.Tensor:
    """Build an identity relative pose targeting the provided frame intrinsics."""
    rel_pose = torch.zeros_like(target_pose)
    rel_pose[..., 6] = 1.0
    rel_pose[..., 7:9] = target_pose[..., 7:9]
    return rel_pose


def resolve_fallback_bridge_rel_pose(
    pose_edge_log,
    ref_frame_id: int,
    target_frame_id: int,
    edge_type: str,
    ref_pose: torch.Tensor,
    target_pose: torch.Tensor,
    allow_missing: bool = False,
    fallback_edge_types: tuple[str, ...] = (),
) -> torch.Tensor | None:
    """Return the stored bridge relative edge for the pair or raise when absent.

    `fallback_edge_types` lets callers accept alternate edge types when the requested type is
    absent — needed for the old log after a prior accepted fallback rewrites it with bridge-only
    edges, so a chained fallback's "normal" lookup can still find the geometrically equivalent
    bridge edge logged during the previous replay.
    """
    if ref_frame_id == target_frame_id:
        return identity_relative_pose(target_pose)

    for et in (edge_type, *fallback_edge_types):
        logged_rel_pose = get_logged_relative_pose(
            pose_edge_log=pose_edge_log,
            ref_frame_id=ref_frame_id,
            target_frame_id=target_frame_id,
            edge_type=et,
            like_pose=target_pose,
        )
        if logged_rel_pose is not None:
            return logged_rel_pose

        reverse_rel_pose = get_logged_relative_pose(
            pose_edge_log=pose_edge_log,
            ref_frame_id=target_frame_id,
            target_frame_id=ref_frame_id,
            edge_type=et,
            like_pose=target_pose,
        )
        if reverse_rel_pose is not None:
            return invert_logged_relative_pose(reverse_rel_pose, target_pose)

    if allow_missing:
        return None

    raise ValueError(
        "Missing logged fallback relative pose edge pair for frames "
        f"({ref_frame_id}, {target_frame_id}) with edge_type={edge_type}"
    )


def fuse_fallback_bridge_pose(
    old_ref_pose: torch.Tensor,
    old_rel_pose: torch.Tensor,
    new_rel_pose: torch.Tensor,
    old_score: float,
    new_score: float,
    scale: float,
) -> torch.Tensor:
    """Blend old/new bridge motion relative to the preserved old anchor pose."""
    new_rel_pose = new_rel_pose.clone()
    new_rel_pose[..., :3] *= float(scale)

    old_weight, new_weight = compute_fallback_blend_weights(old_score, new_score)
    candidate_rel_pose = torch.stack([old_rel_pose, new_rel_pose], dim=1)
    weights = (
        candidate_rel_pose.new_tensor([old_weight, new_weight])
        .unsqueeze(0)
        .expand(candidate_rel_pose.shape[0], -1)
    )
    fused_rel_pose = average_pose_candidates(candidate_rel_pose, weights)
    fused_abs_pose = compose_relative_pose(fused_rel_pose, old_ref_pose.unsqueeze(1))
    return fused_abs_pose[:, 0]


def fuse_fallback_depth_pair(
    old_depth: torch.Tensor,
    new_depth: torch.Tensor,
    old_conf,
    new_conf,
    scale: float,
):
    """Blend old/new bridge depth into the accepted segment's raw depth scale."""
    old_depth_in_new_scale = old_depth.clone()
    if math.isfinite(float(scale)) and float(scale) > 0.0:
        old_depth_in_new_scale = old_depth_in_new_scale / float(scale)

    if isinstance(old_conf, torch.Tensor) and isinstance(new_conf, torch.Tensor):
        old_weight, new_weight = compute_fallback_blend_weights(
            old_conf.float(), new_conf.float()
        )
        while old_weight.dim() < old_depth_in_new_scale.dim():
            old_weight = old_weight.unsqueeze(-1)
            new_weight = new_weight.unsqueeze(-1)
        fused_depth = old_weight * old_depth_in_new_scale + new_weight * new_depth
        fused_conf = old_conf + new_conf
        return fused_depth, fused_conf

    if isinstance(new_conf, torch.Tensor):
        return new_depth, new_conf
    if isinstance(old_conf, torch.Tensor):
        return old_depth_in_new_scale, old_conf

    return 0.5 * (old_depth_in_new_scale + new_depth), None


class ScaleEstimator:
    """Estimate scale factor from bridge frame poses in old and new coordinate systems."""

    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon

    def estimate(
        self,
        old_poses: list[torch.Tensor],
        new_poses: list[torch.Tensor],
        confidences: list[float],
    ) -> float:
        """Estimate scale from pair-wise distance ratios.

        Args:
            old_poses: Bridge frame poses in old coord system, each [B, 9] (pose_enc format).
            new_poses: Bridge frame poses in new coord system, each [B, 9].
            confidences: Per-frame confidence scores.

        Returns:
            Scale factor s such that old_translation ≈ s * new_translation.
        """
        n = len(old_poses)
        ratios = []
        weights = []

        for i in range(n):
            for j in range(i + 1, n):
                d_old = (old_poses[i][0, :3] - old_poses[j][0, :3]).norm().item()
                d_new = (new_poses[i][0, :3] - new_poses[j][0, :3]).norm().item()
                if d_old < self.epsilon or d_new < self.epsilon:
                    continue
                ratios.append(d_old / d_new)
                weights.append(confidences[i] * confidences[j])

        if len(ratios) < 2:
            return 1.0

        return weighted_median(ratios, weights)


@dataclass
class PoseEdge:
    """A single pose graph edge."""

    frame_i: int
    frame_j: int
    rel_pose_enc: torch.Tensor  # [B, 9] pose encoding
    confidence: float
    edge_type: str  # "normal" | "bridge" | "anchor"
    confidence_t: float | None = None
    confidence_r: float | None = None


class PoseEdgeLog:
    """Append-only storage for pose graph edges."""

    def __init__(self):
        self._edges: list[PoseEdge] = []

    def add_edge(
        self,
        frame_i: int,
        frame_j: int,
        rel_pose_enc: torch.Tensor,
        confidence: float,
        edge_type: str,
        confidence_t: float | None = None,
        confidence_r: float | None = None,
    ):
        """Add a single edge."""
        self._edges.append(
            PoseEdge(
                frame_i,
                frame_j,
                rel_pose_enc.detach().cpu(),
                confidence,
                edge_type,
                confidence_t,
                confidence_r,
            )
        )

    def add_edges_from_step(
        self,
        current_frame_id: int,
        memory_frame_ids: list[int],
        rel_pose_enc: torch.Tensor,
        confidences: torch.Tensor,
        edge_type: str,
        confidences_t: torch.Tensor | None = None,
        confidences_r: torch.Tensor | None = None,
        scale_factor: float = 1.0,
    ):
        """Add edges from a forward step. Applies scale to translation."""
        for idx, mem_id in enumerate(memory_frame_ids):
            pose = rel_pose_enc[:, idx].detach().cpu().clone()
            if scale_factor != 1.0:
                pose[:, :3] *= scale_factor
            conf = float(confidences[idx])
            conf_t = float(confidences_t[idx]) if confidences_t is not None else None
            conf_r = float(confidences_r[idx]) if confidences_r is not None else None
            self._edges.append(
                PoseEdge(
                    mem_id, current_frame_id, pose, conf, edge_type, conf_t, conf_r
                )
            )

    def get_all_edges(self) -> list[PoseEdge]:
        """Return all stored edges."""
        return list(self._edges)

    def get_edge(
        self, frame_i: int, frame_j: int, edge_type: str | None = None
    ) -> PoseEdge | None:
        """Return the newest matching edge for a frame pair."""
        for edge in reversed(self._edges):
            if edge.frame_i != frame_i or edge.frame_j != frame_j:
                continue
            if edge_type is not None and edge.edge_type != edge_type:
                continue
            return edge
        return None

    def drop_frames(self, frame_ids: set[int]):
        """Remove all edges that touch dropped frames."""
        self._edges = [
            edge
            for edge in self._edges
            if edge.frame_i not in frame_ids and edge.frame_j not in frame_ids
        ]


class ImageRingBuffer:
    """CPU-side ring buffer for frame images, used for bridge frame re-runs."""

    def __init__(self, capacity: int = 12):
        self.capacity = capacity
        self._buffer: dict[int, torch.Tensor] = {}
        self._order: list[int] = []

    def push(self, frame_id: int, image: torch.Tensor):
        """Store a frame image. Evicts oldest if at capacity."""
        if frame_id in self._buffer:
            self._buffer[frame_id] = image.detach().cpu()
            return
        while len(self._order) >= self.capacity:
            evict_id = self._order.pop(0)
            self._buffer.pop(evict_id, None)
        self._buffer[frame_id] = image.detach().cpu()
        self._order.append(frame_id)

    def get(self, frame_id: int) -> torch.Tensor | None:
        """Retrieve image by frame ID, or None if not available."""
        return self._buffer.get(frame_id)

    def get_multiple(self, frame_ids: list[int]) -> list[torch.Tensor]:
        """Retrieve multiple images. Raises KeyError if any ID is missing."""
        result = []
        for fid in frame_ids:
            img = self._buffer.get(fid)
            if img is None:
                raise KeyError(f"Frame {fid} not in image buffer")
            result.append(img)
        return result

    def has_frame(self, frame_id: int) -> bool:
        """Check if a frame ID is in the buffer."""
        return frame_id in self._buffer


@dataclass
class FallbackAction:
    """Describes a fallback procedure to execute."""

    fallback_frame_id: int
    new_ref_id: int
    bridge_frame_ids: list[int]
    bad_frame_ids: list[int]
    # True when triggered structurally (max_segment_frames) rather than by drought.
    # Forced actions skip the bridge-improvement gate so the KV cache is always cut,
    # even when the rerun does not improve confidence.
    forced: bool = False


class FallbackManager:
    """Orchestrate confidence-based fallback decisions."""

    def __init__(
        self,
        enabled: bool = False,
        drought_length: int = 3,
        drought_threshold: float = 1.0,
        drought_threshold_pct: float = 0.0,
        drought_threshold_warmup_frames: int = 5,
        num_bridge_frames: int = 5,
        min_bridge_baseline_ratio: float = 0.0,
        max_bridge_lookback: int = 0,
        fallback_scale_epsilon: float = 1e-4,
    ):
        self.enabled = enabled
        self.num_bridge_frames = num_bridge_frames
        self.drought_length = drought_length
        self.drought_threshold_pct = max(float(drought_threshold_pct), 0.0)
        self.drought_threshold_warmup_frames = max(
            int(drought_threshold_warmup_frames), 0
        )
        # Parallax-aware bridge selection. When `min_bridge_baseline_ratio > 0`, the manager
        # walks backward through up to `max_bridge_lookback` good frames and accepts only
        # candidates whose translation delta against the last accepted bridge frame exceeds
        # `min_bridge_baseline_ratio * gate_ref`, where `gate_ref` is the 75th percentile of
        # pairwise translation deltas in a broad recent window. Setting `min_bridge_baseline_ratio
        # = 0.0` keeps the legacy consecutive selection.
        self.min_bridge_baseline_ratio = max(float(min_bridge_baseline_ratio), 0.0)
        self.max_bridge_lookback = max(int(max_bridge_lookback), 0)
        self.fallback_scale_epsilon = max(float(fallback_scale_epsilon), 0.0)
        self._detector = DroughtDetector(drought_length, drought_threshold)
        effective_lookback = max(num_bridge_frames, self.max_bridge_lookback)
        max_history = drought_length + effective_lookback + 5
        self._recent_frame_ids: deque[int] = deque(maxlen=max_history)

    def get_drought_stats(
        self, state, current_frame_id: int, confidence: float | None = None
    ) -> DroughtStats:
        """Summarize confidence over the current segment for adaptive thresholds."""
        segment_start = int(getattr(state, "last_fallback_frame_id", -1)) + 1
        segment_frame_ids = [
            int(fid)
            for fid in getattr(state, "frame_order", [])
            if int(fid) >= segment_start and int(fid) <= current_frame_id
        ]
        if current_frame_id not in segment_frame_ids:
            segment_frame_ids.append(int(current_frame_id))
            segment_frame_ids = sorted(dict.fromkeys(segment_frame_ids))

        current_confidence = confidence
        if current_confidence is None:
            current_confidence = getattr(state, "frame_post_scores", {}).get(
                current_frame_id, 0.0
            )
        current_confidence = float(current_confidence)
        if not math.isfinite(current_confidence):
            current_confidence = 0.0

        frame_post_scores = getattr(state, "frame_post_scores", {})
        segment_scores = []
        for fid in segment_frame_ids:
            score = (
                current_confidence
                if fid == current_frame_id
                else frame_post_scores.get(fid, None)
            )
            if score is None:
                continue
            score = float(score)
            if math.isfinite(score):
                segment_scores.append(score)

        if segment_scores:
            max_confidence = max(segment_scores)
            mean_confidence = sum(segment_scores) / len(segment_scores)
        else:
            max_confidence = current_confidence
            mean_confidence = current_confidence

        warmup_candidates = list(segment_frame_ids)
        if segment_start == 0 and warmup_candidates and warmup_candidates[0] == 0:
            warmup_candidates = warmup_candidates[1:]
        if self.drought_threshold_warmup_frames > 0:
            warmup_frame_ids = warmup_candidates[: self.drought_threshold_warmup_frames]
        else:
            warmup_frame_ids = warmup_candidates

        warmup_scores = []
        for fid in warmup_frame_ids:
            score = (
                current_confidence
                if fid == current_frame_id
                else frame_post_scores.get(fid, None)
            )
            if score is None:
                continue
            score = float(score)
            if math.isfinite(score):
                warmup_scores.append(score)

        warmup_confidence = (
            sum(warmup_scores) / len(warmup_scores) if warmup_scores else 0.0
        )
        percent_threshold = 0.0
        if self.drought_threshold_pct > 0.0 and warmup_confidence > 0.0:
            percent_threshold = warmup_confidence * (self.drought_threshold_pct / 100.0)

        absolute_threshold = float(self._detector.drought_threshold)
        effective_threshold = max(absolute_threshold, percent_threshold)
        return DroughtStats(
            segment_frame_ids=segment_frame_ids,
            warmup_frame_ids=warmup_frame_ids,
            current_confidence=current_confidence,
            max_confidence=max_confidence,
            mean_confidence=mean_confidence,
            warmup_confidence=warmup_confidence,
            absolute_threshold=absolute_threshold,
            percent_threshold=percent_threshold,
            effective_threshold=effective_threshold,
        )

    def _select_bridge_frames(self, state, all_good_frames: list[int]) -> list[int] | None:
        """Pick bridge frames spanning enough baseline.

        Default behavior (`min_bridge_baseline_ratio == 0`): take the last `num_bridge_frames`
        consecutive good frames. With parallax gating enabled, walk backward through up to
        `max_bridge_lookback` good frames and accept a candidate only when its translation
        delta against the last accepted bridge frame exceeds `min_bridge_baseline_ratio *
        gate_ref`, where `gate_ref` is the 75th percentile of pairwise translation deltas over
        a broad recent window. Returns `None` when even the broadest window has no motion
        above `fallback_scale_epsilon`, so the caller can abort fallback instead of replaying
        a degenerate bridge.
        """
        if not all_good_frames:
            return None

        fb_idx = len(all_good_frames) - 1

        if self.min_bridge_baseline_ratio <= 0.0:
            bridge_start = max(0, fb_idx - self.num_bridge_frames + 1)
            return list(all_good_frames[bridge_start : fb_idx + 1])

        pose_enc = getattr(state, "frame_pose_enc", {}) or {}
        window_size = min(
            len(all_good_frames),
            max(self.num_bridge_frames * 4, self.max_bridge_lookback * 2, 20),
        )
        recent = all_good_frames[-window_size:]
        deltas: list[float] = []
        for a, b in zip(recent[:-1], recent[1:]):
            pa = pose_enc.get(a)
            pb = pose_enc.get(b)
            if pa is None or pb is None:
                continue
            try:
                d = float(
                    torch.linalg.norm(pa[..., :3].flatten() - pb[..., :3].flatten())
                )
            except Exception:
                continue
            if math.isfinite(d):
                deltas.append(d)

        if not deltas:
            bridge_start = max(0, fb_idx - self.num_bridge_frames + 1)
            return list(all_good_frames[bridge_start : fb_idx + 1])

        sorted_deltas = sorted(deltas)
        max_delta = sorted_deltas[-1]
        if max_delta < self.fallback_scale_epsilon:
            return None

        p75_idx = max(0, (len(sorted_deltas) * 3) // 4)
        gate_ref = max(sorted_deltas[p75_idx], self.fallback_scale_epsilon)
        min_delta = self.min_bridge_baseline_ratio * gate_ref

        lookback_limit = (
            self.max_bridge_lookback if self.max_bridge_lookback > 0 else self.num_bridge_frames
        )
        lookback_limit = max(lookback_limit, self.num_bridge_frames)
        earliest_idx = max(0, fb_idx - lookback_limit + 1)

        selected: list[int] = []
        last_pose = None
        for i in range(fb_idx, earliest_idx - 1, -1):
            if len(selected) >= self.num_bridge_frames:
                break
            fid = all_good_frames[i]
            pose = pose_enc.get(fid)
            if pose is None:
                continue
            if not selected:
                selected.append(fid)
                last_pose = pose
                continue
            try:
                d = float(
                    torch.linalg.norm(
                        pose[..., :3].flatten() - last_pose[..., :3].flatten()
                    )
                )
            except Exception:
                continue
            if not math.isfinite(d):
                continue
            if d >= min_delta:
                selected.append(fid)
                last_pose = pose

        if not selected:
            return None
        selected.sort()
        return selected

    def maybe_trigger(
        self, state, confidence: float, current_frame_id: int
    ) -> FallbackAction | None:
        """Check if fallback should trigger. Returns action or None."""
        if not self.enabled:
            return None

        stats = self.get_drought_stats(state, current_frame_id, confidence=confidence)
        confidence = stats.current_confidence
        self._recent_frame_ids.append(current_frame_id)

        if not self._detector.update(confidence, threshold=stats.effective_threshold):
            return None

        # Drought is a capacity signal; true bad-frame removal is handled by eviction.
        replayable_frames = list(state.frame_order)
        if not replayable_frames:
            self._detector.reset()
            return None
        fallback_frame_id = replayable_frames[-1]

        bridge_frame_ids = self._select_bridge_frames(state, replayable_frames)

        # None signals a degenerate-baseline segment; <2 frames cannot anchor scale.
        if bridge_frame_ids is None or len(bridge_frame_ids) < 2:
            self._detector.reset()
            return None

        # Select new ref: highest-scoring KV cache frame with positive score.
        # Fall back to fallback_frame_id if no such candidate exists.
        candidates = [
            fid
            for fid in state.cache_frame_ids
            if state.frame_scores.get(fid, 0.0) > 0.0
        ]
        if candidates:
            new_ref_id = max(
                candidates, key=lambda fid: state.frame_scores.get(fid, 0.0)
            )
        else:
            new_ref_id = fallback_frame_id

        return FallbackAction(
            fallback_frame_id=fallback_frame_id,
            new_ref_id=new_ref_id,
            bridge_frame_ids=bridge_frame_ids,
            bad_frame_ids=[],
        )

    def on_fallback_complete(self):
        """Called after fallback procedure finishes. Sets cooldown."""
        cooldown = self.num_bridge_frames + self.drought_length
        self._detector.set_cooldown(cooldown)
        self._recent_frame_ids.clear()


class KeyframeRegistry:
    """Stores keyframes for dynamic retention and fallback ref lookup."""

    def __init__(
        self,
        interval: int = 10,
        mode: str = "interval",
        novelty_threshold: float = 0.98,
        max_interval: int = 30,
        max_keyframes: int = 100,
        pose_confidence_ratio: float = 0.0,
        score_provider=None,
    ):
        self.interval = max(int(interval), 1)
        self.mode = str(mode)
        self.novelty_threshold = float(novelty_threshold)
        self.max_interval = max(int(max_interval), 1)
        self.max_keyframes = max(int(max_keyframes), 0)
        # Pose-confidence gate: keyframes are admitted only when reliability >= warmup_mean *
        # pose_confidence_ratio. 0.0 disables the gate; TTT uses 0.8 by default but R3's CLI
        # currently passes 0 (gate off) to avoid stalling the bank during long stationary phases.
        self.pose_confidence_ratio = max(float(pose_confidence_ratio), 0.0)
        if self.mode not in {"interval", "novelty"}:
            raise ValueError(f"Unsupported keyframe mode: {self.mode}")
        # Eviction queries state.frame_scores through this callable. None disables
        # score-aware eviction and falls back to dropping the oldest keyframe.
        self.score_provider = score_provider
        self._keyframes: dict[int, dict] = {}  # frame_id -> {feature, image, pose_enc}
        self._frame_count = 0
        self._frames_since_last_add = 0

    def maybe_add(
        self,
        frame_id: int,
        feature: torch.Tensor,
        image: torch.Tensor,
        pose_enc: torch.Tensor,
        reliability: float = 0.0,
        warmup_confidence: float | None = None,
    ):
        """Add keyframes using fixed-interval or novelty-gated selection.

        `warmup_confidence` is the current segment's warmup-window mean post-softplus score
        (see FallbackManager.get_drought_stats). It anchors the pose-confidence gate so the
        admission bar is fixed by the segment's early behavior. Pass None (or <=0) to bypass.
        """
        if self.max_keyframes == 0:
            return

        self._frame_count += 1
        self._frames_since_last_add += 1
        if self.mode == "novelty":
            self._add_novelty_based(
                frame_id, feature, image, pose_enc, reliability, warmup_confidence
            )
            return
        self._add_interval_based(
            frame_id, feature, image, pose_enc, reliability, warmup_confidence
        )

    def _add_interval_based(
        self,
        frame_id: int,
        feature: torch.Tensor,
        image: torch.Tensor,
        pose_enc: torch.Tensor,
        reliability: float,
        warmup_confidence: float | None,
    ):
        """Store a keyframe every configured interval."""
        if self._frame_count % self.interval == 0:
            if not self._passes_pose_confidence_gate(
                frame_id, reliability, warmup_confidence
            ):
                return
            self._store_keyframe(
                frame_id, feature, image, pose_enc, reliability=reliability
            )
            self._frames_since_last_add = 0

    def _add_novelty_based(
        self,
        frame_id: int,
        feature: torch.Tensor,
        image: torch.Tensor,
        pose_enc: torch.Tensor,
        reliability: float,
        warmup_confidence: float | None,
    ):
        """Store a keyframe when it is novel enough or the gap grows too large."""
        if not self._keyframes:
            self._store_keyframe(
                frame_id, feature, image, pose_enc, reliability=reliability
            )
            self._frames_since_last_add = 0
            return

        if not self._passes_pose_confidence_gate(
            frame_id, reliability, warmup_confidence
        ):
            return

        if self._frames_since_last_add >= self.max_interval:
            self._store_keyframe(
                frame_id, feature, image, pose_enc, reliability=reliability
            )
            self._frames_since_last_add = 0
            return

        summary = self._summarize_feature(feature)
        max_similarity = self._max_similarity_to_bank(summary)
        if max_similarity < self.novelty_threshold:
            self._store_keyframe(
                frame_id,
                feature,
                image,
                pose_enc,
                summary=summary,
                reliability=reliability,
            )
            self._frames_since_last_add = 0

    def _passes_pose_confidence_gate(
        self,
        frame_id: int,
        reliability: float,
        warmup_confidence: float | None,
    ) -> bool:
        """Gate keyframe insertion against the segment's warmup-window mean confidence."""
        if int(frame_id) == 0:
            return True
        if self.pose_confidence_ratio <= 0.0:
            return True
        if warmup_confidence is None:
            return True
        warmup_baseline = float(warmup_confidence)
        if not math.isfinite(warmup_baseline) or warmup_baseline <= 0.0:
            return True
        return float(reliability) >= warmup_baseline * self.pose_confidence_ratio

    def _summarize_feature(self, feature: torch.Tensor | None) -> torch.Tensor | None:
        """Build a normalized summary vector for novelty and similarity checks."""
        summary = summarize_online_frame_feat(feature)
        return summary.detach().cpu() if isinstance(summary, torch.Tensor) else None

    def _store_keyframe(
        self,
        frame_id: int,
        feature: torch.Tensor,
        image: torch.Tensor,
        pose_enc: torch.Tensor,
        summary: torch.Tensor | None = None,
        reliability: float = 0.0,
    ):
        """Persist one keyframe and evict the most redundant entry at capacity."""
        if self.max_keyframes == 0:
            return

        if summary is None:
            summary = self._summarize_feature(feature)

        if len(self._keyframes) >= self.max_keyframes:
            self._evict_lowest_score()

        self._keyframes[frame_id] = {
            "feature": feature.detach().cpu(),
            "image": image.detach().cpu(),
            "pose_enc": pose_enc.detach().cpu() if pose_enc is not None else None,
            "summary": summary,
            "reliability": float(reliability),
        }

    def _get_similarity_vector(self, keyframe: dict) -> torch.Tensor | None:
        """Return the normalized vector used to compare one stored keyframe."""
        summary = keyframe.get("summary")
        if isinstance(summary, torch.Tensor):
            return summary.float().flatten()

        feature = keyframe.get("feature")
        summary = self._summarize_feature(feature)
        if summary is not None:
            keyframe["summary"] = summary
            return summary.float().flatten()
        return None

    def _max_similarity_to_bank(self, summary: torch.Tensor | None) -> float:
        """Compute the largest cosine similarity between a summary and the current bank."""
        if summary is None or not self._keyframes:
            return -1.0

        query = summary.float().flatten()
        best_similarity = -1.0
        for keyframe in self._keyframes.values():
            vector = self._get_similarity_vector(keyframe)
            if vector is None:
                continue
            similarity = float((query * vector).sum())
            if similarity > best_similarity:
                best_similarity = similarity
        return best_similarity

    def _evict_lowest_score(self):
        """Drop the keyframe with the lowest score_provider value, tie-break by oldest frame_id."""
        if len(self._keyframes) < 2:
            return

        if self.score_provider is None:
            # Fall back to oldest-first when no score signal is wired in.
            oldest_frame_id = min(self._keyframes.keys())
            del self._keyframes[oldest_frame_id]
            return

        worst_frame_id = min(
            self._keyframes.keys(),
            key=lambda frame_id: (float(self.score_provider(frame_id)), frame_id),
        )
        del self._keyframes[worst_frame_id]

    def get_latest_before(
        self,
        frame_id: int,
        allowed_ids: set[int] | None = None,
        excluded: set[int] | None = None,
    ) -> int | None:
        """Return the nearest stored keyframe strictly before the given frame ID.

        `allowed_ids` constrains the search to that subset (e.g. currently banked frames);
        `excluded` skips refs that already failed a fallback attempt.
        """
        excluded_set = {int(fid) for fid in (excluded or ())}
        candidates = [
            keyframe_id
            for keyframe_id in self._keyframes
            if keyframe_id < frame_id and keyframe_id not in excluded_set
        ]
        if allowed_ids is not None:
            allowed_set = {int(fid) for fid in allowed_ids}
            candidates = [
                keyframe_id for keyframe_id in candidates if keyframe_id in allowed_set
            ]
        if not candidates:
            return None
        return max(candidates)

    def select_fallback_ref(
        self,
        target_frame_id: int,
        top_k: int = 1,
        recency_alpha: float = 0.1,
        excluded: set[int] | None = None,
        allowed_ids: set[int] | None = None,
    ) -> list[int]:
        """Return keyframe IDs ranked by stored reliability plus a small recency bonus."""
        excluded_set = {int(fid) for fid in (excluded or ())}
        allowed_set = (
            {int(fid) for fid in allowed_ids} if allowed_ids is not None else None
        )
        candidates = [
            keyframe_id
            for keyframe_id in self._keyframes
            if keyframe_id < target_frame_id
            and keyframe_id not in excluded_set
            and (allowed_set is None or keyframe_id in allowed_set)
        ]
        if not candidates:
            return []

        reliabilities = {
            frame_id: float(self._keyframes[frame_id].get("reliability", 0.0))
            for frame_id in candidates
        }
        if all(score == 0.0 for score in reliabilities.values()):
            return [max(candidates)][:top_k]

        min_id = min(candidates)
        span = max(target_frame_id - min_id, 1)

        def score(frame_id: int) -> tuple[float, int]:
            recency_norm = (frame_id - min_id) / span
            return (reliabilities[frame_id] + recency_alpha * recency_norm, frame_id)

        return sorted(candidates, key=score, reverse=True)[:top_k]

    def get_image(self, frame_id: int) -> torch.Tensor | None:
        """Get stored keyframe image."""
        kf = self._keyframes.get(frame_id)
        return kf["image"] if kf else None

    def get_pose(self, frame_id: int) -> torch.Tensor | None:
        """Get stored keyframe pose encoding."""
        kf = self._keyframes.get(frame_id)
        return kf["pose_enc"] if kf else None

    def clear(self):
        """Clear all keyframes."""
        self._keyframes.clear()
        self._frame_count = 0
        self._frames_since_last_add = 0

    def get_keyframe_ids(self) -> list[int]:
        """Return stored keyframe IDs in insertion order."""
        return list(self._keyframes.keys())

    def __len__(self):
        return len(self._keyframes)
