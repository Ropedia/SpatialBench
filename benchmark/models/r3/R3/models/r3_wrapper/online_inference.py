"""Online inference methods for the R3 wrapper."""

import copy
import logging
import math

import addict
import torch
import torch.nn.functional as F

from R3.models.online.fallback import (
    fuse_fallback_bridge_pose,
    fuse_fallback_depth_pair,
    log_online_pose_edges,
    resolve_fallback_bridge_rel_pose,
    resolve_replayable_fallback_ref_id,
    resolve_temporal_fallback_ref_id,
)
from R3.models.online.kv_cache import (
    evict_frame_from_kv as _evict_frame_from_kv_impl,
    get_dynamic_segment_anchor_budget as _get_dynamic_segment_anchor_budget_impl,
    get_dynamic_segment_frame_order as _get_dynamic_segment_frame_order_impl,
    get_online_cache_keep_frame_ids as _get_online_cache_keep_frame_ids_impl,
    prune_kv_cache_list as _prune_kv_cache_list_impl,
    prune_online_similarity_cache as _prune_online_similarity_cache_impl,
    prune_online_state as _prune_online_state_impl,
    set_dynamic_segment_anchor_frame_ids as _set_dynamic_segment_anchor_frame_ids_impl,
    sync_dynamic_memory_bank_ids as _sync_dynamic_memory_bank_ids_impl,
    upgrade_kv_cache_to_buffers as _upgrade_kv_cache_to_buffers_impl,
)
from R3.models.online.pose_resolution import finalize_online_pose_sequence
from R3.models.online.revisit import resolve_online_verbose, run_online_sequence_pass
from R3.models.online.scale_estimation import (
    compute_metric_scale_factor as _compute_metric_scale_factor_impl,
    estimate_scale_from_depth as _estimate_scale_from_depth_impl,
    fallback_improves_bridge_scores as _fallback_improves_bridge_scores_impl,
    resolve_fallback_scale as _resolve_fallback_scale_impl,
)
from R3.models.online.state import OnlineState
from R3.models.r3_wrapper.constants import ONLINE_WRAPPER_OPTION_KEYS
from R3.models.r3_wrapper.paged_kv import (
    ensure_paged_kv_store as _ensure_paged_kv_store_impl,
    populate_paged_kv_from_replay as _populate_paged_kv_from_replay_impl,
    prepare_paged_kv_da3_kwargs as _prepare_paged_kv_da3_kwargs_impl,
    uses_paged_kv_backend as _uses_paged_kv_backend_impl,
    validate_paged_kv_request as _validate_paged_kv_request_impl,
)
from R3.utils.pose_utils import (
    build_online_pose_from_memory,
    compose_relative_pose,
    reconstruct_camera_sequence_from_rel_pose,
    refine_camera_sequence_from_rel_pose,
)


class R3OnlineInferenceMixin:
    def clear_online_state(self):
        """Reset online state and fallback components."""
        self.online_state = None
        self._fallback_manager._detector.reset()
        self._fallback_manager._recent_frame_ids.clear()
        self._pose_edge_log._edges.clear()
        self._historical_bridge_edges.clear()
        self._image_buffer._buffer.clear()
        self._image_buffer._order.clear()
        self._depth_buffer._buffer.clear()
        self._depth_buffer._order.clear()
        self._depth_conf_buffer._buffer.clear()
        self._depth_conf_buffer._order.clear()
        self._keyframe_registry.clear()
        self._previous_bridge_frame_ids.clear()
        self._rejected_ref_ids.clear()
        self._evicted_output_frame_ids.clear()
        self._persistent_post_scores.clear()
        self._metric_bootstrap_images.clear()
        self._metric_bootstrap_depths.clear()
        self._metric_bootstrap_confs.clear()
        self._metric_bootstrap_done = False

    def _create_online_state(self, batch_size: int):
        # Build an empty online state container for sequential inference.
        return OnlineState(batch_size=batch_size)

    def _clone_online_state(self, state):
        # Clone online state so replay passes do not mutate the accepted state.
        if state is None:
            return None
        return copy.deepcopy(state)

    def _clone_online_state_without_kv(self, state):
        # Clone only fallback metadata so bridge replay does not duplicate the live KV cache on GPU.
        if state is None:
            return None

        cloned_state = OnlineState(batch_size=state.batch_size)
        cloned_state.frame_count = int(state.frame_count)
        cloned_state.last_fallback_frame_id = int(state.last_fallback_frame_id)
        cloned_state.tokens_per_frame = state.tokens_per_frame
        cloned_state.cache_frame_ids = list(state.cache_frame_ids)
        cloned_state.frame_order = list(state.frame_order)
        cloned_state.memory_bank_ids = list(state.memory_bank_ids)
        cloned_state.segment_anchor_frame_ids = list(state.segment_anchor_frame_ids)
        cloned_state.frame_feats = dict(state.frame_feats)
        cloned_state.frame_rel_pose_feats = dict(state.frame_rel_pose_feats)
        cloned_state.frame_select_feats = dict(state.frame_select_feats)
        cloned_state.frame_pose_enc = dict(state.frame_pose_enc)
        cloned_state.frame_scores = dict(state.frame_scores)
        cloned_state.frame_post_scores = dict(state.frame_post_scores)
        cloned_state.frame_score_history = {
            frame_id: list(history)
            for frame_id, history in state.frame_score_history.items()
        }
        cloned_state.similarity_cache = dict(state.similarity_cache)
        cloned_state.scale_factor = float(state.scale_factor)
        cloned_state.kv_cache_list = None
        cloned_state.paged_kv_store = None
        return cloned_state

    def _uses_dynamic_kv_cache(self):
        """Return whether the active online cache policy uses the dynamic memory bank."""
        return self.online_kv_cache_mode == "dynamic"

    def _uses_paged_kv_backend(self):
        """Return whether online inference should route global attention through FlashInfer pages."""
        return _uses_paged_kv_backend_impl(self)

    def _validate_paged_kv_request(self, mode: str):
        """Reject paged KV combinations that are not wired in the current rollout phase."""
        return _validate_paged_kv_request_impl(self, mode)

    def _ensure_paged_kv_store(self, state, frame_images):
        """Create the per-state FlashInfer paged KV store on first use."""
        return _ensure_paged_kv_store_impl(self, state, frame_images)

    def _prepare_paged_kv_da3_kwargs(self, state, frame_images, current_frame_id: int):
        """Build DA3 kwargs that expose the current paged KV cache to the ViT."""
        return _prepare_paged_kv_da3_kwargs_impl(
            self, state, frame_images, current_frame_id
        )

    def _populate_paged_kv_from_replay(
        self, state, frame_ids, frame_images_for_shape, kv_cache_list
    ):
        """Copy dense replay K/V into a fresh paged store on ``state``."""
        return _populate_paged_kv_from_replay_impl(
            self, state, frame_ids, frame_images_for_shape, kv_cache_list
        )

    def _make_empty_keyframe_registry(self):
        """Create a fresh keyframe registry with the wrapper's configured selection policy."""
        from R3.models.online.fallback import KeyframeRegistry

        def score_provider(frame_id: int) -> float:
            state = self.online_state
            if state is None:
                return 0.0
            return float(state.frame_scores.get(frame_id, 0.0))

        return KeyframeRegistry(
            interval=self.keyframe_interval,
            mode=self.keyframe_mode,
            novelty_threshold=self.keyframe_novelty_threshold,
            max_interval=self.keyframe_max_interval,
            max_keyframes=self.keyframe_max_keyframes,
            pose_confidence_ratio=self.keyframe_pose_confidence_ratio,
            score_provider=score_provider,
        )

    def _get_dynamic_segment_anchor_budget(self):
        """Return how many frame IDs the dynamic segment should track locally."""
        return _get_dynamic_segment_anchor_budget_impl(
            self.online_kv_cache_mode, self.bank_initial_frames
        )

    def _get_dynamic_segment_frame_order(self, state):
        """Return the frame-order slice that belongs to the active dynamic segment."""
        return _get_dynamic_segment_frame_order_impl(state)

    def _set_dynamic_segment_anchor_frame_ids(
        self, state, frame_ids, preferred_anchor_id=None
    ):
        """Mark a fresh dynamic segment using replay-local frame IDs only."""
        return _set_dynamic_segment_anchor_frame_ids_impl(
            state,
            frame_ids,
            self._get_dynamic_segment_anchor_budget(),
            preferred_anchor_id=preferred_anchor_id,
        )

    def _sync_dynamic_memory_bank_ids(self, state, keyframe_registry=None):
        """Synchronize resident dynamic memory-bank IDs from the registry and initial anchors."""
        registry = (
            self._keyframe_registry if keyframe_registry is None else keyframe_registry
        )
        return _sync_dynamic_memory_bank_ids_impl(
            state,
            self.online_kv_cache_mode,
            self.bank_initial_frames,
            registry,
        )

    def _infer_kv_tokens_per_frame(self, kv_cache_list, num_frames: int):
        """Infer per-frame KV token count from a cache produced for a known number of frames."""
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        for kv_cache in kv_cache_list or []:
            if kv_cache is None or kv_cache[0] is None:
                continue
            keep_len = kv_cache[2] if len(kv_cache) > 2 else kv_cache[0].shape[2]
            if keep_len % num_frames != 0:
                raise ValueError(
                    f"KV cache length {keep_len} is not divisible by frame count {num_frames}"
                )
            return keep_len // num_frames
        return None

    def _apply_online_cache_policy(
        self, state, kv_cache_list, cache_frame_ids, keyframe_registry=None
    ):
        """Apply the configured bounded-cache policy to a state object with updated KV tensors."""
        if self.online_kv_cache_mode == "all":
            state.memory_bank_ids = []
            state.segment_anchor_frame_ids = []
            return kv_cache_list, cache_frame_ids

        self._sync_dynamic_memory_bank_ids(state, keyframe_registry=keyframe_registry)

        keep_frame_ids = self._get_online_cache_keep_frame_ids(state, cache_frame_ids)
        dropped_frame_ids = set(cache_frame_ids) - set(keep_frame_ids)
        kv_cache_list = self._prune_kv_cache_list(
            kv_cache_list,
            cache_frame_ids,
            keep_frame_ids,
            state.tokens_per_frame,
        )
        if state.paged_kv_store is not None:
            for frame_id in dropped_frame_ids:
                state.paged_kv_store.evict_frame(frame_id)
        self._prune_online_state(state, keep_frame_ids)
        cache_frame_ids = keep_frame_ids
        state.memory_bank_ids = [
            frame_id
            for frame_id in state.memory_bank_ids
            if frame_id in cache_frame_ids
        ]
        state.segment_anchor_frame_ids = [
            frame_id
            for frame_id in state.segment_anchor_frame_ids
            if frame_id in cache_frame_ids
        ]
        return kv_cache_list, cache_frame_ids

    def _clear_online_runtime_state(self, state):
        """Drop the live online runtime tensors and frame metadata before a full replay."""
        state.kv_cache_list = None
        state.paged_kv_store = None
        state.tokens_per_frame = None
        state.cache_frame_ids = []
        state.frame_order = []
        state.memory_bank_ids = []
        state.segment_anchor_frame_ids = []
        state.frame_feats = {}
        state.frame_rel_pose_feats = {}
        state.frame_select_feats = {}
        state.frame_pose_enc = {}
        state.frame_scores = {}
        state.frame_post_scores = {}
        state.frame_score_history = {}
        state.similarity_cache = {}
        state.scale_factor = 1.0

    def _should_use_online_mode(self, mode: str):
        return (
            self.online_mode
            and not self.training
            and self._uses_causal_attention()
            and mode in {"causal", "window", "window_wo_sink"}
        )

    def _ensure_online_state(self, batch_size: int):
        if self.online_state is None or self.online_state.batch_size != batch_size:
            self.online_state = self._create_online_state(batch_size)
        return self.online_state

    @staticmethod
    def _get_online_memory_feats(state):
        """Return memory features from all cached frames for the next forward pass."""
        memory_frame_ids = [
            fid for fid in state.cache_frame_ids if fid in state.frame_feats
        ]
        if not memory_frame_ids:
            return None, [], False
        if all(frame_id in state.frame_rel_pose_feats for frame_id in memory_frame_ids):
            memory_feats = torch.cat(
                [state.frame_rel_pose_feats[frame_id] for frame_id in memory_frame_ids],
                dim=1,
            )
            return memory_feats, memory_frame_ids, True
        memory_feats = torch.cat(
            [state.frame_feats[frame_id] for frame_id in memory_frame_ids], dim=1
        )
        return memory_feats, memory_frame_ids, False

    def _update_online_scores(
        self, state, memory_frame_ids, rel_pose_conf, memory_size
    ):
        """Update per-frame confidence scores from relative pose predictions."""
        if memory_size == 0 or rel_pose_conf is None:
            return
        conf_scores = F.softplus(rel_pose_conf[:, :memory_size, memory_size:])
        conf_scores = conf_scores.mean(dim=(0, 2)).detach().cpu()
        for frame_id, score in zip(memory_frame_ids, conf_scores.tolist()):
            prev_score = float(state.frame_scores.get(frame_id, 0.0))
            state.frame_scores[frame_id] = max(prev_score, float(score))

    def _compute_online_frame_post_score(
        self, rel_pose_conf, rel_pose_mask, current_idx: int
    ):
        # Aggregate the current frame's incoming relative-pose confidence into a scalar score.
        if current_idx <= 0:
            return 100.0  # Reference frame: no incoming edges, assume high quality
        if not isinstance(rel_pose_conf, torch.Tensor):
            return 0.0

        candidate_conf = F.softplus(
            rel_pose_conf[:, :current_idx, current_idx].detach().float()
        )
        if isinstance(rel_pose_mask, torch.Tensor):
            candidate_mask = rel_pose_mask[:, :current_idx, current_idx].bool()
            candidate_conf = candidate_conf.masked_select(candidate_mask)
        else:
            candidate_conf = candidate_conf.reshape(-1)

        if candidate_conf.numel() == 0:
            return 0.0
        return float(candidate_conf.mean().detach().cpu().item())

    def _update_online_post_score(self, state, current_frame_id, output):
        # Persist the per-frame post score used for revisit candidate selection.
        rel_pose_conf = self._get_output_value(output, "rel_pose_conf")
        rel_pose_mask = self._get_output_value(output, "rel_pose_mask")
        current_idx = (
            rel_pose_conf.shape[1] - 1 if isinstance(rel_pose_conf, torch.Tensor) else 0
        )
        frame_score = self._compute_online_frame_post_score(
            rel_pose_conf, rel_pose_mask, current_idx
        )
        state.frame_post_scores[current_frame_id] = frame_score
        state.frame_score_history.setdefault(current_frame_id, []).append(frame_score)
        return frame_score

    def _get_online_cache_keep_frame_ids(self, state, cache_frame_ids):
        # Select the bounded set of frame IDs that should remain resident in the KV cache.
        return _get_online_cache_keep_frame_ids_impl(
            state,
            cache_frame_ids,
            self.online_kv_cache_mode,
            self.online_recent_frames,
        )

    def _upgrade_kv_cache_to_buffers(self, kv_cache_list, tokens_per_frame):
        # Convert KV cache tensors to pre-allocated buffers so pruning stays stable and cheap.
        return _upgrade_kv_cache_to_buffers_impl(
            kv_cache_list,
            tokens_per_frame,
            self.online_kv_cache_mode,
            self.online_recent_frames,
            bank_initial_frames=self.bank_initial_frames,
            keyframe_max_keyframes=self.keyframe_max_keyframes,
        )

    def _prune_kv_cache_list(
        self, kv_cache_list, cache_frame_ids, keep_frame_ids, tokens_per_frame
    ):
        # Slice each KV cache block down to the token ranges for the kept frames only.
        return _prune_kv_cache_list_impl(
            kv_cache_list, cache_frame_ids, keep_frame_ids, tokens_per_frame
        )

    def _prune_online_similarity_cache(self, state, keep_frame_ids_set):
        # Remove cached similarities for frames that were dropped from the active state.
        _prune_online_similarity_cache_impl(state, keep_frame_ids_set)

    def _prune_online_state(self, state, keep_frame_ids):
        # Drop state entries for frames that are no longer resident in the active cache.
        _prune_online_state_impl(state, keep_frame_ids)

    def _update_online_state(self, state, output, current_frame_id):
        """Append current frame to online state and apply the configured cache retention policy."""
        state.frame_order.append(current_frame_id)
        state.frame_feats[current_frame_id] = output.cam_feat.detach()
        rel_pose_projected_feat = self._get_output_value(
            output, "rel_pose_projected_feat"
        )
        if isinstance(rel_pose_projected_feat, torch.Tensor):
            state.frame_rel_pose_feats[current_frame_id] = (
                rel_pose_projected_feat.detach()
            )
        memory_select_feat = self._get_output_value(output, "memory_select_feat")
        if not isinstance(memory_select_feat, torch.Tensor):
            memory_select_feat = output.cam_feat
        state.frame_select_feats[current_frame_id] = memory_select_feat.detach()

        if state.paged_kv_store is not None:
            cache_frame_ids = state.cache_frame_ids + [current_frame_id]
            _, cache_frame_ids = self._apply_online_cache_policy(
                state,
                None,
                cache_frame_ids,
            )
            state.cache_frame_ids = cache_frame_ids
            return

        kv_cache_list = self._get_output_value(output, "kv_cache_list")
        if kv_cache_list is not None:
            if state.tokens_per_frame is None:
                state.tokens_per_frame = self._infer_kv_tokens_per_frame(
                    kv_cache_list, num_frames=1
                )
            kv_cache_list = self._upgrade_kv_cache_to_buffers(
                kv_cache_list, state.tokens_per_frame
            )
            cache_frame_ids = state.cache_frame_ids + [current_frame_id]
            kv_cache_list, cache_frame_ids = self._apply_online_cache_policy(
                state, kv_cache_list, cache_frame_ids
            )
            state.kv_cache_list = kv_cache_list
            state.cache_frame_ids = cache_frame_ids

    def _extract_online_step_options(self, kwargs):
        # Extract online wrapper options from kwargs, passing the rest to DA3.
        da3_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ONLINE_WRAPPER_OPTION_KEYS
        }
        rel_pose_reconstruction_kwargs = dict(
            kwargs.get(
                "rel_pose_reconstruction_kwargs",
                self.rel_pose_reconstruction_kwargs,
            )
        )
        online_verbose = resolve_online_verbose(
            online_verbose=kwargs.get("online_verbose"),
            online_memory_verbose=kwargs.get("online_memory_verbose"),
            online_revisit_verbose=None,
        )
        options = {
            "export_feat_layers": kwargs.get(
                "export_feat_layers", self.export_feat_layers
            ),
            "rel_pose_reconstruction_method": kwargs.get(
                "rel_pose_reconstruction_method",
                self.rel_pose_reconstruction_method,
            ),
            "rel_pose_reconstruction_kwargs": rel_pose_reconstruction_kwargs,
            "online_finalize_pose_reconstruction": kwargs.get(
                "online_finalize_pose_reconstruction",
                self.online_finalize_pose_reconstruction,
            ),
            "online_verbose": online_verbose,
            "runtime_stats_every": int(kwargs.get("runtime_stats_every", 0) or 0),
            "runtime_stats_path": kwargs.get("runtime_stats_path", ""),
        }
        return options, da3_kwargs

    def _evict_frame_from_kv(self, state, frame_id):
        """Remove a single frame's KV tokens from the cache."""
        self._evicted_output_frame_ids.add(int(frame_id))
        self._persistent_post_scores.pop(int(frame_id), None)
        state.frame_post_scores.pop(frame_id, None)
        state.frame_score_history.pop(frame_id, None)
        self._pose_edge_log.drop_frames({int(frame_id)})
        self._historical_bridge_edges = [
            edge
            for edge in self._historical_bridge_edges
            if edge.frame_i != int(frame_id) and edge.frame_j != int(frame_id)
        ]
        if state.paged_kv_store is not None:
            state.paged_kv_store.evict_frame(frame_id)
            if frame_id in state.cache_frame_ids:
                state.cache_frame_ids.remove(frame_id)
            state.frame_order = [f for f in state.frame_order if f != frame_id]
            state.frame_feats.pop(frame_id, None)
            state.frame_rel_pose_feats.pop(frame_id, None)
            state.frame_select_feats.pop(frame_id, None)
            state.frame_pose_enc.pop(frame_id, None)
            state.frame_scores.pop(frame_id, None)
            if state.memory_bank_ids:
                state.memory_bank_ids = [
                    cached_id
                    for cached_id in state.memory_bank_ids
                    if cached_id != frame_id
                ]
            state.segment_anchor_frame_ids = [
                anchor_id
                for anchor_id in state.segment_anchor_frame_ids
                if anchor_id != frame_id
            ]
            self._prune_online_similarity_cache(state, set(state.frame_order))
            return
        _evict_frame_from_kv_impl(state, frame_id)

    def _resolve_low_conf_evict_threshold(self, state, current_frame_id, confidence):
        """Resolve absolute/relative KV-eviction threshold for the current frame.

        Relative eviction uses the first warmup frames in the current segment as
        the baseline, but skips the segment anchor because frame 0/reference-only
        scores can be synthetic rather than measured incoming-edge confidence.
        """
        absolute_threshold = max(float(self.evict_low_conf_threshold), 0.0)
        pct = max(float(getattr(self, "evict_low_conf_threshold_pct", 0.0)), 0.0)
        warmup_frames = max(
            int(getattr(self, "evict_low_conf_warmup_frames", 0)), 0
        )
        if pct <= 0.0 or warmup_frames <= 0:
            return absolute_threshold

        segment_start = int(getattr(state, "last_fallback_frame_id", -1)) + 1
        warmup_end = segment_start + warmup_frames
        if int(current_frame_id) < warmup_end:
            return absolute_threshold

        frame_order = [int(fid) for fid in getattr(state, "frame_order", [])]
        warmup_ids = [
            fid for fid in frame_order if segment_start <= fid < warmup_end
        ]
        if warmup_ids and warmup_ids[0] == segment_start:
            warmup_ids = warmup_ids[1:]

        frame_post_scores = getattr(state, "frame_post_scores", {})
        warmup_scores = []
        for fid in warmup_ids:
            score = frame_post_scores.get(fid, None)
            if score is None:
                continue
            score = float(score)
            if math.isfinite(score):
                warmup_scores.append(score)
        if not warmup_scores:
            return absolute_threshold

        baseline = sum(warmup_scores) / len(warmup_scores)
        if not math.isfinite(baseline) or baseline <= 0.0:
            return absolute_threshold
        return max(absolute_threshold, baseline * (pct / 100.0))

    def _can_attempt_fallback(self, state, current_frame_id):
        """Enforce a minimum number of frames between fallback attempts."""
        if self.min_segment_frames <= 0:
            return True
        if state.last_fallback_frame_id >= 0:
            return (
                current_frame_id - state.last_fallback_frame_id
            ) >= self.min_segment_frames
        return (current_frame_id + 1) >= self.min_segment_frames

    def _current_segment_length(self, state, current_frame_id):
        """Return how many frames have elapsed in the current segment."""
        if state.last_fallback_frame_id >= 0:
            return current_frame_id - state.last_fallback_frame_id
        return current_frame_id + 1

    def _build_forced_fallback_action(self, state, current_frame_id):
        """Build a FallbackAction triggered by max_segment_frames (no drought required)."""
        from R3.models.online.fallback import FallbackAction

        n_bridge = self._fallback_manager.num_bridge_frames
        all_frames = list(state.frame_order)
        if len(all_frames) < 2:
            return None
        fallback_frame_id = all_frames[-1]
        # Prefer parallax-aware bridge selection so a stationary tail does not produce a
        # near-duplicate bridge. Forced fallback's job is to always cut the cache, so fall back
        # to consecutive selection if parallax gating returns a degenerate or too-short bridge.
        bridge_frame_ids = self._fallback_manager._select_bridge_frames(state, all_frames)
        if bridge_frame_ids is None or len(bridge_frame_ids) < 2:
            bridge_start = max(0, len(all_frames) - n_bridge)
            bridge_frame_ids = list(all_frames[bridge_start:])
        if len(bridge_frame_ids) < 2:
            return None

        # Initial ref = bridge[0]; _resolve_fallback_ref_id (called below) re-picks honoring
        # fallback_ref_mode + rejected refs, matching TTT's pre-resolve in build_forced_fallback_action.
        action_kwargs = dict(
            fallback_frame_id=fallback_frame_id,
            new_ref_id=int(bridge_frame_ids[0]),
            bridge_frame_ids=bridge_frame_ids,
            bad_frame_ids=[],
            forced=True,
        )
        tentative_action = FallbackAction(**action_kwargs)
        new_ref_id = self._resolve_fallback_ref_id(
            tentative_action,
            self._keyframe_registry,
            banked_frame_ids=state.memory_bank_ids if state.memory_bank_ids else None,
        )

        print(
            f"  [MAX_SEGMENT] forcing fallback at frame {current_frame_id} (segment={self._current_segment_length(state, current_frame_id)})"
        )
        return FallbackAction(
            fallback_frame_id=fallback_frame_id,
            new_ref_id=new_ref_id,
            bridge_frame_ids=bridge_frame_ids,
            bad_frame_ids=[],
            forced=True,
        )

    def _resolve_fallback_ref_id(
        self, action, keyframe_registry, banked_frame_ids=None
    ):
        """Choose the fallback reference frame, honoring bridge mode + rejected-ref suppression.

        `banked_frame_ids` (when provided) constrains the picked ref to currently banked frames.
        `self._rejected_ref_ids` carries refs that already produced a bad replay; we suppress
        them so the next attempt does not re-pick the same anchor. Bridge frames stay eligible
        because the bridge window shifts with each new drought.
        """
        bridge_ref_id = int(action.bridge_frame_ids[0])
        bridge_set = {int(fid) for fid in action.bridge_frame_ids}
        rejected = {int(fid) for fid in self._rejected_ref_ids} - bridge_set
        allowed_ids = None
        if banked_frame_ids is not None:
            allowed_ids = {int(frame_id) for frame_id in banked_frame_ids}

        # Bridge mode is an explicit user override: always anchor on bridge[0], even for forced
        # fallbacks. Otherwise REF_MODE=bridge would silently pick a far keyframe whenever the
        # structural trigger fires.
        if self.fallback_ref_mode == "bridge":
            return bridge_ref_id

        excluded = bridge_set | rejected
        if getattr(keyframe_registry, "mode", "interval") == "novelty" and hasattr(
            keyframe_registry, "select_fallback_ref"
        ):
            candidates = keyframe_registry.select_fallback_ref(
                bridge_ref_id,
                top_k=1,
                allowed_ids=allowed_ids,
                excluded=excluded,
            )
            return int(candidates[0]) if candidates else bridge_ref_id

        keyframe_ref_id = keyframe_registry.get_latest_before(
            bridge_ref_id, allowed_ids=allowed_ids, excluded=excluded
        )
        return int(keyframe_ref_id) if keyframe_ref_id is not None else bridge_ref_id

    def _record_rejected_ref(self, action):
        """Remember a ref that just failed so resolve_fallback_ref_id can skip it next time.

        Bridge frames are kept eligible because the bridge window shifts with each drought; we
        only suppress *non-bridge* (i.e. registry-picked) refs that produced a bad replay.
        """
        ref_id = int(action.new_ref_id)
        if ref_id in {int(fid) for fid in action.bridge_frame_ids}:
            return
        self._rejected_ref_ids.add(ref_id)

    def _cut_dynamic_segment_after_rejected_fallback(self, state, action):
        """Bound the dynamic KV cache after a rejected fallback by reapplying the cache policy.

        Keeps the existing poses/depths (the replay was not good enough), but trims the live
        KV cache so repeated low-confidence rejections cannot let the segment grow past
        max_segment_frames. The bridge head becomes the fresh dynamic-segment anchor.
        """
        if not self._uses_dynamic_kv_cache():
            return
        if not action.bridge_frame_ids:
            return
        no_cache = state.kv_cache_list is None and state.paged_kv_store is None
        bad_dense_cache = (
            state.kv_cache_list is not None and state.tokens_per_frame is None
        )
        if no_cache or bad_dense_cache:
            return
        bridge_ids = [int(fid) for fid in action.bridge_frame_ids]
        self._set_dynamic_segment_anchor_frame_ids(
            state,
            bridge_ids,
            preferred_anchor_id=bridge_ids[0],
        )
        state.kv_cache_list, state.cache_frame_ids = self._apply_online_cache_policy(
            state, state.kv_cache_list, state.cache_frame_ids
        )

    def _cut_live_cache_after_failed_forced_fallback(
        self, state, action, attempt_frame_id: int
    ) -> bool:
        if not action.forced:
            return False
        if state.kv_cache_list is None and state.paged_kv_store is None:
            state.last_fallback_frame_id = attempt_frame_id
            return True
        if state.kv_cache_list is not None and state.tokens_per_frame is None:
            return False

        keep_candidates = list(
            dict.fromkeys(
                [int(action.new_ref_id), *[int(fid) for fid in action.bridge_frame_ids]]
            )
        )
        keep_frame_ids = [
            frame_id for frame_id in keep_candidates if frame_id in state.cache_frame_ids
        ]
        if not keep_frame_ids:
            return False

        dropped_frame_ids = set(state.cache_frame_ids) - set(keep_frame_ids)
        if state.kv_cache_list is not None:
            state.kv_cache_list = self._prune_kv_cache_list(
                state.kv_cache_list,
                state.cache_frame_ids,
                keep_frame_ids,
                state.tokens_per_frame,
            )
        if state.paged_kv_store is not None:
            for frame_id in dropped_frame_ids:
                state.paged_kv_store.evict_frame(frame_id)

        state.cache_frame_ids = keep_frame_ids
        self._prune_online_state(state, keep_frame_ids)
        state.memory_bank_ids = [
            frame_id for frame_id in state.memory_bank_ids if frame_id in keep_frame_ids
        ]
        if self._uses_dynamic_kv_cache():
            self._set_dynamic_segment_anchor_frame_ids(
                state,
                keep_frame_ids,
                preferred_anchor_id=keep_frame_ids[0],
            )
        else:
            state.segment_anchor_frame_ids = []

        state.last_fallback_frame_id = attempt_frame_id
        return True

    def _finish_failed_fallback(self, state, action, attempt_frame_id: int):
        if action.forced:
            if not self._cut_live_cache_after_failed_forced_fallback(
                state, action, attempt_frame_id
            ):
                return
        else:
            state.last_fallback_frame_id = attempt_frame_id
        self._fallback_manager.on_fallback_complete()

    def _fallback_improves_bridge_scores(self, old_scores, new_scores):
        # Require the fallback rerun to improve bridge confidence before accepting it.
        return _fallback_improves_bridge_scores_impl(old_scores, new_scores)

    def _resolve_fallback_scale(self, pose_scale, depth_scale):
        # Prefer the depth-derived scale unless pose agrees closely enough to sharpen it.
        return _resolve_fallback_scale_impl(pose_scale, depth_scale)

    def _compute_metric_scale_factor(self, image, pred_depth, pred_conf):
        # Run DA3-metric and return median(metric / pred), pooling supplied frames.
        return _compute_metric_scale_factor_impl(
            self.da3_metric,
            image,
            pred_depth,
            pred_conf,
            self.metric_min_conf,
        )

    def _apply_metric_scale_correction(
        self, state, correction: float, predictions=None
    ):
        """Rescale accumulated online pose state after multi-frame metric bootstrap."""
        if (
            correction is None
            or not math.isfinite(float(correction))
            or float(correction) <= 0.0
        ):
            return
        factor = float(correction)
        if abs(factor - 1.0) < 1e-9:
            return

        for frame_id, pose in list(state.frame_pose_enc.items()):
            if isinstance(pose, torch.Tensor):
                pose = pose.clone()
                pose[..., :3] *= factor
                state.frame_pose_enc[frame_id] = pose

        for edge in self._pose_edge_log._edges:
            if isinstance(edge.rel_pose_enc, torch.Tensor):
                edge.rel_pose_enc = edge.rel_pose_enc.clone()
                edge.rel_pose_enc[..., :3] *= factor

        if predictions is None:
            return
        if isinstance(predictions.get("pose_enc"), torch.Tensor):
            pose_enc = predictions["pose_enc"].clone()
            pose_enc[..., :3] *= factor
            predictions["pose_enc"] = pose_enc
        if isinstance(predictions.get("pose_enc_list"), list):
            rescaled = []
            for pose_enc in predictions["pose_enc_list"]:
                if isinstance(pose_enc, torch.Tensor):
                    pose_enc = pose_enc.clone()
                    pose_enc[..., :3] *= factor
                rescaled.append(pose_enc)
            predictions["pose_enc_list"] = rescaled

    def _copy_online_state(self, state, new_state):
        # Overwrite the live online state object without duplicating accepted GPU tensors.
        state.batch_size = int(new_state.batch_size)
        state.frame_count = int(new_state.frame_count)
        state.last_fallback_frame_id = int(new_state.last_fallback_frame_id)
        state.kv_cache_list = new_state.kv_cache_list
        state.paged_kv_store = new_state.paged_kv_store
        state.tokens_per_frame = new_state.tokens_per_frame
        state.cache_frame_ids = list(new_state.cache_frame_ids)
        state.frame_order = list(new_state.frame_order)
        state.memory_bank_ids = list(new_state.memory_bank_ids)
        state.segment_anchor_frame_ids = list(new_state.segment_anchor_frame_ids)
        state.frame_feats = dict(new_state.frame_feats)
        state.frame_rel_pose_feats = dict(new_state.frame_rel_pose_feats)
        state.frame_select_feats = dict(new_state.frame_select_feats)
        state.frame_pose_enc = dict(new_state.frame_pose_enc)
        state.frame_scores = dict(new_state.frame_scores)
        state.frame_post_scores = dict(new_state.frame_post_scores)
        state.frame_score_history = {
            frame_id: list(history)
            for frame_id, history in new_state.frame_score_history.items()
        }
        state.similarity_cache = dict(new_state.similarity_cache)
        state.scale_factor = float(new_state.scale_factor)

    def _run_online_step(
        self,
        frame_images,
        state,
        online_options,
        da3_kwargs,
        current_frame_id=None,
        force_reference_token=False,
        pose_max_recent=0,
    ):
        if frame_images.dim() == 4:
            frame_images = frame_images.unsqueeze(1)
        if frame_images.dim() != 5 or frame_images.shape[1] != 1:
            raise ValueError(
                "online step expects frame_images with shape [B, 1, 3, H, W]"
            )

        _, _, _, H, W = frame_images.shape
        if current_frame_id is None:
            current_frame_id = state.frame_count
        memory_feats, memory_frame_ids, memory_feats_projected = (
            self._get_online_memory_feats(state)
        )
        use_paged_kv = self._uses_paged_kv_backend()
        paged_da3_kwargs = (
            self._prepare_paged_kv_da3_kwargs(
                state, frame_images, int(current_frame_id)
            )
            if use_paged_kv
            else {}
        )
        output = self.da3(
            frame_images,
            export_feat_layers=online_options["export_feat_layers"],
            kv_cache_list=None if use_paged_kv else state.kv_cache_list,
            return_kv_cache=not use_paged_kv,
            return_memory_select_feat=True,
            rel_pose_memory_feats=memory_feats,
            rel_pose_memory_feats_projected=memory_feats_projected,
            camera_token_is_reference=(current_frame_id == 0) or force_reference_token,
            frame_ids=torch.tensor(
                [current_frame_id], device=frame_images.device, dtype=torch.long
            ),
            **paged_da3_kwargs,
            **dict(da3_kwargs),
        )

        frame_post_score = self._update_online_post_score(
            state, current_frame_id, output
        )
        # Mirror the freshly computed score into the wrapper-level persistent dict.
        # state.frame_post_scores is wiped by _clear_online_runtime_state on every fallback,
        # so the per-frame visualization/export needs an independent record populated during
        # normal inference. Replayed scores (self._in_fallback) are skipped so the record
        # reflects the original incoming-edge confidence used for drought detection.
        if not self._in_fallback:
            self._persistent_post_scores[int(current_frame_id)] = float(
                frame_post_score
            )

        rel_pose_memory_size = int(
            self._get_output_value(output, "rel_pose_memory_size", 0)
        )
        rel_pose_frame_ids = None
        if rel_pose_memory_size > 0:
            rel_pose_frame_ids = memory_frame_ids + [current_frame_id]
            self._update_online_scores(
                state,
                memory_frame_ids,
                self._get_output_value(output, "rel_pose_conf"),
                rel_pose_memory_size,
            )

        anchored_pose_enc = None
        if rel_pose_frame_ids is not None:
            rel_pose_enc = self._get_output_value(output, "rel_pose_enc")
            rel_pose_conf = self._get_output_value(output, "rel_pose_conf")
            rel_pose_conf_t = self._get_output_value(output, "rel_pose_conf_t")
            rel_pose_conf_r = self._get_output_value(output, "rel_pose_conf_r")
            rel_pose_mask = self._get_output_value(output, "rel_pose_mask")

            if self.online_fallback_enabled:
                log_online_pose_edges(
                    pose_edge_log=self._pose_edge_log,
                    state=state,
                    current_frame_id=current_frame_id,
                    rel_pose_frame_ids=rel_pose_frame_ids,
                    rel_pose_enc=rel_pose_enc,
                    rel_pose_conf=rel_pose_conf,
                    rel_pose_conf_t=rel_pose_conf_t,
                    rel_pose_conf_r=rel_pose_conf_r,
                    rel_pose_mask=rel_pose_mask,
                    in_fallback=self._in_fallback,
                )

            # Scale relative pose translation for coordinate continuity after fallback
            if state.scale_factor != 1.0:
                rel_pose_enc = rel_pose_enc.clone()
                rel_pose_enc[..., :3] *= state.scale_factor

            anchored_pose_enc = build_online_pose_from_memory(
                state,
                rel_pose_enc,
                rel_pose_conf,
                rel_pose_mask,
                rel_pose_frame_ids,
                rel_pose_conf_t=rel_pose_conf_t,
                rel_pose_conf_r=rel_pose_conf_r,
                max_recent=pose_max_recent,
                topn_conf=online_options["rel_pose_reconstruction_kwargs"].get(
                    "topn_conf", 10
                ),
            )

        predictions = self._format_predictions(
            output,
            frame_images,
            H,
            W,
            online_options["rel_pose_reconstruction_method"],
            online_options["rel_pose_reconstruction_kwargs"],
            online_mode=True,
            rel_pose_frame_ids=rel_pose_frame_ids,
            anchored_pose_enc=anchored_pose_enc,
        )

        if "pose_enc" in predictions and predictions["pose_enc"] is not None:
            state.frame_pose_enc[current_frame_id] = predictions["pose_enc"][
                :, -1
            ].detach()

        self._update_online_state(state, output, current_frame_id)
        state.frame_count = max(int(state.frame_count), int(current_frame_id) + 1)

        # Metric-scale anchor at sequence start: run the metric model once on frame 0 and
        # set state.scale_factor so every subsequent depth/translation output is in metric
        # units (rather than the model's internal relative scale).
        if (
            self.metric_scale_enabled
            and not self._in_fallback
            and current_frame_id == 0
            and state.scale_factor == 1.0
        ):
            metric_scale = self._compute_metric_scale_factor(
                image=frame_images,
                pred_depth=predictions.get("depth"),
                pred_conf=predictions.get("depth_conf"),
            )
            if metric_scale is not None and metric_scale > 0:
                state.scale_factor = metric_scale
            else:
                logging.getLogger(__name__).warning(
                    "Metric anchor at frame 0 returned no scale; leaving scale_factor=1.0"
                )

        if (
            self.metric_scale_enabled
            and not self._in_fallback
            and self.metric_bootstrap_frames > 1
            and not self._metric_bootstrap_done
            and current_frame_id < self.metric_bootstrap_frames
        ):
            raw_depth = predictions.get("depth")
            if raw_depth is not None:
                self._metric_bootstrap_images[current_frame_id] = frame_images.detach()
                self._metric_bootstrap_depths[current_frame_id] = raw_depth.detach()
                raw_conf = predictions.get("depth_conf")
                if raw_conf is not None:
                    self._metric_bootstrap_confs[current_frame_id] = raw_conf.detach()

            if current_frame_id == self.metric_bootstrap_frames - 1:
                sorted_frame_ids = sorted(self._metric_bootstrap_images.keys())
                if len(sorted_frame_ids) >= 2:
                    images = [
                        self._metric_bootstrap_images[frame_id]
                        for frame_id in sorted_frame_ids
                    ]
                    depths = [
                        self._metric_bootstrap_depths[frame_id]
                        for frame_id in sorted_frame_ids
                    ]
                    confs = (
                        [
                            self._metric_bootstrap_confs.get(frame_id)
                            for frame_id in sorted_frame_ids
                        ]
                        if self._metric_bootstrap_confs
                        else None
                    )
                    pooled_scale = self._compute_metric_scale_factor(
                        image=images,
                        pred_depth=depths,
                        pred_conf=confs,
                    )
                    if pooled_scale is not None and pooled_scale > 0:
                        old_scale = float(state.scale_factor)
                        if old_scale > 0:
                            correction = float(pooled_scale) / old_scale
                            if abs(correction - 1.0) > 1e-4:
                                state.scale_factor = float(pooled_scale)
                                self._apply_metric_scale_correction(
                                    state, correction, predictions
                                )
                                predictions["metric_scale_correction"] = float(
                                    correction
                                )
                    else:
                        logging.getLogger(__name__).warning(
                            "Metric bootstrap refinement at frame %d returned no pooled scale; keeping initial anchor",
                            current_frame_id,
                        )

                self._metric_bootstrap_done = True
                self._metric_bootstrap_images.clear()
                self._metric_bootstrap_depths.clear()
                self._metric_bootstrap_confs.clear()

        should_register_keyframes = (
            self.online_fallback_enabled or self._uses_dynamic_kv_cache()
        )
        if should_register_keyframes:
            feat = state.frame_select_feats.get(current_frame_id)
            if feat is None:
                feat = state.frame_feats.get(current_frame_id)
            if feat is not None:
                pose = state.frame_pose_enc.get(current_frame_id)
                # Only bank measured incoming confidence as reliability; synthetic reference-only
                # scores (no incoming edges) would otherwise inflate the bank and pull the fallback
                # ref toward synthetic-score frames like frame 0.
                reliability = (
                    float(frame_post_score)
                    if rel_pose_frame_ids is not None and len(rel_pose_frame_ids) > 1
                    else 0.0
                )
                # Anchor the keyframe pose-confidence gate to the current segment's warmup
                # mean so the admission bar is fixed by early-segment behavior rather than
                # drifting with the bank's self-selected mean. During fallback replay the trial
                # segment is just being seeded; pass None so the gate bypasses.
                warmup_confidence = None
                if not self._in_fallback:
                    warmup_stats = self._fallback_manager.get_drought_stats(
                        state, current_frame_id, confidence=frame_post_score
                    )
                    warmup_confidence = warmup_stats.warmup_confidence
                self._keyframe_registry.maybe_add(
                    current_frame_id,
                    feat,
                    frame_images,
                    pose,
                    reliability=reliability,
                    warmup_confidence=warmup_confidence,
                )

        # Fallback: store image and raw depth BEFORE scaling, then check drought
        if self.online_fallback_enabled:
            self._image_buffer.push(current_frame_id, frame_images)
            if "depth" in predictions and predictions["depth"] is not None:
                self._depth_buffer.push(current_frame_id, predictions["depth"])
            if "depth_conf" in predictions and predictions["depth_conf"] is not None:
                self._depth_conf_buffer.push(
                    current_frame_id, predictions["depth_conf"]
                )
            if not self._in_fallback:
                post_score = state.frame_post_scores.get(current_frame_id, float("inf"))
                if self._can_attempt_fallback(state, current_frame_id):
                    action = self._fallback_manager.maybe_trigger(
                        state, post_score, current_frame_id
                    )
                    # Force fallback when segment exceeds max length to bound KV cache
                    if action is None and self.max_segment_frames > 0:
                        segment_len = self._current_segment_length(
                            state, current_frame_id
                        )
                        if segment_len >= self.max_segment_frames:
                            action = self._build_forced_fallback_action(
                                state, current_frame_id
                            )
                    if action:
                        self._execute_fallback(
                            action, state, online_options, da3_kwargs
                        )

        # Evict low-confidence frames from KV cache. This is independent of
        # fallback re-anchoring; fallback only decides whether extra bridge
        # replay/reanchoring should run.
        if not self._in_fallback and not self._uses_dynamic_kv_cache():
            post_score = state.frame_post_scores.get(current_frame_id, float("inf"))
            threshold = self._resolve_low_conf_evict_threshold(
                state, current_frame_id, post_score
            )
            if threshold > 0.0 and post_score < threshold and current_frame_id != 0:
                self._evict_frame_from_kv(state, current_frame_id)

        # Scale depth predictions for coordinate continuity after fallback
        if (
            state.scale_factor != 1.0
            and "depth" in predictions
            and predictions["depth"] is not None
        ):
            predictions["depth"] = predictions["depth"] * state.scale_factor

        return predictions

    def _forward_full_attention_replay(
        self, frame_ids, images, state, reference_frame_id, da3_kwargs=None
    ):
        """Run a single full-attention forward pass over all replay frames.

        Args:
            frame_ids: ordered list of frame IDs [ref, keyframes..., bridge...]
            images: list of image tensors [B, 1, 3, H, W] matching frame_ids
            state: OnlineState (not mutated, used only for context)
            reference_frame_id: which frame ID is the reference token
        Returns:
            dict with depth (dict fid->tensor), depth_conf (dict fid->tensor),
            kv_cache_list, per_frame_pose_enc (dict fid->tensor),
            per_frame_post_scores (dict fid->float), per_frame_cam_feat (dict fid->tensor),
            per_frame_rel_pose_feat (dict fid->tensor), per_frame_select_feat (dict fid->tensor),
            output (raw DA3 output)
        """
        batch_images = torch.cat(images, dim=1)  # [B, N, 3, H, W]
        N = len(frame_ids)
        da3_kwargs = dict(da3_kwargs or {})

        # Temporarily override backbone attention mode to "full"
        backbone = self.da3.backbone
        pretrained = getattr(backbone, "pretrained", backbone)
        orig_causal = pretrained.causal_attn
        orig_mode = pretrained.attention_mode
        pretrained.attention_mode = "full"

        ref_idx = frame_ids.index(reference_frame_id)
        camera_token_is_reference = [False] * N
        camera_token_is_reference[ref_idx] = True

        try:
            output = self.da3(
                batch_images,
                export_feat_layers=self.export_feat_layers,
                kv_cache_list=None,
                return_kv_cache=True,
                return_memory_select_feat=True,
                rel_pose_memory_feats=None,
                camera_token_is_reference=camera_token_is_reference,
                frame_ids=torch.tensor(
                    frame_ids, device=batch_images.device, dtype=torch.long
                ),
                **da3_kwargs,
            )
        finally:
            pretrained.causal_attn = orig_causal
            pretrained.attention_mode = orig_mode

        result = {
            "output": output,
            "kv_cache_list": self._get_output_value(output, "kv_cache_list"),
        }

        # Depth: [B, N, H, W] -> per-frame dict
        depth_map = {}
        depth_conf_map = {}
        if "depth" in output and not isinstance(
            getattr(output, "depth", None), addict.Dict
        ):
            depth = output.depth
            for i, fid in enumerate(frame_ids):
                depth_map[fid] = depth[:, i : i + 1].unsqueeze(-1)
        if "depth_conf" in output and not isinstance(
            getattr(output, "depth_conf", None), addict.Dict
        ):
            depth_conf = output.depth_conf
            for i, fid in enumerate(frame_ids):
                depth_conf_map[fid] = depth_conf[:, i : i + 1]
        result["depth"] = depth_map
        result["depth_conf"] = depth_conf_map

        # Per-frame post scores from rel_pose_conf
        rel_pose_conf = self._get_output_value(output, "rel_pose_conf")
        rel_pose_conf_t = self._get_output_value(output, "rel_pose_conf_t")
        rel_pose_conf_r = self._get_output_value(output, "rel_pose_conf_r")
        rel_pose_mask = self._get_output_value(output, "rel_pose_mask")
        post_scores = {}
        for i, fid in enumerate(frame_ids):
            post_scores[fid] = self._compute_online_frame_post_score(
                rel_pose_conf, rel_pose_mask, i
            )
        result["per_frame_post_scores"] = post_scores

        # Cam features
        cam_feat_map = {}
        rel_pose_feat_map = {}
        select_feat_map = {}
        if hasattr(output, "cam_feat") and output.cam_feat is not None:
            for i, fid in enumerate(frame_ids):
                cam_feat_map[fid] = output.cam_feat[:, i : i + 1].detach()
        rel_pose_projected_feat = self._get_output_value(
            output, "rel_pose_projected_feat"
        )
        if isinstance(rel_pose_projected_feat, torch.Tensor):
            for i, fid in enumerate(frame_ids):
                rel_pose_feat_map[fid] = rel_pose_projected_feat[:, i : i + 1].detach()
        memory_select_feat = self._get_output_value(output, "memory_select_feat")
        if isinstance(memory_select_feat, torch.Tensor):
            for i, fid in enumerate(frame_ids):
                select_feat_map[fid] = memory_select_feat[:, i : i + 1].detach()
        result["per_frame_cam_feat"] = cam_feat_map
        result["per_frame_rel_pose_feat"] = rel_pose_feat_map
        result["per_frame_select_feat"] = select_feat_map

        # Pose encoding from relative predictions
        rel_pose_enc = self._get_output_value(output, "rel_pose_enc")
        pose_enc_map = {}
        if isinstance(rel_pose_enc, torch.Tensor):
            pose_enc = reconstruct_camera_sequence_from_rel_pose(
                rel_pose_enc,
                rel_pose_conf,
                rel_pose_mask,
                pred_rel_conf_t=rel_pose_conf_t,
                pred_rel_conf_r=rel_pose_conf_r,
                **self.rel_pose_reconstruction_kwargs,
            )
            for i, fid in enumerate(frame_ids):
                pose_enc_map[fid] = pose_enc[:, i].detach()
        result["per_frame_pose_enc"] = pose_enc_map

        return result

    def _seed_online_state_from_full_attention_replay(
        self, state, frame_ids, images, replay_result
    ):
        """Initialize online runtime state from a full-attention bootstrap/replay."""
        frame_ids = [int(frame_id) for frame_id in frame_ids]

        if self._uses_paged_kv_backend():
            self._populate_paged_kv_from_replay(
                state,
                frame_ids,
                images[0],
                replay_result["kv_cache_list"],
            )
            state.kv_cache_list = None
        else:
            state.kv_cache_list = replay_result["kv_cache_list"]
        state.cache_frame_ids = list(frame_ids)
        state.frame_order = list(frame_ids)
        if state.kv_cache_list is not None:
            state.tokens_per_frame = self._infer_kv_tokens_per_frame(
                state.kv_cache_list,
                num_frames=len(frame_ids),
            )

        for fid in frame_ids:
            if fid in replay_result["per_frame_pose_enc"]:
                state.frame_pose_enc[fid] = replay_result["per_frame_pose_enc"][fid]
            if fid in replay_result["per_frame_post_scores"]:
                score = float(replay_result["per_frame_post_scores"][fid])
                state.frame_post_scores[fid] = score
                state.frame_scores[fid] = score
                state.frame_score_history.setdefault(fid, []).append(score)
                self._persistent_post_scores[fid] = score
            if fid in replay_result.get("per_frame_cam_feat", {}):
                state.frame_feats[fid] = replay_result["per_frame_cam_feat"][fid]
            if fid in replay_result.get("per_frame_rel_pose_feat", {}):
                state.frame_rel_pose_feats[fid] = replay_result[
                    "per_frame_rel_pose_feat"
                ][fid]
            if fid in replay_result.get("per_frame_select_feat", {}):
                state.frame_select_feats[fid] = replay_result["per_frame_select_feat"][
                    fid
                ]

        state.frame_count = max(int(state.frame_count), max(frame_ids) + 1)

    def _apply_full_attention_bootstrap_metric_scale(
        self, state, images, predictions
    ):
        """Anchor metric scale for a full-attention bootstrap chunk."""
        if not self.metric_scale_enabled or state.scale_factor != 1.0:
            return
        depth = predictions.get("depth")
        if not isinstance(depth, torch.Tensor):
            return

        conf = predictions.get("depth_conf")
        num_frames = int(depth.shape[1])
        metric_frames = max(1, min(num_frames, int(self.metric_bootstrap_frames)))
        if metric_frames > 1:
            image_arg = [
                images[:, idx : idx + 1].detach() for idx in range(metric_frames)
            ]
            depth_arg = [
                depth[:, idx : idx + 1].detach() for idx in range(metric_frames)
            ]
            conf_arg = (
                [conf[:, idx : idx + 1].detach() for idx in range(metric_frames)]
                if isinstance(conf, torch.Tensor)
                else None
            )
        else:
            image_arg = images[:, :1]
            depth_arg = depth[:, :1]
            conf_arg = conf[:, :1] if isinstance(conf, torch.Tensor) else None

        metric_scale = self._compute_metric_scale_factor(
            image=image_arg,
            pred_depth=depth_arg,
            pred_conf=conf_arg,
        )
        if metric_scale is None or metric_scale <= 0:
            logging.getLogger(__name__).warning(
                "Full-attention bootstrap metric anchor returned no scale; leaving scale_factor=1.0"
            )
            return

        state.scale_factor = float(metric_scale)
        self._apply_metric_scale_correction(state, float(metric_scale), predictions)
        predictions["depth"] = predictions["depth"] * float(metric_scale)
        self._metric_bootstrap_done = True
        self._metric_bootstrap_images.clear()
        self._metric_bootstrap_depths.clear()
        self._metric_bootstrap_confs.clear()

    def _register_full_attention_bootstrap_frames(
        self, state, frame_ids, images, predictions
    ):
        """Populate optional fallback/keyframe side buffers for bootstrap frames."""
        should_register_keyframes = (
            self.online_fallback_enabled or self._uses_dynamic_kv_cache()
        )
        depth = predictions.get("depth")
        depth_conf = predictions.get("depth_conf")
        for local_idx, fid in enumerate(frame_ids):
            frame_image = images[:, local_idx : local_idx + 1].detach()
            score = float(state.frame_post_scores.get(fid, 0.0))

            if should_register_keyframes:
                feat = state.frame_select_feats.get(fid)
                if feat is None:
                    feat = state.frame_feats.get(fid)
                pose = state.frame_pose_enc.get(fid)
                if feat is not None and pose is not None:
                    reliability = score if local_idx > 0 else 0.0
                    self._keyframe_registry.maybe_add(
                        fid,
                        feat,
                        frame_image,
                        pose,
                        reliability=reliability,
                        warmup_confidence=None,
                    )

            if self.online_fallback_enabled:
                self._image_buffer.push(fid, frame_image)
                if isinstance(depth, torch.Tensor):
                    self._depth_buffer.push(fid, depth[:, local_idx : local_idx + 1])
                if isinstance(depth_conf, torch.Tensor):
                    self._depth_conf_buffer.push(
                        fid, depth_conf[:, local_idx : local_idx + 1]
                    )

    def _estimate_scale_from_depth(
        self,
        old_depths,
        new_depths,
        old_confs=None,
        new_confs=None,
        frame_weights=None,
        conf_threshold=1.05,
        ransac_iters=200,
        ransac_inlier_ratio_thresh=0.2,
    ):
        """Estimate scale from depth pairs — delegates to scale_estimation module."""
        return _estimate_scale_from_depth_impl(
            self.depth_scale_mode,
            old_depths,
            new_depths,
            old_confs=old_confs,
            new_confs=new_confs,
            frame_weights=frame_weights,
            conf_threshold=conf_threshold,
            ransac_iters=ransac_iters,
            ransac_inlier_ratio_thresh=ransac_inlier_ratio_thresh,
        )

    def _run_segment_pgo(self, state, exclude_frame_ids=None):
        """Run PGO on accumulated pose edges to refine surviving frame poses."""
        edges = self._pose_edge_log.get_all_edges()
        if not edges:
            return

        frame_ids = list(state.frame_order)
        if len(frame_ids) <= 2:
            return

        fid_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
        S = len(frame_ids)
        sample_pose = next(iter(state.frame_pose_enc.values()))
        device, dtype = sample_pose.device, sample_pose.dtype

        # Build dense tensors from sparse edge log
        rel_pose = torch.zeros(1, S, S, 9, device=device, dtype=dtype)
        rel_conf = torch.full((1, S, S), -10.0, device=device, dtype=dtype)
        rel_conf_t = None
        rel_conf_r = None
        rel_mask = torch.zeros(1, S, S, dtype=torch.bool, device=device)
        has_split_confidence = any(
            edge.confidence_t is not None and edge.confidence_r is not None
            for edge in edges
        )
        if has_split_confidence:
            rel_conf_t = torch.full((1, S, S), -10.0, device=device, dtype=dtype)
            rel_conf_r = torch.full((1, S, S), -10.0, device=device, dtype=dtype)

        for edge in edges:
            if edge.frame_i not in fid_to_idx or edge.frame_j not in fid_to_idx:
                continue
            i, j = fid_to_idx[edge.frame_i], fid_to_idx[edge.frame_j]
            rel_pose[0, i, j] = edge.rel_pose_enc[0].to(device=device, dtype=dtype)
            rel_conf[0, i, j] = edge.confidence
            if (
                rel_conf_t is not None
                and edge.confidence_t is not None
                and edge.confidence_r is not None
            ):
                rel_conf_t[0, i, j] = edge.confidence_t
                rel_conf_r[0, i, j] = edge.confidence_r
            rel_mask[0, i, j] = True

        if not rel_mask.any():
            return

        # Init poses from current state
        init_pose = torch.zeros(1, S, 9, device=device, dtype=dtype)
        for fid, idx in fid_to_idx.items():
            pose = state.frame_pose_enc.get(fid)
            if pose is not None:
                init_pose[0, idx] = pose[0] if pose.dim() > 1 else pose

        exclude_indices = None
        if exclude_frame_ids:
            exclude_indices = {
                fid_to_idx[fid] for fid in exclude_frame_ids if fid in fid_to_idx
            }

        refined = refine_camera_sequence_from_rel_pose(
            rel_pose,
            rel_conf,
            rel_mask,
            init_pose,
            pred_rel_conf_t=rel_conf_t,
            pred_rel_conf_r=rel_conf_r,
            score_mode=self.rel_pose_reconstruction_kwargs.get("score_mode", "auto"),
            exclude_indices=exclude_indices,
        )

        n_excluded = len(exclude_indices) if exclude_indices else 0
        for fid, idx in fid_to_idx.items():
            if exclude_indices and idx in exclude_indices:
                continue
            state.frame_pose_enc[fid] = refined[0, idx : idx + 1]
        print(f"  [PGO] refined {S} frames ({n_excluded} excluded)")

    def _execute_fallback(self, action, state, online_options, da3_kwargs):
        """Execute fallback: flush KV, re-run bridge frames, estimate scale."""
        import logging

        logger = logging.getLogger(__name__)
        attempt_frame_id = (
            max(action.bad_frame_ids)
            if action.bad_frame_ids
            else int(action.fallback_frame_id)
        )
        trial_state = self._clone_online_state_without_kv(state)
        old_image_buffer = copy.deepcopy(self._image_buffer)
        old_depth_buffer = copy.deepcopy(self._depth_buffer)
        old_depth_conf_buffer = copy.deepcopy(self._depth_conf_buffer)
        old_keyframe_registry = copy.deepcopy(self._keyframe_registry)
        old_pose_edge_log = copy.deepcopy(self._pose_edge_log)
        trial_image_buffer = type(self._image_buffer)(
            capacity=self._image_buffer.capacity
        )
        trial_depth_buffer = type(self._depth_buffer)(
            capacity=self._depth_buffer.capacity
        )
        trial_depth_conf_buffer = type(self._depth_conf_buffer)(
            capacity=self._depth_conf_buffer.capacity
        )
        trial_keyframe_registry = self._make_empty_keyframe_registry()
        trial_pose_edge_log = type(self._pose_edge_log)()

        bad_set = set(action.bad_frame_ids)
        for fid in action.bad_frame_ids:
            if fid in trial_state.cache_frame_ids:
                trial_state.cache_frame_ids.remove(fid)
            trial_state.frame_order = [
                frame_id for frame_id in trial_state.frame_order if frame_id != fid
            ]
            trial_state.frame_feats.pop(fid, None)
            trial_state.frame_rel_pose_feats.pop(fid, None)
            trial_state.frame_select_feats.pop(fid, None)
            trial_state.frame_pose_enc.pop(fid, None)
            trial_state.frame_scores.pop(fid, None)
            trial_state.frame_post_scores.pop(fid, None)
            trial_state.frame_score_history.pop(fid, None)
            if trial_state.memory_bank_ids:
                trial_state.memory_bank_ids = [
                    cached_id
                    for cached_id in trial_state.memory_bank_ids
                    if cached_id != fid
                ]
        trial_state.similarity_cache = {
            frame_pair: similarity
            for frame_pair, similarity in trial_state.similarity_cache.items()
            if frame_pair[0] not in bad_set and frame_pair[1] not in bad_set
        }
        old_pose_edge_log.drop_frames(bad_set)

        action.new_ref_id = self._resolve_fallback_ref_id(
            action,
            old_keyframe_registry,
            banked_frame_ids=state.memory_bank_ids if state.memory_bank_ids else None,
        )
        if action.new_ref_id not in trial_state.frame_pose_enc:
            keyframe_pose = old_keyframe_registry.get_pose(action.new_ref_id)
            if keyframe_pose is not None:
                trial_state.frame_pose_enc[action.new_ref_id] = keyframe_pose.clone()

        action.bridge_frame_ids = [
            fid
            for fid in action.bridge_frame_ids
            if old_image_buffer.has_frame(fid) and fid in trial_state.frame_pose_enc
        ]
        if len(action.bridge_frame_ids) < 2:
            logger.warning(
                "Fallback skipped: not enough usable bridge frames in buffer"
            )
            self._finish_failed_fallback(state, action, attempt_frame_id)
            return

        # Bridge mode = "ref is always bridge[0]". The image-buffer trim above can drop the
        # original bridge head, leaving the previously chosen ref strictly before the surviving
        # bridge window. Re-pin to the trimmed bridge[0] so the ref/bridge stay adjacent.
        if self.fallback_ref_mode == "bridge":
            action.new_ref_id = int(action.bridge_frame_ids[0])

        # Demote the ref to a bridge frame when the picked ref has no replayable image. The
        # bridge filter above guarantees each surviving entry has an image in the buffer, so
        # falling through to it is safe even when the picked ref happened to equal bridge[0].
        resolved_ref_id = resolve_replayable_fallback_ref_id(
            action.new_ref_id,
            action.bridge_frame_ids,
            old_image_buffer,
            old_keyframe_registry,
        )
        if resolved_ref_id is None:
            logger.warning("Fallback skipped: no replayable ref image available")
            self._finish_failed_fallback(state, action, attempt_frame_id)
            return
        action.new_ref_id = resolved_ref_id

        pose_template = trial_state.frame_pose_enc[action.bridge_frame_ids[0]]
        pose_device = pose_template.device
        pose_dtype = pose_template.dtype
        if action.new_ref_id not in trial_state.frame_pose_enc:
            keyframe_pose = old_keyframe_registry.get_pose(action.new_ref_id)
            if keyframe_pose is not None:
                trial_state.frame_pose_enc[action.new_ref_id] = keyframe_pose.to(
                    device=pose_device, dtype=pose_dtype
                )
        action.new_ref_id = resolve_temporal_fallback_ref_id(
            action.new_ref_id, action.bridge_frame_ids
        )

        old_bridge_poses = [
            trial_state.frame_pose_enc[fid].clone() for fid in action.bridge_frame_ids
        ]
        old_bridge_scores = [
            float(trial_state.frame_post_scores.get(fid, 0.0))
            for fid in action.bridge_frame_ids
        ]
        old_ref_pose = trial_state.frame_pose_enc.get(action.new_ref_id)
        if old_ref_pose is not None:
            old_ref_pose = old_ref_pose.clone()
        old_depths = []
        old_confs = []
        for fid in action.bridge_frame_ids:
            depth = old_depth_buffer.get(fid)
            old_depths.append(depth.clone() if depth is not None else None)
            conf = old_depth_conf_buffer.get(fid)
            old_confs.append(conf.clone() if conf is not None else None)

        # Save new_ref old depth for scale estimation (skip if already in bridge to avoid double-counting)
        ref_in_bridge = action.new_ref_id in set(action.bridge_frame_ids)
        old_ref_depth = None
        old_ref_depth_conf = None
        if not ref_in_bridge:
            d = old_depth_buffer.get(action.new_ref_id)
            old_ref_depth = d.clone() if d is not None else None
            c = old_depth_conf_buffer.get(action.new_ref_id)
            old_ref_depth_conf = c.clone() if c is not None else None

        self._clear_online_runtime_state(trial_state)
        device = next(self.parameters()).device

        if self.fallback_replay_attention == "full":
            # Assemble replay batch: [ref, bridge]
            replay_ids = list(
                dict.fromkeys([action.new_ref_id] + action.bridge_frame_ids)
            )
            replay_images = []
            for fid in replay_ids:
                img = old_image_buffer.get(fid)
                if img is None:
                    img = old_keyframe_registry.get_image(fid)
                if img is None:
                    logger.warning("Fallback skipped: missing image for frame %s", fid)
                    self._finish_failed_fallback(state, action, attempt_frame_id)
                    return
                img = img.to(device)
                while img.dim() < 5:
                    img = img.unsqueeze(0)
                replay_images.append(img)

            replay_result = self._forward_full_attention_replay(
                frame_ids=replay_ids,
                images=replay_images,
                state=trial_state,
                reference_frame_id=action.new_ref_id,
                da3_kwargs=da3_kwargs,
            )

            # Update trial state from replay results. Full-attention replay returns
            # dense K/V; paged mode repopulates its page table from those slices.
            if self._uses_paged_kv_backend():
                self._populate_paged_kv_from_replay(
                    trial_state,
                    replay_ids,
                    replay_images[0],
                    replay_result["kv_cache_list"],
                )
                trial_state.kv_cache_list = None
            else:
                trial_state.kv_cache_list = replay_result["kv_cache_list"]
            trial_state.cache_frame_ids = list(replay_ids)
            trial_state.frame_order = list(replay_ids)
            if trial_state.kv_cache_list is not None:
                trial_state.tokens_per_frame = self._infer_kv_tokens_per_frame(
                    trial_state.kv_cache_list,
                    num_frames=len(replay_ids),
                )
            for fid in replay_ids:
                if fid in replay_result["per_frame_pose_enc"]:
                    trial_state.frame_pose_enc[fid] = replay_result[
                        "per_frame_pose_enc"
                    ][fid]
                if fid in replay_result["per_frame_post_scores"]:
                    trial_state.frame_post_scores[fid] = replay_result[
                        "per_frame_post_scores"
                    ][fid]
                    trial_state.frame_scores[fid] = replay_result[
                        "per_frame_post_scores"
                    ][fid]
                if fid in replay_result.get("per_frame_cam_feat", {}):
                    trial_state.frame_feats[fid] = replay_result["per_frame_cam_feat"][
                        fid
                    ]
                if fid in replay_result.get("per_frame_rel_pose_feat", {}):
                    trial_state.frame_rel_pose_feats[fid] = replay_result[
                        "per_frame_rel_pose_feat"
                    ][fid]
                if fid in replay_result.get("per_frame_select_feat", {}):
                    trial_state.frame_select_feats[fid] = replay_result[
                        "per_frame_select_feat"
                    ][fid]
                source_image = old_image_buffer.get(fid)
                if source_image is None:
                    source_image = old_keyframe_registry.get_image(fid)
                if source_image is not None:
                    trial_image_buffer.push(fid, source_image)

            # Store depth in trial buffers
            for fid, depth_val in replay_result["depth"].items():
                trial_depth_buffer.push(fid, depth_val)
            for fid, conf_val in replay_result["depth_conf"].items():
                trial_depth_conf_buffer.push(fid, conf_val)

            # Log pose edges from full-attention output
            rel_pose_enc_out = self._get_output_value(
                replay_result["output"], "rel_pose_enc"
            )
            rel_pose_conf_out = self._get_output_value(
                replay_result["output"], "rel_pose_conf"
            )
            rel_pose_conf_t_out = self._get_output_value(
                replay_result["output"], "rel_pose_conf_t"
            )
            rel_pose_conf_r_out = self._get_output_value(
                replay_result["output"], "rel_pose_conf_r"
            )
            rel_pose_mask_out = self._get_output_value(
                replay_result["output"], "rel_pose_mask"
            )
            if isinstance(rel_pose_enc_out, torch.Tensor):
                for i, fid in enumerate(replay_ids):
                    if i == 0:
                        continue
                    log_online_pose_edges(
                        pose_edge_log=trial_pose_edge_log,
                        state=trial_state,
                        current_frame_id=fid,
                        rel_pose_frame_ids=replay_ids[: i + 1],
                        rel_pose_enc=rel_pose_enc_out[:, : i + 1, : i + 1],
                        rel_pose_conf=rel_pose_conf_out[:, : i + 1, : i + 1],
                        rel_pose_conf_t=rel_pose_conf_t_out[:, : i + 1, : i + 1]
                        if isinstance(rel_pose_conf_t_out, torch.Tensor)
                        else None,
                        rel_pose_conf_r=rel_pose_conf_r_out[:, : i + 1, : i + 1]
                        if isinstance(rel_pose_conf_r_out, torch.Tensor)
                        else None,
                        rel_pose_mask=rel_pose_mask_out[:, : i + 1, : i + 1],
                        in_fallback=True,
                    )

            for fid in replay_ids:
                feat = trial_state.frame_select_feats.get(fid)
                if feat is None:
                    feat = trial_state.frame_feats.get(fid)
                image = trial_image_buffer.get(fid)
                pose = trial_state.frame_pose_enc.get(fid)
                if feat is not None and image is not None:
                    has_incoming_edges = replay_ids.index(fid) > 0
                    reliability = (
                        float(trial_state.frame_post_scores.get(fid, 0.0))
                        if has_incoming_edges
                        else 0.0
                    )
                    trial_keyframe_registry.maybe_add(
                        fid,
                        feat,
                        image,
                        pose,
                        reliability=reliability,
                    )
        else:
            # --- existing causal replay path ---
            rerun_ids = list(
                dict.fromkeys([action.new_ref_id] + action.bridge_frame_ids)
            )
            rerun_images = []
            for fid in rerun_ids:
                img = old_image_buffer.get(fid)
                if img is None:
                    img = old_keyframe_registry.get_image(fid)
                if img is None:
                    logger.warning("Fallback skipped: missing image for frame %s", fid)
                    self._finish_failed_fallback(state, action, attempt_frame_id)
                    return
                rerun_images.append(img)

            original_image_buffer = self._image_buffer
            original_depth_buffer = self._depth_buffer
            original_depth_conf_buffer = self._depth_conf_buffer
            original_keyframe_registry = self._keyframe_registry
            original_pose_edge_log = self._pose_edge_log

            self._in_fallback = True
            self._image_buffer = trial_image_buffer
            self._depth_buffer = trial_depth_buffer
            self._depth_conf_buffer = trial_depth_conf_buffer
            self._keyframe_registry = trial_keyframe_registry
            self._pose_edge_log = trial_pose_edge_log
            try:
                for fid, img in zip(rerun_ids, rerun_images):
                    self._run_online_step(
                        frame_images=img.to(device),
                        state=trial_state,
                        online_options=online_options,
                        da3_kwargs=da3_kwargs,
                        current_frame_id=fid,
                        force_reference_token=(fid == action.new_ref_id),
                    )
            finally:
                self._in_fallback = False
                self._image_buffer = original_image_buffer
                self._depth_buffer = original_depth_buffer
                self._depth_conf_buffer = original_depth_conf_buffer
                self._keyframe_registry = original_keyframe_registry
                self._pose_edge_log = original_pose_edge_log

        new_bridge_poses = []
        for fid in action.bridge_frame_ids:
            pose = trial_state.frame_pose_enc.get(fid)
            if pose is None:
                logger.warning(
                    "Fallback rejected: rerun missing pose for bridge frame %s", fid
                )
                self._finish_failed_fallback(state, action, attempt_frame_id)
                return
            new_bridge_poses.append(pose.clone())

        bridge_confidences = [
            max(float(trial_state.frame_post_scores.get(fid, 0.0)), 1e-4)
            for fid in action.bridge_frame_ids
        ]
        pose_scale = self._scale_estimator.estimate(
            old_bridge_poses, new_bridge_poses, bridge_confidences
        )
        new_bridge_scores = [
            float(trial_state.frame_post_scores.get(fid, 0.0))
            for fid in action.bridge_frame_ids
        ]
        # Forced (max_segment) fallbacks must always cut the KV cache even when confidence
        # does not improve; otherwise repeated rejections let the segment grow unbounded.
        if (
            not action.forced
            and not self.fallback_skip_confidence_check
            and not self._fallback_improves_bridge_scores(
                old_bridge_scores, new_bridge_scores
            )
        ):
            logger.warning(
                "Fallback rejected: bridge confidence did not improve (old=%.4f, new=%.4f)",
                sum(old_bridge_scores) / len(old_bridge_scores),
                sum(new_bridge_scores) / len(new_bridge_scores),
            )
            state.last_fallback_frame_id = attempt_frame_id
            self._record_rejected_ref(action)
            # Keep the original poses/depths (the replay was not good enough), but in dynamic
            # KV cache mode still re-anchor the live segment at the ref/bridge so repeated
            # low-confidence rejections cannot let the cache grow past max_segment_frames.
            self._cut_dynamic_segment_after_rejected_fallback(state, action)
            self._fallback_manager.on_fallback_complete()
            return

        new_depths = [trial_depth_buffer.get(fid) for fid in action.bridge_frame_ids]
        new_confs = [
            trial_depth_conf_buffer.get(fid) for fid in action.bridge_frame_ids
        ]

        metric_scale = None
        if self.metric_scale_enabled:
            # Pool ref plus bridge frames so fallback scale is not decided by one view.
            metric_frame_ids = []
            for fid in [action.new_ref_id] + list(action.bridge_frame_ids):
                if fid not in metric_frame_ids:
                    metric_frame_ids.append(fid)
            metric_images = []
            metric_depths = []
            metric_confs = []
            for fid in metric_frame_ids:
                img = old_image_buffer.get(fid)
                if img is None:
                    img = old_keyframe_registry.get_image(fid)
                depth_val = trial_depth_buffer.get(fid)
                conf_val = trial_depth_conf_buffer.get(fid)
                if img is None or depth_val is None:
                    continue
                img = img.to(device)
                while img.dim() < 5:
                    img = img.unsqueeze(0)
                metric_images.append(img)
                metric_depths.append(depth_val)
                metric_confs.append(conf_val)
            if metric_images:
                metric_scale = self._compute_metric_scale_factor(
                    image=metric_images,
                    pred_depth=metric_depths,
                    pred_conf=metric_confs
                    if any(c is not None for c in metric_confs)
                    else None,
                )
            if metric_scale is None:
                logger.warning(
                    "Fallback metric scale unavailable; falling back to RANSAC depth scale"
                )

        depth_scale = None
        if not self.metric_scale_enabled or metric_scale is None:
            valid_depth_pairs = []
            for old_depth, new_depth, old_conf, new_conf in zip(
                old_depths, new_depths, old_confs, new_confs
            ):
                if old_depth is not None and new_depth is not None:
                    valid_depth_pairs.append((old_depth, new_depth, old_conf, new_conf))
            # Include new_ref depth pair if it's not already in bridge
            if not ref_in_bridge and old_ref_depth is not None:
                new_ref_depth = trial_depth_buffer.get(action.new_ref_id)
                new_ref_depth_conf = trial_depth_conf_buffer.get(action.new_ref_id)
                if new_ref_depth is not None:
                    valid_depth_pairs.insert(
                        0,
                        (
                            old_ref_depth,
                            new_ref_depth,
                            old_ref_depth_conf,
                            new_ref_depth_conf,
                        ),
                    )
            if valid_depth_pairs:
                depth_scale = self._estimate_scale_from_depth(
                    [pair[0] for pair in valid_depth_pairs],
                    [pair[1] for pair in valid_depth_pairs],
                    old_confs=[pair[2] for pair in valid_depth_pairs],
                    new_confs=[pair[3] for pair in valid_depth_pairs],
                )

        if metric_scale is not None and metric_scale > 0:
            scale = float(metric_scale)
            prev_scale = float(state.scale_factor)
            if prev_scale > 0 and (
                scale / prev_scale > 2.0 or prev_scale / scale > 2.0
            ):
                logger.warning(
                    "Fallback metric scale (%.4f) differs >2x from previous segment (%.4f)",
                    scale,
                    prev_scale,
                )
        else:
            scale = self._resolve_fallback_scale(
                pose_scale=pose_scale, depth_scale=depth_scale
            )
        trial_state.scale_factor = scale

        # Save depth comparison figures for scale debugging
        if self.fallback_debug_dir:
            from R3.models.online.fallback_vis import save_fallback_depth_comparison

            save_fallback_depth_comparison(
                output_dir=self.fallback_debug_dir,
                fallback_idx=self._fallback_debug_count,
                bridge_frame_ids=action.bridge_frame_ids,
                old_depths=old_depths,
                new_depths=new_depths,
                old_confs=old_confs,
                new_confs=new_confs,
                pose_scale=pose_scale,
                depth_scale=depth_scale,
                final_scale=scale,
                old_bridge_scores=old_bridge_scores,
                new_bridge_scores=new_bridge_scores,
            )
            self._fallback_debug_count += 1

        new_ref_pose = trial_state.frame_pose_enc.get(action.new_ref_id)
        if old_ref_pose is None or new_ref_pose is None:
            logger.warning(
                "Fallback rejected: missing reference pose for bridge fusion"
            )
            self._record_rejected_ref(action)
            self._finish_failed_fallback(state, action, attempt_frame_id)
            return
        old_ref_pose = old_ref_pose.to(
            device=new_ref_pose.device, dtype=new_ref_pose.dtype
        )

        bridge_frame_set = set(action.bridge_frame_ids)
        allow_missing_old_ref_edge = action.new_ref_id not in bridge_frame_set
        for fid, old_pose, new_pose, old_score, new_score in zip(
            action.bridge_frame_ids,
            old_bridge_poses,
            new_bridge_poses,
            old_bridge_scores,
            new_bridge_scores,
        ):
            old_rel_pose = resolve_fallback_bridge_rel_pose(
                pose_edge_log=old_pose_edge_log,
                ref_frame_id=action.new_ref_id,
                target_frame_id=fid,
                edge_type="normal",
                ref_pose=old_ref_pose,
                target_pose=old_pose,
                allow_missing=allow_missing_old_ref_edge,
                fallback_edge_types=("bridge",),
            )
            new_rel_pose = resolve_fallback_bridge_rel_pose(
                pose_edge_log=trial_pose_edge_log,
                ref_frame_id=action.new_ref_id,
                target_frame_id=fid,
                edge_type="bridge",
                ref_pose=new_ref_pose,
                target_pose=new_pose,
            )
            if old_rel_pose is None:
                scaled_new_rel_pose = new_rel_pose.clone()
                scaled_new_rel_pose[..., :3] *= float(scale)
                trial_state.frame_pose_enc[fid] = compose_relative_pose(
                    scaled_new_rel_pose.unsqueeze(1),
                    old_ref_pose.unsqueeze(1),
                )[:, 0]
                continue
            trial_state.frame_pose_enc[fid] = fuse_fallback_bridge_pose(
                old_ref_pose=old_ref_pose,
                old_rel_pose=old_rel_pose,
                new_rel_pose=new_rel_pose,
                old_score=old_score,
                new_score=new_score,
                scale=scale,
            )

        for fid, old_depth, new_depth, old_conf, new_conf in zip(
            action.bridge_frame_ids,
            old_depths,
            new_depths,
            old_confs,
            new_confs,
        ):
            if old_depth is None or new_depth is None:
                continue
            fused_depth, fused_conf = fuse_fallback_depth_pair(
                old_depth=old_depth,
                new_depth=new_depth,
                old_conf=old_conf,
                new_conf=new_conf,
                scale=scale,
            )
            trial_depth_buffer.push(fid, fused_depth)
            if fused_conf is not None:
                trial_depth_conf_buffer.push(fid, fused_conf)

        if action.new_ref_id not in set(action.bridge_frame_ids):
            restored_ref_pose = old_ref_pose
            if restored_ref_pose is None:
                restored_ref_pose = old_keyframe_registry.get_pose(action.new_ref_id)
            if restored_ref_pose is not None:
                trial_state.frame_pose_enc[action.new_ref_id] = (
                    restored_ref_pose.clone()
                )

        if self._uses_dynamic_kv_cache():
            anchor_candidates = list(
                dict.fromkeys([action.new_ref_id, *action.bridge_frame_ids])
            )
            self._set_dynamic_segment_anchor_frame_ids(
                trial_state,
                anchor_candidates,
                preferred_anchor_id=action.new_ref_id,
            )

        if (
            trial_state.kv_cache_list is not None
            and trial_state.tokens_per_frame is not None
        ):
            trial_state.kv_cache_list, trial_state.cache_frame_ids = (
                self._apply_online_cache_policy(
                    trial_state,
                    trial_state.kv_cache_list,
                    trial_state.cache_frame_ids,
                    keyframe_registry=trial_keyframe_registry,
                )
            )
        elif trial_state.paged_kv_store is not None:
            _, trial_state.cache_frame_ids = self._apply_online_cache_policy(
                trial_state,
                None,
                trial_state.cache_frame_ids,
                keyframe_registry=trial_keyframe_registry,
            )
        trial_state.last_fallback_frame_id = attempt_frame_id

        self._copy_online_state(state, trial_state)
        self._image_buffer = trial_image_buffer
        self._depth_buffer = trial_depth_buffer
        self._depth_conf_buffer = trial_depth_conf_buffer
        self._keyframe_registry = trial_keyframe_registry
        # Snapshot bridge edges before the live log is replaced. Without this, every accepted
        # fallback wipes out earlier bridges and only the most recent bridge survives in the
        # exported log.
        self._historical_bridge_edges.extend(
            edge for edge in trial_pose_edge_log._edges if edge.edge_type == "bridge"
        )
        self._pose_edge_log = trial_pose_edge_log

        # Registry is replaced with the trial one (which only contains replay frames), so
        # pre-accept rejected refs are no longer reachable and must not gate future resolves.
        self._rejected_ref_ids.clear()
        self._fallback_manager.on_fallback_complete()
        print(
            f"  [FALLBACK] accepted: ref={action.new_ref_id}, bridge={action.bridge_frame_ids}, scale={scale:.4f}"
        )

        self._metric_bootstrap_images.clear()
        self._metric_bootstrap_depths.clear()
        self._metric_bootstrap_confs.clear()
        self._metric_bootstrap_done = True

        # Run segment PGO: refine all poses, excluding both previous and current fallback's bridge frames
        current_bridge_set = set(action.bridge_frame_ids)
        if not self.disable_segment_pgo:
            self._run_segment_pgo(
                state,
                exclude_frame_ids=self._previous_bridge_frame_ids | current_bridge_set,
            )
        self._previous_bridge_frame_ids = current_bridge_set

    def _forward_online(self, images, mode: str, **kwargs):
        # Online forward: process frames sequentially with full causal KV cache.
        if images.dim() == 4:
            images = images.unsqueeze(1)
        self._validate_paged_kv_request(mode)

        B, S, _, _, _ = images.shape
        pose_max_recent = int(kwargs.pop("pose_max_recent", 0))
        bootstrap_full_attention_frames = int(
            kwargs.pop("bootstrap_full_attention_frames", 0) or 0
        )
        state = self._ensure_online_state(B)
        online_options, da3_kwargs = self._extract_online_step_options(kwargs)
        base_state = state
        frame_ids = list(range(int(state.frame_count), int(state.frame_count) + S))

        bootstrap_predictions = None
        bootstrap_frame_ids = []
        if (
            bootstrap_full_attention_frames > 1
            and int(state.frame_count) == 0
            and S > 1
        ):
            if self.online_kv_cache_mode != "all":
                raise ValueError(
                    "bootstrap_full_attention_frames currently requires online_kv_cache_mode='all'"
                )
            bootstrap_n = min(int(bootstrap_full_attention_frames), int(S))
            bootstrap_frame_ids = frame_ids[:bootstrap_n]
            bootstrap_images = [
                images[:, idx : idx + 1] for idx in range(bootstrap_n)
            ]
            if online_options["online_verbose"]:
                print(
                    f"[R3][online-bootstrap] full attention on first {bootstrap_n} frames"
                )
            replay_result = self._forward_full_attention_replay(
                frame_ids=bootstrap_frame_ids,
                images=bootstrap_images,
                state=state,
                reference_frame_id=bootstrap_frame_ids[0],
                da3_kwargs=da3_kwargs,
            )
            self._seed_online_state_from_full_attention_replay(
                state,
                bootstrap_frame_ids,
                bootstrap_images,
                replay_result,
            )
            _, _, _, H, W = images.shape
            bootstrap_predictions = self._format_predictions(
                replay_result["output"],
                images[:, :bootstrap_n],
                H,
                W,
                online_options["rel_pose_reconstruction_method"],
                online_options["rel_pose_reconstruction_kwargs"],
                online_mode=False,
            )
            bootstrap_predictions["rel_pose_frame_ids"] = bootstrap_frame_ids
            self._apply_full_attention_bootstrap_metric_scale(
                state,
                images[:, :bootstrap_n],
                bootstrap_predictions,
            )
            self._register_full_attention_bootstrap_frames(
                state,
                bootstrap_frame_ids,
                images[:, :bootstrap_n],
                bootstrap_predictions,
            )
            base_state = state

        remaining_start = len(bootstrap_frame_ids)
        if remaining_start < S:
            remaining_predictions, updated_state = run_online_sequence_pass(
                self,
                images[:, remaining_start:],
                online_options=online_options,
                da3_kwargs=da3_kwargs,
                base_state=base_state,
                frame_ids=frame_ids[remaining_start:],
                output_frame_ids=frame_ids[remaining_start:],
                pose_max_recent=pose_max_recent,
            )
            self.online_state = updated_state
            if bootstrap_predictions is not None:
                merged_predictions = self._merge_online_step_predictions(
                    [bootstrap_predictions, remaining_predictions]
                )
                retained_frame_ids = list(bootstrap_frame_ids) + list(
                    remaining_predictions.get(
                        "output_frame_ids", frame_ids[remaining_start:]
                    )
                )
                merged_predictions["frame_post_scores"] = {
                    frame_id: float(
                        self._persistent_post_scores.get(
                            frame_id,
                            updated_state.frame_post_scores.get(
                                frame_id, float("inf")
                            ),
                        )
                    )
                    for frame_id in retained_frame_ids
                }
                merged_predictions["frame_score_history"] = {
                    frame_id: list(
                        updated_state.frame_score_history.get(frame_id, [])
                    )
                    for frame_id in retained_frame_ids
                }
                merged_predictions["output_frame_ids"] = retained_frame_ids
            else:
                merged_predictions = remaining_predictions
        else:
            updated_state = state
            self.online_state = updated_state
            merged_predictions = bootstrap_predictions
            retained_frame_ids = list(bootstrap_frame_ids)
            merged_predictions["frame_post_scores"] = {
                frame_id: float(
                    self._persistent_post_scores.get(
                        frame_id,
                        updated_state.frame_post_scores.get(frame_id, float("inf")),
                    )
                )
                for frame_id in retained_frame_ids
            }
            merged_predictions["frame_score_history"] = {
                frame_id: list(updated_state.frame_score_history.get(frame_id, []))
                for frame_id in retained_frame_ids
            }
            merged_predictions["output_frame_ids"] = retained_frame_ids

        if S > 1 and online_options["online_finalize_pose_reconstruction"]:
            output_frame_ids = merged_predictions.get("output_frame_ids", frame_ids)
            finalized_pose_enc = finalize_online_pose_sequence(
                merged_predictions,
                rel_pose_reconstruction_method=online_options[
                    "rel_pose_reconstruction_method"
                ],
                rel_pose_reconstruction_kwargs=online_options[
                    "rel_pose_reconstruction_kwargs"
                ],
                output_frame_ids=output_frame_ids,
            )
            if finalized_pose_enc is not None:
                merged_predictions["pose_enc_online_local"] = merged_predictions[
                    "pose_enc"
                ]
                merged_predictions["pose_enc"] = finalized_pose_enc

        return merged_predictions
