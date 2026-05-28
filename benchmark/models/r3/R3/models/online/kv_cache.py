"""KV cache management utilities extracted from R3 for independent testability."""

import torch


def get_online_cache_keep_frame_ids(
    state, cache_frame_ids, kv_cache_mode, recent_frames
):
    """Select the bounded set of frame IDs that should remain resident in the KV cache."""
    if kv_cache_mode == "all":
        return list(cache_frame_ids)

    regular_ids = list(cache_frame_ids)
    recent_count = max(recent_frames, 1)
    recent_ids = regular_ids[-recent_count:]
    keep_set = set(recent_ids)
    memory_bank_ids = list(getattr(state, "memory_bank_ids", []))
    keep_set |= set(memory_bank_ids)
    return [frame_id for frame_id in cache_frame_ids if frame_id in keep_set]


def get_dynamic_segment_anchor_budget(kv_cache_mode, bank_initial_frames):
    """Return how many frame IDs the dynamic segment should track locally."""
    if kv_cache_mode != "dynamic":
        return 0
    return max(int(bank_initial_frames), 1)


def get_dynamic_segment_frame_order(state):
    """Return the frame-order slice that belongs to the active dynamic segment."""
    frame_order = list(state.frame_order)
    if not frame_order:
        return []

    anchor_ids = [
        frame_id
        for frame_id in getattr(state, "segment_anchor_frame_ids", [])
        if frame_id in frame_order
    ]
    if not anchor_ids:
        return frame_order

    start_frame_id = anchor_ids[0]
    try:
        start_index = frame_order.index(start_frame_id)
    except ValueError:
        state.segment_anchor_frame_ids = []
        return frame_order
    return frame_order[start_index:]


def seed_dynamic_segment_anchor_frame_ids(state, anchor_budget):
    """Seed or refresh tracked anchor IDs for the active dynamic segment."""
    if anchor_budget <= 0:
        state.segment_anchor_frame_ids = []
        return []

    frame_order_set = set(state.frame_order)
    anchor_ids = []
    for frame_id in getattr(state, "segment_anchor_frame_ids", []):
        if frame_id in frame_order_set and frame_id not in anchor_ids:
            anchor_ids.append(frame_id)

    state.segment_anchor_frame_ids = list(anchor_ids)
    candidate_ids = (
        get_dynamic_segment_frame_order(state)
        if anchor_ids
        else list(state.frame_order)
    )
    for frame_id in candidate_ids:
        if frame_id in anchor_ids:
            continue
        anchor_ids.append(frame_id)
        if len(anchor_ids) >= anchor_budget:
            break

    state.segment_anchor_frame_ids = anchor_ids[:anchor_budget]
    return list(state.segment_anchor_frame_ids)


def set_dynamic_segment_anchor_frame_ids(
    state, frame_ids, anchor_budget, preferred_anchor_id=None
):
    """Mark a fresh dynamic segment using replay-local frame IDs only."""
    if anchor_budget <= 0:
        state.segment_anchor_frame_ids = []
        return []

    frame_order_set = set(state.frame_order)
    candidate_ids = list(frame_ids)
    if preferred_anchor_id is not None and preferred_anchor_id in candidate_ids:
        candidate_ids = [preferred_anchor_id] + [
            frame_id for frame_id in candidate_ids if frame_id != preferred_anchor_id
        ]

    state.segment_anchor_frame_ids = [
        frame_id for frame_id in candidate_ids if frame_id in frame_order_set
    ][:anchor_budget]
    return seed_dynamic_segment_anchor_frame_ids(state, anchor_budget)


def sync_dynamic_memory_bank_ids(
    state, kv_cache_mode, bank_initial_frames, keyframe_registry
):
    """Synchronize resident dynamic memory-bank IDs from the active segment and registry."""
    if kv_cache_mode != "dynamic":
        state.memory_bank_ids = []
        state.segment_anchor_frame_ids = []
        return []

    anchor_budget = get_dynamic_segment_anchor_budget(
        kv_cache_mode, bank_initial_frames
    )
    anchor_ids = seed_dynamic_segment_anchor_frame_ids(state, anchor_budget)
    segment_frame_ids = get_dynamic_segment_frame_order(state)
    segment_frame_id_set = set(segment_frame_ids)
    keyframe_ids = set()
    if keyframe_registry is not None:
        keyframe_ids = {
            frame_id
            for frame_id in keyframe_registry.get_keyframe_ids()
            if frame_id in segment_frame_id_set
        }

    pinned_anchor_ids = set(anchor_ids[:bank_initial_frames])
    protected_ids = keyframe_ids | pinned_anchor_ids
    state.memory_bank_ids = [
        frame_id for frame_id in segment_frame_ids if frame_id in protected_ids
    ]
    return list(state.memory_bank_ids)


def upgrade_kv_cache_to_buffers(
    kv_cache_list,
    tokens_per_frame,
    kv_cache_mode,
    recent_frames,
    bank_initial_frames=0,
    keyframe_max_keyframes=0,
):
    """Convert KV cache tensors to pre-allocated buffers so pruning stays stable and cheap."""
    if kv_cache_list is None or tokens_per_frame is None:
        return kv_cache_list

    max_frames = None
    if kv_cache_mode == "dynamic":
        max_frames = max(
            keyframe_max_keyframes
            + max(bank_initial_frames, 0)
            + max(recent_frames, 1)
            + 1,
            1,
        )

    upgraded_kv_cache_list = []
    for kv_cache in kv_cache_list:
        if kv_cache is None or len(kv_cache) != 2:
            upgraded_kv_cache_list.append(kv_cache)
            continue

        k_cache, v_cache = kv_cache
        if k_cache is None or v_cache is None:
            upgraded_kv_cache_list.append(kv_cache)
            continue

        batch_size, num_heads, keep_len, head_dim = k_cache.shape
        target_size = keep_len * 2
        if max_frames is not None:
            target_size = max(target_size, tokens_per_frame * max_frames)

        k_buf = torch.zeros(
            (batch_size, num_heads, target_size, head_dim),
            dtype=k_cache.dtype,
            device=k_cache.device,
        )
        v_buf = torch.zeros(
            (batch_size, num_heads, target_size, head_dim),
            dtype=v_cache.dtype,
            device=v_cache.device,
        )
        k_buf[:, :, :keep_len] = k_cache
        v_buf[:, :, :keep_len] = v_cache
        upgraded_kv_cache_list.append([k_buf, v_buf, keep_len])
    return upgraded_kv_cache_list


def prune_kv_cache_list(
    kv_cache_list, cache_frame_ids, keep_frame_ids, tokens_per_frame
):
    """Slice each KV cache block down to the token ranges for the kept frames only."""
    if (
        kv_cache_list is None
        or tokens_per_frame is None
        or cache_frame_ids == keep_frame_ids
    ):
        return kv_cache_list

    frame_positions = {
        frame_id: position for position, frame_id in enumerate(cache_frame_ids)
    }
    keep_positions = [frame_positions[frame_id] for frame_id in keep_frame_ids]
    if not keep_positions:
        return [
            [None, None] if kv_cache is not None else None for kv_cache in kv_cache_list
        ]

    token_device = None
    for kv_cache in kv_cache_list:
        if kv_cache is not None and kv_cache[0] is not None:
            token_device = kv_cache[0].device
            break
    if token_device is None:
        return kv_cache_list

    base_token_indices = torch.arange(
        tokens_per_frame, device=token_device, dtype=torch.long
    ).unsqueeze(0)
    frame_offsets = torch.tensor(
        keep_positions, device=token_device, dtype=torch.long
    ).unsqueeze(1)
    token_indices = (frame_offsets * tokens_per_frame + base_token_indices).reshape(-1)

    pruned_kv_cache_list = []
    expected_tokens = len(cache_frame_ids) * tokens_per_frame
    keep_len = token_indices.shape[0]
    for kv_cache in kv_cache_list:
        if kv_cache is None or kv_cache[0] is None or kv_cache[1] is None:
            pruned_kv_cache_list.append(kv_cache)
            continue

        current_keep_len = kv_cache[2] if len(kv_cache) > 2 else kv_cache[0].shape[2]
        if current_keep_len < expected_tokens:
            raise ValueError(
                f"KV cache length {current_keep_len} is smaller than expected token count {expected_tokens}"
            )

        if len(kv_cache) > 2:
            k_buf, v_buf, _ = kv_cache
            k_buf[:, :, :keep_len] = k_buf.index_select(2, token_indices)
            v_buf[:, :, :keep_len] = v_buf.index_select(2, token_indices)
            pruned_kv_cache_list.append([k_buf, v_buf, keep_len])
        else:
            pruned_kv_cache_list.append(
                [
                    kv_cache[0].index_select(2, token_indices),
                    kv_cache[1].index_select(2, token_indices),
                ]
            )
    return pruned_kv_cache_list


def prune_online_similarity_cache(state, keep_frame_ids_set):
    """Remove cached similarities for frames that were dropped from the active state."""
    state.similarity_cache = {
        frame_pair: similarity
        for frame_pair, similarity in state.similarity_cache.items()
        if frame_pair[0] in keep_frame_ids_set and frame_pair[1] in keep_frame_ids_set
    }


def prune_online_state(state, keep_frame_ids):
    """Drop state entries for frames that are no longer resident in the active cache."""
    keep_frame_ids_set = set(keep_frame_ids)
    state.frame_order = [
        frame_id for frame_id in state.frame_order if frame_id in keep_frame_ids_set
    ]
    state.frame_feats = {
        frame_id: feat
        for frame_id, feat in state.frame_feats.items()
        if frame_id in keep_frame_ids_set
    }
    state.frame_rel_pose_feats = {
        frame_id: feat
        for frame_id, feat in state.frame_rel_pose_feats.items()
        if frame_id in keep_frame_ids_set
    }
    state.frame_select_feats = {
        frame_id: feat
        for frame_id, feat in state.frame_select_feats.items()
        if frame_id in keep_frame_ids_set
    }
    state.frame_pose_enc = {
        frame_id: pose_enc
        for frame_id, pose_enc in state.frame_pose_enc.items()
        if frame_id in keep_frame_ids_set
    }
    state.frame_scores = {
        frame_id: score
        for frame_id, score in state.frame_scores.items()
        if frame_id in keep_frame_ids_set
    }
    state.segment_anchor_frame_ids = [
        frame_id
        for frame_id in getattr(state, "segment_anchor_frame_ids", [])
        if frame_id in keep_frame_ids_set
    ]
    # frame_post_scores and frame_score_history are read at end-of-run for the per-frame pose
    # confidence aggregate; pruning them along with the KV cache makes most frames look like
    # conf=0 in the final summary even though they were processed fine. Keep them in sync with
    # TTT (accumulate-forever) instead.
    prune_online_similarity_cache(state, keep_frame_ids_set)


def evict_frame_from_kv(state, frame_id):
    """Remove a single frame's KV tokens from the cache."""
    if frame_id not in state.cache_frame_ids:
        return

    pos = state.cache_frame_ids.index(frame_id)
    tokens_per_frame = state.tokens_per_frame
    if tokens_per_frame is None or state.kv_cache_list is None:
        return

    token_start = pos * tokens_per_frame
    token_end = token_start + tokens_per_frame

    for i, kv_cache in enumerate(state.kv_cache_list):
        if kv_cache is None:
            continue
        k_buf, v_buf = kv_cache[0], kv_cache[1]
        if k_buf is None or v_buf is None:
            continue
        has_keep_len = len(kv_cache) > 2
        keep_len = kv_cache[2] if has_keep_len else k_buf.shape[2]
        if keep_len < token_end:
            raise ValueError(
                f"KV cache length {keep_len} is smaller than token range end {token_end}"
            )
        if has_keep_len:
            # Pre-allocated buffer: shift in-place and adjust keep_len
            if keep_len > token_end:
                k_buf[:, :, token_start : keep_len - tokens_per_frame] = k_buf[
                    :, :, token_end:keep_len
                ].clone()
                v_buf[:, :, token_start : keep_len - tokens_per_frame] = v_buf[
                    :, :, token_end:keep_len
                ].clone()
            state.kv_cache_list[i] = [k_buf, v_buf, max(0, keep_len - tokens_per_frame)]
        else:
            # 2-element format: create new sliced tensors
            k_new = torch.cat(
                [k_buf[:, :, :token_start], k_buf[:, :, token_end:keep_len]], dim=2
            )
            v_new = torch.cat(
                [v_buf[:, :, :token_start], v_buf[:, :, token_end:keep_len]], dim=2
            )
            state.kv_cache_list[i] = [k_new, v_new]

    # Remove from state
    state.cache_frame_ids.remove(frame_id)
    state.frame_order = [f for f in state.frame_order if f != frame_id]
    state.frame_feats.pop(frame_id, None)
    state.frame_rel_pose_feats.pop(frame_id, None)
    state.frame_select_feats.pop(frame_id, None)
    state.frame_pose_enc.pop(frame_id, None)
    state.frame_scores.pop(frame_id, None)
    # Keep frame_post_scores / frame_score_history so the end-of-run pose-conf aggregate
    # reports a non-zero value for evicted frames.
    if state.memory_bank_ids:
        state.memory_bank_ids = [
            cached_id for cached_id in state.memory_bank_ids if cached_id != frame_id
        ]
    state.segment_anchor_frame_ids = [
        anchor_id
        for anchor_id in getattr(state, "segment_anchor_frame_ids", [])
        if anchor_id != frame_id
    ]
    prune_online_similarity_cache(state, set(state.frame_order))
