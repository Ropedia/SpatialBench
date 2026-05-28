"""Paged KV backend glue for the R3 online wrapper.

These helpers read configuration from the wrapper and the underlying DA3 backbone,
then create, route, or repopulate the FlashInfer paged KV store. They are split out
of `da3_wrapper.py` to keep that file focused on top-level forward/state orchestration;
the actual paged store implementation lives in `flashinfer_backend.py`.
"""

import torch

from R3.models.online.flashinfer_backend import create_paged_kv_store


def uses_paged_kv_backend(wrapper) -> bool:
    """Return whether online inference should route global attention through FlashInfer pages."""
    return wrapper.online_kv_backend == "paged"


def get_backbone_attention_mode(wrapper) -> str:
    """Read the configured global attention mode from the underlying DA3 backbone."""
    backbone_model = wrapper._get_backbone_model()
    return str(getattr(backbone_model, "attention_mode", "causal"))


def validate_paged_kv_request(wrapper, mode: str) -> None:
    """Reject paged KV combinations that are not wired in the current rollout phase."""
    if not uses_paged_kv_backend(wrapper):
        return
    if mode != "causal":
        raise NotImplementedError(
            "online_kv_backend='paged' currently supports mode='causal' only"
        )
    if get_backbone_attention_mode(wrapper) != "causal":
        raise NotImplementedError(
            "online_kv_backend='paged' requires the DA3 backbone attention_mode='causal'"
        )


def infer_paged_tokens_per_frame(wrapper, frame_images) -> int:
    """Infer ViT token count for one frame from image shape and backbone patch metadata."""
    backbone_model = wrapper._get_backbone_model()
    patch_size = getattr(
        backbone_model, "patch_size", getattr(wrapper.da3, "PATCH_SIZE", 14)
    )
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"Invalid backbone patch_size: {patch_size}")
    height, width = int(frame_images.shape[-2]), int(frame_images.shape[-1])
    num_register_tokens = int(getattr(backbone_model, "num_register_tokens", 0))
    return 1 + num_register_tokens + (height // patch_h) * (width // patch_w)


def get_paged_global_layer_count(wrapper) -> int:
    """Count standard global attention layers backed by the paged KV store."""
    backbone_model = wrapper._get_backbone_model()
    alt_start = int(getattr(backbone_model, "alt_start", -1))
    if alt_start < 0:
        raise RuntimeError(
            "Paged KV backend requires a DA3 backbone with global attention layers"
        )

    blocks = getattr(backbone_model, "blocks", None)
    block_count = (
        len(blocks)
        if blocks is not None
        else int(getattr(backbone_model, "n_blocks", 0))
    )
    use_gla = bool(getattr(backbone_model, "use_gla", False))
    full_attention_idx = getattr(backbone_model, "full_attention_idx", None)
    count = 0
    for block_idx in range(block_count):
        is_global_layer = block_idx >= alt_start and block_idx % 2 == 1
        is_gla_layer = use_gla and (
            full_attention_idx is None or block_idx not in full_attention_idx
        )
        if is_global_layer and not is_gla_layer:
            count += 1
    if count <= 0:
        raise RuntimeError(
            "Paged KV backend found no standard global attention layers to cache"
        )
    return count


def get_paged_attention_shape(wrapper, frame_images):
    """Resolve heads and head dimension used by the DA3 global attention blocks."""
    backbone_model = wrapper._get_backbone_model()
    num_heads = int(getattr(backbone_model, "num_heads", 0))
    embed_dim = int(getattr(backbone_model, "embed_dim", 0))
    if num_heads <= 0 or embed_dim <= 0 or embed_dim % num_heads != 0:
        raise RuntimeError(
            f"Cannot infer paged KV attention shape from backbone: embed_dim={embed_dim}, num_heads={num_heads}"
        )
    return get_paged_global_layer_count(wrapper), num_heads, embed_dim // num_heads


def infer_online_cache_dtype(wrapper, frame_images):
    """Choose the dtype used for paged KV buffers."""
    device_type = frame_images.device.type
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except TypeError:
        if device_type == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    try:
        first_param = next(wrapper.da3.parameters())
        if frame_images.is_cuda and first_param.dtype == torch.float32:
            return torch.bfloat16
        return first_param.dtype
    except StopIteration:
        return frame_images.dtype


def ensure_paged_kv_store(wrapper, state, frame_images):
    """Create the per-state FlashInfer paged KV store on first use."""
    if state.paged_kv_store is not None:
        return state.paged_kv_store

    tokens_per_frame = infer_paged_tokens_per_frame(wrapper, frame_images)
    num_layers, num_heads, head_dim = get_paged_attention_shape(wrapper, frame_images)
    state.tokens_per_frame = tokens_per_frame
    state.paged_kv_store = create_paged_kv_store(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        tokens_per_frame=tokens_per_frame,
        dtype=infer_online_cache_dtype(wrapper, frame_images),
        device=frame_images.device,
        kv_cache_mode=wrapper.online_kv_cache_mode,
        recent_frames=wrapper.online_recent_frames,
        bank_initial_frames=wrapper.bank_initial_frames,
        keyframe_max_keyframes=wrapper.keyframe_max_keyframes,
        flashinfer_page_size=wrapper.flashinfer_page_size,
    )
    return state.paged_kv_store


def prepare_paged_kv_da3_kwargs(
    wrapper, state, frame_images, current_frame_id: int
) -> dict:
    """Build DA3 kwargs that expose the current paged KV cache to the ViT."""
    store = ensure_paged_kv_store(wrapper, state, frame_images)
    active_frame_ids = list(state.cache_frame_ids) + [int(current_frame_id)]
    return {
        "paged_kv_store": store,
        "paged_new_frame_id": int(current_frame_id),
        "paged_active_frame_ids": active_frame_ids,
    }


def populate_paged_kv_from_replay(
    wrapper, state, frame_ids, frame_images_for_shape, kv_cache_list
) -> None:
    """Copy dense per-layer K/V from a full-attention replay into a fresh paged store on `state`.

    The replay path runs through SDPA (the paged store is not passed in), so the returned
    `kv_cache_list` holds dense `[B, H, N*tokens_per_frame, D]` tensors per global layer. To
    keep paged inference going after the fallback, slice each layer per frame and write the
    slices into the trial state's paged store, preserving the replay frame order so future
    `begin_step` calls can reference these frames as active.
    """
    if state.paged_kv_store is not None:
        for fid in list(state.paged_kv_store.page_table.keys()):
            state.paged_kv_store.evict_frame(fid)
    store = ensure_paged_kv_store(wrapper, state, frame_images_for_shape)
    n = len(frame_ids)
    tokens_per_frame = store.tokens_per_frame
    paged_layer_idx = 0
    for kv in kv_cache_list or []:
        if kv is None or kv[0] is None or kv[1] is None:
            continue
        if paged_layer_idx >= store.num_layers:
            raise RuntimeError(
                f"Replay produced more global K/V layers ({paged_layer_idx + 1}) "
                f"than paged store has ({store.num_layers})"
            )
        k_dense, v_dense = kv[0], kv[1]
        if k_dense.shape[2] != n * tokens_per_frame:
            raise ValueError(
                f"Dense KV layer length {k_dense.shape[2]} does not match expected "
                f"{n} frames × {tokens_per_frame} tokens"
            )
        for i, fid in enumerate(frame_ids):
            store.allocate_frame(int(fid))
            k_slice = k_dense[
                :, :, i * tokens_per_frame : (i + 1) * tokens_per_frame, :
            ]
            v_slice = v_dense[
                :, :, i * tokens_per_frame : (i + 1) * tokens_per_frame, :
            ]
            if k_slice.dtype != store.dtype:
                k_slice = k_slice.to(store.dtype)
                v_slice = v_slice.to(store.dtype)
            store.write_kv(
                paged_layer_idx, int(fid), k_slice.contiguous(), v_slice.contiguous()
            )
        paged_layer_idx += 1
    if paged_layer_idx != store.num_layers:
        raise RuntimeError(
            f"Replay produced K/V for {paged_layer_idx} layers but paged store expects {store.num_layers}"
        )
