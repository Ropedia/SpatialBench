"""FlashInfer paged KV cache backend for online inference.

v1 design: page_size == tokens_per_frame so each frame occupies exactly one page,
which keeps FlashInfer's paged-KV invariant ("all pages full except last") trivially
satisfied across the concatenation of any active subset of frames. See
`agent/architecture/flashinfer_paged_kv.md` for the design notes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def max_frames_for_mode(
    kv_cache_mode: str,
    recent_frames: int,
    bank_initial_frames: int = 0,
    keyframe_max_keyframes: int = 0,
    default_all_mode_capacity: int = 1024,
) -> int:
    """Budget identical to `upgrade_kv_cache_to_buffers` in kv_cache.py."""
    recent = max(recent_frames, 1)
    if kv_cache_mode == "dynamic":
        budget = max(
            keyframe_max_keyframes + max(bank_initial_frames, 0) + recent + 1, 1
        )
    elif kv_cache_mode == "all":
        # Unbounded in the dense path; we pre-allocate a generous ceiling for paged
        # and fail loudly if the run exceeds it (the user can grow explicitly later).
        budget = max(default_all_mode_capacity, 1)
    else:
        raise ValueError(f"Unsupported kv_cache_mode: {kv_cache_mode}")
    return budget


def _require_flashinfer():
    try:
        import flashinfer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "FlashInfer is not installed but online_kv_backend='paged' was requested. "
            "See agent/architecture/flashinfer_paged_kv.md for install steps."
        ) from exc
    return flashinfer


class PagedKVStore:
    """Per-layer paged KV cache backed by FlashInfer's BatchPrefillWithPagedKVCacheWrapper.

    One instance per OnlineState. Owns the paged K/V tensors for every global causal
    attention layer, a shared page_table keyed by frame_id, and a FlashInfer wrapper
    that is re-planned at the start of every step.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        tokens_per_frame: int,
        dtype: torch.dtype,
        device: torch.device,
        max_frames: int,
        workspace_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if tokens_per_frame <= 0:
            raise ValueError(
                f"tokens_per_frame must be positive, got {tokens_per_frame}"
            )
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")

        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.tokens_per_frame = int(tokens_per_frame)
        self.page_size = int(tokens_per_frame)  # v1: one page per frame
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_frames = int(max_frames)
        self.num_pages = int(max_frames)

        # Per-layer paged K and V buffers in FlashInfer NHD layout.
        shape = (self.num_pages, self.page_size, self.num_heads, self.head_dim)
        self.paged_k = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.paged_v = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

        self.page_table: dict[int, list[int]] = {}
        # Stack (pop/append at end). Initialize in reverse so the first pop returns page 0.
        self.free_pages: list[int] = list(range(self.num_pages - 1, -1, -1))

        self._workspace: Optional[torch.Tensor] = None
        self._wrapper = None
        self._workspace_bytes = int(workspace_bytes)

        # Current-step metadata (set by begin_step, cleared by end_step).
        self._step_new_frame_id: Optional[int] = None
        self._step_active_frame_ids: Optional[list[int]] = None
        self._step_planned: bool = False

    def __deepcopy__(self, memo):
        """Clone cache tensors while leaving the FlashInfer wrapper unplanned."""
        cloned = self.__class__(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            tokens_per_frame=self.tokens_per_frame,
            dtype=self.dtype,
            device=self.device,
            max_frames=self.max_frames,
            workspace_bytes=self._workspace_bytes,
        )
        memo[id(self)] = cloned
        cloned.paged_k = [paged_k.clone() for paged_k in self.paged_k]
        cloned.paged_v = [paged_v.clone() for paged_v in self.paged_v]
        cloned.page_table = {
            int(frame_id): list(page_ids)
            for frame_id, page_ids in self.page_table.items()
        }
        cloned.free_pages = list(self.free_pages)
        return cloned

    # --- allocation ---------------------------------------------------------------

    def has_frame(self, frame_id: int) -> bool:
        return frame_id in self.page_table

    def allocate_frame(self, frame_id: int) -> list[int]:
        """Reserve the pages needed for a single frame. v1: exactly one page per frame."""
        if frame_id in self.page_table:
            return list(self.page_table[frame_id])
        if not self.free_pages:
            raise RuntimeError(
                f"PagedKVStore out of pages (pool size={self.num_pages}, "
                f"active={len(self.page_table)}). Increase max_frames on the retention policy."
            )
        page_id = self.free_pages.pop()
        self.page_table[frame_id] = [page_id]
        return [page_id]

    def evict_frame(self, frame_id: int) -> None:
        """Return a frame's pages to the free list and drop its page_table entry."""
        pages = self.page_table.pop(frame_id, None)
        if not pages:
            return
        for page_id in pages:
            self.free_pages.append(page_id)

    # --- writes -------------------------------------------------------------------

    def write_kv(
        self, layer_idx: int, frame_id: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Copy a frame's K/V for one layer into its reserved page(s).

        k, v shape: (B=1, num_heads, tokens_per_frame, head_dim) — same as attention.py emits.
        """
        if frame_id not in self.page_table:
            raise RuntimeError(
                f"write_kv called before allocate_frame for frame_id={frame_id}"
            )
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        if k.shape != v.shape:
            raise ValueError(
                f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}"
            )
        expected = (1, self.num_heads, self.tokens_per_frame, self.head_dim)
        if tuple(k.shape) != expected:
            raise ValueError(f"expected k shape {expected}, got {tuple(k.shape)}")

        page_ids = self.page_table[frame_id]
        if len(page_ids) != 1:
            raise RuntimeError(
                f"v1 expects exactly one page per frame; got {len(page_ids)} for frame {frame_id}"
            )
        page_id = page_ids[0]
        # (1, H, T, D) -> (T, H, D)
        k_nhd = k.squeeze(0).transpose(0, 1).contiguous()
        v_nhd = v.squeeze(0).transpose(0, 1).contiguous()
        self.paged_k[layer_idx][page_id].copy_(k_nhd)
        self.paged_v[layer_idx][page_id].copy_(v_nhd)

    def read_kv(
        self, layer_idx: int, frame_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read back a frame's K/V for unit testing. Returns (1, H, T, D) tensors."""
        if frame_id not in self.page_table:
            raise RuntimeError(f"read_kv for unknown frame_id={frame_id}")
        page_ids = self.page_table[frame_id]
        if len(page_ids) != 1:
            raise RuntimeError(f"v1 expects one page per frame; got {len(page_ids)}")
        page_id = page_ids[0]
        k_nhd = self.paged_k[layer_idx][page_id]
        v_nhd = self.paged_v[layer_idx][page_id]
        # (T, H, D) -> (1, H, T, D)
        k = k_nhd.transpose(0, 1).unsqueeze(0).contiguous()
        v = v_nhd.transpose(0, 1).unsqueeze(0).contiguous()
        return k, v

    # --- attention plan / run -----------------------------------------------------

    def _ensure_wrapper(self):
        if self._wrapper is not None:
            return self._wrapper
        flashinfer = _require_flashinfer()
        self._workspace = torch.empty(
            self._workspace_bytes, dtype=torch.uint8, device=self.device
        )
        self._wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD"
        )
        return self._wrapper

    def begin_step(self, new_frame_id: int, active_frame_ids: list[int]) -> None:
        """Allocate the new frame's page and plan the FlashInfer prefill wrapper.

        `active_frame_ids` must include `new_frame_id` (at any position) — all frames
        whose KV should be visible to this step's queries.
        """
        if new_frame_id not in active_frame_ids:
            raise ValueError(
                f"active_frame_ids {active_frame_ids} must include new_frame_id {new_frame_id}"
            )
        missing = [
            fid
            for fid in active_frame_ids
            if fid != new_frame_id and fid not in self.page_table
        ]
        if missing:
            raise RuntimeError(f"active frames not in paged cache: {missing}")

        if new_frame_id not in self.page_table:
            self.allocate_frame(new_frame_id)

        # Build flat list of page ids in frame order.
        paged_kv_indices = []
        for fid in active_frame_ids:
            paged_kv_indices.extend(self.page_table[fid])
        num_pages = len(paged_kv_indices)

        wrapper = self._ensure_wrapper()
        qo_indptr = torch.tensor(
            [0, self.tokens_per_frame], dtype=torch.int32, device=self.device
        )
        paged_kv_indptr = torch.tensor(
            [0, num_pages], dtype=torch.int32, device=self.device
        )
        paged_kv_indices_t = torch.tensor(
            paged_kv_indices, dtype=torch.int32, device=self.device
        )
        # Every page is fully filled under v1's one-page-per-frame layout.
        paged_kv_last_page_len = torch.tensor(
            [self.page_size], dtype=torch.int32, device=self.device
        )

        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices_t,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            page_size=self.page_size,
            causal=False,
            q_data_type=self.dtype,
        )

        self._step_new_frame_id = new_frame_id
        self._step_active_frame_ids = list(active_frame_ids)
        self._step_planned = True

    def run_attention(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor:
        """Run the planned prefill for one layer.

        q shape: (B=1, num_heads, tokens_per_frame, head_dim) — matches attention.py post-RoPE layout.
        Returns: (B=1, num_heads, tokens_per_frame, head_dim).
        """
        if not self._step_planned:
            raise RuntimeError("run_attention called outside begin_step / end_step")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        expected = (1, self.num_heads, self.tokens_per_frame, self.head_dim)
        if tuple(q.shape) != expected:
            raise ValueError(f"expected q shape {expected}, got {tuple(q.shape)}")

        q_nhd = q.squeeze(0).transpose(0, 1).contiguous()  # (T, H, D)
        out_nhd = self._wrapper.run(
            q_nhd, (self.paged_k[layer_idx], self.paged_v[layer_idx])
        )
        # (T, H, D) -> (1, H, T, D)
        return out_nhd.transpose(0, 1).unsqueeze(0).contiguous()

    def end_step(self) -> None:
        self._step_new_frame_id = None
        self._step_active_frame_ids = None
        self._step_planned = False

    # --- introspection ------------------------------------------------------------

    @property
    def num_free_pages(self) -> int:
        return len(self.free_pages)

    @property
    def active_frame_ids(self) -> list[int]:
        return list(self.page_table.keys())


def create_paged_kv_store(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    tokens_per_frame: int,
    dtype: torch.dtype,
    device: torch.device,
    kv_cache_mode: str,
    recent_frames: int,
    bank_initial_frames: int = 0,
    keyframe_max_keyframes: int = 0,
    flashinfer_page_size: int = 0,
) -> PagedKVStore:
    """Factory: size the pool from the retention policy and construct the store.

    `flashinfer_page_size` is accepted for API stability; v1 ignores non-zero values
    that do not equal `tokens_per_frame` and raises a clear error, so configs that
    set it (for future compatibility) still fail fast.
    """
    if flashinfer_page_size not in (0, tokens_per_frame):
        raise NotImplementedError(
            f"flashinfer_page_size={flashinfer_page_size} not supported in v1. "
            f"Use 0 (auto, one page per frame) or {tokens_per_frame} explicitly. "
            f"See agent/architecture/flashinfer_paged_kv.md."
        )
    max_frames = max_frames_for_mode(
        kv_cache_mode,
        recent_frames=recent_frames,
        bank_initial_frames=bank_initial_frames,
        keyframe_max_keyframes=keyframe_max_keyframes,
    )
    return PagedKVStore(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        tokens_per_frame=tokens_per_frame,
        dtype=dtype,
        device=device,
        max_frames=max_frames,
    )


def pages_per_frame(tokens_per_frame: int, page_size: int) -> int:
    """Number of pages a single frame occupies. v1 always returns 1 (page_size == tokens_per_frame)."""
    return math.ceil(tokens_per_frame / page_size)
