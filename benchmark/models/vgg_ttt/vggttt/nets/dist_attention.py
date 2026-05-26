# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Distributed scaled-dot-product attention with sequence-parallel support and entropy-invariance scaling."""

import logging
import math
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from vggttt.utils.dist import get_sp_stream
from vggttt.utils.dist_att import DistributedAttention

_logger = logging.getLogger(__name__)


def get_scale(N: int, scale: float, max_train_len: int | None = None) -> float:
    if max_train_len is None:
        return scale

    rank = os.environ.get("RANK", "0")

    # Entropy invariance scaling log_{train_len}(N)
    entropy_scaling = max(1.0, math.log(N) / math.log(max_train_len))
    _logger.debug(
        "[RANK %s] Entropy-based scaling: %s, final scale: %s, N: %s, max_train_len: %s",
        rank,
        entropy_scaling,
        entropy_scaling * scale,
        N,
        max_train_len,
    )
    return entropy_scaling * scale


class DistributedSDPA(DistributedAttention):
    """A wrapper around ``DistributedAttention`` that handles sequence parallelism when the sequence length is not a
    multiple of the sequence parallel world size.
    """

    def __init__(
        self,
        sp_group: dist.ProcessGroup | None = None,
        sp_stream=None,
        max_train_len: int | None = None,
    ) -> None:
        super().__init__(F.scaled_dot_product_attention)
        self.set_context_parallel_group(sp_group, get_sp_stream())
        self.max_train_len = max_train_len
        self.sp_group = sp_group
        self.gather_idx = 2
        self.scatter_idx = 1

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        """Forward pass.

        Args:
            query, key, value: Query, key, and value tensors of shape (B, num_heads, global_seq_length / seq_parallel_world_size, D).
        """
        N_local = query.shape[self.gather_idx]
        dim_k = key.shape[-1]
        scale = kwargs.pop("scale", dim_k**-0.5)

        if self.sp_group is None or self.sp_group.size() == 1:
            scale = get_scale(N_local, scale=scale, max_train_len=self.max_train_len)
            return F.scaled_dot_product_attention(query, key, value, scale=scale, *args, **kwargs)

        seq_parallel_world_size = self.sp_group.size()

        # Share sequence length across ranks participating in the same sequence parallelism group.
        with torch.no_grad():
            local_len = torch.tensor([N_local], device=query.device, dtype=torch.long)
            sp_lens = [torch.zeros_like(local_len) for _ in range(seq_parallel_world_size)]
            dist.all_gather(sp_lens, local_len, group=self.sp_group)
            sp_lens = [l for l in sp_lens]

        # Overwrite the scale with the true scaling based on global sequence length
        N_global = sum(sp_lens)
        scale = get_scale(N_global, scale=scale, max_train_len=self.max_train_len)

        # Make sequence length of each rank participating in the same sequence parallelism group the same.
        # Required by the underlying ``DistributedAttention`` implementation.
        pad_len = 0

        # Build attention mask for positions that are padded.
        max_len = max(sp_lens)
        att_mask = torch.ones(seq_parallel_world_size, max_len, device=query.device, dtype=torch.bool)
        for i, l in enumerate(sp_lens):
            att_mask[i, l:] = False
        att_mask = att_mask.flatten()
        att_mask = att_mask.reshape(1, 1, 1, -1)  # (S,) -> (B, H, S_q, S_k)

        pad_len = max_len - N_local
        _logger.debug(
            f"[RANK {dist.get_rank(self.sp_group)}] DistributedAttentionVarLen: Pad length: {pad_len}, local_len: {N_local}, max len: {max_len}, total_len: {N_global}, att_mask_shape: {att_mask.shape}, sp_lens: {sp_lens}"
        )

        # Pad the sequence dimension with zeros so that the local sequence length is consistent across ranks.
        if pad_len > 0:

            def _pad_to_len(t: Tensor, length: int, dim: int) -> Tensor:
                """Pad ``t`` along ``dim`` with zeros so that the size increases by
                ``length`` elements.

                Args:
                    t: Input tensor.
                    length: Number of elements to pad.
                    dim: Dimension along which to pad.

                Returns:
                    Padded tensor residing on the same device/dtype as ``t``.
                """
                if length == 0:
                    return t

                pad_shape = list(t.shape)
                pad_shape[dim] = length
                pad_tensor = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad_tensor], dim=dim)

            query = _pad_to_len(query, pad_len, self.gather_idx)
            key = _pad_to_len(key, pad_len, self.gather_idx)
            value = _pad_to_len(value, pad_len, self.gather_idx)

        out = super().forward(query, key, value, attn_mask=att_mask, scale=scale, *args, **kwargs)

        # Remove the padding that we added earlier so that the caller receives
        # a tensor that matches the original (unpadded) sequence length.
        if pad_len > 0:
            slc = [slice(None)] * out.ndim
            slc[self.gather_idx] = slice(0, N_local)
            out = out[tuple(slc)]
        return out
