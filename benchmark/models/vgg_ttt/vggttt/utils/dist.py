# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

logger = logging.getLogger(__name__)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def gather_varlen_tensor(t: torch.Tensor, dim: int = 0, group=None):
    """Gather tensors with different size in ``dim`` using padding + all_gather.

    Returns the concatenated tensor.
    """
    if group is None or dist.get_world_size(group) == 1:
        return t

    world_size_local = dist.get_world_size(group)

    # 1. Share the size of ``dim`` across ranks using no_grad for metadata.
    with torch.no_grad():
        local_len = torch.tensor([t.shape[dim]], device=t.device, dtype=torch.long)
        len_list = [torch.zeros_like(local_len) for _ in range(world_size_local)]
        dist.all_gather(len_list, local_len, group=group)
        lens = [int(l.item()) for l in len_list]
        max_len = max(lens)

    # 2. Pad to the maximum length so that shapes match.
    pad_shape = list(t.shape)
    pad_shape[dim] = max_len
    t_padded = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    slc = [slice(None)] * t.ndim
    slc[dim] = slice(0, t.shape[dim])
    t_padded[tuple(slc)] = t

    # 3. All-gather the padded tensors.
    gather_list = all_gather(t_padded, group=group)

    # 4. Unpad and concatenate.
    parts = []
    for i, g in enumerate(gather_list):
        slc_i = [slice(None)] * t.ndim
        slc_i[dim] = slice(0, lens[i])
        parts.append(g[tuple(slc_i)])
    return torch.cat(parts, dim=dim)


_SP_GROUP: dist.ProcessGroup | None = None
_SP_STREAM: torch.cuda.Stream | None = None


def init_sp_group(
    rank: int,
    world_size: int,
    sequence_parallel_size: int,
):
    if sequence_parallel_size == 1:
        return

    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    assert world_size % sequence_parallel_size == 0, (
        f"World size ({world_size}) must be divisible by sequence parallel size ({sequence_parallel_size})"
    )

    global _SP_GROUP, _SP_STREAM

    try:
        _SP_STREAM = torch.cuda.Stream()
    except RuntimeError:
        _SP_STREAM = None

    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            _SP_GROUP = group


def get_sp_group() -> dist.ProcessGroup:
    """Get the sequence parallel group the caller rank belongs to."""
    return _SP_GROUP


def get_sp_stream() -> torch.cuda.Stream:
    """Get the sequence parallel stream the caller rank belongs to."""
    return _SP_STREAM
