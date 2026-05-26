# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging

import numpy as np
import torch
from einops import asnumpy as einops_asnumpy

_logger = logging.getLogger(__name__)


def move_to_device(data: torch.Tensor | dict | list, device: torch.device) -> torch.Tensor | dict | list:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    if isinstance(data, torch.Tensor):
        # Base case: if data is a tensor, move it to the specified device
        return data.to(device)
    elif isinstance(data, list):
        # Recursive case: if data is a list, apply the function to each element
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        # Recursive case: if data is a dictionary, apply the function to each value
        return {key: move_to_device(value, device) for key, value in data.items()}

    # If the data is neither a tensor, list, nor dictionary, return it as is
    return data


def summarize_tensor(x):
    if x.numel() == 0:
        return None
    return f"\033[34m{str(tuple(x.shape)).ljust(24)}\033[0m (\033[31mmin {x.min().item():+.4f}\033[0m / \033[32mmean {x.mean().item():+.4f}\033[0m / \033[33mmax {x.max().item():+.4f}\033[0m)"


def asnumpy(value):
    """Convert supported tensor-like inputs to NumPy arrays."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, (list, tuple)):
        return np.array(value)

    if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
        value = value.to(torch.float32)
    return einops_asnumpy(value)


def compute_adaptive_minibatch_size(estimated_memory_per_sample_mb: int, memory_safety_factor: float = 0.8) -> int:
    """
    Compute adaptive minibatch size based on available PyTorch memory.

    Args:
        estimated_memory_per_sample_mb: Estimated memory per sample in MB
        memory_safety_factor: Safety factor to avoid OOM (0.95 = use 95% of available memory)

    Returns:
        Computed minibatch size
    """
    torch.cuda.empty_cache()
    available_memory = torch.cuda.mem_get_info()[0]  # Free memory in bytes
    usable_memory = available_memory * memory_safety_factor  # Use safety factor to avoid OOM

    # Determine minibatch size based on available memory
    # 680 MB per sample (upper bound profiling using a 518 x 518 input)
    max_estimated_memory_per_sample = estimated_memory_per_sample_mb * 1024 * 1024
    computed_minibatch_size = int(usable_memory / max_estimated_memory_per_sample)
    if computed_minibatch_size < 1:
        computed_minibatch_size = 1
    return computed_minibatch_size
