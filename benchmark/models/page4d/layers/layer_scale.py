# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Fix for CUDA misaligned address error: avoid problematic in-place operations
        # The original x.mul_(self.gamma) can cause memory alignment issues on CUDA
        # Use out-of-place operations that are memory-safe
        if self.inplace:
            # Instead of in-place operation, use a safe approach that achieves similar memory efficiency
            result = x * self.gamma
            # If the tensor is contiguous and we need in-place behavior, copy back safely
            if x.is_contiguous() and result.shape == x.shape:
                x.copy_(result)
                return x
            else:
                return result
        else:
            return x * self.gamma
