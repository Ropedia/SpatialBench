# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from typing import Callable

from torch import Tensor, nn

from vggttt.nets.dist_attention import DistributedSDPA
from vggttt.utils.dist import get_sp_group


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        max_train_len: int | None = None,
        seq_parallel: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert fused_attn, "Fused attention is required"

        self.max_train_len = max_train_len
        self.seq_parallel = seq_parallel

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        # For forward hooks
        self.q_store = nn.Identity()
        self.k_store = nn.Identity()

        self._att = None

    @property
    def att(self) -> Callable:
        # Defer instantiation of the distributed attention layer until the first forward pass
        # when the distributed context is available.
        sp_group = get_sp_group() if self.seq_parallel else None
        if self._att is None:
            self.dist_attn = DistributedSDPA(sp_group, max_train_len=self.max_train_len)
            self._att = self.dist_attn.forward
        return self._att

    def forward(self, x: Tensor, pos=None, pos_max=None, **kwargs) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )  # B, N, 3, h, d -> (3, B, h, N, d)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos, pos_max=pos_max)
            k = self.rope(k, pos, pos_max=pos_max)

        q = self.q_store(q)
        k = self.k_store(k)

        x = self.att(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
