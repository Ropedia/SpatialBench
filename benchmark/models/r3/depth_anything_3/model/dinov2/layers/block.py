# flake8: noqa: F821
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
from typing import Callable, Optional
import torch
from torch import Tensor, nn

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

logger = logging.getLogger("dinov2")
XFORMERS_AVAILABLE = True


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        ln_eps: float = 1e-6,
        attn_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim, eps=ln_eps)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
            **(attn_kwargs or {}),
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, eps=ln_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
        self,
        x: Tensor,
        pos=None,
        temporal_pos=None,
        attn_mask=None,
        kv_cache=None,
        flex_attn_params=None,
        paged_ctx=None,
    ) -> Tensor:
        def attn_residual_func(
            x: Tensor,
            pos=None,
            temporal_pos=None,
            attn_mask=None,
            kv_cache=None,
            flex_attn_params=None,
            paged_ctx=None,
        ) -> Tensor:
            attn_out = self.attn(
                self.norm1(x),
                pos=pos,
                temporal_pos=temporal_pos,
                attn_mask=attn_mask,
                kv_cache=kv_cache,
                flex_attn_params=flex_attn_params,
                paged_ctx=paged_ctx,
            )
            if kv_cache is not None:
                attn_out, kv_cache = attn_out
                return self.ls1(attn_out), kv_cache
            return self.ls1(attn_out)

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            assert kv_cache is None, "kv_cache is not supported with stochastic depth"
            assert paged_ctx is None, "paged_ctx is not supported with stochastic depth"
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=lambda x, pos=pos: attn_residual_func(
                    x,
                    pos=pos,
                    temporal_pos=temporal_pos,
                    attn_mask=attn_mask,
                    kv_cache=kv_cache,
                    flex_attn_params=flex_attn_params,
                    paged_ctx=paged_ctx,
                ),
                sample_drop_ratio=self.sample_drop_ratio,
                pos=pos,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            if kv_cache is not None:
                delta_x, kv_cache = attn_residual_func(
                    x,
                    pos=pos,
                    temporal_pos=temporal_pos,
                    attn_mask=attn_mask,
                    kv_cache=kv_cache,
                    flex_attn_params=flex_attn_params,
                    paged_ctx=paged_ctx,
                )
            else:
                delta_x = attn_residual_func(
                    x,
                    pos=pos,
                    temporal_pos=temporal_pos,
                    attn_mask=attn_mask,
                    kv_cache=kv_cache,
                    flex_attn_params=flex_attn_params,
                    paged_ctx=paged_ctx,
                )
            x = x + self.drop_path1(delta_x)
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            if kv_cache is not None:
                delta_x, kv_cache = attn_residual_func(
                    x,
                    pos=pos,
                    temporal_pos=temporal_pos,
                    attn_mask=attn_mask,
                    kv_cache=kv_cache,
                    flex_attn_params=flex_attn_params,
                    paged_ctx=paged_ctx,
                )
            else:
                delta_x = attn_residual_func(
                    x,
                    pos=pos,
                    temporal_pos=temporal_pos,
                    attn_mask=attn_mask,
                    kv_cache=kv_cache,
                    flex_attn_params=flex_attn_params,
                )
            x = x + delta_x
            x = x + ffn_residual_func(x)
        if kv_cache is not None:
            return x, kv_cache
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos: Optional[Tensor] = None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor
