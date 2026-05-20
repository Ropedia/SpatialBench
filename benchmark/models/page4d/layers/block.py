# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings
import pdb
import torch
from torch import nn, Tensor
import math

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp
import torch.nn.functional as F
from PIL import Image
import numpy as np

XFORMERS_AVAILABLE = False

def mask_alpha(step: int, hold: int = 0, end: int = 0) -> float:
    if step < hold: return 1.0
    elif step >= end: return 0.0
    u = (step - hold) / float(end - hold)  # in (0,1)
    return 0.5 * (1.0 + math.cos(math.pi * u))

def save_to_bimask(mask, path_name):
    # mask shape: (H, W)
    mask = mask.detach().to(torch.float32)     # <-- convert from bfloat16 to float32
    H, W = mask.shape
    # Step 1: normalize to 0~1
    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    # Step 2: resize
    scale = 14
    new_H = H * scale
    new_W = W * scale
    # convert to PIL-compatible uint8 image
    mask_img = (mask_norm.cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_img)
    mask_img = mask_img.resize((new_W, new_H), Image.BILINEAR)
    # Step 3: Save
    mask_img.save(path_name)

class SpatialMaskHead_IMP(nn.Module):
    def __init__(self, d, alpha_init: float = 1.0, tau_init: float = 1.0,
                 mask_hold_start=0, mask_hold_end=0):
        super().__init__()
        self.head0 = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, groups=d), # depthwise: 保持单调性
            nn.GELU(),
            nn.Conv2d(d, 256, 1),                   # pointwise: 跨通道 mixing
            nn.GELU(),
            nn.Conv2d(256, 1, 1),)
        self.step = 0.
        self.scale = 64.
        self.mask_hold_start = mask_hold_start
        self.mask_hold_end = mask_hold_end

    def normalize(self, h0):
        mean = h0.mean(dim=(2, 3), keepdim=True)                          # per-channel, over spatial
        std = h0.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        h0_norm = (h0 - mean) / std
        return h0_norm

    def freeze_grad(self):
        for p in self.head0.parameters():
            p.requires_grad = False

    def forward(self, x, patch_start, H, W):  # x: (B,S,P,d)
        # image_x = x[0, 0, 5:, :].view(H, W, -1)
        # save_to_bimask(image_x.mean(dim=-1), '/workspace/code/12_4d/VGGT-4D_T/training_bash/print_version0.png')
        if self.training: self.step += 1.0
        alpha = mask_alpha(self.step, self.mask_hold_start, self.mask_hold_end) * self.scale
        if alpha == 0.0: self.freeze_grad()
        B, S, P, d = x.shape
        xs = x.view(B * S, P, d)[:, patch_start:, :]            # (B*S, H*W, d)
        h0 = xs.transpose(1, 2).reshape(B * S, d, H, W)
        h0_norm = self.normalize(h0)
        # save_to_bimask(h0_norm[0].mean(dim=0), '/workspace/code/12_4d/VGGT-4D_T/training_bash/print_version1.png')
        m_logit = self.head0(h0_norm)                                # (B*S,1,H,W)
        # save_to_bimask(m_logit[0,0], '/workspace/code/12_4d/VGGT-4D_T/training_bash/print_version2.png')
        m_logit = torch.sigmoid(m_logit + h0_norm.mean(dim=1, keepdim=True).detach()) - 0.5
        # save_to_bimask(m_logit[0,0], '/workspace/code/12_4d/VGGT-4D_T/training_bash/print_version3.png')
        m_logit = m_logit.view(B, S, H*W)
        key_vis = x.new_zeros(B, S, P)
        key_vis[:, :, patch_start:] = m_logit 
        key_vis = key_vis.view(B, 1, S*P)
        # save_to_bimask(key_vis.view(B, S, P)[0, 0, 5:].view(H, W), '/workspace/code/12_4d/VGGT-4D_T/training_bash/print_version4.png')
        cam_row_mask = x.new_zeros(B, S, P)
        cam_row_mask[:, :, :patch_start] = alpha
        cam_row_mask[:, :, patch_start:] = 0
        cam_row_mask = cam_row_mask.view(B, S*P)
        return cam_row_mask, key_vis.to(cam_row_mask.dtype)

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
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
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None, temporal_features = None, S=None, P=None, attn_mask=None, attn_value=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, temporal_features = None, S=None, P=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos, temporal_features=temporal_features, S=S, P=P, attn_mask=attn_mask, attn_value=attn_value))
        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, temporal_features=temporal_features, S=S, P=P, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, temporal_features=temporal_features, S=S, P=P))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos, temporal_features=temporal_features, S=S, P=P)
            x = x + ffn_residual_func(x)
        return x

def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
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
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
