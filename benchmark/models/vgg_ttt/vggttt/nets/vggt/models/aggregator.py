# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging
from functools import partial
from typing import Any, List, Tuple

import torch
from torch import nn

from vggttt.nets.vggt.layers import PatchEmbed
from vggttt.nets.vggt.layers.attention import Attention
from vggttt.nets.vggt.layers.block import Block
from vggttt.nets.vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from vggttt.utils.dist import get_sp_group
from vggttt.utils.optim import checkpoint

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        gradient_checkpoint: bool = False,
        ttt_query_images: int = 0,
        global_attn_class: type[nn.Module] = Attention,
        permutation_invariant: bool = False,
    ):
        super().__init__()
        assert aa_block_size == 1
        self.gradient_checkpoint = gradient_checkpoint
        self.ttt_query_images = ttt_query_images
        self.permutation_invariant = permutation_invariant

        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    attn_class=partial(global_attn_class, seq_parallel=True),
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            from vggttt.nets.vggt.layers import vision_transformer as vit

            vit_models = {
                "dinov2_vitl14_reg": vit.vit_large,
                "dinov2_vitb14_reg": vit.vit_base,
                "dinov2_vits14_reg": vit.vit_small,
                "dinov2_vitg2_reg": vit.vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
                use_checkpoint=self.gradient_checkpoint,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                del self.patch_embed.mask_token

    def forward(
        self,
        images: torch.Tensor,
        intermediate_layers_to_return: tuple[int, ...] = (4, 11, 17, 23),
        ttt_end: int | None = None,
        attn_kwargs: dict[str, Any] = {},
        add_first_view_token: bool = True,
        memory_efficient_inference: bool = False,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        sp_group = get_sp_group()

        # Expand camera and register tokens to match batch size and sequence length
        if self.permutation_invariant:
            add_first_view_token = False
        else:
            if sp_group is not None and sp_group.rank() != 0:
                add_first_view_token = False

        camera_token = slice_expand_and_flatten(self.camera_token, B, S, add_first_view_token)
        register_token = slice_expand_and_flatten(self.register_token, B, S, add_first_view_token)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        pos_max = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            pos_max = max(H // self.patch_size, W // self.patch_size) + 1

            if self.patch_start_idx > 0:
                # do not use position embedding for special tokens (camera and register tokens)
                # so set pos to 0 for the special tokens
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for aa_block_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens,
                        B,
                        S,
                        P,
                        C,
                        frame_idx,
                        pos=pos,
                        pos_max=pos_max,
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens,
                        B,
                        S,
                        P,
                        C,
                        global_idx,
                        pos=pos,
                        pos_max=pos_max,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        num_prefix_tokens=self.patch_start_idx,
                        **attn_kwargs,
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            block_starting_layer = aa_block_idx * self.aa_block_size
            for i in range(len(frame_intermediates)):
                # Only store the intermediates we need for downstream layers (reducing inference memory usage)
                if block_starting_layer + i not in intermediate_layers_to_return:
                    continue

                # concat frame and global intermediates, [B x S x P x 2C]
                assert len(frame_intermediates) == 1 and len(global_intermediates) == 1
                concat_inter = torch.cat([frame_intermediates[0], global_intermediates[0]], dim=-1)
                output_list.append(concat_inter if not memory_efficient_inference else concat_inter.cpu())

        return output_list, self.patch_start_idx, pos

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, pos_max=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        use_grad_checkpoint = self.gradient_checkpoint and self.training
        for _ in range(self.aa_block_size):
            block = self.frame_blocks[frame_idx]
            tokens = checkpoint(partial(block, pos=pos, pos_max=pos_max), tokens, enabled=use_grad_checkpoint)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, **kwargs):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        use_grad_checkpoint = self.gradient_checkpoint and self.training
        for _ in range(self.aa_block_size):
            block = self.global_blocks[global_idx]
            tokens = checkpoint(partial(block, pos=pos, **kwargs), tokens, enabled=use_grad_checkpoint)

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S, add_first_view_token: bool = True):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    import os

    rank = os.environ.get("RANK", 0)
    if add_first_view_token:
        logger.debug("[RANK %s] Assigning canonical image token.", rank)
        # Slice out the "query" tokens => shape (1, 1, ...)
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        # Slice out the "other" tokens => shape (1, S-1, ...)
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        # Concatenate => shape (B, S, ...)
        combined = torch.cat([query, others], dim=1)
    else:
        logger.debug("[RANK %s] NOT USING canonical image token.", rank)
        # If it is not the first rank in sequence parallel, we need to use only the "other" tokens
        combined = token_tensor[:, 1:, ...].expand(B, S, *token_tensor.shape[2:])

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
