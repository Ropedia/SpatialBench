# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
from typing import Callable, List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from functools import partial

from depth_anything_3.utils.logger import logger

from .layers import LayerScale  # noqa: F401
from .layers import Mlp  # noqa: F401
from .layers import (  # noqa: F401
    Attention,
    Block,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding1D,
    RotaryPositionEmbedding2D,
    SwiGLUFFNFused,
)
from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION

# logger = logging.getLogger("dinov2")


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        plus_cam_token=False,
        cat_token=True,
        use_full_image_chunk=False,
        gradient_checkpointing=False,
        causal_attn=False,
        attention_mode="causal",  # "causal", "window", "window_wo_sink"
        attention_window_size=8,
        use_headwise_attn_output_gate=False,
        causal_random_frame_drop_ratio=0.0,
        use_global_frame_rope=False,
        global_frame_rope_freq=100,
        global_frame_rope_ratio=0.25,
        use_reference_token=True,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating
                positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating
                positional embeddings
            gradient_checkpointing (bool): whether to use gradient checkpointing
        """
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_full_image_chunk = use_full_image_chunk
        self.causal_attn = causal_attn
        self.attention_mode = attention_mode
        self.attention_window_size = max(int(attention_window_size), 1)
        self.use_headwise_attn_output_gate = bool(use_headwise_attn_output_gate)
        self.causal_random_frame_drop_ratio = float(causal_random_frame_drop_ratio)
        self.use_global_frame_rope = bool(use_global_frame_rope)
        self.global_frame_rope_freq = float(global_frame_rope_freq)
        self.global_frame_rope_ratio = float(global_frame_rope_ratio)
        self.use_reference_token = bool(use_reference_token)
        
        assert attention_mode in ["causal", "window", "window_wo_sink", "full"], (
            "Invalid attention mode"
        )
        if not 0.0 <= self.causal_random_frame_drop_ratio <= 1.0:
            raise ValueError("causal_random_frame_drop_ratio must be in [0, 1]")
        if self.causal_attn:
            print("Using causal attention with mode:", attention_mode)
        self.gradient_checkpointing = gradient_checkpointing
        # Cache for deterministic attention masks to avoid repeated large allocations.
        self._attn_mask_cache = {}
        self._dense_mask_warn_logged = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )
        # patch_start_idx counts cls + register tokens that precede patch tokens for RoPE position preparation
        self.patch_start_idx = 1 + num_register_tokens

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        if self.rope_start != -1:
            self.rope = (
                RotaryPositionEmbedding2D(frequency=rope_freq)
                if rope_freq > 0
                else None
            )
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
        self.temporal_rope = (
            RotaryPositionEmbedding1D(frequency=self.global_frame_rope_freq)
            if self.use_global_frame_rope
            else None
        )

        head_dim = embed_dim // num_heads
        if self.use_global_frame_rope:
            self.temporal_rope_dim = int(head_dim * self.global_frame_rope_ratio)
            self.temporal_rope_dim = self.temporal_rope_dim - (
                self.temporal_rope_dim % 2
            )
            if self.temporal_rope_dim < 2:
                raise ValueError(
                    f"temporal_rope_dim must be >= 2, got {self.temporal_rope_dim} "
                    f"(head_dim={head_dim}, ratio={self.global_frame_rope_ratio})"
                )
            spatial_dim = head_dim - self.temporal_rope_dim
            if spatial_dim % 4 != 0:
                raise ValueError(
                    f"spatial_dim must be divisible by 4 for 2D RoPE, got {spatial_dim} "
                    f"(head_dim={head_dim}, temporal_rope_dim={self.temporal_rope_dim})"
                )
        else:
            self.temporal_rope_dim = 0

        # --- BLOCK GENERATION LOOP ---
        blocks_list = []
        for i in range(depth):
            # Define if this is a global layer based on existing logic
            is_global_layer = alt_start != -1 and i >= alt_start and i % 2 == 1

            cur_attn_class = Attention

            attn_kwargs = None
            if is_global_layer:
                attn_kwargs = {
                    "headwise_attn_output_gate": self.use_headwise_attn_output_gate,
                    "temporal_rope": self.temporal_rope,
                    "temporal_rope_dim": self.temporal_rope_dim,
                }

            blk = block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
                attn_class=cur_attn_class,  # Explicitly pass the class
                attn_kwargs=attn_kwargs,
            )
            blocks_list.append(blk)

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

    def _sample_causal_dropped_frame_pairs(self, num_views: int, device: torch.device):
        """Sample per-query frame drop pairs for causal global attention during training."""
        if (
            not self.training
            or self.causal_random_frame_drop_ratio <= 0.0
            or num_views <= 1
        ):
            return None
        dropped_frame_pairs = torch.zeros(
            (num_views, num_views), device=device, dtype=torch.bool
        )
        sample_shape = (num_views, num_views)

        if self.attention_mode == "causal":
            historical_mask = torch.tril(
                torch.ones(sample_shape, device=device, dtype=torch.bool), diagonal=-1
            )
        elif self.attention_mode == "window":
            query_idx = torch.arange(num_views, device=device)[:, None]
            key_idx = torch.arange(num_views, device=device)[None, :]
            historical_mask = key_idx < query_idx
            window_mask = key_idx >= (query_idx - self.attention_window_size + 1)
            sink_mask = key_idx == 0
            historical_mask = historical_mask & (window_mask | sink_mask)
        elif self.attention_mode == "window_wo_sink":
            query_idx = torch.arange(num_views, device=device)[:, None]
            key_idx = torch.arange(num_views, device=device)[None, :]
            historical_mask = key_idx < query_idx
            window_mask = key_idx >= (query_idx - self.attention_window_size + 1)
            historical_mask = historical_mask & window_mask
        else:
            return None

        random_mask = (
            torch.rand(sample_shape, device=device)
            < self.causal_random_frame_drop_ratio
        )
        dropped_frame_pairs[historical_mask] = random_mask[historical_mask]
        # Never drop frame 0 — preserve it as a soft coordinate anchor even
        # without a reference token, so later frames always see the origin.
        dropped_frame_pairs[:, 0] = False
        return dropped_frame_pairs

    def _create_attn_mask(
        self,
        S: int,
        P: int,
        mode: str,
        dtype: torch.dtype,
        device: torch.device,
        dropped_frame_pairs=None,
    ) -> torch.Tensor:
        """Build an additive float attention mask for the given sequence shape and mode.

        Deterministic masks (no random frame drops) are cached on CPU to avoid
        rebuilding every forward pass. Cached masks are moved to the target device
        on each call — CPU memory is cheap, GPU memory is not.
        """
        # Cache key ignores device — masks are stored on CPU and moved on demand.
        cache_key = (S, P, mode, dtype) if dropped_frame_pairs is None else None
        if cache_key is not None and cache_key in self._attn_mask_cache:
            return self._attn_mask_cache[cache_key].to(device, non_blocking=True)

        N = S * P
        if N > 16384 and not self._dense_mask_warn_logged:
            logger.warning(
                f"Creating dense attention mask ({N}, {N}) = "
                f"{N * N * 2 / 1024**2:.0f} MB bf16. "
                f"Consider enabling flex_attn or reducing max_sequence_length."
            )
            self._dense_mask_warn_logged = True
        dropped_frame_pairs_tensor = None
        if dropped_frame_pairs is not None:
            dropped_frame_pairs_tensor = torch.as_tensor(
                dropped_frame_pairs, device=device, dtype=torch.bool
            )
            if dropped_frame_pairs_tensor.shape != (S, S):
                raise ValueError(
                    f"dropped_frame_pairs must have shape {(S, S)}, got {tuple(dropped_frame_pairs_tensor.shape)}"
                )

        if mode == "causal":
            mask = torch.zeros((N, N), dtype=dtype, device=device)
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P
                mask[curr_view_start:curr_view_end, curr_view_end:] = float("-inf")
                if dropped_frame_pairs_tensor is not None:
                    row_drop = dropped_frame_pairs_tensor[i].repeat_interleave(P)
                    row_drop[curr_view_start:curr_view_end] = False
                    mask[curr_view_start:curr_view_end, row_drop] = float("-inf")
        elif mode == "window":
            window_size = self.attention_window_size
            mask = torch.zeros((N, N), dtype=dtype, device=device)
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P
                mask[curr_view_start:curr_view_end, P:] = float("-inf")
                start_view = max(1, i - window_size + 1)
                mask[curr_view_start:curr_view_end, start_view * P : (i + 1) * P] = 0
                if dropped_frame_pairs_tensor is not None:
                    row_drop = dropped_frame_pairs_tensor[i].repeat_interleave(P)
                    row_drop[curr_view_start:curr_view_end] = False
                    mask[curr_view_start:curr_view_end, row_drop] = float("-inf")
        elif mode == "window_wo_sink":
            window_size = self.attention_window_size
            mask = torch.zeros((N, N), dtype=dtype, device=device)
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P
                mask[curr_view_start:curr_view_end, :] = float("-inf")
                start_view = max(0, i - window_size + 1)
                mask[curr_view_start:curr_view_end, start_view * P : curr_view_end] = 0
                if dropped_frame_pairs_tensor is not None:
                    row_drop = dropped_frame_pairs_tensor[i].repeat_interleave(P)
                    row_drop[curr_view_start:curr_view_end] = False
                    mask[curr_view_start:curr_view_end, row_drop] = float("-inf")
        elif mode == "full":
            mask = None
        else:
            raise NotImplementedError(f"Unknown attention mode: {mode}")

        if cache_key is not None and mask is not None:
            self._attn_mask_cache[cache_key] = mask.cpu()
        return mask

    def _get_flex_attn_params(
        self,
        S: int,
        P: int,
        attn_type: str,
        has_external_attn_mask: bool,
        dropped_frame_pairs=None,
    ):
        # Fall back to dense SDPA masks during training ONLY when random frame
        # drops are active (non-deterministic mask that cannot be LRU-cached).
        # When drops are inactive (dropped_frame_pairs is None), flex_attn is
        # safe: create_block_mask_cached caches by shape and the number of
        # unique shapes is bounded by #resolutions × #sequence_lengths (~105).
        if self.training and dropped_frame_pairs is not None:
            return None
        if attn_type != "global" or not self.causal_attn or has_external_attn_mask:
            return None

        if self.attention_mode == "causal":
            window_size = S
            sink_window = False
        elif self.attention_mode == "window":
            window_size = self.attention_window_size
            sink_window = True
        elif self.attention_mode == "window_wo_sink":
            window_size = self.attention_window_size
            sink_window = False
        else:
            return None

        return {
            "use_flex_attn": True,
            "block_size": P,
            "look_forward": 0,
            "look_backward": max(window_size - 1, 0),
            "sink_window": sink_window,
            "dropped_frame_pairs": dropped_frame_pairs,
        }

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the
            # interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using
            # both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_cls_token(self, B, S):
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )
        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = (
                    torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                )
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _prepare_frame_ids(self, batch_size, num_views, device, frame_ids=None):
        """Builds a batch-aligned tensor of frame ids for temporal attention."""
        if frame_ids is None:
            return (
                torch.arange(num_views, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        frame_ids = torch.as_tensor(frame_ids, device=device, dtype=torch.long)
        if frame_ids.ndim == 1:
            if frame_ids.shape[0] != num_views:
                raise ValueError(
                    f"frame_ids must have length {num_views}, got {frame_ids.shape[0]}"
                )
            return frame_ids.unsqueeze(0).expand(batch_size, -1)
        if frame_ids.ndim == 2:
            if frame_ids.shape != (batch_size, num_views):
                raise ValueError(
                    f"frame_ids must have shape {(batch_size, num_views)}, got {tuple(frame_ids.shape)}"
                )
            return frame_ids
        raise ValueError("frame_ids must be a 1D or 2D tensor-like input")

    def _prepare_temporal_positions(self, frame_ids, tokens_per_view):
        """Repeats per-frame ids across all tokens in each frame for global attention."""
        if frame_ids is None:
            return None
        return frame_ids[:, :, None].expand(-1, -1, tokens_per_view)

    def _get_intermediate_layers_not_chunked(
        self, x, n=1, export_feat_layers=[], **kwargs
    ):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        return_kv_cache = bool(kwargs.get("return_kv_cache", False))
        return_memory_select_feat = bool(kwargs.get("return_memory_select_feat", False))
        paged_kv_store = kwargs.get("paged_kv_store")
        paged_new_frame_id = kwargs.get("paged_new_frame_id")
        paged_active_frame_ids = kwargs.get("paged_active_frame_ids")
        if paged_kv_store is not None:
            if x.shape[1] != 1:
                raise NotImplementedError(
                    f"paged_kv_store requires exactly one new frame per forward (got S={x.shape[1]})"
                )
            if paged_new_frame_id is None or paged_active_frame_ids is None:
                raise ValueError(
                    "paged_kv_store requires paged_new_frame_id and paged_active_frame_ids kwargs"
                )
            paged_kv_store.begin_step(
                new_frame_id=int(paged_new_frame_id),
                active_frame_ids=[int(fid) for fid in paged_active_frame_ids],
            )
        kv_cache_list = kwargs.get("kv_cache_list")
        if return_kv_cache and kv_cache_list is None and paged_kv_store is None:
            kv_cache_list = [[None, None] for _ in range(total_block_len)]
        next_kv_cache_list = list(kv_cache_list) if kv_cache_list is not None else None
        paged_global_layer_idx = 0
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
        frame_ids = self._prepare_frame_ids(
            B, S, x.device, frame_ids=kwargs.get("frame_ids")
        )
        temporal_pos = (
            self._prepare_temporal_positions(frame_ids, x.shape[2])
            if self.use_global_frame_rope
            else None
        )
        memory_select_feat = None

        dropped_frame_pairs = None
        if self.causal_attn and not return_kv_cache:
            dropped_frame_pairs = self._sample_causal_dropped_frame_pairs(S, x.device)

        # Defer dense mask creation — flex_attn handles masking via block masks,
        # so we only allocate the O(N^2) dense mask when actually needed as fallback.
        _causal_mask_memo = [None]
        _mask_S, _mask_P = S, x.shape[2]
        _mask_dtype, _mask_device = x.dtype, x.device

        def _get_causal_mask():
            """Lazily create the dense attention mask. Returns None for non-causal modes."""
            if _causal_mask_memo[0] is None and self.causal_attn:
                _causal_mask_memo[0] = self._create_attn_mask(
                    _mask_S,
                    _mask_P,
                    self.attention_mode,
                    _mask_dtype,
                    _mask_device,
                    dropped_frame_pairs=dropped_frame_pairs,
                )
            return _causal_mask_memo[0]

        for i, blk in enumerate(self.blocks):
            if hasattr(blk, "attn"):
                if hasattr(blk.attn, "_runtime_patch_hw"):
                    blk.attn._runtime_patch_hw = (
                        H // self.patch_size,
                        W // self.patch_size,
                    )
                if hasattr(blk.attn, "_runtime_num_special_tokens"):
                    blk.attn._runtime_num_special_tokens = 1 + self.num_register_tokens

            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            # if (
            #     self.alt_start != -1
            #     and (i == self.alt_start - 1)
            #     and x.shape[1] >= THRESH_FOR_REF_SELECTION
            #     and not self.training
            # ):
            #     # Select reference view using configured strategy
            #     strategy = kwargs.get("ref_view_strategy", "first")
            #     logger.info(f"Selecting reference view using strategy: {strategy}")
            #     b_idx = select_reference_view(x, strategy=strategy)
            #     # Reorder views to place reference view first
            #     x = reorder_by_reference(x, b_idx)
            #     local_x = reorder_by_reference(local_x, b_idx)
            if (
                return_memory_select_feat
                and memory_select_feat is None
                and self.alt_start != -1
                and (i == self.alt_start - 1)
            ):
                # if x.shape[2] > self.patch_start_idx:
                #     memory_select_feat = x[:, :, self.patch_start_idx :, :]
                # else:
                memory_select_feat = (
                    x[:, :, self.patch_start_idx :, :].detach().mean(dim=2)
                )  # B S C
                # print("x shape for memory select feature:", x.shape)
                # print(f"Memory select feature shape: {memory_select_feat.shape}")
                # print("Current block index for memory select feature:", i)
            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    # logger.info("Using camera conditions provided by the user")
                    cam_token = kwargs.get("cam_token")
                else:
                    camera_token_is_reference = kwargs.get(
                        "camera_token_is_reference", None
                    )
                    if not self.use_reference_token:
                        # All frames use source token (index 1).
                        cam_token = self.camera_token[:, 1:].expand(B, S, -1)
                    elif S == 1 and camera_token_is_reference is not None:
                        token_idx = 0 if camera_token_is_reference else 1
                        cam_token = self.camera_token[
                            :, token_idx : token_idx + 1
                        ].expand(B, -1, -1)
                    else:
                        ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                        src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                        cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token
                # if True:
                #     if kwargs.get("cam_token", None) is not None:
                #         # logger.info("Using camera conditions provided by the user")
                #         cam_token = kwargs.get("cam_token")
                #     else:
                #         if kwargs.get("relative_cam", False):
                #             # Relative-camera mode: use source token for every frame.
                #             # exit(0)
                #             cam_token = self.camera_token[:, 1:].expand(B, S, -1)
                #         else:
                #             # Absolute-camera mode: first frame gets ref token, others get source token.
                #             ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                #             src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                #             cam_token = torch.cat([ref_token, src_token], dim=1)
                #     x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                curr_attn_type = "global"
                use_paged = paged_kv_store is not None and self.causal_attn
                use_kv_cache = (
                    next_kv_cache_list is not None
                    and self.causal_attn
                    and not use_paged
                )
                flex_attn_params = self._get_flex_attn_params(
                    S=S,
                    P=x.shape[2],
                    attn_type=curr_attn_type,
                    has_external_attn_mask=("attn_mask" in kwargs),
                    dropped_frame_pairs=dropped_frame_pairs,
                )
                # Only create the dense O(N²) mask when flex_attn is unavailable.
                # When flex_attn is active, masking is handled via block masks.
                if use_kv_cache or use_paged or flex_attn_params is not None:
                    current_attn_mask = kwargs.get("attn_mask", None)
                else:
                    current_attn_mask = kwargs.get("attn_mask", _get_causal_mask())
                block_kv_cache = next_kv_cache_list[i] if use_kv_cache else None
                block_paged_ctx = (
                    (paged_kv_store, paged_global_layer_idx) if use_paged else None
                )
                block_output = self.process_attention(
                    x,
                    blk,
                    curr_attn_type,
                    pos=g_pos,
                    temporal_pos=temporal_pos,
                    attn_mask=current_attn_mask,
                    kv_cache=block_kv_cache,
                    flex_attn_params=flex_attn_params,
                    paged_ctx=block_paged_ctx,
                )
                if use_kv_cache:
                    x, next_kv_cache_list[i] = block_output
                else:
                    x = block_output
                if use_paged:
                    paged_global_layer_idx += 1
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                # Restore original view order if reordering was applied
                if (
                    x.shape[1] >= THRESH_FOR_REF_SELECTION
                    and self.alt_start != -1
                    and "b_idx" in locals()
                ):
                    pass
                    # out_x = restore_original_order(out_x, b_idx) #FIXME
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)

        if paged_kv_store is not None:
            paged_kv_store.end_step()

        if return_kv_cache:
            if return_memory_select_feat:
                return output, aux_output, next_kv_cache_list, memory_select_feat
            return output, aux_output, next_kv_cache_list
        if return_memory_select_feat:
            return output, aux_output, memory_select_feat
        return output, aux_output

    def process_attention(
        self,
        x,
        block,
        attn_type="global",
        pos=None,
        temporal_pos=None,
        attn_mask=None,
        kv_cache=None,
        flex_attn_params=None,
        paged_ctx=None,
    ):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
            flex_attn_params = None
        elif attn_type == "global":
            # Global attention within the same batch sample
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
            if temporal_pos is not None:
                temporal_pos = rearrange(temporal_pos, "b s n -> b (s n)")
        elif attn_type == "linear":
            attn_mask = None
            flex_attn_params = None
            temporal_pos = None
            paged_ctx = None
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        if kv_cache is not None or paged_ctx is not None:
            flex_attn_params = None

        if self.gradient_checkpointing and self.training:
            assert kv_cache is None, (
                "kv_cache is not supported with gradient checkpointing"
            )
            assert paged_ctx is None, (
                "paged_ctx is not supported with gradient checkpointing"
            )
            x = torch.utils.checkpoint.checkpoint(
                block,
                x,
                pos,
                temporal_pos,
                attn_mask,
                kv_cache,
                flex_attn_params,
                use_reentrant=False,
            )
        else:
            x = block(
                x,
                pos=pos,
                temporal_pos=temporal_pos,
                attn_mask=attn_mask,
                kv_cache=kv_cache,
                flex_attn_params=flex_attn_params,
                paged_ctx=paged_ctx,
            )

        next_kv_cache = None
        if kv_cache is not None:
            x, next_kv_cache = x

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        elif attn_type == "linear":
            pass

        if kv_cache is not None:
            return x, next_kv_cache
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        export_feat_layers: List[int] = [],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        result = self._get_intermediate_layers_not_chunked(
            x, n, export_feat_layers=export_feat_layers, **kwargs
        )
        return_kv_cache = kwargs.get("return_kv_cache", False)
        return_memory_select_feat = kwargs.get("return_memory_select_feat", False)
        if return_kv_cache and return_memory_select_feat:
            outputs, aux_outputs, kv_cache_list, memory_select_feat = result
        elif return_kv_cache:
            outputs, aux_outputs, kv_cache_list = result
        elif return_memory_select_feat:
            outputs, aux_outputs, memory_select_feat = result
        else:
            outputs, aux_outputs = result
        camera_tokens = [out[0] for out in outputs]
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [
                        out[1][..., : self.embed_dim],
                        self.norm(out[1][..., self.embed_dim :]),
                    ],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        aux_outputs = [self.norm(out) for out in aux_outputs]
        outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
        aux_outputs = [
            torch.cat(
                (out[..., :1, :], out[..., 1 + self.num_register_tokens :, :]), dim=-2
            )
            for out in aux_outputs
        ]
        outputs = (tuple(zip(outputs, camera_tokens)), aux_outputs)
        if return_kv_cache and return_memory_select_feat:
            return outputs[0], outputs[1], kv_cache_list, memory_select_feat
        if return_kv_cache:
            return outputs[0], outputs[1], kv_cache_list
        if return_memory_select_feat:
            return outputs[0], outputs[1], memory_select_feat
        return outputs


def vit_small(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, depth=40, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=depth,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
