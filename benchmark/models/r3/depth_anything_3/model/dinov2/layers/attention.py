# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from .flex_attn import (
        create_block_mask_compiled,
        create_block_mask_cached,
        custom_mask_mod_with_params,
        flex_attention_compiled,
    )

    FLEX_ATTENTION_AVAILABLE = True
except Exception:
    FLEX_ATTENTION_AVAILABLE = False

logger = logging.getLogger("dinov2")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        headwise_attn_output_gate: bool = False,
        temporal_rope=None,
        temporal_rope_dim: int = 0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = fused_attn
        self.headwise_attn_output_gate = bool(headwise_attn_output_gate)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.gate_proj = None
        if self.headwise_attn_output_gate:
            self.gate_proj = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 4.0)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.temporal_rope = temporal_rope
        self.temporal_rope_dim = int(temporal_rope_dim)
        self._flex_mask_mod_cache = {}
        self._flex_warning_logged = False

    @staticmethod
    def _split_spatial_temporal_rope_dims(
        tensor: Tensor, temporal_dim: int
    ) -> tuple[Tensor, Tensor]:
        """Split each spatial axis symmetrically, reserving axis tails for temporal RoPE."""
        head_dim = tensor.shape[-1]
        half_dim = head_dim // 2
        temporal_per_axis = temporal_dim // 2
        spatial_per_axis = half_dim - temporal_per_axis
        first_axis, second_axis = tensor[..., :half_dim], tensor[..., half_dim:]
        spatial = torch.cat(
            [first_axis[..., :spatial_per_axis], second_axis[..., :spatial_per_axis]],
            dim=-1,
        )
        temporal = torch.cat(
            [first_axis[..., spatial_per_axis:], second_axis[..., spatial_per_axis:]],
            dim=-1,
        )
        return spatial, temporal

    @staticmethod
    def _merge_spatial_temporal_rope_dims(spatial: Tensor, temporal: Tensor) -> Tensor:
        spatial_half = spatial.shape[-1] // 2
        temporal_half = temporal.shape[-1] // 2
        first_axis = torch.cat(
            [spatial[..., :spatial_half], temporal[..., :temporal_half]], dim=-1
        )
        second_axis = torch.cat(
            [spatial[..., spatial_half:], temporal[..., temporal_half:]], dim=-1
        )
        return torch.cat([first_axis, second_axis], dim=-1)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not self.headwise_attn_output_gate:
            return

        gate_keys = [f"{prefix}gate_proj.weight", f"{prefix}gate_proj.bias"]
        for gate_key in gate_keys:
            if gate_key not in state_dict and gate_key in missing_keys:
                missing_keys.remove(gate_key)

    def _get_flex_mask_mod(
        self,
        block_size: int,
        look_forward: int,
        look_backward: int,
        sink_window: bool,
        dropped_frame_pairs=None,
    ):
        if dropped_frame_pairs is not None:
            return custom_mask_mod_with_params(
                block_size=block_size,
                look_forward=look_forward,
                look_backward=look_backward,
                sink_window=sink_window,
                dropped_frame_pairs=dropped_frame_pairs,
            )

        key = (block_size, look_forward, look_backward, sink_window)
        if key not in self._flex_mask_mod_cache:
            self._flex_mask_mod_cache[key] = custom_mask_mod_with_params(
                block_size=block_size,
                look_forward=look_forward,
                look_backward=look_backward,
                sink_window=sink_window,
            )
        return self._flex_mask_mod_cache[key]

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
        if paged_ctx is not None and kv_cache is not None:
            raise RuntimeError(
                "paged_ctx and dense kv_cache cannot be combined in the same Attention.forward call"
            )
        B, N, C = x.shape
        qkv = self.qkv(x)
        gate_scores = None
        if self.headwise_attn_output_gate:
            gate_scores = torch.sigmoid(self.gate_proj(x)).view(B, N, self.num_heads, 1)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        has_spatial = self.rope is not None and pos is not None
        has_temporal = (
            self.temporal_rope is not None
            and temporal_pos is not None
            and self.temporal_rope_dim > 0
        )

        if has_spatial and has_temporal:
            d = self.temporal_rope_dim
            q_s, q_t = self._split_spatial_temporal_rope_dims(q, d)
            k_s, k_t = self._split_spatial_temporal_rope_dims(k, d)
            q_s = self.rope(q_s, pos)
            k_s = self.rope(k_s, pos)
            q_t = self.temporal_rope(q_t, temporal_pos)
            k_t = self.temporal_rope(k_t, temporal_pos)
            q = self._merge_spatial_temporal_rope_dims(q_s, q_t)
            k = self._merge_spatial_temporal_rope_dims(k_s, k_t)
        elif has_spatial:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if paged_ctx is not None:
            paged_kv_store, paged_layer_idx = paged_ctx
            q = q.to(paged_kv_store.dtype)
            k = k.to(paged_kv_store.dtype)
            v = v.to(paged_kv_store.dtype)
            paged_kv_store.write_kv(
                paged_layer_idx, paged_kv_store._step_new_frame_id, k, v
            )
            x = paged_kv_store.run_attention(paged_layer_idx, q)
            x = x.transpose(1, 2)
            if gate_scores is not None:
                x = x * gate_scores
            x = x.reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        if kv_cache is not None:
            if len(kv_cache) == 3:
                k_buf, v_buf, cache_len = kv_cache
                n_new = k.shape[2]

                # Dynamic resize in case we exceed pre-allocated buffer
                if cache_len + n_new > k_buf.shape[2]:
                    new_size = max((cache_len + n_new) * 2, k_buf.shape[2] * 2)
                    new_k_buf = torch.zeros(
                        (k_buf.shape[0], k_buf.shape[1], new_size, k_buf.shape[3]),
                        dtype=k_buf.dtype,
                        device=k_buf.device,
                    )
                    new_v_buf = torch.zeros(
                        (v_buf.shape[0], v_buf.shape[1], new_size, v_buf.shape[3]),
                        dtype=v_buf.dtype,
                        device=v_buf.device,
                    )
                    new_k_buf[:, :, :cache_len] = k_buf[:, :, :cache_len]
                    new_v_buf[:, :, :cache_len] = v_buf[:, :, :cache_len]
                    k_buf = new_k_buf
                    v_buf = new_v_buf

                k_buf[:, :, cache_len : cache_len + n_new] = k
                v_buf[:, :, cache_len : cache_len + n_new] = v
                k = k_buf[:, :, : cache_len + n_new]
                v = v_buf[:, :, : cache_len + n_new]
                kv_cache = [k_buf, v_buf, cache_len + n_new]
            else:
                k_cache, v_cache = kv_cache
                if k_cache is not None and v_cache is not None:
                    k = torch.cat([k_cache, k], dim=2)
                    v = torch.cat([v_cache, v], dim=2)
                kv_cache = [k, v]

        use_flex_attn = (
            flex_attn_params is not None
            and flex_attn_params.get("use_flex_attn", False)
            and kv_cache is None
        )
        if use_flex_attn and FLEX_ATTENTION_AVAILABLE and q.is_cuda:
            try:
                block_size = int(flex_attn_params["block_size"])
                look_forward = int(flex_attn_params.get("look_forward", 0))
                look_backward = int(flex_attn_params["look_backward"])
                sink_window = bool(flex_attn_params.get("sink_window", False))
                dropped_frame_pairs = flex_attn_params.get("dropped_frame_pairs")

                mask_mod = self._get_flex_mask_mod(
                    block_size=block_size,
                    look_forward=look_forward,
                    look_backward=look_backward,
                    sink_window=sink_window,
                    dropped_frame_pairs=dropped_frame_pairs,
                )
                if dropped_frame_pairs is None:
                    block_mask = create_block_mask_cached(
                        mask_mod=mask_mod,
                        B=1,
                        H=1,
                        M=N,
                        N=N,
                        device=x.device,
                    )
                else:
                    block_mask = create_block_mask_compiled(
                        mask_mod,
                        1,
                        1,
                        N,
                        N,
                        device=x.device,
                    )
                common_dtype = torch.promote_types(
                    q.dtype, torch.promote_types(k.dtype, v.dtype)
                )
                q_flex = q if q.dtype == common_dtype else q.to(common_dtype)
                k_flex = k if k.dtype == common_dtype else k.to(common_dtype)
                v_flex = v if v.dtype == common_dtype else v.to(common_dtype)
                x = flex_attention_compiled(
                    q_flex, k_flex, v_flex, block_mask=block_mask
                )
            except Exception as e:
                if not self._flex_warning_logged:
                    logger.warning(
                        f"Falling back to SDPA attention because flex attention failed: {e}"
                    )
                    self._flex_warning_logged = True
                use_flex_attn = False
        else:
            if use_flex_attn and not self._flex_warning_logged:
                if not FLEX_ATTENTION_AVAILABLE:
                    logger.warning(
                        "Falling back to SDPA attention because flex attention is unavailable."
                    )
                elif not q.is_cuda:
                    logger.warning(
                        "Falling back to SDPA attention because flex attention requires CUDA tensors."
                    )
                self._flex_warning_logged = True
            use_flex_attn = False

        if not use_flex_attn and self.fused_attn:
            if (
                attn_mask is not None
                and attn_mask.dim() == 2
                and attn_mask.shape[1] == N
                and attn_mask.shape[0] == N
            ):
                pass
            elif attn_mask is not None:
                attn_mask = attn_mask[:, None].repeat(1, self.num_heads, 1, 1)

            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        elif not use_flex_attn:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2)
        if gate_scores is not None:
            x = x * gate_scores
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if kv_cache is not None:
            return x, kv_cache
        return x

    def _forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        gate_scores = None
        if self.headwise_attn_output_gate:
            gate_scores = torch.sigmoid(self.gate_proj(x)).view(B, N, self.num_heads, 1)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if gate_scores is not None:
            x = x * gate_scores
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
