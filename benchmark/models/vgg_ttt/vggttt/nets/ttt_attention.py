# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Fast weight attention layer for test-time training."""

import logging
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vggttt.data.utils import move_to_device
from vggttt.utils.dist import get_sp_group

from .ttt import TTTOperator, apply_chunked, fast_weight_swish_glu_weight_norm_mini_batch_apply, inv_softplus

_logger = logging.getLogger(__name__)


class ShortConv(nn.Module):
    """Wrapper around nn.Conv2d for short convs and channels last."""

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(dim, dim, groups=dim, kernel_size=kernel_size, padding=padding, stride=1, bias=False)
        self.conf = self.conv.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor, patch_h: int, patch_w: int, num_suffix_tokens: int, num_prefix_tokens: int):
        """Forward pass.

        Args:
            x: Input tensor [b, num_heads, tokens, d] where tokens is structured as
                num_images * [num_prefix_tokens + (patch_h * patch_w)] + num_suffix_tokens.
            patch_h: Patch height.
            patch_w: Patch width.
            num_suffix_tokens: Number of tokens that are appended to the FULL sequence.
            num_prefix_tokens: Number of tokens that are prepended to each image.
        """
        b, num_heads, num_tokens, d = x.shape
        num_img_tokens = patch_h * patch_w

        # Suffix tokens are per-sequence
        x, suffix = torch.split(x, [num_tokens - num_suffix_tokens, num_suffix_tokens], dim=2)

        # Prefix tokens are per-image
        num_per_img_tokens = num_img_tokens + num_prefix_tokens
        x = rearrange(x, "b heads (n t) d -> b heads n t d", t=num_per_img_tokens)
        prefix, x = torch.split(x, [num_prefix_tokens, num_img_tokens], dim=3)
        x = rearrange(x, "b heads n (h w) d -> (b n) (heads d) h w", h=patch_h, w=patch_w)

        x = self.conv(x.to(memory_format=torch.channels_last))

        x = rearrange(x, "(b n) (heads d) h w -> b heads n (h w) d", b=b, heads=num_heads, d=d)
        x = torch.cat([prefix, x], dim=3)
        x = rearrange(x, "b heads n t d -> b heads (n t) d")
        x = torch.cat([x, suffix], dim=2)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class FastWeightAttention(nn.Module):
    """Adaptation of Attention layer in VGGT for loading weights but using TTT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
        mlp_ratio: int = 1,
        base_lr: float = 0.01,
        muon_update_steps: int = 4,
        seq_parallel: bool = False,
        short_conv_size_qkv: tuple[int, int, int] = (0, 0, 0),
        norm_ttt_grad: bool = False,
        div_lr_by_seq_len: bool = False,
        num_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.muon_update_steps = muon_update_steps
        self.seq_parallel = seq_parallel
        self.norm_ttt_grad = norm_ttt_grad
        self.div_lr_by_seq_len = div_lr_by_seq_len
        self.num_steps = num_steps

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.using_short_conv_qkv = any(short_conv_size_qkv)
        self.short_conv_qkv = nn.ModuleList()
        for kernel_size in short_conv_size_qkv:
            if kernel_size > 0:
                self.short_conv_qkv.append(ShortConv(dim, kernel_size))
            else:
                self.short_conv_qkv.append(Identity())

        #########################################################
        #  Fast weights
        #########################################################
        d_in = d_out = self.head_dim
        d_h = int(self.head_dim * mlp_ratio)

        gain = math.sqrt(2)  # for relu activations
        self.register_buffer("w0", torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in))
        self.register_buffer("w1", torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h))
        self.register_buffer("w2", torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in))

        self.lr_dim = self.num_heads
        self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)

        self.state: None | dict[str, torch.Tensor] = None
        self.state_tracking = False

        self.logging_hook = nn.Identity()
        self.q_hook = nn.Identity()
        self.k_hook = nn.Identity()

    def set_state_tracking(self, enable: bool):
        self.state_tracking = enable
        if not enable:
            self.state = None

    def reset_state(self):
        self.state = None

    def _compute_qkv_lr_impl(
        self,
        x: torch.Tensor,
        pos=None,
        pos_max=None,
        patch_h: int | None = None,
        patch_w: int | None = None,
        num_suffix_tokens: int = 0,
        num_prefix_tokens: int = 0,
    ):
        """Compute Q, K, V and learning rates for input x.

        Returns:
            Tuple of (q, k, v, lr0, lr1, lr2) all appropriately shaped
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.using_short_conv_qkv and N > num_prefix_tokens + num_suffix_tokens:
            assert patch_h is not None and patch_w is not None
            # B, N, 3, h, d -> (3, B, h, N, d)
            q, k, v = qkv.unbind(0)
            q = F.silu(self.short_conv_qkv[0](q, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens))
            k = F.silu(self.short_conv_qkv[1](k, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens))
            v = F.silu(self.short_conv_qkv[2](v, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens))
        else:
            # B, N, 3, h, d -> (3, B, h, N, d)
            q, k, v = F.silu(qkv, inplace=True).unbind(0)

        if self.rope is not None:
            q = self.rope(q, pos, pos_max=pos_max)
            k = self.rope(k, pos, pos_max=pos_max)

        q, k, v = map(lambda t: rearrange(t, "b h l d -> (b h) l d", h=self.num_heads, b=B), (q, k, v))

        # Normalize q and k
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        q = self.q_hook(q)
        k = self.k_hook(k)

        # Compute learning rates
        lr = self.lr_fc(x)  # [b, l, lr_dim]
        lr = torch.nn.functional.softplus(lr + self.base_lr_inv)
        if self.div_lr_by_seq_len:
            lr = lr / N
        lr0, lr1, lr2 = rearrange(lr, "b l (lrs h d) -> lrs (b h) l d", lrs=3, h=self.num_heads)
        return {"q": q, "k": k, "v": v, "lr0": lr0, "lr1": lr1, "lr2": lr2}

    def _translate_operators_for_chunk(
        self,
        ttt_op_order: list[TTTOperator],
        chunk_size: int | None,
        N: int,
    ) -> list[TTTOperator]:
        """Translate global TTTOperators to local chunk indices.

        Args:
            ttt_op_order: List of operators with global indices
            chunk_start, chunk_end: Global range of current chunk
            N: Total sequence length

        Returns:
            List of TTTOperators with local indices for this chunk
        """
        if chunk_size is None:
            return ttt_op_order

        local_ops = []

        for op_idx, op in enumerate(ttt_op_order):
            op_start = op.start
            op_end = op.end or N

            if op.compute_grad:
                for chunk_start in range(op_start, op_end, chunk_size):
                    local_ops.append(
                        TTTOperator(
                            start=chunk_start,
                            end=chunk_start + chunk_size,
                            compute_grad=op.compute_grad,
                            update=False,
                            apply=False,
                        )
                    )

            if op.update:
                local_ops.append(
                    TTTOperator(
                        start=op_start,
                        end=op_end,
                        compute_grad=False,
                        update=True,
                        apply=False,
                    )
                )

            if op.apply:
                for chunk_start in range(op_start, op_end, chunk_size):
                    local_ops.append(
                        TTTOperator(
                            start=chunk_start,
                            end=chunk_start + chunk_size,
                            compute_grad=False,
                            update=False,
                            apply=True,
                        )
                    )

        return local_ops

    def _compute_qkv_lr(
        self,
        x: torch.Tensor,
        pos=None,
        pos_max=None,
        patch_h: int | None = None,
        patch_w: int | None = None,
        num_suffix_tokens: int = 0,
        num_prefix_tokens: int = 0,
        chunk_size: int | None = None,
        offload_to_cpu: bool = False,
    ):
        """Chunked forward pass with CPU-to-GPU transfers.

        This processes the input in chunks, reusing fast_weight_swish_glu_weight_norm_mini_batch_apply
        by translating global TTTOperators to local chunk indices.

        Args:
            x: Input tensor [b, l, d], may be on CPU
            w0, w1, w2: Initial weight tensors
            ttt_op_order: List of TTTOperators with global indices
            sp_group: Sequence parallel process group
            pos, pos_max, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens:
                Parameters for QKV computation

        Returns:
            Tuple of (output, state_dict)
        """
        if chunk_size is None:
            return self._compute_qkv_lr_impl(x, pos, pos_max, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens)

        B, N, C = x.shape
        chunk_size = chunk_size or N

        accum = []
        start_end = self.get_chunk_boundaries(N, chunk_size, num_prefix_tokens, num_suffix_tokens, patch_h, patch_w)
        for chunk_start, chunk_end in start_end:
            chunk_prefix_tokens = num_prefix_tokens
            chunk_suffix_tokens = max(0, num_suffix_tokens - (N - chunk_end))

            pos_chunk = None
            if pos is not None and self.rope is not None:
                pos_chunk = pos[:, chunk_start:chunk_end]

            x_chunk = x[:, chunk_start:chunk_end, :]

            # Compute QKV and learning rates for this chunk
            out = self._compute_qkv_lr_impl(
                x_chunk, pos_chunk, pos_max, patch_h, patch_w, chunk_suffix_tokens, chunk_prefix_tokens
            )
            accum.append(out) if not offload_to_cpu else accum.append(move_to_device(out, torch.device("cpu")))
        return {k: torch.cat([v[k] for v in accum], dim=1) for k in accum[0].keys()}

    def forward(
        self,
        x: torch.Tensor,
        pos=None,
        info=None,
        pos_max=None,
        patch_h: int | None = None,
        patch_w: int | None = None,
        num_suffix_tokens: int = 0,
        num_prefix_tokens: int = 0,
        chunk_size: int | None = None,
        track_details: bool = False,
        use_best_weights: bool = False,
        offload_to_cpu: bool = False,
        lr_decay: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        x: (b, l, d)
        """
        _logger.debug("TTTAttention: Forward pass, chunk_size: %s, offload_to_cpu: %s", chunk_size, offload_to_cpu)
        B, N, C = x.shape

        # Initialize weights
        if self.state is not None:
            state = self.state
        else:
            state = {
                "w0": self.w0.repeat(x.shape[0], 1, 1),
                "w1": self.w1.repeat(x.shape[0], 1, 1),
                "w2": self.w2.repeat(x.shape[0], 1, 1),
            }

        # Get TTT operation order
        if info:
            ttt_op_order = info["ttt_op_order"]
        else:
            # Default: Update with everything and apply
            ttt_op_order = [
                *([TTTOperator(start=0, end=N, compute_grad=True, update=True, apply=False)] * self.num_steps),
                TTTOperator(start=0, end=N, compute_grad=False, update=False, apply=True),
            ]

        qkv_lr = self._compute_qkv_lr(
            x, pos, pos_max, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens, chunk_size, offload_to_cpu
        )

        ttt_op_order = self._translate_operators_for_chunk(ttt_op_order, chunk_size, N)
        output, state, details = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            **state,
            **qkv_lr,
            ttt_ua_order=ttt_op_order,
            muon_update_steps=self.muon_update_steps,
            sp_group=get_sp_group() if self.seq_parallel else None,
            norm_grad=self.norm_ttt_grad,
            track_details=track_details,
            use_best_weights=use_best_weights,
            offload_to_cpu=offload_to_cpu,
            lr_decay=lr_decay,
        )
        self.logging_hook(details)

        output = rearrange(output, "(b h) l d -> b l (h d)", h=self.num_heads, b=B)
        output = apply_chunked(self._proj, output, dim=1, chunk_size=chunk_size, offload_to_cpu=offload_to_cpu)

        if self.state_tracking:
            self.state = state
        return output.cuda(non_blocking=True)

    def _proj(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
            f"Muon update steps: {self.muon_update_steps}, "
            f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, "
        )

    def get_chunk_boundaries(
        self,
        N: int,
        chunk_size: int | None,
        num_prefix_tokens: int,
        num_suffix_tokens: int,
        patch_h: int | None = None,
        patch_w: int | None = None,
    ) -> list[tuple[int, int]]:
        if chunk_size is None:
            return [(0, N)]

        if patch_h is None or patch_w is None:
            img_starts = list(range(0, N, chunk_size))
        else:
            img_starts = list(range(0, N - num_suffix_tokens, num_prefix_tokens + patch_h * patch_w))

        start_end = []
        cur_start = 0
        for img_start in img_starts:
            if img_start - cur_start >= chunk_size:
                start_end.append((cur_start, img_start))
                cur_start = img_start
        start_end.append((cur_start, N))
        return start_end
