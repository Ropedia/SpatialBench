import logging
import os
import warnings
import pdb
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import math

XFORMERS_AVAILABLE = False

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
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
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

    def forward(self, x: Tensor, pos=None, temporal_features=None, S=None, P=None, attn_mask=None, attn_value=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # [B, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos); k = self.rope(k, pos)

        Visual = False
        import math
        if Visual and not self.training and B > 0:
            with torch.no_grad():
                q0 = q[0, 0, :]    # �~O~V batch=0, head=0, �~I~M200 token
                k0 = k[0, 0, :]
                logits = (q0 @ k0.transpose(0, 1)) / math.sqrt(self.head_dim)
                print("QK/sqrt(d) range:", logits.min().item(), logits.max().item())

        if attn_mask is not None and attn_mask.any():
            attn_mask = attn_mask[:, None, :, None]
            attn_value = attn_value[:, :, :, None]
            #  head
            B, H, N, D = q.shape
            r = attn_mask.expand(B, H, N, 1)
            c = attn_value.expand(B, H, N, 1)
            D_aug = ((D + 1 + 7) // 8) * 8      #  64+1 �~F~R 72
            q = q * (D_aug / D) ** 0.5; 
            q_bias = r * D_aug ** 0.5
            k = k; k_bias = c
            zero_tail7 = torch.zeros(B, H, N, 7, device=attn_mask.device, dtype=attn_mask.dtype)
            zero_tail8 = torch.zeros(B, H, N, 8, device=attn_mask.device, dtype=attn_mask.dtype)
            q = torch.cat([q, q_bias, zero_tail7], dim=-1)  # (B,H,N->S*P,d+1)
            k = torch.cat([k, k_bias, zero_tail7], dim=-1)  # (B,H,N,d+1)
            v = torch.cat([v, zero_tail8], dim=-1)  # (B,H,N,d+1)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        if attn_mask is not None and attn_mask.any():
            x = x[:, :, :, :D]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, temporal_features=None, S=None, P=None, attn_mask=None, attn_value=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        if temporal_features is not None:
            q = q + 0.1 * temporal_features.view(B, N, self.num_heads*self.head_dim).view(B, N, self.num_heads, self.head_dim) # [B, num_heads, N, head_dim]
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, attn_mask=attn_mask, attn_value=attn_value)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
