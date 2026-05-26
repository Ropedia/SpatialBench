# SPDX-FileCopyrightText: Copyright (c) 2025 Tianyuan Zhang, Hao Tan
# SPDX-License-Identifier: MIT
#
# This file is adapted from LaCT:
#   https://github.com/a1600012888/LaCT/blob/main/lact_nvs/lact_ttt.py
# Original work licensed under the MIT License:
#   https://github.com/a1600012888/LaCT/blob/main/LICENSE
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Test-time training utilities."""

import logging
import math
from collections import defaultdict
from typing import Callable, NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_reduce

from vggttt.data.utils import move_to_device

logger = logging.getLogger(__name__)


class TTTOperator(NamedTuple):
    start: int
    end: int | None
    update: bool
    apply: bool
    compute_grad: bool


def inv_softplus(x: float):
    y = x + math.log(-math.expm1(-x))
    return y


def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int):
    """Newton-Schulz orthogonalisation.

    Args:
        G: [b, d, d] input matrices to orthogonalise.
        steps: number of Newton-Schulz iterations.

    Returns:
        X: [b, d, d] orthogonalised matrices.
    """
    if steps < 0:
        return G

    assert len(G.shape) == 3

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.clone()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = torch.bmm(X, X.mT)
        # B = b * A + c * A @ A
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        # X = a * X + B @ X
        X = torch.baddbmm(X, B, X, beta=a)

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


def fast_weight_swish_glu_fwd(
    q: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
):
    return F.silu(q @ w0, inplace=True) * (q @ w2) @ w1


@torch.enable_grad()
def fast_weight_swish_glu_vjp(
    k: torch.Tensor,
    v: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
):
    w0.requires_grad_()
    w1.requires_grad_()
    w2.requires_grad_()

    v_pred = fast_weight_swish_glu_fwd(k, w0, w1, w2)
    mul = v_pred * v  # loss = dot(v_pred, v)

    w0_grad = torch.autograd.grad((lr0 * mul).sum(), w0, create_graph=True)[0]
    w1_grad = torch.autograd.grad((lr1 * mul).sum(), w1, create_graph=True)[0]
    w2_grad = torch.autograd.grad((lr2 * mul).sum(), w2, create_graph=True)[0]
    return w0_grad, w1_grad, w2_grad


def apply_chunked(
    func: Callable,
    x: torch.Tensor,
    *args,
    dim: int = 1,
    chunk_size: int | None = 128,
    offload_to_cpu: bool = False,
    **kwargs,
):
    if chunk_size is None:
        return func(x, *args, **kwargs)

    out = []
    for chunk in x.split(chunk_size, dim):
        chunk_out = func(chunk.cuda(non_blocking=True), *args, **kwargs)
        out.append(chunk_out if not offload_to_cpu else move_to_device(chunk_out, torch.device("cpu")))

    if not out:
        return out

    if isinstance(out[0], dict):
        return {k: torch.cat([v[k] for v in out], dim=dim) for k in out[0]}
    return torch.cat(out, dim=dim)


@torch.no_grad()
def compute_error(q: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, v: torch.Tensor):
    errors = []
    for qi, vi in zip(torch.split(q, 1024, dim=1), torch.split(v, 1024, dim=1)):
        error = (fast_weight_swish_glu_fwd(qi, w0, w1, w2) - vi).norm(dim=-1)
        errors.append(error)
    errors = torch.cat(errors, dim=1)
    return errors


def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_ua_order: list[TTTOperator],
    muon_update_steps: int = 0,
    momentum: float = 1.0,
    lr_decay: float = 1.0,
    sp_group: dist.ProcessGroup | None = None,  # Process group for sequence parallel
    auto_grad: bool = False,
    norm_grad: bool = False,
    track_details: bool = False,
    use_best_weights: bool = False,
    offload_to_cpu: bool = False,
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    B, S, D = q.shape
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    grads = {"w0": torch.zeros_like(w0), "w1": torch.zeros_like(w1), "w2": torch.zeros_like(w2)}

    details = defaultdict(list)
    if track_details or use_best_weights:
        with torch.no_grad():
            error = compute_error(q, w0, w1, w2, v)
            details["error"].append(error.mean())

        if use_best_weights:
            best_weights = {"w0": w0.detach().clone(), "w1": w1.detach().clone(), "w2": w2.detach().clone()}

    # TODO: Mini-batches + multiple steps?
    step = 0
    for op in ttt_ua_order:
        start = op.start
        end = op.end or S
        w0_now, w1_now, w2_now = w0, w1, w2

        if op.compute_grad:
            # NOTE: All of the below code (magically??) works when start = end and grads will be 0 so no additional care
            # needs to be taken for the sequence parallel setting
            ki, vi = k[:, start:end, :].cuda(non_blocking=True), v[:, start:end, :].cuda(non_blocking=True)  # bf16
            lr0i = lr0[:, start:end, :].cuda(non_blocking=True)  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :].cuda(non_blocking=True)  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :].cuda(non_blocking=True)  # [b, l, d/1] fp32

            if auto_grad:
                # Compute gradients using torch.autograd.grad
                w0_grad, w1_grad, w2_grad = fast_weight_swish_glu_vjp(ki, vi, w0_now, w1_now, w2_now, lr0i, lr1i, lr2i)
            else:
                # Compute gradients manually: this is about 1.5x faster than using torch.autograd.grad
                gate_before_act = ki @ w0_now  # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                hidden_before_mul = ki @ w2_now  # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

                dhidden = vi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
                dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
                dgate = dhidden * hidden_before_mul
                dgate_before_act = silu_backprop(dgate, gate_before_act)

                # Compute gradients
                # w1.grad = -matmul(hidden.transpose(-1, -2), v * lr1) # [b, dh, d] = [b, dh, l] x [b, l, d] # [b, d, dh] = [b, d, l] x [b, l, dh]
                w1_grad = (hidden * lr1i).transpose(-1, -2) @ vi
                # w0.grad = -matmul((k * lr0).transpose(-1, -2), dgate_before_act)
                w0_grad = (ki * lr0i).transpose(-1, -2) @ dgate_before_act
                # w2.grad = -matmul((k * lr2).transpose(-1, -2), dhidden_before_gate)
                w2_grad = (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul

            # Scale gradient by the sequence length
            #  (effectively optimizing the mean of the reconstruction error instead of the sum)
            if norm_grad:
                norm_factor = end - start + 1e-8
                w1_grad = w1_grad / norm_factor
                w0_grad = w0_grad / norm_factor
                w2_grad = w2_grad / norm_factor

            grads["w0"] += w0_grad
            grads["w1"] += w1_grad
            grads["w2"] += w2_grad

        if op.update:
            # Synchronize gradients across the provided process group for sequence parallel
            if sp_group is not None and sp_group.size() > 1:
                sp_group.barrier()  # all ranks need to be done with their local gradients before all_reduce
                grads["w1"] = all_reduce(grads["w1"], op=dist.ReduceOp.SUM, group=sp_group)
                grads["w0"] = all_reduce(grads["w0"], op=dist.ReduceOp.SUM, group=sp_group)
                grads["w2"] = all_reduce(grads["w2"], op=dist.ReduceOp.SUM, group=sp_group)

            weight = lr_decay**step
            w1_now = w1_now + weight * zeropower_via_newtonschulz5(grads["w1"], muon_update_steps)
            w0_now = w0_now + weight * zeropower_via_newtonschulz5(grads["w0"], muon_update_steps)
            w2_now = w2_now + weight * zeropower_via_newtonschulz5(grads["w2"], muon_update_steps)

            if track_details:
                details["grads"].append({k: v.detach().clone().cpu() for k, v in grads.items()})

            # Reset gradients
            grads = {k: torch.zeros_like(v) for k, v in grads.items()}

            # do weight norm here
            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            if track_details or use_best_weights:
                with torch.no_grad():
                    error = compute_error(q, w0_now, w1_now, w2_now, v).mean()
                    if use_best_weights and error < details["error"][-1]:
                        best_weights = {
                            "w0": w0_now.detach().clone(),
                            "w1": w1_now.detach().clone(),
                            "w2": w2_now.detach().clone(),
                        }
                    details["error"].append(error)

            step += 1
            w0, w1, w2 = w0_now, w1_now, w2_now

        if op.apply:
            # Only calculate the output in the last repeat.
            if use_best_weights:
                w0, w1, w2 = best_weights["w0"], best_weights["w1"], best_weights["w2"]

            qi = q[:, start:end, :].cuda(non_blocking=True)
            oi = fast_weight_swish_glu_fwd(qi, w0, w1, w2)
            output.append(oi if not offload_to_cpu else move_to_device(oi, torch.device("cpu")))

    output = torch.cat(output, dim=1)

    return output, {"w0": w0, "w1": w1, "w2": w2}, details
