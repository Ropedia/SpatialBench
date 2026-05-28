from typing import Optional, Set

import torch
import torch.nn.functional as F

from depth_anything_3.utils.geometry import (
    affine_inverse,
    as_homogeneous,
    mat_to_quat,
    quat_to_mat,
)


def pose_encoding_to_hmat(pose_enc: torch.Tensor) -> torch.Tensor:
    """Convert pose encoding [t, q, fov] to homogeneous transform matrices."""
    R = quat_to_mat(pose_enc[..., 3:7])
    T = pose_enc[..., :3]
    extrinsics = torch.cat([R, T.unsqueeze(-1)], dim=-1)
    return as_homogeneous(extrinsics)


def hmat_to_pose_encoding(hmat: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    """Convert homogeneous transform matrices back to pose encoding."""
    R = hmat[..., :3, :3]
    T = hmat[..., :3, 3]
    q = mat_to_quat(R)
    return torch.cat([T, q, fov], dim=-1)


def relative_pose_from_absolute_pose(ref_pose_enc: torch.Tensor, target_pose_enc: torch.Tensor) -> torch.Tensor:
    """Express a target absolute pose relative to a reference absolute pose."""
    ref_hmat = pose_encoding_to_hmat(ref_pose_enc)
    target_hmat = pose_encoding_to_hmat(target_pose_enc)
    rel_hmat = target_hmat @ affine_inverse(ref_hmat)
    return hmat_to_pose_encoding(rel_hmat, target_pose_enc[..., 7:9])


_pose_encoding_to_hmat = pose_encoding_to_hmat
_hmat_to_pose_encoding = hmat_to_pose_encoding


def _pairwise_relative_pose_from_absolute_pose_enc(abs_pose_enc: torch.Tensor):
    """Build pairwise relative pose encoding from absolute pose encoding."""
    abs_hmat = pose_encoding_to_hmat(abs_pose_enc)
    abs_inv = affine_inverse(abs_hmat)
    rel_hmat = abs_hmat[:, None, :, :, :] @ abs_inv[:, :, None, :, :]

    rel_R = rel_hmat[..., :3, :3]
    rel_t = rel_hmat[..., :3, 3]
    rel_q = mat_to_quat(rel_R)

    fov_j = abs_pose_enc[..., 7:9]
    rel_fov = fov_j[:, None, :, :].expand(-1, abs_pose_enc.shape[1], -1, -1)

    rel_pose_enc = torch.cat([rel_t, rel_q, rel_fov], dim=-1)
    return rel_pose_enc


def _sparse_relative_pose_from_absolute_pose_enc(
    abs_pose_enc: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor
) -> torch.Tensor:
    """Build pairwise relative pose encoding for valid sparse edges from absolute pose encoding."""
    abs_hmat = pose_encoding_to_hmat(abs_pose_enc)
    abs_inv = affine_inverse(abs_hmat)

    rel_hmat = abs_hmat[dst_idx] @ abs_inv[src_idx]

    rel_R = rel_hmat[..., :3, :3]
    rel_t = rel_hmat[..., :3, 3]
    rel_q = mat_to_quat(rel_R)

    rel_fov = abs_pose_enc[dst_idx, 7:9]
    return torch.cat([rel_t, rel_q, rel_fov], dim=-1)


def _relative_confidence_to_weight(
    pred_rel_conf: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Map raw relative confidence logits to strictly positive weights."""
    return F.softplus(pred_rel_conf) + eps


def _resolve_reconstruction_confidences(
    pred_rel_conf: torch.Tensor,
    pred_rel_conf_t: torch.Tensor | None = None,
    pred_rel_conf_r: torch.Tensor | None = None,
    score_mode: str = "auto",
    *,
    auto_split_mode: str = "separate",
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Resolve translation/rotation confidences and whether true split weighting is active."""
    has_true_split_confidence = pred_rel_conf_t is not None and pred_rel_conf_r is not None
    fallback_conf = pred_rel_conf
    if fallback_conf is None:
        fallback_conf = pred_rel_conf_r if pred_rel_conf_r is not None else pred_rel_conf_t

    resolved_conf_t = fallback_conf if pred_rel_conf_t is None else pred_rel_conf_t
    resolved_conf_r = fallback_conf if pred_rel_conf_r is None else pred_rel_conf_r

    mode = str(score_mode)
    if mode == "auto":
        mode = auto_split_mode if has_true_split_confidence else "shared"

    if mode == "shared":
        if not isinstance(fallback_conf, torch.Tensor):
            raise ValueError("score_mode='shared' requires pred_rel_conf.")
        return fallback_conf, fallback_conf, False

    if mode == "separate":
        if not has_true_split_confidence:
            if fallback_conf is None:
                raise ValueError("score_mode='separate' requires relative confidence predictions.")
            return fallback_conf, fallback_conf, False
        return resolved_conf_t, resolved_conf_r, True

    if resolved_conf_t is None or resolved_conf_r is None:
        raise ValueError(f"score_mode='{mode}' requires relative confidence predictions.")

    if mode == "mean":
        scalar_conf = 0.5 * (resolved_conf_t + resolved_conf_r)
    elif mode == "min":
        scalar_conf = torch.minimum(resolved_conf_t, resolved_conf_r)
    elif mode == "translation":
        scalar_conf = resolved_conf_t
    elif mode == "rotation":
        scalar_conf = resolved_conf_r
    else:
        raise ValueError(f"Unknown reconstruction confidence score_mode: {score_mode}")

    return scalar_conf, scalar_conf, False


def _quat_cosine_loss(pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """Sign-invariant cosine loss for quaternion pairs."""
    pred_q = F.normalize(pred_q, dim=-1, eps=1e-8)
    target_q = F.normalize(target_q, dim=-1, eps=1e-8)
    return 1.0 - (pred_q * target_q).sum(dim=-1).abs()


def _geman_mcclure_loss(residual_sq: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Geman-McClure robust loss: caps penalty at 1.0 for arbitrarily large errors."""
    return residual_sq / (residual_sq + c * c)


def _dynamic_covariance_scaling(sq_residuals: torch.Tensor, phi: float = 1.0) -> torch.Tensor:
    """Dynamic Covariance Scaling: downweight edges with large current residuals."""
    return torch.clamp((2.0 * phi) / (phi + sq_residuals + 1e-8), max=1.0)


def _filter_implausible_edges(
    src_idx: torch.Tensor, dst_idx: torch.Tensor, rel_t: torch.Tensor, max_translation_per_frame: float = 2.0
) -> torch.Tensor:
    """Filter edges with physically implausible translation given frame gap."""
    frame_gap = (dst_idx - src_idx).abs().float().clamp(min=1.0)
    return rel_t.norm(dim=-1) <= frame_gap * max_translation_per_frame


def _build_pgo_fixed_frame_mask(seq_len: int, keyframe_stride: int, device: torch.device) -> torch.Tensor:
    """Build a fixed-frame mask for greedy keyframe scaffolding during PGO."""
    fixed_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    fixed_mask[0] = True
    if keyframe_stride > 0:
        fixed_mask[::keyframe_stride] = True
        fixed_mask[-1] = True
    return fixed_mask


def compose_relative_pose(rel_pose_enc: torch.Tensor, abs_pose_enc: torch.Tensor):
    rel_rot = quat_to_mat(rel_pose_enc[..., 3:7])
    abs_rot = quat_to_mat(abs_pose_enc[..., 3:7])
    abs_trans = abs_pose_enc[..., :3].unsqueeze(-1)

    out_rot = rel_rot @ abs_rot
    out_trans = (rel_rot @ abs_trans).squeeze(-1) + rel_pose_enc[..., :3]
    out_quat = mat_to_quat(out_rot)
    return torch.cat([out_trans, out_quat, rel_pose_enc[..., 7:9]], dim=-1)


def _normalize_masked_weights(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert masked confidence logits into normalized averaging weights."""
    weights = torch.softmax(logits, dim=1)
    weights = weights * mask.to(weights.dtype)
    weight_sum = weights.sum(dim=1, keepdim=True)
    return weights / weight_sum.clamp_min(1e-8), weight_sum.squeeze(1) > 0


def average_pose_candidates(
    candidate_pose_enc: torch.Tensor,
    weights: torch.Tensor,
    rotation_weights: torch.Tensor | None = None,
):
    candidate_trans = candidate_pose_enc[..., :3]
    candidate_quat = candidate_pose_enc[..., 3:7]
    candidate_fov = candidate_pose_enc[..., 7:9]
    rotation_weights = weights if rotation_weights is None else rotation_weights

    ref_quat = candidate_quat[:, :1]
    sign = torch.where(
        (candidate_quat * ref_quat).sum(dim=-1, keepdim=True) < 0,
        -1.0,
        1.0,
    )
    candidate_quat = candidate_quat * sign

    avg_trans = (weights.unsqueeze(-1) * candidate_trans).sum(dim=1)
    avg_quat = F.normalize(
        (rotation_weights.unsqueeze(-1) * candidate_quat).sum(dim=1),
        dim=-1,
        eps=1e-8,
    )
    avg_fov = (weights.unsqueeze(-1) * candidate_fov).sum(dim=1)
    return torch.cat([avg_trans, avg_quat, avg_fov], dim=-1).unsqueeze(1)


def build_online_pose_from_memory(
    state,
    rel_pose_enc,
    rel_pose_conf,
    rel_pose_mask,
    rel_pose_frame_ids,
    rel_pose_conf_t=None,
    rel_pose_conf_r=None,
    topn_conf=10,
    max_recent=0,
):
    """Build absolute pose for current frame from memory frames' relative predictions."""
    if not (
        isinstance(rel_pose_enc, torch.Tensor)
        and isinstance(rel_pose_conf, torch.Tensor)
        and isinstance(rel_pose_mask, torch.Tensor)
    ):
        return None
    if topn_conf == -1:
        topn_conf = rel_pose_conf.shape[1] // 2

    current_idx = len(rel_pose_frame_ids) - 1
    candidate_indices = []
    candidate_abs_pose = []
    for idx, frame_id in enumerate(rel_pose_frame_ids[:-1]):
        if frame_id in state["frame_pose_enc"]:
            candidate_indices.append(idx)
            candidate_abs_pose.append(state["frame_pose_enc"][frame_id])

    if max_recent > 0 and len(candidate_indices) > max_recent:
        paired = sorted(
            zip(candidate_indices, candidate_abs_pose),
            key=lambda x: rel_pose_frame_ids[x[0]],
            reverse=True,
        )
        paired = paired[:max_recent]
        paired.sort(key=lambda x: rel_pose_frame_ids[x[0]])
        candidate_indices = [p[0] for p in paired]
        candidate_abs_pose = [p[1] for p in paired]

    if not candidate_indices:
        return None

    abs_pose_enc = torch.stack(candidate_abs_pose, dim=1)
    rel_pose_to_current = rel_pose_enc[:, candidate_indices, current_idx]
    candidate_pose_enc = compose_relative_pose(rel_pose_to_current, abs_pose_enc)

    candidate_conf = rel_pose_conf[:, candidate_indices, current_idx]
    candidate_mask = rel_pose_mask[:, candidate_indices, current_idx]
    candidate_conf = candidate_conf.masked_fill(~candidate_mask, -1e9)

    if topn_conf is not None:
        topk = min(max(int(topn_conf), 1), candidate_conf.shape[1])
        topk_idx = torch.topk(candidate_conf, k=topk, dim=1).indices
        topk_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)
        candidate_mask = candidate_mask & topk_mask
        candidate_conf = candidate_conf.masked_fill(~candidate_mask, -1e9)

    candidate_conf_t = candidate_conf
    candidate_conf_r = candidate_conf
    if isinstance(rel_pose_conf_t, torch.Tensor) and isinstance(rel_pose_conf_r, torch.Tensor):
        candidate_conf_t = rel_pose_conf_t[:, candidate_indices, current_idx].masked_fill(~candidate_mask, -1e9)
        candidate_conf_r = rel_pose_conf_r[:, candidate_indices, current_idx].masked_fill(~candidate_mask, -1e9)

    weights, valid_translation_weights = _normalize_masked_weights(candidate_conf_t, candidate_mask)
    rotation_weights, valid_rotation_weights = _normalize_masked_weights(candidate_conf_r, candidate_mask)
    valid_weights = valid_translation_weights & valid_rotation_weights
    if not valid_weights.any():
        return None

    return average_pose_candidates(candidate_pose_enc, weights, rotation_weights=rotation_weights)


def _average_candidate_pose(
    candidate_hmat: torch.Tensor,
    candidate_fov: torch.Tensor,
    weights: torch.Tensor,
    rotation_candidate_hmat: torch.Tensor | None = None,
    rotation_weights: torch.Tensor | None = None,
):
    rotation_candidate_hmat = candidate_hmat if rotation_candidate_hmat is None else rotation_candidate_hmat
    candidate_pose_enc = hmat_to_pose_encoding(candidate_hmat, candidate_fov)
    if rotation_candidate_hmat is not candidate_hmat:
        candidate_pose_enc = candidate_pose_enc.clone()
        candidate_pose_enc[:, 3:7] = mat_to_quat(rotation_candidate_hmat[:, :3, :3])

    avg_pose_enc = average_pose_candidates(
        candidate_pose_enc.unsqueeze(0),
        weights.unsqueeze(0),
        rotation_weights=None if rotation_weights is None else rotation_weights.unsqueeze(0),
    )[0, 0]
    avg_R = quat_to_mat(avg_pose_enc[3:7].unsqueeze(0))[0]
    return avg_R, avg_pose_enc[:3], avg_pose_enc[7:9]


def _restrict_rel_pose_mask_to_anchor(pred_rel_mask: torch.Tensor, anchor_only_index: int | None) -> torch.Tensor:
    """Restrict relative-pose reconstruction to edges attached to one anchor frame."""
    valid_mask = pred_rel_mask.bool()
    if anchor_only_index is None:
        return valid_mask

    seq_len = valid_mask.shape[-1]
    anchor_only_index = int(anchor_only_index)
    if not 0 <= anchor_only_index < seq_len:
        raise ValueError(f"anchor_only_index must be in [0, {seq_len}), got {anchor_only_index}")

    anchor_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    anchor_mask[..., anchor_only_index, :] = True
    anchor_mask[..., :, anchor_only_index] = True
    return valid_mask & anchor_mask


def _reconstruct_camera_sequence_greedy(
    pred_rel_pose_enc: torch.Tensor,
    pred_rel_conf_t: torch.Tensor,
    pred_rel_mask: torch.Tensor,
    topn_conf: int | str | None = 10,
    pred_rel_conf_r: torch.Tensor | None = None,
    candidate_selection: str = "translation",
    candidate_quantile: float | None = None,
    anchor_only_index: int | None = None,
) -> torch.Tensor:
    """Greedily reconstruct absolute camera sequence from relative predictions.

    Translation and rotation share the same candidate set during top-k selection.
    `candidate_selection` chooses whether that set is ranked by translation,
    rotation, or the mean of both confidences.
    """
    B, S = pred_rel_pose_enc.shape[:2]
    device = pred_rel_pose_enc.device
    dtype = pred_rel_pose_enc.dtype
    valid_mask = _restrict_rel_pose_mask_to_anchor(pred_rel_mask, anchor_only_index)
    pred_rel_conf_r = pred_rel_conf_t if pred_rel_conf_r is None else pred_rel_conf_r
    candidate_selection = str(candidate_selection)
    if candidate_selection not in {"translation", "rotation", "both"}:
        raise ValueError(
            "candidate_selection must be one of {'translation', 'rotation', 'both'}, "
            f"got {candidate_selection!r}."
        )
    if candidate_quantile is not None:
        candidate_quantile = float(candidate_quantile)

    rel_hmat = torch.zeros(B, S, S, 4, 4, device=device, dtype=dtype)
    rel_hmat[..., :3, :3] = quat_to_mat(pred_rel_pose_enc[..., 3:7])
    rel_hmat[..., :3, 3] = pred_rel_pose_enc[..., :3]
    rel_hmat[..., 3, 3] = 1.0

    abs_hmat = torch.zeros(B, S, 4, 4, device=device, dtype=dtype)
    abs_hmat[..., 3, 3] = 1.0
    abs_hmat[:, 0, :3, :3] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

    out_fov = torch.zeros(B, S, 2, device=device, dtype=dtype)
    out_fov[:, 0] = pred_rel_pose_enc[:, 0, 0, 7:9]

    use_all_candidates = False
    if topn_conf is None:
        use_all_candidates = True
    elif isinstance(topn_conf, str):
        if topn_conf != "all":
            raise ValueError(f"topn_conf must be an integer, None, or 'all', got {topn_conf!r}.")
        use_all_candidates = True
    else:
        topn_conf = max(int(topn_conf), 1)

    for b_idx in range(B):
        for j in range(1, S):
            valid_i = valid_mask[b_idx, :j, j]
            if valid_i.any():
                conf_t_vec = pred_rel_conf_t[b_idx, :j, j].masked_fill(~valid_i, -1e9)
                conf_r_vec = pred_rel_conf_r[b_idx, :j, j].masked_fill(~valid_i, -1e9)
                if candidate_selection == "translation":
                    selection_conf_vec = conf_t_vec
                elif candidate_selection == "rotation":
                    selection_conf_vec = conf_r_vec
                else:
                    selection_conf_vec = 0.5 * (conf_t_vec + conf_r_vec)
                candidate_i = valid_i
                if candidate_quantile is not None:
                    valid_scores = selection_conf_vec[valid_i]
                    cutoff = torch.quantile(valid_scores.float(), candidate_quantile).to(selection_conf_vec.dtype)
                    candidate_i = valid_i & (selection_conf_vec >= cutoff)
                    if not candidate_i.any():
                        candidate_i = torch.zeros_like(valid_i, dtype=torch.bool)
                        candidate_i[int(torch.argmax(selection_conf_vec).item())] = True

                ranking_conf_vec = selection_conf_vec.masked_fill(~candidate_i, -1e9)
                valid_count = int(candidate_i.sum().item())
                cur_topn = valid_count if use_all_candidates else min(topn_conf, valid_count)

                if cur_topn == 1:
                    if use_all_candidates:
                        best_i = int(torch.where(candidate_i)[0][0].item())
                    else:
                        best_i = int(torch.argmax(ranking_conf_vec).item())
                    best_hmat = rel_hmat[b_idx, best_i, j] @ abs_hmat[b_idx, best_i]
                    abs_hmat[b_idx, j, :3, :3] = best_hmat[:3, :3]
                    abs_hmat[b_idx, j, :3, 3] = best_hmat[:3, 3]
                    out_fov[b_idx, j] = pred_rel_pose_enc[b_idx, best_i, j, 7:9]
                else:
                    if use_all_candidates:
                        top_idx = torch.where(candidate_i)[0]
                    else:
                        _, top_idx = torch.topk(ranking_conf_vec, k=cur_topn, dim=0)
                    weights_t = torch.softmax(conf_t_vec[top_idx], dim=0)
                    weights_r = torch.softmax(conf_r_vec[top_idx], dim=0)

                    candidate_hmat = rel_hmat[b_idx, top_idx, j] @ abs_hmat[b_idx, top_idx]
                    candidate_fov = pred_rel_pose_enc[b_idx, top_idx, j, 7:9]
                    avg_R, avg_t, avg_fov = _average_candidate_pose(
                        candidate_hmat,
                        candidate_fov,
                        weights_t,
                        rotation_weights=weights_r,
                    )

                    abs_hmat[b_idx, j, :3, :3] = avg_R
                    abs_hmat[b_idx, j, :3, 3] = avg_t
                    out_fov[b_idx, j] = avg_fov
            else:
                abs_hmat[b_idx, j] = abs_hmat[b_idx, j - 1]
                out_fov[b_idx, j] = out_fov[b_idx, j - 1]

    return hmat_to_pose_encoding(abs_hmat, out_fov)


def _reconstruct_camera_sequence_pgo_single(
    pred_rel_pose_enc: torch.Tensor,
    pred_rel_conf: torch.Tensor,
    pred_rel_mask: torch.Tensor,
    init_pose_enc: torch.Tensor,
    pred_rel_conf_t: torch.Tensor | None = None,
    pred_rel_conf_r: torch.Tensor | None = None,
    score_mode: str = "auto",
    pgo_num_iters: int = 100,
    pgo_lr: float = 0.05,
    weight_T: float = 1.0,
    weight_R: float = 1.0,
    weight_fl: float = 0.25,
    init_prior_weight: float = 0.01,
    conf_eps: float = 1e-4,
    edge_percentile_cutoff: float = 0.0,
    keyframe_stride: int = 0,
    geman_mcclure_c: float = 0.0,
    dcs_phi: float = 0.0,
    max_translation_per_frame: float = 0.0,
    exclude_indices: Optional[Set[int]] = None,
    pgo_optimizer: str = "lbfgs",
    anchor_only_index: int | None = None,
) -> torch.Tensor:
    """Refine absolute poses from pairwise relative edges with a lightweight PGO."""
    # Build robust edge weights by confidence percentile filtering only.
    seq_len = pred_rel_pose_enc.shape[0]
    valid_mask = _restrict_rel_pose_mask_to_anchor(pred_rel_mask, anchor_only_index)
    if seq_len <= 1 or not valid_mask.any():
        return init_pose_enc
    print("Running PGO refinement with {} valid edges...".format(valid_mask.sum().item()))

    src_idx, dst_idx = torch.where(valid_mask)
    raw_edge_weights = _relative_confidence_to_weight(pred_rel_conf, eps=conf_eps)[src_idx, dst_idx]
    pgo_rel_conf_t, pgo_rel_conf_r, uses_split_confidence = _resolve_reconstruction_confidences(
        pred_rel_conf,
        pred_rel_conf_t=pred_rel_conf_t,
        pred_rel_conf_r=pred_rel_conf_r,
        score_mode=score_mode,
        auto_split_mode="separate",
    )
    raw_edge_weights_t = _relative_confidence_to_weight(pgo_rel_conf_t, eps=conf_eps)[src_idx, dst_idx]
    raw_edge_weights_r = _relative_confidence_to_weight(pgo_rel_conf_r, eps=conf_eps)[src_idx, dst_idx]
    print(
        "PGO edge weight min/max before percentile filtering: {:.6f}/{:.6f}".format(
            raw_edge_weights.min().item(),
            raw_edge_weights.max().item(),
        )
    )

    if edge_percentile_cutoff > 0:
        cutoff = torch.quantile(raw_edge_weights, edge_percentile_cutoff)
        # Keep edges on the percentile boundary so repeated-confidence edges do not
        # get over-pruned when many of them share the same score.
        keep_mask = raw_edge_weights >= cutoff
        if not keep_mask.any():
            keep_mask = raw_edge_weights >= cutoff
    else:
        keep_mask = torch.ones_like(raw_edge_weights, dtype=torch.bool)

    src_idx = src_idx[keep_mask]
    dst_idx = dst_idx[keep_mask]
    edge_weights = raw_edge_weights[keep_mask]
    edge_weights_t = raw_edge_weights_t[keep_mask]
    edge_weights_r = raw_edge_weights_r[keep_mask]

    # Filter out edges touching excluded frames
    if exclude_indices:
        excl_t = torch.tensor(sorted(exclude_indices), dtype=src_idx.dtype, device=src_idx.device)
        not_excluded = ~(torch.isin(src_idx, excl_t) | torch.isin(dst_idx, excl_t))
        if not not_excluded.any():
            return init_pose_enc
        src_idx = src_idx[not_excluded]
        dst_idx = dst_idx[not_excluded]
        edge_weights = edge_weights[not_excluded]
        edge_weights_t = edge_weights_t[not_excluded]
        edge_weights_r = edge_weights_r[not_excluded]

    # Physical plausibility pre-filter: remove edges with impossible translations
    if max_translation_per_frame > 0:
        gt_rel_t_pre = pred_rel_pose_enc[src_idx, dst_idx, :3]
        plausible = _filter_implausible_edges(src_idx, dst_idx, gt_rel_t_pre, max_translation_per_frame)
        if plausible.any():
            src_idx = src_idx[plausible]
            dst_idx = dst_idx[plausible]
            edge_weights = edge_weights[plausible]
            edge_weights_t = edge_weights_t[plausible]
            edge_weights_r = edge_weights_r[plausible]

    gt_rel_pose_enc = pred_rel_pose_enc[src_idx, dst_idx]
    gt_rel_t = gt_rel_pose_enc[..., :3]
    gt_rel_q = gt_rel_pose_enc[..., 3:7]
    gt_rel_fov = gt_rel_pose_enc[..., 7:9]

    with torch.no_grad():
        init_rel_pose_enc = _sparse_relative_pose_from_absolute_pose_enc(init_pose_enc, src_idx, dst_idx)
        init_rel_t_err = F.smooth_l1_loss(init_rel_pose_enc[..., :3], gt_rel_t, reduction="none", beta=0.1).mean(dim=-1)

        init_rel_q_err = _quat_cosine_loss(init_rel_pose_enc[..., 3:7], gt_rel_q)

        init_rel_fov_err = F.smooth_l1_loss(init_rel_pose_enc[..., 7:9], gt_rel_fov, reduction="none", beta=0.1).mean(
            dim=-1
        )
        init_edge_loss = weight_T * init_rel_t_err + weight_R * init_rel_q_err + 0.0 * init_rel_fov_err
        if init_edge_loss.numel() == 0 or init_edge_loss.max().item() < 1e-12:
            return init_pose_enc

    fixed_mask = _build_pgo_fixed_frame_mask(seq_len, int(keyframe_stride), init_pose_enc.device)
    # Fix excluded frames so their poses stay at greedy init
    if exclude_indices:
        for idx in exclude_indices:
            if 0 <= idx < seq_len:
                fixed_mask[idx] = True
    opt_idx = torch.where(~fixed_mask)[0]
    if opt_idx.numel() == 0:
        return init_pose_enc

    init_opt_pose = init_pose_enc[opt_idx].clone()
    valid_weight_sum = edge_weights.sum()

    if valid_weight_sum.item() <= 0:
        return init_pose_enc

    # --- LM optimizer path ---
    if pgo_optimizer == "lm":
        n_opt = opt_idx.numel()
        init_params_flat = init_opt_pose[:, :7].reshape(-1).detach()

        def residual_fn(params_flat):
            """Build flat residual vector for LM: translation, rotation, and prior terms."""
            params_7 = params_flat.reshape(n_opt, 7)
            cur_pose = init_pose_enc[:, :7].clone().detach()
            cur_pose[opt_idx] = torch.cat(
                [params_7[:, :3], F.normalize(params_7[:, 3:7], dim=-1, eps=1e-8)],
                dim=-1,
            )
            abs_pose_enc = torch.cat([cur_pose, init_pose_enc[:, 7:9].detach()], dim=-1)
            cur_rel = _sparse_relative_pose_from_absolute_pose_enc(abs_pose_enc, src_idx, dst_idx)

            residuals = []

            # Per-edge translation residuals [E*3]
            t_diff = cur_rel[..., :3] - gt_rel_t
            if geman_mcclure_c > 0:
                t_sq = t_diff.pow(2).sum(dim=-1, keepdim=True)
                irls_w = (geman_mcclure_c / (t_sq + geman_mcclure_c**2).sqrt()).detach()
                t_base_weights = edge_weights_t if uses_split_confidence else edge_weights
                t_res = (t_base_weights.unsqueeze(-1) * weight_T * irls_w).sqrt() * t_diff
            elif dcs_phi > 0:
                t_sq = t_diff.pow(2).sum(dim=-1, keepdim=True)
                dcs_w = torch.clamp((2.0 * dcs_phi) / (dcs_phi + t_sq + 1e-8), max=1.0).detach()
                t_base_weights = edge_weights_t if uses_split_confidence else edge_weights
                t_res = (t_base_weights.unsqueeze(-1) * weight_T * dcs_w).sqrt() * t_diff
            else:
                t_base_weights = edge_weights_t if uses_split_confidence else edge_weights
                t_res = (t_base_weights * weight_T).sqrt().unsqueeze(-1) * t_diff
            residuals.append(t_res.reshape(-1))

            # Per-edge rotation residuals [E]
            q_pred = F.normalize(cur_rel[..., 3:7], dim=-1, eps=1e-8)
            q_gt = F.normalize(gt_rel_q, dim=-1, eps=1e-8)
            cos_dist = 1.0 - (q_pred * q_gt).sum(dim=-1).abs()
            r_base_weights = edge_weights_r if uses_split_confidence else edge_weights
            r_res = (r_base_weights * weight_R).sqrt() * cos_dist.sqrt().clamp(min=1e-12)
            residuals.append(r_res)

            # Prior residuals [n_opt*7]
            if init_prior_weight > 0:
                prior_res = (init_prior_weight**0.5) * (params_flat - init_params_flat)
                residuals.append(prior_res)

            return torch.cat(residuals)

        x0 = init_opt_pose[:, :7].reshape(-1).detach()
        opt_result = _levenberg_marquardt_pgo(residual_fn, x0, max_iters=max(1, pgo_num_iters), lambda_init=1e-4)

        refined_pose = init_pose_enc.clone()
        opt_7 = opt_result.reshape(n_opt, 7)
        refined_pose[opt_idx, :7] = torch.cat(
            [opt_7[:, :3], F.normalize(opt_7[:, 3:7], dim=-1, eps=1e-8)],
            dim=-1,
        )
        return refined_pose.detach()

    # --- L-BFGS optimizer path (default) ---
    pose_params = torch.nn.Parameter(init_pose_enc[opt_idx, :7].clone())

    optimizer = torch.optim.LBFGS(
        [pose_params],
        lr=float(pgo_lr),
        max_iter=15,
        line_search_fn="strong_wolfe",
    )

    with torch.enable_grad():

        def closure():
            optimizer.zero_grad()
            cur_pose = init_pose_enc[:, :7].clone()
            cur_pose[opt_idx] = torch.cat(
                [
                    pose_params[:, :3],
                    F.normalize(pose_params[:, 3:7], dim=-1, eps=1e-8),
                ],
                dim=-1,
            )
            abs_pose_enc = torch.cat([cur_pose, init_pose_enc[:, 7:9]], dim=-1)

            cur_rel_pose_enc = _sparse_relative_pose_from_absolute_pose_enc(abs_pose_enc, src_idx, dst_idx)

            # Translation loss: Geman-McClure (robust) or Huber
            if geman_mcclure_c > 0:
                t_sq = (cur_rel_pose_enc[..., :3] - gt_rel_t).pow(2).sum(dim=-1)
                rel_t_err = _geman_mcclure_loss(t_sq, c=geman_mcclure_c)
            else:
                rel_t_err = F.smooth_l1_loss(cur_rel_pose_enc[..., :3], gt_rel_t, reduction="none", beta=0.1).mean(
                    dim=-1
                )

            rel_q_pred = cur_rel_pose_enc[..., 3:7]
            rel_q_err = _quat_cosine_loss(rel_q_pred, gt_rel_q)

            rel_fov_err = F.smooth_l1_loss(cur_rel_pose_enc[..., 7:9], gt_rel_fov, reduction="none", beta=0.1).mean(
                dim=-1
            )

            if uses_split_confidence:
                eff_w_t = edge_weights_t
                eff_w_r = edge_weights_r

                # Dynamic Covariance Scaling: modulate edge weights by current residual magnitude
                if dcs_phi > 0:
                    sq_res = (cur_rel_pose_enc[..., :3] - gt_rel_t).pow(2).sum(dim=-1) + rel_q_err.pow(2)
                    dcs_scale = _dynamic_covariance_scaling(sq_res, phi=dcs_phi)
                    eff_w_t = edge_weights_t * dcs_scale
                    eff_w_r = edge_weights_r * dcs_scale

                loss = rel_t_err.new_tensor(0.0)
                if weight_T > 0:
                    loss = loss + weight_T * (eff_w_t * rel_t_err).sum() / eff_w_t.sum().clamp(min=1e-8)
                if weight_R > 0:
                    loss = loss + weight_R * (eff_w_r * rel_q_err).sum() / eff_w_r.sum().clamp(min=1e-8)
                if weight_fl > 0:
                    loss = loss + 0.0 * rel_fov_err.mean()
            else:
                edge_loss = weight_T * rel_t_err + weight_R * rel_q_err + 0.0 * rel_fov_err

                if dcs_phi > 0:
                    sq_res = (cur_rel_pose_enc[..., :3] - gt_rel_t).pow(2).sum(dim=-1) + rel_q_err.pow(2)
                    dcs_scale = _dynamic_covariance_scaling(sq_res, phi=dcs_phi)
                    eff_w = edge_weights * dcs_scale
                    loss = (eff_w * edge_loss).sum() / eff_w.sum().clamp(min=1e-8)
                else:
                    loss = (edge_weights * edge_loss).sum() / (valid_weight_sum + 1e-8)

            if init_prior_weight > 0:
                pose_prior = (cur_pose[opt_idx] - init_opt_pose[:, :7]).pow(2).mean(dim=-1)

                loss = loss + init_prior_weight * (pose_prior.mean())
            loss.backward()
            return loss

        # Run LBFGS scaled by original iterations request
        for _ in range(max(1, int(pgo_num_iters) // 10)):
            optimizer.step(closure)

        refined_pose = init_pose_enc.clone()
        refined_pose[opt_idx, :7] = torch.cat(
            [pose_params[:, :3], F.normalize(pose_params[:, 3:7], dim=-1, eps=1e-8)],
            dim=-1,
        )

    return refined_pose.detach()


def _reconstruct_camera_sequence_pgo(
    pred_rel_pose_enc: torch.Tensor,
    pred_rel_conf: torch.Tensor,
    pred_rel_mask: torch.Tensor,
    topn_conf: int | str | None = 10,
    greedy_pred_rel_conf_t: torch.Tensor | None = None,
    greedy_pred_rel_conf_r: torch.Tensor | None = None,
    greedy_candidate_selection: str = "translation",
    candidate_quantile: float | None = None,
    pgo_pred_rel_conf_t: torch.Tensor | None = None,
    pgo_pred_rel_conf_r: torch.Tensor | None = None,
    score_mode: str = "auto",
    pgo_num_iters: int = 100,
    pgo_lr: float = 0.05,
    pgo_weight_T: float = 1.0,
    pgo_weight_R: float = 1.0,
    pgo_weight_fl: float = 0.25,
    pgo_init_prior_weight: float = 0.01,
    conf_eps: float = 1e-4,
    edge_percentile_cutoff: float = 0.0,
    pgo_keyframe_stride: int = 0,
    geman_mcclure_c: float = 0.0,
    dcs_phi: float = 0.0,
    max_translation_per_frame: float = 0.0,
    exclude_indices: Optional[Set[int]] = None,
    pgo_optimizer: str = "lbfgs",
    anchor_only_index: int | None = None,
) -> torch.Tensor:
    init_pose_enc = _reconstruct_camera_sequence_greedy(
        pred_rel_pose_enc,
        greedy_pred_rel_conf_t if greedy_pred_rel_conf_t is not None else pred_rel_conf,
        pred_rel_mask,
        topn_conf=topn_conf,
        pred_rel_conf_r=greedy_pred_rel_conf_r,
        candidate_selection=greedy_candidate_selection,
        candidate_quantile=candidate_quantile,
        anchor_only_index=anchor_only_index,
    )

    refined_pose = []
    for b_idx in range(pred_rel_pose_enc.shape[0]):
        refined_pose.append(
            _reconstruct_camera_sequence_pgo_single(
                pred_rel_pose_enc=pred_rel_pose_enc[b_idx],
                pred_rel_conf=pred_rel_conf[b_idx],
                pred_rel_mask=pred_rel_mask[b_idx],
                init_pose_enc=init_pose_enc[b_idx],
                pred_rel_conf_t=pgo_pred_rel_conf_t[b_idx] if pgo_pred_rel_conf_t is not None else None,
                pred_rel_conf_r=pgo_pred_rel_conf_r[b_idx] if pgo_pred_rel_conf_r is not None else None,
                score_mode=score_mode,
                pgo_num_iters=pgo_num_iters,
                pgo_lr=pgo_lr,
                weight_T=pgo_weight_T,
                weight_R=pgo_weight_R,
                weight_fl=pgo_weight_fl,
                init_prior_weight=pgo_init_prior_weight,
                conf_eps=conf_eps,
                edge_percentile_cutoff=edge_percentile_cutoff,
                keyframe_stride=pgo_keyframe_stride,
                geman_mcclure_c=geman_mcclure_c,
                dcs_phi=dcs_phi,
                max_translation_per_frame=max_translation_per_frame,
                exclude_indices=exclude_indices,
                pgo_optimizer=pgo_optimizer,
                anchor_only_index=anchor_only_index,
            )
        )
    return torch.stack(refined_pose, dim=0)


def reconstruct_camera_sequence_from_rel_pose(
    pred_rel_pose_enc: torch.Tensor,
    pred_rel_conf: torch.Tensor,
    pred_rel_mask: torch.Tensor,
    pred_rel_conf_t: torch.Tensor | None = None,
    pred_rel_conf_r: torch.Tensor | None = None,
    topn_conf: int | str | None = 10,
    method: str = "greedy",
    score_mode: str = "auto",
    candidate_selection: str = "translation",
    candidate_quantile: float | None = None,
    pgo_num_iters: int = 100,
    pgo_lr: float = 0.05,
    pgo_weight_T: float = 1.0,
    pgo_weight_R: float = 1.0,
    pgo_weight_fl: float = 0.1,
    pgo_init_prior_weight: float = 0.01,
    conf_eps: float = 1e-4,
    edge_percentile_cutoff: float = 0.0,
    pgo_keyframe_stride: int = 0,
    geman_mcclure_c: float = 0.0,
    dcs_phi: float = 0.0,
    max_translation_per_frame: float = 0.0,
    exclude_indices: Optional[Set[int]] = None,
    pgo_optimizer: str = "lbfgs",
    anchor_only_index: int | None = None,
) -> torch.Tensor:
    """Reconstruct absolute camera sequence from relative predictions."""
    greedy_conf_t, greedy_conf_r, _ = _resolve_reconstruction_confidences(
        pred_rel_conf,
        pred_rel_conf_t=pred_rel_conf_t,
        pred_rel_conf_r=pred_rel_conf_r,
        score_mode=score_mode,
        auto_split_mode="separate",
    )

    if method == "greedy":
        return _reconstruct_camera_sequence_greedy(
            pred_rel_pose_enc,
            greedy_conf_t,
            pred_rel_mask,
            topn_conf=topn_conf,
            pred_rel_conf_r=greedy_conf_r,
            candidate_selection=candidate_selection,
            candidate_quantile=candidate_quantile,
            anchor_only_index=anchor_only_index,
        )

    if method == "pgo":
        scalar_score_mode = "mean" if score_mode == "separate" else score_mode
        score_conf, _, _ = _resolve_reconstruction_confidences(
            pred_rel_conf,
            pred_rel_conf_t=pred_rel_conf_t,
            pred_rel_conf_r=pred_rel_conf_r,
            score_mode=scalar_score_mode,
            auto_split_mode="mean",
        )
        return _reconstruct_camera_sequence_pgo(
            pred_rel_pose_enc,
            score_conf,
            pred_rel_mask,
            topn_conf=topn_conf,
            greedy_pred_rel_conf_t=greedy_conf_t,
            greedy_pred_rel_conf_r=greedy_conf_r,
            greedy_candidate_selection=candidate_selection,
            candidate_quantile=candidate_quantile,
            pgo_pred_rel_conf_t=pred_rel_conf_t,
            pgo_pred_rel_conf_r=pred_rel_conf_r,
            score_mode=score_mode,
            pgo_num_iters=pgo_num_iters,
            pgo_lr=pgo_lr,
            pgo_weight_T=pgo_weight_T,
            pgo_weight_R=pgo_weight_R,
            pgo_weight_fl=pgo_weight_fl,
            pgo_init_prior_weight=pgo_init_prior_weight,
            conf_eps=conf_eps,
            edge_percentile_cutoff=edge_percentile_cutoff,
            pgo_keyframe_stride=pgo_keyframe_stride,
            geman_mcclure_c=geman_mcclure_c,
            dcs_phi=dcs_phi,
            max_translation_per_frame=max_translation_per_frame,
            exclude_indices=exclude_indices,
            pgo_optimizer=pgo_optimizer,
            anchor_only_index=anchor_only_index,
        )

    raise ValueError(f"Unknown reconstruction method: {method}")


def refine_camera_sequence_from_rel_pose(
    pred_rel_pose_enc: torch.Tensor,
    pred_rel_conf: torch.Tensor,
    pred_rel_mask: torch.Tensor,
    init_pose_enc: torch.Tensor,
    pred_rel_conf_t: torch.Tensor | None = None,
    pred_rel_conf_r: torch.Tensor | None = None,
    method: str = "pgo",
    score_mode: str = "auto",
    candidate_quantile: float | None = None,
    pgo_num_iters: int = 100,
    pgo_lr: float = 0.05,
    pgo_weight_T: float = 1.0,
    pgo_weight_R: float = 1.0,
    pgo_weight_fl: float = 0.25,
    pgo_init_prior_weight: float = 0.01,
    conf_eps: float = 1e-4,
    edge_percentile_cutoff: float = 0.0,
    pgo_keyframe_stride: int = 0,
    geman_mcclure_c: float = 0.0,
    dcs_phi: float = 0.0,
    max_translation_per_frame: float = 0.0,
    exclude_indices: Optional[Set[int]] = None,
    pgo_optimizer: str = "lbfgs",
    anchor_only_index: int | None = None,
) -> torch.Tensor:
    """Refine an existing absolute camera sequence using relative predictions."""
    if method != "pgo":
        raise ValueError(f"Unsupported refinement method: {method}")

    refined_pose = []
    for b_idx in range(pred_rel_pose_enc.shape[0]):
        refined_pose.append(
            _reconstruct_camera_sequence_pgo_single(
                pred_rel_pose_enc=pred_rel_pose_enc[b_idx],
                pred_rel_conf=pred_rel_conf[b_idx],
                pred_rel_mask=pred_rel_mask[b_idx],
                init_pose_enc=init_pose_enc[b_idx],
                pred_rel_conf_t=pred_rel_conf_t[b_idx] if pred_rel_conf_t is not None else None,
                pred_rel_conf_r=pred_rel_conf_r[b_idx] if pred_rel_conf_r is not None else None,
                score_mode=score_mode,
                pgo_num_iters=pgo_num_iters,
                pgo_lr=pgo_lr,
                weight_T=pgo_weight_T,
                weight_R=pgo_weight_R,
                weight_fl=pgo_weight_fl,
                init_prior_weight=pgo_init_prior_weight,
                conf_eps=conf_eps,
                edge_percentile_cutoff=edge_percentile_cutoff,
                keyframe_stride=pgo_keyframe_stride,
                geman_mcclure_c=geman_mcclure_c,
                dcs_phi=dcs_phi,
                max_translation_per_frame=max_translation_per_frame,
                exclude_indices=exclude_indices,
                pgo_optimizer=pgo_optimizer,
                anchor_only_index=anchor_only_index,
            )
        )
    return torch.stack(refined_pose, dim=0)


def _levenberg_marquardt_pgo(
    residual_fn,
    x0: torch.Tensor,
    max_iters: int = 50,
    lambda_init: float = 1e-4,
) -> torch.Tensor:
    """Hand-rolled Levenberg-Marquardt optimizer for PGO (VGGT-Long style)."""
    x = x0.clone().detach().requires_grad_(False)
    lam = lambda_init

    r = residual_fn(x)
    best_cost = (r * r).sum().item()

    for _ in range(max_iters):
        x_req = x.detach().requires_grad_(True)
        J = torch.autograd.functional.jacobian(residual_fn, x_req, vectorize=True)
        r = residual_fn(x_req.detach())

        JtJ = J.T @ J
        Jtr = J.T @ r

        diag = JtJ.diag().clamp(min=1e-12)
        delta = torch.linalg.solve(JtJ + lam * torch.diag(diag), -Jtr)

        x_new = x + delta
        r_new = residual_fn(x_new.detach())
        new_cost = (r_new * r_new).sum().item()

        if new_cost < best_cost:
            x = x_new.detach()
            best_cost = new_cost
            lam = max(lam / 2.0, 1e-10)
        else:
            lam = min(lam * 2.0, 1e6)

    return x.detach()
