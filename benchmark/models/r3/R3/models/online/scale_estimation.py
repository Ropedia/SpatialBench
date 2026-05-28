"""Scale estimation utilities extracted from R3 for independent testability."""

import math

import torch


def estimate_scale_weighted_median(
    old_depths, new_depths, old_confs=None, new_confs=None, frame_weights=None
):
    """Weighted-median scale estimation (pre-RANSAC, top-25% confidence pixels)."""
    all_ratios = []
    all_weights = []
    for idx, (old_d, new_d) in enumerate(zip(old_depths, new_depths)):
        old_d = old_d.squeeze().float()
        new_d = new_d.squeeze().float()
        valid = (
            (old_d > 1e-6)
            & (new_d > 1e-6)
            & torch.isfinite(old_d)
            & torch.isfinite(new_d)
        )
        if (
            old_confs is not None
            and idx < len(old_confs)
            and old_confs[idx] is not None
            and new_confs is not None
            and idx < len(new_confs)
            and new_confs[idx] is not None
        ):
            old_c = old_confs[idx].squeeze().float()
            new_c = new_confs[idx].squeeze().float()
            min_conf = torch.minimum(old_c, new_c)
            conf_vals = min_conf[valid]
            if conf_vals.numel() > 0:
                top_k = max(100, int(conf_vals.numel() * 0.25))
                if top_k < conf_vals.numel():
                    threshold = torch.topk(conf_vals, top_k).values[-1]
                    valid = valid & (min_conf >= threshold)
        if valid.sum() < 100:
            continue
        ratios = old_d[valid] / new_d[valid]
        all_ratios.append(ratios)
        fw = (
            frame_weights[idx]
            if frame_weights is not None and idx < len(frame_weights)
            else 1.0
        )
        all_weights.append(ratios.new_full((ratios.numel(),), fw))

    if not all_ratios:
        return 1.0

    all_ratios = torch.cat(all_ratios)
    all_weights = torch.cat(all_weights)
    sorted_indices = all_ratios.argsort()
    sorted_weights = all_weights[sorted_indices]
    cumsum = sorted_weights.cumsum(0)
    median_idx = int((cumsum >= cumsum[-1] / 2.0).nonzero(as_tuple=False)[0].item())
    scale = float(all_ratios[sorted_indices[median_idx]].item())
    if scale < 0.01 or scale > 100:
        return 1.0
    return scale


def estimate_scale_huber(
    old_depths,
    new_depths,
    old_confs=None,
    new_confs=None,
    use_conf_weights=False,
    downsample=4,
    num_iters=20,
):
    """Estimate scale via Huber regression across all bridge frames jointly.

    Solves old_d = scale * new_d using iteratively reweighted least squares with Huber loss.
    All bridge frames are concatenated into one regression problem.
    """
    all_old = []
    all_new = []
    all_weights = []

    for idx, (old_d, new_d) in enumerate(zip(old_depths, new_depths)):
        old_d = old_d.squeeze().float()
        new_d = new_d.squeeze().float()

        # Downsample to reduce computation
        if downsample > 1:
            old_d = old_d[::downsample, ::downsample]
            new_d = new_d[::downsample, ::downsample]

        valid = (
            (old_d > 1e-6)
            & (new_d > 1e-6)
            & torch.isfinite(old_d)
            & torch.isfinite(new_d)
        )

        if valid.sum() < 10:
            continue

        all_old.append(old_d[valid])
        all_new.append(new_d[valid])

        # Per-pixel confidence weights
        if use_conf_weights and old_confs is not None and new_confs is not None:
            old_c = old_confs[idx].squeeze().float()
            new_c = new_confs[idx].squeeze().float()
            if downsample > 1:
                old_c = old_c[::downsample, ::downsample]
                new_c = new_c[::downsample, ::downsample]
            w = torch.minimum(old_c[valid], new_c[valid]).clamp(min=0.0)
            all_weights.append(w)
        else:
            all_weights.append(torch.ones_like(old_d[valid]))

    if not all_old:
        return 1.0

    old_flat = torch.cat(all_old)
    new_flat = torch.cat(all_new)
    weights = torch.cat(all_weights)

    n = old_flat.shape[0]
    if n < 10:
        return 1.0

    # Initialize scale with weighted median of ratios
    ratios = old_flat / new_flat
    scale = float(ratios.median().item())

    # Iteratively reweighted least squares with Huber loss
    # Adaptive delta: fraction of initial median absolute residual for scale invariance
    init_residuals = (old_flat - scale * new_flat).abs()
    delta = max(float(init_residuals.median().item()) * 0.5, 1e-6)
    for _ in range(num_iters):
        residuals = old_flat - scale * new_flat
        abs_res = residuals.abs()
        # Huber weights: 1 for |r| < delta, delta/|r| for |r| >= delta
        huber_w = torch.where(
            abs_res < delta, torch.ones_like(abs_res), delta / abs_res.clamp(min=1e-8)
        )
        combined_w = weights * huber_w
        # Weighted least squares: scale = sum(w * old * new) / sum(w * new^2)
        denom = (combined_w * new_flat * new_flat).sum()
        if denom < 1e-8:
            break
        scale = float(((combined_w * old_flat * new_flat).sum() / denom).item())

    if scale < 0.01 or scale > 100:
        return 1.0
    return scale


def estimate_scale_from_depth(
    mode,
    old_depths,
    new_depths,
    old_confs=None,
    new_confs=None,
    frame_weights=None,
    conf_threshold=1.05,
    ransac_iters=200,
    ransac_inlier_ratio_thresh=0.2,
):
    """Estimate scale from depth pairs using the specified mode.

    Dispatches to weighted_median, huber, or RANSAC depending on *mode*.

    Args:
        mode: one of "weighted_median", "huber", "huber_conf", "ransac" (default)
        old_depths: list of depth tensors from before re-run
        new_depths: list of depth tensors from after re-run
        old_confs: list of depth confidence tensors (optional)
        new_confs: list of depth confidence tensors (optional)
        frame_weights: unused, kept for call-site compatibility
        conf_threshold: min confidence to include a pixel (default 1.05)
        ransac_iters: number of RANSAC iterations
        ransac_inlier_ratio_thresh: relative inlier threshold around candidate scale
    Returns:
        scale factor (float), 1.0 if estimation fails
    """
    # Dispatch to mode-specific implementation
    if mode == "weighted_median":
        return estimate_scale_weighted_median(
            old_depths, new_depths, old_confs, new_confs, frame_weights
        )
    elif mode == "huber":
        return estimate_scale_huber(
            old_depths, new_depths, old_confs, new_confs, use_conf_weights=False
        )
    elif mode == "huber_conf":
        return estimate_scale_huber(
            old_depths, new_depths, old_confs, new_confs, use_conf_weights=True
        )
    # else: fall through to existing RANSAC code below

    all_old = []
    all_new = []
    for idx, (old_d, new_d) in enumerate(zip(old_depths, new_depths)):
        old_d = old_d.squeeze().float()
        new_d = new_d.squeeze().float()
        valid = (
            (old_d > 1e-6)
            & (new_d > 1e-6)
            & torch.isfinite(old_d)
            & torch.isfinite(new_d)
        )
        # Filter by new_conf only — old conf is unreliable (fallback was triggered by low confidence)
        if (
            new_confs is not None
            and idx < len(new_confs)
            and new_confs[idx] is not None
        ):
            new_c = new_confs[idx].squeeze().float()
            valid = valid & (new_c > conf_threshold)
        if valid.sum() < 10:
            continue
        all_old.append(old_d[valid])
        all_new.append(new_d[valid])

    if not all_old:
        return 1.0

    old_flat = torch.cat(all_old)
    new_flat = torch.cat(all_new)
    n = old_flat.shape[0]
    if n < 10:
        return 1.0

    # RANSAC: fit old_d = s * new_d (single parameter, sample one pixel per iteration)
    ratios = old_flat / new_flat

    best_scale = 1.0
    best_inlier_count = 0

    num_samples = min(ransac_iters, n)
    sample_indices = (
        torch.randperm(n)[:num_samples]
        if n <= ransac_iters
        else torch.randint(0, n, (num_samples,))
    )

    for idx_t in sample_indices:
        s_candidate = float(ratios[idx_t].item())
        if s_candidate < 0.01 or s_candidate > 100:
            continue
        # Inlier = pixel where log-ratio deviation < threshold (symmetric in scale direction)
        log_deviation = (ratios / s_candidate).clamp(min=1e-6).log().abs()
        inlier_mask = log_deviation < ransac_inlier_ratio_thresh
        inlier_count = int(inlier_mask.sum().item())
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            # Refine: least squares on inliers (old = s * new → s = sum(old*new) / sum(new^2))
            old_in = old_flat[inlier_mask]
            new_in = new_flat[inlier_mask]
            denom = (new_in * new_in).sum()
            if denom > 1e-8:
                best_scale = float(((old_in * new_in).sum() / denom).item())
            else:
                best_scale = s_candidate

    if best_scale < 0.01 or best_scale > 100:
        return 1.0
    return best_scale


def resolve_fallback_scale(pose_scale, depth_scale):
    """Prefer the depth-derived scale unless pose agrees closely enough to sharpen it."""
    pose_scale = float(pose_scale)
    if depth_scale is None:
        return pose_scale

    depth_scale = float(depth_scale)
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        return pose_scale
    if not math.isfinite(pose_scale) or pose_scale <= 0.0:
        return depth_scale

    # Treat the estimator's 1.0 output as a weak pose signal when depth is available.
    if pose_scale == 1.0:
        return depth_scale

    disagreement_ratio = max(pose_scale, depth_scale) / max(
        min(pose_scale, depth_scale), 1e-8
    )
    if disagreement_ratio > 1.25:
        return depth_scale

    return math.sqrt(pose_scale * depth_scale)


def fallback_improves_bridge_scores(old_scores, new_scores):
    """Require the fallback rerun to improve bridge confidence before accepting it."""
    if not old_scores or not new_scores or len(old_scores) != len(new_scores):
        return False
    old_mean = sum(old_scores) / len(old_scores)
    new_mean = sum(new_scores) / len(new_scores)
    return new_mean >= old_mean and new_scores[-1] >= old_scores[-1]


def _stack_per_frame_tensors(tensors, view_dim):
    """Stack per-frame tensors along the view dimension for batched metric inference."""
    if isinstance(tensors, torch.Tensor):
        return tensors

    normalized = []
    for tensor in tensors:
        if tensor is None:
            return None
        while tensor.dim() < view_dim - 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == view_dim - 1:
            tensor = tensor.unsqueeze(1)
        normalized.append(tensor)
    if not normalized:
        return None
    return torch.cat(normalized, dim=1)


def compute_metric_scale_factor(
    metric_model, image, pred_depth, pred_conf, metric_min_conf
):
    """Run DA3-metric and return median(metric / pred) pooled across one or more frames.

    Inputs:
      metric_model: the frozen DA3 metric network, or None if metric anchoring is disabled.
      image: [B, N, 3, H, W], [B, 3, H, W], or a list of per-frame tensors.
      pred_depth: scale-1.0 depth prediction(s), tensor or list.
      pred_conf: corresponding confidence map(s), tensor/list or None.
      metric_min_conf: minimum per-pixel confidence required to count the pixel.
    Returns a float scale factor, or None when no pixels pass the validity mask.
    """
    if metric_model is None:
        return None

    image = _stack_per_frame_tensors(image, view_dim=5)
    if image is None:
        return None
    if image.dim() == 4:
        image = image.unsqueeze(1)

    pred_depth = _stack_per_frame_tensors(pred_depth, view_dim=5)
    if pred_depth is None:
        return None
    pred = pred_depth.squeeze(-1) if pred_depth.dim() == 5 else pred_depth
    target_device = pred.device

    pred_conf = (
        _stack_per_frame_tensors(pred_conf, view_dim=4)
        if pred_conf is not None
        else None
    )
    conf = pred_conf.to(device=target_device) if pred_conf is not None else None

    metric_out = metric_model(image)
    metric_depth = (
        metric_out.depth if hasattr(metric_out, "depth") else metric_out["depth"]
    )
    metric_depth = metric_depth.to(device=target_device)

    pred_flat = pred.float().reshape(-1)
    metric_flat = metric_depth.float().reshape(-1)
    conf_flat = conf.float().reshape(-1) if conf is not None else None

    if metric_flat.numel() != pred_flat.numel():
        # Metric and main models run at the same resolution — mismatch signals a bug we
        # want to see rather than silently alias pixels.
        raise ValueError(
            f"metric depth shape {tuple(metric_depth.shape)} does not match predicted depth "
            f"{tuple(pred.shape)}; cannot align pixel-wise."
        )

    eps = 1e-4
    valid = (metric_flat > eps) & (pred_flat > eps)
    if conf_flat is not None:
        valid &= conf_flat > metric_min_conf

    if valid.sum().item() == 0:
        return None

    ratio = metric_flat[valid] / pred_flat[valid]
    # Clip the extreme 5% tails to guard against a few stray high-ratio pixels (often
    # near saturated/sky regions that slip past the conf mask).
    lo, hi = torch.quantile(ratio, torch.tensor([0.05, 0.95], device=ratio.device))
    trimmed = ratio[(ratio >= lo) & (ratio <= hi)]
    if trimmed.numel() == 0:
        return None
    return float(trimmed.median().item())
