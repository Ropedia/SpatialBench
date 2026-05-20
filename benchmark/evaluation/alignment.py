"""
Alignment methods between predictions and GT:
- median_scale_alignment: median scaling (standard method for monocular depth)
- lstsq_alignment: least-squares affine alignment
- procrustes_alignment: Procrustes similarity transform alignment (pose)
"""
import numpy as np


def median_scale_alignment(pred_depth, gt_depth, valid_mask):
    """Median scaling alignment: scale = median(gt) / median(pred).

    Args:
        pred_depth: (H, W) predicted depth
        gt_depth: (H, W) GT depth
        valid_mask: (H, W) bool, valid region

    Returns:
        aligned_pred: (H, W) aligned predicted depth
        scale: float, scale factor
    """
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]

    if len(pred_valid) == 0 or np.median(pred_valid) < 1e-8:
        return pred_depth.copy(), 1.0

    scale = float(np.median(gt_valid) / np.median(pred_valid))
    aligned_pred = pred_depth * scale
    return aligned_pred, scale


def lstsq_alignment(pred_depth, gt_depth, valid_mask):
    """Least-squares affine alignment: gt ≈ s * pred + t.

    Args:
        pred_depth: (H, W)
        gt_depth: (H, W)
        valid_mask: (H, W) bool

    Returns:
        aligned_pred: (H, W)
        s: float, scale
        t: float, offset
    """
    pred_valid = pred_depth[valid_mask].flatten()
    gt_valid = gt_depth[valid_mask].flatten()

    if len(pred_valid) < 2:
        return pred_depth.copy(), 1.0, 0.0

    # Build [pred, 1] @ [s, t]^T = gt
    A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
    result = np.linalg.lstsq(A, gt_valid, rcond=None)
    s, t = result[0]

    aligned_pred = pred_depth * s + t
    return aligned_pred, float(s), float(t)


def procrustes_alignment(pred_poses, gt_poses):
    """Procrustes similarity-transform alignment: find optimal s, R, t such that gt ≈ s * R @ pred + t.

    Aligns predicted poses to GT poses (based on camera centers).

    Args:
        pred_poses: (N, 3, 4) cam2world
        gt_poses: (N, 3, 4) cam2world

    Returns:
        aligned_poses: (N, 3, 4) aligned predicted poses
        sim3: dict with 's', 'R', 't'
    """
    # Extract camera centers
    pred_centers = pred_poses[:, :3, 3].copy()  # (N, 3)
    gt_centers = gt_poses[:, :3, 3].copy()      # (N, 3)

    # Decentralize
    pred_mean = pred_centers.mean(axis=0)
    gt_mean = gt_centers.mean(axis=0)
    pred_centered = pred_centers - pred_mean
    gt_centered = gt_centers - gt_mean

    # SVD to solve for optimal rotation
    H = pred_centered.T @ gt_centered  # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T  # (3, 3)

    # Solve for scale
    pred_scale = np.sqrt((pred_centered ** 2).sum())
    gt_scale = np.sqrt((gt_centered ** 2).sum())
    if pred_scale < 1e-8:
        s = 1.0
    else:
        s = gt_scale / pred_scale

    # Solve for translation
    t = gt_mean - s * R @ pred_mean

    # Apply transform to all poses
    N = len(pred_poses)
    aligned_poses = np.zeros_like(pred_poses)
    for i in range(N):
        # Rotation part
        aligned_poses[i, :3, :3] = R @ pred_poses[i, :3, :3]
        # Translation part
        aligned_poses[i, :3, 3] = s * R @ pred_poses[i, :3, 3] + t

    sim3 = {'s': float(s), 'R': R, 't': t}
    return aligned_poses, sim3
