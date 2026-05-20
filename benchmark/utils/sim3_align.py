"""
SIM(3) alignment utility functions.
Used for inter-chunk alignment of long-sequence models (VGGT-Long, Pi-Long, DA3-Streaming).
Core functions extracted from VGGT-Long/loop_utils/sim3utils.py, with numba/loop closure dependencies removed.
"""
import numpy as np


def estimate_sim3(source_points, target_points):
    """Estimate SIM(3) transform: source -> target.

    Use SVD to solve for s, R, t such that target ≈ s * R @ source + t.

    Args:
        source_points: (M, 3) source points
        target_points: (M, 3) target points

    Returns:
        s: float, scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt((src_centered ** 2).sum(axis=1).mean())
    scale_tgt = np.sqrt((tgt_centered ** 2).sum(axis=1).mean())
    s = scale_tgt / (scale_src + 1e-12)

    src_scaled = src_centered * s

    H = src_scaled.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def weighted_estimate_sim3(source_points, target_points, weights):
    """Weighted SIM(3) estimation: source -> target.

    Args:
        source_points: (M, 3)
        target_points: (M, 3)
        weights: (M,) weights

    Returns:
        s, R, t
    """
    w = weights / (weights.sum() + 1e-12)

    mu_src = (w[:, None] * source_points).sum(axis=0)
    mu_tgt = (w[:, None] * target_points).sum(axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt((w * (src_centered ** 2).sum(axis=1)).sum())
    scale_tgt = np.sqrt((w * (tgt_centered ** 2).sum(axis=1)).sum())
    s = scale_tgt / (scale_src + 1e-12)

    src_scaled = src_centered * s

    H = (w[:, None] * src_scaled).T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def align_overlapping_chunks(point_map1, conf1, point_map2, conf2,
                             conf_threshold=None, use_weighted=True):
    """Align point clouds of two overlapping chunks: point_map2 → point_map1 coordinate frame.

    Args:
        point_map1: (B, H, W, 3) world-coordinate points of the overlapping region in the first chunk
        conf1: (B, H, W) confidence
        point_map2: (B, H, W, 3) world-coordinate points of the overlapping region in the second chunk
        conf2: (B, H, W) confidence
        conf_threshold: confidence threshold; computed automatically when None
        use_weighted: whether to use weighted SIM3

    Returns:
        s, R, t: SIM(3) transform parameters
    """
    b = min(point_map1.shape[0], point_map2.shape[0])

    if conf_threshold is None:
        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

    aligned_pts1 = []
    aligned_pts2 = []
    all_weights = []

    for i in range(b):
        mask1 = conf1[i] > conf_threshold
        mask2 = conf2[i] > conf_threshold
        valid_mask = mask1 & mask2

        # Exclude NaN/Inf points
        valid_mask = valid_mask & np.isfinite(point_map1[i]).all(axis=-1)
        valid_mask = valid_mask & np.isfinite(point_map2[i]).all(axis=-1)

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        pts1 = point_map1[i][idx]
        pts2 = point_map2[i][idx]

        aligned_pts1.append(pts1)
        aligned_pts2.append(pts2)

        if use_weighted:
            combined_conf = np.sqrt(conf1[i][idx] * conf2[i][idx])
            all_weights.append(combined_conf)

    if len(aligned_pts1) == 0:
        print("[WARNING] No matching point pairs found, using identity transform")
        return 1.0, np.eye(3), np.zeros(3)

    all_pts1 = np.concatenate(aligned_pts1, axis=0)
    all_pts2 = np.concatenate(aligned_pts2, axis=0)

    print(f"  [SIM3] {all_pts1.shape[0]} corresponding points")

    if use_weighted and all_weights:
        weights = np.concatenate(all_weights, axis=0)
        s, R, t = weighted_estimate_sim3(all_pts2, all_pts1, weights)
    else:
        s, R, t = estimate_sim3(all_pts2, all_pts1)

    print(f"  [SIM3] scale={s:.4f}")
    return s, R, t


def accumulate_sim3_transforms(transforms):
    """Accumulate adjacent SIM(3) transforms into transforms from frame 0 to each frame.

    Args:
        transforms: list of (s, R, t) tuples

    Returns:
        list of cumulative (s, R, t)
    """
    if not transforms:
        return []

    cumulative = [transforms[0]]

    for i in range(1, len(transforms)):
        s_prev, R_prev, t_prev = cumulative[i - 1]
        s_next, R_next, t_next = transforms[i]
        R_new = R_prev @ R_next
        s_new = s_prev * s_next
        t_new = s_prev * (R_prev @ t_next) + t_prev
        cumulative.append((s_new, R_new, t_new))

    return cumulative


def apply_sim3_to_points(point_maps, s, R, t):
    """Apply SIM(3) transform to a point cloud.

    Args:
        point_maps: (B, H, W, 3) or (N, 3)
        s: scale
        R: (3, 3)
        t: (3,)

    Returns:
        transformed point cloud with the same shape as the input
    """
    original_shape = point_maps.shape
    pts = point_maps.reshape(-1, 3)
    transformed = s * (R @ pts.T).T + t
    return transformed.reshape(original_shape)


def apply_sim3_to_c2w(c2w_44, s, R, t):
    """Apply SIM(3) transform to cam2world 4x4 pose matrices.

    Args:
        c2w_44: (N, 4, 4) cam2world homogeneous matrices
        s: scale
        R: (3, 3)
        t: (3,)

    Returns:
        (N, 4, 4) transformed cam2world
    """
    N = c2w_44.shape[0]
    S = np.eye(4, dtype=np.float64)
    S[:3, :3] = s * R
    S[:3, 3] = t

    result = np.zeros_like(c2w_44)
    for i in range(N):
        transformed = S @ c2w_44[i]
        # Normalize the rotation part (remove scale)
        transformed[:3, :3] /= s
        result[i] = transformed
    return result


def depth_to_world_points(depth, intrinsics, extrinsics_w2c, device=None):
    """Compute world-coordinate points from depth map + intrinsics + extrinsics (w2c).

    Args:
        depth: (N, H, W) numpy
        intrinsics: (N, 3, 3) numpy
        extrinsics_w2c: (N, 3, 4) numpy, world-to-camera

    Returns:
        world_points: (N, H, W, 3) numpy
    """
    import torch as _torch

    input_is_numpy = isinstance(depth, np.ndarray)
    if input_is_numpy:
        depth_t = _torch.tensor(depth, dtype=_torch.float32)
        K_t = _torch.tensor(intrinsics, dtype=_torch.float32)
        ext_t = _torch.tensor(extrinsics_w2c, dtype=_torch.float32)
    else:
        depth_t, K_t, ext_t = depth, intrinsics, extrinsics_w2c

    if device is not None:
        depth_t = depth_t.to(device)
        K_t = K_t.to(device)
        ext_t = ext_t.to(device)

    dev = depth_t.device
    N, H, W = depth_t.shape

    u = _torch.arange(W, device=dev).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = _torch.arange(H, device=dev).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = _torch.ones((N, H, W, 1), device=dev)
    pixel_coords = _torch.cat([u, v, ones], dim=-1)  # (N, H, W, 3)

    K_inv = _torch.inverse(K_t)
    camera_coords = _torch.einsum("nij,nhwj->nhwi", K_inv, pixel_coords)
    camera_coords = camera_coords * depth_t.unsqueeze(-1)
    camera_coords_homo = _torch.cat([camera_coords, ones], dim=-1)

    ext_44 = _torch.zeros(N, 4, 4, device=dev)
    ext_44[:, :3, :4] = ext_t
    ext_44[:, 3, 3] = 1.0
    c2w = _torch.inverse(ext_44)

    world_coords = _torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    world_points = world_coords[..., :3]

    if input_is_numpy:
        return world_points.cpu().numpy()
    return world_points
