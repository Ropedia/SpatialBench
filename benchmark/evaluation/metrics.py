"""
Evaluation metric computations: depth, camera pose, point cloud.
"""
import glob
import os
import time

import numpy as np
from itertools import combinations

# Number of parallel cKDTree workers. Under multi-process concurrency, workers=-1
# would spawn cpu_count threads per process, causing severe CPU contention.
# Default is cpu_count//4 (suitable for 4-process concurrency); can be overridden
# via the BENCHMARK_PC_WORKERS environment variable.
_KDTREE_WORKERS = int(
    os.environ.get("BENCHMARK_PC_WORKERS",
                   max(1, (os.cpu_count() or 4) // 4))
)

# ============================================================
# Dataset whitelist for point cloud evaluation + GT mesh path resolution
# ============================================================

# Only these datasets run pointcloud evaluation at medium / dense density
# Note: tanks_and_temples is temporarily excluded -- the large pred/GT scale
# mismatch makes cKDTree query degenerate, and a single-scene pointcloud eval
# often takes 300-1500s, slowing down the medium evaluation main loop.
POINTCLOUD_ELIGIBLE_DATASETS = {
    "scannetpp", "nrgbd", "dtu", "7scenes", "hiroom",
}
POINTCLOUD_ELIGIBLE_DENSITIES = {"medium", "dense"}


# Pointcloud evaluation parameters aligned with DA3 (Depth-Anything-3/src/depth_anything_3/bench).
#
# DA3 pipeline (see Depth-Anything-3/src/depth_anything_3/bench/datasets/*.eval3d and
# bench/utils.py::evaluate_3d_reconstruction):
#   1. Prediction side: per-view depth -> TSDFVolume (voxel_length, sdf_trunc) -> mesh
#              -> sample_points_uniformly(sampling_number)
#   2. GT side:  mesh -> sample_points_uniformly(sampling_number)
#   3. Only scannetpp does AABB crop (0.1m margin); other datasets do not crop
#   4. voxel_down_sample(down_sample) on both sides
#   5. KDTree chamfer -> acc/comp/overall/precision/recall/fscore
#
# Parameter source: Depth-Anything-3/src/depth_anything_3/utils/constants.py
#   SCANNETPP_*:   voxel=0.02,       sdf_trunc=0.15, max_depth=5.0,       down=0.02,       thr=0.05
#   SEVENSCENES_*: voxel=4/512,      sdf_trunc=0.04, max_depth=1e6,       down=4/512,      thr=0.05
#   HIROOM_*:      voxel=4/512,      sdf_trunc=0.04, max_depth=1e4,       down=4/512,      thr=0.05
#   ETH3D_*:       voxel=4/512*5,    sdf_trunc=0.04*5, max_depth=1e5,     down=4/512*5,    thr=0.05*5
# nrgbd / dtu are not in DA3's recon bench; here we set custom values referring to comparable scales,
# and retain the AABB crop to speed up KDTree query.
POINTCLOUD_EVAL_PARAMS = {
    "scannetpp": {"down_sample": 0.02,          "threshold": 0.05,      "crop_margin": 0.1},
    "7scenes":   {"down_sample": 4.0 / 512,     "threshold": 0.05,      "crop_margin": None},
    "hiroom":    {"down_sample": 4.0 / 512,     "threshold": 0.05,      "crop_margin": None},
    "eth3d":     {"down_sample": 4.0 / 512 * 5, "threshold": 0.05 * 5,  "crop_margin": None},
    "nrgbd":     {"down_sample": 0.02,          "threshold": 0.05,      "crop_margin": 0.1},
    "dtu":       {"down_sample": 0.01,          "threshold": 0.05,      "crop_margin": 0.1},
}
_POINTCLOUD_EVAL_DEFAULT = {"down_sample": 0.02, "threshold": 0.05, "crop_margin": 0.1}

# Prediction-side TSDF fusion parameters (aligned with DA3 constants.py). Unconfigured datasets fall back to the scannetpp default.
POINTCLOUD_FUSION_PARAMS = {
    "scannetpp": {"voxel_length": 0.02,          "sdf_trunc": 0.15,     "max_depth": 5.0,        "sampling_number": 1_000_000},
    "7scenes":   {"voxel_length": 4.0 / 512,     "sdf_trunc": 0.04,     "max_depth": 1_000_000.0, "sampling_number": 1_000_000},
    "hiroom":    {"voxel_length": 4.0 / 512,     "sdf_trunc": 0.04,     "max_depth": 10_000.0,    "sampling_number": 1_000_000},
    "eth3d":     {"voxel_length": 4.0 / 512 * 5, "sdf_trunc": 0.04 * 5, "max_depth": 100_000.0,   "sampling_number": 1_000_000},
}
_POINTCLOUD_FUSION_DEFAULT = POINTCLOUD_FUSION_PARAMS["scannetpp"]


def get_pointcloud_eval_params(source_dataset):
    """Return (down_sample, threshold, crop_margin) per dataset. Returns the default if unconfigured."""
    return POINTCLOUD_EVAL_PARAMS.get(source_dataset, _POINTCLOUD_EVAL_DEFAULT)


def get_pointcloud_fusion_params(source_dataset):
    """Return DA3-aligned TSDF fusion parameters per dataset (voxel_length / sdf_trunc / max_depth / sampling_number)."""
    return POINTCLOUD_FUSION_PARAMS.get(source_dataset, _POINTCLOUD_FUSION_DEFAULT)


def should_run_pointcloud_eval(source_dataset, tags):
    """Whether to run pointcloud evaluation for this scene.

    Returns True only when source_dataset is in POINTCLOUD_ELIGIBLE_DATASETS
    and view_density is medium / dense.
    """
    if source_dataset not in POINTCLOUD_ELIGIBLE_DATASETS:
        return False
    density = (tags or {}).get("view_density")
    return density in POINTCLOUD_ELIGIBLE_DENSITIES


def get_gt_mesh_path(source_dataset, pointcloud_root, scene_path):
    """Return the scene's GT mesh / point cloud .ply path (returns None if missing).

    Conventions:
        Point-cloud GT files are stored under:
            {pointcloud_root}/{source_dataset}/{scene}/...

        Most scene paths map directly after dropping the final sample suffix
        from scene_index.json, for example:
            7scenes/chess_seq-05/0      -> pointcloud/7scenes/chess_seq-05/*.ply
            dtu/rgbd_dtu_scan4/0        -> pointcloud/dtu/rgbd_dtu_scan4/*.ply
            nrgbd/kitchen/0             -> pointcloud/nrgbd/kitchen/*.ply
            hiroom/828738/cam_sampled_08 -> pointcloud/hiroom/828738/*.ply
    """
    if (source_dataset not in POINTCLOUD_ELIGIBLE_DATASETS or
            pointcloud_root is None or scene_path is None):
        return None

    dataset_root = os.path.join(pointcloud_root, source_dataset)
    scene_path = os.path.normpath(scene_path).strip(os.sep)
    if not scene_path or scene_path == ".":
        return None

    def _choose_ply(mesh_dir, preferred_token=None):
        if not os.path.isdir(mesh_dir):
            return None
        plys = sorted(glob.glob(os.path.join(mesh_dir, "*.ply")))
        if not plys:
            return None
        if preferred_token and not preferred_token.isdigit():
            preferred = [p for p in plys
                         if preferred_token in os.path.basename(p)]
            if preferred:
                return preferred[0]
        return plys[0]

    parts = scene_path.split(os.sep)
    candidates = [(os.path.join(dataset_root, *parts), None)]
    for end in range(len(parts) - 1, 0, -1):
        candidates.append((
            os.path.join(dataset_root, *parts[:end]),
            parts[end],
        ))

    seen = set()
    for mesh_dir, preferred_token in candidates:
        if mesh_dir in seen:
            continue
        seen.add(mesh_dir)
        ply = _choose_ply(mesh_dir, preferred_token)
        if ply is not None:
            return ply
    return None


def load_gt_pointcloud_from_mesh(mesh_path, num_points=1_000_000, seed=0):
    """Load the GT point cloud from a .ply file. If it is a triangle mesh, uniformly sample points; if it is already a point cloud, return the vertices directly.

    Args:
        mesh_path: .ply file path
        num_points: number of points to sample if it is a mesh; upper bound for downsampling if it is already a point cloud and too large
        seed: random seed for downsampling (only applies to random downsampling of existing vertices)

    Returns:
        np.ndarray (M, 3) float32
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        pts = np.asarray(pcd.points, dtype=np.float32)
    else:
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        if pts.shape[0] == 0:
            pcd = o3d.io.read_point_cloud(mesh_path)
            pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] > num_points:
            rng = np.random.RandomState(seed)
            idx = rng.choice(pts.shape[0], num_points, replace=False)
            pts = pts[idx]

    return np.ascontiguousarray(pts, dtype=np.float32)


# ============================================================
# Depth metrics
# ============================================================

def compute_depth_metrics(pred, gt, valid_mask):
    """Compute the standard depth evaluation metrics.

    Args:
        pred: (H, W) predicted depth (already aligned)
        gt: (H, W) GT depth
        valid_mask: (H, W) bool

    Returns:
        dict: abs_rel, sq_rel, rmse, log_rmse,
              delta_1.25, delta_1.25^2, delta_1.25^3
    """
    pred = np.asanyarray(pred)
    gt = np.asanyarray(gt)
    vm = np.asarray(valid_mask, dtype=bool)
    print(
        f"[metric:depth] pred shape={pred.shape} dtype={pred.dtype} | "
        f"gt shape={gt.shape} dtype={gt.dtype} | "
        f"valid_mask shape={vm.shape} valid={int(vm.sum())}/{vm.size}"
    )
    if pred.shape == vm.shape and gt.shape == vm.shape and vm.any():
        pv, gv = pred[vm], gt[vm]
        print(
            f"[metric:depth] masked pixels n={pv.size} | "
            f"pred min/max/mean={np.nanmin(pv):.6g}/{np.nanmax(pv):.6g}/{np.nanmean(pv):.6g} | "
            f"gt min/max/mean={np.nanmin(gv):.6g}/{np.nanmax(gv):.6g}/{np.nanmean(gv):.6g}"
        )

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    if len(pred_valid) == 0:
        return {k: float('nan') for k in [
            'abs_rel', 'sq_rel', 'rmse', 'log_rmse',
            'delta_1.25', 'delta_1.25^2', 'delta_1.25^3',
            'inlier_1.03', 'inlier_1.05', 'inlier_1.10',
        ]}

    # Filter out zero and negative values
    mask = (pred_valid > 1e-6) & (gt_valid > 1e-6)
    pred_valid = pred_valid[mask]
    gt_valid = gt_valid[mask]

    if len(pred_valid) == 0:
        return {k: float('nan') for k in [
            'abs_rel', 'sq_rel', 'rmse', 'log_rmse',
            'delta_1.25', 'delta_1.25^2', 'delta_1.25^3',
            'inlier_1.03', 'inlier_1.05', 'inlier_1.10',
        ]}

    # AbsRel
    abs_rel = float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid))

    # SqRel
    sq_rel = float(np.mean(((pred_valid - gt_valid) ** 2) / gt_valid))

    # RMSE
    rmse = float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2)))

    # Log RMSE
    log_rmse = float(np.sqrt(np.mean((np.log(pred_valid) - np.log(gt_valid)) ** 2)))

    # Delta thresholds
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < 1.25 ** 2))
    delta_3 = float(np.mean(ratio < 1.25 ** 3))

    # Inlier Ratio (RobustMVD): max(pred/gt, gt/pred) < τ
    inlier_1_03 = float(np.mean(ratio < 1.03))
    inlier_1_05 = float(np.mean(ratio < 1.05))
    inlier_1_10 = float(np.mean(ratio < 1.10))

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'log_rmse': log_rmse,
        'delta_1.25': delta_1,
        'delta_1.25^2': delta_2,
        'delta_1.25^3': delta_3,
        'inlier_1_03': inlier_1_03,
        'inlier_1_05': inlier_1_05,
        'inlier_1_10': inlier_1_10,
    }


# ============================================================
# TGM: Temporal Geometric Motion
# ============================================================

def compute_tgm_metric(pred_depth, gt_depth, valid_mask):
    """Temporal Geometric Motion (TGM): compares depth changes between each pair of adjacent frames.

    Formula:
        TGM = (1/(N-1)) * sum_{i=1}^{N-1} | ||d_{i+1}-d_i||_1 - ||g_{i+1}-g_i||_1 |

    Here ||.||_1 is the per-pixel mean absolute value (MAE), computed only over pixels
    that are valid in both frames, to avoid losing comparability due to differing
    counts of valid pixels. pred_depth should already be aligned to the GT scale.

    Args:
        pred_depth: (N, H, W) aligned predicted depth
        gt_depth:   (N, H, W) GT depth
        valid_mask: (N, H, W) bool

    Returns:
        dict: {'tgm': float}  returns NaN if N<2 or there are no valid adjacent frame pairs.
    """
    pred = np.asarray(pred_depth)
    gt = np.asarray(gt_depth)
    vm = np.asarray(valid_mask, dtype=bool)
    N = len(pred)
    print(
        f"[metric:tgm] pred shape={pred.shape} dtype={pred.dtype} | "
        f"gt shape={gt.shape} dtype={gt.dtype} | "
        f"valid_mask shape={vm.shape} | N={N}"
    )

    if N < 2:
        return {'tgm': float('nan')}

    diffs = []
    for i in range(N - 1):
        m = vm[i] & vm[i + 1]
        if not m.any():
            continue
        pred_delta = float(np.mean(np.abs(pred[i + 1][m] - pred[i][m])))
        gt_delta = float(np.mean(np.abs(gt[i + 1][m] - gt[i][m])))
        diffs.append(abs(pred_delta - gt_delta))

    if not diffs:
        return {'tgm': float('nan')}

    return {'tgm': float(np.mean(diffs))}


# ============================================================
# Camera pose metrics (aligned with DA3 cameras_evaluation, pure numpy)
# ============================================================

def _c2w_to_w2c(c2w):
    """(N, 3, 4) c2w -> (N, 3, 4) w2c.  R' = R^T, t' = -R^T @ t."""
    N = c2w.shape[0]
    w2c = np.zeros_like(c2w)
    for i in range(N):
        R, t = c2w[i, :3, :3], c2w[i, :3, 3]
        w2c[i, :3, :3] = R.T
        w2c[i, :3, 3] = -R.T @ t
    return w2c


def _invert_se3_batch(se3_44):
    """Invert (N, 4, 4) SE3 numpy batch."""
    out = se3_44.copy()
    R = se3_44[:, :3, :3]
    t = se3_44[:, :3, 3:]
    R_inv = R.transpose(0, 2, 1)             # (N, 3, 3)
    t_inv = -np.matmul(R_inv, t)              # (N, 3, 1)
    out[:, :3, :3] = R_inv
    out[:, :3, 3:] = t_inv
    return out


def _mat_to_quat(R):
    """Rotation matrix (N, 3, 3) -> unit quaternion (N, 4).  Shepperd method."""
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)

    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    trace = m00 + m11 + m22

    # case 0: trace > 0
    c0 = trace > 0
    if c0.any():
        s = np.sqrt(np.maximum(trace[c0] + 1.0, 1e-10)) * 2
        q[c0, 0] = 0.25 * s
        q[c0, 1] = (R[c0, 2, 1] - R[c0, 1, 2]) / s
        q[c0, 2] = (R[c0, 0, 2] - R[c0, 2, 0]) / s
        q[c0, 3] = (R[c0, 1, 0] - R[c0, 0, 1]) / s

    # case 1: m00 largest diagonal
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    if c1.any():
        s = np.sqrt(np.maximum(1.0 + m00[c1] - m11[c1] - m22[c1], 1e-10)) * 2
        q[c1, 0] = (R[c1, 2, 1] - R[c1, 1, 2]) / s
        q[c1, 1] = 0.25 * s
        q[c1, 2] = (R[c1, 0, 1] + R[c1, 1, 0]) / s
        q[c1, 3] = (R[c1, 0, 2] + R[c1, 2, 0]) / s

    # case 2: m11 largest diagonal
    c2 = ~c0 & ~c1 & (m11 > m22)
    if c2.any():
        s = np.sqrt(np.maximum(1.0 + m11[c2] - m00[c2] - m22[c2], 1e-10)) * 2
        q[c2, 0] = (R[c2, 0, 2] - R[c2, 2, 0]) / s
        q[c2, 1] = (R[c2, 0, 1] + R[c2, 1, 0]) / s
        q[c2, 2] = 0.25 * s
        q[c2, 3] = (R[c2, 1, 2] + R[c2, 2, 1]) / s

    # case 3: m22 largest diagonal
    c3 = ~c0 & ~c1 & ~c2
    if c3.any():
        s = np.sqrt(np.maximum(1.0 + m22[c3] - m00[c3] - m11[c3], 1e-10)) * 2
        q[c3, 0] = (R[c3, 1, 0] - R[c3, 0, 1]) / s
        q[c3, 1] = (R[c3, 0, 2] + R[c3, 2, 0]) / s
        q[c3, 2] = (R[c3, 1, 2] + R[c3, 2, 1]) / s
        q[c3, 3] = 0.25 * s

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norms, 1e-10)
    return q


def _rotation_angle_quat(R_gt, R_pred, eps=1e-15):
    """Quaternion-based rotation error (degrees). Matches DA3 rotation_angle."""
    q_pred = _mat_to_quat(R_pred)
    q_gt = _mat_to_quat(R_gt)
    dot2 = np.sum(q_pred * q_gt, axis=1) ** 2
    loss_q = np.clip(1.0 - dot2, eps, None)
    err_q = np.arccos(np.clip(1.0 - 2.0 * loss_q, -1.0, 1.0))
    return np.degrees(err_q)


def _translation_angle(t_gt, t_pred, eps=1e-15):
    """Translation angle error (degrees) with ambiguity handling. Matches DA3."""
    t_pred_n = t_pred / (np.linalg.norm(t_pred, axis=1, keepdims=True) + eps)
    t_gt_n = t_gt / (np.linalg.norm(t_gt, axis=1, keepdims=True) + eps)
    dot2 = np.sum(t_pred_n * t_gt_n, axis=1) ** 2
    loss_t = np.clip(1.0 - dot2, eps, None)
    err_t = np.degrees(np.arccos(np.clip(np.sqrt(1.0 - loss_t), -1.0, 1.0)))
    err_t[~np.isfinite(err_t)] = 1e6
    # ambiguity: min(angle, 180 - angle)
    err_t = np.minimum(err_t, np.abs(180.0 - err_t))
    return err_t


def _calculate_auc_np(r_error, t_error, max_threshold=30):
    """AUC via histogram bins. Matches DA3 calculate_auc_np."""
    error_matrix = np.stack([r_error, t_error], axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return float(np.mean(np.cumsum(normalized_histogram)))


def cameras_evaluation(gt_w2c, pred_w2c, num_frames):
    """Pose evaluation fully aligned with DA3 cameras_evaluation (pure numpy).

    Args:
        gt_w2c: (N, 3, 4) w2c numpy
        pred_w2c: (N, 3, 4) w2c numpy
        num_frames: int

    Returns:
        Racc_5, Tacc_5, Racc_3, Tacc_3, rError (np), tError (np)
    """
    # (N, 3, 4) -> (N, 4, 4)
    def to_44(m34):
        N = m34.shape[0]
        m44 = np.zeros((N, 4, 4), dtype=np.float64)
        m44[:, :3, :] = m34
        m44[:, 3, 3] = 1.0
        return m44

    pred_se3 = to_44(pred_w2c.astype(np.float64))
    gt_se3 = to_44(gt_w2c.astype(np.float64))

    # build all pair indices
    pairs = np.array(list(combinations(range(num_frames), 2)))
    i1, i2 = pairs[:, 0], pairs[:, 1]

    # relative poses: inv(T_i) @ T_j
    rel_gt = np.matmul(_invert_se3_batch(gt_se3[i1]), gt_se3[i2])
    rel_pred = np.matmul(_invert_se3_batch(pred_se3[i1]), pred_se3[i2])

    rel_rangle_deg = _rotation_angle_quat(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    rel_tangle_deg = _translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])

    Racc_5 = float(np.mean(rel_rangle_deg < 5))
    Tacc_5 = float(np.mean(rel_tangle_deg < 5))
    Racc_3 = float(np.mean(rel_rangle_deg < 3))
    Tacc_3 = float(np.mean(rel_tangle_deg < 3))

    return (Racc_5, Tacc_5, Racc_3, Tacc_3,
            rel_rangle_deg.astype(np.float32),
            rel_tangle_deg.astype(np.float32))


def compute_pose_metrics(pred_w2c, gt_w2c):
    """Compute camera pose evaluation metrics (aligned with DA3 cameras_evaluation).

    Args:
        pred_w2c: (N, 3, 4) world-to-camera numpy
        gt_w2c: (N, 3, 4) world-to-camera numpy (already normalized by normalize_gt_poses)

    Returns:
        dict: racc_3, racc_5, tacc_3, tacc_5, auc_3, auc_5, auc_15, auc_30
    """
    pw = np.asarray(pred_w2c)
    gw = np.asarray(gt_w2c)
    N = len(pred_w2c)
    print(
        f"[metric:pose] pred_w2c shape={pw.shape} dtype={pw.dtype} | "
        f"gt_w2c shape={gw.shape} dtype={gw.dtype} | N={N}"
    )

    nan_result = {k: float('nan') for k in [
        'racc_3', 'racc_5', 'tacc_3', 'tacc_5',
        'auc_3', 'auc_5', 'auc_15', 'auc_30'
    ]}
    if N < 2:
        return nan_result

    try:
        Racc_5, Tacc_5, Racc_3, Tacc_3, rError, tError = cameras_evaluation(
            gt_w2c, pred_w2c, num_frames=N
        )
    except Exception:
        return nan_result

    return {
        'racc_3': float(Racc_3),
        'racc_5': float(Racc_5),
        'tacc_3': float(Tacc_3),
        'tacc_5': float(Tacc_5),
        'auc_3': _calculate_auc_np(rError, tError, max_threshold=3),
        'auc_5': _calculate_auc_np(rError, tError, max_threshold=5),
        'auc_15': _calculate_auc_np(rError, tError, max_threshold=15),
        'auc_30': _calculate_auc_np(rError, tError, max_threshold=30),
    }


# ============================================================
# Fast3R official c2w pairwise pose evaluation
# Reference: fast3r/fast3r/eval/cam_pose_metric.py
# Uses c2w directly for pairwise computation; the global coordinate frame
# cancels out automatically, no GT alignment is needed
# ============================================================

def _rotation_angle_trace(R_gt, R_pred):
    """Trace-based rotation angle (degrees): acos((trace(R1 @ R2^T) - 1) / 2).
    Matches Fast3R so3_relative_angle.
    """
    R12 = np.matmul(R_gt, R_pred.transpose(0, 2, 1))  # (N, 3, 3)
    trace = np.trace(R12, axis1=1, axis2=2)  # (N,)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def cameras_evaluation_c2w(pred_c2w, gt_c2w, num_frames):
    """Fast3R official c2w pairwise pose evaluation (pure numpy).

    Aligned with camera_to_rel_deg in fast3r/fast3r/eval/cam_pose_metric.py.
    Uses c2w directly for relative pose: rel = inv(c2w_i) @ c2w_j.
    The global coordinate frame cancels out automatically, so GT need not be aligned to the first frame.

    Args:
        pred_c2w: (N, 3, 4) cam2world numpy
        gt_c2w: (N, 3, 4) cam2world numpy (raw GT, no normalization needed)
        num_frames: int

    Returns:
        Racc_5, Tacc_5, Racc_3, Tacc_3, rError (np), tError (np)
    """
    def to_44(m34):
        N = m34.shape[0]
        m44 = np.zeros((N, 4, 4), dtype=np.float64)
        m44[:, :3, :] = m34
        m44[:, 3, 3] = 1.0
        return m44

    pred_se3 = to_44(pred_c2w.astype(np.float64))
    gt_se3 = to_44(gt_c2w.astype(np.float64))

    # all pairs
    pairs = np.array(list(combinations(range(num_frames), 2)))
    i1, i2 = pairs[:, 0], pairs[:, 1]

    # relative poses: inv(c2w_i) @ c2w_j  (global coordinate frame cancels out)
    rel_gt = np.matmul(_invert_se3_batch(gt_se3[i1]), gt_se3[i2])
    rel_pred = np.matmul(_invert_se3_batch(pred_se3[i1]), pred_se3[i2])

    # rotation: trace-based (matches Fast3R so3_relative_angle)
    rel_rangle_deg = _rotation_angle_trace(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    # translation: direction angle (matches Fast3R compare_translation_by_angle)
    rel_tangle_deg = _translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])

    Racc_5 = float(np.mean(rel_rangle_deg < 5))
    Tacc_5 = float(np.mean(rel_tangle_deg < 5))
    Racc_3 = float(np.mean(rel_rangle_deg < 3))
    Tacc_3 = float(np.mean(rel_tangle_deg < 3))

    return (Racc_5, Tacc_5, Racc_3, Tacc_3,
            rel_rangle_deg.astype(np.float32),
            rel_tangle_deg.astype(np.float32))


def compute_pose_metrics_c2w(pred_c2w, gt_c2w):
    """Fast3R official c2w pose evaluation metrics.

    Args:
        pred_c2w: (N, 3, 4) cam2world numpy
        gt_c2w: (N, 3, 4) cam2world numpy (raw GT)

    Returns:
        dict: racc_3, racc_5, tacc_3, tacc_5, auc_3, auc_5, auc_15, auc_30
    """
    pc = np.asarray(pred_c2w)
    gc = np.asarray(gt_c2w)
    N = len(pred_c2w)
    print(
        f"[metric:pose_c2w] pred_c2w shape={pc.shape} dtype={pc.dtype} | "
        f"gt_c2w shape={gc.shape} dtype={gc.dtype} | N={N}"
    )

    nan_result = {k: float('nan') for k in [
        'racc_3', 'racc_5', 'tacc_3', 'tacc_5',
        'auc_3', 'auc_5', 'auc_15', 'auc_30'
    ]}
    if N < 2:
        return nan_result

    try:
        Racc_5, Tacc_5, Racc_3, Tacc_3, rError, tError = cameras_evaluation_c2w(
            pred_c2w, gt_c2w, num_frames=N
        )
    except Exception:
        return nan_result

    return {
        'racc_3': float(Racc_3),
        'racc_5': float(Racc_5),
        'tacc_3': float(Tacc_3),
        'tacc_5': float(Tacc_5),
        'auc_3': _calculate_auc_np(rError, tError, max_threshold=3),
        'auc_5': _calculate_auc_np(rError, tError, max_threshold=5),
        'auc_15': _calculate_auc_np(rError, tError, max_threshold=15),
        'auc_30': _calculate_auc_np(rError, tError, max_threshold=30),
    }


# ============================================================
# Trajectory metrics: ATE, RPEt, RPEr (after Sim(3) alignment)
# Evaluated via the evo library, matching the DROID-SLAM (Teed & Deng, 2021) protocol
# Reference: https://github.com/princeton-vl/DROID-SLAM/issues/156
#   - Use PoseRelation.rotation_angle_deg to correctly compute rotation error
# ============================================================

def _c2w_to_se3_list(c2w):
    """(N, 3, 4) c2w -> list of (4, 4) SE3 numpy matrices."""
    poses = []
    for i in range(len(c2w)):
        m = np.eye(4, dtype=np.float64)
        m[:3, :] = c2w[i]
        poses.append(m)
    return poses


def compute_trajectory_metrics(pred_c2w, gt_c2w):
    """Compute trajectory metrics after Sim(3) alignment: ATE, RPEt, RPEr.

    Uses the evo library (DROID-SLAM evaluation standard):
    - ATE: APE translation_part (RMSE), Sim(3) aligned
    - RPE_t: RPE translation_part (mean), delta=1 frame
    - RPE_r: RPE rotation_angle_deg (mean), delta=1 frame
      (cf. DROID-SLAM issue #156, using evo's rotation_angle_deg)

    Args:
        pred_c2w: (N, 3, 4) cam-to-world numpy
        gt_c2w: (N, 3, 4) cam-to-world numpy

    Returns:
        dict: ate, rpe_t, rpe_r
    """
    nan_result = {k: float('nan') for k in ['ate', 'rpe_t', 'rpe_r']}
    pp = np.asarray(pred_c2w)
    gg = np.asarray(gt_c2w)
    N = len(pred_c2w)
    if N >= 2:
        p_t = pp[:, :3, 3].astype(np.float64)
        g_t = gg[:, :3, 3].astype(np.float64)
        print(
            f"[metric:trajectory] pred_c2w shape={pp.shape} dtype={pp.dtype} | "
            f"gt_c2w shape={gg.shape} dtype={gg.dtype} | N={N} | "
            f"pred t min/max/mean={p_t.min():.6g}/{p_t.max():.6g}/{p_t.mean():.6g} | "
            f"gt t min/max/mean={g_t.min():.6g}/{g_t.max():.6g}/{g_t.mean():.6g}"
        )
    else:
        print(
            f"[metric:trajectory] pred_c2w shape={pp.shape} dtype={pp.dtype} | "
            f"gt_c2w shape={gg.shape} dtype={gg.dtype} | N={N} (skip, N<2)"
        )

    if N < 2:
        return nan_result

    try:
        import copy
        from evo.core.trajectory import PosePath3D
        from evo.core import metrics
        from evo.core.metrics import PoseRelation, StatisticsType, Unit

        traj_pred = PosePath3D(poses_se3=_c2w_to_se3_list(pred_c2w))
        traj_gt = PosePath3D(poses_se3=_c2w_to_se3_list(gt_c2w))

        # Sim(3) alignment: Umeyama align + scale correction
        traj_pred_aligned = copy.deepcopy(traj_pred)
        traj_pred_aligned.align(traj_gt, correct_scale=True)

        # ATE: APE translation_part, RMSE
        ape_metric = metrics.APE(PoseRelation.translation_part)
        ape_metric.process_data((traj_gt, traj_pred_aligned))
        ate = ape_metric.get_statistic(StatisticsType.rmse)

        # RPE translation: delta=1 frame, mean
        rpe_t_metric = metrics.RPE(
            PoseRelation.translation_part,
            delta=1, delta_unit=Unit.frames, all_pairs=False)
        rpe_t_metric.process_data((traj_gt, traj_pred_aligned))
        rpe_t = rpe_t_metric.get_statistic(StatisticsType.mean)

        # RPE rotation: rotation_angle_deg, delta=1 frame, mean
        rpe_r_metric = metrics.RPE(
            PoseRelation.rotation_angle_deg,
            delta=1, delta_unit=Unit.frames, all_pairs=False)
        rpe_r_metric.process_data((traj_gt, traj_pred_aligned))
        rpe_r = rpe_r_metric.get_statistic(StatisticsType.mean)

        return {
            'ate': float(ate),
            'rpe_t': float(rpe_t),
            'rpe_r': float(rpe_r),
        }
    except ImportError:
        # Fall back to the numpy implementation when evo is not installed
        return _compute_trajectory_metrics_numpy(pred_c2w, gt_c2w)
    except Exception:
        # evo's Umeyama alignment throws GeometryException on degenerate trajectories
        # (coplanar/colinear); fall back to the numpy implementation.
        return _compute_trajectory_metrics_numpy(pred_c2w, gt_c2w)


def _compute_trajectory_metrics_numpy(pred_c2w, gt_c2w):
    """numpy fallback (used when evo is not installed)."""
    nan_result = {k: float('nan') for k in ['ate', 'rpe_t', 'rpe_r']}
    N = len(pred_c2w)
    if N < 2:
        return nan_result

    try:
        pred_pos = pred_c2w[:, :3, 3].astype(np.float64)
        gt_pos = gt_c2w[:, :3, 3].astype(np.float64)

        # Umeyama Sim(3) alignment
        pred, gt = pred_pos, gt_pos
        mu_p, mu_g = pred.mean(0), gt.mean(0)
        pc, gc = pred - mu_p, gt - mu_g
        H = pc.T @ gc / N
        U, D, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1.0, 1.0, np.sign(d)])
        R_align = Vt.T @ S @ U.T
        var_p = np.sum(pc ** 2) / N
        s = np.trace(np.diag(D) @ S) / max(var_p, 1e-10)
        t_align = mu_g - s * R_align @ mu_p

        # Aligned c2w
        pred_64 = pred_c2w.astype(np.float64)
        aligned = np.zeros_like(pred_64)
        for i in range(N):
            aligned[i, :3, :3] = R_align @ pred_64[i, :3, :3]
            aligned[i, :3, 3] = s * R_align @ pred_64[i, :3, 3] + t_align

        # ATE
        ate = float(np.sqrt(np.mean(np.sum(
            (aligned[:, :3, 3] - gt_pos) ** 2, axis=1))))

        # RPE (consecutive frames)
        def to_44(m34):
            m44 = np.zeros((m34.shape[0], 4, 4), dtype=np.float64)
            m44[:, :3, :] = m34
            m44[:, 3, 3] = 1.0
            return m44

        a44 = to_44(aligned)
        g44 = to_44(gt_c2w.astype(np.float64))
        rpe_t_list, rpe_r_list = [], []
        for i in range(N - 1):
            rel_p = np.linalg.inv(a44[i]) @ a44[i + 1]
            rel_g = np.linalg.inv(g44[i]) @ g44[i + 1]
            rpe = np.linalg.inv(rel_g) @ rel_p
            rpe_t_list.append(np.linalg.norm(rpe[:3, 3]))
            cos_a = np.clip((np.trace(rpe[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            rpe_r_list.append(float(np.degrees(np.arccos(cos_a))))

        return {
            'ate': ate,
            'rpe_t': float(np.mean(rpe_t_list)),
            'rpe_r': float(np.mean(rpe_r_list)),
        }
    except Exception:
        return nan_result


# ============================================================
# Point cloud metrics
# ============================================================

def _voxel_downsample_np(points, voxel_size):
    """Pure numpy voxel downsampling: each voxel keeps the mean of its interior points.

    Very close to open3d.voxel_down_sample but does not depend on Open3D.
    """
    if voxel_size is None or voxel_size <= 0 or len(points) == 0:
        return np.ascontiguousarray(points, dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32)
    keys = np.floor(pts / float(voxel_size)).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    pts_sorted = pts[order]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    sums = np.add.reduceat(pts_sorted, offsets[:-1], axis=0)
    return (sums / counts[:, None]).astype(np.float32)


def compute_pointcloud_metrics(pred_pc, gt_pc, threshold=0.05,
                               down_sample=0.02, crop_margin=None):
    """3D reconstruction metrics (strictly aligned with the DA3 evaluate_3d_reconstruction + scannetpp.eval3d protocol).

    Pipeline:
      1. (Only when crop_margin is not None) AABB-crop pred to the GT bounding box +/- crop_margin.
         DA3 only does this step in scannetpp.eval3d; other datasets' eval3d use the shared
         evaluate_3d_reconstruction with no crop; therefore crop_margin defaults to None.
      2. voxel_down_sample(down_sample) is applied to both pred and GT.
      3. Bidirectional nearest-neighbor chamfer via cKDTree (workers=_KDTREE_WORKERS parallelism).
      4. For empty point clouds, return the DA3 sentinel: acc/comp/overall=+inf, precision/recall/fscore=0.

    Args:
        pred_pc: (M, 3) predicted point cloud (already aligned to the GT frame, usually sampled from a TSDF mesh)
        gt_pc: (K, 3) GT point cloud (uniformly sampled from the GT mesh)
        threshold: F-score distance threshold (meters); see POINTCLOUD_EVAL_PARAMS[...].threshold
        down_sample: voxel downsampling cell size (meters); None/0 = skip
        crop_margin: AABB crop margin (meters); None = skip crop (DA3 non-scannetpp behavior)

    Returns:
        dict: {f_score, overall, acc, comp, precision, recall}
    """
    empty_result = {
        'f_score': 0.0, 'overall': float('inf'),
        'acc': float('inf'), 'comp': float('inf'),
        'precision': 0.0, 'recall': 0.0,
    }
    if len(pred_pc) == 0 or len(gt_pc) == 0:
        return empty_result

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {k: float('nan') for k in empty_result}

    pred_pc = np.ascontiguousarray(pred_pc, dtype=np.float32)
    gt_pc = np.ascontiguousarray(gt_pc, dtype=np.float32)

    t0 = time.time()
    n_pred_raw, n_gt_raw = len(pred_pc), len(gt_pc)

    # (1) AABB crop (scannetpp only; skipped for other datasets where crop_margin=None).
    #     DA3 behavior: strict crop; if the result is empty, return the sentinel directly without falling back to the raw points.
    if crop_margin is not None and crop_margin >= 0:
        lo = gt_pc.min(axis=0) - float(crop_margin)
        hi = gt_pc.max(axis=0) + float(crop_margin)
        mask = np.all((pred_pc >= lo) & (pred_pc <= hi), axis=1)
        pred_pc = pred_pc[mask]

    # (2) voxel downsample both pred and GT.
    if down_sample is not None and down_sample > 0:
        pred_pc = _voxel_downsample_np(pred_pc, down_sample)
        gt_pc = _voxel_downsample_np(gt_pc, down_sample)

    print(
        f"[metric:pointcloud] pred n={n_pred_raw}->{len(pred_pc)} | "
        f"gt n={n_gt_raw}->{len(gt_pc)} | "
        f"down_sample={down_sample} crop_margin={crop_margin} threshold={threshold}"
    )

    if len(pred_pc) == 0 or len(gt_pc) == 0:
        return empty_result

    # (3) chamfer via cKDTree; cap the thread count to avoid CPU contention under multi-process concurrency.
    tree_pred = cKDTree(pred_pc)
    tree_gt = cKDTree(gt_pc)
    dist_pred_to_gt, _ = tree_gt.query(pred_pc, workers=_KDTREE_WORKERS)
    dist_gt_to_pred, _ = tree_pred.query(gt_pc, workers=_KDTREE_WORKERS)

    accuracy = float(np.mean(dist_pred_to_gt))
    completeness = float(np.mean(dist_gt_to_pred))
    overall = (accuracy + completeness) / 2

    precision = float(np.mean(dist_pred_to_gt < threshold))
    recall = float(np.mean(dist_gt_to_pred < threshold))
    if precision + recall < 1e-8:
        f_score = 0.0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    print(f"[metric:pointcloud] elapsed={time.time()-t0:.2f}s "
          f"f_score={f_score:.4f} overall={overall:.4f}")

    return {
        'f_score': float(f_score),
        'overall': float(overall),
        'acc': float(accuracy),
        'comp': float(completeness),
        'precision': float(precision),
        'recall': float(recall),
    }


def align_pointcloud_procrustes(pred_pc, gt_pc, max_samples=10000):
    """Procrustes-align the predicted point cloud to the GT point cloud (s, R, t).

    Used in pred_pose mode, where the predicted point cloud may be in a different frame/scale.
    Subsamples a subset to compute the alignment, then applies it to all points.

    Args:
        pred_pc: (M, 3)
        gt_pc: (K, 3)
        max_samples: maximum number of sample points used to compute the alignment

    Returns:
        aligned_pc: (M, 3) aligned predicted point cloud
    """
    if len(pred_pc) == 0 or len(gt_pc) == 0:
        return pred_pc

    from scipy.spatial import cKDTree

    # Use nearest-neighbor correspondences for Procrustes
    # First subsample gt, then find corresponding points in pred
    if len(gt_pc) > max_samples:
        idx = np.random.choice(len(gt_pc), max_samples, replace=False)
        gt_sub = gt_pc[idx]
    else:
        gt_sub = gt_pc

    tree_pred = cKDTree(pred_pc)
    _, nn_idx = tree_pred.query(gt_sub, workers=_KDTREE_WORKERS)
    pred_sub = pred_pc[nn_idx]

    # Procrustes: gt ≈ s * R @ pred + t
    pred_mean = pred_sub.mean(axis=0)
    gt_mean = gt_sub.mean(axis=0)
    pred_c = pred_sub - pred_mean
    gt_c = gt_sub - gt_mean

    H = pred_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_mat @ U.T

    pred_scale = np.sqrt((pred_c ** 2).sum())
    gt_scale = np.sqrt((gt_c ** 2).sum())
    s = gt_scale / max(pred_scale, 1e-8)

    t = gt_mean - s * R @ pred_mean

    aligned_pc = (s * (pred_pc @ R.T)) + t
    return aligned_pc.astype(np.float32)


def fuse_depth_to_pointcloud(depths, extrinsics, intrinsics, images_raw,
                             voxel_length=None, sdf_trunc=None,
                             max_depth=None, num_points=None,
                             source_dataset=None,
                             save_mesh_path=None, save_pcd_path=None):
    """Multi-frame depth maps -> TSDF fusion -> mesh -> uniformly sampled point cloud (DA3-aligned).

    Reproduces the three steps from DA3 (Depth-Anything-3/src/depth_anything_3/bench/utils.py):
        create_tsdf_volume -> fuse_depth_to_tsdf -> sample_points_from_mesh

    Relies on Open3D's ``o3d.pipelines.integration.ScalableTSDFVolume``. In Open3D 0.17,
    ``integrate`` segfaults in some environments; upgrade to 0.18+.

    When parameters are not explicitly passed, the original DA3 values are taken from
    POINTCLOUD_FUSION_PARAMS by source_dataset; if not configured, fall back to the
    scannetpp default (voxel=0.02, sdf_trunc=0.15, max_depth=5.0).

    Args:
        depths: (N, H, W) float32 depth maps, already aligned to metric scale (meters)
        extrinsics: (N, 3, 4) or (N, 4, 4) cam2world (internally inverted to world2cam)
        intrinsics: (N, 3, 3) or (3, 3) (the latter is broadcast to N)
        images_raw: color images, several forms supported
            - torch.Tensor / np.ndarray (N, 3, H, W) float [0, 1]
            - np.ndarray (N, H, W, 3) uint8 [0, 255]
            - None: use a grayscale placeholder image
        voxel_length / sdf_trunc / max_depth / num_points: explicit values override the per-dataset defaults
        source_dataset: determines the default TSDF parameters (see POINTCLOUD_FUSION_PARAMS)
        save_mesh_path: if not None, save the TSDF-extracted triangle mesh to this .ply path (debug)
        save_pcd_path:  if not None, save the sampled point cloud to this .ply path (debug)

    Returns:
        np.ndarray (M, 3) float32: world-coordinate point cloud uniformly sampled from the TSDF mesh
    """
    import open3d as o3d

    fp = get_pointcloud_fusion_params(source_dataset)
    voxel_length = float(voxel_length if voxel_length is not None else fp["voxel_length"])
    sdf_trunc    = float(sdf_trunc    if sdf_trunc    is not None else fp["sdf_trunc"])
    max_depth    = float(max_depth    if max_depth    is not None else fp["max_depth"])
    num_points   = int(  num_points   if num_points   is not None else fp["sampling_number"])

    depths = np.asarray(depths, dtype=np.float32)
    if depths.ndim != 3:
        return np.zeros((0, 3), dtype=np.float32)
    N, Hd, Wd = depths.shape

    # Extrinsics: cam2world (N,3,4)/(N,4,4) -> world2cam (N,4,4) float64
    ext = np.asarray(extrinsics, dtype=np.float64)
    if ext.shape == (N, 3, 4):
        ext4 = np.broadcast_to(np.eye(4, dtype=np.float64), (N, 4, 4)).copy()
        ext4[:, :3, :] = ext
        ext = ext4
    if ext.shape != (N, 4, 4):
        raise ValueError(f"extrinsics shape {ext.shape} not in {{(N,3,4), (N,4,4)}}")
    w2c = np.linalg.inv(ext)

    # Intrinsics (3,3) -> (N,3,3)
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape == (3, 3):
        K = np.broadcast_to(K, (N, 3, 3)).copy()
    if K.shape != (N, 3, 3):
        raise ValueError(f"intrinsics shape {K.shape} not in {{(3,3), (N,3,3)}}")

    # Colors: convert to (N, H, W, 3) uint8 at the same resolution as depth
    imgs = _prepare_tsdf_colors(images_raw, N=N, Hd=Hd, Wd=Wd)

    # Build TSDF volume, integrate, extract mesh
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrated = 0
    for i in range(N):
        depth_i = depths[i]
        finite = np.isfinite(depth_i) & (depth_i > 1e-6)
        if not finite.any():
            continue
        depth_clean = np.where(finite, depth_i, 0.0).astype(np.float32)

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(imgs[i]))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_clean))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        Ki = K[i]
        ixt_o3d = o3d.camera.PinholeCameraIntrinsic(
            Wd, Hd,
            float(Ki[0, 0]), float(Ki[1, 1]),
            float(Ki[0, 2]), float(Ki[1, 2]),
        )
        volume.integrate(rgbd, ixt_o3d, w2c[i])
        integrated += 1

    if integrated == 0:
        return np.zeros((0, 3), dtype=np.float32)

    mesh = volume.extract_triangle_mesh()
    if len(mesh.triangles) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if save_mesh_path:
        os.makedirs(os.path.dirname(save_mesh_path) or ".", exist_ok=True)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(save_mesh_path, mesh)
        print(f"[fuse] saved pred mesh -> {save_mesh_path} "
              f"(V={len(mesh.vertices)} F={len(mesh.triangles)})")

    # DA3 sample_points_from_mesh: uniform sampling, with a seeded random fallback if it fails
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    except Exception:
        rng = np.random.default_rng(seed=42)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rng.uniform(-1.0, 1.0, size=(num_points, 3)))

    if save_pcd_path:
        os.makedirs(os.path.dirname(save_pcd_path) or ".", exist_ok=True)
        o3d.io.write_point_cloud(save_pcd_path, pcd)
        print(f"[fuse] saved pred pcd  -> {save_pcd_path} (N={len(pcd.points)})")

    pts = np.asarray(pcd.points, dtype=np.float32)
    return np.ascontiguousarray(pts)


def save_pointcloud_ply(points, path):
    """Save an (M,3) numpy point cloud as .ply (for debug / visualization)."""
    if points is None or len(points) == 0:
        return
    import open3d as o3d
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    o3d.io.write_point_cloud(path, pcd)
    print(f"[save_pointcloud_ply] saved {path} (N={len(points)})")


def _prepare_tsdf_colors(images_raw, N, Hd, Wd):
    """Normalize various color input forms into (N, Hd, Wd, 3) uint8."""
    if images_raw is None:
        return np.full((N, Hd, Wd, 3), 128, dtype=np.uint8)

    imgs = images_raw
    try:
        import torch
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.detach().cpu().numpy()
    except ImportError:
        pass
    imgs = np.asarray(imgs)

    # (N, 3, H, W) -> (N, H, W, 3)
    if imgs.ndim == 4 and imgs.shape[1] == 3 and imgs.shape[-1] != 3:
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    if imgs.ndim != 4 or imgs.shape[-1] != 3 or imgs.shape[0] != N:
        return np.full((N, Hd, Wd, 3), 128, dtype=np.uint8)

    # float [0,1] -> uint8 [0,255]
    if imgs.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(imgs)) <= 1.5 else 1.0
        imgs = np.clip(imgs * scale, 0.0, 255.0).astype(np.uint8)

    # Align resolution to depth
    Hi, Wi = imgs.shape[1], imgs.shape[2]
    if (Hi, Wi) != (Hd, Wd):
        try:
            import cv2
            imgs = np.stack(
                [cv2.resize(im, (Wd, Hd), interpolation=cv2.INTER_AREA) for im in imgs],
                axis=0,
            )
        except Exception:
            return np.full((N, Hd, Wd, 3), 128, dtype=np.uint8)

    return np.ascontiguousarray(imgs, dtype=np.uint8)


def unproject_to_pointcloud(depths, extrinsics, intrinsics, valid_masks):
    """Unproject depth maps into a world-coordinate point cloud.

    Args:
        depths: (N, H, W) depth maps
        extrinsics: (N, 3, 4) cam2world
        intrinsics: (N, 3, 3) intrinsics
        valid_masks: (N, H, W) bool

    Returns:
        np.ndarray (M, 3): merged world-coordinate point cloud
    """
    all_points = []
    N, H, W = depths.shape

    for i in range(N):
        mask = valid_masks[i]
        if not mask.any():
            continue

        depth = depths[i]
        K = intrinsics[i]
        pose = extrinsics[i]  # (3, 4) cam2world

        # Pixel coordinates
        v, u = np.where(mask)
        z = depth[v, u]

        # Camera coordinates
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        pts_cam = np.stack([x, y, z], axis=1)  # (M, 3)

        # World coordinates
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_world = (R @ pts_cam.T).T + t  # (M, 3)

        all_points.append(pts_world)

    if not all_points:
        return np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(all_points, axis=0).astype(np.float32)
