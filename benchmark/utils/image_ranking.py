"""
Camera pose distance and ranking utilities.
Extracted from dataloader/utils/image_ranking.py, used for benchmark frame selection.
"""
import numpy as np


def rotation_angle(R1, R2):
    R = R1.T @ R2
    val = (np.trace(R) - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    angle_rad = np.arccos(val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def rotation_angle_batch(R1, R2):
    R1_t = np.transpose(R1, (0, 2, 1))[:, np.newaxis, :, :]
    R2_b = R2[np.newaxis, :, :, :]
    R_mult = np.matmul(R1_t, R2_b)
    trace_vals = R_mult[..., 0, 0] + R_mult[..., 1, 1] + R_mult[..., 2, 2]
    val = (trace_vals - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    angle_rad = np.arccos(val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg / 180.0


def extrinsic_distance_batch(extrinsics, lambda_t=1.0):
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    rot_diff = rotation_angle_batch(R, R)
    t_i = t[:, np.newaxis, :]
    t_j = t[np.newaxis, :, :]
    trans_diff = np.linalg.norm(t_i - t_j, axis=2)
    dists = rot_diff + lambda_t * trans_diff
    return dists


def rotation_angle_batch_chunked(R, chunk_size):
    N = R.shape[0]
    rot_diff = np.empty((N, N), dtype=np.float32)
    R_t = R.transpose(0, 2, 1)

    for i_start in range(0, N, chunk_size):
        i_end = min(N, i_start + chunk_size)
        R_i_t = R_t[i_start:i_end]
        for j_start in range(0, N, chunk_size):
            j_end = min(N, j_start + chunk_size)
            R_j = R[j_start:j_end]
            R_mult = R_i_t[:, np.newaxis, :, :] @ R_j[np.newaxis, :, :, :]
            trace_vals = R_mult[..., 0, 0] + R_mult[..., 1, 1] + R_mult[..., 2, 2]
            val = (trace_vals - 1.0) / 2.0
            np.clip(val, -1.0, 1.0, out=val)
            angle_rad = np.arccos(val)
            angle_deg = np.degrees(angle_rad)
            block_rot_diff = angle_deg / 180.0
            rot_diff[i_start:i_end, j_start:j_end] = block_rot_diff.astype(np.float32)
    return rot_diff


def extrinsic_distance_batch_chunked(extrinsics, lambda_t=1.0, chunk_size=1000):
    R = extrinsics[:, :3, :3].astype(np.float32)
    t = extrinsics[:, :3, 3].astype(np.float32)
    N = R.shape[0]
    rot_diff = rotation_angle_batch_chunked(R, chunk_size)
    dists = np.empty((N, N), dtype=np.float32)
    for i_start in range(0, N, chunk_size):
        i_end = min(N, i_start + chunk_size)
        t_i = t[i_start:i_end]
        for j_start in range(0, N, chunk_size):
            j_end = min(N, j_start + chunk_size)
            t_j = t[j_start:j_end]
            diff = t_i[:, None, :] - t_j[None, :, :]
            trans_diff = np.linalg.norm(diff, axis=2)
            dists[i_start:i_end, j_start:j_end] = rot_diff[i_start:i_end, j_start:j_end] + lambda_t * trans_diff
    return dists
