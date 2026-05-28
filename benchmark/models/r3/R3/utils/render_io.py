"""Loaders for the standard run-output layout (camera/, color/, conf/, depth/)."""

import glob
import os

import imageio.v2 as iio
import numpy as np


def load_frames(data_dir):
    """Scan depth/ and return sorted integer frame IDs."""
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth", "*.npy")))
    return [int(os.path.splitext(os.path.basename(f))[0]) for f in depth_files]


def load_frame(data_dir, frame_id):
    """Load one frame's depth, conf, color, pose, intrinsics."""
    tag = f"{frame_id:06d}"
    depth = np.load(os.path.join(data_dir, "depth", f"{tag}.npy"))
    conf = np.load(os.path.join(data_dir, "conf", f"{tag}.npy"))
    color = iio.imread(os.path.join(data_dir, "color", f"{tag}.png"))
    cam = np.load(os.path.join(data_dir, "camera", f"{tag}.npz"))
    pose = cam["pose"]
    if pose.shape == (3, 4):
        pose_4x4 = np.eye(4, dtype=pose.dtype)
        pose_4x4[:3, :] = pose
        pose = pose_4x4
    return {"depth": depth, "conf": conf, "color": color, "pose": pose, "intrinsics": cam["intrinsics"]}
