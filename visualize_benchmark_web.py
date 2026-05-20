#!/usr/bin/env python3
"""Web-based Benchmark GT Point Cloud Viewer.

Reads scene indices from benchmark/scene_indices/, loads GT depth + pose + intrinsics,
unprojects to 3D point cloud, and displays in an interactive Three.js viewer.

Supports filtering by dataset, view density (sparse/medium/dense), environment,
dynamics, and view type.

Usage:
    python visualize_benchmark_web.py
    python visualize_benchmark_web.py --benchmark-root SpatialBenchmark --port 8082
"""

import argparse
import io
import json
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')  # 启用 EXR 支持 (Waymo 深度图)
import struct
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark.datasets.benchmark_dataset import BenchmarkDataset
from benchmark.utils.visualization import save_pointcloud_glb

import cv2
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

# ── Config ──
SCENE_INDEX_DIR: Path = Path("benchmark/scene_indices")
SCENE_INDEX_PATH: Path = SCENE_INDEX_DIR / "all_scenes.json"
BENCHMARK_ROOT: Path = Path("SpatialBenchmark")
DATA_ROOTS: dict = {}          # Legacy placeholder; unified reader uses BENCHMARK_ROOT.
Z_FAR_DEFAULT = 10.0
RESOLUTION = (518, 378)        # (W, H) — same as benchmark configs
GLB_OUTPUT_DIR: Path = Path("glb_output")

# Per-dataset default z_far — kept in sync with DEFAULT_Z_FAR in data_readers.py
DATASET_Z_FAR: dict = {
    "droid":              1.0,   # DroidReader
    "ropedia":            5.0,   # RopediaReader
    "tum":                5.0,   # TumReader
    "nrgbd":             10.0,   # NrgbdReader
    "7scenes":           10.0,   # SevenScenesReader
    "adt":               10.0,   # AdtReader
    "robotwin":           3.0,   # RoboTwinReader
    "rlbench":            3.0,   # RLBenchReader
    "robolab":            3.0,   # RoboLabReader
    "dtu":                3.0,   # DtuReader
    "eth3d":             30.0,   # Eth3dReader
    "tanks_and_temples": 30.0,   # TanksAndTemplesReader
    "omniworld":         50.0,   # OmniworldReader
    "lingbot":           20.0,   # LingbotReader
    "hiroom":            10.0,   # HiroomReader
    "scannetpp":         10.0,   # ScannetppReader
    "spatialvid":        50.0,   # SpatialVidReader
    "vkitti":            80.0,   # VkittiReader
    "waymo":             50.0,   # WaymoReader
    "kitti_odometry":    80.0,   # KittiOdometryReader
}

app = FastAPI(title="Benchmark GT Point Cloud Viewer")

# ── Scene index cache ──
_all_scenes: list = []         # flat list of scene dicts
_dataset_names: list = []      # unique dataset names
_benchmark_dataset: BenchmarkDataset | None = None
_scene_to_dataset_idx: dict = {}


def _load_all_scenes():
    """Load all scene indices from JSON files."""
    global _all_scenes, _dataset_names
    index_path = SCENE_INDEX_PATH
    if index_path.exists():
        with open(index_path) as f:
            _all_scenes = json.load(f)
    else:
        # Fallback: merge individual files
        _all_scenes = []
        for p in sorted(SCENE_INDEX_DIR.glob("*_scenes.json")):
            if p.name == "all_scenes.json":
                continue
            with open(p) as f:
                _all_scenes.extend(json.load(f))

    _dataset_names = sorted(set(s["source_dataset"] for s in _all_scenes))
    print(f"[SceneIndex] Loaded {len(_all_scenes)} scenes from {len(_dataset_names)} datasets")


def _init_benchmark_dataset():
    """Initialize the same unified loader used by benchmark/evaluation."""
    global _benchmark_dataset, _scene_to_dataset_idx, DATASET_Z_FAR

    _benchmark_dataset = BenchmarkDataset(
        scene_index_path=str(SCENE_INDEX_PATH),
        benchmark_root=str(BENCHMARK_ROOT),
        tags=None,
        tag_operator="AND",
        max_scenes=None,
    )
    _scene_to_dataset_idx = {
        s["scene_id"]: i for i, s in enumerate(_benchmark_dataset.scenes)
    }
    DATASET_Z_FAR = {
        name: float(reader.DEFAULT_Z_FAR)
        for name, reader in _benchmark_dataset._readers.items()
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _get_data_root(dataset: str) -> Path:
    density = "unknown"
    return BENCHMARK_ROOT / density / dataset


def _load_scene_data_raw(scene: dict, z_far: float, use_depth_mask: bool = True, conf_threshold: float = 0.0):
    """Load a scene through BenchmarkDataset's current unified data pipeline.

    Current SpatialBenchmark layout:
        <benchmark_root>/<density>/<dataset>/<scene_path>/{images,depths,poses,intrinsics,...}

    This function adapts the BenchmarkDataset output to the older web-viewer
    return shape: lists of RGB uint8 images, depth maps, c2w extrinsics, and
    intrinsics. Dataset-specific path and depth decoding logic lives in
    benchmark/datasets/data_readers.py.
    """
    if _benchmark_dataset is None:
        raise RuntimeError("BenchmarkDataset is not initialized")

    dataset = scene["source_dataset"]
    scene_id = scene["scene_id"]
    ds_idx = _scene_to_dataset_idx.get(scene_id)
    if ds_idx is None:
        raise KeyError(f"Scene {scene_id!r} not found in BenchmarkDataset")

    reader = _benchmark_dataset._readers.get(dataset)
    if reader is None:
        raise ValueError(f"No reader for dataset {dataset!r}")

    # Endpoint controls are applied by temporarily overriding the reader settings
    # before BenchmarkDataset calls reader.read_scene().
    had_z_far = 'DEFAULT_Z_FAR' in reader.__dict__
    orig_z_far = getattr(reader, 'DEFAULT_Z_FAR', None)
    had_depth_masks = '_USE_DEPTH_MASKS' in reader.__dict__
    orig_depth_masks = getattr(reader, '_USE_DEPTH_MASKS', True)
    had_aliasing_masks = '_USE_ALIASING_MASKS' in reader.__dict__
    orig_aliasing_masks = getattr(reader, '_USE_ALIASING_MASKS', True)
    had_conf = 'conf_threshold' in reader.__dict__
    orig_conf = getattr(reader, 'conf_threshold', None)

    try:
        reader.DEFAULT_Z_FAR = float(z_far)
        if not use_depth_mask:
            reader._USE_DEPTH_MASKS = False
            reader._USE_ALIASING_MASKS = False
        if hasattr(reader, 'conf_threshold'):
            reader.conf_threshold = float(conf_threshold)

        loaded = _benchmark_dataset[ds_idx]
    finally:
        if had_z_far:
            reader.DEFAULT_Z_FAR = orig_z_far
        elif 'DEFAULT_Z_FAR' in reader.__dict__:
            delattr(reader, 'DEFAULT_Z_FAR')

        if had_depth_masks:
            reader._USE_DEPTH_MASKS = orig_depth_masks
        elif '_USE_DEPTH_MASKS' in reader.__dict__:
            delattr(reader, '_USE_DEPTH_MASKS')

        if had_aliasing_masks:
            reader._USE_ALIASING_MASKS = orig_aliasing_masks
        elif '_USE_ALIASING_MASKS' in reader.__dict__:
            delattr(reader, '_USE_ALIASING_MASKS')

        if had_conf:
            reader.conf_threshold = orig_conf
        elif 'conf_threshold' in reader.__dict__:
            delattr(reader, 'conf_threshold')

    images_np = loaded["images_raw"].permute(0, 2, 3, 1).cpu().numpy()
    images = [
        np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        for img in images_np
    ]
    depths = [np.ascontiguousarray(d.astype(np.float32)) for d in loaded["depth"]]
    extrinsics = [
        np.ascontiguousarray(e.astype(np.float32))
        for e in loaded["extrinsic"]
    ]
    intrinsics = [
        np.ascontiguousarray(k.astype(np.float32))
        for k in loaded["intrinsic"]
    ]

    return {
        "images": images,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "K": intrinsics[0] if intrinsics else np.eye(3, dtype=np.float32),
    }


def _load_droid_style(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True, pose_scale: float = 1.0):
    """Load DROID-style data (shared by droid, tum, bonn, nrgbd, 7scenes, tanks).

    pose_scale: multiplier for pose translation. 1.0 (default) for most datasets.
                0.001 for DTU where poses are in mm but depth /1000 converts to meters.
    """
    rgb_dir = scene_dir / "images" / "left"
    depth_dir = scene_dir / "depth_npy"
    depth_mask_dir = scene_dir / "depth_mask"

    rgb_paths = sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem))
    depth_paths = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem.replace("_depth", "")))
    depth_mask_paths = sorted(depth_mask_dir.glob("*.png")) if depth_mask_dir.exists() else []

    # Poses
    pose_path = scene_dir / "poses_ma" / "poses_depth_ba.npy"
    if not pose_path.exists():
        pose_path = scene_dir / "poses_ma" / "poses.npy"
    all_poses = np.load(str(pose_path)).astype(np.float32)
    print(f"using pose from {pose_path}")
    
    # Intrinsics
    intr_dir = scene_dir / "intrinsics"
    K_path = list(intr_dir.glob("*left.npy")) or list(intr_dir.glob("*.npy"))
    K = np.load(str(K_path[0])).astype(np.float32) if K_path else np.eye(3, dtype=np.float32)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths) or idx >= len(depth_paths):
            continue
        # RGB
        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Depth
        d_raw = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_UNCHANGED)
        if d_raw is not None:
            depth = d_raw.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0
            # Apply depth mask if available
            if use_depth_mask and depth_mask_paths and idx < len(depth_mask_paths):
                mask = cv2.imread(str(depth_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    depth[mask == 0] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((480, 640), dtype=np.float32)

        images.append(img)
        depths.append(depth)
        pose = all_poses[min(idx, len(all_poses) - 1)].copy()
        if pose_scale != 1.0:
            pose[:3, 3] *= pose_scale
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_ropedia(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True, conf_threshold: float = 0.0):
    """Load Ropedia/ViPE format data."""
    rgb_dir = scene_dir / "images" / "left"
    # Prefer depths_ropedia/ (raw), fall back to depths/ (cleaned)
    depth_dir = scene_dir / "depths_ropedia"
    if not depth_dir.exists():
        depth_dir = scene_dir / "depths"
    depth_mask_dir = scene_dir / "depth_mask"
    conf_mask_dir = scene_dir / "conf_mask"
    has_conf = conf_mask_dir.is_dir()

    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    depth_mask_paths = sorted(depth_mask_dir.glob("*.png")) if depth_mask_dir.exists() else []

    # Try loading poses: prefer pose/ (refined), fall back to pose_from_hdf5/
    pose_file = scene_dir / "pose" / "left.npz"
    if not pose_file.exists():
        pose_file = scene_dir / "pose_from_hdf5" / "left.npz"
    pose_data = np.load(str(pose_file))
    all_poses = pose_data["data"].astype(np.float32)
    pose_inds = pose_data.get("inds", None)
    if all_poses.ndim == 3 and all_poses.shape[1] == 4 and all_poses.shape[2] == 4:
        all_poses = all_poses[:, :3, :]

    # Intrinsics: read from annotation.hdf5 if available, else hardcode
    ann_path = scene_dir / "annotation.hdf5"
    if ann_path.exists():
        with h5py.File(str(ann_path), "r") as f:
            k_vals = f["calibration/cam01/K"][:].tolist()  # [fx, fy, cx, cy]
        K = np.array([[k_vals[0], 0, k_vals[2]],
                       [0, k_vals[1], k_vals[3]],
                       [0, 0, 1]], dtype=np.float32)
    else:
        K = np.array([[200, 0, 256], [0, 200, 256], [0, 0, 1]], dtype=np.float32)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(all_poses):
            continue

        # Determine file names based on pose_inds if available
        if pose_inds is not None and idx < len(pose_inds):
            rgb_fname = f"frame_{int(pose_inds[idx]):05d}_rgb.png"
            depth_fname = f"{int(pose_inds[idx]):06d}.png"
        else:
            rgb_fname = None
            depth_fname = None

        # RGB
        if rgb_fname and (rgb_dir / rgb_fname).exists():
            img = cv2.cvtColor(cv2.imread(str(rgb_dir / rgb_fname)), cv2.COLOR_BGR2RGB)
        elif idx < len(rgb_paths):
            img = cv2.cvtColor(cv2.imread(str(rgb_paths[idx])), cv2.COLOR_BGR2RGB)
        else:
            continue

        # Depth
        if depth_fname and (depth_dir / depth_fname).exists():
            d_raw = cv2.imread(str(depth_dir / depth_fname), cv2.IMREAD_ANYDEPTH)
        elif idx < len(depth_paths):
            d_raw = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_ANYDEPTH)
        else:
            continue

        depth = d_raw.astype(np.float32) / 1000.0
        depth[~np.isfinite(depth)] = 0

        # Apply depth mask if available
        if use_depth_mask and depth_mask_paths:
            dm_path = depth_mask_dir / rgb_fname if rgb_fname else None
            if dm_path and dm_path.exists():
                mask = cv2.imread(str(dm_path), cv2.IMREAD_UNCHANGED)
                depth[mask == 0] = 0
            elif idx < len(depth_mask_paths):
                mask = cv2.imread(str(depth_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
                depth[mask == 0] = 0

        depth[depth > z_far] = 0

        # Confidence mask filtering
        if has_conf and conf_threshold > 0:
            conf_fname = rgb_fname if rgb_fname else None
            conf_path = conf_mask_dir / conf_fname if conf_fname else None
            if conf_path and conf_path.exists():
                conf_raw = cv2.imread(str(conf_path), cv2.IMREAD_GRAYSCALE)
                if conf_raw is not None:
                    conf = conf_raw.astype(np.float32) / 255.0
                    depth[conf < conf_threshold] = 0

        # Center crop RGB to depth size if needed
        H_d, W_d = depth.shape
        H_r, W_r = img.shape[:2]
        if H_r != H_d or W_r != W_d:
            if H_r >= H_d and W_r >= W_d:
                t = (H_r - H_d) // 2
                l = (W_r - W_d) // 2
                img = img[t:t + H_d, l:l + W_d]

        images.append(img)
        depths.append(depth)
        extrinsics.append(all_poses[min(idx, len(all_poses) - 1)])

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_adt(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load ADT (Aria Digital Twin) format data.

    Layout:
      rgb_rectified/*.png       — timestamp-named RGB images
      depth_rectified/*.png     — uint16 PNG depth (mm), timestamp-named
      depth_mask/*.png          — optional binary mask from depth cleaning
      intrinsic.npy             — 3x3 camera intrinsic matrix
      camera_to_world/*.npy     — per-frame 3x4 cam-to-world poses
    """
    rgb_dir = scene_dir / "rgb_rectified"
    depth_dir = scene_dir / "depth_rectified"
    depth_mask_dir = scene_dir / "depth_mask"
    pose_dir = scene_dir / "camera_to_world"

    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    pose_paths = sorted(pose_dir.glob("*.npy"))
    depth_mask_paths = sorted(depth_mask_dir.glob("*.png")) if depth_mask_dir.exists() else []

    # Intrinsics
    K_path = scene_dir / "intrinsic.npy"
    K = np.load(str(K_path)).astype(np.float32) if K_path.exists() else np.eye(3, dtype=np.float32)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths) or idx >= len(depth_paths) or idx >= len(pose_paths):
            continue

        # RGB
        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((512, 512, 3), dtype=np.uint8)

        # Depth (uint16 mm -> metres)
        d_raw = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_UNCHANGED)
        if d_raw is not None:
            depth = d_raw.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0
            # Apply depth mask if available
            if use_depth_mask and depth_mask_paths and idx < len(depth_mask_paths):
                mask = cv2.imread(str(depth_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    depth[mask == 0] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((512, 512), dtype=np.float32)

        # Pose (3x4 cam-to-world)
        pose = np.load(str(pose_paths[idx])).astype(np.float32)
        if pose.shape == (4, 4):
            pose = pose[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_robotwin(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load RoboTwin format data.

    Layout:
      camera_data/images/*.png       — RGB images
      camera_data/depths/*.png       — uint16 PNG depth (mm)
      camera_data/extrinsics/*.npy   — per-frame 3x4 cam-to-world poses
      camera_data/intrinsics/*.npy   — per-frame 3x3 intrinsics
    """
    rgb_dir = scene_dir / "camera_data" / "images"
    depth_dir = scene_dir / "camera_data" / "depths"
    depth_mask_dir = scene_dir / "depth_mask"
    extrinsic_dir = scene_dir / "camera_data" / "extrinsics"
    intrinsic_dir = scene_dir / "camera_data" / "intrinsics"

    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    depth_mask_paths = sorted(depth_mask_dir.glob("*.png")) if depth_mask_dir.exists() else []
    extrinsic_paths = sorted(extrinsic_dir.glob("*.npy"))
    intrinsic_paths = sorted(intrinsic_dir.glob("*.npy"))

    # Use first frame's intrinsic as shared K (they are typically identical)
    K = np.load(str(intrinsic_paths[0])).astype(np.float32) if intrinsic_paths else np.eye(3, dtype=np.float32)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths) or idx >= len(depth_paths) or idx >= len(extrinsic_paths):
            continue

        # RGB
        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Depth (uint16 mm -> metres)
        d_raw = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_ANYDEPTH)
        if d_raw is not None:
            depth = d_raw.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0
            # Apply depth mask if available
            if use_depth_mask and depth_mask_paths and idx < len(depth_mask_paths):
                mask = cv2.imread(str(depth_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    depth[mask == 0] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((720, 1280), dtype=np.float32)

        # Pose (3x4 cam-to-world)
        pose = np.load(str(extrinsic_paths[idx])).astype(np.float32)
        if pose.shape == (4, 4):
            pose = pose[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_robolab(scene_dir: Path, frame_indices: list, z_far: float):
    """Load RoboLab (Isaac Sim) format data.

    Layout:
      rgb/{XXXXXX}.png     — RGB images (1280x720)
      depth.npy            — float32 (T, H, W), z-depth in meters (invalid=0)
      c2w.npy              — float32 (T, 4, 4), cam2world (OpenCV convention)
      K.npy                — float32 (3, 3), shared intrinsics
    """
    rgb_dir = scene_dir / "rgb"
    rgb_paths = sorted(rgb_dir.glob("*.png"))

    K = np.load(str(scene_dir / "K.npy")).astype(np.float32)
    all_depth = np.load(str(scene_dir / "depth.npy"))         # (T, H, W) float32
    all_c2w = np.load(str(scene_dir / "c2w.npy")).astype(np.float32)  # (T, 4, 4)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths) or idx >= len(all_depth) or idx >= len(all_c2w):
            continue

        # RGB
        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Depth: 已为米, 直接使用
        depth = np.ascontiguousarray(all_depth[idx].astype(np.float32))
        depth[~np.isfinite(depth)] = 0
        depth[depth > z_far] = 0

        # Pose: 4x4 cam2world -> (3, 4)
        pose = all_c2w[idx][:3, :].astype(np.float32)

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_rlbench(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load RLBench format data.

    Layout:
      images/frame_XXXX.png           — RGB images
      depth/frame_XXXX.png            — uint16 PNG depth (mm)
      pose/frame_XXXX.npy             — per-frame 4x4 cam-to-world poses
      intrinsic/frame_XXXX.npy        — per-frame 3x3 intrinsics (PyRep convention)
    """
    rgb_dir = scene_dir / "images"
    depth_dir = scene_dir / "depth"
    depth_mask_dir = scene_dir / "depth_mask"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"

    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    depth_mask_paths = sorted(depth_mask_dir.glob("*.png")) if depth_mask_dir.exists() else []
    pose_paths = sorted(pose_dir.glob("*.npy"))
    intrinsic_paths = sorted(intrinsic_dir.glob("*.npy"))

    # Use first frame's intrinsic as shared K for visualization (after abs correction)
    K = np.load(str(intrinsic_paths[0])).astype(np.float32) if intrinsic_paths else np.eye(3, dtype=np.float32)
    K[0, 0] = abs(K[0, 0])
    K[1, 1] = abs(K[1, 1])

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths) or idx >= len(depth_paths) or idx >= len(pose_paths):
            continue

        # RGB
        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Depth (uint16 mm -> metres)
        d_raw = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_ANYDEPTH)
        if d_raw is not None:
            depth = d_raw.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0
            # Apply depth mask if available
            if use_depth_mask and depth_mask_paths and idx < len(depth_mask_paths):
                mask = cv2.imread(str(depth_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    depth[mask == 0] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((720, 1280), dtype=np.float32)

        # Per-frame intrinsic: handle PyRep negative fx/fy by flipping image
        K_frame = np.load(str(intrinsic_paths[idx])).astype(np.float32) if idx < len(intrinsic_paths) else K.copy()
        if K_frame[0, 0] < 0:
            img = img[:, ::-1, :].copy()
            depth = depth[:, ::-1].copy()
            K_frame[0, 2] = (img.shape[1] - 1) - K_frame[0, 2]
            K_frame[0, 0] = abs(K_frame[0, 0])
        if K_frame[1, 1] < 0:
            img = img[::-1, :, :].copy()
            depth = depth[::-1, :].copy()
            K_frame[1, 2] = (img.shape[0] - 1) - K_frame[1, 2]
            K_frame[1, 1] = abs(K_frame[1, 1])

        # Pose (4x4 cam-to-world -> 3x4)
        pose = np.load(str(pose_paths[idx])).astype(np.float32)
        if pose.shape == (4, 4):
            pose = pose[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_tanks_raw(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load Tanks & Temples raw format data.

    Layout:
      images/*.jpg                    — RGB images
      depth/*.npz                     — float32 depth maps (key='arr_0')
      {scene_name}_COLMAP_SfM.log     — extrinsics (4x4 per frame)
    """
    rgb_dir = scene_dir / "images"
    depth_dir = scene_dir / "depth"
    depth_mask_dir = scene_dir / "depth_mask"

    rgb_paths = sorted(rgb_dir.glob("*.jpg"))
    depth_paths = sorted(depth_dir.glob("*.npz"))
    depth_mask_paths = sorted(depth_mask_dir.glob("*.png")) if depth_mask_dir.exists() else []

    # Read extrinsics from COLMAP log
    log_files = list(scene_dir.glob("*_COLMAP_SfM.log"))
    if not log_files:
        raise FileNotFoundError(f"No *_COLMAP_SfM.log in {scene_dir}")

    all_extrinsics = []
    with open(str(log_files[0]), 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if len(line.split()) == 3:
            i += 1
            mat_lines = lines[i:i + 4]
            mat = np.array([[float(x) for x in l.split()] for l in mat_lines], dtype=np.float32)
            all_extrinsics.append(mat)
            i += 4
        else:
            i += 1

    # Estimate intrinsics from first image size
    SCENE_IMAGE_SIZES = {
        'Barn': (1920, 1080),
        'Church': (1961, 1091),
        'Courthouse': (1962, 1091),
        'Ignatius': (1961, 1091),
    }
    scene_name = scene_dir.name
    if scene_name in SCENE_IMAGE_SIZES:
        W, H = SCENE_IMAGE_SIZES[scene_name]
    elif rgb_paths:
        img_tmp = cv2.imread(str(rgb_paths[0]))
        H, W = img_tmp.shape[:2]
    else:
        W, H = 1920, 1080

    fx = fy = 0.7 * W
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=np.float32)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths) or idx >= len(depth_paths) or idx >= len(all_extrinsics):
            continue

        # RGB
        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((H, W, 3), dtype=np.uint8)

        # Depth: npz float32
        depth = np.load(str(depth_paths[idx]))['arr_0'].astype(np.float32)
        depth[~np.isfinite(depth)] = 0
        # Apply depth mask if available
        if use_depth_mask and depth_mask_paths and idx < len(depth_mask_paths):
            mask = cv2.imread(str(depth_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
            if mask is not None:
                depth[mask == 0] = 0
        depth[depth > z_far] = 0

        # Extrinsic: (4, 4) -> (3, 4)
        pose = all_extrinsics[idx]
        if pose.shape == (4, 4):
            pose = pose[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_omniworld(scene_dir: Path, scene_path: str, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load OmniWorld-Game data (metric depth, metric poses).

    scene_dir 实际上是 data_root/scene_id，scene_path = "scene_id/split_N"。
    深度和位姿均乘以 metric_scale 转为米制。
    """
    import csv as _csv
    import json as _json
    from scipy.spatial.transform import Rotation as _R

    parts = scene_path.split('/')
    scene_id = parts[0]
    split_idx = int(parts[1].replace('split_', ''))

    # 加载 metric_scale
    csv_path = scene_dir.parent / "omniworld_game_metadata.csv"
    metric_scale = 1.0
    if csv_path.exists():
        with open(str(csv_path), 'r', encoding='utf-8', newline='') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row["UID"] == scene_id:
                    metric_scale = float(row["Metric Scale"])
                    break

    # scene_dir 已经是 data_root / scene_id
    with open(str(scene_dir / "split_info.json"), 'r') as f:
        split_info = _json.load(f)
    global_indices = split_info["split"][split_idx]

    # 加载相机参数
    cam_file = scene_dir / "camera" / f"split_{split_idx}.json"
    with open(str(cam_file), 'r') as f:
        cam = _json.load(f)

    # 内参 (取均值 focal)
    focal_mean = float(np.mean(cam["focals"]))
    K = np.array([
        [focal_mean, 0, cam["cx"]],
        [0, focal_mean, cam["cy"]],
        [0, 0, 1],
    ], dtype=np.float32)

    # 外参 w2c -> c2w, c2w 平移 × metric_scale
    quat_wxyz = np.array(cam["quats"], dtype=np.float64)
    trans = np.array(cam["trans"], dtype=np.float64)
    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    quat_wxyz = quat_wxyz / norms
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)
    rotations = _R.from_quat(quat_xyzw).as_matrix()

    S = len(cam["quats"])
    w2c = np.repeat(np.eye(4, dtype=np.float64)[None], S, axis=0)
    w2c[:, :3, :3] = rotations
    w2c[:, :3, 3] = trans
    c2w = np.linalg.inv(w2c)
    c2w[:, :3, 3] *= metric_scale

    depth_mask_dir = scene_dir / "depth_mask"
    sky_mask_dir = scene_dir / "sky_mask"

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(global_indices):
            continue
        gi = global_indices[idx]

        # RGB
        img = cv2.imread(str(scene_dir / "color" / f"{gi:06d}.png"))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Depth (反深度解码 × metric_scale = 米)
        d_raw = cv2.imread(str(scene_dir / "depth" / f"{gi:06d}.png"), cv2.IMREAD_UNCHANGED)
        if d_raw is not None:
            depthmap = d_raw.astype(np.float32) / 65535.0
            near_mask = depthmap < 0.0015
            depth_sky_mask = depthmap > (65500.0 / 65535.0)
            near, far = 1.0, 1000.0
            depthmap = depthmap / (far - depthmap * (far - near)) / 0.004
            valid = ~(near_mask | depth_sky_mask)
            depthmap[~valid] = 0
            depthmap[valid] *= metric_scale

            # 应用 depth_mask (飞点过滤)
            if use_depth_mask and depth_mask_dir.exists():
                mask_path = depth_mask_dir / f"{gi:06d}.png"
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        depthmap[mask == 0] = 0

            # 应用 sky_mask (天空区域置零)
            if use_depth_mask and sky_mask_dir.exists():
                sky_path = sky_mask_dir / f"{gi:06d}.png"
                if sky_path.exists():
                    sky_m = cv2.imread(str(sky_path), cv2.IMREAD_UNCHANGED)
                    if sky_m is not None:
                        depthmap[sky_m > 0] = 0

            depthmap[depthmap > z_far] = 0
        else:
            depthmap = np.zeros((720, 1280), dtype=np.float32)

        pose = c2w[idx, :3, :].astype(np.float32)

        images.append(img)
        depths.append(depthmap)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_spatialvid(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load SpatialVID scene data.

    Layout:
      images/000001.jpg ...       — RGB images (1-indexed)
      depth/{ann_idx:06d}.png     — inverse depth (uint16)
      indexes.txt                 — annotation_idx -> video_frame_idx
      poses.npy                   — (N, 7) [tx, ty, tz, qx, qy, qz, qw] w2c
      intrinsics.npy              — (N, 4) [fx, fy, cx, cy] normalized
      depth_mask/{ann_idx:06d}.png — flying point mask (optional)
      sky_mask/{ann_idx:06d}.png   — sky mask (optional)
    """
    from scipy.spatial.transform import Rotation as R

    # Parse indexes.txt
    ann_to_video = {}
    idx_path = scene_dir / "indexes.txt"
    if idx_path.exists():
        for line in idx_path.read_text().strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                ann_to_video[int(parts[0])] = int(parts[1])

    # Load poses (N, 7): w2c -> c2w
    all_poses = np.load(str(scene_dir / "poses.npy")).astype(np.float64)
    N = len(all_poses)
    quat_xyzw = all_poses[:, 3:7]
    rotations = R.from_quat(quat_xyzw).as_matrix()
    translations = all_poses[:, :3]

    w2c = np.repeat(np.eye(4, dtype=np.float64)[None], N, axis=0)
    w2c[:, :3, :3] = rotations
    w2c[:, :3, 3] = translations
    c2w = np.linalg.inv(w2c)
    all_c2w = c2w[:, :3, :].astype(np.float32)

    # Load intrinsics (N, 4): normalized -> pixel
    all_intrin = np.load(str(scene_dir / "intrinsics.npy")).astype(np.float32)

    # Get image size from first image
    img_dir = scene_dir / "images"
    sample_imgs = sorted(img_dir.glob("*.jpg"))
    if not sample_imgs:
        sample_imgs = sorted(img_dir.glob("*.png"))
    first_img = cv2.imread(str(sample_imgs[0]))
    img_H, img_W = first_img.shape[:2]

    # Build shared K from first frame
    fx_n, fy_n, cx_n, cy_n = all_intrin[0]
    K = np.array([
        [fx_n * img_W, 0, cx_n * img_W],
        [0, fy_n * img_H, cy_n * img_H],
        [0, 0, 1],
    ], dtype=np.float32)

    depth_dir = scene_dir / "depth"
    has_depth = depth_dir.is_dir()
    depth_mask_dir = scene_dir / "depth_mask"
    sky_mask_dir = scene_dir / "sky_mask"

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        video_frame_idx = ann_to_video.get(idx, idx)

        # RGB (1-indexed filenames)
        rgb_path = img_dir / f"{video_frame_idx + 1:06d}.jpg"
        if not rgb_path.exists():
            rgb_path = img_dir / f"{video_frame_idx + 1:06d}.png"
        img = cv2.imread(str(rgb_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((img_H, img_W, 3), dtype=np.uint8)

        # Depth (inverse depth)
        if has_depth:
            dp = depth_dir / f"{idx:06d}.png"
            if dp.exists():
                d_raw = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
                disp = d_raw.astype(np.float32) / 65535.0
                valid = disp > 1e-6
                depth = np.zeros_like(disp)
                depth[valid] = 1.0 / disp[valid]
                depth[~np.isfinite(depth)] = 0
            else:
                depth = np.zeros((img_H, img_W), dtype=np.float32)
        else:
            depth = np.zeros((img_H, img_W), dtype=np.float32)

        # Apply masks
        if use_depth_mask:
            mask_path = depth_mask_dir / f"{idx:06d}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    depth[mask == 0] = 0
            sky_path = sky_mask_dir / f"{idx:06d}.png"
            if sky_path.exists():
                sky_mask = cv2.imread(str(sky_path), cv2.IMREAD_UNCHANGED)
                if sky_mask is not None:
                    depth[sky_mask > 0] = 0

        depth[depth > z_far] = 0

        images.append(img)
        depths.append(depth)
        extrinsics.append(all_c2w[idx])

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_vkitti(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load Virtual KITTI 2 scene data.

    Layout (scene_dir = {data_root}/{scene}/{sub_scene}/Camera_0):
      {XXXXX}_rgb.jpg      — RGB 1242×375
      {XXXXX}_depth.png    — uint16, /100 -> meters, sky=65535
      {XXXXX}_cam.npz      — camera_pose (4×4 w2c), camera_intrinsics (3×3)
      depth_mask/*.png     — binary mask (optional)
    """
    rgb_paths = sorted(scene_dir.glob("*_rgb.jpg"))
    depth_paths = sorted(scene_dir.glob("*_depth.png"))
    cam_paths = sorted(scene_dir.glob("*_cam.npz"))
    depth_mask_dir = scene_dir / "depth_mask"
    depth_mask_paths = sorted(depth_mask_dir.glob("*.png")) if depth_mask_dir.is_dir() else []

    # 从第一帧加载内参 (共享)
    cam0 = np.load(str(cam_paths[0]))
    K = cam0['camera_intrinsics'].astype(np.float32)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths) or idx >= len(depth_paths) or idx >= len(cam_paths):
            continue

        # RGB
        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((375, 1242, 3), dtype=np.uint8)

        # Depth: uint16 / 100.0 -> meters
        d_raw = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if d_raw is not None:
            depth = d_raw.astype(np.float32) / 100.0
            depth[~np.isfinite(depth)] = 0
            # 飞点过滤
            if use_depth_mask and depth_mask_paths and idx < len(depth_mask_paths):
                mask = cv2.imread(str(depth_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    depth[mask == 0] = 0
            depth[depth >= 655.0] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((375, 1242), dtype=np.float32)

        # Pose: camera_pose 已是 c2w (4×4), 直接取 (3×4)
        cam_data = np.load(str(cam_paths[idx]))
        pose = cam_data['camera_pose'].astype(np.float32)[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _parse_kitti_calib(path: Path) -> dict:
    """解析 KITTI Odometry calib.txt → {key: (3,4) float32}，key ∈ {P0,P1,P2,P3,Tr}。"""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, vals = line.partition(':')
            parts = vals.split()
            if len(parts) < 12:
                continue
            out[key.strip()] = np.array([float(x) for x in parts[:12]], dtype=np.float32).reshape(3, 4)
    return out


def _parse_kitti_poses(path: Path) -> np.ndarray:
    """解析 poses/XX.txt，每行 12 float，reshape (N,3,4) c2w。文件不存在返回空数组。"""
    if not path.is_file():
        return np.zeros((0, 3, 4), dtype=np.float32)
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 12:
                continue
            poses.append(np.array(vals[:12], dtype=np.float32).reshape(3, 4))
    return np.stack(poses) if poses else np.zeros((0, 3, 4), dtype=np.float32)


def _load_kitti_odometry(data_root: Path, scene_path: str, frame_indices: list):
    """Load KITTI Odometry scene data (镜像 KittiOdometryReader)。

    Layout:
      {data_root}/sequences/{scene_path}/image_2/{XXXXXX}.png   — 左彩色 rectified RGB
      {data_root}/sequences/{scene_path}/calib.txt              — P0 P1 P2 P3 Tr
      {data_root}/poses/{scene_path}.txt                        — 3×4 c2w per frame (仅 00-10 有)

    KITTI Odometry 无稠密深度 GT，depth 以零填充，
    点云接口会返回空（_unproject_frame 用 depth>0.01 过滤），
    前端仍可用相机锥 + RGB + 轨迹查看。
    """
    seq_dir = data_root / "sequences" / scene_path
    rgb_dir = seq_dir / "image_2"
    calib_path = seq_dir / "calib.txt"
    poses_path = data_root / "poses" / f"{scene_path}.txt"

    rgb_paths = sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem))

    # 内参: 取 P2 的左上 3x3 (KITTI Odometry 左 rectified 相机)
    calib = _parse_kitti_calib(calib_path) if calib_path.is_file() else {}
    P2 = calib.get('P2')
    K = P2[:3, :3].astype(np.float32).copy() if P2 is not None else np.eye(3, dtype=np.float32)

    all_poses = _parse_kitti_poses(poses_path)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths):
            continue

        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((375, 1242, 3), dtype=np.uint8)

        # 无深度 GT：零深度占位
        H, W = img.shape[:2]
        depth = np.zeros((H, W), dtype=np.float32)

        # 位姿: c2w (3×4)，若缺失 (test 序列 11-21 或文件缺失) 则回退单位阵
        if idx < len(all_poses):
            pose = all_poses[idx].astype(np.float32)
        else:
            pose = np.eye(4, dtype=np.float32)[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_waymo(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load Waymo Open Dataset scene data.

    Layout (scene_dir = {data_root}/{scene_name}):
      {XXXXX}_1.jpg    — RGB 512×341
      {XXXXX}_1.exr    — float32 depth (meters, LiDAR)
      {XXXXX}_1.npz    — cam2world (4×4), intrinsics (3×3)
      depth_mask/*.png  — binary mask (optional)
      sky_mask/*.png    — binary sky mask (optional)
    """
    rgb_paths = sorted(scene_dir.glob("*.jpg"))
    depth_paths = sorted(scene_dir.glob("*.exr"))
    anno_paths = sorted(scene_dir.glob("*.npz"))
    depth_mask_dir = scene_dir / "depth_mask"
    depth_mask_paths = sorted(depth_mask_dir.glob("*.png")) if depth_mask_dir.is_dir() else []
    sky_mask_dir = scene_dir / "sky_mask"
    sky_mask_paths = sorted(sky_mask_dir.glob("*.png")) if sky_mask_dir.is_dir() else []

    # 从第一帧加载内参 (共享)
    anno0 = np.load(str(anno_paths[0]))
    K = anno0['intrinsics'].astype(np.float32)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(rgb_paths) or idx >= len(depth_paths) or idx >= len(anno_paths):
            continue

        # RGB
        img = cv2.imread(str(rgb_paths[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((341, 512, 3), dtype=np.uint8)

        # Depth: float32 EXR, 已为米制 (IMREAD_ANYDEPTH 保留原始 float32)
        d = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_ANYDEPTH)
        if d is not None:
            depth = d.astype(np.float32)
            depth[~np.isfinite(depth)] = 0
            # 飞点过滤
            if use_depth_mask and depth_mask_paths and idx < len(depth_mask_paths):
                mask = cv2.imread(str(depth_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    depth[mask == 0] = 0
            # Sky mask
            if use_depth_mask and sky_mask_paths and idx < len(sky_mask_paths):
                sky = cv2.imread(str(sky_mask_paths[idx]), cv2.IMREAD_UNCHANGED)
                if sky is not None:
                    depth[sky > 0] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((341, 512), dtype=np.float32)

        # Pose: cam2world (4×4), 取 (3×4)
        anno = np.load(str(anno_paths[idx]))
        pose = anno['cam2world'].astype(np.float32)[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_lingbot(scene_dir: Path, frame_indices: list, z_far: float, use_depth_mask: bool = True):
    """Load Lingbot single-frame data.

    Layout:
      rgb.jpg          — RGB image
      depth.png        — uint16 PNG depth (mm)
      intrinsic.txt    — 3x3 intrinsic matrix in plain text
      depth_mask.png   — binary mask from flying-point cleaning (optional)
      sky_mask.png     — binary sky mask for outdoor scenes (optional)
    """
    # Intrinsics from txt
    K = np.loadtxt(str(scene_dir / "intrinsic.txt")).astype(np.float32)

    # Load masks if available
    depth_mask_path = scene_dir / "depth_mask.png"
    sky_mask_path = scene_dir / "sky_mask.png"

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx != 0:
            continue  # single-frame dataset, only index 0

        # RGB
        img = cv2.imread(str(scene_dir / "rgb.jpg"))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Depth (uint16 mm -> metres)
        d_raw = cv2.imread(str(scene_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
        if d_raw is not None:
            depth = d_raw.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0
            # Apply depth_mask (flying-point removal) + sky_mask
            if use_depth_mask:
                if depth_mask_path.exists():
                    mask = cv2.imread(str(depth_mask_path), cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        depth[mask == 0] = 0
                if sky_mask_path.exists():
                    sky_mask = cv2.imread(str(sky_mask_path), cv2.IMREAD_UNCHANGED)
                    if sky_mask is not None:
                        depth[sky_mask > 0] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((720, 1280), dtype=np.float32)

        # Identity pose (single frame, no extrinsics)
        pose = np.eye(3, 4, dtype=np.float32)

        images.append(img)
        depths.append(depth)
        extrinsics.append(pose)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_eth3d(scene_dir: Path, frame_indices: list, z_far: float):
    """Load ETH3D multi-view stereo data.

    深度为 float32 二进制文件 (单位: 米, 无效值: inf)，pose 为 COLMAP w2c 需转 c2w。
    每帧可能有不同 camera_id，使用第一帧的内参作为共享 K (可视化用途)。
    """
    import re

    def _quat_to_R(qw, qx, qy, qz):
        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        q /= np.linalg.norm(q)
        qw, qx, qy, qz = q
        return np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
        ], dtype=np.float64)

    def _dsc_num(name):
        m = re.search(r'DSC_(\d+)', name)
        return int(m.group(1)) if m else 0

    # Parse cameras.txt
    K_dict = {}
    with open(scene_dir / "dslr_calibration_jpg" / "cameras.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            fx, fy, cx, cy = map(float, parts[4:8])
            K_dict[cam_id] = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

    # Parse images.txt -> sorted frames
    frames = []
    with open(scene_dir / "dslr_calibration_jpg" / "images.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        R = _quat_to_R(qw, qx, qy, qz)
        T_cw = np.eye(4, dtype=np.float64)
        T_cw[:3,:3] = R; T_cw[:3,3] = [tx, ty, tz]
        T_wc = np.linalg.inv(T_cw)
        c2w = T_wc[:3,:].astype(np.float32)
        K = K_dict.get(cam_id, np.eye(3, dtype=np.float32))
        frames.append((name, c2w, K))
    frames.sort(key=lambda x: _dsc_num(x[0]))

    # 使用第一帧内参作为共享 K (可视化用)
    K = frames[0][2] if frames else np.eye(3, dtype=np.float32)

    rgb_dir = scene_dir / "images" / "dslr_images"
    depth_dir = scene_dir / "ground_truth_depth" / "dslr_images"

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(frames):
            continue
        name, c2w, _ = frames[idx]
        fname = Path(name).name

        # RGB
        img = cv2.imread(str(rgb_dir / fname))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((4032, 6048, 3), dtype=np.uint8)

        # Depth: float32 binary
        depth_path = depth_dir / fname
        if depth_path.exists():
            depth = np.fromfile(str(depth_path), dtype=np.float32).reshape(4032, 6048)
            depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((4032, 6048), dtype=np.float32)

        images.append(img)
        depths.append(depth)
        extrinsics.append(c2w)

    return {
        "images": images, "depths": depths,
        "extrinsics": extrinsics, "K": K,
    }


def _load_hiroom(scene_dir: Path, frame_indices: list, z_far: float):
    """Load HiRoom format (indoor simulation, static).

    Layout:
        image/{idx}.jpg              (RGB)
        depth/{idx}.png              (uint16, raw / 655.35 -> meters)
        pose/{idx}.npy               (4×4 world2cam; inverted to c2w here)
        cam_K.npy                    (3×3 shared intrinsics)
        aliasing_mask/{idx}.png      (uint8 0/255, >0 = invalid)
    """
    image_dir = scene_dir / "image"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    mask_dir = scene_dir / "aliasing_mask"

    img_paths = sorted(image_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    K = np.load(str(scene_dir / "cam_K.npy")).astype(np.float32)

    HIROOM_DEPTH_SCALE = 65535.0 / 100.0
    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(img_paths):
            continue
        img_path = img_paths[idx]
        stem = img_path.stem

        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)

        d_raw = cv2.imread(str(depth_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if d_raw is not None:
            depth = d_raw.astype(np.float32) / HIROOM_DEPTH_SCALE
            depth[~np.isfinite(depth)] = 0
            mask_path = mask_dir / f"{stem}.png"
            if mask_path.exists():
                alias = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if alias is not None:
                    depth[alias > 0] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((1024, 1024), dtype=np.float32)

        # Pose: w2c (4, 4) -> c2w (3, 4)
        w2c = np.load(str(pose_dir / f"{stem}.npy")).astype(np.float32)
        if w2c.shape == (3, 4):
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = w2c
            w2c = T
        c2w = np.linalg.inv(w2c).astype(np.float32)[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(c2w)

    return {"images": images, "depths": depths, "extrinsics": extrinsics, "K": K}


def _load_scannetpp(scene_dir: Path, frame_indices: list, z_far: float):
    """Load ScanNet++ iPhone subset (indoor real, static).

    Layout:
        merge_dslr_iphone/
            colmap/sparse_render_rgb/   (COLMAP binary, DSLR + iPhone)
            images/iphone/frame_XXXXXX.jpg
            render_depth/frame_XXXXXX.png  (uint16, /1000 -> meters)

    Only iPhone frames are used. RGB is undistorted to rectified space so it
    aligns with the rendered depth and rectified intrinsics.
    """
    import pycolmap

    merged = scene_dir / "merge_dslr_iphone"
    colmap_path = merged / "colmap" / "sparse_render_rgb"
    image_dir = merged / "images"
    depth_dir = merged / "render_depth"

    rec = pycolmap.Reconstruction(str(colmap_path))

    # 收集 iPhone 帧 (按 name 排序与 build_scannetpp_index 保持一致)
    frames = []
    for im in rec.images.values():
        if "iphone" not in im.name:
            continue
        frames.append((im.name, im, rec.cameras[im.camera_id]))
    frames.sort(key=lambda x: x[0])

    # 共享 rectified K (首帧相机参数)
    first_cam = frames[0][2]
    params = np.asarray(first_cam.params, dtype=np.float64)
    fx, fy, cx, cy = params[:4]
    dist = np.zeros(5, dtype=np.float32)
    if first_cam.model_name == "OPENCV" and len(params) >= 8:
        dist[:4] = params[4:8]
    K_raw = np.array([
        [fx, 0,  cx - 0.5],
        [0,  fy, cy - 0.5],
        [0,  0,  1.0     ],
    ], dtype=np.float32)
    W, H = first_cam.width, first_cam.height
    K_rect, _ = cv2.getOptimalNewCameraMatrix(K_raw, dist, (W, H), 1, (W, H))
    K_rect = np.asarray(K_rect, dtype=np.float32)

    images, depths, extrinsics = [], [], []
    for idx in frame_indices:
        if idx >= len(frames):
            continue
        name, im, cam = frames[idx]

        # Per-frame K (都是 iphone 相机, 一般一致)
        p_params = np.asarray(cam.params, dtype=np.float64)
        p_dist = np.zeros(5, dtype=np.float32)
        if cam.model_name == "OPENCV" and len(p_params) >= 8:
            p_dist[:4] = p_params[4:8]
        p_K_raw = np.array([
            [p_params[0], 0, p_params[2] - 0.5],
            [0, p_params[1], p_params[3] - 0.5],
            [0, 0, 1.0],
        ], dtype=np.float32)
        p_K_rect, _ = cv2.getOptimalNewCameraMatrix(
            p_K_raw, p_dist, (cam.width, cam.height), 1, (cam.width, cam.height)
        )
        p_K_rect = np.asarray(p_K_rect, dtype=np.float32)

        img_path = image_dir / name
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.undistort(img, p_K_raw, p_dist, newCameraMatrix=p_K_rect)
        else:
            img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

        frame_stem = Path(name).stem
        d_raw = cv2.imread(str(depth_dir / f"{frame_stem}.png"), cv2.IMREAD_UNCHANGED)
        if d_raw is not None:
            depth = d_raw.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0
            depth[depth > z_far] = 0
        else:
            depth = np.zeros((cam.height, cam.width), dtype=np.float32)

        # Pose: w2c -> c2w
        rigid = im.cam_from_world()
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = rigid.matrix()
        c2w = np.linalg.inv(w2c).astype(np.float32)[:3, :]

        images.append(img)
        depths.append(depth)
        extrinsics.append(c2w)

    return {"images": images, "depths": depths, "extrinsics": extrinsics, "K": K_rect}


def _unproject_frame(depth, K, pose, downsample=4, max_pts=500000):
    """Unproject a single depth map to world coordinates."""
    H, W = depth.shape
    v_idx = np.arange(0, H, downsample)
    u_idx = np.arange(0, W, downsample)
    uu, vv = np.meshgrid(u_idx, v_idx)
    dd = depth[vv, uu]
    valid = (dd > 0.01)

    u_v = uu[valid].astype(np.float32)
    v_v = vv[valid].astype(np.float32)
    z = dd[valid]

    x = (u_v - K[0, 2]) / K[0, 0] * z
    y = (v_v - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=-1)

    R, t = pose[:3, :3], pose[:3, 3]
    pts_world = (pts_cam @ R.T + t).astype(np.float32)

    if len(pts_world) > max_pts:
        idx = np.random.choice(len(pts_world), max_pts, replace=False)
        pts_world = pts_world[idx]
        valid_indices = np.where(valid)
        sampled_v = valid_indices[0][idx]
        sampled_u = valid_indices[1][idx]
        return pts_world, sampled_v * downsample, sampled_u * downsample
    else:
        valid_indices = np.where(valid)
        return pts_world, valid_indices[0] * downsample, valid_indices[1] * downsample


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
def list_datasets():
    return _dataset_names


@app.get("/api/dataset_z_far")
def dataset_z_far():
    """Return per-dataset default z_far mapping."""
    return DATASET_Z_FAR


@app.get("/api/tags")
def list_tags():
    """Return all unique tag axes and values."""
    axes = {}
    for s in _all_scenes:
        for axis, val in s.get("tags", {}).items():
            axes.setdefault(axis, set()).add(val)
    return {k: sorted(v) for k, v in axes.items()}


@app.get("/api/scenes")
def list_scenes(
    dataset: str = Query(None),
    view_density: str = Query(None),
    environment: str = Query(None),
    dynamics: str = Query(None),
    view_type: str = Query(None),
    data_type: str = Query(None),
):
    """List scenes with optional filtering."""
    result = _all_scenes
    if dataset:
        result = [s for s in result if s["source_dataset"] == dataset]
    if view_density:
        result = [s for s in result if s.get("tags", {}).get("view_density") == view_density]
    if environment:
        result = [s for s in result if s.get("tags", {}).get("environment") == environment]
    if dynamics:
        result = [s for s in result if s.get("tags", {}).get("dynamics") == dynamics]
    if view_type:
        result = [s for s in result if s.get("tags", {}).get("view_type") == view_type]
    if data_type:
        result = [s for s in result if s.get("tags", {}).get("data_type") == data_type]
    return [
        {
            "scene_id": s["scene_id"],
            "source_dataset": s["source_dataset"],
            "tags": s.get("tags", {}),
            "num_frames": len(s["frame_indices"]),
            "num_frames_total": s.get("num_frames_total", 0),
        }
        for s in result
    ]


@app.get("/api/scene/{scene_id}/rgb/{frame_idx}")
def get_scene_rgb(scene_id: str, frame_idx: int, z_far: float = Query(10.0), depth_mask: bool = Query(True), conf_threshold: float = Query(0.0)):
    """Get RGB image for a specific frame within a scene."""
    scene = _find_scene(scene_id)
    if scene is None:
        return Response(status_code=404)

    data = _load_scene_data_raw(scene, z_far, use_depth_mask=depth_mask, conf_threshold=conf_threshold)
    if frame_idx >= len(data["images"]):
        return Response(status_code=404)

    img = cv2.cvtColor(data["images"][frame_idx], cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/scene/{scene_id}/depth_viz/{frame_idx}")
def get_scene_depth_viz(scene_id: str, frame_idx: int, z_far: float = Query(10.0), depth_mask: bool = Query(True), conf_threshold: float = Query(0.0)):
    """Get colorized depth for a specific frame."""
    scene = _find_scene(scene_id)
    if scene is None:
        return Response(status_code=404)

    data = _load_scene_data_raw(scene, z_far, use_depth_mask=depth_mask, conf_threshold=conf_threshold)
    if frame_idx >= len(data["depths"]):
        return Response(status_code=404)

    depth = data["depths"][frame_idx]
    valid = depth > 0.01
    depth_norm = np.zeros_like(depth)
    if valid.any():
        depth_norm[valid] = np.clip(depth[valid] / z_far, 0, 1)
    colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0

    _, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/scene/{scene_id}/pcd")
def get_scene_pcd(
    scene_id: str,
    z_far: float = Query(10.0),
    downsample: int = Query(4),
    max_pts: int = Query(800000),
    depth_mask: bool = Query(True),
    conf_threshold: float = Query(0.0),
):
    """Return merged point cloud for all frames in the scene.

    Binary format: int32(N) | float32[N*3](xyz) | uint8[N*3](rgb)
    """
    scene = _find_scene(scene_id)
    if scene is None:
        return Response(status_code=404)

    data = _load_scene_data_raw(scene, z_far, use_depth_mask=depth_mask, conf_threshold=conf_threshold)

    all_pts = []
    all_rgb = []

    for i in range(len(data["depths"])):
        depth = data["depths"][i]
        pose = data["extrinsics"][i]
        img = data["images"][i]
        K = data["intrinsics"][i] if "intrinsics" in data else data["K"]

        pts, vs, us = _unproject_frame(depth, K, pose, downsample, max_pts)
        if len(pts) == 0:
            continue

        # Sample colors
        vs_c = np.clip(vs, 0, img.shape[0] - 1).astype(int)
        us_c = np.clip(us, 0, img.shape[1] - 1).astype(int)
        colors = img[vs_c, us_c]

        all_pts.append(pts)
        all_rgb.append(colors)

    if not all_pts:
        buf = io.BytesIO()
        buf.write(struct.pack("<i", 0))
        return Response(content=buf.getvalue(), media_type="application/octet-stream")

    pts_all = np.concatenate(all_pts, axis=0)
    rgb_all = np.concatenate(all_rgb, axis=0)

    # Limit total points
    if len(pts_all) > max_pts:
        idx = np.random.choice(len(pts_all), max_pts, replace=False)
        pts_all = pts_all[idx]
        rgb_all = rgb_all[idx]

    buf = io.BytesIO()
    buf.write(struct.pack("<i", len(pts_all)))
    buf.write(pts_all.astype(np.float32).tobytes())
    buf.write(rgb_all.astype(np.uint8).tobytes())
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


@app.get("/api/scene/{scene_id}/cameras")
def get_scene_cameras(scene_id: str, z_far: float = Query(10.0), depth_mask: bool = Query(True), conf_threshold: float = Query(0.0)):
    """Return camera positions and frustum data for all frames.

    Binary format: int32(N) | float32[N*12](poses 3x4) | float32[9](K 3x3)
    """
    scene = _find_scene(scene_id)
    if scene is None:
        return Response(status_code=404)

    data = _load_scene_data_raw(scene, z_far, use_depth_mask=depth_mask, conf_threshold=conf_threshold)
    K = data["K"]
    poses = np.array(data["extrinsics"], dtype=np.float32)

    buf = io.BytesIO()
    buf.write(struct.pack("<i", len(poses)))
    buf.write(poses.tobytes())
    buf.write(K.astype(np.float32).tobytes())
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


def _find_scene(scene_id: str):
    for s in _all_scenes:
        if s["scene_id"] == scene_id:
            return s
    return None


@app.post("/api/scene/{scene_id}/export_glb")
def export_scene_glb(
    scene_id: str,
    z_far: float = Query(10.0),
    downsample: int = Query(1),
    max_pts: int = Query(3_000_000),
    depth_mask: bool = Query(True),
    conf_threshold: float = Query(0.0),
    output_dir: str = Query("glb_output"),
    frustum_scale: float = Query(0.0),
):
    """Export current scene as GLB point cloud with coloured camera frustums.

    Uses the same data pipeline as the web viewer so the result exactly matches
    what is displayed.  Saves to {output_dir}/{source_dataset}/{view_density}/{scene_id}.glb
    and returns JSON with the output path and stats.
    """
    scene = _find_scene(scene_id)
    if scene is None:
        return JSONResponse(status_code=404, content={"error": f"Scene {scene_id!r} not found"})

    source_dataset = scene["source_dataset"]
    density = scene.get("tags", {}).get("view_density", "unknown")

    out_dir = Path(output_dir) / source_dataset / density
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_id}.glb"

    try:
        data = _load_scene_data_raw(scene, z_far, use_depth_mask=depth_mask, conf_threshold=conf_threshold)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load data: {e}"})

    all_pts, all_rgb = [], []

    per_frame_max = max(1, max_pts // max(len(data["depths"]), 1))
    for i in range(len(data["depths"])):
        depth = data["depths"][i]
        pose  = data["extrinsics"][i]
        img   = data["images"][i]
        K = data["intrinsics"][i] if "intrinsics" in data else data["K"]

        pts, vs, us = _unproject_frame(depth, K, pose, downsample, per_frame_max)
        if len(pts) == 0:
            continue

        vs_c = np.clip(vs, 0, img.shape[0] - 1).astype(int)
        us_c = np.clip(us, 0, img.shape[1] - 1).astype(int)
        all_pts.append(pts)
        all_rgb.append(img[vs_c, us_c])

    if not all_pts:
        return JSONResponse(status_code=422, content={"error": "No valid points in scene"})

    pts_all = np.concatenate(all_pts, axis=0)
    rgb_all = np.concatenate(all_rgb, axis=0).astype(np.float32) / 255.0

    if len(pts_all) > max_pts:
        idx = np.random.choice(len(pts_all), max_pts, replace=False)
        pts_all = pts_all[idx]
        rgb_all = rgb_all[idx]

    # ── camera params for frustums ──
    poses_arr = np.array(data["extrinsics"], dtype=np.float32)   # (N, 3, 4)
    N = len(poses_arr)
    K_arr = np.array(
        data.get("intrinsics", [data["K"]] * N), dtype=np.float32
    )                                                           # (N, 3, 3)
    H, W = data["depths"][0].shape if data["depths"] else (480, 640)

    # ── frustum scale: use viewer value if provided, else auto ──
    if frustum_scale > 0:
        fscale = frustum_scale
    else:
        cam_centers = poses_arr[:, :3, 3]
        traj_span = float(np.linalg.norm(cam_centers.max(0) - cam_centers.min(0))) if N > 1 else 0.0
        depths_stacked = np.stack(data["depths"])
        valid_d = depths_stacked[(depths_stacked > 0) & np.isfinite(depths_stacked)]
        med_depth = float(np.median(valid_d)) if len(valid_d) > 0 else 1.0
        fscale = float(np.clip(max(traj_span, med_depth) * 0.05, 0.02, 5.0))

    try:
        save_pointcloud_glb(
            pts_all, rgb_all, str(out_path),
            extrinsics=poses_arr,
            intrinsics=K_arr,
            image_size=(W, H),
            frustum_scale=fscale,
            max_pts=max_pts,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"GLB export failed: {e}"})

    size_mb = round(out_path.stat().st_size / (1024 ** 2), 2)
    print(f"[GLB] {out_path}  {len(pts_all)} pts  {size_mb} MB  frustum={fscale:.3f}m")
    return {
        "status": "ok",
        "path": str(out_path),
        "num_points": int(len(pts_all)),
        "size_mb": size_mb,
        "frustum_scale": round(fscale, 4),
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark GT Point Cloud Viewer</title>
<style>
:root{
  --bg:#f7f8fa;--sf:#ffffff;--sf2:#eef1f5;--bd:#d8dde5;
  --tx:#1a2030;--txm:#6a7585;--pr:#2f6fd8;--pr2:#e3ecfb;
  --rd:#d83a3a;--gn:#2f9a4e;--yl:#c48510;--cy:#1fa3bd;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;overflow:hidden;height:100vh;display:flex;flex-direction:column}

/* ── sidebar ── */
.sidebar{width:320px;background:var(--sf);border-right:1px solid var(--bd);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.sidebar .logo{padding:14px 16px 10px;font-size:15px;font-weight:600;border-bottom:1px solid var(--bd);letter-spacing:-.3px}
.sidebar .logo span{color:var(--pr);font-weight:700}

.filter-section{padding:10px 14px;border-bottom:1px solid var(--bd)}
.filter-section h3{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:var(--txm);margin-bottom:6px}
.filter-row{display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap}
.filter-row select{background:var(--sf2);color:var(--tx);border:1px solid var(--bd);padding:5px 8px;border-radius:6px;font-size:12px;flex:1;min-width:0;cursor:pointer;outline:none;transition:border-color .15s}
.filter-row select:hover{border-color:var(--pr)}
.filter-row select:focus{border-color:var(--pr);box-shadow:0 0 0 2px rgba(47,111,216,.18)}

.scene-list{flex:1;overflow-y:auto;padding:6px}
.scene-list::-webkit-scrollbar{width:5px}
.scene-list::-webkit-scrollbar-track{background:transparent}
.scene-list::-webkit-scrollbar-thumb{background:var(--bd);border-radius:3px}
.scene-item{padding:8px 10px;border-radius:6px;cursor:pointer;margin-bottom:2px;transition:background .12s;display:flex;flex-direction:column;gap:2px}
.scene-item:hover{background:var(--sf2)}
.scene-item.active{background:var(--pr2);border:1px solid rgba(47,111,216,.35)}
.scene-item .name{font-size:12px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.scene-item .meta{font-size:10px;color:var(--txm);display:flex;gap:6px}
.scene-item .tag{background:rgba(47,111,216,.12);color:var(--pr);padding:1px 5px;border-radius:3px;font-size:9px;font-weight:500}
.scene-item .tag.ds{background:rgba(31,163,189,.14);color:var(--cy)}
.scene-count{padding:6px 14px;font-size:11px;color:var(--txm);border-bottom:1px solid var(--bd)}

/* ── main layout ── */
.main-area{flex:1;display:flex;overflow:hidden}
.content{flex:1;display:flex;flex-direction:column;overflow:hidden}

/* ── toolbar ── */
.toolbar{display:flex;align-items:center;gap:10px;padding:8px 14px;background:var(--sf);border-bottom:1px solid var(--bd);flex-wrap:wrap}
.toolbar .btn{background:var(--sf2);color:var(--tx);border:1px solid var(--bd);padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px;transition:all .12s;white-space:nowrap}
.toolbar .btn:hover{border-color:var(--pr);background:var(--pr2)}
.toolbar .btn.on{background:var(--pr2);border-color:var(--pr);color:var(--pr)}
.ctrl-group{display:flex;align-items:center;gap:5px}
.ctrl-group label{color:var(--txm);font-size:11px;white-space:nowrap}
.ctrl-group input[type=range]{width:100px;accent-color:var(--pr);height:3px}
.ctrl-group .val{color:var(--pr);min-width:36px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
.sep{width:1px;height:20px;background:var(--bd)}
.badge{padding:3px 10px;border-radius:10px;font-size:11px;font-variant-numeric:tabular-nums}

/* ── panels ── */
.panels{flex:1;display:flex;overflow:hidden}

/* images panel */
.img-panel{width:280px;display:flex;flex-direction:column;border-right:1px solid var(--bd);overflow:hidden;flex-shrink:0}
.img-panel.hidden{display:none}
.img-sec{flex:1;min-height:0;padding:4px 6px;display:flex;flex-direction:column}
.img-lbl{font-size:10px;color:var(--txm);text-transform:uppercase;letter-spacing:.6px;margin-bottom:2px;flex-shrink:0}
.img-sec img{width:100%;flex:1;min-height:0;object-fit:contain;border-radius:4px;background:var(--sf2);display:block}
.frame-nav{display:flex;align-items:center;gap:4px;padding:6px 10px;background:var(--sf);border-top:1px solid var(--bd)}
.frame-nav button{background:var(--sf2);color:var(--tx);border:1px solid var(--bd);width:28px;height:26px;border-radius:4px;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center}
.frame-nav button:hover{background:var(--pr2);border-color:var(--pr)}
.frame-nav .finfo{flex:1;text-align:center;font-size:11px;color:var(--txm);font-variant-numeric:tabular-nums}

/* 3D panel */
.view3d{flex:1;position:relative}
.view3d canvas{width:100%;height:100%;display:block}
.pcd-info{position:absolute;top:8px;left:8px;background:rgba(255,255,255,.88);padding:5px 12px;border-radius:6px;font-size:11px;color:var(--txm);font-variant-numeric:tabular-nums;backdrop-filter:blur(6px);border:1px solid var(--bd)}
.loading-overlay{position:absolute;inset:0;background:rgba(247,248,250,.8);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px;z-index:10;backdrop-filter:blur(4px)}
.loading-overlay.hidden{display:none}
.spinner{width:32px;height:32px;border:3px solid var(--bd);border-top-color:var(--pr);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{color:var(--txm);font-size:12px}

/* ── export toast ── */
#export-toast{position:fixed;bottom:24px;right:24px;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:10px 16px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);z-index:100;display:none;max-width:380px;word-break:break-all}
#export-toast.show{display:block}
#export-toast.ok{border-color:var(--gn);color:var(--gn)}
#export-toast.err{border-color:var(--rd);color:var(--rd)}
</style>
</head>
<body>

<div class="main-area">
  <!-- sidebar -->
  <div class="sidebar">
    <div class="logo">Spatial <span>Benchmark</span> Viewer</div>
    <div class="filter-section">
      <h3>Filters</h3>
      <div class="filter-row">
        <select id="f-dataset"><option value="">All Datasets</option></select>
        <select id="f-density"><option value="">All Density</option></select>
      </div>
      <div class="filter-row">
        <select id="f-env"><option value="">All Env</option></select>
        <select id="f-dynamics"><option value="">All Dynamics</option></select>
      </div>
      <div class="filter-row">
        <select id="f-viewtype"><option value="">All View Type</option></select>
        <select id="f-data_type"><option value="">All Sim/Real</option></select>
      </div>
    </div>
    <div class="scene-count" id="scene-count">0 scenes</div>
    <div class="scene-list" id="scene-list"></div>
  </div>

  <!-- content -->
  <div class="content">
    <!-- toolbar -->
    <div class="toolbar">
      <button class="btn on" id="btn-images" title="Toggle image panel">Images</button>
      <button class="btn on" id="btn-cameras" title="Toggle camera frustums">Cameras</button>
      <button class="btn" id="btn-axes" title="Toggle axes">Axes</button>
      <button class="btn on" id="btn-depthmask" title="Toggle depth mask">Depth Mask</button>
      <div class="sep"></div>
      <div class="ctrl-group">
        <label>Downsample</label>
        <input type="range" id="r-ds" min="1" max="8" step="1" value="4">
        <span class="val" id="v-ds">4</span>
      </div>
      <div class="ctrl-group">
        <label>Point Size</label>
        <input type="range" id="r-psize" min="1" max="50" step="1" value="8">
        <span class="val" id="v-psize">0.008</span>
      </div>
      <div class="ctrl-group">
        <label>Z Far</label>
        <input type="range" id="r-zfar" min="0.5" max="100" step="0.5" value="10">
        <span class="val" id="v-zfar">10.0</span>
      </div>
      <div class="ctrl-group" id="conf-group" style="display:none">
        <label>Conf</label>
        <input type="range" id="r-conf" min="0" max="1" step="0.05" value="0">
        <span class="val" id="v-conf">0.00</span>
      </div>
      <div class="sep"></div>
      <div class="ctrl-group">
        <label>Frustum</label>
        <input type="range" id="r-fscale" min="0.005" max="2.0" step="0.005" value="0.04">
        <span class="val" id="v-fscale">0.040</span>
      </div>
      <div class="sep"></div>
      <button class="btn" id="btn-export-glb" title="Export current scene as GLB to glb_output/">Export GLB</button>
      <span class="badge" id="status" style="color:var(--gn)">Ready</span>
    </div>

    <!-- panels -->
    <div class="panels">
      <!-- image panel -->
      <div class="img-panel" id="img-panel">
        <div class="img-sec"><div class="img-lbl">RGB</div><img id="img-rgb" alt="RGB"></div>
        <div class="img-sec"><div class="img-lbl">Depth</div><img id="img-depth" alt="Depth"></div>
        <div class="frame-nav">
          <button id="btn-prev">&#9664;</button>
          <span class="finfo" id="frame-info">- / -</span>
          <button id="btn-next">&#9654;</button>
        </div>
      </div>

      <!-- 3D view -->
      <div class="view3d">
        <canvas id="cv"></canvas>
        <div class="pcd-info" id="pcd-info">Select a scene</div>
        <div class="loading-overlay hidden" id="loading">
          <div class="spinner"></div>
          <div class="loading-text">Loading point cloud...</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script type="importmap">{"imports":{"three":"https://esm.sh/three@0.169.0","three/addons/":"https://esm.sh/three@0.169.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const $=id=>document.getElementById(id);
const json=u=>fetch(u).then(r=>r.json());

// ── State ──
const S={
  sceneId:null, sceneData:null,
  frameIdx:0, numFrames:0,
  ds:4, ptSize:.008, zFar:10, confThreshold:0,
  showImages:true, showCams:true, showAxes:false, depthMask:true,
  frustumScale:0.04, dataset:'',
};
let datasetZFar={}; // dataset_name -> default z_far

// ── Three.js ──
const canvas=$('cv');
const scene3=new THREE.Scene();
scene3.background=new THREE.Color(0xf7f8fa);
const cam3=new THREE.PerspectiveCamera(60,1,.005,500);
cam3.position.set(0,-0.5,-2);
cam3.up.set(0,-1,0);
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
const ctrl3=new OrbitControls(cam3,canvas);
ctrl3.enableDamping=true;ctrl3.dampingFactor=.12;

// Grid + axes
const gridHelper=new THREE.GridHelper(10,20,0xcfd4de,0xcfd4de);
gridHelper.rotation.x=Math.PI/2;
scene3.add(gridHelper);
const axesHelper=new THREE.AxesHelper(.3);
axesHelper.visible=false;
scene3.add(axesHelper);
scene3.add(new THREE.AmbientLight(0xffffff,.5));

// Point cloud
const pGeo=new THREE.BufferGeometry();
const pMat=new THREE.PointsMaterial({size:S.ptSize,vertexColors:true,sizeAttenuation:true});
const pObj=new THREE.Points(pGeo,pMat);
scene3.add(pObj);

// Camera frustums group
const camGroup=new THREE.Group();
scene3.add(camGroup);

function resize(){
  const r=canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width,r.height);
  cam3.aspect=r.width/r.height;
  cam3.updateProjectionMatrix();
}
window.addEventListener('resize',resize);
(function anim(){requestAnimationFrame(anim);ctrl3.update();renderer.render(scene3,cam3);})();

// ── Frustum drawing ──
function hsvToHex(i, n){
  const h=(i+0.5)/Math.max(n,1), s=0.85, v=0.95;
  const c=v*s, x=c*(1-Math.abs(h*6%2-1)), m=v-c;
  let r,g,b;
  const h6=h*6;
  if(h6<1){r=c;g=x;b=0;}else if(h6<2){r=x;g=c;b=0;}
  else if(h6<3){r=0;g=c;b=x;}else if(h6<4){r=0;g=x;b=c;}
  else if(h6<5){r=x;g=0;b=c;}else{r=c;g=0;b=x;}
  return(Math.round((r+m)*255)<<16)|(Math.round((g+m)*255)<<8)|Math.round((b+m)*255);
}

function drawFrustums(poses, K, activeIdx){
  while(camGroup.children.length){
    const c=camGroup.children[0];
    c.geometry?.dispose();c.material?.dispose();camGroup.remove(c);
  }
  if(!S.showCams) return;

  const fx=K[0],fy=K[4],cx=K[6],cy=K[7];
  const w=640,h=480,sc=S.frustumScale;
  const n=poses.length/12|0;

  for(let i=0;i<n;i++){
    const p=poses.slice(i*12,(i+1)*12);
    const R=new THREE.Matrix3();
    R.set(p[0],p[1],p[2], p[4],p[5],p[6], p[8],p[9],p[10]);
    const t=new THREE.Vector3(p[3],p[7],p[11]);

    const corners=[[0,0],[w,0],[w,h],[0,h]].map(([u,v])=>[
      (u-cx)/fx*sc,(v-cy)/fy*sc,sc
    ]);
    const o=t.clone();
    const wc=corners.map(c=>{const v=new THREE.Vector3(...c);v.applyMatrix3(R);v.add(t);return v;});

    const pts=[];
    for(const c of wc){pts.push(o.x,o.y,o.z,c.x,c.y,c.z);}
    for(let j=0;j<4;j++){const a=wc[j],b=wc[(j+1)%4];pts.push(a.x,a.y,a.z,b.x,b.y,b.z);}

    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
    const isActive=i===activeIdx;
    const color=isActive?0xffffff:hsvToHex(i,n);
    const frust=new THREE.LineSegments(g,new THREE.LineBasicMaterial({
      color,linewidth:isActive?2:1,opacity:isActive?1:.7,transparent:true
    }));
    camGroup.add(frust);
  }
}

// ── Filter UI ──
const fDataset=$('f-dataset'), fDensity=$('f-density'), fEnv=$('f-env'), fDyn=$('f-dynamics'), fView=$('f-viewtype'), fSim=$('f-data_type');
const sceneList=$('scene-list'), sceneCount=$('scene-count');

async function initFilters(){
  const tags=await json('/api/tags');
  const ds=await json('/api/datasets');
  ds.forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;fDataset.appendChild(o);});
  (tags.view_density||[]).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;fDensity.appendChild(o);});
  (tags.environment||[]).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;fEnv.appendChild(o);});
  (tags.dynamics||[]).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;fDyn.appendChild(o);});
  (tags.view_type||[]).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;fView.appendChild(o);});
  (tags.data_type||[]).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;fSim.appendChild(o);});
}

async function loadSceneList(){
  const params=new URLSearchParams();
  if(fDataset.value)params.set('dataset',fDataset.value);
  if(fDensity.value)params.set('view_density',fDensity.value);
  if(fEnv.value)params.set('environment',fEnv.value);
  if(fDyn.value)params.set('dynamics',fDyn.value);
  if(fView.value)params.set('view_type',fView.value);
  if(fSim.value)params.set('data_type',fSim.value);

  const scenes=await json('/api/scenes?'+params);
  sceneCount.textContent=`${scenes.length} scenes`;
  sceneList.innerHTML='';

  for(const s of scenes){
    const div=document.createElement('div');
    div.className='scene-item';
    if(s.scene_id===S.sceneId) div.classList.add('active');
    div.innerHTML=`
      <div class="name">${s.scene_id}</div>
      <div class="meta">
        <span class="tag ds">${s.source_dataset}</span>
        <span class="tag">${s.tags.view_density||'?'}</span>
        <span class="tag">${s.tags.environment||'?'}</span>
        <span>${s.num_frames} frames</span>
      </div>
    `;
    div.onclick=()=>selectScene(s.scene_id, s.num_frames, s.source_dataset);
    sceneList.appendChild(div);
  }
}

[fDataset,fDensity,fEnv,fDyn,fView,fSim].forEach(el=>el.onchange=loadSceneList);

// ── Scene selection ──
let loadSeq=0;

async function selectScene(sceneId, numFrames, dataset, keepZFar){
  const sceneChanged = sceneId !== S.sceneId;
  S.sceneId=sceneId;
  S.numFrames=numFrames;
  if(sceneChanged) S.frameIdx=0;
  updateFrameInfo();

  // Auto-set z_far from dataset default only on scene change, not on slider adjust
  if(!keepZFar && sceneChanged && dataset && datasetZFar[dataset]!=null){
    S.zFar=datasetZFar[dataset];
    $('r-zfar').value=S.zFar;
    $('v-zfar').textContent=S.zFar.toFixed(1);
  }

  // Show confidence slider only for ropedia
  S.dataset=dataset||'';
  $('conf-group').style.display=(dataset==='ropedia')?'flex':'none';

  // Highlight in list
  document.querySelectorAll('.scene-item').forEach(el=>{
    el.classList.toggle('active',el.querySelector('.name').textContent===sceneId);
  });

  // Load point cloud + cameras
  const seq=++loadSeq;
  $('loading').classList.remove('hidden');
  $('status').textContent='Loading...';$('status').style.color='var(--pr)';

  try{
    const [pcdBuf,camBuf]=await Promise.all([
      fetch(`/api/scene/${sceneId}/pcd?z_far=${S.zFar}&downsample=${S.ds}&depth_mask=${S.depthMask}&conf_threshold=${S.confThreshold}`).then(r=>r.arrayBuffer()),
      fetch(`/api/scene/${sceneId}/cameras?z_far=${S.zFar}&depth_mask=${S.depthMask}&conf_threshold=${S.confThreshold}`).then(r=>r.arrayBuffer()),
    ]);
    if(seq!==loadSeq)return;

    // Parse point cloud
    const dv=new DataView(pcdBuf);
    const N=dv.getInt32(0,true);
    const pos=new Float32Array(pcdBuf,4,N*3);
    const colU8=new Uint8Array(pcdBuf,4+N*12,N*3);
    const col=new Float32Array(N*3);
    for(let i=0;i<colU8.length;i++)col[i]=colU8[i]/255;

    pGeo.setAttribute('position',new THREE.BufferAttribute(pos,3));
    pGeo.setAttribute('color',new THREE.BufferAttribute(col,3));
    pGeo.computeBoundingSphere();
    $('pcd-info').textContent=`${N.toLocaleString()} points | ${numFrames} frames`;

    // Parse cameras
    const cdv=new DataView(camBuf);
    const nCams=cdv.getInt32(0,true);
    const camPoses=new Float32Array(camBuf,4,nCams*12);
    const camK=new Float32Array(camBuf,4+nCams*48,9);
    S.camPoses=camPoses;
    S.camK=camK;
    S.nCams=nCams;
    drawFrustums(camPoses,camK,S.frameIdx);

    // Auto-center camera
    if(pGeo.boundingSphere){
      const c=pGeo.boundingSphere.center;
      const r=pGeo.boundingSphere.radius;
      ctrl3.target.copy(c);
      cam3.position.set(c.x,c.y-r*.8,c.z-r*2);
      ctrl3.update();
    }

    // Load frame images
    loadFrameImages();

    $('status').textContent='Ready';$('status').style.color='var(--gn)';
  }catch(e){
    console.error(e);
    if(seq===loadSeq){$('status').textContent='Error';$('status').style.color='var(--rd)';}
  }finally{
    if(seq===loadSeq)$('loading').classList.add('hidden');
  }
}

function loadFrameImages(){
  if(!S.sceneId)return;
  $('img-rgb').src=`/api/scene/${S.sceneId}/rgb/${S.frameIdx}?z_far=${S.zFar}&depth_mask=${S.depthMask}&conf_threshold=${S.confThreshold}`;
  $('img-depth').src=`/api/scene/${S.sceneId}/depth_viz/${S.frameIdx}?z_far=${S.zFar}&depth_mask=${S.depthMask}&conf_threshold=${S.confThreshold}`;
  // Update frustum highlight
  if(S.camPoses) drawFrustums(S.camPoses,S.camK,S.frameIdx);
}

function updateFrameInfo(){
  $('frame-info').textContent=`${S.frameIdx+1} / ${S.numFrames}`;
}

$('btn-prev').onclick=()=>{if(S.numFrames>0){S.frameIdx=Math.max(0,S.frameIdx-1);updateFrameInfo();loadFrameImages();}};
$('btn-next').onclick=()=>{if(S.numFrames>0){S.frameIdx=Math.min(S.numFrames-1,S.frameIdx+1);updateFrameInfo();loadFrameImages();}};

// ── Toolbar ──
$('btn-images').onclick=()=>{
  S.showImages=!S.showImages;
  $('btn-images').classList.toggle('on',S.showImages);
  $('img-panel').classList.toggle('hidden',!S.showImages);
  resize();
};
$('btn-cameras').onclick=()=>{
  S.showCams=!S.showCams;
  $('btn-cameras').classList.toggle('on',S.showCams);
  if(S.camPoses) drawFrustums(S.camPoses,S.camK,S.frameIdx);
};
$('btn-axes').onclick=()=>{
  S.showAxes=!S.showAxes;
  $('btn-axes').classList.toggle('on',S.showAxes);
  axesHelper.visible=S.showAxes;
};
$('btn-depthmask').onclick=()=>{
  S.depthMask=!S.depthMask;
  $('btn-depthmask').classList.toggle('on',S.depthMask);
  if(S.sceneId)selectScene(S.sceneId,S.numFrames,S.dataset,true);
};

$('r-ds').oninput=function(){S.ds=+this.value;$('v-ds').textContent=S.ds;};
$('r-ds').onchange=()=>{if(S.sceneId)selectScene(S.sceneId,S.numFrames,S.dataset,true);};

$('r-psize').oninput=function(){
  const v=+this.value/1000;
  S.ptSize=v;$('v-psize').textContent=v.toFixed(3);
  pMat.size=v;
};

$('r-zfar').oninput=function(){S.zFar=+this.value;$('v-zfar').textContent=S.zFar.toFixed(1);};
$('r-zfar').onchange=()=>{if(S.sceneId)selectScene(S.sceneId,S.numFrames,S.dataset,true);};

$('r-conf').oninput=function(){S.confThreshold=+this.value;$('v-conf').textContent=S.confThreshold.toFixed(2);};
$('r-conf').onchange=()=>{if(S.sceneId)selectScene(S.sceneId,S.numFrames,S.dataset,true);};

$('r-fscale').oninput=function(){
  S.frustumScale=+this.value;
  $('v-fscale').textContent=S.frustumScale.toFixed(3);
  if(S.camPoses) drawFrustums(S.camPoses,S.camK,S.frameIdx);
};

// ── Export GLB ──
let _toastTimer=null;
function showToast(msg, ok=true){
  const el=$('export-toast');
  el.textContent=msg;
  el.className='show '+(ok?'ok':'err');
  if(_toastTimer) clearTimeout(_toastTimer);
  _toastTimer=setTimeout(()=>{el.className='';},6000);
}

$('btn-export-glb').onclick=async()=>{
  if(!S.sceneId){showToast('No scene selected',false);return;}
  const btn=$('btn-export-glb');
  btn.disabled=true;btn.textContent='Exporting…';
  try{
    const params=new URLSearchParams({
      z_far:S.zFar, downsample:S.ds,
      depth_mask:S.depthMask, conf_threshold:S.confThreshold,
      frustum_scale:S.frustumScale,
    });
    const res=await fetch(`/api/scene/${S.sceneId}/export_glb?`+params,{method:'POST'});
    const data=await res.json();
    if(res.ok && data.status==='ok'){
      showToast(`✓ ${data.path}  ${data.num_points.toLocaleString()} pts  ${data.size_mb}MB  frustum=${data.frustum_scale}m`,true);
    } else {
      showToast('Export failed: '+(data.error||res.statusText),false);
    }
  }catch(e){
    showToast('Export error: '+e,false);
  }finally{
    btn.disabled=false;btn.textContent='Export GLB';
  }
};

// ── Keyboard ──
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT')return;
  switch(e.key){
    case 'ArrowLeft':e.preventDefault();$('btn-prev').click();break;
    case 'ArrowRight':e.preventDefault();$('btn-next').click();break;
  }
});

// ── Init ──
resize();
datasetZFar=await json('/api/dataset_z_far');
await initFilters();
await loadSceneList();
</script>
<div id="export-toast"></div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global SCENE_INDEX_DIR, SCENE_INDEX_PATH, BENCHMARK_ROOT

    parser = argparse.ArgumentParser(description="Benchmark GT Point Cloud Web Viewer")
    parser.add_argument("--scene-index", type=str, default=None,
                        help="Scene index JSON, e.g. benchmark/scene_indices/all_scenes.json")
    parser.add_argument("--scene-index-dir", type=str, default="benchmark/scene_indices",
                        help="Directory containing all_scenes.json (kept for backward compatibility)")
    parser.add_argument("--benchmark-root", type=str, default="SpatialBenchmark",
                        help="SpatialBenchmark root containing single/sparse/medium/dense")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    SCENE_INDEX_DIR = Path(args.scene_index_dir)
    SCENE_INDEX_PATH = (
        Path(args.scene_index)
        if args.scene_index else
        SCENE_INDEX_DIR / "all_scenes.json"
    )
    BENCHMARK_ROOT = Path(args.benchmark_root)

    _load_all_scenes()
    _init_benchmark_dataset()

    print(f"Scene index : {SCENE_INDEX_PATH.resolve()}")
    print(f"Benchmark   : {BENCHMARK_ROOT.resolve()}")
    print(f"Total scenes: {len(_all_scenes)}")
    print(f"Datasets    : {_dataset_names}")
    print(f"Server      : http://localhost:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
