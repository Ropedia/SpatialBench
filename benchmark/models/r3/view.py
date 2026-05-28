"""Viser viewer for inference run outputs.

Two modes (chosen automatically from --data_dir):

* **Single run**: path is a run dir (contains `depth/` or staged `points.npz`). Loads it on startup.
* **Batch parent**: path is a parent dir (e.g. `scratch/demo`). Discovers all
  child run dirs and exposes a dropdown to lazy-load any sequence on demand.

Loading is on-the-fly from per-frame `depth/`, `color/`, `camera/`, `conf/` files, or
from a staged sparse `points.npz` reference cloud plus `camera/*.npz` files. The
on-disk artifacts stay full-density. All downsampling happens here, controlled by
GUI sliders.

Features:
  * Per-frame view: show only the point cloud from the current frame.
  * Frame-by-frame playback at user-selected fps.
  * Click any camera frustum to jump the viewer there.
  * Tilt / roll / FOV tweaks applied on top of the current view.
  * Keypoint trajectory: snapshot view poses, slerp/lerp between them, play or render mp4.
  * Single-frame video: scrub through frames from a fixed viewpoint and dump mp4.

Usage:
    conda activate r3
    python view.py --data_dir scratch/demo/indoor_metric
    python view.py --data_dir scratch/demo
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

import imageio.v2 as iio
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import viser  # noqa: E402
import viser.transforms as vt  # noqa: E402

from R3.utils.render_io import load_frame, load_frames  # noqa: E402
from R3.utils.components_geometry import depth_to_cam_coords_points  # noqa: E402


class _FrameRenderError(RuntimeError):
    """Wrap a per-frame render failure so encoder fallback does not mask it."""

    def __init__(self, frame_idx, cause):
        self.frame_idx = frame_idx
        self.cause = cause
        super().__init__(f"frame {frame_idx}: {cause}")


def _write_video_streaming(out_path, n_frames, fps, render_frame, on_progress=None):
    """Render and write video frames one at a time, retrying without libx264 on encoder failure."""
    fps_int = int(round(fps))
    libx264_error = None
    for codec in ("libx264", None):
        writer = None
        try:
            kwargs = {"fps": fps_int}
            if codec is not None:
                kwargs["codec"] = codec
            writer = iio.get_writer(out_path, **kwargs)
            for frame_idx in range(n_frames):
                try:
                    img = render_frame(frame_idx)
                except Exception as e:
                    raise _FrameRenderError(frame_idx, e) from e
                writer.append_data(img)
                if on_progress is not None:
                    on_progress(frame_idx)
            writer.close()
            writer = None
            return
        except _FrameRenderError:
            raise
        except Exception as e:
            if codec is not None:
                libx264_error = e
                continue
            if libx264_error is not None:
                raise RuntimeError(
                    f"default video writer failed after libx264 failed; libx264 error: {libx264_error}; "
                    f"default writer error: {e}"
                ) from e
            raise
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass


def _frustum_color(idx: int, total: int):
    """Rainbow HSV color cycling so the camera trajectory is visually distinguishable."""
    h = (idx / max(1, total)) % 1.0
    c = 1.0
    x = (1 - abs((h * 6) % 2 - 1)) * c
    if h < 1 / 6:
        r, g, b = c, x, 0.0
    elif h < 2 / 6:
        r, g, b = x, c, 0.0
    elif h < 3 / 6:
        r, g, b = 0.0, c, x
    elif h < 4 / 6:
        r, g, b = 0.0, x, c
    elif h < 5 / 6:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return (int(r * 255), int(g * 255), int(b * 255))


def _frame_to_world(frame):
    """Lift one frame's depth/color/conf into world-space arrays."""
    pts_cam = depth_to_cam_coords_points(frame["depth"], frame["intrinsics"])
    pose = frame["pose"]
    pts_world = (pts_cam.reshape(-1, 3) @ pose[:3, :3].T) + pose[:3, 3]
    colors = frame["color"].reshape(-1, 3).astype(np.uint8)
    conf = frame["conf"].reshape(-1).astype(np.float32)
    return pts_world.astype(np.float32), colors, conf


def _load_one_worker(data_dir, fid, conf_threshold_initial):
    """Module-level for ProcessPoolExecutor pickling. Loads + reprojects + conf-filters one frame."""
    frame = load_frame(data_dir, fid)
    pts, colors, conf = _frame_to_world(frame)
    mask = conf > conf_threshold_initial
    return (
        pts[mask],
        colors[mask],
        conf[mask],
        frame["pose"].astype(np.float32),
        frame["intrinsics"].astype(np.float32),
    )


def load_dense(data_dir, conf_threshold_initial=1.0, verbose=True, num_workers=16, progress_cb=None):
    """Read all frames in a run dir and return aggregated arrays plus per-frame offsets."""
    sparse_path = os.path.join(data_dir, "points.npz")
    if os.path.isfile(sparse_path):
        return load_sparse_reference(data_dir, sparse_path, verbose=verbose)

    frame_ids = load_frames(data_dir)
    if not frame_ids:
        raise ValueError(f"No frames found under {data_dir}/depth/")

    n = len(frame_ids)
    pts_list = [None] * n
    col_list = [None] * n
    conf_list = [None] * n
    cam_c2ws = [None] * n
    cam_intrinsics = [None] * n
    t0 = time.time()
    workers = max(1, min(int(num_workers), n))
    # ProcessPoolExecutor sidesteps the GIL — depth_to_cam_coords_points is the bottleneck,
    # not disk I/O. Pickle overhead is small relative to per-frame reprojection cost.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_load_one_worker, data_dir, fid, conf_threshold_initial): i for i, fid in enumerate(frame_ids)}
        done = 0
        # Collect in submission order so the per-frame arrays line up with frame_ids.
        for fut, i in list(futures.items()):
            pts, colors, conf, pose, K = fut.result()
            pts_list[i] = pts
            col_list[i] = colors
            conf_list[i] = conf
            cam_c2ws[i] = pose
            cam_intrinsics[i] = K
            done += 1
            if progress_cb is not None and (done % 25 == 0 or done == n):
                progress_cb(done, n)
            if verbose and done % 100 == 0:
                print(f"  loaded {done}/{n} frames ({time.time() - t0:.1f}s)")

    counts = [len(p) for p in pts_list]
    frame_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    pts = np.concatenate(pts_list) if pts_list else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate(col_list) if col_list else np.zeros((0, 3), dtype=np.uint8)
    conf = np.concatenate(conf_list) if conf_list else np.zeros((0,), dtype=np.float32)
    if verbose:
        print(f"  total points after conf>{conf_threshold_initial}: {len(pts):,} in {time.time() - t0:.1f}s")
    return {
        "points": pts,
        "colors": colors,
        "conf": conf,
        "cam_c2ws": np.stack(cam_c2ws),
        "cam_intrinsics": np.stack(cam_intrinsics),
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        "frame_offsets": frame_offsets,
        "has_per_frame_points": True,
    }


def load_sparse_reference(data_dir, sparse_path, verbose=True):
    """Read a staged sparse reference cloud and all camera frustums."""
    payload = np.load(sparse_path)
    pts = payload["points"].astype(np.float32)
    colors = payload["colors"]
    if colors.dtype == np.uint8:
        colors = colors.astype(np.uint8)
    else:
        colors = colors.astype(np.float32)
        if colors.max(initial=0.0) <= 1.0:
            colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    conf = np.full((len(pts),), 2.0, dtype=np.float32)

    cam_files = sorted(glob.glob(os.path.join(data_dir, "camera", "*.npz")))
    if not cam_files:
        raise ValueError(f"No cameras found under {data_dir}/camera/")
    cam_c2ws, cam_intrinsics, frame_ids = [], [], []
    for f in cam_files:
        cam = np.load(f)
        pose = cam["pose"]
        if pose.shape == (3, 4):
            pose_4x4 = np.eye(4, dtype=pose.dtype)
            pose_4x4[:3, :] = pose
            pose = pose_4x4
        cam_c2ws.append(pose.astype(np.float32))
        cam_intrinsics.append(cam["intrinsics"].astype(np.float32))
        frame_ids.append(int(os.path.splitext(os.path.basename(f))[0]))

    if verbose:
        print(f"  loaded sparse reference: {len(pts):,} points, {len(cam_c2ws)} cameras")
    return {
        "points": pts,
        "colors": colors,
        "conf": conf,
        "cam_c2ws": np.stack(cam_c2ws),
        "cam_intrinsics": np.stack(cam_intrinsics),
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        "frame_offsets": np.array([0, len(pts)], dtype=np.int64),
        "has_per_frame_points": False,
    }


def _stride_downsample(pts, colors, conf, target_count):
    """Stride-decimate to about `target_count` points (no-op if already small enough)."""
    if target_count <= 0 or len(pts) <= target_count:
        return pts, colors, conf
    stride = max(1, len(pts) // target_count)
    return pts[::stride][:target_count], colors[::stride][:target_count], conf[::stride][:target_count]


def _filter_finite_cloud(pts, colors, conf):
    """Drop non-finite cloud rows before sending arrays to Viser."""
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
    if valid.all():
        return pts, colors, conf
    return pts[valid], colors[valid], conf[valid]


def _is_run_dir(path):
    """Return True if `path` looks like a dense run dir or staged sparse reference dir."""
    return os.path.isdir(path) and (
        os.path.isdir(os.path.join(path, "depth")) or os.path.isfile(os.path.join(path, "points.npz"))
    )


def _discover_runs(parent, method=None):
    """Map run-name -> path for single-run, batch-parent, or scene/method layouts.

    Layouts handled:
      * `parent` itself is a run dir.
      * `parent/<run>` (flat batch).
      * `parent/<scene>/<method>` (e.g. `demo_tmp/<scene>/da3`). When `method` is set,
        only that subdir is picked and named `<scene>`; otherwise all method subdirs
        are picked and named `<scene>/<method>`.
    """
    if _is_run_dir(parent):
        return {os.path.basename(parent.rstrip("/")): parent}
    runs = {}
    for name in sorted(os.listdir(parent)):
        path = os.path.join(parent, name)
        if not os.path.isdir(path):
            continue
        if _is_run_dir(path):
            runs[name] = path
            continue
        # scene dir holding multiple method subdirs
        for sub in sorted(os.listdir(path)):
            sub_path = os.path.join(path, sub)
            if not _is_run_dir(sub_path):
                continue
            if method is not None and sub != method:
                continue
            key = name if method is not None else f"{name}/{sub}"
            runs[key] = sub_path
    return runs


def _intrinsics_to_fov_aspect(intrinsics):
    """Convert pinhole intrinsics to (vertical FOV in radians, aspect ratio)."""
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    fov_y = 2.0 * np.arctan2(cy, fy)
    aspect = (cx / cy) * (fy / fx) if cy > 0 and fx > 0 else 1.0
    return fov_y, aspect


def _c2w_to_viser_pose(c2w):
    """Convert OpenCV c2w (4x4 or 3x4) to a Viser camera pose."""
    R = np.asarray(c2w[:3, :3], dtype=np.float64)
    pos = np.asarray(c2w[:3, 3], dtype=np.float64)
    wxyz = vt.SO3.from_matrix(R).parameters().astype(np.float64)
    return pos, wxyz


def _slerp_wxyz(q0, q1, t):
    """Spherical-linear interpolation between two wxyz quaternions (shortest path)."""
    a_params = np.asarray(q0, dtype=np.float64)
    b_params = np.asarray(q1, dtype=np.float64)
    if float(np.dot(a_params, b_params)) < 0:
        b_params = -b_params
    a = vt.SO3(a_params)
    b = vt.SO3(b_params)
    delta = a.inverse() @ b
    return (a @ vt.SO3.exp(t * delta.log())).parameters().astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Run output dir or parent dir of multiple runs")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument(
        "--initial_conf",
        type=float,
        default=1.0,
        help="Loose conf threshold applied at load (slider re-thresholds higher; load lower than the slider min so users can dial down)",
    )
    parser.add_argument(
        "--initial_max_points", type=int, default=1_500_000, help="Initial downsample target for the displayed cloud"
    )
    parser.add_argument(
        "--pc_frame_start",
        type=int,
        default=0,
        help="Initial inclusive frame index for aggregate point-cloud display",
    )
    parser.add_argument(
        "--pc_frame_end",
        type=int,
        default=-1,
        help="Initial inclusive frame index for aggregate point-cloud display (-1 means last frame)",
    )
    parser.add_argument(
        "--screenshot_dir",
        type=str,
        default="",
        help="Where to save screenshots / rendered videos (default: <data_dir>/screenshots/)",
    )
    parser.add_argument(
        "--auto_load_first", action="store_true", help="Auto-load the first sequence on startup (batch mode only)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="When --data_dir is a parent of <scene>/<method> dirs, restrict to this method subdir (e.g. da3).",
    )
    parser.add_argument(
        "--load_workers",
        type=int,
        default=16,
        help="Threads used to read per-frame depth/conf/color/camera files in parallel.",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=3,
        help="Number of recently-loaded sequences to keep in memory (LRU). 0 disables caching.",
    )
    parser.add_argument(
        "--up_direction",
        type=str,
        default="-y",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        help="World up direction for Viser navigation. DL3DV/nerfstudio scenes usually use +z.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.data_dir)
    runs = _discover_runs(root, method=args.method)
    if not runs:
        raise SystemExit(f"No run dirs (with depth/ or points.npz) found under {root}")
    is_batch = not _is_run_dir(root)

    screenshot_root = args.screenshot_dir or os.path.join(root, "screenshots")
    os.makedirs(screenshot_root, exist_ok=True)

    print(f"Discovered {len(runs)} run dir(s) under {root}:")
    for name in runs:
        print(f"  - {name}")

    print(f"Starting Viser server on port {args.port} ...")
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction(args.up_direction)

    state: dict = {
        "name": None,
        "data": None,
        "pc_handle": None,
        "cam_handles": {},
        "kp_handles": [],
        "keypoints": [],
        "playing": False,
        "screenshot_dir": screenshot_root,
        "cache": OrderedDict(),  # name -> data, LRU
    }

    # --- GUI layout -------------------------------------------------------------
    sequence_names = list(runs.keys())
    with server.gui.add_folder("Sequence"):
        gui_seq = (
            server.gui.add_dropdown("Sequence", options=sequence_names, initial_value=sequence_names[0])
            if is_batch
            else None
        )
        gui_load_btn = server.gui.add_button("Load sequence")
        gui_clear_btn = server.gui.add_button("Clear scene") if is_batch else None
        initial_status = (
            f"{len(runs)} run(s) discovered — pick one and click Load" if is_batch else "click Load to begin"
        )
        gui_status = server.gui.add_text("Status", initial_value=initial_status)

    with server.gui.add_folder("Display"):
        gui_n_points = server.gui.add_slider(
            "Displayed points (k)",
            min=10,
            max=20000,
            step=10,
            initial_value=max(10, min(args.initial_max_points // 1000, 1500)),
        )
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.0005,
            max=0.05,
            step=0.0005,
            initial_value=0.001,
        )
        gui_conf = server.gui.add_slider(
            "Conf threshold",
            min=1.0,
            max=10.0,
            step=0.05,
            initial_value=max(2.0, float(args.initial_conf)),
        )

    with server.gui.add_folder("Cameras"):
        gui_show_cams = server.gui.add_checkbox("Show camera frustums", initial_value=True)
        gui_cam_size = server.gui.add_slider(
            "Frustum size",
            min=0.001,
            max=2.0,
            step=0.001,
            initial_value=0.01,
        )
        gui_cam_stride = server.gui.add_slider(
            "Frustum stride",
            min=1,
            max=50,
            step=1,
            initial_value=1,
        )

    with server.gui.add_folder("Frames / playback"):
        gui_single = server.gui.add_checkbox("Single-frame mode", initial_value=False)
        gui_frame = server.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0)
        gui_pc_start = server.gui.add_slider("Cloud frame start", min=0, max=0, step=1, initial_value=0)
        gui_pc_end = server.gui.add_slider("Cloud frame end", min=0, max=0, step=1, initial_value=0)
        gui_play_fps = server.gui.add_slider("Play FPS", min=1, max=60, step=1, initial_value=10)
        gui_play_btn = server.gui.add_button("Play / Pause")
        gui_goto_btn = server.gui.add_button("Go to current frame's camera")
        gui_render_singles_btn = server.gui.add_button("Render single-frame video")

    with server.gui.add_folder("Keypoints / trajectory"):
        gui_kp_count = server.gui.add_text("Keypoints", initial_value="0", disabled=True)
        gui_kp_add_btn = server.gui.add_button("Add keypoint (current view)")
        gui_kp_clear_btn = server.gui.add_button("Clear keypoints")
        gui_traj_dur = server.gui.add_slider("Duration (s)", min=1.0, max=60.0, step=0.5, initial_value=8.0)
        gui_traj_fps = server.gui.add_slider("Render FPS", min=1, max=60, step=1, initial_value=24)
        gui_traj_loop = server.gui.add_checkbox("Loop back to first", initial_value=False)
        gui_traj_play_btn = server.gui.add_button("Play trajectory")
        gui_traj_render_btn = server.gui.add_button("Render trajectory video")

    with server.gui.add_folder("Navigation"):
        gui_align_up_btn = server.gui.add_button("Set world up to current camera")
        gui_move_speed = server.gui.add_slider(
            "Move step (scene units)",
            min=0.001,
            max=5.0,
            step=0.001,
            initial_value=0.05,
        )
        gui_move_fwd_btn = server.gui.add_button("Forward")
        gui_move_back_btn = server.gui.add_button("Backward")
        gui_move_left_btn = server.gui.add_button("Left")
        gui_move_right_btn = server.gui.add_button("Right")
        gui_move_up_btn = server.gui.add_button("Up")
        gui_move_down_btn = server.gui.add_button("Down")

    with server.gui.add_folder("View tweaks"):
        gui_view_fov = server.gui.add_slider("FOV (deg)", min=10.0, max=120.0, step=0.5, initial_value=60.0)
        gui_view_tilt = server.gui.add_slider("Tilt (deg)", min=-89.0, max=89.0, step=0.5, initial_value=0.0)
        gui_view_roll = server.gui.add_slider("Roll (deg)", min=-180.0, max=180.0, step=0.5, initial_value=0.0)
        gui_view_apply_btn = server.gui.add_button("Apply tweaks (then reset sliders)")

    with server.gui.add_folder("Screenshot"):
        gui_shot_w = server.gui.add_slider("Width", min=320, max=3840, step=10, initial_value=1280)
        gui_shot_h = server.gui.add_slider("Height", min=240, max=2160, step=10, initial_value=720)
        gui_shot_btn = server.gui.add_button("Take screenshot")

    with server.gui.add_folder("Saved views"):
        gui_view_name = server.gui.add_text("Name", initial_value="view")
        gui_view_save_btn = server.gui.add_button("Save view (current camera)")
        gui_view_load_path = server.gui.add_text("Load path", initial_value="")
        gui_view_load_btn = server.gui.add_button("Load view")

    # --- Helpers -----------------------------------------------------------------
    def _first_client():
        clients = server.get_clients()
        if not clients:
            return None
        return next(iter(clients.values()))

    def _clear_scene():
        state["playing"] = False
        state["data"] = None
        if state["pc_handle"] is not None:
            state["pc_handle"].remove()
            state["pc_handle"] = None
        for h in state["cam_handles"].values():
            h.remove()
        state["cam_handles"] = {}
        for h in state["kp_handles"]:
            h.remove()
        state["kp_handles"] = []
        state["keypoints"] = []
        gui_kp_count.value = "0"
        try:
            gui_frame.value = 0
        except Exception:
            pass
        state["name"] = None

    def _frame_slice(idx):
        data = state["data"]
        if not data.get("has_per_frame_points", True):
            return data["points"], data["colors"], data["conf"]
        offs = data["frame_offsets"]
        a, b = int(offs[idx]), int(offs[idx + 1])
        return data["points"][a:b], data["colors"][a:b], data["conf"][a:b]

    def _frame_range_slice(start_idx, end_idx):
        """Return points from an inclusive frame-index range for dense per-frame runs."""
        data = state["data"]
        if not data.get("has_per_frame_points", True):
            return data["points"], data["colors"], data["conf"], "all frames"
        n_frames = len(data["cam_c2ws"])
        lo = max(0, min(int(start_idx), int(end_idx), n_frames - 1))
        hi = min(n_frames - 1, max(int(start_idx), int(end_idx), 0))
        offs = data["frame_offsets"]
        a, b = int(offs[lo]), int(offs[hi + 1])
        return data["points"][a:b], data["colors"][a:b], data["conf"][a:b], f"frames {lo}-{hi}"

    def _refresh_pointcloud():
        data = state["data"]
        if data is None:
            return
        thr = float(gui_conf.value)
        target = int(gui_n_points.value) * 1000
        if gui_single.value:
            idx = int(gui_frame.value)
            idx = max(0, min(idx, len(data["cam_c2ws"]) - 1))
            pts_all, cols_all, conf_all = _frame_slice(idx)
            label = f"frame {idx}"
        else:
            pts_all, cols_all, conf_all, label = _frame_range_slice(gui_pc_start.value, gui_pc_end.value)
        mask = conf_all > thr
        pts = pts_all[mask]
        cols = cols_all[mask]
        conf = conf_all[mask]
        pts, cols, conf = _filter_finite_cloud(pts, cols, conf)
        pts_ds, cols_ds, _ = _stride_downsample(pts, cols, conf, target)
        if state["pc_handle"] is not None:
            state["pc_handle"].remove()
        state["pc_handle"] = server.scene.add_point_cloud(
            "/cloud",
            points=pts_ds,
            colors=cols_ds,
            point_size=float(gui_point_size.value),
        )
        gui_status.value = f"[{state['name']}] {label}: {len(pts_ds):,} / {len(pts):,} pts (conf>{thr:.2f})"

    def _refresh_cameras():
        data = state["data"]
        if data is None:
            return
        for h in state["cam_handles"].values():
            h.remove()
        state["cam_handles"] = {}
        if not gui_show_cams.value:
            return
        scale = float(gui_cam_size.value)
        c2ws = data["cam_c2ws"]
        intr = data["cam_intrinsics"]
        n = len(c2ws)
        cam_stride = max(1, int(gui_cam_stride.value))
        for idx in range(0, n, cam_stride):
            try:
                fov_y, aspect = _intrinsics_to_fov_aspect(intr[idx])
                pos, wxyz = _c2w_to_viser_pose(c2ws[idx])
                if not (
                    np.isfinite(fov_y) and np.isfinite(aspect) and np.isfinite(pos).all() and np.isfinite(wxyz).all()
                ):
                    continue
            except Exception:
                continue
            handle = server.scene.add_camera_frustum(
                f"/cams/{idx:05d}",
                fov=float(fov_y),
                aspect=float(aspect),
                scale=scale,
                color=_frustum_color(idx, n),
                position=pos,
                wxyz=wxyz,
            )
            handle.on_click(_make_frustum_click(idx))
            state["cam_handles"][idx] = handle

    def _make_frustum_click(idx):
        def _cb(event):
            client = getattr(event, "client", None) or _first_client()
            if client is None:
                return
            _goto_frame(client, idx)

        return _cb

    def _goto_frame(client, idx):
        data = state["data"]
        c2w = data["cam_c2ws"][idx]
        K = data["cam_intrinsics"][idx]
        pos, wxyz = _c2w_to_viser_pose(c2w)
        fov_y, _ = _intrinsics_to_fov_aspect(K)
        with client.atomic():
            client.camera.position = pos
            client.camera.wxyz = wxyz
            client.camera.fov = float(fov_y)
        gui_status.value = f"viewer -> camera at frame {idx}"

    def _load_selected():
        name = gui_seq.value if gui_seq is not None else next(iter(runs))
        path = runs[name]
        state["playing"] = False
        cached = state["cache"].pop(name, None)
        if cached is not None:
            state["cache"][name] = cached  # bump to MRU
            data = cached
            gui_status.value = f"loaded {name} from cache"
        else:
            gui_status.value = f"loading {name} ..."
            t0 = time.time()

            def _on_progress(done, total):
                gui_status.value = f"loading {name}: {done}/{total} frames"

            try:
                data = load_dense(
                    path,
                    conf_threshold_initial=args.initial_conf,
                    num_workers=int(args.load_workers),
                    progress_cb=_on_progress,
                )
            except Exception as e:
                gui_status.value = f"load failed: {e}"
                return
            gui_status.value = f"loaded {name} in {time.time() - t0:.1f}s"
            if int(args.cache_size) > 0:
                state["cache"][name] = data
                while len(state["cache"]) > int(args.cache_size):
                    state["cache"].popitem(last=False)
        _clear_scene()
        state["name"] = name
        state["data"] = data
        n_full = len(data["points"])
        n_frames = len(data["cam_c2ws"])
        try:
            gui_n_points.max = max(10, n_full // 1000)
        except Exception:
            pass
        try:
            gui_frame.max = max(0, n_frames - 1)
            gui_frame.value = 0
            gui_pc_start.max = max(0, n_frames - 1)
            gui_pc_end.max = max(0, n_frames - 1)
            pc_start = max(0, min(int(args.pc_frame_start), n_frames - 1))
            pc_end_arg = n_frames - 1 if int(args.pc_frame_end) < 0 else int(args.pc_frame_end)
            pc_end = max(0, min(pc_end_arg, n_frames - 1))
            gui_pc_start.value = pc_start
            gui_pc_end.value = pc_end
        except Exception:
            pass
        state["screenshot_dir"] = os.path.join(screenshot_root, name)
        os.makedirs(state["screenshot_dir"], exist_ok=True)
        _refresh_pointcloud()
        _refresh_cameras()

    # --- Keypoint helpers --------------------------------------------------------
    def _redraw_keypoint_markers():
        for h in state["kp_handles"]:
            h.remove()
        state["kp_handles"] = []
        for i, kp in enumerate(state["keypoints"]):
            try:
                handle = server.scene.add_camera_frustum(
                    f"/kps/{i:03d}",
                    fov=float(kp["fov"]),
                    aspect=16.0 / 9.0,
                    scale=float(gui_cam_size.value) * 1.5,
                    color=(255, 180, 0),
                    position=kp["position"],
                    wxyz=kp["wxyz"],
                )
                state["kp_handles"].append(handle)
            except Exception:
                pass
        gui_kp_count.value = str(len(state["keypoints"]))

    def _interp_trajectory(n_frames, loop):
        kps = list(state["keypoints"])
        if loop and len(kps) >= 2:
            kps = kps + [kps[0]]
        if len(kps) < 2:
            return []
        seg_count = len(kps) - 1
        out = []
        for i in range(n_frames):
            u = i / max(1, n_frames - 1)
            s = u * seg_count
            seg = min(int(np.floor(s)), seg_count - 1)
            local = float(s - seg)
            a, b = kps[seg], kps[seg + 1]
            pos = (1.0 - local) * np.asarray(a["position"]) + local * np.asarray(b["position"])
            wxyz = _slerp_wxyz(a["wxyz"], b["wxyz"], local)
            fov = (1.0 - local) * float(a["fov"]) + local * float(b["fov"])
            out.append((pos, wxyz, fov))
        return out

    def _set_camera(client, pos, wxyz, fov):
        with client.atomic():
            client.camera.position = pos
            client.camera.wxyz = wxyz
            client.camera.fov = float(fov)

    # --- Long-running tasks (run in threads to keep GUI responsive) -------------
    def _run_play_trajectory(client):
        fps = float(gui_traj_fps.value)
        dur = float(gui_traj_dur.value)
        loop = bool(gui_traj_loop.value)
        n = max(2, int(round(fps * dur)))
        traj = _interp_trajectory(n, loop)
        if not traj:
            gui_status.value = "need >=2 keypoints to play trajectory"
            return
        gui_status.value = f"playing trajectory ({n} frames @ {fps:.0f} fps)"
        period = 1.0 / max(1.0, fps)
        for pos, wxyz, fov in traj:
            t0 = time.time()
            try:
                _set_camera(client, pos, wxyz, fov)
            except Exception as e:
                gui_status.value = f"playback aborted: {e}"
                return
            dt = time.time() - t0
            if period - dt > 0:
                time.sleep(period - dt)
        gui_status.value = "trajectory playback done"

    def _run_render_trajectory(client):
        fps = float(gui_traj_fps.value)
        dur = float(gui_traj_dur.value)
        loop = bool(gui_traj_loop.value)
        h = int(gui_shot_h.value)
        w = int(gui_shot_w.value)
        n = max(2, int(round(fps * dur)))
        traj = _interp_trajectory(n, loop)
        if not traj:
            gui_status.value = "need >=2 keypoints to render trajectory"
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(state["screenshot_dir"], f"trajectory_{ts}.mp4")
        gui_status.value = f"rendering trajectory ({n} frames) ..."

        def _render_frame(k):
            pos, wxyz, fov = traj[k]
            _set_camera(client, pos, wxyz, fov)
            time.sleep(1.0 / 60.0)  # let scene settle
            return client.camera.get_render(height=h, width=w)

        def _progress(k):
            if (k + 1) % 20 == 0:
                gui_status.value = f"rendering trajectory: {k + 1}/{n}"

        try:
            _write_video_streaming(out_path, n, fps, _render_frame, _progress)
        except _FrameRenderError as e:
            gui_status.value = f"render aborted at frame {e.frame_idx}: {e.cause}"
            return
        except Exception as e:
            gui_status.value = f"video write failed: {e}"
            return
        gui_status.value = f"saved {out_path}"
        print(f"trajectory video -> {out_path}")

    def _run_render_singles(client):
        data = state["data"]
        if data is None:
            gui_status.value = "no sequence loaded"
            return
        h = int(gui_shot_h.value)
        w = int(gui_shot_w.value)
        fps = float(gui_play_fps.value)
        n = len(data["cam_c2ws"])
        prev_single = bool(gui_single.value)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(state["screenshot_dir"], f"singles_{ts}.mp4")
        gui_status.value = f"rendering single-frame video ({n} frames) ..."

        def _render_frame(i):
            gui_frame.value = i  # triggers _refresh_pointcloud via on_update
            time.sleep(0.03)  # give the scene a moment to update
            return client.camera.get_render(height=h, width=w)

        def _progress(i):
            if (i + 1) % 25 == 0:
                gui_status.value = f"rendering single-frame video: {i + 1}/{n}"

        try:
            if not prev_single:
                gui_single.value = True
            _write_video_streaming(out_path, n, fps, _render_frame, _progress)
        except _FrameRenderError as e:
            gui_status.value = f"render aborted at frame {e.frame_idx}: {e.cause}"
            return
        except Exception as e:
            gui_status.value = f"video write failed: {e}"
            return
        finally:
            try:
                gui_single.value = prev_single
            except Exception:
                pass
        gui_status.value = f"saved {out_path}"
        print(f"single-frame video -> {out_path}")

    def _run_play_loop():
        period = lambda: 1.0 / max(1.0, float(gui_play_fps.value))  # noqa: E731
        while state["playing"]:
            data = state["data"]
            if data is None:
                break
            n = len(data["cam_c2ws"])
            i = (int(gui_frame.value) + 1) % max(1, n)
            try:
                gui_frame.value = i
            except Exception:
                break
            time.sleep(period())
        state["playing"] = False

    # --- Event wiring -----------------------------------------------------------
    @gui_load_btn.on_click
    def _(_):
        _load_selected()

    if gui_clear_btn is not None:

        @gui_clear_btn.on_click
        def _(_):
            _clear_scene()
            gui_status.value = "scene cleared"

    @gui_n_points.on_update
    def _(_):
        _refresh_pointcloud()

    @gui_point_size.on_update
    def _(_):
        if state["pc_handle"] is not None:
            state["pc_handle"].point_size = float(gui_point_size.value)

    @gui_conf.on_update
    def _(_):
        _refresh_pointcloud()

    @gui_show_cams.on_update
    def _(_):
        _refresh_cameras()

    @gui_cam_size.on_update
    def _(_):
        _refresh_cameras()

    @gui_cam_stride.on_update
    def _(_):
        _refresh_cameras()

    @gui_single.on_update
    def _(_):
        _refresh_pointcloud()

    @gui_frame.on_update
    def _(_):
        if state["data"] is None:
            return
        if gui_single.value:
            _refresh_pointcloud()

    @gui_pc_start.on_update
    def _(_):
        if state["data"] is None:
            return
        if not gui_single.value:
            _refresh_pointcloud()

    @gui_pc_end.on_update
    def _(_):
        if state["data"] is None:
            return
        if not gui_single.value:
            _refresh_pointcloud()

    @gui_play_btn.on_click
    def _(_):
        if state["playing"]:
            state["playing"] = False
            gui_status.value = "playback paused"
            return
        if state["data"] is None:
            gui_status.value = "load a sequence first"
            return
        state["playing"] = True
        gui_status.value = "playing frames"
        threading.Thread(target=_run_play_loop, daemon=True).start()

    @gui_goto_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client or _first_client()
        if client is None or state["data"] is None:
            gui_status.value = "need connected client and loaded sequence"
            return
        _goto_frame(client, int(gui_frame.value))

    @gui_render_singles_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "no connected client to render from"
            return
        threading.Thread(target=_run_render_singles, args=(client,), daemon=True).start()

    @gui_kp_add_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "no connected client to snapshot"
            return
        kp = {
            "position": np.asarray(client.camera.position, dtype=np.float64).copy(),
            "wxyz": np.asarray(client.camera.wxyz, dtype=np.float64).copy(),
            "fov": float(client.camera.fov),
        }
        state["keypoints"].append(kp)
        _redraw_keypoint_markers()
        gui_status.value = f"keypoint #{len(state['keypoints'])} added"

    @gui_kp_clear_btn.on_click
    def _(_):
        state["keypoints"] = []
        _redraw_keypoint_markers()
        gui_status.value = "keypoints cleared"

    @gui_traj_play_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "no connected client"
            return
        threading.Thread(target=_run_play_trajectory, args=(client,), daemon=True).start()

    @gui_traj_render_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "no connected client"
            return
        threading.Thread(target=_run_render_trajectory, args=(client,), daemon=True).start()

    def _camera_world_axes(client):
        """Return camera-local movement axes in world coordinates."""
        R = vt.SO3(np.asarray(client.camera.wxyz, dtype=np.float64)).as_matrix()
        right = R @ np.array([1.0, 0.0, 0.0])
        # Viser CameraHandle follows OpenCV camera axes: +Z looks forward and
        # -Y is camera up.
        up = R @ np.array([0.0, -1.0, 0.0])
        forward = R @ np.array([0.0, 0.0, 1.0])
        return right, up, forward

    @gui_align_up_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "no connected client"
            return
        _, up, _ = _camera_world_axes(client)
        norm = float(np.linalg.norm(up))
        if not np.isfinite(norm) or norm < 1e-6:
            gui_status.value = "invalid camera orientation"
            return
        up = up / norm
        server.scene.set_up_direction(tuple(float(x) for x in up))
        gui_status.value = f"world up -> ({up[0]:+.2f}, {up[1]:+.2f}, {up[2]:+.2f})"

    def _move_camera(client, dx_right, dy_up, dz_forward):
        right, up, forward = _camera_world_axes(client)
        step = float(gui_move_speed.value)
        if not np.isfinite(step) or step <= 0.0:
            gui_status.value = "invalid move step"
            return
        delta = step * (dx_right * right + dy_up * up + dz_forward * forward)
        new_pos = np.asarray(client.camera.position, dtype=np.float64) + delta
        # Viser's position setter translates look_at by the same offset, so this
        # preserves the view direction while moving through the scene.
        client.camera.position = new_pos
        dist = float(np.linalg.norm(delta))
        gui_status.value = f"moved by {dist:.3f} scene units"

    def _make_move_cb(dx, dy, dz):
        def _cb(event: viser.GuiEvent):
            client = event.client or _first_client()
            if client is None:
                gui_status.value = "no connected client"
                return
            _move_camera(client, dx, dy, dz)

        return _cb

    gui_move_fwd_btn.on_click(_make_move_cb(0, 0, 1))
    gui_move_back_btn.on_click(_make_move_cb(0, 0, -1))
    gui_move_left_btn.on_click(_make_move_cb(-1, 0, 0))
    gui_move_right_btn.on_click(_make_move_cb(1, 0, 0))
    gui_move_up_btn.on_click(_make_move_cb(0, 1, 0))
    gui_move_down_btn.on_click(_make_move_cb(0, -1, 0))

    @gui_view_apply_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "no connected client"
            return
        # Compose tilt (camera-local X) then roll (camera-local Z) onto current orientation.
        tilt = float(np.deg2rad(gui_view_tilt.value))
        roll = float(np.deg2rad(gui_view_roll.value))
        fov_rad = float(np.deg2rad(gui_view_fov.value))
        cur = vt.SO3(np.asarray(client.camera.wxyz, dtype=np.float64))
        new_R = cur @ vt.SO3.from_x_radians(tilt) @ vt.SO3.from_z_radians(roll)
        with client.atomic():
            client.camera.wxyz = new_R.parameters().astype(np.float64)
            client.camera.fov = fov_rad
        gui_view_tilt.value = 0.0
        gui_view_roll.value = 0.0
        # Sync FOV slider to the new camera FOV so the user sees what's active.
        gui_view_fov.value = float(np.rad2deg(fov_rad))
        gui_status.value = "view tweaks applied"

    @gui_shot_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "screenshot needs a connected client"
            return
        h = int(gui_shot_h.value)
        w = int(gui_shot_w.value)
        try:
            img = client.camera.get_render(height=h, width=w)
        except Exception as e:
            gui_status.value = f"screenshot failed: {e}"
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(state["screenshot_dir"], f"viewer_{ts}.png")
        iio.imwrite(path, img)
        gui_status.value = f"saved {path}"
        print(f"screenshot -> {path}")

    @gui_view_save_btn.on_click
    def _(event: viser.GuiEvent):
        # Persist the current viser camera to JSON for external replay tools.
        # against any aligned method point cloud.
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "no connected client to save view"
            return
        import json as _json

        view = {
            "position": np.asarray(client.camera.position, dtype=np.float64).tolist(),
            "wxyz": np.asarray(client.camera.wxyz, dtype=np.float64).tolist(),
            "fov": float(client.camera.fov),
            "height": int(gui_shot_h.value),
            "width": int(gui_shot_w.value),
            "source_data_dir": os.path.abspath(runs.get(state["name"], "") if state["name"] else root),
            "name": str(gui_view_name.value).strip() or "view",
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(state["screenshot_dir"], f"view_{view['name']}_{ts}.json")
        with open(path, "w") as f:
            _json.dump(view, f, indent=2)
        gui_view_load_path.value = path
        gui_status.value = f"saved {path}"
        print(f"view -> {path}")

    @gui_view_load_btn.on_click
    def _(event: viser.GuiEvent):
        # Restore a previously-saved viser camera so the user can iterate on a viewpoint.
        client = event.client or _first_client()
        if client is None:
            gui_status.value = "no connected client to load view"
            return
        path = str(gui_view_load_path.value).strip()
        if not path or not os.path.exists(path):
            gui_status.value = f"view path not found: {path}"
            return
        import json as _json

        with open(path) as f:
            view = _json.load(f)
        with client.atomic():
            client.camera.position = np.asarray(view["position"], dtype=np.float64)
            client.camera.wxyz = np.asarray(view["wxyz"], dtype=np.float64)
            client.camera.fov = float(view["fov"])
        gui_shot_h.value = int(view.get("height", gui_shot_h.value))
        gui_shot_w.value = int(view.get("width", gui_shot_w.value))
        gui_view_fov.value = float(np.rad2deg(view["fov"]))
        gui_status.value = f"loaded {os.path.basename(path)}"

    # Keep GUI FOV slider mirroring the actual camera when a new client connects.
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        try:
            gui_view_fov.value = float(np.rad2deg(client.camera.fov))
        except Exception:
            pass

    # --- Initial load -----------------------------------------------------------
    if not is_batch or args.auto_load_first:
        _load_selected()

    print(f"Viser ready at http://0.0.0.0:{args.port} — Ctrl-C to stop.")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
