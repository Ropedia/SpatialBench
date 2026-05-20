"""
SpatialBench unified scene reader.

The dataset uses a unified directory format (per-scene):

    <data_root>/<scene_path>/
        images/             RGB images (positional frames per lexicographically sorted filename)
        depths/             depth maps (same order as images/; encoding in meta.json depth_format)
        poses/              per-frame .npy cam2world (3, 4) or (4, 4); identity when missing
        intrinsics/
            intrinsic.npy   shared mode (3, 3)
            <stem>.npy      per_frame mode (3, 3)
        meta.json           scene metadata (depth_format / intrinsic_mode / ...)
        depth_masks/        optional; uint8, 0=invalid
        aliasing_masks/     optional (hiroom); uint8
        sky_masks/          optional; uint8, !=0 indicates sky
        conf_masks/         optional (ropedia); uint16, /65535 -> [0,1]

frame_indices in scene_index.json are 0-based positional indices into the sorted images/.
"""
import csv
import glob
import json
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')  # Waymo EXR depth maps
import re

import cv2
import numpy as np
import PIL.Image

from benchmark.utils.cropping import (
    rescale_image_depthmap,
    crop_image_depthmap,
    camera_matrix_of_crop,
    bbox_from_intrinsics_in_out,
)
from benchmark.utils.misc import threshold_depth_map


_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')
_DEPTH_EXTS = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.exr', '.EXR', '.npy')


def deterministic_resize(image, depthmap, intrinsics, resolution):
    """Deterministic resize: principal-point-centered crop + scale to target resolution."""
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)

    if min_margin_x > W / 5 and min_margin_y > H / 5:
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        image, depthmap, intrinsics = crop_image_depthmap(
            image, depthmap, intrinsics, (l, t, r, b)
        )

    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = rescale_image_depthmap(
        image, depthmap, intrinsics, target_resolution
    )

    intrinsics2 = camera_matrix_of_crop(
        intrinsics, image.size, resolution, offset_factor=0.5
    )
    crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = crop_image_depthmap(
        image, depthmap, intrinsics, crop_bbox
    )
    return image, depthmap, intrinsics2


def _list_sorted(dir_path, exts):
    """List files in dir_path matching exts, sorted lexicographically by filename."""
    if not os.path.isdir(dir_path):
        return []
    files = []
    for f in os.listdir(dir_path):
        if f.endswith(exts):
            files.append(os.path.join(dir_path, f))
    files.sort()
    return files


def _decode_depth(path, depth_format, meta):
    """Decode a depth map per depth_format; returns (depth, sky_mask_or_None)."""
    if depth_format in ('uint16_png_div1000', 'uint16_png_div1000_rectified'):
        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        return raw.astype(np.float32) / 1000.0, None

    if depth_format == 'uint16_png_div100':
        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        depth = raw.astype(np.float32) / 100.0
        sky_mask = raw >= 65500
        depth[sky_mask] = 0
        return depth, sky_mask

    if depth_format == 'uint16_png_div_655.35':
        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        return raw.astype(np.float32) / 655.35, None

    if depth_format == 'uint16_png_inverse_encoded':
        # OmniWorld: depth = decode(raw/65535) * metric_scale; sky if raw > 65500
        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        metric_scale = float(meta.get('metric_scale', 1.0))
        depthmap = raw.astype(np.float32) / 65535.0
        near_mask = depthmap < 0.0015
        sky_mask = depthmap > (65500.0 / 65535.0)
        near, far = 1.0, 1000.0
        depthmap = depthmap / (far - depthmap * (far - near)) / 0.004
        invalid = near_mask | sky_mask
        depthmap[invalid] = 0
        depthmap[~invalid] *= metric_scale
        return depthmap, sky_mask

    if depth_format == 'npy_float32_meters':
        return np.load(path).astype(np.float32), None

    if depth_format == 'exr_float32':
        d = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32), None

    if depth_format.startswith('float32_binary'):
        m = re.search(r'_(\d+)x(\d+)', depth_format)
        if not m:
            raise ValueError(f"Cannot parse size from depth_format: {depth_format}")
        H, W = int(m.group(1)), int(m.group(2))
        d = np.fromfile(path, dtype=np.float32).reshape(H, W)
        return d, None

    if depth_format == 'none':
        return None, None

    raise ValueError(f"Unknown depth_format: {depth_format}")


def _load_pose(path):
    """Load a .npy pose file; uniformly returns (3, 4) cam2world float32."""
    p = np.load(path).astype(np.float32)
    if p.shape == (4, 4):
        p = p[:3, :]
    return p


def _resize_mask(mask, target_wh):
    """Resize a bool/uint8 mask to target resolution using nearest-neighbor interpolation."""
    W, H = target_wh
    m = (mask.astype(np.uint8) if mask.dtype != np.uint8 else mask) > 0
    if m.shape == (H, W):
        return m
    return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0


class BaseReader:
    """Base class for unified-format scene readers; auto-dispatches depth encodings per meta.json.

    Subclasses only need to override DEFAULT_RESOLUTION / DEFAULT_Z_FAR or the optional
    _postprocess hook.
    """

    DEFAULT_RESOLUTION = (518, 378)
    DEFAULT_Z_FAR = 10.0
    _resolution_override = None

    # For simulated data, mask filtering can be disabled (depth itself is accurate; masks
    # accidentally kill valid pixels)
    _USE_DEPTH_MASKS = True
    _USE_ALIASING_MASKS = True

    # Non-frame files to exclude from the intrinsics directory (per_frame mode)
    _NON_FRAME_INTRINSIC_FILES = (
        'intrinsic.npy', 'K_rect.npy', 'K_distorted.npy', 'distortion_coeffs.npy',
    )

    def _compute_resolution(self, peek_image_path=None):
        if self._resolution_override is None:
            return self.DEFAULT_RESOLUTION
        if isinstance(self._resolution_override, (list, tuple)):
            return tuple(self._resolution_override)
        if isinstance(self._resolution_override, dict) and peek_image_path:
            img = PIL.Image.open(peek_image_path)
            orig_W, orig_H = img.size
            img.close()
            align = self._resolution_override.get('align', 16)
            target_w = self._resolution_override['width']
            target_h = round(orig_H / orig_W * target_w / align) * align
            return (target_w, target_h)
        return self.DEFAULT_RESOLUTION

    def _postprocess(self, rgb, depth, K, pose, sky_mask, meta, idx):
        """Override in subclasses for extra per-frame processing (e.g. scannetpp undistortion)."""
        return rgb, depth, K, pose, sky_mask

    @staticmethod
    def _apply_pyrep_flip(rgb, depth, K, info):
        """Pre-baked compensation for PyRep's negative fx/fy intrinsics: flip image+depth by
        basename and adjust K's principal point.

        SpatialBench's RLBench data has already converted negative fx/fy into positive K +
        meta.pyrep_flip (per-frame {flip_lr, flip_ud}). Without flipping, the point cloud is
        mirrored.
        """
        H, W = depth.shape
        K = K.copy()
        if info.get('flip_lr'):
            rgb = rgb.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            depth = np.ascontiguousarray(np.flip(depth, axis=1))
            K[0, 2] = (W - 1) - K[0, 2]
        if info.get('flip_ud'):
            rgb = rgb.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            depth = np.ascontiguousarray(np.flip(depth, axis=0))
            K[1, 2] = (H - 1) - K[1, 2]
        return rgb, depth, K

    def read_scene(self, data_root, scene_path, frame_indices):
        scene_dir = os.path.join(data_root, scene_path)

        with open(os.path.join(scene_dir, 'meta.json'), 'r') as f:
            meta = json.load(f)

        rgb_paths = _list_sorted(os.path.join(scene_dir, 'images'), _IMAGE_EXTS)
        if not rgb_paths:
            raise FileNotFoundError(f"No images in {scene_dir}/images")
        depth_paths = _list_sorted(os.path.join(scene_dir, 'depths'), _DEPTH_EXTS)
        pose_paths = _list_sorted(os.path.join(scene_dir, 'poses'), ('.npy',))

        # Intrinsics
        intrinsic_dir = os.path.join(scene_dir, 'intrinsics')
        intrinsic_mode = meta.get('intrinsic_mode', 'shared')
        shared_K = None
        per_frame_K = None
        if intrinsic_mode == 'shared':
            shared_path = os.path.join(intrinsic_dir, 'intrinsic.npy')
            shared_K = np.load(shared_path).astype(np.float32)
        else:
            per_frame_K = [
                p for p in _list_sorted(intrinsic_dir, ('.npy',))
                if os.path.basename(p) not in self._NON_FRAME_INTRINSIC_FILES
            ]

        # Optional mask directories
        dm_paths = _list_sorted(os.path.join(scene_dir, 'depth_masks'), _IMAGE_EXTS) or None
        am_paths = _list_sorted(os.path.join(scene_dir, 'aliasing_masks'), _IMAGE_EXTS) or None
        sky_paths = _list_sorted(os.path.join(scene_dir, 'sky_masks'), _IMAGE_EXTS) or None
        conf_paths = _list_sorted(os.path.join(scene_dir, 'conf_masks'), _IMAGE_EXTS) or None

        depth_format = meta.get('depth_format', 'none')
        has_depth = depth_format != 'none' and bool(depth_paths)
        has_pose = bool(pose_paths) and meta.get('pose', '') != 'none'

        resolution = self._compute_resolution(rgb_paths[frame_indices[0]])
        z_far = self.DEFAULT_Z_FAR
        conf_threshold = getattr(self, 'conf_threshold', 0.3)

        # PyRep flip metadata: basename -> {flip_lr, flip_ud} (RLBench-specific)
        pyrep_flip_list = meta.get('pyrep_flip')
        flip_by_basename = (
            {it['basename']: it for it in pyrep_flip_list} if pyrep_flip_list else None
        )

        images, depths, extrinsics, intrinsics = [], [], [], []
        sky_masks = []
        any_sky = False

        for idx in frame_indices:
            rgb = PIL.Image.open(rgb_paths[idx]).convert("RGB")

            # ---- Depth ----
            sky_mask_decoded = None
            if has_depth:
                depth, sky_mask_decoded = _decode_depth(depth_paths[idx], depth_format, meta)
                depth = depth.astype(np.float32)
                depth[~np.isfinite(depth)] = 0
            else:
                W, H = rgb.size
                depth = np.zeros((H, W), dtype=np.float32)

            # depth_masks / aliasing_masks: 0 -> invalid
            if dm_paths and self._USE_DEPTH_MASKS:
                mask = cv2.imread(dm_paths[idx], cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.shape == depth.shape:
                    depth[mask == 0] = 0
            if am_paths and self._USE_ALIASING_MASKS:
                mask = cv2.imread(am_paths[idx], cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.shape == depth.shape:
                    depth[mask == 0] = 0

            # conf_masks: threshold filtering (ropedia: uint16/65535)
            if conf_paths and idx < len(conf_paths):
                conf_raw = cv2.imread(conf_paths[idx], cv2.IMREAD_UNCHANGED)
                if conf_raw is not None and conf_raw.shape == depth.shape:
                    denom = 65535.0 if conf_raw.dtype == np.uint16 else 255.0
                    conf = conf_raw.astype(np.float32) / denom
                    depth[conf < conf_threshold] = 0

            # z_far clipping
            depth[depth > z_far] = 0

            # sky_masks: file takes precedence over decode
            sky_mask = sky_mask_decoded
            if sky_paths and idx < len(sky_paths):
                sm = cv2.imread(sky_paths[idx], cv2.IMREAD_GRAYSCALE)
                if sm is not None:
                    sky_mask = sm > 0
                    if sky_mask.shape == depth.shape:
                        depth[sky_mask] = 0

            # ---- Pose ----
            if has_pose and idx < len(pose_paths):
                pose = _load_pose(pose_paths[idx])
            else:
                pose = np.eye(4, dtype=np.float32)[:3, :]

            # ---- Intrinsic ----
            if shared_K is not None:
                K = shared_K.copy()
            else:
                K = np.load(per_frame_K[idx]).astype(np.float32)

            # ---- PyRep flip (RLBench): flip image/depth/K by basename ----
            if flip_by_basename is not None:
                bn = os.path.splitext(os.path.basename(rgb_paths[idx]))[0]
                info = flip_by_basename.get(bn)
                if info is not None and (info.get('flip_lr') or info.get('flip_ud')):
                    rgb, depth, K = self._apply_pyrep_flip(rgb, depth, K, info)

            # ---- Subclass hook ----
            rgb, depth, K, pose, sky_mask = self._postprocess(
                rgb, depth, K, pose, sky_mask, meta, idx
            )

            # ---- Resize ----
            rgb, depth, K = deterministic_resize(rgb, depth, K, resolution)
            if sky_mask is not None:
                sky_mask = _resize_mask(sky_mask, rgb.size)
                any_sky = True
            sky_masks.append(sky_mask)

            images.append(rgb)
            depths.append(depth)
            extrinsics.append(pose)
            intrinsics.append(K)

        out = {
            'images': images,
            'depths': depths,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
        if any_sky:
            out['sky_masks'] = sky_masks
        return out


# ============================================================
# Per-dataset subclasses: only override default resolution/z_far; a few need a hook
# ============================================================


class DroidReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 294)
    DEFAULT_Z_FAR = 1.0


class TumReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 392)
    DEFAULT_Z_FAR = 5.0


class NrgbdReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 392)
    DEFAULT_Z_FAR = 10.0


class SevenScenesReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 392)
    DEFAULT_Z_FAR = 10.0


class AdtReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 518)
    DEFAULT_Z_FAR = 10.0


class RopediaReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 518)
    DEFAULT_Z_FAR = 8.0

    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold

    def _postprocess(self, rgb, depth, K, pose, sky_mask, meta, idx):
        # The high-proportion far-end depth is very likely noise; reuse the old reader's 80th
        # percentile filter
        depth = threshold_depth_map(depth, max_percentile=80, min_percentile=-1)
        return rgb, depth, K, pose, sky_mask


class RLBenchReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 294)
    DEFAULT_Z_FAR = 3.0


class RoboTwinReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 294)
    DEFAULT_Z_FAR = 3.0


class RoboLabReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 294)
    DEFAULT_Z_FAR = 3.0


class DtuReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 378)
    DEFAULT_Z_FAR = 3.0


class Eth3dReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 350)
    DEFAULT_Z_FAR = 30.0


class TanksAndTemplesReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 378)
    DEFAULT_Z_FAR = 30.0


class OmniworldReader(BaseReader):
    """OmniWorld-Game: inverse-encoded depth (metric_scale is already multiplied into depth in
    _decode_depth).

    The c2w translations in pose files are already metric (multiplied by metric_scale during
    preprocessing); do not multiply again, otherwise trajectories are scaled up ~400x and the
    whole point cloud appears scattered over kilometers.
    depth_masks act as a flying-points filter for simulated rendering (at depth discontinuities at
    boundaries); keep them - sim does not mean no flying points.
    """
    DEFAULT_RESOLUTION = (518, 294)
    DEFAULT_Z_FAR = 50.0
    _resolution_override = {"width": 518, "align": 14}


class LingbotReader(BaseReader):
    # Three source categories have different resolutions (RobbyReal 1280x720, RobbySimVal 1280x960,
    # RobbyVla 640x480); using 518 as width, align to a multiple of 14 by the original aspect
    # ratio (ViT patch_size=14).
    DEFAULT_RESOLUTION = (518, 392)
    DEFAULT_Z_FAR = 20.0
    _resolution_override = {"width": 518, "align": 14}


class HiroomReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 518)
    DEFAULT_Z_FAR = 10.0
    # hiroom's aliasing_masks only flag ~0.6% of anti-aliased edge pixels (uint8 255=aliased),
    # but BaseReader's `depth[mask==0]=0` semantics would instead keep that 0.6% and kill 99.4%
    # of pixels.
    # Simulated depth is already accurate, so simply disable this mask.
    _USE_ALIASING_MASKS = False


class ScannetppReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 378)
    DEFAULT_Z_FAR = 10.0

    def __init__(self):
        self._undistort_cache = {}

    def _get_undistort_maps(self, scene_dir, image_size):
        """Cache cv2.initUndistortRectifyMap results (per-scene)."""
        key = (scene_dir, image_size)
        if key in self._undistort_cache:
            return self._undistort_cache[key]
        K_d = np.load(os.path.join(scene_dir, 'intrinsics', 'K_distorted.npy')).astype(np.float32)
        dist = np.load(os.path.join(scene_dir, 'intrinsics', 'distortion_coeffs.npy')).astype(np.float32)
        K_rect = np.load(os.path.join(scene_dir, 'intrinsics', 'intrinsic.npy')).astype(np.float32)
        W, H = image_size
        map1, map2 = cv2.initUndistortRectifyMap(K_d, dist, None, K_rect, (W, H), cv2.CV_16SC2)
        self._undistort_cache[key] = (map1, map2)
        return map1, map2

    def _postprocess(self, rgb, depth, K, pose, sky_mask, meta, idx):
        # Images are stored as the original distorted, depth is already rectified; here we use
        # K_rect + distortion coefficients to undistort the image into K_rect space
        if meta.get('image_state') != 'distorted_original':
            return rgb, depth, K, pose, sky_mask
        scene_dir = meta.get('_scene_dir')
        if not scene_dir:
            return rgb, depth, K, pose, sky_mask
        rgb_np = np.array(rgb)
        H, W = rgb_np.shape[:2]
        map1, map2 = self._get_undistort_maps(scene_dir, (W, H))
        rgb_rect = cv2.remap(rgb_np, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return PIL.Image.fromarray(rgb_rect), depth, K, pose, sky_mask

    def read_scene(self, data_root, scene_path, frame_indices):
        # Inject scene_dir into meta so that _postprocess can access the undistortion matrices
        scene_dir = os.path.join(data_root, scene_path)
        meta_path = os.path.join(scene_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta['_scene_dir'] = scene_dir
        # Temporarily written back so the base class can read it
        # Switch to overriding base-class logic directly: copy the base implementation and inject
        # _scene_dir after loading meta. To avoid duplication, via a monkey approach we instead
        # call the base implementation and rely on the file meta; changed to reading _scene_dir
        # from the meta field already passed into _postprocess. Since base does not pass
        # _scene_dir, we take a custom path here.
        return self._read_scene_with_meta(data_root, scene_path, frame_indices, meta)

    def _read_scene_with_meta(self, data_root, scene_path, frame_indices, meta):
        # Directly reuse the base-class logic; only pass the already-loaded meta through to
        # _postprocess. Use a lightweight implementation that copies the loop body of
        # BaseReader.read_scene.
        scene_dir = os.path.join(data_root, scene_path)
        rgb_paths = _list_sorted(os.path.join(scene_dir, 'images'), _IMAGE_EXTS)
        depth_paths = _list_sorted(os.path.join(scene_dir, 'depths'), _DEPTH_EXTS)
        pose_paths = _list_sorted(os.path.join(scene_dir, 'poses'), ('.npy',))
        intrinsic_dir = os.path.join(scene_dir, 'intrinsics')
        shared_K = np.load(os.path.join(intrinsic_dir, 'intrinsic.npy')).astype(np.float32)

        depth_format = meta.get('depth_format', 'none')
        has_depth = depth_format != 'none' and bool(depth_paths)
        has_pose = bool(pose_paths) and meta.get('pose', '') != 'none'

        resolution = self._compute_resolution(rgb_paths[frame_indices[0]])
        z_far = self.DEFAULT_Z_FAR

        images, depths, extrinsics, intrinsics = [], [], [], []
        for idx in frame_indices:
            rgb = PIL.Image.open(rgb_paths[idx]).convert("RGB")
            if has_depth:
                depth, _ = _decode_depth(depth_paths[idx], depth_format, meta)
                depth = depth.astype(np.float32)
                depth[~np.isfinite(depth)] = 0
            else:
                W, H = rgb.size
                depth = np.zeros((H, W), dtype=np.float32)
            depth[depth > z_far] = 0

            if has_pose:
                pose = _load_pose(pose_paths[idx])
            else:
                pose = np.eye(4, dtype=np.float32)[:3, :]

            K = shared_K.copy()
            rgb, depth, K, pose, _ = self._postprocess(rgb, depth, K, pose, None, meta, idx)
            rgb, depth, K = deterministic_resize(rgb, depth, K, resolution)

            images.append(rgb)
            depths.append(depth)
            extrinsics.append(pose)
            intrinsics.append(K)

        return {
            'images': images,
            'depths': depths,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }


class VkittiReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 154)
    DEFAULT_Z_FAR = 80.0


class WaymoReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 350)
    DEFAULT_Z_FAR = 50.0


class KittiOdometryReader(BaseReader):
    DEFAULT_RESOLUTION = (518, 154)
    DEFAULT_Z_FAR = 80.0
