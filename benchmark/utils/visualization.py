"""
Visualization utilities: point cloud GLB export, camera frustum visualization, projection-based coloring, scene visualization.
"""
import os
import numpy as np

from benchmark.evaluation.metrics import unproject_to_pointcloud


# ============================================================
# Camera frustum utilities
# ============================================================

def _hsv_to_rgb(h, s, v):
    """HSV -> RGB (float [0,1])."""
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def _camera_color(i, n):
    """Generate a unique color for the i-th camera (uint8 RGB)."""
    h = (i + 0.5) / max(n, 1)
    r, g, b = _hsv_to_rgb(h, 0.85, 0.95)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def camera_frustum_lines(K, c2w, W, H, scale=0.1):
    """Generate frustum line segments for a single camera (world coordinates).

    Args:
        K: (3, 3) intrinsics
        c2w: (3, 4) cam2world
        W, H: image width and height
        scale: frustum size scale

    Returns:
        (8, 2, 3) float64 — 8 line segments, each with 2 endpoints
    """
    corners = np.array([
        [0, 0, 1.0],
        [W - 1, 0, 1.0],
        [W - 1, H - 1, 1.0],
        [0, H - 1, 1.0],
    ], dtype=np.float64)

    K_inv = np.linalg.inv(K.astype(np.float64))
    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)

    # Camera center (world coordinates)
    Cw = t.copy()

    # Pixel corners → camera coords on z=1 plane → scale → world coords
    rays = (K_inv @ corners.T).T  # (4, 3)
    z = rays[:, 2:3].copy()
    z[z == 0] = 1.0
    plane_cam = (rays / z) * scale  # (4, 3)
    plane_w = (R @ plane_cam.T).T + t  # (4, 3)

    segs = []
    # Camera center to the 4 corner points
    for k in range(4):
        segs.append(np.stack([Cw, plane_w[k]], axis=0))
    # 4 rectangle edges
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        segs.append(np.stack([plane_w[a], plane_w[b]], axis=0))

    return np.stack(segs, axis=0)  # (8, 2, 3)


def build_frustum_geometries(extrinsics, intrinsics, image_size, scale=0.1):
    """Generate a list of trimesh line-segment geometries for all frustums of a set of cameras.

    Args:
        extrinsics: (N, 3, 4) cam2world
        intrinsics: (N, 3, 3) intrinsics
        image_size: (W, H) or (N, 2) per-frame
        scale: frustum size

    Returns:
        list[trimesh.path.Path3D] — one line-segment geometry per camera
    """
    import trimesh

    N = len(extrinsics)
    geoms = []

    for i in range(N):
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2 and not hasattr(image_size[0], '__len__'):
            W, H = image_size
        else:
            W, H = image_size[i]

        segs = camera_frustum_lines(intrinsics[i], extrinsics[i], W, H, scale)
        path = trimesh.load_path(segs)
        color = _camera_color(i, N)
        if hasattr(path, "colors"):
            path.colors = np.tile(color, (len(path.entities), 1))
        geoms.append(path)

    return geoms


def _segment_to_cylinder(p0, p1, radius, sections=6):
    """Convert a line segment between two points into a thin cylinder trimesh.Trimesh.

    Used to export frustum line segments as a triangle mesh that MeshLab can render reliably
    (avoiding the GLB LINES primitive, which MeshLab supports poorly).
    """
    import trimesh

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return None
    direction /= length

    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)

    z = np.array([0.0, 0.0, 1.0])
    cos_a = float(np.dot(z, direction))
    if cos_a > 0.99999:
        R = np.eye(4)
    elif cos_a < -0.99999:
        R = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        axis = np.cross(z, direction)
        axis /= np.linalg.norm(axis)
        angle = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        R = trimesh.transformations.rotation_matrix(angle, axis)

    T = np.eye(4)
    T[:3, 3] = (p0 + p1) * 0.5

    cyl.apply_transform(T @ R)
    return cyl


def build_frustum_mesh_geometries(extrinsics, intrinsics, image_size, scale=0.1, line_radius=None):
    """Convert each camera's frustum line segments into thin-cylinder triangle meshes (one merged Mesh per camera).

    Compared to build_frustum_geometries() (which returns Path3D/LINES), triangle meshes render
    more reliably in tools like MeshLab and Blender and do not trigger crashes related to the LINES primitive.

    Args:
        extrinsics: (N, 3, 4) cam2world
        intrinsics: (N, 3, 3) intrinsics
        image_size: (W, H) or (N, 2)
        scale: frustum size
        line_radius: cylinder radius (None = automatically use scale*0.015)

    Returns:
        list[trimesh.Trimesh] — one colored Mesh per camera
    """
    import trimesh

    N = len(extrinsics)
    if line_radius is None:
        line_radius = max(scale * 0.015, 0.001)

    geoms = []
    for i in range(N):
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2 and not hasattr(image_size[0], '__len__'):
            W, H = image_size
        else:
            W, H = image_size[i]

        segs = camera_frustum_lines(intrinsics[i], extrinsics[i], W, H, scale)
        color = _camera_color(i, N)
        rgba = np.array([color[0], color[1], color[2], 255], dtype=np.uint8)

        cyls = []
        for seg in segs:
            cyl = _segment_to_cylinder(seg[0], seg[1], line_radius, sections=6)
            if cyl is None:
                continue
            cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
            cyls.append(cyl)

        if not cyls:
            continue

        merged = trimesh.util.concatenate(cyls)
        geoms.append(merged)

    return geoms


# ============================================================
# GLB export
# ============================================================

def save_pointcloud_glb(points, colors, output_path,
                        extrinsics=None, intrinsics=None, image_size=None,
                        frustum_scale=0.1, max_pts=500_000,
                        frustums_as_mesh=True):
    """Save point cloud + optional camera frustums to a GLB file.

    Args:
        points: (M, 3) world-coordinate points
        colors: (M, 3) RGB colors [0, 1]
        output_path: output .glb path
        extrinsics: (N, 3, 4) cam2world; when provided, draw camera frustums
        intrinsics: (N, 3, 3) intrinsics
        image_size: (W, H) image size
        frustum_scale: frustum size scale
        max_pts: maximum number of points (randomly downsampled when exceeded)
        frustums_as_mesh: True = export frustums as triangle meshes (stable in MeshLab),
                          False = export as Path3D/LINES (clearer in three.js/Web,
                          but may crash MeshLab)
    """
    try:
        import trimesh
    except ImportError:
        print(f"  [WARN] trimesh not installed, skipping GLB export: {output_path}")
        return

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)

    # Filter NaN/Inf (one of the common causes of MeshLab crashes)
    if len(points) > 0:
        finite = np.isfinite(points).all(axis=1)
        if not finite.all():
            points = points[finite]
            colors = colors[finite]

    if len(points) == 0:
        print(f"  [WARN] no finite points to export: {output_path}")
        return

    if len(points) > max_pts:
        idx = np.random.choice(len(points), max_pts, replace=False)
        points = points[idx]
        colors = colors[idx]

    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    rgba = np.concatenate([colors_uint8, np.full((len(colors_uint8), 1), 255, dtype=np.uint8)], axis=1)

    cloud = trimesh.PointCloud(vertices=points, colors=rgba)

    # If camera parameters are provided, add frustums
    if extrinsics is not None and intrinsics is not None and image_size is not None:
        scene = trimesh.Scene([cloud])
        if frustums_as_mesh:
            frustums = build_frustum_mesh_geometries(
                extrinsics, intrinsics, image_size, frustum_scale,
            )
        else:
            frustums = build_frustum_geometries(
                extrinsics, intrinsics, image_size, frustum_scale,
            )
        for geom in frustums:
            scene.add_geometry(geom)
        scene.export(output_path)
    else:
        cloud.export(output_path)


def color_pointcloud_by_projection(points, extrinsics, intrinsics, images_raw):
    """Project the point cloud onto each view image to pick colors.

    For each 3D point, project it to the nearest camera view and sample an RGB color from the image.
    Points that cannot be projected use gray (0.6, 0.6, 0.6).

    Args:
        points: (M, 3) world-coordinate points
        extrinsics: (N, 3, 4) cam2world
        intrinsics: (N, 3, 3) intrinsics
        images_raw: Tensor (N, 3, H, W) [0, 1]

    Returns:
        colors: (M, 3) float32 [0, 1]
    """
    M = len(points)
    N = len(extrinsics)
    if M == 0:
        return np.zeros((0, 3), dtype=np.float32)

    _, _, H, W = images_raw.shape
    colors = np.full((M, 3), 0.6, dtype=np.float32)
    min_depth = np.full(M, np.inf, dtype=np.float32)

    for i in range(N):
        R_c2w = extrinsics[i, :3, :3]
        t_c2w = extrinsics[i, :3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w

        pts_cam = (R_w2c @ points.T).T + t_w2c
        z = pts_cam[:, 2]

        valid = z > 0.01
        if not valid.any():
            continue

        K = intrinsics[i]
        u = (K[0, 0] * pts_cam[:, 0] / z + K[0, 2]).astype(np.int32)
        v = (K[1, 1] * pts_cam[:, 1] / z + K[1, 2]).astype(np.int32)

        in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        closer = z < min_depth
        use = in_bounds & closer

        if not use.any():
            continue

        img = images_raw[i].permute(1, 2, 0).numpy()
        idx_use = np.where(use)[0]
        colors[idx_use] = img[v[idx_use], u[idx_use]]
        min_depth[idx_use] = z[idx_use]

    return colors


def _collect_colors(images_raw, mask_per_frame):
    """Collect pixel colors from each frame's image according to a mask.

    Args:
        images_raw: Tensor (N, 3, H, W) [0, 1]
        mask_per_frame: (N, H, W) bool

    Returns:
        (M, 3) float32
    """
    colors = []
    N = len(mask_per_frame)
    for i in range(N):
        mask = mask_per_frame[i]
        if not mask.any():
            continue
        img = images_raw[i].permute(1, 2, 0).numpy()
        colors.append(img[mask])
    return np.concatenate(colors, axis=0) if colors else np.zeros((0, 3), dtype=np.float32)


def save_scene_inputs(scene, inputs_dir):
    """Save the scene's input images to inputs_dir/<scene_id>/.

    Each frame is exported as a PNG, with the filename containing the original frame_index for easy lookup in the source dataset.

    Args:
        scene: dict from BenchmarkDataset
        inputs_dir: parent directory; the function will create a <scene_id>/ subdirectory inside it
    """
    from PIL import Image

    scene_id = scene["scene_id"]
    images_raw = scene["images_raw"]  # (N, 3, H, W) float in [0, 1]
    frame_indices = scene.get("frame_indices") or list(range(len(images_raw)))

    out_dir = os.path.join(inputs_dir, scene_id)
    os.makedirs(out_dir, exist_ok=True)

    arr = (images_raw.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    arr = np.transpose(arr, (0, 2, 3, 1))  # (N, H, W, 3)
    for i, fidx in enumerate(frame_indices):
        Image.fromarray(arr[i]).save(
            os.path.join(out_dir, f"frame_{int(fidx):06d}.png")
        )


def visualize_gt(scene, output_dir):
    """Save the GT point cloud + camera frustums as a GLB file.

    All models share the same GT visualization function.

    Args:
        scene: dict from BenchmarkDataset
        output_dir: output directory
    """
    scene_id = scene["scene_id"]
    os.makedirs(output_dir, exist_ok=True)

    gt_depth = scene["depth"]
    gt_poses = scene["extrinsic"]
    gt_intrinsic = scene["intrinsic"]
    valid_mask = scene["valid_mask"]
    images_raw = scene["images_raw"]
    N, _, H, W = images_raw.shape

    gt_points = unproject_to_pointcloud(gt_depth, gt_poses, gt_intrinsic, valid_mask)
    if len(gt_points) > 0:
        gt_colors = _collect_colors(images_raw, valid_mask)
        gt_glb_path = os.path.join(output_dir, f"{scene_id}_gt.glb")
        save_pointcloud_glb(gt_points, gt_colors, gt_glb_path,
                            extrinsics=gt_poses, intrinsics=gt_intrinsic,
                            frustum_scale=0.1, image_size=(W, H))
        print(f"    GT  -> {gt_glb_path} ({len(gt_points)} pts, {N} cams)")


def visualize_scene(scene, predictions, adapter,
                    gt_dir=None, pred_dir=None, output_dir=None,
                    z_far=None, vis_conf_percent=50.0):
    """Save the GT and predicted point clouds for one scene as GLB files.

    GT visualization uses the unified visualize_gt();
    prediction visualization is delegated to adapter.visualize_prediction() (customizable per model).

    Args:
        scene: dict from BenchmarkDataset
        predictions: dict from adapter.predict()
        adapter: ModelAdapter instance
        gt_dir: GT GLB output directory
        pred_dir: Pred GLB output directory
        output_dir: legacy-interface compatibility; if gt_dir/pred_dir are unspecified, both use this directory
        z_far: maximum depth filter (None = no filtering)
        vis_conf_percent: filter out the lowest N% confidence points (0 = no filter, 50 = filter the lowest 50%)
    """
    if gt_dir is None:
        gt_dir = output_dir
    if pred_dir is None:
        pred_dir = output_dir
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # GT point cloud (shared by all models)
    visualize_gt(scene, gt_dir)

    # If the scene has a sky_mask, zero out the sky region in the predicted depth before visualization
    sky_mask = scene.get('sky_mask')  # (N, H, W) bool or None
    if sky_mask is not None and 'pred_depth' in predictions:
        pred_depth = predictions['pred_depth'].copy()
        pred_depth[sky_mask] = 0
        predictions = {**predictions, 'pred_depth': pred_depth}

    # Predicted point cloud (delegated to each model's adapter)
    adapter.visualize_prediction(
        scene, predictions, pred_dir,
        z_far=z_far, vis_conf_percent=vis_conf_percent,
    )
