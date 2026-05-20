"""
HunyuanWorld-Mirror model adapter.
Use WorldMirror.from_pretrained() to load the model and forward() for inference.

HunyuanWorld-Mirror outputs:
  - depth: (B, S, H, W, 1) Z-depth in camera frame (relative, not metric)
  - depth_conf: (B, S, H, W) depth confidence
  - pts3d: (B, S, H, W, 3) world-coordinate points
  - pts3d_conf: (B, S, H, W) point-cloud confidence
  - camera_poses: (B, S, 4, 4) cam2world pose (OpenCV coordinate system)
  - camera_intrs: (B, S, 3, 3) intrinsics
  - normals: (B, S, H, W, 3) surface normals

Note: WorldMirror camera_poses is already in cam2world format.
     Input is [0, 1] range tensor (the model does not apply ImageNet normalization internally).
     Image dimensions must be aligned to 14 multiple (patch_size=14).
"""
import sys
import os
import numpy as np
import torch

_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
_WORLDMIRROR_ROOT = os.path.join(_MODELS_ROOT, 'worldmirror_root')
sys.path.insert(0, _WORLDMIRROR_ROOT)

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter


PATCH_SIZE = 14


def _inv_se3(mat):
    """SE(3) closed-form inverse transform, numerically stable, supports single (4,4) or batched (N,4,4).

    For SE(3) matrix [R|t; 0 1], its inverse is [R^T | -R^T @ t; 0 1], 
    No LU decomposition is needed, avoiding singular-matrix risk.
    """
    single = mat.ndim == 2
    if single:
        mat = mat[np.newaxis]
    R = mat[:, :3, :3]                      # (N, 3, 3)
    t = mat[:, :3, 3:]                      # (N, 3, 1)
    Rt = np.swapaxes(R, -1, -2)             # R^T, (N, 3, 3)
    inv = np.zeros_like(mat)
    inv[:, :3, :3] = Rt
    inv[:, :3, 3:] = -np.matmul(Rt, t)     # -R^T @ t
    inv[:, 3, 3] = 1.0
    return inv[0] if single else inv


@register_adapter("worldmirror")
class WorldMirrorAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.cond_flags = [0, 0, 0]  # condition flags [pose, depth, intrinsic]

    def name(self):
        return "WorldMirror"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from src.models.models.worldmirror import WorldMirror

        if checkpoint and os.path.isdir(checkpoint):
            self.model = WorldMirror.from_pretrained(checkpoint)
            print(f"[WorldMirrorAdapter] Model loaded from {checkpoint}")
        else:
            repo_id = checkpoint or "tencent/HunyuanWorld-Mirror"
            if weights_dir:
                from benchmark.utils.hf_weights import ensure_hf_snapshot
                snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
                self.model = WorldMirror.from_pretrained(snapshot_dir)
                print(f"[WorldMirrorAdapter] Model loaded from {repo_id} -> {snapshot_dir}")
            else:
                self.model = WorldMirror.from_pretrained(repo_id)
                print(f"[WorldMirrorAdapter] Model loaded from {repo_id}")

        self.model = self.model.to(device)
        # SpatialBench evaluates WorldMirror depth/pose/trajectory outputs and does not consume
        # Gaussian splat predictions. Disable this optional branch so gsplat is not required.
        if getattr(self.model, "enable_gs", False):
            self.model.enable_gs = False
        self.model.eval()
        print(f"[WorldMirrorAdapter] Model on {device}")

    def supports_gt_prior(self):
        return {'pose': True, 'depth': True, 'intrinsic': True, 'partial': False}

    def predict(self, scene, gt_config=None):
        """Run WorldMirror inference.

        WorldMirror input: views={"img": (1, N, 3, H, W)} in [0,1], cond_flags=[0,0,0]
        WorldMirror outputs: camera_poses (B, S, 4, 4) c2w, depth (B, S, H, W, 1),
                         pts3d (B, S, H, W, 3), camera_intrs (B, S, 3, 3)

        Args:
            scene: dict from BenchmarkDataset
            gt_config: optional dict with keys:
                use_pose, use_depth, use_intrinsic (bool)
                gt_frame_indices (list[int])
        """
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # All data readers already guarantee DEFAULT_RESOLUTION is divisible by 14, 
        # Assert here instead of silently resizing to avoid hiding resolution issues
        assert H % PATCH_SIZE == 0 and W % PATCH_SIZE == 0,\
            f"Input size ({H}, {W}) must be divisible by patch_size={PATCH_SIZE}. "\
            f"Use resolution_override in config to set a compatible resolution."

        # WorldMirror expects (1, N, 3, H, W) in [0, 1]
        images_input = images_raw.unsqueeze(0).to(self.device)

        # ---- Build GT prior input (all-or-nothing) ----
        # cond_flags order: [pose, depth, rays/intrinsic]
        cond_flags = list(self.cond_flags)
        views_dict = {"img": images_input}

        has_camera_gt = gt_config and (gt_config.get('use_pose') or gt_config.get('use_intrinsic'))
        has_depth_gt = gt_config and gt_config.get('use_depth')
        if has_camera_gt or has_depth_gt:

            if gt_config.get('use_pose'):
                cond_flags[0] = 1
                # camera_poses: (1, N, 4, 4) c2w, align to the first frame
                gt_c2w = scene['extrinsic']  # (N, 3, 4)
                c2w_44 = np.zeros((N, 4, 4), dtype=np.float32)
                c2w_44[:, :3, :] = gt_c2w
                c2w_44[:, 3, 3] = 1.0
                # align to the first frame: c2w_aligned[i] = inv(c2w[0]) @ c2w[i]
                c2w0_inv = _inv_se3(c2w_44[0])
                c2w_aligned = np.matmul(c2w0_inv, c2w_44)
                views_dict["camera_poses"] = torch.from_numpy(
                    c2w_aligned[np.newaxis]
                ).to(self.device)  # (1, N, 4, 4)

            if gt_config.get('use_depth'):
                cond_flags[1] = 1
                # depthmap: (1, N, H, W)
                views_dict["depthmap"] = torch.from_numpy(
                    scene['depth'].astype(np.float32)[np.newaxis]
                ).to(self.device)  # (1, N, H, W)

            if gt_config.get('use_intrinsic'):
                cond_flags[2] = 1
                # camera_intrs: (1, N, 3, 3)
                views_dict["camera_intrs"] = torch.from_numpy(
                    scene['intrinsic'].astype(np.float32)[np.newaxis]
                ).to(self.device)  # (1, N, 3, 3)

            if sum(cond_flags) > 0:
                print(f"    [WorldMirror] GT prior: cond_flags={cond_flags} "
                      f"(pose={bool(cond_flags[0])}, depth={bool(cond_flags[1])}, "
                      f"intrinsic={bool(cond_flags[2])})")

        use_amp = self.device != "cpu" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_amp else torch.float32

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                predictions = self.model(
                    views=views_dict,
                    cond_flags=cond_flags,
                )

        result = {}

        # ---- depth: (B, S, H, W, 1) -> (N, H, W) ----
        if "depth" in predictions:
            depth = predictions["depth"][0, :, :, :, 0]  # (S, H, W)
            result['pred_depth'] = depth.float().cpu().numpy().astype(np.float32)

        # ---- pose: camera_poses (B, S, 4, 4) c2w ----
        if "camera_poses" in predictions:
            c2w = predictions["camera_poses"][0].float().cpu().numpy()  # (S, 4, 4)
            # align to the first frame: c2w_aligned[i] = inv(c2w[0]) @ c2w[i]
            c2w0_inv = _inv_se3(c2w[0])  # (4, 4) closed-form SE3 inverse
            c2w_aligned = np.matmul(c2w0_inv, c2w)  # (N, 4, 4)
            result['pred_pose'] = c2w_aligned[:, :3, :4].astype(np.float32)
            # w2c = inv(c2w), closed-form SE3 inverse
            w2c = _inv_se3(c2w_aligned)
            result['w2c_extrinsics'] = w2c[:, :3, :4].astype(np.float32)

        # ---- intrinsics: camera_intrs (B, S, 3, 3) ----
        if "camera_intrs" in predictions:
            K = predictions["camera_intrs"][0].float().cpu().numpy()  # (S, 3, 3)
            result['pred_intrinsic'] = K.astype(np.float32)

        # ---- point cloud: pts3d (B, S, H, W, 3) world coordinates ----
        if "pts3d" in predictions:
            world_points = predictions["pts3d"][0].float().cpu().numpy()  # (S, H, W, 3)

            # point-cloud confidence
            if "pts3d_conf" in predictions:
                conf = predictions["pts3d_conf"][0].float().cpu().numpy()  # (S, H, W)
            else:
                conf = np.ones((N, H, W), dtype=np.float32)

            # Align the point cloud to the first-frame coordinate system (consistent with pose alignment)
            if "camera_poses" in predictions:
                # pts3d_aligned = c2w0_inv @ pts3d_homo
                all_points = []
                for i in range(N):
                    mask = conf[i] > np.percentile(conf[i], 10)
                    pts = world_points[i][mask]  # (M, 3)
                    if len(pts) > 0:
                        # Transform to the first-frame coordinate system
                        pts_homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
                        pts_aligned = (c2w0_inv @ pts_homo.T).T[:, :3]
                        all_points.append(pts_aligned)
                if all_points:
                    result['pred_pointcloud'] = np.concatenate(
                        all_points, axis=0).astype(np.float32)

        # ---- confidence: Use depth_conf or pts3d_conf ----
        conf_key = "depth_conf" if "depth_conf" in predictions else "pts3d_conf"
        if conf_key in predictions:
            pred_conf = predictions[conf_key][0].float().cpu().numpy()  # (S, H, W)
            result['pred_confidence'] = pred_conf.astype(np.float32)

        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """GT c2w align to the first frame + apply_pointcloud_normalization then convert to w2c.

        Aligned with training multiview_dataset.py processing flow:
        1. cam_align: align to the first frame
        2. Use aligned c2w + depth + intrinsic back-projection to obtain world-coordinate points
        3. apply_pointcloud_normalization: compute average point distance as norm_factor, 
           scale camera translation
        4. Convert to w2c returns
        """
        gt_c2w = scene["extrinsic"]  # (N, 3, 4) c2w
        N = gt_c2w.shape[0]

        # Expand to 4x4
        c2w_44 = np.zeros((N, 4, 4), dtype=np.float64)
        c2w_44[:, :3, :] = gt_c2w
        c2w_44[:, 3, 3] = 1.0

        # 1) align to the first frame (closed-form SE3 inverse)
        c2w0_inv = _inv_se3(c2w_44[0])
        c2w_aligned = np.matmul(c2w0_inv, c2w_44)

        # 2) Back-project world-coordinate points and compute norm_factor
        depth = scene.get('depth')        # (N, H, W)
        intrinsic = scene.get('intrinsic')  # (N, 3, 3)

        norm_factor = 1.0
        if depth is not None and intrinsic is not None:
            all_pts = []
            all_valid = []
            for i in range(N):
                H, W = depth[i].shape
                fx, fy = intrinsic[i, 0, 0], intrinsic[i, 1, 1]
                cx, cy = intrinsic[i, 0, 2], intrinsic[i, 1, 2]
                u, v = np.meshgrid(np.arange(W), np.arange(H))
                z = depth[i].astype(np.float64)
                x_cam = (u - cx) * z / fx
                y_cam = (v - cy) * z / fy
                X_cam = np.stack([x_cam, y_cam, z], axis=-1)  # (H, W, 3)
                valid = (z > 0) & np.isfinite(z)

                # Transform to the aligned world coordinate system
                R = c2w_aligned[i, :3, :3]
                t = c2w_aligned[i, :3, 3]
                X_world = np.einsum('ij,hwj->hwi', R, X_cam) + t

                all_pts.append(X_world.reshape(-1, 3))
                all_valid.append(valid.reshape(-1))

            all_pts = np.concatenate(all_pts, axis=0)
            all_valid = np.concatenate(all_valid, axis=0)

            # Aligned with training code apply_pointcloud_normalization aligns with: avg_dis
            all_pts[~all_valid] = np.nan
            dis = np.linalg.norm(all_pts, ord=2, axis=-1)
            nf = np.nanmean(dis)
            if np.isfinite(nf) and nf > 1e-8:
                norm_factor = nf

        # 3) scale camera translation
        c2w_aligned[:, :3, 3] /= norm_factor

        # 4) Convert to w2c (closed-form SE3 inverse)
        w2c = _inv_se3(c2w_aligned)
        return w2c[:, :3, :4].astype(np.float32)
