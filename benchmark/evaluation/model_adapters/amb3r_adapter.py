"""
AMB3R model adapter.
AMB3R = Accurate feed-forward Metric-scale 3D Reconstruction with Backend.
Multi-view 3D reconstruction model based on a VGGT encoder-decoder and Point Transformer V3 backend.

AMB3R outputs:
  - depth_metric: (B, T, H, W, 1) metric depth (metric scale)
  - pose: (B, T, 4, 4) cam2world pose
  - world_points: (B, T, H, W, 3) 3D points in world coordinates
  - world_points_conf: (B, T, H, W, 1) confidence
  - extrinsic: (B, T, 3, 4) world2cam extrinsics
  - intrinsic: (B, T, 3, 3) intrinsics
  - pts3d_by_unprojection: (B, T, H, W, 3) 3D points back-projected from depth

Note:
  - AMB3R input requires [-1, 1] normalization (not ImageNet normalization)
  - Resolution requirement: H/W must be divisible by 14 (patch tokenization); 518x392 is recommended
  - use bfloat16 mixed-precision inference
  - pose output is c2w (4x4); extrinsic output is w2c (3x4)
"""
import sys
import os
import types
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager

_AMB3R_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'amb3r_root')
_AMB3R_THIRDPARTY = os.path.join(_AMB3R_ROOT, 'thirdparty')


@contextmanager
def _amb3r_context():
    """Temporarily manage sys.path and sys.modules, isolate amb3r dependencies such as vggt/dust3r/croco.

    amb3r dependency chain:
      amb3r_root/amb3r/ - main package (import vggt, ptv3, moge)
      amb3r_root/thirdparty/ - contains vggt, ptv3, moge, croco, dust3r etc.
    These modules may conflict with benchmark/models/ modules with the same names under benchmark/models/ and must be isolated.
    """
    _prefixes = ('vggt', 'dust3r', 'croco', 'moge', 'ptv3', 'segformer',
                 'depth_anything_3', 'robustmvd', 'amb3r')

    saved_modules = {k: v for k, v in sys.modules.items()
                     if any(k == p or k.startswith(p + '.') for p in _prefixes)}
    for k in saved_modules:
        del sys.modules[k]

    saved_path = sys.path[:]
    _root_abs = os.path.abspath(_AMB3R_ROOT)
    _thirdparty_abs = os.path.abspath(_AMB3R_THIRDPARTY)

    # Build a clean path: remove paths that may conflict
    clean_path = []
    known = {_root_abs, _thirdparty_abs}
    for p in sys.path:
        pa = os.path.abspath(p)
        if pa in known:
            continue
        if os.path.isdir(os.path.join(pa, 'vggt')) and pa != _thirdparty_abs:
            continue
        clean_path.append(p)
    sys.path[:] = [_root_abs, _thirdparty_abs] + clean_path

    try:
        yield
    finally:
        cut_modules = [k for k in sys.modules
                       if any(k == p or k.startswith(p + '.') for p in _prefixes)]
        for k in cut_modules:
            del sys.modules[k]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter


def _patched_resize_feat(self, feat, scale_factor=None, target_size=None):
    """Drop-in replacement for AMB3R.resize_feat that chunks F.interpolate.

    Why: upsample_bilinear2d_nhwc uses int32 indexing, so output element count
    must stay below INT_MAX. Long sequences (e.g. 1000 frames at 1024×56×74)
    overflow. Split the merged batch dim into chunks and concat.
    """
    if scale_factor is None and target_size is None:
        raise ValueError("Either scale_factor or target_size must be provided")
    if scale_factor is not None:
        target_size = (int(feat.shape[2] * scale_factor), int(feat.shape[3] * scale_factor))

    Bs, T, H, W, C = feat.shape
    feat = feat.permute(0, 1, 4, 2, 3).flatten(0, 1)
    chunk_size = 256
    if feat.shape[0] > chunk_size:
        feat = torch.cat(
            [F.interpolate(c, size=target_size, mode='bilinear', align_corners=False)
             for c in feat.split(chunk_size, dim=0)],
            dim=0,
        )
    else:
        feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
    feat = feat.permute(0, 2, 3, 1).view(Bs, T, target_size[0], target_size[1], C)
    return feat


@register_adapter("amb3r")
class AMB3RAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        # AMB3R inference parameters
        self.resolution = (518, 392)  # (W, H) - must be divisible by 14
        self.data_type = "bf16"  # bf16 | fp16 | fp32

    def name(self):
        return "AMB3R"

    def configure(self, **kwargs):
        for key, val in kwargs.items():
            if val is None:
                continue
            if key == "resolution":
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    self.resolution = tuple(int(x) for x in val)
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device

        with _amb3r_context():
            from amb3r.model import AMB3R

            self.model = AMB3R(device=device, metric_scale=True)

            if checkpoint and os.path.isfile(checkpoint):
                self.model.load_weights(checkpoint, data_type=self.data_type)
                print(f"[AMB3RAdapter] Model loaded from: {checkpoint}")
            elif checkpoint and os.path.isdir(checkpoint):
                # Find .pt file
                for f in os.listdir(checkpoint):
                    if f.endswith('.pt') or f.endswith('.pth'):
                        ckpt_path = os.path.join(checkpoint, f)
                        self.model.load_weights(ckpt_path, data_type=self.data_type)
                        print(f"[AMB3RAdapter] Model loaded from: {ckpt_path}")
                        break
            else:
                # Try downloading from HuggingFace
                repo_id = checkpoint or "hywang01/AMB3R"
                from benchmark.utils.hf_weights import ensure_hf_snapshot
                from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR
                weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
                try:
                    snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
                    for f in os.listdir(snapshot_dir):
                        if f.endswith('.pt') or f.endswith('.pth'):
                            ckpt_path = os.path.join(snapshot_dir, f)
                            self.model.load_weights(ckpt_path, data_type=self.data_type)
                            print(f"[AMB3RAdapter] Model loaded from HF: {ckpt_path}")
                            break
                except Exception as e:
                    print(f"[AMB3RAdapter] Warning: could not load from HF ({e}), trying local")

        self.model = self.model.to(device)
        self.model.eval()
        self.model.resize_feat = types.MethodType(_patched_resize_feat, self.model)
        print(f"[AMB3RAdapter] AMB3R on {device}")

    def predict(self, scene):
        """Run AMB3R inference.

        Flow:
        1. Convert benchmark images to AMB3R input format ([-1, 1] normalization, (B, T, 3, H, W))
        2. Call model.run_amb3r_benchmark() (frontend + backend)
        3. Extract depth, pose, pointcloud, confidence, intrinsics
        """
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # AMB3R use [-1, 1] normalization: img = img * 2 - 1
        images_normed = images_raw * 2.0 - 1.0  # [-1, 1]

        # Build AMB3R input: (B=1, T=N, 3, H, W)
        frames = {
            'images': images_normed.unsqueeze(0).to(self.device),  # (1, N, 3, H, W)
        }

        with _amb3r_context():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    res = self.model.run_amb3r_benchmark(frames)

        result = {}

        def _tensor_depth_to_numpy(tensor):
            depth = tensor[0].detach().cpu().float().numpy()
            pred_depth = depth.squeeze(-1) if depth.ndim == 4 and depth.shape[-1] == 1 else depth
            if pred_depth.shape[1] != H or pred_depth.shape[2] != W:
                import cv2
                pred_depth = np.stack([
                    cv2.resize(pred_depth[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    for i in range(N)
                ])
            return pred_depth.astype(np.float32)

        def _tensor_points_to_numpy(tensor):
            pts = tensor[0].detach().cpu().float().numpy()
            if pts.shape[1] != H or pts.shape[2] != W:
                import cv2
                pts = np.stack([
                    cv2.resize(pts[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    for i in range(N)
                ])
            return pts.astype(np.float32)

        # pose: c2w (B, T, 4, 4) -> (N, 3, 4)
        if 'pose' in res and res['pose'] is not None:
            poses_c2w = res['pose'][0].detach().cpu().float().numpy()  # (N, 4, 4)
            result['pred_pose'] = poses_c2w[:, :3, :4].astype(np.float32)

            # w2c: c2w inverse transform (closed-form SE3)
            w2c_list = []
            for i in range(N):
                R = poses_c2w[i, :3, :3]
                t = poses_c2w[i, :3, 3]
                w2c = np.zeros((3, 4), dtype=np.float32)
                w2c[:3, :3] = R.T
                w2c[:3, 3] = -R.T @ t
                w2c_list.append(w2c)
            result['w2c_extrinsics'] = np.stack(w2c_list).astype(np.float32)

        if 'depth' in res and res['depth'] is not None:
            result['pred_depth_raw'] = _tensor_depth_to_numpy(res['depth'])

        # depth_metric is the metric-scale depth produced by AMB3R's metric head.
        if 'depth_metric' in res and res['depth_metric'] is not None:
            result['pred_depth'] = _tensor_depth_to_numpy(res['depth_metric'])
        elif 'pred_depth_raw' in result:
            result['pred_depth'] = result['pred_depth_raw']

        # intrinsics: (B, T, 3, 3) -> (N, 3, 3)
        if 'intrinsic' in res and res['intrinsic'] is not None:
            pred_intrinsic = res['intrinsic'][0].detach().cpu().float().numpy()  # (N, 3, 3)
            result['pred_intrinsic'] = pred_intrinsic.astype(np.float32)

        if 'world_points_conf' in res and res['world_points_conf'] is not None:
            conf = res['world_points_conf'][0].detach().cpu().float().numpy()
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf.squeeze(-1)
            if conf.shape[1] != H or conf.shape[2] != W:
                import cv2
                conf = np.stack([
                    cv2.resize(conf[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    for i in range(N)
                ])
            result['pred_confidence'] = conf.astype(np.float32)

        # Match amb3r/demo.py: use backend-refined world_points as the primary
        # point-cloud reconstruction. Point-cloud evaluation has an AMB3R-specific
        # branch in run_benchmark.py that consumes this field directly.
        if 'world_points' in res and res['world_points'] is not None:
            from benchmark.utils.visualization import _collect_colors

            pred_world_points = _tensor_points_to_numpy(res['world_points'])
            result['pred_world_points'] = pred_world_points

            valid = np.isfinite(pred_world_points).all(axis=-1)
            result['pred_pointcloud_mask'] = valid
            if valid.any():
                result['pred_pointcloud'] = pred_world_points[valid].astype(np.float32)
                result['pred_pointcloud_colors'] = _collect_colors(images_raw, valid)

        torch.cuda.empty_cache()
        return result

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """Visualize raw AMB3R depth back-projected with AMB3R camera pose."""
        from benchmark.evaluation.metrics import unproject_to_pointcloud
        from benchmark.utils.visualization import save_pointcloud_glb, _collect_colors

        if ("pred_depth_raw" not in predictions or
                "pred_pose" not in predictions or
                "pred_intrinsic" not in predictions):
            return super().visualize_prediction(
                scene, predictions, output_dir,
                z_far=z_far, vis_conf_percent=vis_conf_percent)

        scene_id = scene["scene_id"]
        images_raw = scene["images_raw"]
        pred_depth = predictions["pred_depth_raw"]
        pred_pose = predictions["pred_pose"]
        pred_intrinsic = predictions["pred_intrinsic"]

        pred_valid = (pred_depth > 1e-6) & np.isfinite(pred_depth)
        if z_far is not None:
            pred_valid = pred_valid & (pred_depth < z_far)

        sky_mask = scene.get("sky_mask")
        if sky_mask is not None:
            pred_valid = pred_valid & ~sky_mask

        if not pred_valid.any():
            return

        pred_points = unproject_to_pointcloud(
            pred_depth, pred_pose, pred_intrinsic, pred_valid)
        if len(pred_points) == 0:
            return

        pred_colors = _collect_colors(images_raw, pred_valid)
        pred_glb_path = os.path.join(output_dir, f"{scene_id}_pred_raw_depth_pose.glb")

        N, _, H, W = images_raw.shape
        save_pointcloud_glb(pred_points, pred_colors, pred_glb_path,
                            extrinsics=pred_pose, intrinsics=pred_intrinsic,
                            frustum_scale=0.04, image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} raw-depth pts, "
              f"z_far={z_far})")

    def supports_metric_depth(self):
        return True

    def requires_intrinsics(self):
        return False

    def normalize_gt_poses(self, scene):
        """AMB3R use VGGT normalization method: align-to-first + scale."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
