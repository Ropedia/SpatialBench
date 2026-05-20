"""
MAPAnything model adapter.
Use MapAnything infer() high-level API for inference.

MAPAnything outputs (per view):
  - depth_z: (B, H, W, 1) Z depth
  - camera_poses: (B, 4, 4) cam2world pose
  - pts3d: (B, H, W, 3) world-coordinate points
  - conf: (B, H, W) confidence
Note: MAPAnything camera_poses is already in cam2world format, matching benchmark GT.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter


@register_adapter("mapanything")
class MAPAnythingAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.memory_efficient_inference = True
        self.use_amp = True
        self.apply_mask = False
        self.mask_edges = False

    def name(self):
        return "MAPAnything"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        from mapanything.models.mapanything.model import MapAnything

        if checkpoint and os.path.isdir(checkpoint):
            self.model = MapAnything.from_pretrained(checkpoint)
            print(f"[MAPAnythingAdapter] Model loaded from {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            self.model = MapAnything.from_pretrained(checkpoint)
            print(f"[MAPAnythingAdapter] Model loaded from {checkpoint}")
        else:
            self.model = MapAnything.from_pretrained("facebook/map-anything")
            print("[MAPAnythingAdapter] Model loaded from HuggingFace Hub")

        self.model = self.model.to(device)
        self.model.eval()

        # Get model data_norm_type
        self.data_norm_type = self.model.encoder.data_norm_type
        print(f"[MAPAnythingAdapter] Model on {device}, data_norm_type={self.data_norm_type}")

    def supports_gt_prior(self):
        return {'pose': True, 'depth': True, 'intrinsic': True, 'partial': True}

    def predict(self, scene, gt_config=None):
        """Run MAPAnything inference.

        Use MapAnything infer() API.
        Benchmark images are already ImageNet-normalized (consistent with dinov2), 
        can be used directly as MapAnything img input.

        Args:
            scene: dict from BenchmarkDataset
            gt_config: optional dict with keys:
                use_pose, use_depth, use_intrinsic (bool)
                gt_frame_indices (list[int])
        """
        images = scene['images']          # (N, 3, H, W) ImageNet normalization
        intrinsic = scene['intrinsic']    # (N, 3, 3) numpy
        N, _, H, W = images.shape

        # Parse GT prior config (camera and depth separate indices)
        camera_gt_set = set()
        depth_gt_set = set()
        use_pose = use_depth = use_intrinsic = False
        if gt_config:
            use_pose = gt_config.get('use_pose', False)
            use_depth = gt_config.get('use_depth', False)
            use_intrinsic = gt_config.get('use_intrinsic', False)
            if use_pose or use_intrinsic:
                camera_gt_set = set(gt_config.get('camera_gt_indices',
                                                   gt_config.get('gt_frame_indices', [])))
            if use_depth:
                depth_gt_set = set(gt_config.get('depth_gt_indices',
                                                  gt_config.get('gt_frame_indices', [])))

        if (camera_gt_set or depth_gt_set):
            print(f"    [MAPAnything] GT prior: camera_indices={sorted(camera_gt_set)}, "
                  f"depth_indices={sorted(depth_gt_set)}, "
                  f"pose={use_pose}, depth={use_depth}, intrinsic={use_intrinsic}")

        # Build MapAnything input views:
        # each view corresponds to one frame, batch_size=1
        views = []
        for i in range(N):
            view = {
                'img': images[i:i+1].to(self.device),             # (1, 3, H, W)
                'data_norm_type': [self.data_norm_type],
                'intrinsics': torch.from_numpy(
                    intrinsic[i:i+1]
                ).float().to(self.device),                        # (1, 3, 3)
            }

            # Inject priors for GT frames (camera and depth use separate indices)
            if i in camera_gt_set:
                if use_pose:
                    c2w_34 = scene['extrinsic'][i]  # (3, 4) c2w
                    c2w_44 = np.eye(4, dtype=np.float32)
                    c2w_44[:3, :] = c2w_34
                    view['camera_poses'] = torch.from_numpy(
                        c2w_44[np.newaxis]
                    ).float().to(self.device)  # (1, 4, 4)

            if i in depth_gt_set:
                if use_depth:
                    # depth_z: (1, H, W) - spatially matches img (1, 3, H, W) spatial dimensions match
                    view['depth_z'] = torch.from_numpy(
                        scene['depth'][i:i+1].astype(np.float32)
                    ).to(self.device)  # (1, H, W)
                    view['is_metric_scale'] = torch.tensor([True], device=self.device)  # (1,)

            views.append(view)

        # Run inference
        preds = self.model.infer(
            views,
            memory_efficient_inference=self.memory_efficient_inference,
            use_amp=self.use_amp,
            apply_mask=self.apply_mask,
            mask_edges=self.mask_edges,
            ignore_calibration_inputs=not use_intrinsic,
            ignore_depth_inputs=not use_depth,
            ignore_pose_inputs=not use_pose,
        )

        result = {}

        # depth: depth_z (1, H, W, 1) -> (H, W) per view, stack into (N, H, W)
        depth_list = []
        for pred in preds:
            if 'depth_z' in pred:
                dz = pred['depth_z'][0, :, :, 0]  # (H, W)
                depth_list.append(dz.cpu().numpy())
        if depth_list:
            result['pred_depth'] = np.stack(depth_list).astype(np.float32)

        # pose: camera_poses (1, 4, 4) cam2world -> stack into (N, 4, 4)
        pose_list = []
        for pred in preds:
            if 'camera_poses' in pred:
                c2w = pred['camera_poses'][0].cpu().numpy()  # (4, 4)
                pose_list.append(c2w)
        if pose_list:
            c2w_all = np.stack(pose_list).astype(np.float64)  # (N, 4, 4)
            result['pred_pose'] = c2w_all[:, :3, :4].astype(np.float32)
            w2c_all = self._invert_se3(result['pred_pose'])  # (N, 3, 4)
            result['w2c_extrinsics'] = w2c_all.astype(np.float32)

        # point cloud: pts3d (1, H, W, 3) per view
        all_points = []
        for i, pred in enumerate(preds):
            if 'pts3d' not in pred:
                continue
            pts = pred['pts3d'][0].cpu().numpy()  # (H, W, 3)
            # Filter by confidence
            if 'conf' in pred:
                conf = pred['conf'][0].cpu().numpy()  # (H, W)
                mask = conf > np.percentile(conf, 10)
            else:
                mask = np.ones((H, W), dtype=bool)
            # Exclude invalid depth points
            if 'depth_z' in pred:
                dz = pred['depth_z'][0, :, :, 0].cpu().numpy()
                mask = mask & (dz > 0) & np.isfinite(dz)
            valid_pts = pts[mask]
            if len(valid_pts) > 0:
                all_points.append(valid_pts)
        if all_points:
            result['pred_pointcloud'] = np.concatenate(
                all_points, axis=0
            ).astype(np.float32)

        # confidence: conf (1, H, W) per view -> (N, H, W)
        conf_list = []
        for pred in preds:
            if 'conf' in pred:
                conf_list.append(pred['conf'][0].cpu().numpy())
        if conf_list:
            result['pred_confidence'] = np.stack(conf_list).astype(np.float32)

        return result

    def supports_metric_depth(self):
        return True
    
    def normalize_gt_poses(self, scene):
        """VGGT normalization: align to first camera + scale by avg point distance.

        consistent with training normalize_camera_extrinsics_and_points_batch consistent.
        """
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
        
    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """MAPAnything visualization: depth + pose back-projection, percentile confidence filtering."""
        from benchmark.evaluation.metrics import unproject_to_pointcloud
        from benchmark.utils.visualization import (
            save_pointcloud_glb, _collect_colors,
        )

        scene_id = scene["scene_id"]
        images_raw = scene["images_raw"]
        pred_depth = predictions.get("pred_depth")
        pred_conf = predictions.get("pred_confidence")
        if pred_depth is None:
            return

        pred_poses = predictions.get("pred_pose", scene["extrinsic"])
        # MAPAnything does not output pred_intrinsic; use GT intrinsics
        intrinsic = scene["intrinsic"]

        pred_valid = (pred_depth > 0) & np.isfinite(pred_depth)
        if z_far is not None:
            pred_valid = pred_valid & (pred_depth < z_far)

        sky_mask = scene.get("sky_mask")
        if sky_mask is not None:
            pred_valid = pred_valid & ~sky_mask

        if pred_conf is not None and vis_conf_percent > 0:
            conf_valid = pred_conf[pred_valid]
            if len(conf_valid) > 0:
                threshold_val = np.percentile(conf_valid, vis_conf_percent)
                pred_valid = pred_valid & (pred_conf >= threshold_val)
                print(f"    Conf filter: percentile={vis_conf_percent}%, "
                      f"threshold={threshold_val:.4f}, "
                      f"range=[{conf_valid.min():.4f}, {conf_valid.max():.4f}]")

        pred_points = unproject_to_pointcloud(
            pred_depth, pred_poses, intrinsic, pred_valid)
        if len(pred_points) == 0:
            return

        pred_colors = _collect_colors(images_raw, pred_valid)
        suffix = "_pred_pred_pose"
        if vis_conf_percent > 0:
            suffix += f"_top{int(100 - vis_conf_percent)}pct"
        pred_glb_path = os.path.join(output_dir, f"{scene_id}{suffix}.glb")
        N, _, H, W = images_raw.shape
        save_pointcloud_glb(pred_points, pred_colors, pred_glb_path,
                            extrinsics=pred_poses, intrinsics=intrinsic,
                            frustum_scale=0.1, image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
              f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")
