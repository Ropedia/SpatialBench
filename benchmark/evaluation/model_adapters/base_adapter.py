"""
Abstract base class for model adapters: defines the standard interface between the benchmark and the models.
All model adapters must implement this interface.
"""
import os
from abc import ABC, abstractmethod

import numpy as np


class ModelAdapter(ABC):
    """Standard interface for 3D reconstruction models.

    Each adapter wraps a specific model (VGGT, DA3, CUT3R, etc.) and
    converts the benchmark data format to/from the model's input/output format.

    To implement a new model adapter you only need to:
    1. Subclass ModelAdapter
    2. Implement the name(), load_model(), and predict() methods
    3. Register it with @register_adapter("model_name")
    4. (Optional) Override visualize_prediction() to customize visualization
    """

    @abstractmethod
    def name(self):
        """Model name (used in reports)."""
        pass

    @abstractmethod
    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        """Load model weights.

        Args:
            checkpoint: model weights path (optional; some models have a default path)
            device: inference device
            weights_dir: if checkpoint is a HuggingFace hub repo_id, weights can be downloaded/cached into this directory
        """
        pass

    @abstractmethod
    def predict(self, scene):
        """Run inference on a single scene.

        Args:
            scene: dict from BenchmarkDataset.__getitem__(), containing:
                images: Tensor (N, 3, H, W) - ImageNet normalized
                images_raw: Tensor (N, 3, H, W) - unnormalized
                intrinsic: np.ndarray (N, 3, 3)
                depth: np.ndarray (N, H, W) - GT depth (must not be used for inference)
                extrinsic: np.ndarray (N, 3, 4) - GT pose (must not be used for inference)

        Returns:
            dict, may contain any subset of the following (the framework skips missing ones):
                pred_depth: np.ndarray (N, H, W) - predicted depth
                pred_pose: np.ndarray (N, 3, 4) - predicted cam2world pose
                pred_pointcloud: np.ndarray (M, 3) - predicted point cloud
                pred_confidence: np.ndarray (N, H, W) - per-pixel confidence
        """
        pass

    def prepare(self, scene):
        """Optional: data preparation stage. Called by the benchmark before predict(),
        NOT counted in inference time. Default is a no-op.

        Use case: when the adapter needs to do disk I/O or expensive data preprocessing
        inside predict() (such as writing in-memory tensors to disk as PNGs or
        building heavy batch structures), moving that work to prepare() lets time_s
        reflect only the actual inference + post-processing time.

        Convention: all results should be saved as attributes on self (e.g. self._batches),
        and the subsequent predict(scene) reuses those attributes. Subclasses may also do nothing.
        """
        return None

    def supports_metric_depth(self):
        """Whether the model outputs metric-scale depth (no alignment required).
        Default False (relative depth, requires median-scale alignment).
        """
        return False

    def supports_gt_prior(self):
        """Return the GT prior injection capabilities supported by the model.

        Returns:
            dict: {
                'pose': bool,       # whether GT camera pose injection is supported
                'depth': bool,      # whether GT depth injection is supported
                'intrinsic': bool,  # whether GT intrinsic injection is supported
                'partial': bool,    # whether per-frame selective injection is supported (True=can specify a subset of frames, False=all or none)
            }
        """
        return {'pose': False, 'depth': False, 'intrinsic': False, 'partial': False}

    def configure(self, **kwargs):
        """Inject model-specific inference parameters from a config file.

        run_benchmark passes non-standard fields from the YAML (such as chunk_size, overlap,
        ref_view_strategy, etc.) to the adapter through this method.
        Default implementation: only sets attributes that already exist on the adapter; unknown keys are ignored.
        Subclasses may override for validation or conversion.
        """
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def requires_intrinsics(self):
        """Whether intrinsics are required as input. Default True."""
        return True

    def max_frames(self):
        """Maximum number of frames the model can process at once. None means unlimited."""
        return None

    def get_model_params(self):
        """Compute model parameter counts and return as a dict.

        Returns:
            dict: {
                "total_params": int,        # total parameter count
                "trainable_params": int,     # trainable parameter count
                "frozen_params": int,        # frozen parameter count
                "total_params_M": float,     # total parameter count (millions)
            }
            Returns an empty dict if the model is not loaded or is not an nn.Module.
        """
        model = getattr(self, 'model', None)
        if model is None:
            return {}
        try:
            import torch.nn as nn
            if not isinstance(model, nn.Module):
                return {}
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total - trainable
            return {
                "total_params": total,
                "trainable_params": trainable,
                "frozen_params": frozen,
                "total_params_M": round(total / 1e6, 2),
            }
        except Exception:
            return {}

    def normalize_gt_poses(self, scene):
        """Normalize GT poses to the same coordinate frame as the model prediction and return w2c.

        Different models may use different normalization strategies (align first, scale by points, etc.).
        The default does no processing and simply converts GT c2w to w2c and returns it.
        Subclasses may override as needed.

        Args:
            scene: dict from BenchmarkDataset, containing extrinsic, depth, intrinsic, valid_mask, world_points, etc.

        Returns:
            np.ndarray (N, 3, 4) normalized GT world-to-camera poses
        """
        return self._invert_se3(scene["extrinsic"])

    @staticmethod
    def _invert_se3(se3_34):
        """Invert (N, 3, 4) SE3 matrices → (N, 3, 4).  R' = R^T, t' = -R^T @ t."""
        N = se3_34.shape[0]
        out = np.zeros_like(se3_34)
        for i in range(N):
            R, t = se3_34[i, :3, :3], se3_34[i, :3, 3]
            out[i, :3, :3] = R.T
            out[i, :3, 3] = -R.T @ t
        return out

    @staticmethod
    def normalize_camera_extrinsics_and_points(gt_c2w, world_points, valid_mask):
        """Normalization aligned with DA3's training-time normalize_camera_extrinsics_and_points_batch.

        Align to first camera + scale by avg point distance.
        Input is in c2w format (benchmark standard); internally converts to w2c, computes, and returns normalized w2c.

        Args:
            gt_c2w: (N, 3, 4) cam2world
            world_points: (N, H, W, 3) world-coordinate points
            valid_mask: (N, H, W) bool

        Returns:
            gt_w2c_norm: (N, 3, 4) normalized world-to-camera
        """
        N = gt_c2w.shape[0]

        # c2w -> w2c
        gt_w2c = np.zeros((N, 3, 4), dtype=np.float32)
        for i in range(N):
            R, t = gt_c2w[i, :3, :3], gt_c2w[i, :3, 3]
            gt_w2c[i, :3, :3] = R.T
            gt_w2c[i, :3, 3] = -R.T @ t

        # --- 1) align to first camera: T_i' = T_i @ T_0^{-1} ---
        # T_0^{-1} = c2w of first camera
        R0, t0 = gt_w2c[0, :3, :3], gt_w2c[0, :3, 3]

        w2c_44 = np.zeros((N, 4, 4), dtype=np.float32)
        w2c_44[:, :3, :] = gt_w2c
        w2c_44[:, 3, 3] = 1.0

        inv0 = np.eye(4, dtype=np.float32)
        inv0[:3, :3] = R0.T
        inv0[:3, 3] = -R0.T @ t0
        new_w2c_44 = w2c_44 @ inv0  # (N, 4, 4)

        # --- 2) world points -> first camera coord system ---
        # new_wp = R0 @ world_points + t0  (equivalent to DA3's world_points @ R.T + t)
        new_wp = np.einsum('ij,nhwj->nhwi', R0, world_points) + t0  # (N, H, W, 3)

        # --- 3) scale by avg point distance ---
        dist = np.linalg.norm(new_wp, axis=-1)  # (N, H, W)
        mask_f = valid_mask.astype(np.float32)
        dist_sum = (dist * mask_f).sum()
        valid_count = mask_f.sum()
        avg_scale = np.clip(dist_sum / (valid_count + 1e-3), 1e-6, 1e6)

        new_w2c_44[:, :3, 3] /= avg_scale

        # Return normalized w2c (N, 3, 4)
        return new_w2c_44[:, :3, :].copy()

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """Visualize the model's prediction and save it as a GLB point cloud file.

        The default implementation supports two paths:
          A) The model outputs pred_pointcloud directly (without pred_depth)
          B) Unproject using pred_depth + pred_pose + intrinsic

        Subclasses may override this method to customize visualization logic.

        Args:
            scene: dict from BenchmarkDataset
            predictions: dict from self.predict()
            output_dir: output directory
            z_far: maximum-depth filter (None = no filtering)
            vis_conf_percent: drop the lowest-confidence N% of points (0 = no filtering, 50 = drop the lowest 50%)
        """
        from benchmark.evaluation.metrics import unproject_to_pointcloud
        from benchmark.utils.visualization import (
            save_pointcloud_glb, color_pointcloud_by_projection, _collect_colors,
        )

        scene_id = scene["scene_id"]
        gt_poses = scene["extrinsic"]
        gt_intrinsic = scene["intrinsic"]
        images_raw = scene["images_raw"]

        pred_points = None
        pred_colors = None
        suffix = "_pred"

        if "pred_pointcloud" in predictions and "pred_depth" not in predictions:
            # Path A: directly use the point cloud produced by the model
            pred_points = predictions["pred_pointcloud"]

            if "pred_pointcloud_colors" in predictions:
                pred_colors = predictions["pred_pointcloud_colors"]
            elif "pred_pose" in predictions and "pred_intrinsic" in predictions:
                pred_colors = color_pointcloud_by_projection(
                    pred_points, predictions["pred_pose"],
                    predictions["pred_intrinsic"], images_raw)
            elif "pred_pose" in predictions:
                pred_colors = color_pointcloud_by_projection(
                    pred_points, predictions["pred_pose"], gt_intrinsic, images_raw)
            else:
                pred_colors = np.full((len(pred_points), 3), 0.6, dtype=np.float32)

            suffix = "_pred_pointcloud"
        elif "pred_depth" in predictions:
            # Path B: depth + pose + intrinsic unprojection
            pred_depth = predictions["pred_depth"]
            pred_conf = predictions.get("pred_confidence", None)

            if "pred_pose" in predictions:
                pred_poses = predictions["pred_pose"]
                suffix = "_pred_pred_pose"
            else:
                pred_poses = gt_poses
                suffix = "_pred_gt_pose"

            pred_intrinsic = predictions.get("pred_intrinsic", gt_intrinsic)

            if z_far is not None:
                pred_valid = (pred_depth > 0) & np.isfinite(pred_depth) & (pred_depth < z_far)
            else:
                pred_valid = (pred_depth > 0) & np.isfinite(pred_depth)

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
                          f"conf range=[{conf_valid.min():.4f}, {conf_valid.max():.4f}]")

            pred_points = unproject_to_pointcloud(
                pred_depth, pred_poses, pred_intrinsic, pred_valid)
            if len(pred_points) > 0:
                pred_colors = _collect_colors(images_raw, pred_valid)
        else:
            return

        if pred_points is not None and len(pred_points) > 0:
            if vis_conf_percent > 0:
                suffix += f"_top{int(100 - vis_conf_percent)}pct"
            pred_glb_path = os.path.join(output_dir, f"{scene_id}{suffix}.glb")

            # Camera frustum: prefer pred_pose, fall back to gt_pose
            vis_poses = predictions.get("pred_pose", gt_poses)
            vis_intrinsic = predictions.get("pred_intrinsic", gt_intrinsic)
            N, _, H, W = images_raw.shape

            save_pointcloud_glb(pred_points, pred_colors, pred_glb_path,
                                extrinsics=vis_poses, intrinsics=vis_intrinsic,
                                frustum_scale=0.04, image_size=(W, H))
            print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
                  f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")
