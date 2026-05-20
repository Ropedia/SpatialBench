"""
Pi3 model adapter.
Use the Pi3 forward() API for inference.

Pi3 outputs:
  - camera_poses: (B, N, 4, 4) cam2world pose (OpenCV coordinate system)
  - points: (B, N, H, W, 3) world-coordinate points
  - local_points: (B, N, H, W, 3) camera-coordinate points
  - conf: (B, N, H, W, 1) confidence (requires sigmoid)

Note: Pi3 camera_poses is already in cam2world format, matching benchmark GT.
     Pi3 internally applies ImageNet normalization, input should be [0, 1] range.
     Pi3 does not output intrinsics; estimate them via recover_intrinsic_from_rays_d from local_points.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.models.omnivggt.utils.geometry import closed_form_inverse_se3
from pi3.utils.geometry import recover_intrinsic_from_rays_d


@register_adapter("pi3")
class Pi3Adapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"

    def name(self):
        return "Pi3"

    def load_model(self, checkpoint=None, device="cuda"):
        self.device = device
        from pi3.models.pi3 import Pi3 as Pi3Model

        if checkpoint and os.path.isdir(checkpoint):
            self.model = Pi3Model.from_pretrained(checkpoint)
            print(f"[Pi3Adapter] Model loaded from {checkpoint}")
        elif checkpoint and os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = Pi3Model()
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[Pi3Adapter] Model loaded from {checkpoint}")
        else:
            self.model = Pi3Model.from_pretrained("yyfz233/Pi3")
            print("[Pi3Adapter] Model loaded from HuggingFace Hub")

        self.model = self.model.to(device)
        self.model.eval()
        print(f"[Pi3Adapter] Model on {device}")

    def predict(self, scene):
        """Run Pi3 inference.

        Pi3 input: imgs (B, N, 3, H, W) in [0, 1] (the model applies ImageNet normalization)
        Pi3 outputs: camera_poses (B, N, 4, 4) cam2world, points (B, N, H, W, 3),
                   local_points (B, N, H, W, 3), conf (B, N, H, W, 1)
        """
        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N, _, H, W = images_raw.shape

        # Pi3 expects (B, N, 3, H, W)
        images_input = images_raw.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            outputs = self.model(images_input)

        result = {}

        # depth: local_points[..., 2] i.e. Z-depth (B, N, H, W)
        if 'local_points' in outputs:
            local_depth = outputs['local_points'][0, :, :, :, 2]  # (N, H, W)
            result['pred_depth'] = local_depth.cpu().numpy().astype(np.float32)


        # pose: camera_poses (B, N, 4, 4) is already cam2world
        # Pi3 output c2w is not aligned to the first frame and needs align-to-first:
        #   c2w_aligned[i] = inv(c2w[0]) @ c2w[i]  (the first frame becomes identity)
        if 'camera_poses' in outputs:
            c2w = outputs['camera_poses'][0].cpu().numpy()  # (N, 4, 4)
            # align to the first frame
            c2w0_inv = closed_form_inverse_se3(c2w[0:1])[0]  # (4, 4)
            c2w = np.matmul(c2w0_inv, c2w)  # (N, 4, 4)
            result['pred_pose'] = c2w[:, :3, :4].astype(np.float32)
            w2c = closed_form_inverse_se3(c2w)  # (N, 4, 4)
            result['w2c_extrinsics'] = w2c[:, :3, :4].astype(np.float32)


        # point cloud: points (B, N, H, W, 3) world coordinates
        if 'points' in outputs:
            world_points = outputs['points'][0].cpu().numpy()  # (N, H, W, 3)

            # conf requires sigmoid
            if 'conf' in outputs:
                conf = torch.sigmoid(outputs['conf'][0, :, :, :, 0]).cpu().numpy()  # (N, H, W)
            else:
                conf = np.ones((N, H, W), dtype=np.float32)

            all_points = []
            for i in range(N):
                mask = conf[i] > 0.05
                # Exclude invalid depth
                if 'local_points' in outputs:
                    dz = outputs['local_points'][0, i, :, :, 2].cpu().numpy()
                    mask = mask & (dz > 0) & np.isfinite(dz)
                pts = world_points[i][mask]
                if len(pts) > 0:
                    all_points.append(pts)

            if all_points:
                result['pred_pointcloud'] = np.concatenate(
                    all_points, axis=0
                ).astype(np.float32)

        # confidence: sigmoid(conf) -> (N, H, W)
        if 'conf' in outputs:
            conf_np = torch.sigmoid(outputs['conf'][0, :, :, :, 0]).cpu().numpy()
            result['pred_confidence'] = conf_np.astype(np.float32)

        # intrinsicsestimate: Recover intrinsics from rays_d of local_points
        if 'local_points' in outputs:
            rays_d = torch.nn.functional.normalize(outputs['local_points'][0], dim=-1)  # (N, H, W, 3)
            K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)  # (N, 3, 3)
            result['pred_intrinsic'] = K.cpu().numpy().astype(np.float32)
            print(f"[Pi3Adapter] Estimated first frame intrinsic:\n{K[0].cpu().numpy()}")

        return result

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        """GT c2w align to the first framethen convert to w2c, consistent with predict processing.

        aligned_c2w[i] = inv(c2w[0]) @ c2w[i], thenConvert to w2c.
        """
        gt_c2w = scene["extrinsic"]  # (N, 3, 4)
        N = gt_c2w.shape[0]
        # Expand to 4x4
        c2w_44 = np.zeros((N, 4, 4), dtype=np.float64)
        c2w_44[:, :3, :] = gt_c2w
        c2w_44[:, 3, 3] = 1.0
        # align to the first frame
        c2w0_inv = closed_form_inverse_se3(c2w_44[0:1])[0]  # (4, 4)
        c2w_aligned = np.matmul(c2w0_inv, c2w_44)  # (N, 4, 4)
        # Convert to w2c
        w2c = closed_form_inverse_se3(c2w_aligned)  # (N, 4, 4)
        return w2c[:, :3, :4].astype(np.float32)

    def visualize_prediction(self, scene, predictions, output_dir,
                             z_far=None, vis_conf_percent=50.0):
        """Pi3 visualization: use world points, sigmoid confidence filtering.

        Pi3 confidence is sigmoid-activated (range [0, 1]), 
        predict() uses a > 0.05 threshold to filter the point cloud.
        """
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
        # Prefer estimated intrinsics, fall back to GT intrinsics
        intrinsic = predictions.get("pred_intrinsic", scene["intrinsic"])

        pred_valid = (pred_depth > 0) & np.isfinite(pred_depth)
        if z_far is not None:
            pred_valid = pred_valid & (pred_depth < z_far)

        # sigmoid confidencepercentile filtering
        if pred_conf is not None and vis_conf_percent > 0:
            conf_valid = pred_conf[pred_valid]
            if len(conf_valid) > 0:
                threshold_val = np.percentile(conf_valid, vis_conf_percent)
                pred_valid = pred_valid & (pred_conf >= threshold_val)
                print(f"    Conf filter (sigmoid): percentile={vis_conf_percent}%, "
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
                            image_size=(W, H))
        print(f"    Pred -> {pred_glb_path} ({len(pred_points)} pts, "
              f"z_far={z_far}, top {int(100 - vis_conf_percent)}%)")
