# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from addict import Dict
from omegaconf import DictConfig, OmegaConf

from depth_anything_3.cfg import create_object
from depth_anything_3.model.cam_dec import CameraDec, CameraDecRel
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from depth_anything_3.utils.geometry import (
    affine_inverse,
    as_homogeneous,
    map_pdf_to_opacity,
)
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DPT or DualDPT for depth prediction
    - Optional camera decoders for pose estimation
    - Optional GSDPT for 3DGS prediction

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
        - extrinsics: Camera extrinsics (B, N, 4, 4)
        - intrinsics: Camera intrinsics (B, N, 3, 3)
        - gaussians: 3D Gaussian Splats (world space), type: model.gs_adapter.Gaussians
        - aux: Auxiliary features for specified layers
    """

    # Patch size for feature extraction
    PATCH_SIZE = 14

    def __init__(
        self,
        net,
        head,
        cam_dec=None,
        cam_enc=None,
        gs_head=None,
        gs_adapter=None,
        relative_cam=False,
        enable_aux_cam_decs=True,
    ):
        """
        Initialize DepthAnything3Net with given yaml-initialized configuration.
        """
        super().__init__()
        self.backbone = (
            net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        )  # depth_anything_3.model.dinov2.dinov2
        self.head = (
            head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        )
        self.cam_dec, self.cam_enc = None, None
        self.aux_cam_decs = nn.ModuleList()
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec
                if isinstance(cam_dec, nn.Module)
                else create_object(_wrap_cfg(cam_dec))
            )
            if (
                relative_cam
                and isinstance(self.cam_dec, CameraDec)
                and not isinstance(self.cam_dec, CameraDecRel)
            ):
                dim_in = self.cam_dec.fc_t.in_features
                rel_cam_dec = CameraDecRel(dim_in=dim_in)
                rel_cam_dec.load_state_dict(self.cam_dec.state_dict(), strict=False)
                self.cam_dec = rel_cam_dec
            if cam_enc is not None:
                self.cam_enc = (
                    cam_enc
                    if isinstance(cam_enc, nn.Module)
                    else create_object(_wrap_cfg(cam_enc))
                )
            # Aux camera decoders supervise absolute-camera heads only. In
            # relative-camera mode their outputs are unused by the loss, and
            # running three extra pairwise rel-heads wastes a large amount of
            # memory.
            if enable_aux_cam_decs and not relative_cam:
                for _ in range(3):
                    if isinstance(cam_dec, nn.Module):
                        self.aux_cam_decs.append(copy.deepcopy(cam_dec))
                    else:
                        self.aux_cam_decs.append(create_object(_wrap_cfg(cam_dec)))
        self.gs_adapter, self.gs_head = None, None
        self.relative_cam = relative_cam
        if gs_head is not None and gs_adapter is not None:
            self.gs_adapter = (
                gs_adapter
                if isinstance(gs_adapter, nn.Module)
                else create_object(_wrap_cfg(gs_adapter))
            )
            gs_out_dim = self.gs_adapter.d_in + 1
            if isinstance(gs_head, nn.Module):
                assert gs_head.out_dim == gs_out_dim, (
                    f"gs_head.out_dim should be {gs_out_dim}, got {gs_head.out_dim}"
                )
                self.gs_head = gs_head
            else:
                assert gs_head["output_dim"] == gs_out_dim, (
                    f"gs_head output_dim should set to {gs_out_dim}, got {gs_head['output_dim']}"
                )
                self.gs_head = create_object(_wrap_cfg(gs_head))

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        kv_cache_list: list | None = None,
        return_kv_cache: bool = False,
        rel_pose_memory_feats: torch.Tensor | None = None,
        rel_pose_memory_feats_projected: bool = False,
        camera_token_is_reference: bool | None = None,
        return_memory_select_feat: bool = False,
        frame_ids: torch.Tensor | None = None,
        paged_kv_store: object | None = None,
        paged_new_frame_id: int | None = None,
        paged_active_frame_ids: list | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4)
            intrinsics: Camera intrinsics (B, N, 3, 3)
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        # Extract features using backbone
        if extrinsics is not None:
            if self.cam_enc is None:
                raise RuntimeError(
                    "Extrinsics were provided but cam_enc is None. "
                    "Pose-conditioned input requires the model to be built with a cam_enc head."
                )
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        backbone_output = self.backbone(
            x,
            cam_token=cam_token,
            frame_ids=frame_ids,
            export_feat_layers=export_feat_layers,
            ref_view_strategy=ref_view_strategy,
            kv_cache_list=kv_cache_list,
            return_kv_cache=return_kv_cache,
            camera_token_is_reference=camera_token_is_reference,
            relative_cam=self.relative_cam,
            return_memory_select_feat=return_memory_select_feat,
            paged_kv_store=paged_kv_store,
            paged_new_frame_id=paged_new_frame_id,
            paged_active_frame_ids=paged_active_frame_ids,
        )
        if return_kv_cache and return_memory_select_feat:
            feats, aux_feats, kv_cache_list, memory_select_feat = backbone_output
        elif return_kv_cache:
            feats, aux_feats, kv_cache_list = backbone_output
            memory_select_feat = None
        elif return_memory_select_feat:
            feats, aux_feats, memory_select_feat = backbone_output
        else:
            feats, aux_feats = backbone_output
            memory_select_feat = None
        # feats = [[item for item in feat] for feat in feats]
        H, W = x.shape[-2], x.shape[-1]

        # Process features through depth head
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, H, W)
            if use_ray_pose:
                output = self._process_ray_pose_estimation(output, H, W)
            else:
                output = self._process_camera_estimation(
                    feats,
                    H,
                    W,
                    output,
                    rel_pose_memory_feats=rel_pose_memory_feats,
                    rel_pose_memory_feats_projected=rel_pose_memory_feats_projected,
                )
            if infer_gs:
                output = self._process_gs_head(
                    feats, H, W, output, x, extrinsics, intrinsics
                )

        output = self._process_mono_sky_estimation(output)

        # Extract auxiliary features if requested
        output.aux = self._extract_auxiliary_features(
            aux_feats, export_feat_layers, H, W
        )
        output.cam_feat = feats[-1][1]
        output.memory_select_feat = (
            memory_select_feat if memory_select_feat is not None else output.cam_feat
        )
        if return_kv_cache:
            output.kv_cache_list = kv_cache_list

        return output

    def _process_mono_sky_estimation(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process mono sky estimation."""
        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return output
        if (~non_sky_mask).sum() <= 10:
            return output

        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(
                0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device
            )
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = torch.quantile(sampled_depth, 0.99)

        # Set sky regions to maximum depth and high confidence
        output.depth, _ = set_sky_regions_to_max_depth(
            output.depth, None, non_sky_mask, max_depth=non_sky_max
        )
        return output

    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        if "ray" in output and "ray_conf" in output:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = (
                get_extrinsic_from_camray(
                    output.ray,
                    output.ray_conf,
                    output.ray.shape[-3],
                    output.ray.shape[-2],
                )
            )
            pred_extrinsic = affine_inverse(pred_extrinsic)  # w2c -> c2w
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            pred_intrinsic = (
                torch.eye(3, 3)[None, None]
                .repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1)
                .clone()
                .to(pred_extrinsic.device)
            )
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            del output.ray
            del output.ray_conf
            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0, chunk_size=4)

    def _process_camera_estimation(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        rel_pose_memory_feats: torch.Tensor | None = None,
        rel_pose_memory_feats_projected: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            is_causal = False
            attention_mode = "causal"
            if hasattr(self.backbone, "pretrained") and hasattr(
                self.backbone.pretrained, "causal_attn"
            ):
                is_causal = bool(self.backbone.pretrained.causal_attn)
                attention_mode = getattr(
                    self.backbone.pretrained, "attention_mode", attention_mode
                )
                attention_window_size = int(
                    getattr(
                        self.backbone.pretrained,
                        "attention_window_size",
                        8,
                    )
                )
            elif hasattr(self.backbone, "causal_attn"):
                is_causal = bool(self.backbone.causal_attn)
                attention_mode = getattr(
                    self.backbone, "attention_mode", attention_mode
                )
                attention_window_size = int(
                    getattr(
                        self.backbone,
                        "attention_window_size",
                        8,
                    )
                )

            cam_output = self.cam_dec(
                feats[-1][1],
                causal=is_causal,
                attention_mode=attention_mode,
                attention_window_size=attention_window_size,
                memory_feat=rel_pose_memory_feats,
                memory_feat_projected=rel_pose_memory_feats_projected,
            )
            if isinstance(cam_output, dict):
                pose_enc = cam_output["abs_pose_enc"]
                output.rel_pose_enc = cam_output["rel_pose_enc"]
                output.rel_pose_conf = cam_output["rel_pose_conf"]
                if "rel_pose_conf_t" in cam_output:
                    output.rel_pose_conf_t = cam_output["rel_pose_conf_t"]
                if "rel_pose_conf_r" in cam_output:
                    output.rel_pose_conf_r = cam_output["rel_pose_conf_r"]
                output.rel_pose_mask = cam_output["rel_pose_mask"]
                if "rel_pose_memory_size" in cam_output:
                    output.rel_pose_memory_size = cam_output["rel_pose_memory_size"]
                if "rel_pose_projected_feat" in cam_output:
                    output.rel_pose_projected_feat = cam_output[
                        "rel_pose_projected_feat"
                    ]
                if cam_output.get("scale_pred") is not None:
                    output.scale_pred = cam_output["scale_pred"]
            else:
                pose_enc = cam_output
            # Remove ray information as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics.
            # `pose_encoding_to_extri_intri` returns c2w here.
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)  # store as w2c for downstream
            output.intrinsics = ixt
            output.pose_enc_list = [pose_enc]

            # Readout camera poses from intermediate camera tokens
            if len(self.aux_cam_decs) > 0 and not self.relative_cam:
                output.pose_enc_list_aux = []
                # print("number of feats:", len(feats))
                for i in range(min(len(feats) - 1, len(self.aux_cam_decs))):
                    aux_output = self.aux_cam_decs[i](feats[i][1])
                    if isinstance(aux_output, dict):
                        aux_pose_enc = aux_output["abs_pose_enc"]
                    else:
                        aux_pose_enc = aux_output
                    output.pose_enc_list_aux.append(aux_pose_enc)

        return output

    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Process 3DGS parameters estimation if 3DGS head is available."""
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, (
            "must provide MV depth for the GS head."
        )

        # The depth is defined in the DA3 model's camera space,
        # so even with provided GT camera poses,
        # we instead use the predicted camera poses for better alignment.
        ctx_extr = output.get("extrinsics", None)
        ctx_intr = output.get("intrinsics", None)
        assert ctx_extr is not None and ctx_intr is not None, (
            "must process camera info first if GT is not available"
        )

        gt_extr = extrinsics
        # homo the extr if needed
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        # forward through the gs_dpt head to get 'camera space' parameters
        gs_outs = self.gs_head(
            feats=feats,
            H=H,
            W=W,
            patch_start_idx=0,
            images=in_images,
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        # convert to 'world space' 3DGS parameters; ready to export and render
        # gt_extr could be None, and will be used to align the pose scale if available
        gs_world = self.gs_adapter(
            extrinsics=ctx_extr,
            intrinsics=ctx_intr,
            depths=output.depth,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H, W),
            gt_extrinsics=gt_extr,
        )
        output.gaussians = gs_world

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # feat: [B, S, N, C]
            # Check if we have a CLS/Camera token
            num_patches = (H // self.PATCH_SIZE) * (W // self.PATCH_SIZE)
            if feat.shape[2] == num_patches + 1:
                # Separate CLS token and patch tokens
                cls_token = feat[:, :, 0, :]
                patch_tokens = feat[:, :, 1:, :]
                aux_features[f"cam_layer_{feat_layer}"] = cls_token
            else:
                # Assume only patch tokens
                patch_tokens = feat

            # Reshape features to spatial dimensions
            feat_reshaped = patch_tokens.reshape(
                [
                    patch_tokens.shape[0],
                    patch_tokens.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    patch_tokens.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features


class NestedDepthAnything3Net(nn.Module):
    """
    Nested Depth Anything 3 network with metric scaling capabilities.

    This network combines two DepthAnything3Net branches:
    - Main branch: Standard depth estimation
    - Metric branch: Metric depth estimation for scaling alignment

    The network performs depth alignment using least squares scaling
    and handles sky region masking for improved depth estimation.

    Args:
        preset: Configuration for the main depth estimation branch
        second_preset: Configuration for the metric depth branch
    """

    def __init__(self, anyview: DictConfig, metric: DictConfig):
        """
        Initialize NestedDepthAnything3Net with two branches.

        Args:
            preset: Configuration for main depth estimation branch
            second_preset: Configuration for metric depth branch
        """
        super().__init__()
        self.da3 = create_object(anyview)
        self.da3_metric = create_object(metric)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both branches with metric scaling alignment.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view

        Returns:
            Dictionary containing aligned depth predictions and camera parameters
        """
        # Get predictions from both branches
        output = self.da3(
            x,
            extrinsics,
            intrinsics,
            export_feat_layers=export_feat_layers,
            infer_gs=infer_gs,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
        )
        metric_output = self.da3_metric(x)

        # Apply metric scaling and alignment
        output = self._apply_metric_scaling(output, metric_output)
        output = self._apply_depth_alignment(output, metric_output)
        output = self._handle_sky_regions(output, metric_output)

        return output

    def _apply_metric_scaling(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply metric scaling to the metric depth output."""
        # Scale metric depth based on camera intrinsics
        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )
        return output

    def _apply_depth_alignment(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply depth alignment using least squares scaling."""
        # Compute non-sky mask
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Ensure we have enough non-sky pixels
        assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"

        # Sample depth confidence for quantile computation
        depth_conf_ns = output.depth_conf[non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(
            depth_conf_ns, max_samples=100000
        )
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        # Compute alignment mask
        align_mask = compute_alignment_mask(
            output.depth_conf,
            non_sky_mask,
            output.depth,
            metric_output.depth,
            median_conf,
        )

        # Compute scale factor using least squares
        valid_depth = output.depth[align_mask]
        valid_metric_depth = metric_output.depth[align_mask]
        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

        # Apply scaling to depth and extrinsics
        output.depth *= scale_factor
        output.extrinsics[:, :, :3, 3] *= scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()

        return output

    def _handle_sky_regions(
        self,
        output: Dict[str, torch.Tensor],
        metric_output: Dict[str, torch.Tensor],
        sky_depth_def: float = 200.0,
    ) -> Dict[str, torch.Tensor]:
        """Handle sky regions by setting them to maximum depth."""
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Compute maximum depth for non-sky regions
        # Use sampling to safely compute quantile on large tensors
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(
                0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device
            )
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)

        # Set sky regions to maximum depth and high confidence
        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )

        return output
