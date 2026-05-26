# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging
from collections import defaultdict
from functools import partial
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from hydra.utils import instantiate
from torch.distributed.nn.functional import all_reduce

from vggttt.data.utils import compute_adaptive_minibatch_size
from vggttt.nets.ttt import TTTOperator
from vggttt.nets.vggt.heads.camera_head import CameraHead
from vggttt.nets.vggt.heads.dpt_head import DPTHead
from vggttt.nets.vggt.heads.track_head import TrackHead
from vggttt.nets.vggt.layers.attention import Attention
from vggttt.nets.vggt.models.aggregator import Aggregator
from vggttt.nets.vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggttt.nets.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggttt.utils.dist import get_sp_group

_logger = logging.getLogger(__name__)


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        intermediate_layer_idx: Sequence[int] = (4, 11, 17, 23),
        gradient_checkpoint: bool = True,
        ttt_query_images: int = 0,
        global_attn_class: type[nn.Module] | dict = Attention,
        camera_attn_class: type[nn.Module] | dict = Attention,
        init_weights: str | None = "VGGT-1B",
        use_track_head: bool = True,
        use_point_head: bool = True,
        patch_embed: str = "dinov2_vitl14_reg",
        rope_freq: int = 100,
        max_train_len: int | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.ttt_query_images = ttt_query_images
        self.init_weights = init_weights
        self.patch_embed = patch_embed

        global_attn_class = instantiate(global_attn_class) if isinstance(global_attn_class, dict) else global_attn_class
        camera_attn_class = instantiate(camera_attn_class) if isinstance(camera_attn_class, dict) else camera_attn_class

        if max(intermediate_layer_idx) != depth - 1:
            max_intermediate_layer_idx = max(intermediate_layer_idx)
            raise ValueError(
                f"The last intermediate layer to take features (idx {max_intermediate_layer_idx}) from must be the same"
                f" as the depth of the aggregator ({depth})"
            )
        self.intermediate_layer_idx = intermediate_layer_idx

        self.aggregator = Aggregator(
            # HACK: Hard-code image size here to re-use the pre-trained encoder. It will resize the learnable positional
            # encodings to whatever the input image size is.
            img_size=518,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            gradient_checkpoint=gradient_checkpoint,
            ttt_query_images=ttt_query_images,
            global_attn_class=partial(global_attn_class, seq_parallel=True, max_train_len=max_train_len),
            patch_embed=patch_embed,
            rope_freq=rope_freq,
        )
        self.camera_head = CameraHead(
            dim_in=2 * embed_dim,
            ttt_query_images=ttt_query_images,
            attn_class=partial(camera_attn_class, seq_parallel=True),
            gradient_checkpoint=False,
        )

        self.point_head: DPTHead | None = None
        if use_point_head:
            self.point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                use_gradient_checkpointing=gradient_checkpoint,
            )

        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
            use_gradient_checkpointing=gradient_checkpoint,
        )

        self.track_head: TrackHead | None = None
        if use_track_head:
            self.track_head = TrackHead(
                dim_in=2 * embed_dim,
                patch_size=patch_size,
                use_gradient_checkpoint=gradient_checkpoint,
            )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.to(memory_format=torch.channels_last)

    def load_pretrained_weights(self):
        if self.init_weights == "VGGT-1B":
            _logger.info("Loading VGGT-1B weights.")
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            missing_keys, unexpected_keys = self.load_state_dict(torch.hub.load_state_dict_from_url(_URL), strict=False)
            _logger.info(f"Missing keys: {missing_keys}")
            _logger.info(f"Unexpected keys: {unexpected_keys}")
        elif self.patch_embed.startswith("dinov2"):
            model = torch.hub.load("facebookresearch/dinov2", self.patch_embed)
            missing_keys, unexpected_keys = self.aggregator.patch_embed.load_state_dict(
                model.state_dict(), strict=False
            )
            _logger.info(f"Missing keys: {missing_keys}")
            _logger.info(f"Unexpected keys: {unexpected_keys}")

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor | None = None,
        frames_chunk_size: int = 9999,
        **kwargs,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
        """

        # Use queries from sequence parallel rank 0
        sp_group = get_sp_group()
        if sp_group is not None and sp_group.size() > 1:
            if sp_group.rank() != 0:
                query_points = torch.zeros_like(query_points)
            query_points = all_reduce(query_points, op=dist.ReduceOp.SUM, group=sp_group)

        ttt_end = self.get_ttt_end(images)

        aggregated_tokens_list, patch_start_idx, _ = self.aggregator(
            images,
            intermediate_layers_to_return=self.intermediate_layer_idx,
            ttt_end=ttt_end,
        )

        predictions = defaultdict(dict)
        if self.camera_head is not None:
            cam_tokens = aggregated_tokens_list[-1][:, :, 0]
            pose_enc_list = self.camera_head(cam_tokens)
            predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
            predictions["pose_enc_list"] = pose_enc_list

        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=frames_chunk_size,
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=frames_chunk_size,
            )
            predictions["global"]["pts3d"] = pts3d
            predictions["global"]["conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list  # [-1]  # track of the last iteration
            predictions["track_vis"] = vis
            predictions["track_conf"] = conf

        predictions["images"] = images
        return {k: v for k, v in predictions.items()}

    def get_ttt_end(self, images: torch.Tensor):
        """Find the image sequence bounds for TTT training. Considers sequence parallel setups.

        Given sequence parallel group with two workers with images sharded into [0 1 2] | [3 4]
        -> S=3 for first and S=2 for second worker
        Here we determine the start and end indices in the sharded sequence, i.e.,
        -> For first worker: sp_start = 0, sp_end = 3
        -> For second worker: sp_start = 3, sp_end = 5
        -> S_global = 5

        We then determine the end index based on the number of images that should not be considered for TTT updates
        (ttt_query_images).

        If ttt_query_images=1
        -> For first worker: end = 3 (== sp_end)
        -> For second worker: end = 1  (==sp_end - 1)
        """
        B, S, _, H, W = images.shape
        sp_start = 0
        S_global = S

        sp_group = get_sp_group()
        if sp_group and sp_group.size() > 1:
            sp_rank = dist.get_rank(sp_group)
            sp_world_size = dist.get_world_size(sp_group)
            local_len = torch.tensor([S], device=images.device, dtype=torch.long)
            sp_lens = [torch.zeros_like(local_len) for _ in range(sp_world_size)]
            dist.all_gather(sp_lens, local_len, group=sp_group)
            sp_lens = torch.cat([torch.zeros_like(local_len)] + sp_lens)
            sp_starts = torch.cumsum(sp_lens, dim=0)
            sp_start = sp_starts[sp_rank]
            S_global = sp_lens.sum()
        sp_end = sp_start + S
        end = min(S_global - self.ttt_query_images, sp_end) - sp_start
        _logger.debug(
            "TTT image ranges. Total images %s, local images %s, Query images %s, rank %s, sp range (%s, %s), ttt_end %s",
            S_global,
            S,
            self.ttt_query_images,
            sp_group.rank() if sp_group is not None else None,
            sp_start,
            sp_end,
            end,
        )
        return end

    @torch.inference_mode()
    def infer(
        self,
        images: torch.Tensor,
        attn_kwargs: dict[str, Any] = {},
        num_ttt_steps: int | None = 1,
        dtype: torch.dtype | None = None,
        log_ttt_details: bool = False,
        ttt_op_order: list[TTTOperator] | None = None,
        memory_efficient_inference: bool = False,
        use_global_pred: bool = False,
        existing_cam_tokens: torch.Tensor | None = None,
        offload_to_cpu: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Inference forward pass.

        Args:
            images: Input images of shape [#images, 3, H, W] in range [0, 1].

        Returns:
            Dict containing the predicted outputs with the following keys:
                - 'pose':        [#images, 4, 4]  Camera-to-world transformation
                - 'intrinsics':  [#images, 3, 3]  Pinhole camera matrix
                - 'pts3d':       [#images, height, width, 3]  Per-pixel points in world coordinates
                - 'conf':        [#images, height, width]  Per-pixel confidence in range ]1, inf[
                - 'depth':       [#images, height, width, 1]  Per-pixel depth
        """
        N, _, H, W = images.shape

        import os

        world_size = int(os.environ.get("WORLD_SIZE", 1))

        global_num_imgs = N * world_size
        if num_ttt_steps is None and not ttt_op_order:
            num_ttt_steps = 2 + global_num_imgs // 1000
            _logger.info("Using %s TTT steps for %s images", num_ttt_steps, N)

        all_tokens_same_op_order = [
            *[TTTOperator(start=0, end=None, compute_grad=True, update=True, apply=False)] * num_ttt_steps,
            TTTOperator(start=0, end=None, compute_grad=False, update=False, apply=True),
        ]
        # Run the model
        attn_kwargs = attn_kwargs or {
            "info": {
                "ttt_op_order": ttt_op_order or all_tokens_same_op_order,
            },
            # max 10 images at the same time (given patch size 14 and img size 518)
            "chunk_size": 20 * 37 * 37 if memory_efficient_inference else None,
            "track_details": log_ttt_details,
            "offload_to_cpu": offload_to_cpu,
        }


        images = images[None]  # add batch dimension
        dtype = dtype or torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            aggregated_tokens_list, ps_idx, _ = self.aggregator(
                images,
                attn_kwargs=attn_kwargs,
                add_first_view_token=existing_cam_tokens is None,
                memory_efficient_inference=memory_efficient_inference,
            )

            # Predict Cameras
            # Use tokens from the last block for camera prediction.
            cam_tokens = aggregated_tokens_list[-1][:, :, 0]  # [B, S, C]
            if existing_cam_tokens is not None:
                cam_tokens = torch.cat([existing_cam_tokens.to(cam_tokens), cam_tokens], dim=1)

        with torch.autocast("cuda", enabled=False):
            pose_enc = self.camera_head(cam_tokens.clone().cuda(non_blocking=True))[-1]

            if existing_cam_tokens is not None:
                pose_enc = pose_enc[:, existing_cam_tokens.shape[1] :]

            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W))

            # Predict Depth Maps
            minibatch_size = compute_adaptive_minibatch_size(estimated_memory_per_sample_mb=1000)
            depth_map, depth_conf = self.depth_head(
                aggregated_tokens_list, images, ps_idx, frames_chunk_size=minibatch_size
            )

            if use_global_pred and self.point_head is not None:
                # Predict Point Maps
                _logger.info("Predicting point maps with global prediction")
                point_map, point_conf = self.point_head(
                    aggregated_tokens_list, images, ps_idx, frames_chunk_size=minibatch_size
                )
                point_map = point_map.squeeze(0).cpu()
                point_conf = point_conf.squeeze(0).cpu()
            else:
                # Construct 3D Points from Depth Maps and Cameras
                # which usually leads to more accurate 3D points than point map branch
                point_map = unproject_depth_map_to_point_map(
                    depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
                )
                point_map = torch.from_numpy(point_map).to(dtype=torch.float32)
                point_conf = depth_conf.squeeze(0).cpu()

        return {
            "pose": closed_form_inverse_se3(extrinsic.squeeze(0)).cpu(),
            "intrinsics": intrinsic.squeeze(0).cpu(),
            "pts3d": point_map,
            "conf": point_conf,
            "depth": depth_map.squeeze(0).cpu(),
            "cam_tokens": cam_tokens.clone(),
        }

    def device(self):
        return next(self.parameters()).device

    def map(self, inp_views: list[dict[str, torch.Tensor]] | torch.Tensor, **kwargs):
        if isinstance(inp_views, torch.Tensor):
            images = inp_views
        else:
            images = torch.cat([view["img"] for view in inp_views]).to(self.device())

        # Reset state
        _set_state_tracking(self, False)
        self.cam_tokens = None

        # Enable state tracking
        _set_state_tracking(self, True)
        out = self.infer(images=images, **kwargs)
        self.cam_tokens = out.pop("cam_tokens")

        return out

    def query(self, inp_views: list[dict[str, torch.Tensor]] | torch.Tensor, **kwargs):
        if isinstance(inp_views, torch.Tensor):
            images = inp_views
        else:
            images = torch.cat([view["img"] for view in inp_views]).to(self.device())

        # Query with state
        ttt_op_order = [
            TTTOperator(start=0, end=None, compute_grad=False, update=False, apply=True),
        ]
        assert self.cam_tokens is not None, "Camera tokens are not cached"
        return self.infer(images, ttt_op_order=ttt_op_order, existing_cam_tokens=self.cam_tokens, **kwargs)


def _set_state_tracking(model: torch.nn.Module, enable: bool):
    for m in model.modules():
        if isinstance(m, torch.nn.Module) and hasattr(m, "set_state_tracking"):
            m.set_state_tracking(enable)
            if not enable:
                m.reset_state()
