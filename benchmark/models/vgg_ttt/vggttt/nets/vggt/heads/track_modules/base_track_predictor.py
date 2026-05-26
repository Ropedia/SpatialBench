# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from torch.distributed.nn.functional import broadcast

from vggttt.utils.dist import get_sp_group
from vggttt.utils.optim import checkpoint

from .blocks import CorrBlock, EfficientUpdateFormer
from .modules import Mlp
from .utils import get_2d_embedding, get_2d_sincos_pos_embed, sample_features4d

_logger = logging.getLogger(__name__)


class BaseTrackerPredictor(nn.Module):
    def __init__(
        self,
        stride=1,
        corr_levels=5,
        corr_radius=4,
        latent_dim=128,
        hidden_size=384,
        use_spaceatt=True,
        depth=6,
        predict_conf=True,
        use_gradient_checkpoint=False,
    ):
        super().__init__()
        """
        The base template to create a track predictor

        Modified from https://github.com/facebookresearch/co-tracker/
        and https://github.com/facebookresearch/vggsfm
        """

        self.stride = stride
        self.latent_dim = latent_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.hidden_size = hidden_size
        self.predict_conf = predict_conf

        self.flows_emb_dim = latent_dim // 2

        self.corr_mlp = Mlp(
            in_features=self.corr_levels * (self.corr_radius * 2 + 1) ** 2,
            hidden_features=self.hidden_size,
            out_features=self.latent_dim,
        )

        self.transformer_dim = self.latent_dim + self.latent_dim + self.latent_dim + 4

        self.query_ref_token = nn.Parameter(torch.randn(1, 2, self.transformer_dim))

        space_depth = depth if use_spaceatt else 0
        time_depth = depth

        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.transformer_dim,
            hidden_size=self.hidden_size,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=use_spaceatt,
        )

        self.fmap_norm = nn.LayerNorm(self.latent_dim)
        self.ffeat_norm = nn.GroupNorm(1, self.latent_dim)

        # A linear layer to update track feats at each iteration
        self.ffeat_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())

        self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        if predict_conf:
            self.conf_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        self.use_checkpoint = use_gradient_checkpoint

    def _init_track_feats(self, fmaps: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        B, S, C, _, _ = fmaps.shape

        track_feats = sample_features4d(fmaps[:, 0], query_points).contiguous()

        sp_group = get_sp_group()
        if sp_group is not None:
            # Broadcast the track_feats from the rank 0 of the sequence parallel group to all ranks
            sp_zero_global_rank = dist.get_global_rank(group=sp_group, group_rank=0)
            track_feats = broadcast(track_feats, src=sp_zero_global_rank, group=sp_group)

        # Each rank handles its own sequence length S, no need to sync across ranks
        result = track_feats.unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, C
        return result

    def _forward_iteration(self, coords, track_feats, fcorr_fn, query_points, query_pos_emb, B, N, S):
        """Single iteration of the tracking refinement loop."""
        # Detach the gradients from the last iteration
        coords = coords.detach()

        fcorrs = fcorr_fn.corr_sample(track_feats, coords)

        corr_dim = fcorrs.shape[3]
        fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, corr_dim)
        fcorrs_ = self.corr_mlp(fcorrs_)

        # Movement of current coords relative to query points
        flows = (coords - query_points[:, None]).permute(0, 2, 1, 3).reshape(B * N, S, 2)
        flows_emb = get_2d_embedding(flows, self.flows_emb_dim, cat_coords=False).to(flows)

        # (In my trials, it is also okay to just add the flows_emb instead of concat)
        flows_emb = torch.cat([flows_emb, flows], dim=-1)

        track_feats_ = track_feats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

        # Concatenate them as the input for the transformers
        transformer_input = torch.cat([flows_emb, fcorrs_, track_feats_], dim=2)

        if transformer_input.shape[2] < self.transformer_dim:
            # pad the features to match the dimension
            pad_dim = self.transformer_dim - transformer_input.shape[2]
            pad = torch.zeros_like(flows_emb[..., 0:pad_dim])
            transformer_input = torch.cat([transformer_input, pad], dim=2)

        x = transformer_input + query_pos_emb

        # Add the query ref token to the track feats
        # Query ref_token[[0]] should only be on rank 0
        sp_group = get_sp_group()
        if sp_group is None or sp_group.rank() == 0:
            query_ref_token = torch.cat(
                [self.query_ref_token[:, [0]], self.query_ref_token[:, [1]].expand(-1, S - 1, -1)], dim=1
            )
        else:
            query_ref_token = self.query_ref_token[:, [1]].expand(-1, S, -1)

        x = x + query_ref_token.to(x)

        # B, N, S, C
        x = rearrange(x, "(b n) s d -> b n s d", b=B)

        # Compute the delta coordinates and delta track features
        delta, _ = self.updateformer(x)

        delta = rearrange(delta, "b n s d -> (b n) s d", b=B)
        delta_coords_ = delta[:, :, :2]
        delta_feats_ = delta[:, :, 2:]

        track_feats_ = track_feats_.reshape(B * N * S, self.latent_dim)
        delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)

        # Update the track features
        track_feats_ = self.ffeat_updater(self.ffeat_norm(delta_feats_)) + track_feats_

        track_feats = track_feats_.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)  # BxSxNxC

        # B x S x N x 2
        coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

        # Force coord0 as query for first frame (only if this rank contains frame 0)
        if sp_group is None or sp_group.rank() == 0:
            coords[:, 0] = query_points  # B, N, 2

        return coords, track_feats

    def forward(self, query_points, fmaps, iters=6, return_feat=False, down_ratio=1, apply_sigmoid=True):
        """
        query_points: B x N x 2, the number of batches, tracks, and xy
        fmaps: B x S x C x HH x WW, the number of batches, frames, and feature dimension.
                note HH and WW is the size of feature maps instead of original images
        """
        B, N, D = query_points.shape
        B, S, C, HH, WW = fmaps.shape

        assert D == 2, "Input points must be 2D coordinates"

        # apply a layernorm to fmaps here
        fmaps = self.fmap_norm(fmaps.permute(0, 1, 3, 4, 2))
        fmaps = fmaps.permute(0, 1, 4, 2, 3)

        # Scale the input query_points because we may downsample the images
        # by down_ratio or self.stride
        # e.g., if a 3x1024x1024 image is processed to a 128x256x256 feature map
        # its query_points should be query_points/4
        if down_ratio > 1:
            query_points = query_points / float(down_ratio)

        query_points = query_points / float(self.stride)

        # Initialize positions and features of all tracks as the query points and features at the query points of the
        # first frame respectively.
        coords = query_points.clone().unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, 2
        track_feats = self._init_track_feats(fmaps, query_points)  # B, S, N, C

        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        coord_preds = []

        # Iterative Refinement
        pos_embed = get_2d_sincos_pos_embed(
            self.transformer_dim, grid_size=(HH, WW), device=query_points.device, omega_0=10_000, dtype=fmaps.dtype
        )

        # 2D positional embed
        query_pos_emb = sample_features4d(pos_embed.expand(B, -1, -1, -1), query_points)
        query_pos_emb = rearrange(query_pos_emb, "b n c -> (b n) () c")

        for _ in range(iters):
            coords, track_feats = checkpoint(
                self._forward_iteration,
                coords,
                track_feats,
                fcorr_fn,
                query_points,
                query_pos_emb,
                B,
                N,
                S,
                enabled=self.use_checkpoint and self.training,
            )

            # The predicted tracks are in the original image scale
            if down_ratio > 1:
                coord_pred = coords * self.stride * down_ratio
            else:
                coord_pred = coords * self.stride

            coord_preds.append(coord_pred)

        # B, S, N
        track_feats_flat = track_feats.reshape(B * S * N, self.latent_dim)

        vis_e = self.vis_predictor(track_feats_flat).view(B, S, N)
        if apply_sigmoid:
            vis_e = torch.sigmoid(vis_e)

        if self.predict_conf:
            conf_e = self.conf_predictor(track_feats_flat).view(B, S, N)
            if apply_sigmoid:
                conf_e = torch.sigmoid(conf_e)
        else:
            conf_e = None

        if return_feat:
            return coord_preds, vis_e, track_feats, conf_e
        else:
            return coord_preds, vis_e, conf_e
