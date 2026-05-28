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

import torch
import torch.nn as nn


class CameraDec(nn.Module):
    def __init__(
        self,
        dim_in=1536,
        attention_mode=None,
        attention_window_size=8,
        use_scale_head=False,
        separate_rel_pose_confidence=False,
    ):
        super().__init__()
        # Kept as no-op compatibility knobs for configs that switch from
        # CameraDecRel to CameraDec while inheriting relative-head settings.
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())

    def forward(self, feat, camera_encoding=None, *args, **kwargs):
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)
        out_t = self.fc_t(feat).reshape(B, N, 3)
        if camera_encoding is None:
            out_qvec = self.fc_qvec(feat).reshape(B, N, 4)
            out_fov = self.fc_fov(feat).reshape(B, N, 2)
        else:
            out_qvec = camera_encoding[..., 3:7]
            out_fov = camera_encoding[..., -2:]
        pose_enc = torch.cat([out_t, out_qvec, out_fov], dim=-1)
        return pose_enc


class CameraDecRel(CameraDec):
    """Relative-camera decoder.

    This head predicts:
    - absolute pose encoding for each camera token (same as CameraDec), and
    - pairwise relative pose deltas (t, qvec) + confidence for token pairs (i, j).

    FoV is predicted only once per current frame j from the absolute branch,
    then broadcast to all (i, j) pairs.
    """

    def __init__(
        self,
        dim_in=1536,
        attention_mode=None,
        attention_window_size=8,
        use_scale_head=False,
        separate_rel_pose_confidence=False,
    ):
        super().__init__(dim_in=dim_in)
        output_dim = dim_in
        self.rel_backbone = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_rel_t = nn.Linear(output_dim, 3)
        self.fc_rel_qvec = nn.Linear(output_dim, 4)
        self.fc_rel_conf = nn.Linear(output_dim, 1)
        self.separate_rel_pose_confidence = bool(separate_rel_pose_confidence)
        if self.separate_rel_pose_confidence:
            self.fc_rel_conf_t = nn.Linear(output_dim, 1)
        self.fc_abs_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())
        self.is_relative_head = True
        self.attention_mode = attention_mode
        self.attention_window_size = attention_window_size

        # Optional scale head: predicts metric scale ratio from reference token.
        self.use_scale_head = use_scale_head
        if use_scale_head:
            self.fc_scale = nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.ReLU(),
                nn.Linear(output_dim // 4, 1),
            )

    @staticmethod
    def _aggregate_rel_confidence_logits(rel_conf_t: torch.Tensor, rel_conf_r: torch.Tensor) -> torch.Tensor:
        """Build the compatibility scalar confidence from split logits."""
        return 0.5 * (rel_conf_t + rel_conf_r)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Map legacy shared-confidence weights into split heads when needed."""
        if self.separate_rel_pose_confidence:
            old_weight_key = prefix + "fc_rel_conf.weight"
            old_bias_key = prefix + "fc_rel_conf.bias"
            new_weight_key = prefix + "fc_rel_conf_t.weight"
            new_bias_key = prefix + "fc_rel_conf_t.bias"

            if new_weight_key not in state_dict and old_weight_key in state_dict:
                state_dict[new_weight_key] = state_dict[old_weight_key].clone()
            if new_bias_key not in state_dict and old_bias_key in state_dict:
                state_dict[new_bias_key] = state_dict[old_bias_key].clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _build_rel_pair_mask(
        num_views: int,
        device: torch.device,
        causal: bool = False,
        attention_mode: str = "causal",
        attention_window_size: int = 8,
    ) -> torch.Tensor:
        if not causal:
            return ~torch.eye(num_views, device=device, dtype=torch.bool)

        rel_mask = torch.zeros((num_views, num_views), device=device, dtype=torch.bool)
        window_size = max(int(attention_window_size), 1)

        if attention_mode == "causal":
            return torch.triu(
                torch.ones((num_views, num_views), device=device, dtype=torch.bool),
                diagonal=1,
            )

        if attention_mode == "window":
            for j in range(1, num_views):
                start_view = max(1, j - window_size + 1)
                rel_mask[start_view:j, j] = True
                rel_mask[0, j] = True
            return rel_mask

        if attention_mode == "window_wo_sink":
            for j in range(1, num_views):
                start_view = max(0, j - window_size + 1)
                rel_mask[start_view:j, j] = True
            return rel_mask

        if attention_mode == "full":
            return ~torch.eye(num_views, device=device, dtype=torch.bool)

        raise ValueError(f"Unknown attention_mode: {attention_mode}")

    def forward(
        self,
        feat,
        camera_encoding=None,
        causal: bool = False,
        attention_mode: str = "causal",
        attention_window_size: int = 8,
        memory_feat=None,
        memory_feat_projected: bool = False,
        *args,
        **kwargs,
    ):
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)
        out_t = self.fc_t(feat).reshape(B, N, 3)
        if camera_encoding is None:
            out_qvec = self.fc_qvec(feat).reshape(B, N, 4)
            out_fov = self.fc_fov(feat).reshape(B, N, 2)
        else:
            out_qvec = camera_encoding[..., 3:7]
            out_fov = camera_encoding[..., -2:]
        pose_enc = torch.cat([out_t, out_qvec, out_fov], dim=-1)

        feat_abs = feat.reshape(B, N, -1)

        # Predict metric scale ratio from reference token (frame 0).
        scale_pred = None
        if self.use_scale_head:
            raw = self.fc_scale(feat_abs[:, 0])  # (B, 1)
            scale_pred = torch.exp(raw.squeeze(-1).clamp(-10, 10))  # (B,)

        if memory_feat is not None:
            M = memory_feat.shape[1]
            if memory_feat_projected:
                memory_feat = memory_feat.reshape(B, M, -1)
            else:
                memory_feat = memory_feat.reshape(B * M, -1)
                memory_feat = self.backbone(memory_feat).reshape(B, M, -1)

            rel_t = feat_abs.new_zeros((B, M + N, M + N, 3), dtype=torch.float32)
            rel_qvec = feat_abs.new_zeros((B, M + N, M + N, 4), dtype=torch.float32)
            rel_conf = feat_abs.new_zeros((B, M + N, M + N), dtype=torch.float32)
            rel_conf_t = None
            rel_conf_r = None
            if self.separate_rel_pose_confidence:
                rel_conf_t = feat_abs.new_zeros((B, M + N, M + N), dtype=torch.float32)
                rel_conf_r = feat_abs.new_zeros((B, M + N, M + N), dtype=torch.float32)
            rel_fov = feat_abs.new_zeros((B, M + N, M + N, 2), dtype=torch.float32)
            rel_mask = torch.zeros((B, M + N, M + N), device=feat_abs.device, dtype=torch.bool)

            curr_idx = torch.arange(N, device=feat_abs.device) + M
            rel_fov[:, curr_idx, curr_idx, :] = out_fov

            if M > 0 and N > 0:
                feat_i = memory_feat[:, :, None, :].expand(-1, -1, N, -1)
                feat_j = feat_abs[:, None, :, :].expand(-1, M, -1, -1)
                pair_feat = torch.cat([feat_i, feat_j], dim=-1).reshape(B * M * N, -1)

                rel_feat = self.rel_backbone(pair_feat)
                rel_t_vals = self.fc_rel_t(rel_feat).reshape(B, M, N, 3)
                rel_qvec_vals = self.fc_rel_qvec(rel_feat).reshape(B, M, N, 4)
                if self.separate_rel_pose_confidence:
                    rel_conf_t_vals = self.fc_rel_conf_t(rel_feat).reshape(B, M, N)
                    rel_conf_r_vals = self.fc_rel_conf(rel_feat).reshape(B, M, N)
                    rel_conf_vals = self._aggregate_rel_confidence_logits(rel_conf_t_vals, rel_conf_r_vals)
                else:
                    rel_conf_vals = self.fc_rel_conf(rel_feat).reshape(B, M, N)
                rel_fov_vals = out_fov[:, None, :, :].expand(-1, M, -1, -1)

                rel_t[:, :M, M:, :] = rel_t_vals
                rel_qvec[:, :M, M:, :] = rel_qvec_vals
                rel_conf[:, :M, M:] = rel_conf_vals
                if self.separate_rel_pose_confidence:
                    rel_conf_t[:, :M, M:] = rel_conf_t_vals
                    rel_conf_r[:, :M, M:] = rel_conf_r_vals
                rel_fov[:, :M, M:, :] = rel_fov_vals
                rel_mask[:, :M, M:] = True

            rel_pose_enc = torch.cat([rel_t, rel_qvec, rel_fov], dim=-1)

            output = {
                "abs_pose_enc": pose_enc,
                "rel_pose_enc": rel_pose_enc,
                "rel_pose_conf": rel_conf,
                "rel_pose_mask": rel_mask,
                "rel_pose_memory_size": M,
                "rel_pose_projected_feat": feat_abs,
                "scale_pred": scale_pred,
            }
            if self.separate_rel_pose_confidence:
                output["rel_pose_conf_t"] = rel_conf_t
                output["rel_pose_conf_r"] = rel_conf_r

            return output

        attention_mode = attention_mode if self.attention_mode is None else self.attention_mode

        pair_mask = self._build_rel_pair_mask(
            num_views=N,
            device=feat_abs.device,
            causal=causal,
            attention_mode=attention_mode,
            attention_window_size=attention_window_size,
        )
        pair_idx = pair_mask.nonzero(as_tuple=False).T

        i_idx, j_idx = pair_idx[0], pair_idx[1]
        num_pairs = i_idx.numel()

        rel_t = feat_abs.new_zeros((B, N, N, 3), dtype=torch.float32)
        rel_qvec = feat_abs.new_zeros((B, N, N, 4), dtype=torch.float32)
        rel_conf = feat_abs.new_zeros((B, N, N), dtype=torch.float32)
        rel_conf_t = None
        rel_conf_r = None
        if self.separate_rel_pose_confidence:
            rel_conf_t = feat_abs.new_zeros((B, N, N), dtype=torch.float32)
            rel_conf_r = feat_abs.new_zeros((B, N, N), dtype=torch.float32)
        rel_fov = feat_abs.new_zeros((B, N, N, 2), dtype=torch.float32)
        diag_idx = torch.arange(N, device=feat_abs.device)
        rel_fov[:, diag_idx, diag_idx, :] = out_fov.to(torch.float32)

        if num_pairs > 0:
            feat_i = feat_abs[:, i_idx, :]
            feat_j = feat_abs[:, j_idx, :]
            pair_feat = torch.cat([feat_i, feat_j], dim=-1).reshape(B * num_pairs, -1)

            rel_feat = self.rel_backbone(pair_feat)
            rel_t_vals = self.fc_rel_t(rel_feat).reshape(B, num_pairs, 3).to(torch.float32)
            rel_qvec_vals = self.fc_rel_qvec(rel_feat).reshape(B, num_pairs, 4).to(torch.float32)
            if self.separate_rel_pose_confidence:
                rel_conf_t_vals = self.fc_rel_conf_t(rel_feat).reshape(B, num_pairs).to(torch.float32)
                rel_conf_r_vals = self.fc_rel_conf(rel_feat).reshape(B, num_pairs).to(torch.float32)
                rel_conf_vals = self._aggregate_rel_confidence_logits(rel_conf_t_vals, rel_conf_r_vals)
            else:
                rel_conf_vals = self.fc_rel_conf(rel_feat).reshape(B, num_pairs).to(torch.float32)
            # rel_fov_vals = self.fc_abs_fov(rel_feat).reshape(B, num_pairs, 2)
            # Broadcast the FoV from the current frame j to all pairs (i, j), use out_fov from the absolute branch for better stability
            rel_fov_vals = out_fov[:, j_idx, :].reshape(B, num_pairs, 2).to(torch.float32)

            rel_t[:, i_idx, j_idx, :] = rel_t_vals
            rel_qvec[:, i_idx, j_idx, :] = rel_qvec_vals
            rel_conf[:, i_idx, j_idx] = rel_conf_vals
            if self.separate_rel_pose_confidence:
                rel_conf_t[:, i_idx, j_idx] = rel_conf_t_vals
                rel_conf_r[:, i_idx, j_idx] = rel_conf_r_vals
            rel_fov[:, i_idx, j_idx, :] = rel_fov_vals

        rel_pose_enc = torch.cat([rel_t, rel_qvec, rel_fov], dim=-1)

        rel_mask = pair_mask
        rel_mask = rel_mask.unsqueeze(0).expand(B, -1, -1)

        output = {
            "abs_pose_enc": pose_enc,
            "rel_pose_enc": rel_pose_enc,
            "rel_pose_conf": rel_conf,
            "rel_pose_mask": rel_mask,
            "rel_pose_projected_feat": feat_abs,
            "scale_pred": scale_pred,
        }
        if self.separate_rel_pose_confidence:
            output["rel_pose_conf_t"] = rel_conf_t
            output["rel_pose_conf_r"] = rel_conf_r

        return output
