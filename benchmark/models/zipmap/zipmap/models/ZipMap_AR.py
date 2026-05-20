
import collections
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin 

from zipmap.models.aggregator_ttt import Aggregator
from zipmap.heads.camera_head import CameraHead, CameraHead_MLP
from zipmap.heads.dpt_head_vggt_legacy import DPTHead
import random

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])


class ZipMap(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False,
                 enable_camera_mlp=True,
                 enable_local_point=True,
                 enable_depth=True,
                 enable_nvs=False,
                 ttt_config=None,
                 nvs_config=None,
                 other_config=None,
                 ):
        super().__init__()
        self.ttt_config = ttt_config
        self.other_config = other_config
        self.nvs_config = nvs_config
        if self.other_config is None:
            self.other_config = {}

        affine_invariant = self.other_config.get("affine_invariant", False)
        mixed_image_tokenization = self.other_config.get("mixed_image_tokenization", False)

        other_config_for_aggregator = {
            "affine_invariant": affine_invariant,
            "mixed_image_tokenization": mixed_image_tokenization,
        }

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, ttt_config=ttt_config, other_config=other_config_for_aggregator, nvs_config=nvs_config)

        if enable_camera:
            self.camera_head = CameraHead(dim_in=2 * embed_dim)
        else:
            self.camera_head = None

        if enable_camera_mlp:
            self.camera_mlp_head = CameraHead_MLP(dim_in=2 * embed_dim)
        else:
            self.camera_mlp_head = None

        if enable_local_point:
            use_gradient_checkpointing_local_point = self.other_config.get("use_gradient_checkpointing_local_point", False)
            self.local_point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=3,
                activation="xy_exp",
                conf_activation="none", # no confidence for local points
                use_gradient_checkpointing=use_gradient_checkpointing_local_point,
                 # ! vggt set this to True but you should manually check set it to be False otherwise dpt residual block will have bugs
                use_inplace=False
            )
        else:
            self.local_point_head = None
        if enable_depth:
            use_gradient_checkpointing_depth = self.other_config.get("use_gradient_checkpointing_depth", False)
            self.depth_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
                use_gradient_checkpointing=use_gradient_checkpointing_depth,
                # ! vggt set this to True but you should manually check set it to be False otherwise dpt residual block will have bugs
                use_inplace=False
            )
        else:
            self.depth_head = None


        # design for training
        self.random_window_size_rgn = random.Random()
        self.random_window_size_rgn.seed(0)
        self.random_window_size_list = [1, 2, 4, 6, 8, 12, 24]

    def forward(self, images: torch.Tensor, query_info: torch.Tensor = None, store_state: bool = False, window_size: int = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_info (torch.Tensor, optional): Query information for NVS,
                Default: None
            store_state (bool, optional): Whether to store the TTT state for future queries.
            window_size (int, optional): Window size (frame number) for TTT operations. Default is None.

        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        info = {
            # "ttt_op_order": ttt_op_order, # define later in forward
            "store_state": store_state, # if to store the state_list for future queries
        }

        if window_size is not None:
            info["window_size"] = window_size
        elif "window_size" in self.ttt_config:
            if self.ttt_config["window_size"] == "random":
                rand_idx = self.random_window_size_rgn.randint(0, len(self.random_window_size_list) - 1)
                info["window_size"] = self.random_window_size_list[rand_idx]
            else:
                info["window_size"] = self.ttt_config["window_size"]
        
        # print cur rank and window size
        if torch.distributed.is_initialized():
            cur_rank = torch.distributed.get_rank()
            print(f"[Rank {cur_rank}] with window size: {info['window_size']}")
        aggregated_tokens_list, patch_start_idx, state_list = self.aggregator(images, target_query_conditions=None, info=info)
        
        input_view_num = images.shape[1]
        input_img_aggregated_tokens_list = [tokens[:, :input_view_num, :] for tokens in aggregated_tokens_list]

        predictions = {}
        with torch.amp.autocast(device_type='cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(input_img_aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list # var type: a list of predictions["pose_enc"] of each iteration
            
            if self.camera_mlp_head is not None:
                # Extract camera tokens (index 0) from [B, S, P, 2C] -> [B, S, 2C]
                camera_tokens = input_img_aggregated_tokens_list[-1][:, :, 0]
                pose_enc_mlp_list = [self.camera_mlp_head(camera_tokens)]
                predictions["pose_enc"] = pose_enc_mlp_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_mlp_list"] = pose_enc_mlp_list # var type: a list of predictions["pose_enc"] of each iteration
            
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    input_img_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
    
            if self.local_point_head is not None:
                pts3d, pts3d_conf = self.local_point_head(
                    input_img_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["local_points"] = pts3d
                predictions["local_points_conf"] = pts3d_conf


            predictions["images"] = images  # store the images for visualization during inference

        predictions["state_list"] = state_list # cache the TTT state_list for future queries
        


        return predictions


    
