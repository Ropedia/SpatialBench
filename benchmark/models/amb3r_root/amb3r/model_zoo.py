import sys
import os
import torch
import torch.nn as nn
import numpy as np
from .model import AMB3R

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

def load_model(model_name, ckpt_path=None):
    if model_name == 'amb3r':
        model = AMB3R()
        if ckpt_path is not None:
            model.load_weights(ckpt_path)
    
    elif model_name == 'da3':
        model = DA3(ckpt_path=ckpt_path) if ckpt_path is not None else DA3()

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model


class DA3(nn.Module):
    def __init__(self, device='cuda', ckpt_path="depth-anything/DA3NESTED-GIANT-LARGE"):
        super().__init__()

        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.utils.geometry import unproject_depth

        self.model = DepthAnything3.from_pretrained(ckpt_path).to(device).eval()


        self.device = device
        self.name = 'da3'
    

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        def select_by_index(l, idx):
            """Select an element from a list by an index. Supports data batches with different indices.

            Args:
                l (list): List with potentially batched data items.
                idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
                    Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
            """
            if isinstance(idx, int):
                ret = l[idx]
            else:
                indices = idx
                ret = []
                for batch_idx, idx in enumerate(indices):
                    ret.append(l[idx][batch_idx])

                if isinstance(ret[0], np.ndarray):
                    ret = np.stack(ret, 0)
                else:
                    ret = torch.stack(ret, 0)

            return ret

        def exclude_index(l, exclude_idx):
            """Selects all element from a list, excluding a specific index. Supports data batches with different indices.

            Args:
                l (list): List with potentially batched data items.
                idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
                    Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
            """
            if isinstance(exclude_idx, int):
                ret = [ele for idx, ele in enumerate(l) if idx != exclude_idx]
            else:
                exclude_indices = exclude_idx
                ret = []
                for batch_idx, exclude_idx in enumerate(exclude_indices):
                    ret.append([ele[batch_idx] for idx, ele in enumerate(l) if idx != exclude_idx])

                transposed = list(zip(*ret))
                if isinstance(transposed[0][0], np.ndarray):
                    ret = [np.stack(ele, 0) for ele in transposed]
                else:
                    ret = [torch.stack(ele, 0) for ele in transposed]

            return ret
    
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images = [image_key] + images_source
        images = np.stack(images, 1) / 255.0  # Normalize images to [0, 1]
        images = images * 2.0 - 1.0  # Normalize to [-1, 1]

        images = torch.from_numpy(images).float()
        
        # Ensure channel dimension is at position 1: (B, C, H, W) or (B, N, C, H, W)
        if images.shape[-1] == 3:
            images = images.permute(0, 1, 4, 2, 3) if images.dim() == 5 else images.permute(0, 3, 1, 2)


        frames_ = {
            'images': images.to(self.device),
            'keyview_idx': keyview_idx
        }

        frames = {
            'frames': frames_,
        }

        return frames

    
    def _normalize_images(self, imgs):
        """Normalize images from [-1, 1] to ImageNet-normalized range."""
        imgs = (imgs + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 1, 3, 1, 1)
        return (imgs - mean) / std

    def forward(self, frames):
        # input should be (-1, 1)
        imgs = frames['images']  # (B, N, 3, H, W)
        B, nimgs, three, H, W = imgs.shape
        imgs = self._normalize_images(imgs)

        raw_output = self.model.forward(imgs, None, None, [], False, 
                                        use_ray_pose=False, ref_view_strategy='first')

        # Convert raw output to prediction
        prediction = self.model._convert_to_prediction(raw_output)

        # Align prediction to extrinsics
        prediction = self.model._align_to_input_extrinsics_intrinsics(
            None, None, prediction, True
        )

        # Build c2w from w2c (N, 3, 4) -> (N, 4, 4) -> invert
        w2c_34 = torch.from_numpy(prediction.extrinsics).to(imgs.device)  # (N, 3, 4)
        N = w2c_34.shape[0]
        bottom = torch.tensor([0, 0, 0, 1], dtype=w2c_34.dtype, device=imgs.device).view(1, 1, 4).expand(N, -1, -1)
        w2c = torch.cat([w2c_34, bottom], dim=1)  # (N, 4, 4)
        c2w = torch.inverse(w2c)  # (N, 4, 4)

        # Build depth and intrinsics tensors
        intrinsics_t = torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(imgs.device)  # (1, N, 3, 3)
        depth_t = torch.from_numpy(prediction.depth).unsqueeze(0).unsqueeze(-1).to(imgs.device)  # (1, N, H, W, 1)

        # Unproject depth to world points
        from depth_anything_3.utils.geometry import unproject_depth
        world_points = unproject_depth(depth_t, intrinsics_t, c2w.unsqueeze(0))  # (1, N, H, W, 3)
        conf = torch.from_numpy(prediction.conf).unsqueeze(0).to(imgs.device)  # (1, N, H, W)

        return {
            'world_points': world_points.float(),        # (1, N, H, W, 3)
            'world_points_conf': conf.float(),            # (1, N, H, W)
            'depth': depth_t.squeeze(-1).float(),         # (1, N, H, W)
            'pose': c2w.unsqueeze(0).float(),             # (1, N, 4, 4)
        }
    
    
    def output_adapter(self, model_output):
        aux = {}

        # model_output is the dict returned by forward()
        depth = model_output['depth'][:, 0].cpu().numpy()  # (1, H, W)

        pred = {
            'depth': depth[None]  # (1, 1, H, W)
        }

        return pred, aux
    

    def run_amb3r_benchmark(self, frames):
        return self.forward(frames)

    
    @torch.inference_mode()
    def run_amb3r_vo(self, frames, cfg, keyframe_memory):
        return self.forward(frames)


