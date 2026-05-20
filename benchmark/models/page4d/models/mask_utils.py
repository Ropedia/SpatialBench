import torch
from PIL import Image
import math
import matplotlib

def normalize_feature_batch(feature):
    return (feature - feature.min(dim=1, keepdim=True)[0]) / (feature.max(dim=1, keepdim=True)[0] - feature.min(dim=1, keepdim=True)[0])

def otsu_threshold_batch(img: torch.Tensor) -> torch.Tensor:
    """
    Batched Otsu's thresholding without Python loops.
    Args: img: (B, *) tensor, any range.
    Returns: mask: (B, *) bool tensor
    """
    B, _ = img.shape
    img = img.detach()
    # 1) Normalize each image to [0, 255]
    img_min = img.view(B, -1).min(dim=1)[0].view(B, 1)
    img_max = img.view(B, -1).max(dim=1)[0].view(B, 1)
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)
    img_uint8 = (img_norm * 255).to(torch.uint8)  # (B, H, W)
    # 2) Flatten
    flat = img_uint8.view(B, -1)  # (B, N)
    N = flat.size(1)
    # 3) Compute batched histograms using bincount + offset
    offsets = torch.arange(B, device=img.device) * 256  # batch offset
    flat_offsets = flat + offsets[:, None]  # (B, N)
    hist_all = torch.bincount(flat_offsets.view(-1), minlength=B * 256)
    hist = hist_all.view(B, 256).float()  # (B, 256)
    # 4) Probabilities
    prob = hist / N  # (B, 256)
    # 5) Cumulative sums
    omega = torch.cumsum(prob, dim=1)  # (B, 256)
    mu = torch.cumsum(prob * torch.arange(256, device=img.device), dim=1)  # (B, 256)
    mu_t = mu[:, -1].unsqueeze(1)  # (B, 1)
    # 6) Between-class variance
    numerator = (mu_t * omega - mu) ** 2
    denominator = omega * (1 - omega)
    variance = torch.zeros_like(numerator)
    valid = denominator > 0
    variance[valid] = numerator[valid] / denominator[valid]
    # 7) Best thresholds for each batch
    best_thresh = variance.argmax(dim=1)  # (B,)
    # 8) Apply thresholds (vectorized comparison)
    thresholds = best_thresh.view(B, 1)
    mask = img_uint8 > thresholds  # (B, H, W)
    mask = (1 - mask.float()) * (- math.log(2.0)) 
    return mask

def process_mask(img: torch.Tensor, B, S, P, patch_start_idx, H, W) -> torch.Tensor:
    """
    Batched Otsu's thresholding without Python loops.
    Args:
        img: (B,S,P,C) tensor, any range.
    Returns:
        mask: (B,S,P*P) bool tensor
    """
    img = img.mean(dim=-1)
    img = img.view(B*S, P)[:, patch_start_idx:]
    img = normalize_feature_batch(img)
    mask = otsu_threshold_batch(img)

    input_mask = mask.view(B, S, (P-patch_start_idx))
    padding_mask = torch.zeros(B, S, patch_start_idx).to(img.device)
    input_mask = torch.cat([padding_mask, input_mask], dim=-1).view(B, S*P)
    
    whole_mask = torch.zeros(B, P*S, P*S).to(img.device)
    whole_mask[:, 0, :] = input_mask

    Visual = False
    if Visual:
        _image_ = normalize_feature_batch(img)
        _image_ = confidence_to_rgb(_image_[0].view(H,W), cmap_name='YlOrRd')
        Dynamic_mask_img_ = Image.fromarray((_image_.cpu().view(H,W,3).clamp(0, 1) * 255).byte().numpy()).resize((400,400), resample=Image.BILINEAR)  
        Dynamic_mask_img_.save("_global_feature0.png")

        Dynamic_mask_img_ = Image.fromarray(((mask[0]+1).cpu().view(H,W).clamp(0, 1) * 255).byte().numpy()).resize((400,400), resample=Image.BILINEAR)  
        Dynamic_mask_img_.save("_global_feature_mask0.png")
    return whole_mask

def confidence_to_rgb(confidence_map: torch.Tensor, cmap_name='Reds') -> torch.Tensor:
    """
    Converts a 2D confidence map (H, W) to an RGB image (H, W, 3).
    Args:
        confidence_map: torch.Tensor of shape (H, W), values in [0, 1] or [min, max]
        cmap_name: str, e.g. 'Reds', 'plasma', 'hot', 'viridis', etc.
    Returns: torch.Tensor of shape (H, W, 3) with RGB values in [0, 1]
    """
    # Normalize confidence to [0, 1]
    conf = confidence_map.clone()
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    # Convert to numpy and apply colormap
    conf_np = conf.cpu().numpy()
    colormap = matplotlib.colormaps.get_cmap(cmap_name)
    colored = colormap(conf_np)  # (H, W, 4), includes alpha
    # Remove alpha and convert to torch
    rgb = torch.from_numpy(colored[:, :, :3]).float()
    return rgb
