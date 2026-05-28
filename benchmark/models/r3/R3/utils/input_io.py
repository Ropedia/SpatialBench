"""Image/video input helpers for R3 inference demos.

The image resizing/cropping helper is adapted from DUSt3R image utilities:
Meta / DUSt3R utility code under its upstream repository license, and Naver
DUSt3R image utilities under CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import glob
import os
import tempfile
from copy import deepcopy

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORT_ENABLED = True
except ImportError:
    HEIF_SUPPORT_ENABLED = False


IMG_NORM = tvf.Compose(
    [tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def parse_seq_path(path: str):
    """Return sorted image paths for a directory, or extract video frames to a temp dir.

    Returns:
        tuple[list[str], str | None]: image paths and a temporary directory to remove
        after loading. The temp directory is only used for video inputs.
    """
    if os.path.isdir(path):
        img_paths = sorted(glob.glob(os.path.join(path, "*")))
        return img_paths, None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps == 0:
        cap.release()
        raise ValueError(f"Error: Video FPS is 0 for {path}")

    frame_indices = list(range(total_frames))
    print(f" - Video FPS: {video_fps}, Frame Interval: 1, Total Frames to Read: {len(frame_indices)}")

    tmpdirname = tempfile.mkdtemp()
    img_paths = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = os.path.join(tmpdirname, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        img_paths.append(frame_path)

    cap.release()
    return img_paths, tmpdirname


def _resize_pil_image(img, long_edge_size):
    scale = long_edge_size / max(img.size)
    new_size = tuple(int(round(x * scale)) for x in img.size)
    interp = PIL.Image.LANCZOS if scale < 1 else PIL.Image.BICUBIC
    return img.resize(new_size, interp)


def load_images(
    folder_or_list,
    size,
    square_ok=False,
    verbose=True,
    rotate_clockwise_90=False,
    crop_to_landscape=False,
    patch_size=16,
):
    """Open images and convert them to R3 view input dictionaries."""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list
    else:
        raise ValueError(f"bad folder_or_list={folder_or_list!r}")

    supported_ext = [".jpg", ".jpeg", ".png"]
    if HEIF_SUPPORT_ENABLED:
        supported_ext += [".heic", ".heif"]
    supported_ext = tuple(supported_ext)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_ext):
            continue

        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")

        if rotate_clockwise_90:
            img = img.rotate(-90, expand=True)

        if crop_to_landscape:
            desired_aspect = 4 / 3
            width, height = img.size
            current_aspect = width / height
            if current_aspect > desired_aspect:
                new_width = int(height * desired_aspect)
                left = (width - new_width) // 2
                img = img.crop((left, 0, left + new_width, height))
            else:
                new_height = int(width / desired_aspect)
                top = (height - new_height) // 2
                img = img.crop((0, top, width, top + new_height))

        w1, h1 = img.size
        if size == 224:
            img = _resize_pil_image(img, round(size * max(w1 / h1, h1 / w1)))
        else:
            img = _resize_pil_image(img, size)

        w, h = img.size
        cx, cy = w // 2, h // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw = ((2 * cx) // patch_size) * patch_size // 2
            halfh = ((2 * cy) // patch_size) * patch_size // 2
            if not square_ok and w == h:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        w2, h2 = img.size
        if verbose and len(imgs) == 0:
            print(f" - first image {path} with resolution {w1}x{h1} --> {w2}x{h2}")

        imgs.append(
            {
                "img": IMG_NORM(img)[None],
                "true_shape": np.int32([img.size[::-1]]),
                "idx": len(imgs),
                "instance": str(len(imgs)),
            }
        )

    assert imgs, "no images found at " + root
    if verbose:
        shape = imgs[0]["true_shape"][0]
        print(f" (Found {len(imgs)} images, resized to {shape[1]}x{shape[0]})")
    return imgs


def prepare_image_views(img_paths, size: int, revisit: int = 1, update: bool = True):
    """Load images and convert them to R3 view dictionaries."""
    images = load_images(img_paths, size=size, patch_size=14)
    views = []
    for idx, image in enumerate(images):
        img = image["img"]
        view = {
            "img": img,
            "ray_map": torch.full(
                (img.shape[0], 6, img.shape[-2], img.shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(image["true_shape"]),
            "idx": idx,
            "instance": str(idx),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)

    if revisit > 1:
        revisited = []
        for r in range(revisit):
            for idx, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + idx
                new_view["instance"] = str(r * len(views) + idx)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                revisited.append(new_view)
        return revisited

    return views
