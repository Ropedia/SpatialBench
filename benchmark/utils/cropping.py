"""
Image/depth map cropping and resizing utilities.
Extracted from dataloader/utils/cropping.py, with inlined colmap/opencv intrinsics conversion,
removing the dependency on src.utils.geometry.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import PIL.Image

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


# ---- Intrinsics coordinate system conversion (originally from src.utils.geometry) ----

def colmap_to_opencv_intrinsics(K):
    """COLMAP intrinsics -> OpenCV intrinsics (principal point offset -0.5)"""
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """OpenCV intrinsics -> COLMAP intrinsics (principal point offset +0.5)"""
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


# ---- ImageList utility class ----

class ImageList:
    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


# ---- Crop/resize functions ----

def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    """Jointly resize image + depthmap so that (W, H) >= output_resolution."""
    image = ImageList(image)
    input_resolution = np.array(image.size)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """Crop image + depthmap and adjust intrinsics."""
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    depthmap = depthmap[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    """Compute the intrinsics matrix after cropping/resizing."""
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    """Compute the crop bbox from input/output intrinsics."""
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l + out_width, t + out_height)
    return crop_bbox
