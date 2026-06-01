"""Image preprocessing helpers shared by dataset, training, and the viewers.

Raw cam1/cam2 PNGs saved by ``camera.DualCameraRecorder`` are not pre-cropped
or rotated. Downstream consumers apply::

    rotate(image, ScrewConfig.rotateN degrees CW)  →  crop(image, ScrewConfig.roiN)

The helpers below accept either a PIL ``Image`` or a numpy ndarray and return
the same type to keep the calling site simple.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from PIL import Image


ImageLike = Union[Image.Image, np.ndarray]

# Clockwise-degrees → PIL Transpose constant.
# PIL .transpose(ROTATE_*) is COUNTER-clockwise, so the mapping below uses the
# CCW value that is the complement of the requested CW rotation.
_PIL_CW_TO_TRANSPOSE = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def rotate_image(img: ImageLike, degrees: int) -> ImageLike:
    """Rotate a PIL Image or HxWxC ndarray clockwise by 0/90/180/270 degrees.

    Lossless (just swaps/flips axes). Returns the same type as the input.
    """
    if degrees == 0:
        return img
    if degrees not in _PIL_CW_TO_TRANSPOSE:
        raise ValueError(
            f"degrees must be 0/90/180/270 (clockwise), got {degrees}"
        )
    if isinstance(img, Image.Image):
        return img.transpose(_PIL_CW_TO_TRANSPOSE[degrees])
    arr = np.asarray(img)
    # np.rot90 is CCW; rotating CW by N*90 == rotating CCW by -N*90
    return np.rot90(arr, k=-(degrees // 90))


def apply_roi(img: ImageLike, roi: Tuple[int, int, int, int]) -> ImageLike:
    """Crop to ``roi = (left, upper, right, lower)`` (PIL .crop convention)."""
    left, upper, right, lower = (int(v) for v in roi)
    if isinstance(img, Image.Image):
        return img.crop((left, upper, right, lower))
    arr = np.asarray(img)
    return arr[upper:lower, left:right]


def apply_rotation_and_roi(
    img: ImageLike,
    degrees: int,
    roi: Tuple[int, int, int, int],
) -> ImageLike:
    """Rotate (CW) then crop to ROI. ROI coords live in the ROTATED frame."""
    return apply_roi(rotate_image(img, degrees), roi)
