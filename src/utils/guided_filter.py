from __future__ import annotations

import cv2
import numpy as np


def _box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    ksize = 2 * radius + 1
    return cv2.boxFilter(
        img, ddepth=-1, ksize=(ksize, ksize), borderType=cv2.BORDER_REFLECT
    )


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 60,
    eps: float = 1e-3,
) -> np.ndarray:
    if guide.ndim == 3:
        guide_u8 = (
            guide if guide.dtype == np.uint8
            else np.clip(guide * 255.0, 0, 255).astype(np.uint8)
        )
        guide_gray = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY).astype(
            np.float32
        ) * (1.0 / 255.0)
    else:
        guide_gray = guide.astype(np.float32, copy=False)

    src_f = src.astype(np.float32, copy=False)

    mean_I = _box_filter(guide_gray, radius)
    mean_p = _box_filter(src_f, radius)
    corr_Ip = _box_filter(guide_gray * src_f, radius)
    corr_II = _box_filter(guide_gray * guide_gray, radius)

    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    return _box_filter(a, radius) * guide_gray + _box_filter(b, radius)
