"""
src/metrics/edge_entropy.py
---------------------------
No-reference metrics:
  - Shannon entropy: ảnh càng "giàu thông tin" thì entropy càng cao.
  - Edge visibility: số lượng cạnh thấy được, dựa trên gradient
    (cải biên từ Hautiere 2008 - "Blind contrast enhancement
    assessment by gradient ratioing at visible edges").
"""
from __future__ import annotations

import numpy as np
import cv2
from skimage.measure import shannon_entropy


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def compute_entropy(img: np.ndarray) -> float:
    """Shannon entropy của ảnh grayscale (bit). Cao hơn = giàu thông tin hơn."""
    gray = _to_gray_u8(img)
    return float(shannon_entropy(gray))


def compute_edge_visibility(
    img: np.ndarray, threshold: int = 20
) -> float:
    """Tỷ lệ pixel có gradient magnitude > threshold.

    Đơn giản hoá của Hautiere e-metric: đếm số "visible edge" trong ảnh.
    Giá trị càng cao thì càng nhiều chi tiết/cạnh.
    """
    gray = _to_gray_u8(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > threshold).mean())