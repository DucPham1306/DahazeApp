
from __future__ import annotations

import cv2
import numpy as np

from ..utils.guided_filter import guided_filter



THETA0 = 0.121779
THETA1 = 0.959710
THETA2 = -0.780245
SIGMA = 0.041337


def compute_depth_map(img: np.ndarray, min_filter_size: int = 15) -> np.ndarray:

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    S = hsv[..., 1]
    V = hsv[..., 2]

    d = THETA0 + THETA1 * V + THETA2 * S

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (min_filter_size, min_filter_size)
    )
    d = cv2.erode(d, kernel)
    return d


def estimate_atmospheric_light_cap(
    img: np.ndarray, depth: np.ndarray, top_percent: float = 0.001
) -> np.ndarray:

    h, w = depth.shape
    n_top = max(int(h * w * top_percent), 1)
    flat_depth = depth.ravel()
    flat_img = img.reshape(-1, 3)
    idx = np.argpartition(flat_depth, -n_top)[-n_top:]
    brightest = flat_img[idx]
    A = brightest[np.argmax(brightest.sum(axis=1))]
    return A.astype(np.float32)


def dehaze_cap(
    img: np.ndarray,
    beta: float = 1.0,
    min_filter_size: int = 15,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
    t0: float = 0.1,
) -> np.ndarray:

    assert img.ndim == 3 and img.shape[2] == 3
    if img.dtype not in (np.float32, np.float64):
        img = img.astype(np.float32) / 255.0

    depth = compute_depth_map(img, min_filter_size)
    depth_refined = guided_filter(img, depth, guided_radius, guided_eps)

    A = estimate_atmospheric_light_cap(img, depth_refined)

    t = np.exp(-beta * depth_refined)
    t = np.clip(t, t0, 1.0)
    t3 = np.repeat(t[..., np.newaxis], 3, axis=2)

    J = (img - A) / t3 + A
    return np.clip(J, 0, 1)