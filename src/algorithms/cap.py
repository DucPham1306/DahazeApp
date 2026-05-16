from __future__ import annotations

import cv2
import numpy as np

from ..utils.guided_filter import guided_filter

THETA0 = 0.121779
THETA1 = 0.959710
THETA2 = -0.780245
SIGMA = 0.041337


def compute_depth_map(img: np.ndarray, min_filter_size: int = 15) -> np.ndarray:
    if img.dtype == np.uint8:
        img_u8 = img
    else:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.float32) * (1.0 / 255.0)
    V = hsv[..., 2].astype(np.float32) * (1.0 / 255.0)
    d = THETA0 + THETA1 * V + THETA2 * S
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (min_filter_size, min_filter_size)
    )
    return cv2.erode(d, kernel)


def estimate_atmospheric_light_cap(
    img: np.ndarray, depth: np.ndarray, top_percent: float = 0.001
) -> np.ndarray:
    n_pixels = depth.size
    n_top = max(int(n_pixels * top_percent), 1)
    flat_img = img.reshape(n_pixels, 3)
    idx = np.argpartition(depth.ravel(), -n_top)[-n_top:]
    candidates = flat_img[idx]
    return candidates[candidates.sum(axis=1).argmax()].astype(np.float32)


def dehaze_cap(
    img: np.ndarray,
    beta: float = 1.0,
    min_filter_size: int = 15,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
    t0: float = 0.1,
) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input phải là ảnh RGB 3 kênh")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    depth = compute_depth_map(img, min_filter_size)
    depth_refined = guided_filter(img, depth, guided_radius, guided_eps)
    A = estimate_atmospheric_light_cap(img, depth_refined)
    t = np.maximum(np.exp(-beta * depth_refined), t0)[..., None]
    return np.clip((img - A) / t + A, 0.0, 1.0)
