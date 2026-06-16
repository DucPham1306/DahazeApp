from __future__ import annotations

import cv2
import numpy as np

from ..utils.guided_filter import guided_filter

def dark_channel(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    min_channel = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)

def estimate_atmospheric_light(
    img: np.ndarray, dark: np.ndarray, top_percent: float = 0.001
) -> np.ndarray:
    n_pixels = dark.size
    n_top = max(int(n_pixels * top_percent), 1)
    flat_img = img.reshape(n_pixels, 3)
    idx = np.argpartition(dark.ravel(), -n_top)[-n_top:]
    candidates = flat_img[idx]
    return candidates[candidates.sum(axis=1).argmax()].astype(np.float32)


def estimate_transmission(
    img: np.ndarray, A: np.ndarray, omega: float = 0.95, patch_size: int = 15
) -> np.ndarray:
    normed = np.clip(img / A, 0.0, 1.0)
    return 1.0 - omega * dark_channel(normed, patch_size)


def recover(
    img: np.ndarray, A: np.ndarray, t: np.ndarray, t0: float = 0.1
) -> np.ndarray:
    t_safe = np.maximum(t, t0)[..., None]
    return np.clip((img - A) / t_safe + A, 0.0, 1.0)


def dehaze_dcp(
    img: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
    return_intermediate: bool = False,
):
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input phải là ảnh RGB 3 kênh")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    dark = dark_channel(img, patch_size)
    A = estimate_atmospheric_light(img, dark)
    t_coarse = estimate_transmission(img, A, omega, patch_size)
    t_refined = guided_filter(img, t_coarse, guided_radius, guided_eps)
    J = recover(img, A, t_refined, t0)

    if return_intermediate:
        return J, {"A": A, "t_coarse": t_coarse, "t_refined": t_refined, "dark": dark}
    return J
