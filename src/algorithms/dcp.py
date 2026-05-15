from __future__ import annotations

import cv2
import numpy as np

from ..utils.guided_filter import guided_filter


# --------------------------- Các bước con --------------------------- #
def dark_channel(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def estimate_atmospheric_light(
    img: np.ndarray, dark: np.ndarray, top_percent: float = 0.001
) -> np.ndarray:
    h, w = dark.shape
    n_pixels = h * w
    n_top = max(int(n_pixels * top_percent), 1)

    dark_flat = dark.ravel()
    img_flat = img.reshape(n_pixels, 3)

    idx = np.argpartition(dark_flat, -n_top)[-n_top:]
    brightest = img_flat[idx]

    A = brightest[np.argmax(brightest.sum(axis=1))]
    return A.astype(np.float32)  


def estimate_transmission(
    img: np.ndarray, A: np.ndarray, omega: float = 0.95, patch_size: int = 15
) -> np.ndarray:
  
    normed = img / A.reshape(1, 1, 3)
    normed = np.clip(normed, 0, 1)
    t = 1.0 - omega * dark_channel(normed, patch_size)
    return t


def recover(
    img: np.ndarray, A: np.ndarray, t: np.ndarray, t0: float = 0.1
) -> np.ndarray:

    t = np.clip(t, t0, 1.0)
    t3 = np.repeat(t[..., np.newaxis], 3, axis=2)
    J = (img - A) / t3 + A
    return np.clip(J, 0, 1)


# --------------------------- API chính --------------------------- #
def dehaze_dcp(
    img: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
    return_intermediate: bool = False,
):

    assert img.ndim == 3 and img.shape[2] == 3, "Cần ảnh RGB 3 kênh"
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    dark = dark_channel(img, patch_size)
    A = estimate_atmospheric_light(img, dark)
    t_coarse = estimate_transmission(img, A, omega, patch_size)
    t_refined = guided_filter(img, t_coarse, guided_radius, guided_eps)
    J = recover(img, A, t_refined, t0)

    if return_intermediate:
        return J, {"A": A, "t_coarse": t_coarse, "t_refined": t_refined, "dark": dark}
    return J