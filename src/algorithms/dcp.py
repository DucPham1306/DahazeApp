"""
src/algorithms/dcp.py
---------------------
Dark Channel Prior (He, Sun, Tang - CVPR 2009, PAMI 2011).

Mô hình vật lý:
    I(x) = J(x) * t(x) + A * (1 - t(x))

Các bước:
  1) Tính dark channel:
        J_dark(x) = min_{c in RGB}( min_{y in Ω(x)} J^c(y) )
  2) Ước lượng atmospheric light A:
        - Lấy top 0.1% pixel sáng nhất của dark channel
        - Trong các pixel đó, chọn pixel có intensity cao nhất trong ảnh gốc
  3) Ước lượng transmission thô:
        t_thô(x) = 1 - ω * dark_channel(I / A)   với ω ≈ 0.95
  4) Refine t(x) bằng Guided Filter (thay cho soft matting gốc).
  5) Khôi phục ảnh:
        J(x) = (I(x) - A) / max(t(x), t0) + A   với t0 = 0.1
"""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.guided_filter import guided_filter


# --------------------------- Các bước con --------------------------- #
def dark_channel(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Tính dark channel với cửa sổ patch_size x patch_size."""
    # min theo trục kênh màu trước -> min theo không gian bằng erode
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def estimate_atmospheric_light(
    img: np.ndarray, dark: np.ndarray, top_percent: float = 0.001
) -> np.ndarray:
    """Ước lượng atmospheric light A (vector 3 kênh).

    Lấy top 0.1% pixel sáng nhất của dark channel, trong các pixel đó
    chọn pixel có tổng cường độ lớn nhất trong ảnh gốc -> đó là A.
    """
    h, w = dark.shape
    n_pixels = h * w
    n_top = max(int(n_pixels * top_percent), 1)

    dark_flat = dark.ravel()
    img_flat = img.reshape(n_pixels, 3)

    # indices của n_top pixel sáng nhất trong dark channel
    idx = np.argpartition(dark_flat, -n_top)[-n_top:]
    brightest = img_flat[idx]
    # pixel có tổng intensity cao nhất -> A
    A = brightest[np.argmax(brightest.sum(axis=1))]
    return A.astype(np.float32)  # shape (3,)


def estimate_transmission(
    img: np.ndarray, A: np.ndarray, omega: float = 0.95, patch_size: int = 15
) -> np.ndarray:
    """t(x) = 1 - ω * dark_channel(I / A)."""
    normed = img / A.reshape(1, 1, 3)
    normed = np.clip(normed, 0, 1)
    t = 1.0 - omega * dark_channel(normed, patch_size)
    return t


def recover(
    img: np.ndarray, A: np.ndarray, t: np.ndarray, t0: float = 0.1
) -> np.ndarray:
    """Khôi phục J từ I, A, t."""
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
    """Khử sương mù bằng Dark Channel Prior + Guided Filter refine.

    Args:
        img: RGB float [0,1], shape (H, W, 3).
        patch_size: kích thước patch cho dark channel.
        omega: tỷ lệ giữ lại 1 phần sương để tự nhiên (0.9–0.95).
        t0: cận dưới của t để tránh chia 0.
        guided_radius / guided_eps: tham số guided filter.
        return_intermediate: nếu True trả về cả A, t, dark để debug.

    Returns:
        J: ảnh đã khử sương mù, float [0,1].
    """
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