"""
src/algorithms/cap.py
---------------------
Color Attenuation Prior (Zhu, Mai, Shao - IEEE TIP 2015).

Ý tưởng:
  Quan sát rằng với ảnh có sương mù, độ chênh lệch giữa brightness (V)
  và saturation (S) trong HSV gần như tỷ lệ thuận với độ dày sương mù,
  tức là tỷ lệ thuận với độ sâu scene d(x):

        d(x) = θ0 + θ1 * V(x) + θ2 * S(x) + ε(x)

  Trong paper gốc tác giả học các tham số bằng supervised learning trên
  dataset ảnh nhân tạo, thu được:
        θ0 =  0.121779
        θ1 =  0.959710
        θ2 = -0.780245
        σ  =  0.041337   (std của noise ε ~ N(0, σ^2))

Từ depth -> transmission:
        t(x) = exp(-β * d(x))
Sau đó refine t bằng Guided Filter, ước lượng A, và khôi phục như DCP.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.guided_filter import guided_filter


# Tham số đã học sẵn trong paper
THETA0 = 0.121779
THETA1 = 0.959710
THETA2 = -0.780245
SIGMA = 0.041337


def compute_depth_map(img: np.ndarray, min_filter_size: int = 15) -> np.ndarray:
    """Ước lượng depth map theo mô hình CAP."""
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    S = hsv[..., 1]
    V = hsv[..., 2]

    d = THETA0 + THETA1 * V + THETA2 * S
    # Min filter cục bộ để khử nhiễu kết cấu nhỏ -> depth mượt hơn
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (min_filter_size, min_filter_size)
    )
    d = cv2.erode(d, kernel)
    return d


def estimate_atmospheric_light_cap(
    img: np.ndarray, depth: np.ndarray, top_percent: float = 0.001
) -> np.ndarray:
    """A = pixel xa nhất (depth cao nhất) trong ảnh (top 0.1%)."""
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
    """Khử sương mù bằng Color Attenuation Prior.

    Args:
        img: RGB float [0,1].
        beta: hệ số tán xạ (1.0 theo paper; tăng để khử mạnh hơn).
        min_filter_size: kích thước min filter áp lên depth map.
        guided_radius / guided_eps: tham số refine.
        t0: cận dưới của t.
    """
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