"""
src/metrics/no_reference.py
---------------------------
Các chỉ số No-Reference (không cần ảnh gốc) đánh giá chất lượng ảnh
sau khi khử sương mù.

Tất cả hàm nhận ảnh RGB float [0,1] hoặc uint8, trả về 1 số float.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------- Helpers ---------------------- #
def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


# ---------------------- Metrics ---------------------- #
def compute_contrast(img: np.ndarray) -> float:
    """RMS contrast = std(luminance) / 255. Càng cao, tương phản càng mạnh.

    Giá trị trong khoảng ~[0, 0.5]. Ảnh sương mù thường có contrast thấp.
    """
    gray = _to_gray_u8(img).astype(np.float32)
    return float(gray.std() / 255.0)


def compute_brightness(img: np.ndarray) -> float:
    """Độ sáng trung bình (mean luminance) chuẩn hoá [0,1].

    Ảnh có sương mù thường sáng đều (brightness cao). Sau khử sương,
    brightness thường giảm về mức tự nhiên hơn.
    """
    gray = _to_gray_u8(img).astype(np.float32)
    return float(gray.mean() / 255.0)


def compute_saturation(img: np.ndarray) -> float:
    """Độ bão hoà màu trung bình (S trong HSV) chuẩn hoá [0,1].

    Sương mù làm saturation giảm mạnh (ảnh trắng đục).
    Khử sương tốt => saturation tăng.
    """
    rgb_u8 = _to_rgb_u8(img)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.float32)
    return float(S.mean() / 255.0)


def compute_colorfulness(img: np.ndarray) -> float:
    """Colorfulness metric (Hasler & Süsstrunk 2003).

    Càng cao, ảnh càng "giàu màu". Ảnh sương mù gần grayscale nên
    colorfulness rất thấp (~5-10). Khử sương tốt có thể lên ~30-50.

    Formula:
        rg = R - G
        yb = 0.5(R + G) - B
        C  = sqrt(std(rg)^2 + std(yb)^2) + 0.3 * sqrt(mean(rg)^2 + mean(yb)^2)
    """
    rgb = _to_rgb_u8(img).astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def compute_avg_gradient(img: np.ndarray) -> float:
    """Average gradient (độ sắc nét trung bình).

    Tính sqrt((∂I/∂x)^2 + (∂I/∂y)^2) rồi lấy trung bình.
    Càng cao => ảnh càng sắc nét, nhiều chi tiết.
    """
    gray = _to_gray_u8(img).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def compute_laplacian_var(img: np.ndarray) -> float:
    """Variance of Laplacian - chỉ số blur/sharpness phổ biến.

    Càng cao = càng sắc nét. Càng thấp = càng mờ. Ảnh sương mù có
    Laplacian variance thấp vì chi tiết bị che.
    """
    gray = _to_gray_u8(img)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def compute_fade_like(img: np.ndarray) -> float:
    """Ước lượng "độ mù" đơn giản (FADE-like), càng THẤP càng ít sương.

    Ý tưởng: trong ảnh mù, dark channel sáng và saturation thấp.
    Chỉ số = mean(dark_channel) - mean(saturation), range ~[-1, 1].
    Ảnh rõ: dark_channel thấp, saturation cao => chỉ số âm.
    Ảnh mù: dark_channel cao, saturation thấp => chỉ số dương.
    """
    rgb = _to_rgb_u8(img).astype(np.float32) / 255.0
    dc = np.min(rgb, axis=2).mean()
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32).mean() / 255.0
    return float(dc - sat)