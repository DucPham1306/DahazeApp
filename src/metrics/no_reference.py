from __future__ import annotations

import cv2
import numpy as np


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


def compute_contrast(img: np.ndarray) -> float:
    return float(_to_gray_u8(img).std() / 255.0)


def compute_brightness(img: np.ndarray) -> float:
    return float(_to_gray_u8(img).mean() / 255.0)


def compute_saturation(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(_to_rgb_u8(img), cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1].mean() / 255.0)


def compute_colorfulness(img: np.ndarray) -> float:
    rgb = _to_rgb_u8(img).astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_root = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean_root = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return std_root + 0.3 * mean_root


def compute_avg_gradient(img: np.ndarray) -> float:
    gray = _to_gray_u8(img).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(cv2.magnitude(gx, gy).mean())


def compute_laplacian_var(img: np.ndarray) -> float:
    return float(cv2.Laplacian(_to_gray_u8(img), cv2.CV_32F).var())


def compute_fade_like(img: np.ndarray) -> float:
    rgb_u8 = _to_rgb_u8(img)
    rgb = rgb_u8.astype(np.float32) * (1.0 / 255.0)
    dc = float(rgb.min(axis=2).mean())
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean() / 255.0)
    return dc - sat
