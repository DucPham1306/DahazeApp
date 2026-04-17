"""
src/metrics/psnr_ssim.py
------------------------
Full-reference metrics: PSNR và SSIM.
Dùng scikit-image.
"""
from __future__ import annotations

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _prepare(img: np.ndarray) -> np.ndarray:
    """Đảm bảo float32 trong [0,1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0, 1)


def _align_shapes(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nếu kích thước khác nhau (thường gặp với SOTS) thì resize gt về pred."""
    if pred.shape[:2] != gt.shape[:2]:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
    return pred, gt


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR (dB). Càng cao càng tốt."""
    pred = _prepare(pred)
    gt = _prepare(gt)
    pred, gt = _align_shapes(pred, gt)
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM [0,1]. Càng gần 1 càng tốt."""
    pred = _prepare(pred)
    gt = _prepare(gt)
    pred, gt = _align_shapes(pred, gt)
    return float(
        structural_similarity(gt, pred, channel_axis=2, data_range=1.0)
    )