from __future__ import annotations

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _prepare(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) * (1.0 / 255.0)
    return np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)


def _align_shapes(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape[:2] != gt.shape[:2]:
        gt = cv2.resize(
            gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA
        )
    return pred, gt


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _align_shapes(_prepare(pred), _prepare(gt))
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _align_shapes(_prepare(pred), _prepare(gt))
    return float(
        structural_similarity(gt, pred, channel_axis=2, data_range=1.0)
    )
