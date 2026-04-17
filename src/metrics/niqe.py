"""
src/metrics/niqe.py
-------------------
NIQE (Natural Image Quality Evaluator) - no-reference metric.
Càng THẤP càng tốt.

Cài đặt full NIQE cần model file pretrained (modelparameters.mat).
Để đơn giản và không phụ thuộc nặng, ta dùng pyiqa nếu có; nếu không
có sẵn thì trả về None để benchmark vẫn chạy.

Cài đặt tuỳ chọn:
    pip install pyiqa torch
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

_NIQE_METRIC = None
_NIQE_READY: Optional[bool] = None  # None = chưa thử, True/False = đã thử


def _init_niqe():
    """Lazy init pyiqa metric (chỉ load khi lần đầu gọi)."""
    global _NIQE_METRIC, _NIQE_READY
    if _NIQE_READY is not None:
        return
    try:
        import pyiqa  # noqa
        import torch  # noqa

        _NIQE_METRIC = pyiqa.create_metric("niqe", as_loss=False)
        _NIQE_READY = True
    except Exception as e:  # pragma: no cover
        warnings.warn(
            f"[metrics.niqe] pyiqa không khả dụng ({e}). "
            "NIQE sẽ trả về None. Cài: pip install pyiqa torch"
        )
        _NIQE_READY = False


def compute_niqe(img: np.ndarray) -> Optional[float]:
    """Tính NIQE cho 1 ảnh RGB float [0,1]. Trả None nếu chưa cài pyiqa."""
    _init_niqe()
    if not _NIQE_READY:
        return None

    import torch

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # pyiqa yêu cầu tensor (N, C, H, W) float [0,1]
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        score = _NIQE_METRIC(t)
    return float(score.item())