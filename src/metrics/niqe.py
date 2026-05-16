from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

_NIQE_METRIC = None
_NIQE_READY: Optional[bool] = None


def _init_niqe() -> None:
    global _NIQE_METRIC, _NIQE_READY
    if _NIQE_READY is not None:
        return
    try:
        import pyiqa
        import torch
        _NIQE_METRIC = pyiqa.create_metric("niqe", as_loss=False)
        _NIQE_READY = True
    except Exception as e:
        warnings.warn(
            f"[metrics.niqe] pyiqa không khả dụng ({e}). "
            "NIQE sẽ trả về None. Cài: pip install pyiqa torch"
        )
        _NIQE_READY = False


def compute_niqe(img: np.ndarray) -> Optional[float]:
    _init_niqe()
    if not _NIQE_READY:
        return None

    import torch

    if img.dtype == np.uint8:
        img = img.astype(np.float32) * (1.0 / 255.0)

    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        score = _NIQE_METRIC(t)
    return float(score.item())
