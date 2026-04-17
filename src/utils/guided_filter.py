"""
src/utils/guided_filter.py
--------------------------
Guided Filter (He et al. 2013) dùng để refine transmission map.

Thay cho soft matting kinh điển trong paper DCP gốc (chậm, phức tạp),
guided filter cho chất lượng tương đương nhưng nhanh hơn ~100 lần.

Tham khảo: http://kaiminghe.com/eccv10/
"""
from __future__ import annotations

import cv2
import numpy as np


def _box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """Box filter dùng cv2.blur (rất nhanh, trung bình trong cửa sổ 2r+1)."""
    ksize = 2 * radius + 1
    return cv2.blur(img, (ksize, ksize))


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 60,
    eps: float = 1e-3,
) -> np.ndarray:
    """Guided filter.

    Args:
        guide: ảnh dẫn (thường là ảnh gốc grayscale hoặc RGB), float [0,1].
        src:   ảnh cần refine (ví dụ transmission map thô), float [0,1].
        radius: bán kính cửa sổ (thường 40-80 cho ảnh ~480x640).
        eps:    regularization parameter (1e-3 ~ 1e-4).

    Returns:
        Ảnh đã refine, cùng shape với src.
    """
    # Dùng grayscale làm guide cho đơn giản (nếu guide là RGB)
    if guide.ndim == 3:
        guide_gray = cv2.cvtColor(
            (guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0
    else:
        guide_gray = guide.astype(np.float32)

    src = src.astype(np.float32)

    mean_I = _box_filter(guide_gray, radius)
    mean_p = _box_filter(src, radius)
    mean_Ip = _box_filter(guide_gray * src, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_filter(guide_gray * guide_gray, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    return mean_a * guide_gray + mean_b