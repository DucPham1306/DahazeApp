"""
src/algorithms/clahe.py
-----------------------
Contrast Limited Adaptive Histogram Equalization.

Không phải phương pháp dehaze đúng nghĩa, nhưng là baseline mạnh để so sánh
vì nó tăng cường tương phản cục bộ rất hiệu quả trong ảnh sương mù nhẹ.

Thực hiện:
  - Chuyển ảnh sang không gian LAB.
  - Áp CLAHE lên kênh L (độ sáng).
  - Merge lại và convert về RGB.
"""
from __future__ import annotations

import cv2
import numpy as np


def dehaze_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """Áp CLAHE để tăng tương phản cục bộ.

    Args:
        img: RGB float [0,1].
        clip_limit: giới hạn cắt histogram (2–4 là hợp lý).
        tile_grid_size: kích thước tile (8x8 là mặc định tốt).

    Returns:
        Ảnh RGB float [0,1] đã tăng tương phản.
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Cần ảnh RGB 3 kênh"
    # CLAHE làm việc trên uint8
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return rgb_eq.astype(np.float32) / 255.0