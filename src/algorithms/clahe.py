
from __future__ import annotations

import cv2
import numpy as np


def dehaze_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:

    assert img.ndim == 3 and img.shape[2] == 3, "Cần ảnh RGB 3 kênh"

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