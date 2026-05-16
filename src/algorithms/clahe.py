from __future__ import annotations

import cv2
import numpy as np

_CLAHE_CACHE: dict[tuple[float, int], "cv2.CLAHE"] = {}


def _get_clahe(clip_limit: float, tile_grid_size: int) -> "cv2.CLAHE":
    key = (round(clip_limit, 4), int(tile_grid_size))
    obj = _CLAHE_CACHE.get(key)
    if obj is None:
        obj = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )
        _CLAHE_CACHE[key] = obj
    return obj


def dehaze_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input phải là ảnh RGB 3 kênh")

    if img.dtype == np.uint8:
        img_u8 = img
    else:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    lab[..., 0] = _get_clahe(clip_limit, tile_grid_size).apply(lab[..., 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32) * (1.0 / 255.0)
