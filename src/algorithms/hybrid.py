from __future__ import annotations

import numpy as np

from .clahe import dehaze_clahe
from .dcp import dehaze_dcp


def dehaze_hybrid(
    img: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid_size: int = 8,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    guided_radius: int = 60,
    guided_eps: float = 1e-3,
    blend: float = 0.0,
) -> np.ndarray:

    pre = dehaze_clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    J = dehaze_dcp(
        pre,
        patch_size=patch_size,
        omega=omega,
        t0=t0,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
    )
    if blend > 0:
        J = np.clip((1.0 - blend) * J + blend * img, 0, 1)
    return J