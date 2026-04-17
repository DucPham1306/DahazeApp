"""
src/algorithms/hybrid.py
------------------------
Pipeline lai CLAHE + DCP - hướng nâng cấp đề xuất trong đề cương.

Ý tưởng:
  1) Dùng CLAHE NHẸ để tăng tương phản cục bộ, giúp ước lượng
     dark channel/transmission ổn định hơn trên vùng bầu trời
     và vùng sương dày.
  2) Chạy DCP trên ảnh đã tăng sáng.
  3) (Tuỳ chọn) blend kết quả với ảnh gốc để tránh over-enhance.
"""
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
    """Pipeline CLAHE -> DCP.

    Args:
        blend: 0.0 => dùng hoàn toàn kết quả DCP;
               0.3 => trộn 70% DCP + 30% ảnh gốc (tránh over-enhance).
    """
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