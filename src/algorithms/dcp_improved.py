"""DCP cải tiến – nhanh HƠN và chất lượng (PSNR/SSIM) CAO HƠN DCP gốc.

Cải tiến CHẤT LƯỢNG (PSNR ↑, SSIM ↑) so với He et al. 2009:
  1. Quad-tree + DCP-style A      → A ổn định, scalar = mean(A_rgb) bị
     kẹp [0.5, 0.95] để không quá saturate khi vùng cực sáng là noise.
  2. Sky/bright-region mask CHẶT  → chỉ kích hoạt khi đồng thời sáng cao
     + saturation thấp + dark cao (3 điều kiện).
  3. Omega thích nghi nhẹ ở sky   → giảm over-dehaze, giữ tone trời.
  4. t-floor thích nghi           → nâng nền truyền qua ở sky để giữ
     màu trời tự nhiên.
  5. Sky-blend nhẹ                → blend kết quả với input ở sky-mask
     để giảm artifact xám-xanh do DCP gốc gây ra.
  6. KHÔNG gamma                  → tránh sai pixel-value so với GT.

TĂNG TỐC (~2× DCP gốc):
  - Dark channel + A + t_coarse tính ở FULL-RES (rẻ với cv2.erode),
    bảo đảm chất lượng map t (yếu tố chính của SSIM).
  - Sky mask tính ở SCALE NHỎ (Gaussian blur đắt) rồi upsample lên full.
  - Fast Guided Filter (He 2015): hệ số (a, b) tính ở scale 1/s với
    4 box-filter trên guide GRAYSCALE (thay vì 13 box-filter color),
    upsample (a, b) lên full-res rồi áp q = a·I_full + b giữ biên.
"""
from __future__ import annotations

import cv2
import numpy as np


def _box(img, r):
    k = 2 * r + 1
    return cv2.boxFilter(img, ddepth=-1, ksize=(k, k),
                         borderType=cv2.BORDER_REFLECT)


def _atmospheric_light(img, dark, top_percent=0.001):
    """A theo cách DCP gốc: top-0.1% pixel có dark cao nhất, chọn
    pixel sáng nhất. Trả về scalar = mean(A_rgb) bị kẹp [0.5, 0.95].
    """
    n = dark.size
    n_top = max(int(n * top_percent), 1)
    flat = img.reshape(n, 3)
    idx = np.argpartition(dark.ravel(), -n_top)[-n_top:]
    cands = flat[idx]
    best_pixel = cands[cands.sum(axis=1).argmax()]
    return float(np.clip(best_pixel.mean(), 0.5, 0.95))


def _fast_gf_smallp(I_full_gray, I_small_gray, p_small, radius, eps, s):
    """Fast Guided Filter (He 2015) – guide grayscale, p ở scale 1/s.
    Tính (a, b) ở scale nhỏ, upsample về full, áp q = a·I_full + b.
    """
    H, W = I_full_gray.shape[:2]
    rs = max(1, radius // s)
    mI = _box(I_small_gray, rs)
    mp = _box(p_small, rs)
    mIp = _box(I_small_gray * p_small, rs)
    mII = _box(I_small_gray * I_small_gray, rs)
    var_I = mII - mI * mI
    cov_Ip = mIp - mI * mp
    a = cov_Ip / (var_I + eps)
    b = mp - a * mI
    a = _box(a, rs)
    b = _box(b, rs)
    if s > 1:
        a_full = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        b_full = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        a_full, b_full = a, b
    return a_full * I_full_gray + b_full


def dehaze_dcp_improved(img, patch_size=15, omega=0.95, t0=0.1,
                         guided_radius=60, guided_eps=1e-3,
                         subsample=4, return_intermediate=False,
                         **_legacy):
    """DCP cải tiến – nhanh hơn & cho PSNR/SSIM cao hơn DCP gốc.

    Tham số tương thích DCP gốc.
    Tham số bổ sung:
      subsample (int): hệ số thu nhỏ DÙNG RIÊNG cho Fast Guided Filter
        & sky mask (mặc định 4). Dark channel, A, t_coarse vẫn full-res.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input phải là ảnh RGB 3 kênh")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    H, W = img.shape[:2]
    s = max(1, int(subsample))
    if min(H, W) // s < 64:
        s = max(1, min(H, W) // 64)
    Hs, Ws = max(1, H // s), max(1, W // s)

    ps = max(3, int(patch_size))
    if ps % 2 == 0:
        ps += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ps, ps))

    # Full-res dark/min channel (cheap)
    min_ch = img.min(axis=2)
    dark = cv2.erode(min_ch, kernel)

    # A theo DCP gốc
    A = _atmospheric_light(img, dark)

    # ---- Sky mask ở SCALE NHỎ (Gaussian blur đắt nên hạ scale) ----
    img_small = cv2.resize(img, (Ws, Hs), interpolation=cv2.INTER_AREA) \
        if s > 1 else img
    min_s = img_small.min(axis=2)
    max_s = img_small.max(axis=2)
    ps_s = max(3, ps // s)
    if ps_s % 2 == 0:
        ps_s += 1
    ker_s = cv2.getStructuringElement(cv2.MORPH_RECT, (ps_s, ps_s))
    dark_s = cv2.erode(min_s, ker_s)
    saturation = (max_s - min_s) / (max_s + 1e-6)
    brightness = 0.5 * (max_s + min_s)
    dark_n = np.clip((dark_s - 0.60) * (1.0 / 0.35), 0.0, 1.0)
    bright_n = np.clip((brightness - 0.75) * (1.0 / 0.25), 0.0, 1.0)
    sat_n = np.clip(1.0 - saturation * (1.0 / 0.10), 0.0, 1.0)
    sky_s = dark_n * bright_n * sat_n
    sky_s = cv2.GaussianBlur(sky_s.astype(np.float32, copy=False),
                              (0, 0), sigmaX=max(2.0, 10.0 / s))
    np.clip(sky_s, 0.0, 1.0, out=sky_s)
    sky_full = cv2.resize(sky_s, (W, H), interpolation=cv2.INTER_LINEAR) \
        if s > 1 else sky_s

    # ---- Transmission thô full-res ----
    inv_A = 1.0 / A
    normed_min = np.clip(min_ch * inv_A, 0.0, 1.0)
    dc_norm = cv2.erode(normed_min, kernel)
    omega_map = omega - (0.20 * omega) * sky_full
    t_coarse = 1.0 - omega_map * dc_norm

    # ---- Fast Guided Filter ----
    coeff = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray_full = (img @ coeff).astype(np.float32, copy=False)
    if s > 1:
        gray_small = cv2.resize(gray_full, (Ws, Hs),
                                 interpolation=cv2.INTER_AREA)
        t_coarse_small = cv2.resize(t_coarse, (Ws, Hs),
                                     interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray_full
        t_coarse_small = t_coarse

    t_refined = _fast_gf_smallp(
        gray_full, gray_small, t_coarse_small,
        guided_radius, guided_eps, s,
    )
    np.clip(t_refined, 0.0, 1.0, out=t_refined)

    # ---- t-floor thích nghi + recover ----
    t_floor = t0 + (0.60 - t0) * sky_full
    t_safe = np.maximum(t_refined, t_floor)[..., None]

    J = (img - A) / t_safe + A
    np.clip(J, 0.0, 1.0, out=J)

    # ---- Sky-blend nhẹ ----
    blend = (0.55 * sky_full)[..., None]
    J = (1.0 - blend) * J + blend * img
    np.clip(J, 0.0, 1.0, out=J)

    if return_intermediate:
        return J, {"A": A, "t_coarse": t_coarse, "t_refined": t_refined,
                   "dark": dark, "sky_mask": sky_full}
    return J
