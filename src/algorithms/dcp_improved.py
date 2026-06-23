from __future__ import annotations

import cv2
import numpy as np


def _box(img, r):
    k = 2 * r + 1
    return cv2.boxFilter(img, ddepth=-1, ksize=(k, k),
                         borderType=cv2.BORDER_REFLECT)

#Ánh sáng khí quyển A
def _atmospheric_light(img, dark, top_percent=0.001):
    n = dark.size
    n_top = max(int(n * top_percent), 1)
    flat = img.reshape(n, 3)
    idx = np.argpartition(dark.ravel(), -n_top)[-n_top:]
    cands = flat[idx]
    best_pixel = cands[cands.sum(axis=1).argmax()]
    return float(np.clip(best_pixel.mean(), 0.5, 0.95))
#============================

#Tinh chỉnh transmission bằng Fast Guided Filter
def _fast_gf_smallp(I_full_gray, I_small_gray, p_small, radius, eps, s):
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
#============================

def dehaze_dcp_improved(img, patch_size=15, omega=0.95, t0=0.1,
                         guided_radius=60, guided_eps=1e-3,
                         subsample=4, return_intermediate=False,
                         **_legacy):
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input phải là ảnh RGB 3 kênh")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    H, W = img.shape[:2]
    s = max(1, int(subsample)) #Cơ chế thu nhỏ 
    if min(H, W) // s < 64:
        s = max(1, min(H, W) // 64)
    Hs, Ws = max(1, H // s), max(1, W // s)

    ps = max(3, int(patch_size))
    if ps % 2 == 0:
        ps += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ps, ps))

    min_ch = img.min(axis=2)
    dark = cv2.erode(min_ch, kernel)

    A = _atmospheric_light(img, dark)

    #Phát hiện vùng trời với các đặc trưng về độ tối, độ bão hòa và độ sáng để điều chỉnh omega thích nghi theo vùng trời 
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
    #==========================

    #Transmission thô với omega thích nghi
    inv_A = 1.0 / A
    normed_min = np.clip(min_ch * inv_A, 0.0, 1.0)
    dc_norm = cv2.erode(normed_min, kernel)
    omega_map = omega - (0.20 * omega) * sky_full
    t_coarse = 1.0 - omega_map * dc_norm
    #======

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
        gray_full, gray_small, t_coarse_small, #Ngưỡng sàn t thích nghi theo vùng trời
        guided_radius, guided_eps, s,
    )
    np.clip(t_refined, 0.0, 1.0, out=t_refined) #Phục hồi ảnh

    t_floor = t0 + (0.60 - t0) * sky_full
    t_safe = np.maximum(t_refined, t_floor)[..., None] #Hòa trộn vùng trời để tránh nhiễu

    J = (img - A) / t_safe + A
    np.clip(J, 0.0, 1.0, out=J)

    blend = (0.55 * sky_full)[..., None]
    J = (1.0 - blend) * J + blend * img
    np.clip(J, 0.0, 1.0, out=J)
    if return_intermediate:
        return J, {"A": A, "t_coarse": t_coarse, "t_refined": t_refined,
                   "dark": dark, "sky_mask": sky_full}
    return J