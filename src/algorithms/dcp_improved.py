"""DCP cải tiến (Improved Dark Channel Prior).

So với DCP gốc (He et al. 2009), phiên bản này khắc phục các nhược điểm:
  1. Quad-tree atmospheric light  ⇒ tránh chọn nhầm vật thể trắng.
  2. Dark channel đa tỉ lệ          ⇒ giảm halo quanh biên vật thể.
  3. Sky-aware transmission         ⇒ giữ màu trời tự nhiên, ít color shift.
  4. Omega thích nghi cục bộ        ⇒ không over-dehaze vùng cận sương mờ.
  5. Color guided filter (3 kênh)   ⇒ biên sắc nét hơn.
  6. Gamma adaptive nhẹ             ⇒ cân bằng độ sáng mà không bị xỉn.
"""
from __future__ import annotations

import cv2
import numpy as np


def _dark_channel(img, patch_size):
    min_channel = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def _multi_scale_dark_channel(img, patches=(7, 15)):
    """Trộn dark channel ở patch nhỏ (giữ chi tiết) và patch lớn (ổn định).

    Ưu tiên patch lớn (0.7) để gần với DCP gốc, pha patch nhỏ (0.3) để
    giảm halo ở biên.
    """
    if len(patches) == 1:
        return _dark_channel(img, patches[0])
    dc_s = _dark_channel(img, patches[0])
    dc_l = _dark_channel(img, patches[-1])
    return 0.3 * dc_s + 0.7 * dc_l


def _quadtree_atmospheric_light(img, min_size=32):
    """Ước lượng A bằng phân vùng quad-tree.

    Tại mỗi bước, chia ảnh thành 4 góc và đi vào nhánh có (mean - std) lớn
    nhất – vùng trời thường sáng và đồng đều. Dừng khi block đủ nhỏ.
    """
    gray = img.mean(axis=2)
    h, w = gray.shape
    x0, y0, x1, y1 = 0, 0, w, h
    while (x1 - x0) > min_size and (y1 - y0) > min_size:
        mx = (x0 + x1) // 2
        my = (y0 + y1) // 2
        quads = [(x0, y0, mx, my), (mx, y0, x1, my),
                 (x0, my, mx, y1), (mx, my, x1, y1)]
        best_score = -np.inf
        best = quads[0]
        for q in quads:
            qx0, qy0, qx1, qy1 = q
            block = gray[qy0:qy1, qx0:qx1]
            if block.size == 0:
                continue
            score = float(block.mean() - block.std())
            if score > best_score:
                best_score = score
                best = q
        x0, y0, x1, y1 = best
    region = img[y0:y1, x0:x1].reshape(-1, 3)
    brightness = region.sum(axis=1)
    n_top = max(int(region.shape[0] * 0.1), 1)
    idx = np.argpartition(brightness, -n_top)[-n_top:]
    A = region[idx].mean(axis=0).astype(np.float32)
    return np.clip(A, 0.5, 1.0)


def _sky_mask(img, dark):
    """Soft mask vùng trời / vùng sáng đồng đều.

    Điều kiện chặt để không kích hoạt sai trên ảnh indoor.
    """
    max_c = img.max(axis=2)
    min_c = img.min(axis=2)
    saturation = (max_c - min_c) / (max_c + 1e-6)
    brightness = img.mean(axis=2)
    dark_n = np.clip((dark - 0.4) / 0.5, 0.0, 1.0)
    bright_n = np.clip((brightness - 0.65) / 0.30, 0.0, 1.0)
    sat_n = np.clip(1.0 - saturation / 0.18, 0.0, 1.0)
    mask = dark_n * bright_n * sat_n
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=8.0)
    return np.clip(mask, 0.0, 1.0)


def _adaptive_transmission(img, A, omega, patches, sky_mask):
    normed = np.clip(img / A, 0.0, 1.0)
    dc_norm = _multi_scale_dark_channel(normed, patches)
    # Vùng trời: omega giảm đến 80% để không gỡ hết sương ⇒ giữ màu trời.
    omega_map = omega * (1.0 - 0.20 * sky_mask)
    return 1.0 - omega_map * dc_norm


def _box(img, r):
    k = 2 * r + 1
    return cv2.boxFilter(img, ddepth=-1, ksize=(k, k),
                         borderType=cv2.BORDER_REFLECT)


def _color_guided_filter(guide, src, radius, eps):
    """Guided filter dùng 3 kênh màu làm guide (He 2013, eq.19).

    Giữ biên tốt hơn so với guide grayscale do khai thác đầy đủ tương phản
    màu giữa các kênh.
    """
    I = guide.astype(np.float32, copy=False)
    p = src.astype(np.float32, copy=False)
    Ir, Ig, Ib = I[..., 0], I[..., 1], I[..., 2]
    mIr, mIg, mIb = _box(Ir, radius), _box(Ig, radius), _box(Ib, radius)
    mp = _box(p, radius)
    cov_r = _box(Ir * p, radius) - mIr * mp
    cov_g = _box(Ig * p, radius) - mIg * mp
    cov_b = _box(Ib * p, radius) - mIb * mp
    vrr = _box(Ir * Ir, radius) - mIr * mIr + eps
    vrg = _box(Ir * Ig, radius) - mIr * mIg
    vrb = _box(Ir * Ib, radius) - mIr * mIb
    vgg = _box(Ig * Ig, radius) - mIg * mIg + eps
    vgb = _box(Ig * Ib, radius) - mIg * mIb
    vbb = _box(Ib * Ib, radius) - mIb * mIb + eps
    # Cofactors / inverse 3x3 phần tử-wise
    c11 = vgg * vbb - vgb * vgb
    c12 = vgb * vrb - vrg * vbb
    c13 = vrg * vgb - vgg * vrb
    c22 = vrr * vbb - vrb * vrb
    c23 = vrg * vrb - vrr * vgb
    c33 = vrr * vgg - vrg * vrg
    det = vrr * c11 + vrg * c12 + vrb * c13
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    a_r = (c11 * cov_r + c12 * cov_g + c13 * cov_b) / det
    a_g = (c12 * cov_r + c22 * cov_g + c23 * cov_b) / det
    a_b = (c13 * cov_r + c23 * cov_g + c33 * cov_b) / det
    b = mp - a_r * mIr - a_g * mIg - a_b * mIb
    return (_box(a_r, radius) * Ir + _box(a_g, radius) * Ig +
            _box(a_b, radius) * Ib + _box(b, radius))


def _recover(img, A, t, t0, sky_mask):
    # Sàn truyền sáng cao hơn ở vùng trời ⇒ tránh khuếch đại quá mức
    t_floor = t0 + (0.4 - t0) * sky_mask
    t_safe = np.maximum(t, t_floor)[..., None]
    return np.clip((img - A) / t_safe + A, 0.0, 1.0)


def _adaptive_gamma(img):
    """Gamma nhẹ – chỉ kích hoạt khi ảnh sau dehaze thực sự tối.

    Tránh việc luôn đẩy luminance về 0.5 (làm sáng quá ảnh indoor có
    ground-truth tối).
    """
    luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    mean_l = float(np.clip(luma.mean(), 0.05, 0.95))
    if mean_l >= 0.45:
        return img
    target = min(mean_l + 0.05, 0.5)
    gamma = float(np.clip(np.log(target) / np.log(mean_l), 0.85, 1.0))
    if abs(gamma - 1.0) < 0.01:
        return img
    return np.power(np.clip(img, 0.0, 1.0), gamma)


def dehaze_dcp_improved(
    img,
    patch_size=15,
    omega=0.95,
    t0=0.1,
    guided_radius=60,
    guided_eps=1e-3,
    use_gamma=True,
    return_intermediate=False,
):
    """DCP cải tiến. Tham số tương thích DCP gốc để dễ benchmark."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input phải là ảnh RGB 3 kênh")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    p_small = max(3, (patch_size // 2) | 1)
    patches = (p_small, patch_size)

    dark = _multi_scale_dark_channel(img, patches)
    A = _quadtree_atmospheric_light(img)
    sky = _sky_mask(img, dark)

    t_coarse = _adaptive_transmission(img, A, omega, patches, sky)
    t_refined = _color_guided_filter(img, t_coarse, guided_radius, guided_eps)
    t_refined = np.clip(t_refined, 0.0, 1.0)

    J = _recover(img, A, t_refined, t0, sky)
    if use_gamma:
        J = _adaptive_gamma(J)

    if return_intermediate:
        return J, {"A": A, "t_coarse": t_coarse, "t_refined": t_refined,
                   "dark": dark, "sky_mask": sky}
    return J
