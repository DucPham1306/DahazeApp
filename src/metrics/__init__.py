"""Các chỉ số đánh giá chất lượng ảnh sau khử sương mù."""
from .psnr_ssim import compute_psnr, compute_ssim
from .edge_entropy import compute_entropy, compute_edge_visibility
from .niqe import compute_niqe
from .no_reference import (
    compute_contrast,
    compute_brightness,
    compute_saturation,
    compute_colorfulness,
    compute_avg_gradient,
    compute_laplacian_var,
    compute_fade_like,
)

__all__ = [
    # Full-reference (cần GT)
    "compute_psnr",
    "compute_ssim",
    # No-reference
    "compute_entropy",
    "compute_edge_visibility",
    "compute_niqe",
    "compute_contrast",
    "compute_brightness",
    "compute_saturation",
    "compute_colorfulness",
    "compute_avg_gradient",
    "compute_laplacian_var",
    "compute_fade_like",
]


# Tiện ích: gom nhiều metric no-reference vào 1 dict
def compute_all_no_reference(img) -> dict:
    """Tính tất cả metric no-reference, trả dict {tên: giá_trị}."""
    result = {
        "Contrast (RMS)": compute_contrast(img), #Độ tương phản
        "Brightness": compute_brightness(img), #Độ sáng
        "Saturation": compute_saturation(img), #Độ bão hòa
        "Colorfulness": compute_colorfulness(img), #Độ sống động màu sắc
        "Avg Gradient": compute_avg_gradient(img), #Độ sắc nét trung bình
        "Sharpness (LapVar)": compute_laplacian_var(img), #Độ sắc nét (biến thiên Laplacian)
        "Entropy (Shannon)": compute_entropy(img), #Độ phức tạp thông tin
        "Edge Visibility": compute_edge_visibility(img), #Độ rõ cạnh
        "Haze Index": compute_fade_like(img), #Chỉ số sương mù (dựa trên histogram)
    }
    niqe = compute_niqe(img)
    if niqe is not None:
        result["NIQE"] = niqe
    return result