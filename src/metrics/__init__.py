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
    "compute_psnr",
    "compute_ssim",
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
    "compute_all_no_reference",
]


def compute_all_no_reference(img) -> dict:
    result = {
        "Contrast (RMS)": compute_contrast(img),
        "Brightness": compute_brightness(img),
        "Saturation": compute_saturation(img),
        "Colorfulness": compute_colorfulness(img),
        "Avg Gradient": compute_avg_gradient(img),
        "Sharpness (LapVar)": compute_laplacian_var(img),
        "Entropy (Shannon)": compute_entropy(img),
        "Edge Visibility": compute_edge_visibility(img),
        "Haze Index": compute_fade_like(img),
    }
    niqe = compute_niqe(img)
    if niqe is not None:
        result["NIQE"] = niqe
    return result
