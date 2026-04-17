"""Các thuật toán khử sương mù."""
from .dcp import dehaze_dcp
from .clahe import dehaze_clahe
from .cap import dehaze_cap
from .hybrid import dehaze_hybrid

ALGORITHMS = {
    "DCP": dehaze_dcp,
    "CLAHE": dehaze_clahe,
    "CAP": dehaze_cap,
    "Hybrid": dehaze_hybrid,
}

__all__ = ["dehaze_dcp", "dehaze_clahe", "dehaze_cap", "dehaze_hybrid", "ALGORITHMS"]