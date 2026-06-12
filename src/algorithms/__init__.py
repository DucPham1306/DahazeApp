from .dcp import dehaze_dcp
from .dcp_improved import dehaze_dcp_improved
from .clahe import dehaze_clahe

ALGORITHMS = {
    "DCP": dehaze_dcp,
    "DCP-Improved": dehaze_dcp_improved,
    "CLAHE": dehaze_clahe,
}

__all__ = ["dehaze_dcp", "dehaze_dcp_improved", "dehaze_clahe", "ALGORITHMS"]
