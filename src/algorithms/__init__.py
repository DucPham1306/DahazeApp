from .dcp import dehaze_dcp
from .dcp_improved import dehaze_dcp_improved
from .clahe import dehaze_clahe
from .cap import dehaze_cap

ALGORITHMS = {
    "DCP": dehaze_dcp,
    "DCP-Improved": dehaze_dcp_improved,
    "CLAHE": dehaze_clahe,
    "CAP": dehaze_cap,
}

__all__ = ["dehaze_dcp", "dehaze_dcp_improved", "dehaze_clahe", "dehaze_cap", "ALGORITHMS"]
