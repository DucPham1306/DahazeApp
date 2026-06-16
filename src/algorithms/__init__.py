from .dcp import dehaze_dcp
from .dcp_improved import dehaze_dcp_improved

ALGORITHMS = {
    "DCP": dehaze_dcp,
    "DCP-Improved": dehaze_dcp_improved,
}

__all__ = ["dehaze_dcp", "dehaze_dcp_improved", "ALGORITHMS"]
