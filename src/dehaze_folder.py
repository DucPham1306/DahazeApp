"""Khử sương mù hàng loạt cho một subset của SOTS.

Đọc tất cả ảnh trong  data/SOTS/<subset>/hazy/  rồi lưu kết quả ra
data/SOTS/<subset>/dehazed/<stem>_dehazed.png  (giống cấu trúc thư mục indoor).

Ví dụ:
    python -m src.dehaze_folder --subset outdoor
    python -m src.dehaze_folder --subset indoor --subset outdoor
    python -m src.dehaze_folder --sots-root data/SOTS --algorithm DCP-Improved
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

from .algorithms import ALGORITHMS
from .utils import load_image, save_image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def dehaze_subset(
    sots_root: Path,
    subset: str,
    algo_name: str,
    out_name: str = "dehazed",
    overwrite: bool = False,
) -> int:
    """Khử sương mù toàn bộ ảnh hazy của một subset. Trả về số ảnh đã lưu."""
    hazy_dir = sots_root / subset / "hazy"
    if not hazy_dir.is_dir():
        print(f"  [bỏ qua] Không tìm thấy thư mục hazy: {hazy_dir}")
        return 0

    out_dir = sots_root / subset / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    algo = ALGORITHMS[algo_name]
    hazy_files = sorted(
        f for f in hazy_dir.iterdir() if f.suffix.lower() in IMG_EXTS
    )

    saved = 0
    for f in tqdm(hazy_files, ncols=80, desc=f">>> {subset.upper()}"):
        out_path = out_dir / f"{f.stem}_dehazed.png"
        if out_path.exists() and not overwrite:
            continue
        try:
            hazy = load_image(str(f), as_float=True)
            pred = algo(hazy)
            save_image(str(out_path), pred)
            saved += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [!] Lỗi với {f.name}: {e}")
    print(f"  Đã lưu {saved} ảnh vào {out_dir}")
    return saved


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Khử sương mù hàng loạt cho subset SOTS, lưu ra <subset>/dehazed/."
    )
    parser.add_argument("--sots-root", default="data/SOTS")
    parser.add_argument(
        "--subset",
        action="append",
        choices=["indoor", "outdoor"],
        help="Subset cần xử lý (lặp lại để chọn nhiều). Mặc định: cả indoor và outdoor.",
    )
    parser.add_argument(
        "--algorithm",
        default="DCP-Improved",
        choices=list(ALGORITHMS.keys()),
    )
    parser.add_argument("--out-name", default="dehazed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè ảnh đã tồn tại (mặc định bỏ qua).",
    )
    args = parser.parse_args(argv)

    subsets = args.subset or ["indoor", "outdoor"]
    root = Path(args.sots_root)

    total = 0
    for subset in subsets:
        total += dehaze_subset(
            root, subset, args.algorithm, args.out_name, args.overwrite
        )
    print(f"\nHoàn tất. Tổng cộng {total} ảnh đã được khử sương mù.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
