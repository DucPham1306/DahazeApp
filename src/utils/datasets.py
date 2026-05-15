from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Các tên folder phổ biến cho ảnh mù và ảnh gốc
HAZY_NAMES = ("hazy", "Hazy", "HAZY", "haze", "input", "foggy")
GT_NAMES = ("gt", "GT", "Gt", "clear", "clean", "target", "ground_truth", "groundtruth")


# ========================================================================= #
#                              Dataclass Dataset
# ========================================================================= #
@dataclass
class Dataset:
    name: str
    pairs: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.pairs)

    @property
    def subsets(self) -> List[str]:
        return sorted({p[0] for p in self.pairs})

    def filter(self, subsets: List[str] | None = None, limit: int = 0) -> List[Tuple[str, str, str]]:
        out: List[Tuple[str, str, str]] = []
        by_subset: dict[str, list] = {}
        for sub, hz, gt in self.pairs:
            if subsets and sub not in subsets:
                continue
            by_subset.setdefault(sub, []).append((sub, hz, gt))
        for sub, items in by_subset.items():
            if limit > 0:
                items = items[:limit]
            out.extend(items)
        return out


# ========================================================================= #
#                        Các hàm ghép cặp hazy <-> gt
# ========================================================================= #
def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _pair_by_filename(hazy_files: list, gt_index: dict) -> list:
    pairs = []
    for h in hazy_files:
        key = _stem(h)
        if key in gt_index:
            pairs.append((h, gt_index[key]))
    return pairs


def _pair_by_prefix(hazy_files: list, gt_index: dict) -> list:
    pairs = []
    for h in hazy_files:
        key = _stem(h).split("_")[0]
        if key in gt_index:
            pairs.append((h, gt_index[key]))
    return pairs


_FIRST_NUM = re.compile(r"\d+")


def _pair_by_first_num(hazy_files: list, gt_index: dict) -> list:
    pairs = []
    for h in hazy_files:
        m = _FIRST_NUM.search(_stem(h))
        if m and m.group() in gt_index:
            pairs.append((h, gt_index[m.group()]))
    return pairs


def _list_images(folder: str | Path) -> list:
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(
        str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXTS
    )


def _build_gt_index(gt_files: list, mode: str) -> dict:
    idx: dict[str, str] = {}
    for f in gt_files:
        if mode == "filename":
            key = _stem(f)
        elif mode == "prefix":
            key = _stem(f).split("_")[0]
        elif mode == "firstnum":
            m = _FIRST_NUM.search(_stem(f))
            if not m:
                continue
            key = m.group()
        else:
            raise ValueError(mode)
        idx.setdefault(key, f)
    return idx


def pair_folders(hazy_dir: str | Path, gt_dir: str | Path) -> List[Tuple[str, str]]:
    hazy_files = _list_images(hazy_dir)
    gt_files = _list_images(gt_dir)
    if not hazy_files or not gt_files:
        return []

    best = []
    for mode in ("filename", "prefix", "firstnum"):
        idx = _build_gt_index(gt_files, mode)
        if mode == "filename":
            pairs = _pair_by_filename(hazy_files, idx)
        elif mode == "prefix":
            pairs = _pair_by_prefix(hazy_files, idx)
        else:
            pairs = _pair_by_first_num(hazy_files, idx)
        if len(pairs) > len(best):
            best = pairs
    return best


# ========================================================================= #
#                        Phát hiện cấu trúc tự động
# ========================================================================= #
def _find_folder(root: Path, candidates: tuple) -> Path | None:
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    return None


def detect_dataset(root: str | Path, name: str | None = None) -> Dataset:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Thư mục không tồn tại: {root}")

    display_name = name or root.name
    pairs_all: List[Tuple[str, str, str]] = []

    # 1) SOTS style
    for subset in ("indoor", "outdoor"):
        sub = root / subset
        if sub.is_dir():
            hz = _find_folder(sub, HAZY_NAMES)
            gt = _find_folder(sub, GT_NAMES)
            if hz and gt:
                for h, g in pair_folders(hz, gt):
                    pairs_all.append((subset, h, g))
    if pairs_all:
        return Dataset(display_name, pairs_all)

    # 2) Flat style: root/{hazy,gt}/
    hz = _find_folder(root, HAZY_NAMES)
    gt = _find_folder(root, GT_NAMES)
    if hz and gt:
        for h, g in pair_folders(hz, gt):
            pairs_all.append(("all", h, g))
        if pairs_all:
            return Dataset(display_name, pairs_all)

    # 3) Đệ quy 1 cấp
    for sub in root.iterdir():
        if sub.is_dir():
            hz = _find_folder(sub, HAZY_NAMES)
            gt = _find_folder(sub, GT_NAMES)
            if hz and gt:
                for h, g in pair_folders(hz, gt):
                    pairs_all.append((sub.name, h, g))
    if pairs_all:
        return Dataset(display_name, pairs_all)

    raise ValueError(
        f"Không phát hiện được cấu trúc dataset hợp lệ trong {root}.\n"
        f"Gợi ý cấu trúc: root/hazy/ + root/gt/  hoặc  "
        f"root/{{indoor,outdoor}}/{{hazy,gt}}/"
    )


def load_custom_dataset(
    hazy_dir: str | Path,
    gt_dir: str | Path,
    name: str = "custom",
    subset: str = "all",
) -> Dataset:
    pairs_raw = pair_folders(hazy_dir, gt_dir)
    if not pairs_raw:
        raise ValueError(
            f"Không ghép được cặp nào giữa {hazy_dir} và {gt_dir}. "
            "Kiểm tra tên file."
        )
    pairs = [(subset, h, g) for h, g in pairs_raw]
    return Dataset(name, pairs)