from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
HAZY_NAMES = ("hazy", "Hazy", "HAZY", "haze", "input", "foggy")
GT_NAMES = ("gt", "GT", "Gt", "clear", "clean", "target", "ground_truth", "groundtruth")
_FIRST_NUM = re.compile(r"\d+")


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

    def filter(
        self, subsets: List[str] | None = None, limit: int = 0
    ) -> List[Tuple[str, str, str]]:
        by_subset: dict[str, list] = {}
        for sub, hz, gt in self.pairs:
            if subsets and sub not in subsets:
                continue
            by_subset.setdefault(sub, []).append((sub, hz, gt))
        out: List[Tuple[str, str, str]] = []
        for items in by_subset.values():
            out.extend(items[:limit] if limit > 0 else items)
        return out


def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _list_images(folder: str | Path) -> list:
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)


def _key(name: str, mode: str) -> str | None:
    s = _stem(name)
    if mode == "filename":
        return s
    if mode == "prefix":
        return s.split("_", 1)[0]
    if mode == "firstnum":
        m = _FIRST_NUM.search(s)
        return m.group() if m else None
    raise ValueError(mode)


def _build_gt_index(gt_files: list, mode: str) -> dict:
    idx: dict[str, str] = {}
    for f in gt_files:
        k = _key(f, mode)
        if k is not None:
            idx.setdefault(k, f)
    return idx


def _pair(hazy_files: list, gt_index: dict, mode: str) -> list:
    pairs = []
    for h in hazy_files:
        k = _key(h, mode)
        if k is not None and k in gt_index:
            pairs.append((h, gt_index[k]))
    return pairs


def pair_folders(hazy_dir: str | Path, gt_dir: str | Path) -> List[Tuple[str, str]]:
    hazy_files = _list_images(hazy_dir)
    gt_files = _list_images(gt_dir)
    if not hazy_files or not gt_files:
        return []

    best: list = []
    for mode in ("filename", "prefix", "firstnum"):
        idx = _build_gt_index(gt_files, mode)
        pairs = _pair(hazy_files, idx, mode)
        if len(pairs) > len(best):
            best = pairs
    return best


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

    for subset in ("indoor", "outdoor"):
        sub = root / subset
        if sub.is_dir():
            hz = _find_folder(sub, HAZY_NAMES)
            gt = _find_folder(sub, GT_NAMES)
            if hz and gt:
                pairs_all.extend((subset, h, g) for h, g in pair_folders(hz, gt))
    if pairs_all:
        return Dataset(display_name, pairs_all)

    hz = _find_folder(root, HAZY_NAMES)
    gt = _find_folder(root, GT_NAMES)
    if hz and gt:
        pairs_all.extend(("all", h, g) for h, g in pair_folders(hz, gt))
        if pairs_all:
            return Dataset(display_name, pairs_all)

    for sub in root.iterdir():
        if sub.is_dir():
            hz = _find_folder(sub, HAZY_NAMES)
            gt = _find_folder(sub, GT_NAMES)
            if hz and gt:
                pairs_all.extend((sub.name, h, g) for h, g in pair_folders(hz, gt))
    if pairs_all:
        return Dataset(display_name, pairs_all)

    raise ValueError(
        f"Không phát hiện được cấu trúc dataset hợp lệ trong {root}.\n"
        f"Gợi ý: root/hazy/ + root/gt/  hoặc  root/{{indoor,outdoor}}/{{hazy,gt}}/"
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
            f"Không ghép được cặp nào giữa {hazy_dir} và {gt_dir}. Kiểm tra tên file."
        )
    return Dataset(name, [(subset, h, g) for h, g in pairs_raw])
