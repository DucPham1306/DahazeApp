from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_image(path: str | Path, as_float: bool = True) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_float:
        img = img.astype(np.float32) * (1.0 / 255.0)
    return img


def save_image(path: str | Path, img: np.ndarray) -> None:
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    out_bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out_bgr)


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) * (1.0 / 255.0)
    return img.astype(np.float32, copy=False)


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _stem_prefix(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    return stem.split("_", 1)[0]


def pair_sots(hazy_dir: str | Path, gt_dir: str | Path) -> List[Tuple[str, str]]:
    hazy_dir = Path(hazy_dir)
    gt_dir = Path(gt_dir)
    if not hazy_dir.is_dir():
        raise FileNotFoundError(f"Thư mục hazy không tồn tại: {hazy_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Thư mục gt không tồn tại: {gt_dir}")

    gt_index: dict[str, str] = {}
    for f in gt_dir.iterdir():
        if f.suffix.lower() in IMG_EXTS:
            gt_index.setdefault(_stem_prefix(f.name), str(f))

    pairs: List[Tuple[str, str]] = []
    for f in sorted(hazy_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        gt_path = gt_index.get(_stem_prefix(f.name))
        if gt_path:
            pairs.append((str(f), gt_path))
    return pairs


def list_sots_pairs(sots_root: str | Path) -> dict[str, List[Tuple[str, str]]]:
    root = Path(sots_root)
    out: dict[str, List[Tuple[str, str]]] = {}
    for subset in ("indoor", "outdoor"):
        hazy = root / subset / "hazy"
        gt = root / subset / "gt"
        if hazy.is_dir() and gt.is_dir():
            out[subset] = pair_sots(hazy, gt)
    return out
