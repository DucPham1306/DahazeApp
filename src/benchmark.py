from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from .algorithms import ALGORITHMS
from .metrics import (
    compute_psnr,
    compute_ssim,
    compute_entropy,
    compute_edge_visibility,
    compute_niqe,
)
from .utils import list_sots_pairs, load_image, save_image


def run_single(
    algo_name: str,
    hazy_path: str,
    gt_path: str,
    save_dir: Path | None,
    subset: str,
    cached_pair: tuple | None = None,
) -> tuple[dict, tuple]:
    if cached_pair is None:
        hazy = load_image(hazy_path, as_float=True)
        gt = load_image(gt_path, as_float=True)
    else:
        hazy, gt = cached_pair

    t0 = time.perf_counter()
    pred = ALGORITHMS[algo_name](hazy)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    metrics = {
        "subset": subset,
        "algorithm": algo_name,
        "image": Path(hazy_path).name,
        "psnr": compute_psnr(pred, gt),
        "ssim": compute_ssim(pred, gt),
        "entropy": compute_entropy(pred),
        "edge_vis": compute_edge_visibility(pred),
        "niqe": compute_niqe(pred),
        "time_ms": elapsed_ms,
    }

    if save_dir is not None:
        out_dir = save_dir / subset / algo_name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(out_dir / Path(hazy_path).name, pred)

    return metrics, (hazy, gt)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark các thuật toán khử sương mù trên SOTS."
    )
    parser.add_argument("--sots-root", default="data/SOTS")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Lưu ảnh kết quả (tốn dung lượng).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Chỉ chạy N ảnh mỗi subset (0 = tất cả).",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images" if args.save_images else None

    pairs_by_subset = list_sots_pairs(args.sots_root)
    if not pairs_by_subset:
        raise SystemExit(
            f"Không tìm thấy dataset tại {args.sots_root}. "
            f"Hãy đảm bảo cấu trúc: SOTS/{{indoor,outdoor}}/{{hazy,gt}}"
        )

    records: list[dict] = []
    for subset, pairs in pairs_by_subset.items():
        if args.limit > 0:
            pairs = pairs[: args.limit]
        for hazy_p, gt_p in tqdm(pairs, ncols=80, desc=f">>> {subset.upper()}"):
            cached: tuple | None = None
            for algo in args.algorithms:
                try:
                    rec, cached = run_single(
                        algo, hazy_p, gt_p, img_dir, subset, cached
                    )
                    records.append(rec)
                except Exception as e:
                    print(f"  [!] Lỗi với {hazy_p} / {algo}: {e}")

    df = pd.DataFrame(records)
    csv_path = out_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nĐã lưu CSV: {csv_path}")

    print("\n===== TỔNG HỢP TRUNG BÌNH =====")
    numeric_cols = ["psnr", "ssim", "entropy", "edge_vis", "niqe", "time_ms"]
    agg_cols = [c for c in numeric_cols if c in df.columns and df[c].notna().any()]
    summary = (
        df.groupby(["subset", "algorithm"])[agg_cols]
        .mean(numeric_only=True)
        .round(4)
    )
    print(summary)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"\nĐã lưu summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
