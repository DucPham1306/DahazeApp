"""
src/gui/benchmark_tab.py
------------------------
Tab "Đánh giá Dataset" — chạy tất cả thuật toán trên toàn bộ SOTS
(hoặc subset + limit), đo PSNR/SSIM + chỉ số no-reference, hiển thị
bảng tổng hợp và cho phép export CSV.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd
from PyQt5 import QtCore, QtWidgets

from ..algorithms import ALGORITHMS
from ..metrics import (
    compute_psnr,
    compute_ssim,
    compute_contrast,
    compute_colorfulness,
    compute_avg_gradient,
    compute_laplacian_var,
    compute_entropy,
    compute_edge_visibility,
    compute_fade_like,
)
from ..utils import list_sots_pairs, load_image


# ========================================================================= #
#                          BenchmarkWorker (QThread)
# ========================================================================= #
class BenchmarkWorker(QtCore.QThread):
    """Chạy benchmark trong thread riêng, emit progress & kết quả."""

    progress = QtCore.pyqtSignal(int, int, str)   # (current, total, msg)
    finished_ok = QtCore.pyqtSignal(list)          # list[dict] records
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        sots_root: str,
        algorithms: List[str],
        subsets: List[str],
        limit: int,
    ):
        super().__init__()
        self.sots_root = sots_root
        self.algorithms = algorithms
        self.subsets = subsets
        self.limit = limit
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            pairs_by_subset = list_sots_pairs(self.sots_root)
            if not pairs_by_subset:
                self.error.emit(
                    f"Không tìm thấy cặp ảnh nào trong {self.sots_root}.\n"
                    "Kiểm tra cấu trúc: SOTS/{indoor,outdoor}/{hazy,gt}/"
                )
                return

            # Lập danh sách task
            tasks = []
            for subset in self.subsets:
                pairs = pairs_by_subset.get(subset, [])
                if self.limit > 0:
                    pairs = pairs[: self.limit]
                for algo in self.algorithms:
                    for hazy_p, gt_p in pairs:
                        tasks.append((subset, algo, hazy_p, gt_p))

            total = len(tasks)
            if total == 0:
                self.error.emit("Không có task nào để chạy. Hãy chọn ít nhất 1 subset và 1 thuật toán.")
                return

            records: list[dict] = []
            for i, (subset, algo, hazy_p, gt_p) in enumerate(tasks):
                if self._abort:
                    break
                name = Path(hazy_p).name
                self.progress.emit(i + 1, total, f"{subset} / {algo} / {name}")

                try:
                    hazy = load_image(hazy_p, as_float=True)
                    gt = load_image(gt_p, as_float=True)

                    t0 = time.perf_counter()
                    pred = ALGORITHMS[algo](hazy)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    records.append(
                        {
                            "subset": subset,
                            "algorithm": algo,
                            "image": name,
                            "psnr": compute_psnr(pred, gt),
                            "ssim": compute_ssim(pred, gt),
                            "contrast": compute_contrast(pred),
                            "colorfulness": compute_colorfulness(pred),
                            "avg_gradient": compute_avg_gradient(pred),
                            "sharpness": compute_laplacian_var(pred),
                            "entropy": compute_entropy(pred),
                            "edge_vis": compute_edge_visibility(pred),
                            "haze_index": compute_fade_like(pred),
                            "time_ms": elapsed_ms,
                        }
                    )
                except Exception as e:  # pragma: no cover
                    # Bỏ qua ảnh lỗi, tiếp tục chạy
                    print(f"[!] Lỗi với {hazy_p}: {e}")

            self.finished_ok.emit(records)
        except Exception as e:  # pragma: no cover
            self.error.emit(str(e))


# ========================================================================= #
#                               BenchmarkTab
# ========================================================================= #
class BenchmarkTab(QtWidgets.QWidget):
    """Giao diện đánh giá dataset."""

    METRIC_COLUMNS = [
        ("psnr", "PSNR ↑"),
        ("ssim", "SSIM ↑"),
        ("contrast", "Contrast ↑"),
        ("colorfulness", "Colorful ↑"),
        ("avg_gradient", "AvgGrad ↑"),
        ("sharpness", "Sharp ↑"),
        ("entropy", "Entropy ↑"),
        ("edge_vis", "EdgeVis ↑"),
        ("haze_index", "Haze ↓"),
        ("time_ms", "Time (ms)"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.records: list[dict] = []
        self.worker: BenchmarkWorker | None = None
        self._build_ui()

    # ---------------------------- UI ---------------------------- #
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(10)

        # --- Cấu hình chạy ---
        cfg_group = QtWidgets.QGroupBox("1. Cấu hình đánh giá")
        cfg_layout = QtWidgets.QGridLayout(cfg_group)

        # Thư mục SOTS
        cfg_layout.addWidget(QtWidgets.QLabel("Thư mục SOTS:"), 0, 0)
        self.ed_root = QtWidgets.QLineEdit("data/SOTS")
        cfg_layout.addWidget(self.ed_root, 0, 1)
        btn_browse = QtWidgets.QPushButton("📁 Duyệt…")
        btn_browse.clicked.connect(self._on_browse)
        cfg_layout.addWidget(btn_browse, 0, 2)

        # Subsets
        cfg_layout.addWidget(QtWidgets.QLabel("Subset:"), 1, 0)
        subset_box = QtWidgets.QWidget()
        subset_h = QtWidgets.QHBoxLayout(subset_box)
        subset_h.setContentsMargins(0, 0, 0, 0)
        self.cb_indoor = QtWidgets.QCheckBox("indoor")
        self.cb_outdoor = QtWidgets.QCheckBox("outdoor")
        self.cb_indoor.setChecked(True)
        self.cb_outdoor.setChecked(True)
        subset_h.addWidget(self.cb_indoor)
        subset_h.addWidget(self.cb_outdoor)
        subset_h.addStretch()
        cfg_layout.addWidget(subset_box, 1, 1, 1, 2)

        # Thuật toán
        cfg_layout.addWidget(QtWidgets.QLabel("Thuật toán:"), 2, 0)
        algo_box = QtWidgets.QWidget()
        algo_h = QtWidgets.QHBoxLayout(algo_box)
        algo_h.setContentsMargins(0, 0, 0, 0)
        self.algo_checks: dict[str, QtWidgets.QCheckBox] = {}
        for a in ALGORITHMS.keys():
            cb = QtWidgets.QCheckBox(a)
            cb.setChecked(True)
            self.algo_checks[a] = cb
            algo_h.addWidget(cb)
        algo_h.addStretch()
        cfg_layout.addWidget(algo_box, 2, 1, 1, 2)

        # Limit
        cfg_layout.addWidget(QtWidgets.QLabel("Số ảnh mỗi subset (0 = tất cả):"), 3, 0)
        self.spin_limit = QtWidgets.QSpinBox()
        self.spin_limit.setRange(0, 100000)
        self.spin_limit.setValue(10)
        cfg_layout.addWidget(self.spin_limit, 3, 1)

        root.addWidget(cfg_group)

        # --- Nút điều khiển ---
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("▶ Bắt đầu đánh giá")
        self.btn_run.setStyleSheet(
            "QPushButton{background:#2d7cff; color:white; padding:8px 18px;"
            "font-weight:600;} QPushButton:disabled{background:#555;}"
        )
        self.btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self.btn_run)

        self.btn_stop = QtWidgets.QPushButton("■ Dừng")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        self.btn_export = QtWidgets.QPushButton("💾 Export CSV…")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        btn_row.addWidget(self.btn_export)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # --- Progress ---
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("%p%  (%v / %m)")
        root.addWidget(self.progress)

        self.lbl_status = QtWidgets.QLabel("Chưa chạy.")
        self.lbl_status.setStyleSheet("color:#666; font-family:Consolas; font-size:11px;")
        root.addWidget(self.lbl_status)

        # --- Kết quả ---
        root.addWidget(QtWidgets.QLabel("<b>2. Kết quả trung bình</b>"))
        self.tbl_summary = self._make_table(
            ["Subset", "Thuật toán", "Số ảnh"]
            + [h for _, h in self.METRIC_COLUMNS]
        )
        root.addWidget(self.tbl_summary, stretch=1)

        # Thông tin nhanh
        self.lbl_best = QtWidgets.QLabel("")
        self.lbl_best.setStyleSheet("padding:6px; background:#fff8e1; border:1px solid #ffd54f;")
        self.lbl_best.setWordWrap(True)
        self.lbl_best.hide()
        root.addWidget(self.lbl_best)

    def _make_table(self, headers: List[str]) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.horizontalHeader().setStretchLastSection(True)
        t.setStyleSheet(
            "QTableWidget{font-family:Consolas; font-size:11px;}"
            "QHeaderView::section{background:#e8eef7; padding:4px; font-weight:600;}"
        )
        return t

    # ---------------------------- Actions ---------------------------- #
    def _on_browse(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Chọn thư mục SOTS (chứa indoor/ và outdoor/)", self.ed_root.text()
        )
        if path:
            self.ed_root.setText(path)

    def _on_run(self) -> None:
        root = self.ed_root.text().strip()
        if not root:
            QtWidgets.QMessageBox.warning(self, "Chú ý", "Hãy chọn thư mục SOTS.")
            return
        if not Path(root).is_dir():
            QtWidgets.QMessageBox.warning(self, "Chú ý", f"Thư mục không tồn tại: {root}")
            return

        subsets = []
        if self.cb_indoor.isChecked():
            subsets.append("indoor")
        if self.cb_outdoor.isChecked():
            subsets.append("outdoor")
        if not subsets:
            QtWidgets.QMessageBox.warning(self, "Chú ý", "Hãy chọn ít nhất 1 subset.")
            return

        algos = [a for a, cb in self.algo_checks.items() if cb.isChecked()]
        if not algos:
            QtWidgets.QMessageBox.warning(self, "Chú ý", "Hãy chọn ít nhất 1 thuật toán.")
            return

        limit = self.spin_limit.value()

        # Reset UI
        self.records = []
        self.tbl_summary.setRowCount(0)
        self.lbl_best.hide()
        self.progress.setValue(0)
        self.btn_export.setEnabled(False)

        self.worker = BenchmarkWorker(root, algos, subsets, limit)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Đang chạy…")

    def _on_stop(self) -> None:
        if self.worker is not None:
            self.worker.abort()
            self.lbl_status.setText("Đang dừng…")

    def _on_progress(self, cur: int, total: int, msg: str) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(cur)
        self.lbl_status.setText(f"[{cur}/{total}] {msg}")

    def _on_done(self, records: list) -> None:
        self.records = records
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(bool(records))
        self.lbl_status.setText(f"Xong. Đã xử lý {len(records)} (subset × algo × ảnh).")
        self._fill_summary_table()
        self._highlight_best()

    def _on_error(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Lỗi benchmark", msg)
        self.lbl_status.setText(f"Lỗi: {msg}")

    # ---------------------------- Rendering ---------------------------- #
    def _fill_summary_table(self) -> None:
        if not self.records:
            return
        df = pd.DataFrame(self.records)

        agg_cols = [k for k, _ in self.METRIC_COLUMNS]
        grouped = (
            df.groupby(["subset", "algorithm"])[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        counts = (
            df.groupby(["subset", "algorithm"])["image"].count().reset_index()
            .rename(columns={"image": "n"})
        )
        merged = grouped.merge(counts, on=["subset", "algorithm"])

        self.tbl_summary.setRowCount(len(merged))
        for i, row in merged.iterrows():
            self.tbl_summary.setItem(i, 0, QtWidgets.QTableWidgetItem(str(row["subset"])))
            self.tbl_summary.setItem(i, 1, QtWidgets.QTableWidgetItem(str(row["algorithm"])))
            self.tbl_summary.setItem(i, 2, QtWidgets.QTableWidgetItem(str(int(row["n"]))))
            for j, (key, _) in enumerate(self.METRIC_COLUMNS):
                val = row[key]
                txt = f"{val:.3f}" if abs(val) < 1000 else f"{val:.1f}"
                self.tbl_summary.setItem(i, 3 + j, QtWidgets.QTableWidgetItem(txt))

        self.tbl_summary.resizeColumnsToContents()

    def _highlight_best(self) -> None:
        """Tóm tắt thuật toán tốt nhất theo PSNR cho mỗi subset."""
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        msgs = []
        for subset in df["subset"].unique():
            sub = df[df["subset"] == subset]
            avg_by_algo = sub.groupby("algorithm")[["psnr", "ssim", "time_ms"]].mean()
            best_psnr = avg_by_algo["psnr"].idxmax()
            best_ssim = avg_by_algo["ssim"].idxmax()
            fastest = avg_by_algo["time_ms"].idxmin()
            msgs.append(
                f"<b>{subset}</b>: "
                f"PSNR cao nhất = <b>{best_psnr}</b> "
                f"({avg_by_algo.loc[best_psnr,'psnr']:.2f} dB) · "
                f"SSIM cao nhất = <b>{best_ssim}</b> "
                f"({avg_by_algo.loc[best_ssim,'ssim']:.4f}) · "
                f"Nhanh nhất = <b>{fastest}</b> "
                f"({avg_by_algo.loc[fastest,'time_ms']:.1f} ms)"
            )
        self.lbl_best.setText("🏆 " + "<br>🏆 ".join(msgs))
        self.lbl_best.show()

    def _on_export(self) -> None:
        if not self.records:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu CSV chi tiết", "benchmark_detail.csv", "CSV (*.csv)"
        )
        if not path:
            return
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)

        # Thêm summary.csv cạnh file chi tiết
        summary_path = str(Path(path).with_name("benchmark_summary.csv"))
        agg_cols = [k for k, _ in self.METRIC_COLUMNS]
        df.groupby(["subset", "algorithm"])[agg_cols].mean(numeric_only=True).to_csv(
            summary_path
        )
        QtWidgets.QMessageBox.information(
            self,
            "Đã lưu",
            f"Chi tiết: {path}\nTổng hợp: {summary_path}",
        )