from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

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
from ..utils import Dataset, detect_dataset, load_custom_dataset, load_image
from .style import BEST_CARD_QSS, INFO_CARD_QSS


class CustomDatasetDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thêm dataset custom")
        self.resize(560, 200)
        form = QtWidgets.QFormLayout(self)
        form.setSpacing(10)
        form.setContentsMargins(16, 16, 16, 16)

        self.ed_name = QtWidgets.QLineEdit("MyDataset")
        form.addRow("Tên dataset:", self.ed_name)

        self.ed_hazy = QtWidgets.QLineEdit()
        btn_h = QtWidgets.QPushButton("📁  Browse")
        btn_h.clicked.connect(
            lambda: self._pick(self.ed_hazy, "Chọn thư mục ảnh HAZY")
        )
        form.addRow("Folder hazy:", self._hbox(self.ed_hazy, btn_h))

        self.ed_gt = QtWidgets.QLineEdit()
        btn_g = QtWidgets.QPushButton("📁  Browse")
        btn_g.clicked.connect(
            lambda: self._pick(self.ed_gt, "Chọn thư mục ảnh GROUND TRUTH")
        )
        form.addRow("Folder gt:", self._hbox(self.ed_gt, btn_g))

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.button(QtWidgets.QDialogButtonBox.Ok).setProperty("variant", "primary")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    @staticmethod
    def _hbox(edit, btn):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(edit, 1)
        h.addWidget(btn)
        return w

    def _pick(self, edit, title):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, title, edit.text())
        if p:
            edit.setText(p)

    def get_values(self):
        return (
            self.ed_name.text().strip(),
            self.ed_hazy.text().strip(),
            self.ed_gt.text().strip(),
        )


class BenchmarkWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished_ok = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        datasets: List[Dataset],
        algorithms: List[str],
        subsets_by_ds: dict[str, List[str]],
        limit: int,
    ):
        super().__init__()
        self.datasets = datasets
        self.algorithms = algorithms
        self.subsets_by_ds = subsets_by_ds
        self.limit = limit
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            tasks = []
            for ds in self.datasets:
                subs = self.subsets_by_ds.get(ds.name) or ds.subsets
                for sub, hz, gt in ds.filter(subs, self.limit):
                    for algo in self.algorithms:
                        tasks.append((ds.name, sub, algo, hz, gt))

            total = len(tasks)
            if total == 0:
                self.error.emit(
                    "Không có task nào để chạy. Kiểm tra dataset + subset + thuật toán."
                )
                return

            records: list[dict] = []
            cache_img: dict[str, tuple] = {}
            for i, (ds_name, subset, algo, hz, gt) in enumerate(tasks):
                if self._abort:
                    break
                name = Path(hz).name
                self.progress.emit(
                    i + 1, total, f"{ds_name} / {subset} / {algo} / {name}"
                )
                try:
                    pair = cache_img.get(hz)
                    if pair is None:
                        pair = (
                            load_image(hz, as_float=True),
                            load_image(gt, as_float=True),
                        )
                        if len(cache_img) > 4:
                            cache_img.clear()
                        cache_img[hz] = pair
                    hazy, gtimg = pair

                    t0 = time.perf_counter()
                    pred = ALGORITHMS[algo](hazy)
                    dt_ms = (time.perf_counter() - t0) * 1000

                    records.append({
                        "dataset": ds_name,
                        "subset": subset,
                        "algorithm": algo,
                        "image": name,
                        "psnr": compute_psnr(pred, gtimg),
                        "ssim": compute_ssim(pred, gtimg),
                        "contrast": compute_contrast(pred),
                        "colorfulness": compute_colorfulness(pred),
                        "avg_gradient": compute_avg_gradient(pred),
                        "sharpness": compute_laplacian_var(pred),
                        "entropy": compute_entropy(pred),
                        "edge_vis": compute_edge_visibility(pred),
                        "haze_index": compute_fade_like(pred),
                        "time_ms": dt_ms,
                    })
                except Exception as e:
                    print(f"[!] Lỗi {hz}: {e}")

            self.finished_ok.emit(records)
        except Exception as e:
            self.error.emit(str(e))


class BenchmarkTab(QtWidgets.QWidget):
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

    DS_COLORS = ["#ffffff", "#eff6ff", "#fef2f2", "#f0fdf4", "#fefce8", "#faf5ff"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.datasets: List[Dataset] = []
        self.records: list[dict] = []
        self.worker: BenchmarkWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(self._build_dataset_box())
        root.addWidget(self._build_config_box())
        root.addLayout(self._build_action_row())
        root.addWidget(self.progress)

        self.lbl_status = QtWidgets.QLabel("Thêm dataset để bắt đầu.")
        self.lbl_status.setStyleSheet(INFO_CARD_QSS)
        root.addWidget(self.lbl_status)

        results_label = QtWidgets.QLabel("📈 Kết quả trung bình")
        results_label.setStyleSheet(
            "font-weight:700; font-size:18px; padding:10px 4px 4px 4px;"
        )
        root.addWidget(results_label)

        self.tbl_summary = self._make_table(
            ["Dataset", "Subset", "Thuật toán", "n"]
            + [h for _, h in self.METRIC_COLUMNS]
        )
        # Làm bảng kết quả trung bình to và dễ đọc hơn
        summary_font = QtGui.QFont("Segoe UI", 11)
        self.tbl_summary.setFont(summary_font)
        header_font = QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold)
        self.tbl_summary.horizontalHeader().setFont(header_font)
        self.tbl_summary.horizontalHeader().setDefaultSectionSize(105)
        self.tbl_summary.horizontalHeader().setMinimumSectionSize(80)
        self.tbl_summary.verticalHeader().setDefaultSectionSize(34)
        self.tbl_summary.setMinimumHeight(380)
        root.addWidget(self.tbl_summary, stretch=3)

        self.lbl_best = QtWidgets.QLabel("")
        self.lbl_best.setStyleSheet(BEST_CARD_QSS)
        self.lbl_best.setWordWrap(True)
        self.lbl_best.hide()
        root.addWidget(self.lbl_best)

    def _build_dataset_box(self) -> QtWidgets.QGroupBox:
        ds_group = QtWidgets.QGroupBox("1. Dataset (có thể thêm nhiều)")
        ds_layout = QtWidgets.QVBoxLayout(ds_group)
        ds_layout.setSpacing(8)

        tb = QtWidgets.QHBoxLayout()
        tb.setSpacing(6)
        btn_add_auto = QtWidgets.QPushButton("➕  Auto-detect")
        btn_add_auto.setProperty("variant", "primary")
        btn_add_auto.setToolTip(
            "Tự động nhận cấu trúc SOTS / O-HAZE / I-HAZE / NH-HAZE / Dense-Haze"
        )
        btn_add_auto.clicked.connect(self._on_add_auto)
        tb.addWidget(btn_add_auto)

        btn_add_custom = QtWidgets.QPushButton("➕  Custom folder")
        btn_add_custom.setToolTip("Chỉ định tay folder hazy + folder gt")
        btn_add_custom.clicked.connect(self._on_add_custom)
        tb.addWidget(btn_add_custom)

        btn_remove = QtWidgets.QPushButton("➖  Xóa")
        btn_remove.clicked.connect(self._on_remove_dataset)
        tb.addWidget(btn_remove)

        btn_clear = QtWidgets.QPushButton("🗑  Xóa tất cả")
        btn_clear.setProperty("variant", "danger")
        btn_clear.clicked.connect(self._on_clear_datasets)
        tb.addWidget(btn_clear)

        tb.addStretch()
        ds_layout.addLayout(tb)

        self.tbl_datasets = QtWidgets.QTableWidget(0, 4)
        self.tbl_datasets.setHorizontalHeaderLabels(
            ["Tên dataset", "Số ảnh", "Subsets", "Đường dẫn"]
        )
        self.tbl_datasets.verticalHeader().setVisible(False)
        self.tbl_datasets.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.tbl_datasets.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.tbl_datasets.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.tbl_datasets.horizontalHeader().setStretchLastSection(True)
        self.tbl_datasets.setMaximumHeight(160)
        self.tbl_datasets.setShowGrid(False)
        ds_layout.addWidget(self.tbl_datasets)
        return ds_group

    def _build_config_box(self) -> QtWidgets.QGroupBox:
        cfg_group = QtWidgets.QGroupBox("2. Cấu hình chạy")
        cfg_layout = QtWidgets.QGridLayout(cfg_group)
        cfg_layout.setSpacing(10)

        cfg_layout.addWidget(QtWidgets.QLabel("Thuật toán:"), 0, 0)
        algo_box = QtWidgets.QWidget()
        algo_h = QtWidgets.QHBoxLayout(algo_box)
        algo_h.setContentsMargins(0, 0, 0, 0)
        algo_h.setSpacing(8)
        self.algo_checks: dict[str, QtWidgets.QCheckBox] = {}
        for a in ALGORITHMS.keys():
            cb = QtWidgets.QCheckBox(a)
            cb.setChecked(True)
            self.algo_checks[a] = cb
            algo_h.addWidget(cb)
        algo_h.addStretch()
        cfg_layout.addWidget(algo_box, 0, 1)

        cfg_layout.addWidget(
            QtWidgets.QLabel("Số ảnh mỗi subset (0 = tất cả):"), 1, 0
        )
        self.spin_limit = QtWidgets.QSpinBox()
        self.spin_limit.setRange(0, 100000)
        self.spin_limit.setValue(10)
        self.spin_limit.setMaximumWidth(120)
        cfg_layout.addWidget(self.spin_limit, 1, 1, alignment=QtCore.Qt.AlignLeft)
        return cfg_group

    def _build_action_row(self) -> QtWidgets.QHBoxLayout:
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_run = QtWidgets.QPushButton("▶  Bắt đầu đánh giá")
        self.btn_run.setProperty("variant", "success")
        self.btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self.btn_run)

        self.btn_stop = QtWidgets.QPushButton("■  Dừng")
        self.btn_stop.setProperty("variant", "danger")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        self.btn_export = QtWidgets.QPushButton("💾  Export CSV…")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        btn_row.addWidget(self.btn_export)

        btn_row.addStretch()

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("%p%  (%v / %m)")
        self.progress.setMinimumWidth(280)
        return btn_row

    def _make_table(self, headers: List[str]) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.horizontalHeader().setStretchLastSection(True)
        t.setShowGrid(False)
        return t

    def _on_add_auto(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Chọn thư mục gốc của dataset", ""
        )
        if not path:
            return
        try:
            ds = detect_dataset(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Không phát hiện được", str(e))
            return
        self._add_dataset(ds, path)

    def _on_add_custom(self) -> None:
        dlg = CustomDatasetDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        name, hazy, gt = dlg.get_values()
        if not (name and hazy and gt):
            return
        try:
            ds = load_custom_dataset(hazy, gt, name=name)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Lỗi", str(e))
            return
        self._add_dataset(ds, f"{hazy}  +  {gt}")

    def _add_dataset(self, ds: Dataset, path_display: str) -> None:
        existing = {d.name for d in self.datasets}
        if ds.name in existing:
            i = 2
            while f"{ds.name}_{i}" in existing:
                i += 1
            ds = Dataset(f"{ds.name}_{i}", ds.pairs)

        self.datasets.append(ds)
        row = self.tbl_datasets.rowCount()
        self.tbl_datasets.insertRow(row)
        for col, val in enumerate([
            ds.name, str(ds.n_images), ", ".join(ds.subsets), path_display,
        ]):
            self.tbl_datasets.setItem(row, col, QtWidgets.QTableWidgetItem(val))
        self.tbl_datasets.resizeColumnsToContents()
        self._update_status()

    def _on_remove_dataset(self) -> None:
        rows = sorted(
            {idx.row() for idx in self.tbl_datasets.selectedIndexes()},
            reverse=True,
        )
        for r in rows:
            del self.datasets[r]
            self.tbl_datasets.removeRow(r)
        self._update_status()

    def _on_clear_datasets(self) -> None:
        self.datasets.clear()
        self.tbl_datasets.setRowCount(0)
        self._update_status()

    def _update_status(self) -> None:
        n_ds = len(self.datasets)
        n_img = sum(d.n_images for d in self.datasets)
        if n_ds == 0:
            self.lbl_status.setText("Chưa có dataset. Nhấn ➕ để thêm.")
        else:
            self.lbl_status.setText(
                f"✓  Đã có {n_ds} dataset, tổng {n_img} ảnh. Sẵn sàng chạy."
            )

    def _on_run(self) -> None:
        if not self.datasets:
            QtWidgets.QMessageBox.warning(
                self, "Chú ý", "Hãy thêm ít nhất 1 dataset."
            )
            return
        algos = [a for a, cb in self.algo_checks.items() if cb.isChecked()]
        if not algos:
            QtWidgets.QMessageBox.warning(
                self, "Chú ý", "Hãy chọn ít nhất 1 thuật toán."
            )
            return

        subsets_by_ds = {d.name: d.subsets for d in self.datasets}
        limit = self.spin_limit.value()

        self.records = []
        self.tbl_summary.setRowCount(0)
        self.lbl_best.hide()
        self.progress.setValue(0)
        self.btn_export.setEnabled(False)

        self.worker = BenchmarkWorker(
            list(self.datasets), algos, subsets_by_ds, limit
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("⏳  Đang chạy…")

    def _on_stop(self) -> None:
        if self.worker is not None:
            self.worker.abort()
            self.lbl_status.setText("⏹  Đang dừng…")

    def _on_progress(self, cur: int, total: int, msg: str) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(cur)
        self.lbl_status.setText(f"[{cur}/{total}]  {msg}")

    def _on_done(self, records: list) -> None:
        self.records = records
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(bool(records))
        self.lbl_status.setText(f"✓  Hoàn tất. Đã xử lý {len(records)} task.")
        self._fill_summary_table()
        self._highlight_best()

    def _on_error(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Lỗi benchmark", msg)
        self.lbl_status.setText(f"✗  Lỗi: {msg}")

    def _fill_summary_table(self) -> None:
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        agg_cols = [k for k, _ in self.METRIC_COLUMNS]
        grouped = (
            df.groupby(["dataset", "subset", "algorithm"])[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        counts = (
            df.groupby(["dataset", "subset", "algorithm"])["image"]
            .count().reset_index().rename(columns={"image": "n"})
        )
        merged = grouped.merge(counts, on=["dataset", "subset", "algorithm"])

        self.tbl_summary.setRowCount(len(merged))
        ds_names = list(merged["dataset"].unique())

        best_per_subset: dict[tuple, dict] = {}
        for (ds, sub), group in merged.groupby(["dataset", "subset"]):
            best = {}
            for k, _ in self.METRIC_COLUMNS:
                if k == "haze_index" or k == "time_ms":
                    best[k] = group[k].idxmin()
                else:
                    best[k] = group[k].idxmax()
            best_per_subset[(ds, sub)] = best

        for i, row in merged.iterrows():
            ds_idx = ds_names.index(row["dataset"])
            bg = QtGui.QColor(self.DS_COLORS[ds_idx % len(self.DS_COLORS)])

            base_vals = [
                row["dataset"], row["subset"], row["algorithm"], str(int(row["n"])),
            ]
            for col, v in enumerate(base_vals):
                it = QtWidgets.QTableWidgetItem(str(v))
                it.setBackground(bg)
                self.tbl_summary.setItem(i, col, it)

            best_for = best_per_subset[(row["dataset"], row["subset"])]
            for j, (key, _) in enumerate(self.METRIC_COLUMNS):
                val = row[key]
                txt = f"{val:.3f}" if abs(val) < 1000 else f"{val:.1f}"
                it = QtWidgets.QTableWidgetItem(txt)
                it.setBackground(bg)
                if best_for.get(key) == i:
                    f = QtGui.QFont()
                    f.setBold(True)
                    it.setFont(f)
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#16a34a")))
                self.tbl_summary.setItem(i, 4 + j, it)

        self.tbl_summary.resizeColumnsToContents()

    def _highlight_best(self) -> None:
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        msgs = []
        for (ds, sub), group in df.groupby(["dataset", "subset"]):
            avg = group.groupby("algorithm")[["psnr", "ssim", "time_ms"]].mean()
            best_psnr = avg["psnr"].idxmax()
            best_ssim = avg["ssim"].idxmax()
            fastest = avg["time_ms"].idxmin()
            msgs.append(
                f"<b>{ds} / {sub}</b> — "
                f"PSNR: <b>{best_psnr}</b> ({avg.loc[best_psnr,'psnr']:.2f} dB) · "
                f"SSIM: <b>{best_ssim}</b> ({avg.loc[best_ssim,'ssim']:.4f}) · "
                f"Nhanh nhất: <b>{fastest}</b> ({avg.loc[fastest,'time_ms']:.1f} ms)"
            )
        self.lbl_best.setText("🏆  " + "<br>🏆  ".join(msgs))
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

        summary_path = str(Path(path).with_name("benchmark_summary.csv"))
        agg_cols = [k for k, _ in self.METRIC_COLUMNS]
        df.groupby(["dataset", "subset", "algorithm"])[agg_cols].mean(
            numeric_only=True
        ).to_csv(summary_path)

        QtWidgets.QMessageBox.information(
            self, "Đã lưu",
            f"Chi tiết: {path}\nTổng hợp: {summary_path}",
        )
