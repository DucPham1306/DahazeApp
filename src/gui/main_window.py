from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..algorithms import ALGORITHMS
from ..metrics import compute_all_no_reference
from ..utils import load_image, save_image, to_uint8
from .benchmark_tab import BenchmarkTab
from .style import (
    APP_QSS,
    IMAGE_VIEW_QSS,
    INFO_CARD_QSS,
    LEGEND_CARD_QSS,
    PRIMARY,
    SUCCESS,
    DANGER,
)


class DehazeWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(np.ndarray, float)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, algo_name: str, img: np.ndarray, params: dict):
        super().__init__()
        self.algo_name = algo_name
        self.img = img
        self.params = params

    def run(self):
        try:
            fn = ALGORITHMS[self.algo_name]
            t0 = time.perf_counter()
            out = fn(self.img, **self.params)
            dt = (time.perf_counter() - t0) * 1000
            self.finished_signal.emit(out, dt)
        except Exception as e:
            self.error_signal.emit(str(e))


class ImageView(QtWidgets.QLabel):
    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setStyleSheet(IMAGE_VIEW_QSS)
        self.setText(placeholder)
        self._pixmap: Optional[QtGui.QPixmap] = None

    def set_image(self, img: np.ndarray) -> None:
        img_u8 = np.ascontiguousarray(to_uint8(img))
        h, w, _ = img_u8.shape
        qimg = QtGui.QImage(img_u8.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(qimg.copy())
        self._rescale()

    def clear_image(self, placeholder: str) -> None:
        self._pixmap = None
        self.setText(placeholder)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class SingleImageTab(QtWidgets.QWidget):
    HIGHER_IS_BETTER = {
        "Contrast (RMS)", "Saturation", "Colorfulness",
        "Avg Gradient", "Sharpness (LapVar)", "Entropy (Shannon)",
        "Edge Visibility",
    }
    LOWER_IS_BETTER = {"Haze Index", "NIQE"}

    PLACEHOLDER_BEFORE = "🌫️\n\nẢNH CÓ SƯƠNG MÙ\nMở ảnh hoặc kéo-thả vào đây"
    PLACEHOLDER_AFTER = "✨\n\nẢNH ĐÃ KHỬ SƯƠNG MÙ\nNhấn ▶ để xử lý"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hazy_img: Optional[np.ndarray] = None
        self.result_img: Optional[np.ndarray] = None
        self.worker: Optional[DehazeWorker] = None
        self._param_values: dict = {}
        self._build_ui()
        self.setAcceptDrops(True)

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_center_panel(), stretch=2)
        root.addWidget(self._build_right_panel())

    def _build_left_panel(self) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        wrapper.setFixedWidth(310)
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        file_box = QtWidgets.QGroupBox("Tệp ảnh")
        file_layout = QtWidgets.QVBoxLayout(file_box)
        file_layout.setSpacing(8)
        btn_open = QtWidgets.QPushButton("📂  Mở ảnh hazy…")
        btn_open.setProperty("variant", "primary")
        btn_open.clicked.connect(self.on_open_hazy)
        file_layout.addWidget(btn_open)

        self.btn_save = QtWidgets.QPushButton("💾  Lưu kết quả…")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        file_layout.addWidget(self.btn_save)

        self.lbl_filename = QtWidgets.QLabel("Chưa chọn ảnh")
        self.lbl_filename.setStyleSheet("color:#64748b; font-size:11px;")
        self.lbl_filename.setWordWrap(True)
        file_layout.addWidget(self.lbl_filename)
        layout.addWidget(file_box)

        algo_box = QtWidgets.QGroupBox("Thuật toán")
        algo_layout = QtWidgets.QVBoxLayout(algo_box)
        algo_layout.setSpacing(10)
        self.cb_algo = QtWidgets.QComboBox()
        self.cb_algo.addItems(list(ALGORITHMS.keys()))
        self.cb_algo.currentTextChanged.connect(self._update_param_panel)
        algo_layout.addWidget(self.cb_algo)

        self.param_widget = QtWidgets.QWidget()
        self.param_layout = QtWidgets.QFormLayout(self.param_widget)
        self.param_layout.setHorizontalSpacing(10)
        self.param_layout.setVerticalSpacing(8)
        self.param_layout.setContentsMargins(0, 4, 0, 0)
        algo_layout.addWidget(self.param_widget)
        layout.addWidget(algo_box)

        run_box = QtWidgets.QGroupBox("Xử lý")
        run_layout = QtWidgets.QVBoxLayout(run_box)
        run_layout.setSpacing(8)
        self.btn_run = QtWidgets.QPushButton("▶  Khử sương mù")
        self.btn_run.setProperty("variant", "success")
        self.btn_run.clicked.connect(self.on_run)
        run_layout.addWidget(self.btn_run)

        self.time_label = QtWidgets.QLabel("⏱  Thời gian xử lý: —")
        self.time_label.setStyleSheet(INFO_CARD_QSS)
        run_layout.addWidget(self.time_label)
        layout.addWidget(run_box)

        layout.addStretch()
        self._update_param_panel(self.cb_algo.currentText())
        return wrapper

    def _build_center_panel(self) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self.view_before = ImageView(self.PLACEHOLDER_BEFORE)
        self.view_after = ImageView(self.PLACEHOLDER_AFTER)

        layout.addWidget(self._labeled_view("BEFORE", self.view_before), 0, 0)
        layout.addWidget(self._labeled_view("AFTER", self.view_after), 0, 1)
        return wrapper

    def _labeled_view(self, title: str, view: ImageView) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        lab = QtWidgets.QLabel(title)
        lab.setStyleSheet(
            f"color:{PRIMARY}; font-weight:700; letter-spacing:1px; font-size:11px;"
        )
        v.addWidget(lab)
        v.addWidget(view)
        return w

    def _build_right_panel(self) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        wrapper.setFixedWidth(360)
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("📊 Chỉ số chất lượng")
        title.setStyleSheet("font-weight:700; font-size:14px; padding:2px 4px;")
        layout.addWidget(title)

        self.metric_table = self._make_metrics_table()
        layout.addWidget(self.metric_table, stretch=1)

        legend = QtWidgets.QLabel(
            "<b>▲ Xanh</b>: tốt hơn &nbsp;·&nbsp; <b>▼ Đỏ</b>: kém hơn<br>"
            "Sau khi khử sương: Contrast / Saturation / Colorfulness / "
            "Sharpness / Entropy thường <b>tăng</b>; Haze Index, NIQE thường <b>giảm</b>."
        )
        legend.setWordWrap(True)
        legend.setStyleSheet(LEGEND_CARD_QSS)
        layout.addWidget(legend)
        return wrapper

    def _make_metrics_table(self) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, 3)
        t.setHorizontalHeaderLabels(["Chỉ số", "Before", "After"])
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        t.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents
        )
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        t.setAlternatingRowColors(True)
        t.setShowGrid(False)
        return t

    def _update_metrics_table(self) -> None:
        t = self.metric_table
        t.setRowCount(0)
        if self.hazy_img is None:
            return

        before = compute_all_no_reference(self.hazy_img)
        after = (
            compute_all_no_reference(self.result_img)
            if self.result_img is not None
            else {k: None for k in before}
        )

        for name, b_val in before.items():
            a_val = after.get(name)
            row = t.rowCount()
            t.insertRow(row)

            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.DemiBold))
            t.setItem(row, 0, name_item)

            t.setItem(row, 1, QtWidgets.QTableWidgetItem(self._fmt(b_val)))

            item_a = QtWidgets.QTableWidgetItem(self._fmt(a_val))
            if a_val is not None and b_val is not None:
                delta = a_val - b_val
                color = self._delta_color(name, delta)
                if color is not None:
                    item_a.setForeground(QtGui.QBrush(QtGui.QColor(color)))
                    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
                    item_a.setText(f"{arrow} {self._fmt(a_val)}")
                    item_a.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
            t.setItem(row, 2, item_a)

        t.resizeRowsToContents()

    @staticmethod
    def _fmt(v) -> str:
        if v is None:
            return "—"
        if abs(v) >= 100:
            return f"{v:.2f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4f}"

    def _delta_color(self, name: str, delta: float) -> Optional[str]:
        if abs(delta) < 1e-6:
            return None
        if name in self.HIGHER_IS_BETTER:
            return SUCCESS if delta > 0 else DANGER
        if name in self.LOWER_IS_BETTER:
            return SUCCESS if delta < 0 else DANGER
        return None

    def _clear_param_layout(self) -> None:
        while self.param_layout.rowCount():
            self.param_layout.removeRow(0)

    def _add_slider(self, name, label, lo, hi, default, step=0.01):
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        steps = int(round((hi - lo) / step))
        slider.setRange(0, steps)
        slider.setValue(int(round((default - lo) / step)))
        val_lbl = QtWidgets.QLabel(f"{default:.3g}")
        val_lbl.setMinimumWidth(48)
        val_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        val_lbl.setStyleSheet(
            f"color:{PRIMARY}; font-family:Consolas; font-weight:600;"
        )

        def on_change(v):
            fv = lo + v * step
            val_lbl.setText(f"{fv:.3g}")
            self._param_values[name] = fv

        slider.valueChanged.connect(on_change)
        self._param_values[name] = default

        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, stretch=1)
        h.addWidget(val_lbl)
        self.param_layout.addRow(label, row)

    def _add_spin_int(self, name, label, lo, hi, default, step=1):
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        spin.valueChanged.connect(
            lambda v, k=name: self._param_values.__setitem__(k, v)
        )
        self._param_values[name] = default
        self.param_layout.addRow(label, spin)

    def _update_param_panel(self, algo: str) -> None:
        self._clear_param_layout()
        self._param_values = {}
        if algo == "DCP":
            self._add_spin_int("patch_size", "Patch size", 3, 31, 15, 2)
            self._add_slider("omega", "Omega", 0.5, 1.0, 0.95, 0.01)
            self._add_slider("t0", "t₀ (min trans.)", 0.01, 0.3, 0.1, 0.01)
            self._add_spin_int("guided_radius", "Guided radius", 10, 120, 60, 5)
        elif algo == "CLAHE":
            self._add_slider("clip_limit", "Clip limit", 1.0, 10.0, 2.0, 0.1)
            self._add_spin_int("tile_grid_size", "Tile grid", 2, 32, 8, 1)
        elif algo == "CAP":
            self._add_slider("beta", "Beta (scattering)", 0.1, 3.0, 1.0, 0.05)
            self._add_spin_int("min_filter_size", "Min filter", 3, 31, 15, 2)
            self._add_slider("t0", "t₀", 0.01, 0.3, 0.1, 0.01)
        elif algo == "Hybrid":
            self._add_slider("clip_limit", "CLAHE clip", 1.0, 5.0, 1.5, 0.1)
            self._add_spin_int("patch_size", "DCP patch", 3, 31, 15, 2)
            self._add_slider("omega", "Omega", 0.5, 1.0, 0.95, 0.01)
            self._add_slider("blend", "Blend w/ orig", 0.0, 0.5, 0.0, 0.02)

    def on_open_hazy(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn ảnh hazy", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if path:
            self._load_hazy(path)

    def _load_hazy(self, path: str) -> None:
        try:
            self.hazy_img = load_image(path, as_float=True)
            self.view_before.set_image(self.hazy_img)
            self.view_after.clear_image(self.PLACEHOLDER_AFTER)
            self.result_img = None
            self.btn_save.setEnabled(False)
            self.time_label.setText("⏱  Thời gian xử lý: —")
            self.lbl_filename.setText(f"📷 {Path(path).name}")
            self._update_metrics_table()
            mw = self.window()
            if hasattr(mw, "statusBar"):
                mw.statusBar().showMessage(f"Đã tải: {Path(path).name}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Lỗi", str(e))

    def on_save(self) -> None:
        if self.result_img is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu ảnh kết quả", "result.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            save_image(path, self.result_img)
            mw = self.window()
            if hasattr(mw, "statusBar"):
                mw.statusBar().showMessage(f"Đã lưu: {path}")

    def on_run(self) -> None:
        if self.hazy_img is None:
            QtWidgets.QMessageBox.information(
                self, "Chú ý", "Hãy mở một ảnh trước."
            )
            return
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳  Đang xử lý…")

        algo = self.cb_algo.currentText()
        self.worker = DehazeWorker(algo, self.hazy_img, dict(self._param_values))
        self.worker.finished_signal.connect(self._on_done)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _on_done(self, result: np.ndarray, dt_ms: float) -> None:
        self.result_img = result
        self.view_after.set_image(result)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Khử sương mù")
        self.btn_save.setEnabled(True)
        self.time_label.setText(f"⏱  Thời gian xử lý: {dt_ms:.1f} ms")
        self._update_metrics_table()

    def _on_error(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Khử sương mù")
        QtWidgets.QMessageBox.critical(self, "Lỗi xử lý", msg)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self._load_hazy(path)
                break


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dehaze Studio — Ứng dụng khử sương mù ảnh số")
        self.resize(1440, 880)
        self.setMinimumSize(1180, 720)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_header())

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tab_single = SingleImageTab(self)
        self.tab_bench = BenchmarkTab(self)
        self.tabs.addTab(self.tab_single, "🖼  Xử lý ảnh đơn")
        self.tabs.addTab(self.tab_bench, "📊  Đánh giá Dataset")
        v.addWidget(self.tabs, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Sẵn sàng.")

    def _build_header(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(58)
        bar.setStyleSheet(f"background:{PRIMARY}; color:white;")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(20, 8, 20, 8)
        h.setSpacing(12)

        title = QtWidgets.QLabel("🌫️  Dehaze Studio")
        title.setStyleSheet(
            "color:white; font-size:18px; font-weight:700; letter-spacing:0.5px;"
        )
        h.addWidget(title)

        subtitle = QtWidgets.QLabel("DCP · CLAHE · CAP · Hybrid")
        subtitle.setStyleSheet(
            "color:rgba(255,255,255,0.85); font-size:12px; padding-left:6px;"
        )
        h.addWidget(subtitle)
        h.addStretch()

        version = QtWidgets.QLabel("v1.1")
        version.setStyleSheet(
            "color:rgba(255,255,255,0.85); font-size:11px;"
            "background:rgba(255,255,255,0.15); padding:4px 10px; border-radius:10px;"
        )
        h.addWidget(version)
        return bar


def run() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    w = MainWindow()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run())
