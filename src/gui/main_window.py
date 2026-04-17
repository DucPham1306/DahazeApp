"""
src/gui/main_window.py
----------------------
Cửa sổ chính của ứng dụng. Chia 2 tab:
  - "Xử lý ảnh đơn" (SingleImageTab)
  - "Đánh giá Dataset" (BenchmarkTab - import từ benchmark_tab.py)
"""
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


# ========================================================================= #
#                        Worker chạy thuật toán ở thread riêng
# ========================================================================= #
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
        except Exception as e:  # pragma: no cover
            self.error_signal.emit(str(e))


# ========================================================================= #
#                        Widget hiển thị ảnh (scale auto)
# ========================================================================= #
class ImageView(QtWidgets.QLabel):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(400, 280)
        self.setStyleSheet(
            "background:#222; color:#aaa; border:1px solid #444;"
            "border-radius:6px; font-size:13px;"
        )
        self.setText(f"{title}\n(Chưa có ảnh)")
        self._pixmap: Optional[QtGui.QPixmap] = None

    def set_image(self, img: np.ndarray) -> None:
        img_u8 = to_uint8(img)
        h, w, _ = img_u8.shape
        qimg = QtGui.QImage(img_u8.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(qimg.copy())
        self._rescale()

    def resizeEvent(self, e):  # noqa: N802
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


# ========================================================================= #
#                              SingleImageTab
# ========================================================================= #
class SingleImageTab(QtWidgets.QWidget):
    """Tab xử lý 1 ảnh: chọn file, chọn thuật toán, điều chỉnh tham số,
    xem Before/After, bảng chỉ số no-reference."""

    HIGHER_IS_BETTER = {
        "Contrast (RMS)", "Saturation", "Colorfulness",
        "Avg Gradient", "Sharpness (LapVar)", "Entropy (Shannon)",
        "Edge Visibility",
    }
    LOWER_IS_BETTER = {"Haze Index", "NIQE"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hazy_img: Optional[np.ndarray] = None
        self.result_img: Optional[np.ndarray] = None
        self.worker: Optional[DehazeWorker] = None
        self._build_ui()
        self.setAcceptDrops(True)

    # ---------------------------- UI ---------------------------- #
    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)

        # --- Panel trái: điều khiển ---
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)

        btn_open = QtWidgets.QPushButton("📂 Mở ảnh hazy…")
        btn_open.clicked.connect(self.on_open_hazy)
        left.addWidget(btn_open)

        left.addWidget(QtWidgets.QLabel("Thuật toán:"))
        self.cb_algo = QtWidgets.QComboBox()
        self.cb_algo.addItems(list(ALGORITHMS.keys()))
        self.cb_algo.currentTextChanged.connect(self._update_param_panel)
        left.addWidget(self.cb_algo)

        self.param_group = QtWidgets.QGroupBox("Tham số")
        self.param_layout = QtWidgets.QFormLayout(self.param_group)
        left.addWidget(self.param_group)

        self.btn_run = QtWidgets.QPushButton("▶ Khử sương mù")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#2d7cff; color:white; padding:8px; font-weight:600;}"
            "QPushButton:disabled{background:#555;}"
        )
        left.addWidget(self.btn_run)

        self.btn_save = QtWidgets.QPushButton("💾 Lưu kết quả…")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        left.addWidget(self.btn_save)

        self.time_label = QtWidgets.QLabel("Thời gian xử lý: —")
        self.time_label.setStyleSheet(
            "background:#f0f4ff; padding:6px; border-radius:4px;"
            "border:1px solid #c7d4ff; font-family:Consolas; font-size:12px;"
        )
        left.addWidget(self.time_label)

        left.addStretch()
        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(300)
        root.addWidget(left_w)

        # --- Panel giữa: Before / After ---
        center_w = QtWidgets.QWidget()
        center = QtWidgets.QGridLayout(center_w)
        self.view_before = ImageView("BEFORE (ảnh có sương mù)")
        self.view_after = ImageView("AFTER (đã khử sương mù)")
        center.addWidget(self._wrap("Before", self.view_before), 0, 0)
        center.addWidget(self._wrap("After", self.view_after), 0, 1)
        root.addWidget(center_w, stretch=2)

        # --- Panel phải: bảng chỉ số ---
        right_w = QtWidgets.QWidget()
        right_w.setFixedWidth(340)
        right = QtWidgets.QVBoxLayout(right_w)
        h = QtWidgets.QLabel("📊 Chỉ số chất lượng ảnh")
        h.setStyleSheet("font-weight:700; font-size:14px; padding:4px;")
        right.addWidget(h)
        self.metric_table = self._make_metrics_table()
        right.addWidget(self.metric_table, stretch=1)
        right.addWidget(self._make_metrics_legend())
        root.addWidget(right_w)

        self._update_param_panel(self.cb_algo.currentText())

    def _wrap(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        lab = QtWidgets.QLabel(title)
        lab.setStyleSheet("font-weight:600; padding:2px;")
        v.addWidget(lab)
        v.addWidget(widget)
        return w

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
        t.setStyleSheet(
            "QTableWidget{font-family:Consolas; font-size:11px;}"
            "QHeaderView::section{background:#e8eef7; padding:4px; font-weight:600;}"
        )
        return t

    def _make_metrics_legend(self) -> QtWidgets.QLabel:
        txt = (
            "<span style='font-size:10px; color:#555;'>"
            "▲ <span style='color:#1a8c36;'>xanh</span>: tốt hơn &nbsp;&nbsp;"
            "▼ <span style='color:#c73030;'>đỏ</span>: xấu hơn<br>"
            "Sau khi khử sương mù, <b>Contrast, Saturation, Colorfulness, "
            "Sharpness, Entropy</b> thường tăng; <b>Haze Index, NIQE</b> thường giảm."
            "</span>"
        )
        lab = QtWidgets.QLabel(txt)
        lab.setWordWrap(True)
        lab.setStyleSheet("padding:6px; background:#fafafa; border:1px solid #ddd;")
        return lab

    # ---------------------------- Metric table ---------------------------- #
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

        for name in before.keys():
            b_val = before[name]
            a_val = after.get(name)

            row = t.rowCount()
            t.insertRow(row)
            t.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            t.setItem(row, 1, QtWidgets.QTableWidgetItem(self._fmt(b_val)))

            item_a = QtWidgets.QTableWidgetItem(self._fmt(a_val))
            if a_val is not None and b_val is not None:
                delta = a_val - b_val
                color = self._delta_color(name, delta)
                if color is not None:
                    item_a.setForeground(QtGui.QBrush(QtGui.QColor(color)))
                    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
                    item_a.setText(f"{arrow} {self._fmt(a_val)}")
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
            return "#1a8c36" if delta > 0 else "#c73030"
        if name in self.LOWER_IS_BETTER:
            return "#1a8c36" if delta < 0 else "#c73030"
        return None

    # ---------------------------- Param panels ---------------------------- #
    def _clear_param_layout(self) -> None:
        while self.param_layout.rowCount():
            self.param_layout.removeRow(0)

    def _add_slider(self, name, label, lo, hi, default, step=0.01):
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        steps = int(round((hi - lo) / step))
        slider.setRange(0, steps)
        slider.setValue(int(round((default - lo) / step)))
        val_lbl = QtWidgets.QLabel(f"{default:.3g}")

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
        spin.valueChanged.connect(lambda v: self._param_values.__setitem__(name, v))
        self._param_values[name] = default
        self.param_layout.addRow(label, spin)

    def _update_param_panel(self, algo: str) -> None:
        self._clear_param_layout()
        self._param_values: dict = {}
        if algo == "DCP":
            self._add_spin_int("patch_size", "Patch size", 3, 31, 15, 2)
            self._add_slider("omega", "Omega", 0.5, 1.0, 0.95, 0.01)
            self._add_slider("t0", "t₀ (min transmission)", 0.01, 0.3, 0.1, 0.01)
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
            self._add_slider("blend", "Blend w/ original", 0.0, 0.5, 0.0, 0.02)

    # ---------------------------- File actions ---------------------------- #
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
            self.view_after.setText("AFTER\n(nhấn ▶ để xử lý)")
            self.result_img = None
            self.btn_save.setEnabled(False)
            self.time_label.setText("Thời gian xử lý: —")
            self._update_metrics_table()
            if self.parent() and hasattr(self.parent(), "statusBar"):
                self.parent().statusBar().showMessage(f"Đã tải: {Path(path).name}")
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

    # ---------------------------- Run ---------------------------- #
    def on_run(self) -> None:
        if self.hazy_img is None:
            QtWidgets.QMessageBox.information(self, "Chú ý", "Hãy mở một ảnh trước.")
            return
        self.btn_run.setEnabled(False)

        algo = self.cb_algo.currentText()
        self.worker = DehazeWorker(algo, self.hazy_img, dict(self._param_values))
        self.worker.finished_signal.connect(self._on_done)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _on_done(self, result: np.ndarray, dt_ms: float) -> None:
        self.result_img = result
        self.view_after.set_image(result)
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.time_label.setText(f"Thời gian xử lý: {dt_ms:.1f} ms")
        self._update_metrics_table()

    def _on_error(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Lỗi xử lý", msg)

    # ---------------------------- Drag & Drop ---------------------------- #
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self._load_hazy(path)
                break


# ========================================================================= #
#                                 MainWindow
# ========================================================================= #
class MainWindow(QtWidgets.QMainWindow):
    """Cửa sổ chính, chứa 2 tab."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ứng dụng khử sương mù ảnh số")
        self.resize(1400, 820)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab{padding:8px 20px; font-size:13px;}"
            "QTabBar::tab:selected{background:#2d7cff; color:white; font-weight:600;}"
        )

        self.tab_single = SingleImageTab(self)
        self.tab_bench = BenchmarkTab(self)

        self.tabs.addTab(self.tab_single, "🖼️  Xử lý ảnh đơn")
        self.tabs.addTab(self.tab_bench, "📊  Đánh giá Dataset")

        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Sẵn sàng.")


# ========================================================================= #
#                                  Entry
# ========================================================================= #
def run() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run())