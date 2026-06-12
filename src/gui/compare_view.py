from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..utils import to_uint8
from .style import PRIMARY, BORDER, CANVAS, CANVAS_TEXT


class SplitCompareView(QtWidgets.QWidget):

    PLACEHOLDER_EMPTY = (
        "🖱️\n\n"
        "CHẾ ĐỘ SO SÁNH THANH TRƯỢT\n"
        "Mở ảnh hazy và xử lý để bắt đầu so sánh"
    )
    PLACEHOLDER_WAIT = (
        "✨\n\nĐÃ CÓ ẢNH HAZY\nNhấn ▶ Khử sương mù để hiển thị so sánh"
    )

    _HANDLE_RADIUS = 18
    _HANDLE_HIT_PX = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self._before_pixmap: Optional[QtGui.QPixmap] = None
        self._after_pixmap: Optional[QtGui.QPixmap] = None
        self._divider_ratio: float = 0.5
        self._dragging: bool = False

        self.setMouseTracking(True)
        self.setMinimumSize(560, 360)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"background:{CANVAS}; border:1px solid {BORDER}; border-radius:16px;"
        )

    def set_before(self, img: np.ndarray) -> None:
        self._before_pixmap = self._np_to_pixmap(img)
        self.update()

    def set_after(self, img: np.ndarray) -> None:
        self._after_pixmap = self._np_to_pixmap(img)
        self.update()

    def clear(self) -> None:
        self._before_pixmap = None
        self._after_pixmap = None
        self._divider_ratio = 0.5
        self.update()

    def set_divider_ratio(self, ratio: float) -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        if ratio != self._divider_ratio:
            self._divider_ratio = ratio
            self.update()

    @staticmethod
    def _np_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
        img_u8 = np.ascontiguousarray(to_uint8(img))
        h, w, _ = img_u8.shape
        qimg = QtGui.QImage(
            img_u8.data, w, h, 3 * w, QtGui.QImage.Format_RGB888
        )
        return QtGui.QPixmap.fromImage(qimg.copy())

    def _image_rect(self) -> QtCore.QRect:
        ref = self._before_pixmap or self._after_pixmap
        if ref is None:
            return self.rect()
        scaled = ref.size()
        scaled.scale(self.size(), QtCore.Qt.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QtCore.QRect(x, y, scaled.width(), scaled.height())

    def _divider_x(self, target: QtCore.QRect) -> int:
        return target.left() + int(target.width() * self._divider_ratio)

    def paintEvent(self, e):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        painter.fillRect(self.rect(), QtGui.QColor(CANVAS))

        if self._before_pixmap is None:
            self._draw_placeholder(painter, self.PLACEHOLDER_EMPTY)
            return
        if self._after_pixmap is None:
            target = self._image_rect()
            painter.drawPixmap(target, self._before_pixmap)
            self._draw_overlay_label(
                painter,
                "BEFORE (chưa có AFTER — nhấn ▶ để xử lý)",
                target.adjusted(0, 0, 0, 0),
                center=True,
            )
            return

        target = self._image_rect()
        divider_x = self._divider_x(target)

        painter.drawPixmap(target, self._after_pixmap)

        if divider_x > target.left():
            clip_w = divider_x - target.left()
            clip_rect = QtCore.QRect(
                target.left(), target.top(), clip_w, target.height()
            )
            painter.save()
            painter.setClipRect(clip_rect)
            painter.drawPixmap(target, self._before_pixmap)
            painter.restore()

        line_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 2)
        painter.setPen(line_pen)
        painter.drawLine(divider_x, target.top(), divider_x, target.bottom())

        shadow = QtGui.QLinearGradient(
            divider_x - 6, 0, divider_x + 6, 0
        )
        shadow.setColorAt(0, QtGui.QColor(0, 0, 0, 0))
        shadow.setColorAt(0.5, QtGui.QColor(0, 0, 0, 60))
        shadow.setColorAt(1, QtGui.QColor(0, 0, 0, 0))
        painter.fillRect(
            QtCore.QRect(divider_x - 6, target.top(), 12, target.height()),
            QtGui.QBrush(shadow),
        )
        painter.setPen(line_pen)
        painter.drawLine(divider_x, target.top(), divider_x, target.bottom())

        center_y = target.center().y()
        r = self._HANDLE_RADIUS
        painter.setPen(QtGui.QPen(QtGui.QColor(PRIMARY), 2))
        painter.setBrush(QtGui.QColor("white"))
        painter.drawEllipse(QtCore.QPoint(divider_x, center_y), r, r)

        arrow_pen = QtGui.QPen(QtGui.QColor(PRIMARY), 2)
        arrow_pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(arrow_pen)
        a = 5
        painter.drawLine(divider_x - a, center_y, divider_x - a - 6, center_y - 5)
        painter.drawLine(divider_x - a, center_y, divider_x - a - 6, center_y + 5)
        painter.drawLine(divider_x + a, center_y, divider_x + a + 6, center_y - 5)
        painter.drawLine(divider_x + a, center_y, divider_x + a + 6, center_y + 5)

        if divider_x - target.left() > 90:
            self._draw_pill(
                painter,
                "BEFORE",
                target.left() + 12,
                target.top() + 12,
                anchor="left",
            )
        if target.right() - divider_x > 90:
            self._draw_pill(
                painter,
                "AFTER",
                target.right() - 12,
                target.top() + 12,
                anchor="right",
            )

    def _draw_placeholder(self, painter: QtGui.QPainter, text: str) -> None:
        painter.setPen(QtGui.QColor(CANVAS_TEXT))
        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
            text,
        )

    def _draw_overlay_label(
        self,
        painter: QtGui.QPainter,
        text: str,
        rect: QtCore.QRect,
        center: bool = False,
    ) -> None:
        painter.save()
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(text)
        th = metrics.height()
        pad_x, pad_y = 10, 6
        bw, bh = tw + 2 * pad_x, th + 2 * pad_y
        if center:
            x = rect.center().x() - bw // 2
            y = rect.bottom() - bh - 12
        else:
            x = rect.left() + 12
            y = rect.top() + 12
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(15, 23, 42, 210))
        painter.drawRoundedRect(QtCore.QRectF(x, y, bw, bh), 6, 6)
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(QtCore.QRectF(x, y, bw, bh), QtCore.Qt.AlignCenter, text)
        painter.restore()

    def _draw_pill(
        self,
        painter: QtGui.QPainter,
        text: str,
        x_anchor: int,
        y_anchor: int,
        anchor: str = "left",
    ) -> None:
        painter.save()
        font = painter.font()
        font.setBold(True)
        font.setPointSize(9)
        font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 1.0)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(text)
        th = metrics.height()
        pad_x, pad_y = 10, 5
        bw, bh = tw + 2 * pad_x, th + 2 * pad_y
        if anchor == "right":
            x = x_anchor - bw
        else:
            x = x_anchor
        y = y_anchor
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(15, 23, 42, 210))
        painter.drawRoundedRect(QtCore.QRectF(x, y, bw, bh), 8, 8)
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(
            QtCore.QRectF(x, y, bw, bh), QtCore.Qt.AlignCenter, text
        )
        painter.restore()

    def _on_handle(self, pos: QtCore.QPoint) -> bool:
        if self._before_pixmap is None or self._after_pixmap is None:
            return False
        target = self._image_rect()
        if not target.contains(pos):
            return False
        return abs(pos.x() - self._divider_x(target)) <= self._HANDLE_HIT_PX

    def _update_cursor(self, pos: QtCore.QPoint) -> None:
        if self._dragging or self._on_handle(pos):
            self.setCursor(QtCore.Qt.SizeHorCursor)
        elif self._image_rect().contains(pos):
            self.setCursor(QtCore.Qt.PointingHandCursor)
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)

    def _move_divider_to(self, x: int) -> None:
        target = self._image_rect()
        if target.width() <= 0:
            return
        ratio = (x - target.left()) / target.width()
        ratio = max(0.0, min(1.0, ratio))
        if ratio != self._divider_ratio:
            self._divider_ratio = ratio
            self.update()

    def mousePressEvent(self, e):
        if e.button() != QtCore.Qt.LeftButton:
            return super().mousePressEvent(e)
        if self._before_pixmap is None or self._after_pixmap is None:
            return
        if self._image_rect().contains(e.pos()):
            self._dragging = True
            self._move_divider_to(e.pos().x())
            self._update_cursor(e.pos())

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._move_divider_to(e.pos().x())
        else:
            self._update_cursor(e.pos())

    def mouseReleaseEvent(self, e):
        if self._dragging and e.button() == QtCore.Qt.LeftButton:
            self._dragging = False
            self._update_cursor(e.pos())

    def mouseDoubleClickEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.set_divider_ratio(0.5)

    def leaveEvent(self, e):
        self.setCursor(QtCore.Qt.ArrowCursor)
        super().leaveEvent(e)
