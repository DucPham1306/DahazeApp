from __future__ import annotations

PRIMARY = "#2563eb"
PRIMARY_HOVER = "#1d4ed8"
PRIMARY_PRESSED = "#1e40af"
SECONDARY_BG = "#f1f5f9"
SECONDARY_HOVER = "#e2e8f0"
DANGER = "#dc2626"
DANGER_HOVER = "#b91c1c"
SUCCESS = "#16a34a"
WARNING = "#ea580c"

BG = "#f8fafc"
CARD = "#ffffff"
BORDER = "#e2e8f0"
TEXT = "#0f172a"
TEXT_MUTED = "#64748b"
ACCENT_BG = "#eff6ff"

APP_QSS = f"""
QMainWindow, QDialog {{
    background: {BG};
    color: {TEXT};
}}
QWidget {{
    color: {TEXT};
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial;
    font-size: 13px;
}}
QLabel {{
    background: transparent;
    color: {TEXT};
}}
QStatusBar {{
    background: {CARD};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
    padding: 4px 10px;
    font-size: 12px;
}}
QGroupBox {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding: 14px 12px 10px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    margin-left: 8px;
    color: {PRIMARY};
    background: {CARD};
}}
QPushButton {{
    background: {SECONDARY_BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {SECONDARY_HOVER};
}}
QPushButton:pressed {{
    background: #cbd5e1;
}}
QPushButton:disabled {{
    background: #f1f5f9;
    color: #94a3b8;
    border-color: #e2e8f0;
}}
QPushButton[variant="primary"] {{
    background: {PRIMARY};
    color: white;
    border: 1px solid {PRIMARY};
    font-weight: 600;
    padding: 9px 18px;
}}
QPushButton[variant="primary"]:hover {{
    background: {PRIMARY_HOVER};
    border-color: {PRIMARY_HOVER};
}}
QPushButton[variant="primary"]:pressed {{
    background: {PRIMARY_PRESSED};
}}
QPushButton[variant="primary"]:disabled {{
    background: #94a3b8;
    border-color: #94a3b8;
    color: #f8fafc;
}}
QPushButton[variant="danger"] {{
    background: {DANGER};
    color: white;
    border: 1px solid {DANGER};
}}
QPushButton[variant="danger"]:hover {{
    background: {DANGER_HOVER};
    border-color: {DANGER_HOVER};
}}
QPushButton[variant="success"] {{
    background: {SUCCESS};
    color: white;
    border: 1px solid {SUCCESS};
    font-weight: 600;
}}
QPushButton[variant="success"]:hover {{
    background: #15803d;
}}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {PRIMARY};
}}
QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{
    border-color: {PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {PRIMARY};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {PRIMARY};
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 8px;
    border: 2px solid white;
}}
QSlider::handle:horizontal:hover {{
    background: {PRIMARY_HOVER};
}}
QCheckBox {{
    spacing: 6px;
    padding: 2px 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #94a3b8;
    border-radius: 4px;
    background: white;
}}
QCheckBox::indicator:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
    image: none;
}}
QProgressBar {{
    background: {SECONDARY_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    text-align: center;
    color: {TEXT};
    height: 18px;
    font-size: 11px;
}}
QProgressBar::chunk {{
    background: {PRIMARY};
    border-radius: 7px;
}}
QTableWidget {{
    background: {CARD};
    alternate-background-color: #f8fafc;
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
    selection-background-color: {ACCENT_BG};
    selection-color: {TEXT};
}}
QHeaderView::section {{
    background: #f1f5f9;
    color: {TEXT};
    padding: 8px 6px;
    border: none;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
QTabWidget::pane {{
    border: none;
    background: {BG};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 10px 22px;
    margin-right: 2px;
    border-bottom: 3px solid transparent;
    font-size: 13px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    color: {PRIMARY};
    border-bottom: 3px solid {PRIMARY};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #cbd5e1;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: #94a3b8;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: #cbd5e1;
    border-radius: 5px;
    min-width: 30px;
}}
QToolTip {{
    background: {TEXT};
    color: white;
    border: none;
    padding: 6px 10px;
    border-radius: 6px;
}}
"""

IMAGE_VIEW_QSS = f"""
background: #0f172a;
color: #cbd5e1;
border: 1px solid {BORDER};
border-radius: 10px;
font-size: 13px;
"""

IMAGE_VIEW_IMG_QSS = f"""
background: transparent;
border: 1px solid {BORDER};
border-radius: 10px;
"""

INFO_CARD_QSS = f"""
background: {ACCENT_BG};
color: {PRIMARY};
border: 1px solid #bfdbfe;
border-radius: 8px;
padding: 8px 10px;
font-family: Consolas, "Cascadia Mono", monospace;
font-size: 12px;
"""

LEGEND_CARD_QSS = f"""
background: #fefce8;
color: #713f12;
border: 1px solid #fde68a;
border-radius: 8px;
padding: 8px 10px;
font-size: 11px;
"""

BEST_CARD_QSS = f"""
background: #ecfdf5;
color: #065f46;
border: 1px solid #a7f3d0;
border-radius: 8px;
padding: 10px 12px;
font-size: 12px;
"""
