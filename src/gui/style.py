from __future__ import annotations


PRIMARY = "#4f46e5"
PRIMARY_HOVER = "#4338ca"
PRIMARY_PRESSED = "#3730a3"

DANGER = "#e11d48"
DANGER_HOVER = "#be123c"
SUCCESS = "#059669"
SUCCESS_HOVER = "#047857"
WARNING = "#d97706"

BG = "#f6f6f8"
CARD = "#ffffff"
BORDER = "#ececef"
BORDER_STRONG = "#dcdce1"
TEXT = "#18181b"
TEXT_MUTED = "#71717a"
TEXT_FAINT = "#a1a1aa"

ACCENT_BG = "#eef2ff"
ACCENT_LINE = "#c7d2fe"

CANVAS = "#101014"
CANVAS_TEXT = "#9ca3af"

TRACK = "#eeeef1"
SECONDARY_BG = TRACK
SECONDARY_BG_HOVER = "#e9e9ec"
SECONDARY_HOVER = SECONDARY_BG_HOVER

APP_QSS = f"""
QMainWindow, QDialog {{
    background: {BG};
    color: {TEXT};
}}
QWidget {{
    color: {TEXT};
    font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial;
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
    padding: 5px 14px;
    font-size: 12px;
}}
QStatusBar::item {{ border: none; }}

/* ---------- Cards (GroupBox) ---------- */
QGroupBox {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    margin-top: 18px;
    padding: 18px 16px 14px 16px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 4px;
    margin-left: 14px;
    color: {TEXT_MUTED};
    background: transparent;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
}}

/* ---------- Buttons ---------- */
QPushButton {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 11px;
    padding: 9px 16px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: {SECONDARY_BG_HOVER};
    border-color: {TEXT_FAINT};
}}
QPushButton:pressed {{
    background: #e8e8eb;
}}
QPushButton:disabled {{
    background: #f4f4f5;
    color: #b4b4bb;
    border-color: {BORDER};
}}
QPushButton:checked {{
    background: {ACCENT_BG};
    color: {PRIMARY};
    border-color: {ACCENT_LINE};
}}

QPushButton[variant="primary"] {{
    background: {PRIMARY};
    color: white;
    border: 1px solid {PRIMARY};
    font-weight: 700;
    padding: 10px 18px;
}}
QPushButton[variant="primary"]:hover {{
    background: {PRIMARY_HOVER};
    border-color: {PRIMARY_HOVER};
}}
QPushButton[variant="primary"]:pressed {{
    background: {PRIMARY_PRESSED};
}}
QPushButton[variant="primary"]:disabled {{
    background: #c7c5f2;
    border-color: #c7c5f2;
    color: #f8fafc;
}}

QPushButton[variant="success"] {{
    background: {SUCCESS};
    color: white;
    border: 1px solid {SUCCESS};
    font-weight: 700;
    padding: 11px 18px;
}}
QPushButton[variant="success"]:hover {{
    background: {SUCCESS_HOVER};
    border-color: {SUCCESS_HOVER};
}}
QPushButton[variant="success"]:disabled {{
    background: #a7d8c8;
    border-color: #a7d8c8;
    color: #f0fdf4;
}}

QPushButton[variant="danger"] {{
    background: {CARD};
    color: {DANGER};
    border: 1px solid #f3c6cf;
    font-weight: 600;
}}
QPushButton[variant="danger"]:hover {{
    background: #fff1f2;
    border-color: {DANGER};
}}

/* ---------- Inputs ---------- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {CARD};
    border: 1px solid {BORDER_STRONG};
    border-radius: 10px;
    padding: 8px 10px;
    selection-background-color: {PRIMARY};
    selection-color: white;
}}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover {{
    border-color: {TEXT_FAINT};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 26px;
}}
QComboBox QAbstractItemView {{
    background: {CARD};
    border: 1px solid {BORDER_STRONG};
    border-radius: 10px;
    padding: 4px;
    selection-background-color: {ACCENT_BG};
    selection-color: {PRIMARY};
    outline: none;
}}

/* ---------- Slider ---------- */
QSlider::groove:horizontal {{
    height: 5px;
    background: {TRACK};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {PRIMARY};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: white;
    width: 18px;
    height: 18px;
    margin: -8px 0;
    border-radius: 9px;
    border: 2px solid {PRIMARY};
}}
QSlider::handle:horizontal:hover {{
    border-color: {PRIMARY_HOVER};
    background: {ACCENT_BG};
}}

/* ---------- CheckBox ---------- */
QCheckBox {{
    spacing: 7px;
    padding: 2px 6px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    background: white;
}}
QCheckBox::indicator:hover {{
    border-color: {PRIMARY};
}}
QCheckBox::indicator:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
    image: none;
}}

/* ---------- Progress ---------- */
QProgressBar {{
    background: {TRACK};
    border: none;
    border-radius: 9px;
    text-align: center;
    color: {TEXT_MUTED};
    height: 20px;
    font-size: 11px;
    font-weight: 600;
}}
QProgressBar::chunk {{
    background: {PRIMARY};
    border-radius: 9px;
}}

/* ---------- Table ---------- */
QTableWidget {{
    background: {CARD};
    alternate-background-color: #fafafa;
    border: 1px solid {BORDER};
    border-radius: 14px;
    gridline-color: {BORDER};
    selection-background-color: {ACCENT_BG};
    selection-color: {TEXT};
}}
QHeaderView::section {{
    background: {CARD};
    color: {TEXT_MUTED};
    padding: 10px 8px;
    border: none;
    border-bottom: 1px solid {BORDER_STRONG};
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.4px;
}}
QTableCornerButton::section {{
    background: {CARD};
    border: none;
}}

/* ---------- Tabs => segmented control ---------- */
QTabWidget::pane {{
    border: none;
    background: {BG};
    top: -1px;
}}
QTabBar {{
    background: {TRACK};
    border-radius: 13px;
    padding: 4px;
    margin: 14px 0 6px 18px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 9px 22px;
    margin: 0;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background: {CARD};
    color: {TEXT};
    font-weight: 700;
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT};
}}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #d4d4d8;
    border-radius: 5px;
    min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{
    background: #b4b4bb;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #d4d4d8;
    border-radius: 5px;
    min-width: 32px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #b4b4bb;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ---------- Tooltip ---------- */
QToolTip {{
    background: {TEXT};
    color: white;
    border: none;
    padding: 7px 11px;
    border-radius: 8px;
    font-size: 12px;
}}
"""

IMAGE_VIEW_QSS = f"""
background: {CANVAS};
color: {CANVAS_TEXT};
border: 1px solid {BORDER};
border-radius: 16px;
font-size: 13px;
"""

IMAGE_VIEW_IMG_QSS = f"""
background: {CANVAS};
border: 1px solid {BORDER};
border-radius: 16px;
"""

INFO_CARD_QSS = f"""
background: {ACCENT_BG};
color: {PRIMARY};
border: 1px solid {ACCENT_LINE};
border-radius: 12px;
padding: 10px 12px;
font-family: "JetBrains Mono", Consolas, "Cascadia Mono", monospace;
font-size: 12px;
font-weight: 600;
"""

LEGEND_CARD_QSS = f"""
background: #fffbeb;
color: #92400e;
border: 1px solid #fde68a;
border-radius: 12px;
padding: 10px 12px;
font-size: 11px;
"""

BEST_CARD_QSS = f"""
background: #ecfdf5;
color: #065f46;
border: 1px solid #a7f3d0;
border-radius: 12px;
padding: 12px 14px;
font-size: 12px;
"""

HEADER_BAR_QSS = f"""
background: {CARD};
border-bottom: 1px solid {BORDER};
"""

HEADER_TITLE_QSS = f"""
color: {TEXT};
font-size: 19px;
font-weight: 800;
letter-spacing: -0.3px;
"""

HEADER_SUBTITLE_QSS = f"""
color: {TEXT_MUTED};
font-size: 12px;
"""

HEADER_BADGE_QSS = f"""
background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {PRIMARY}, stop:1 #7c3aed);
border-radius: 12px;
color: white;
font-size: 18px;
"""

HEADER_CHIP_QSS = f"""
background: {ACCENT_BG};
color: {PRIMARY};
border: 1px solid {ACCENT_LINE};
border-radius: 9px;
padding: 5px 11px;
font-size: 11px;
font-weight: 700;
letter-spacing: 0.3px;
"""

SECTION_LABEL_QSS = f"""
color: {TEXT_MUTED};
font-weight: 700;
letter-spacing: 1.5px;
font-size: 11px;
"""
