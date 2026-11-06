from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Palette:
    bg_app: str = "#F3F6F5"
    bg_card: str = "#FFFFFF"
    bg_header: str = "#0E3B3C"
    bg_subtle: str = "#EEF3F2"
    border: str = "#D5DFDC"
    border_input: str = "#C3D0CC"
    text_main: str = "#15201E"
    text_muted: str = "#5B6B68"
    primary: str = "#0F766E"
    primary_hover: str = "#0B5F58"
    success: str = "#15803D"
    success_hover: str = "#116A33"
    danger: str = "#B42318"
    danger_hover: str = "#921B12"
    warning: str = "#B54708"
    selection: str = "#CDEDE8"
    chip_ready_bg: str = "#ECFDF3"
    chip_ready_fg: str = "#027A48"
    chip_run_bg: str = "#E6F4FF"
    chip_run_fg: str = "#175CD3"
    chip_err_bg: str = "#FEF3F2"
    chip_err_fg: str = "#B42318"


@dataclass(frozen=True)
class Sizing:
    radius_card: int = 12
    radius_input: int = 8
    radius_button: int = 8
    control_h: int = 32
    spacing: int = 12
    font_size: int = 13


PALETTE = Palette()
SIZING = Sizing()


def build_stylesheet(icons_dir: Path | str | None = None) -> str:
    """Hoja de estilo de la app. ``icons_dir`` contiene arrow_up.png, arrow_down.png y check.png."""
    p = PALETTE
    s = SIZING
    icons = Path(icons_dir) if icons_dir else None

    def icon(name: str) -> str:
        if icons is None or not (icons / name).exists():
            return ""
        return "image: url(" + (icons / name).resolve().as_posix() + ");"

    return f"""
    QWidget {{
        color: {p.text_main};
        font-size: {s.font_size}px;
    }}

    QMainWindow, QDialog, QMessageBox, QInputDialog {{
        background: {p.bg_app};
    }}

    QFrame#HeaderCard {{
        background: {p.bg_header};
        border-radius: {s.radius_card}px;
        border: none;
    }}

    QLabel#TitleLabel {{
        color: #FFFFFF;
        font-size: 22px;
        font-weight: 700;
    }}

    QLabel#SubtitleLabel {{
        color: #BFE3DE;
        font-size: 12px;
    }}

    QLabel#StatusChipReady, QLabel#StatusChipRunning, QLabel#StatusChipError {{
        border-radius: 14px;
        padding: 6px 14px;
        font-weight: 700;
        min-width: 96px;
        qproperty-alignment: AlignCenter;
    }}

    QLabel#StatusChipReady {{
        background: {p.chip_ready_bg};
        color: {p.chip_ready_fg};
        border: 1px solid #A6F4C5;
    }}

    QLabel#StatusChipRunning {{
        background: {p.chip_run_bg};
        color: {p.chip_run_fg};
        border: 1px solid #B2DDFF;
    }}

    QLabel#StatusChipError {{
        background: {p.chip_err_bg};
        color: {p.chip_err_fg};
        border: 1px solid #FECDCA;
    }}

    QFrame#Card {{
        background: {p.bg_card};
        border: 1px solid {p.border};
        border-radius: {s.radius_card}px;
    }}

    QFrame#MetricCard {{
        background: {p.bg_subtle};
        border: 1px solid {p.border};
        border-radius: 10px;
    }}

    QLabel#MetricTitle {{
        color: {p.text_muted};
        font-size: 11px;
        font-weight: 600;
    }}

    QLabel#MetricValue {{
        color: {p.text_main};
        font-size: 21px;
        font-weight: 700;
    }}

    QLabel#MetricCaption {{
        color: {p.text_muted};
        font-size: 11px;
    }}

    QLabel#SectionLabel {{
        font-size: 14px;
        font-weight: 600;
    }}

    QGroupBox {{
        background: {p.bg_card};
        border: 1px solid {p.border};
        border-radius: {s.radius_card}px;
        margin-top: 12px;
        padding: 12px 10px 10px 10px;
        font-weight: 700;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {p.primary};
    }}

    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QPlainTextEdit {{
        border: 1px solid {p.border_input};
        border-radius: {s.radius_input}px;
        padding: 5px 8px;
        background: #FFFFFF;
        selection-background-color: {p.selection};
        selection-color: {p.text_main};
    }}

    QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
        border: 1px solid {p.primary};
    }}

    QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
        background: {p.bg_subtle};
        color: {p.text_muted};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox::down-arrow {{
        {icon("arrow_down.png")}
        width: 8px;
        height: 5px;
    }}

    QAbstractSpinBox {{
        padding-right: 22px;
    }}

    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        subcontrol-origin: border;
        width: 20px;
        border: none;
        border-left: 1px solid {p.border};
        background: {p.bg_subtle};
    }}

    QAbstractSpinBox::up-button {{
        subcontrol-position: top right;
        border-top-right-radius: {s.radius_input}px;
    }}

    QAbstractSpinBox::down-button {{
        subcontrol-position: bottom right;
        border-bottom-right-radius: {s.radius_input}px;
    }}

    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background: {p.selection};
    }}

    QAbstractSpinBox::up-arrow {{
        {icon("arrow_up.png")}
        width: 8px;
        height: 5px;
    }}

    QAbstractSpinBox::down-arrow {{
        {icon("arrow_down.png")}
        width: 8px;
        height: 5px;
    }}

    QComboBox QAbstractItemView {{
        background: #FFFFFF;
        border: 1px solid {p.border};
        selection-background-color: {p.selection};
        selection-color: {p.text_main};
    }}

    QTableWidget {{
        border: 1px solid {p.border};
        border-radius: {s.radius_input}px;
        background: #FFFFFF;
        alternate-background-color: #F7FAF9;
        gridline-color: #E6ECEA;
        selection-background-color: {p.selection};
        selection-color: {p.text_main};
    }}

    QHeaderView::section {{
        background: {p.bg_subtle};
        color: {p.text_main};
        border: 0;
        border-right: 1px solid {p.border};
        border-bottom: 1px solid {p.border};
        padding: 7px 8px;
        font-weight: 700;
    }}

    QPushButton {{
        min-height: {s.control_h}px;
        border-radius: {s.radius_button}px;
        border: none;
        padding: 0 14px;
        background: {p.primary};
        color: white;
        font-weight: 600;
    }}

    QPushButton:hover {{
        background: {p.primary_hover};
    }}

    QPushButton:disabled {{
        background: #AFC3BF;
        color: #F1F5F4;
    }}

    QPushButton#PrimaryAction {{
        background: {p.success};
        font-size: 14px;
    }}

    QPushButton#PrimaryAction:hover {{
        background: {p.success_hover};
    }}

    QPushButton#DangerAction {{
        background: {p.danger};
    }}

    QPushButton#DangerAction:hover {{
        background: {p.danger_hover};
    }}

    QPushButton#PrimaryAction:disabled, QPushButton#DangerAction:disabled {{
        background: #AFC3BF;
        color: #F1F5F4;
    }}

    QTabWidget::pane {{
        border: 1px solid {p.border};
        border-radius: {s.radius_card}px;
        top: -1px;
        background: #FFFFFF;
    }}

    QTabBar::tab {{
        background: {p.bg_subtle};
        color: {p.text_muted};
        border: 1px solid {p.border};
        padding: 7px 14px;
        margin-right: 2px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}

    QTabBar::tab:selected {{
        background: #FFFFFF;
        color: {p.primary};
        border-bottom-color: #FFFFFF;
        font-weight: 700;
    }}

    QTabBar::tab:hover:!selected {{
        color: {p.text_main};
    }}

    QProgressBar {{
        border: 1px solid {p.border_input};
        border-radius: 8px;
        text-align: center;
        background: #FFFFFF;
        min-height: 20px;
    }}

    QProgressBar::chunk {{
        background-color: #7FD1C6;
        border-radius: 6px;
    }}

    QCheckBox, QRadioButton {{
        spacing: 8px;
        background: transparent;
    }}

    QCheckBox::indicator, QRadioButton::indicator, QGroupBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid #9DB0AB;
        background: #FFFFFF;
    }}

    QCheckBox::indicator, QGroupBox::indicator {{
        border-radius: 4px;
    }}

    QRadioButton::indicator {{
        border-radius: 8px;
    }}

    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {p.primary};
    }}

    QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
        background: {p.primary};
        border-color: {p.primary};
        {icon("check.png")}
    }}

    QRadioButton::indicator:checked {{
        border-color: {p.primary};
        background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
            stop:0 {p.primary}, stop:0.45 {p.primary}, stop:0.55 #FFFFFF, stop:1 #FFFFFF);
    }}

    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
        background: {p.bg_subtle};
        border-color: {p.border};
    }}

    QSplitter::handle {{
        background: {p.bg_app};
    }}

    QSplitter::handle:hover {{
        background: {p.border};
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical {{
        background: #C6D3CF;
        border-radius: 4px;
        min-height: 28px;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px;
    }}

    QScrollBar::handle:horizontal {{
        background: #C6D3CF;
        border-radius: 4px;
        min-width: 28px;
    }}

    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0;
        height: 0;
    }}

    QLabel#HintLabel {{
        color: {p.text_muted};
        font-size: 12px;
    }}

    QLabel#ErrorLabel {{
        color: {p.danger};
        font-weight: 600;
    }}

    QLabel#OkLabel {{
        color: {p.success};
        font-weight: 600;
    }}

    QToolTip {{
        background: #10201E;
        color: #FFFFFF;
        border: 1px solid #1F3A37;
        padding: 5px;
    }}

    QMenu {{
        background: #FFFFFF;
        border: 1px solid {p.border};
        padding: 4px;
    }}

    QMenu::item {{
        padding: 6px 18px;
        border-radius: 6px;
    }}

    QMenu::item:selected {{
        background: {p.selection};
        color: {p.text_main};
    }}
    """
