from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    bg_app: str = "#F3F6FA"
    bg_card: str = "#FFFFFF"
    bg_header: str = "#0F3554"
    border: str = "#D6DEE8"
    text_main: str = "#17212B"
    text_muted: str = "#5D6B7A"
    primary: str = "#1F4F7A"
    primary_hover: str = "#163C5D"
    success: str = "#1F7A4C"
    success_hover: str = "#18603C"
    danger: str = "#B42318"
    warning: str = "#B54708"
    chip_ready_bg: str = "#ECFDF3"
    chip_ready_fg: str = "#027A48"
    chip_run_bg: str = "#EFF8FF"
    chip_run_fg: str = "#175CD3"
    chip_err_bg: str = "#FEF3F2"
    chip_err_fg: str = "#B42318"


@dataclass(frozen=True)
class Sizing:
    radius_card: int = 12
    radius_input: int = 8
    radius_button: int = 8
    control_h: int = 34
    spacing: int = 12
    font_size: int = 13


PALETTE = Palette()
SIZING = Sizing()


def build_stylesheet() -> str:
    p = PALETTE
    s = SIZING
    return f"""
    QWidget {{
        background: {p.bg_app};
        color: {p.text_main};
        font-size: {s.font_size}px;
    }}

    QMainWindow {{
        background: {p.bg_app};
    }}

    QFrame#HeaderCard {{
        background: {p.bg_header};
        border-radius: {s.radius_card}px;
        border: none;
    }}

    QLabel#TitleLabel {{
        color: #FFFFFF;
        font-size: 21px;
        font-weight: 700;
    }}

    QLabel#SubtitleLabel {{
        color: #D8E8F8;
        font-size: 12px;
    }}

    QLabel#StatusChipReady, QLabel#StatusChipRunning, QLabel#StatusChipError {{
        border-radius: 14px;
        padding: 6px 12px;
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

    QGroupBox {{
        background: {p.bg_card};
        border: 1px solid {p.border};
        border-radius: {s.radius_card}px;
        margin-top: 10px;
        padding: 10px;
        font-weight: 700;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }}

    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QPlainTextEdit, QTableWidget {{
        border: 1px solid #C7D3E0;
        border-radius: {s.radius_input}px;
        padding: 6px;
        background: #FFFFFF;
    }}

    QPushButton {{
        min-height: {s.control_h}px;
        border-radius: {s.radius_button}px;
        border: none;
        padding: 0 12px;
        background: {p.primary};
        color: white;
        font-weight: 600;
    }}

    QPushButton:hover {{
        background: {p.primary_hover};
    }}

    QPushButton:disabled {{
        background: #A4B5C7;
        color: #EAF0F6;
    }}

    QPushButton#PrimaryAction {{
        background: {p.success};
    }}

    QPushButton#PrimaryAction:hover {{
        background: {p.success_hover};
    }}

    QPushButton#DangerAction {{
        background: {p.danger};
    }}

    QPushButton#DangerAction:hover {{
        background: #921B12;
    }}

    QTabWidget::pane {{
        border: 1px solid {p.border};
        border-radius: {s.radius_card}px;
        top: -1px;
        background: #FFFFFF;
    }}

    QTabBar::tab {{
        background: #EDF2F8;
        border: 1px solid #D3DDE8;
        padding: 8px 12px;
        margin-right: 2px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}

    QTabBar::tab:selected {{
        background: #FFFFFF;
        border-bottom-color: #FFFFFF;
        font-weight: 700;
    }}

    QProgressBar {{
        border: 1px solid #C7D3E0;
        border-radius: 8px;
        text-align: center;
        background: #FFFFFF;
        min-height: 20px;
    }}

    QProgressBar::chunk {{
        background-color: {p.primary};
        border-radius: 6px;
    }}

    QHeaderView::section {{
        background: #EEF3F8;
        border: 0;
        border-right: 1px solid #D7DFE8;
        border-bottom: 1px solid #D7DFE8;
        padding: 7px;
        font-weight: 700;
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
        background: #12202F;
        color: #FFFFFF;
        border: 1px solid #1C324A;
        padding: 4px;
    }}
    """
