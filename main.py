import multiprocessing
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import librosa
import numpy as np
import soundfile as sf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCharts import QBarCategoryAxis, QBarSet, QChart, QChartView, QHorizontalBarSeries, QValueAxis
from PySide6.QtCore import QElapsedTimer, QRect, QSettings, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QRadioButton,
    QSplitter,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from audio_analyzer import AudioAnalyzer, AudioWorker, ProfileManager
from ui_theme import PALETTE, build_stylesheet
from validation import (
    ReferenceTable,
    ValidationResult,
    combine_results,
    compare_recording,
    load_reference_table,
    plot_species_recall,
    plot_validation,
)

try:
    import noisereduce as nr
except Exception:  # pragma: no cover - optional dependency fallback
    nr = None


APP_TITLE = "EcoAcoustic Sentinel"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aiff", ".aif"}
CONF_ROLE = int(Qt.UserRole) + 1

SOURCE_LABELS = {
    "local": "CONDESAN",
    "custom": "Perfil",
    "birdnet_global": "BirdNET global",
    "anomaly": "Anomalía",
}
SOURCE_STYLES = {  # (texto, fondo)
    "local": ("#0F5257", "#DDF4F1"),
    "custom": ("#14532D", "#E3F5E8"),
    "birdnet_global": ("#1D4ED8", "#E6EEFF"),
    "anomaly": ("#92400E", "#FFF1DB"),
}
SOURCE_TIPS = {
    "local": "Clasificador local CONDESAN: BirdNET restringido a especies locales con umbral calibrado por especie.",
    "custom": "Especie de tu biblioteca de perfiles (similitud coseno).",
    "birdnet_global": "Detección de BirdNET global (6 522 especies).",
    "anomaly": "Evento que BirdNET detecta pero no coincide con tus perfiles (posible especie nueva).",
}


def _resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative_path
    return Path(__file__).resolve().parent / relative_path


def _confidence_fill(frac: float) -> QColor:
    if frac >= 0.75:
        return QColor("#99E2B4")
    if frac >= 0.5:
        return QColor("#A7E3DC")
    return QColor("#FBD38D")


class NumericItem(QTableWidgetItem):
    """Celda con texto formateado que se ordena por su valor numerico."""

    def __init__(self, value: float, text: str) -> None:
        super().__init__(text)
        self._value = float(value)
        self.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))

    def __lt__(self, other) -> bool:
        if isinstance(other, NumericItem):
            return self._value < other._value
        return super().__lt__(other)


class ConfidenceDelegate(QStyledItemDelegate):
    """Dibuja la confianza como una barra con su porcentaje."""

    def paint(self, painter, option, index) -> None:
        value = index.data(CONF_ROLE)
        if value is None:
            super().paint(painter, option, index)
            return
        frac = max(0.0, min(1.0, float(value)))
        painter.save()
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        rect = option.rect.adjusted(6, 6, -6, -6)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#E5EBEA"))
        painter.drawRoundedRect(rect, 4, 4)
        if frac > 0:
            painter.setBrush(_confidence_fill(frac))
            painter.drawRoundedRect(QRect(rect.left(), rect.top(), max(4, int(rect.width() * frac)), rect.height()), 4, 4)
        painter.setPen(QColor("#10201E"))
        painter.drawText(rect, int(Qt.AlignCenter), f"{frac * 100:.0f} %")
        painter.restore()


@dataclass
class ExportOptions:
    formats: list[str]
    include_zone_metadata: bool = False


class ExportOptionsDialog(QDialog):
    def __init__(self, options: ExportOptions, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configurar exportación")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hint = QLabel("Selecciona los formatos que se generarán en cada análisis.")
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.csv_check = QCheckBox("CSV")
        self.raven_check = QCheckBox("Raven Table")
        self.audacity_check = QCheckBox("Audacity")
        self.kaleidoscope_check = QCheckBox("Kaleidoscope")

        self.csv_check.setChecked("csv" in options.formats)
        self.raven_check.setChecked("table" in options.formats)
        self.audacity_check.setChecked("audacity" in options.formats)
        self.kaleidoscope_check.setChecked("kaleidoscope" in options.formats)

        layout.addWidget(self.csv_check)
        layout.addWidget(self.raven_check)
        layout.addWidget(self.audacity_check)
        layout.addWidget(self.kaleidoscope_check)

        self.zone_meta_check = QCheckBox("Incluir metadatos de zona (lat/lon) en CSV")
        self.zone_meta_check.setChecked(options.include_zone_metadata)
        layout.addWidget(self.zone_meta_check)

        self.error_label = QLabel("")
        self.error_label.setObjectName("ErrorLabel")
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        if not self.selected_formats():
            self.error_label.setText("Debes seleccionar al menos un formato de exportación.")
            return
        self.accept()

    def selected_formats(self) -> list[str]:
        formats: list[str] = []
        if self.csv_check.isChecked():
            formats.append("csv")
        if self.raven_check.isChecked():
            formats.append("table")
        if self.audacity_check.isChecked():
            formats.append("audacity")
        if self.kaleidoscope_check.isChecked():
            formats.append("kaleidoscope")
        return formats

    def include_zone_metadata(self) -> bool:
        return self.zone_meta_check.isChecked()


class SpectrogramWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.figure = Figure(figsize=(6, 2.8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._colorbar = None
        layout.addWidget(self.canvas)
        self.clear_plot()

    def clear_plot(self, message: str = "Selecciona una detección para ver su espectrograma.") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#0f1117")
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=10,
            color="#9aa0bb",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw_idle()

    def plot_segment(
        self,
        audio_path: str,
        *,
        start_sec: float,
        end_sec: float,
        title: str = "",
        use_noise_reduction: bool = False,
        context_sec: float = 1.0,
    ) -> None:
        """Espectrograma de la deteccion con contexto a ambos lados y la ventana resaltada."""
        start_sec, end_sec = float(start_sec), float(end_sec)
        seg_start = max(0.0, start_sec - context_sec)
        seg_duration = max(0.2, (end_sec + context_sec) - seg_start)
        signal, sample_rate = librosa.load(audio_path, sr=None, mono=True, offset=seg_start, duration=seg_duration)
        if signal.size == 0:
            self.clear_plot("No se pudo extraer el audio para el espectrograma.")
            return
        if use_noise_reduction and nr is not None:
            signal = nr.reduce_noise(y=signal, sr=sample_rate, stationary=True, prop_decrease=0.9)

        n_fft = 2048 if sample_rate >= 32000 else 1024
        hop = n_fft // 8
        magnitude = np.abs(librosa.stft(np.asarray(signal, dtype=np.float32), n_fft=n_fft, hop_length=hop, window="hann"))
        ref_amp = float(magnitude.max()) if magnitude.size and magnitude.max() > 0 else 1e-6
        spec_db = librosa.amplitude_to_db(magnitude, ref=ref_amp)
        nyquist_khz = sample_rate / 2000.0
        seg_end = seg_start + len(signal) / float(sample_rate)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = ax.imshow(
            spec_db,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[seg_start, seg_end, 0, nyquist_khz],
            vmin=-80,
            vmax=0,
        )
        ax.set_ylim(0, min(12.0, nyquist_khz))
        ax.axvspan(start_sec, end_sec, color="#5EEAD4", alpha=0.10, lw=0)
        for x in (start_sec, end_sec):
            ax.axvline(x, color="#5EEAD4", lw=1.2, ls="--")
        ax.set_xlabel("Tiempo en la grabación (s)")
        ax.set_ylabel("Frecuencia (kHz)")
        ax.set_title(title or "Espectrograma", fontsize=10, loc="left")
        colorbar = self.figure.colorbar(img, ax=ax, pad=0.01)
        colorbar.set_label("dB")
        self.canvas.draw_idle()

class AudioAnalysisWindow(QMainWindow):
    DEFAULTS = {
        "overlap": 0.30,
        "sensitivity": 1.00,
        "min_conf": 0.75,
        "anomaly_threshold": 0.45,
        "threads": max(1, (os.cpu_count() or 4) // 2),
        "use_multi_vector": True,
        "use_noise_reduction": False,
        "smoothing_window": 3,
        "use_global_discovery": False,
        "use_local_classifier": False,
    }

    PRESETS = {
        "Balanceado": {"overlap": 0.30, "sensitivity": 1.00, "min_conf": 0.75, "threads": max(1, (os.cpu_count() or 4) // 2)},
        "Alta sensibilidad": {"overlap": 0.60, "sensitivity": 1.20, "min_conf": 0.70, "threads": max(1, (os.cpu_count() or 4) // 2)},
        "Precisión alta": {"overlap": 0.20, "sensitivity": 0.90, "min_conf": 0.82, "threads": max(1, (os.cpu_count() or 4) // 2)},
    }

    def __init__(self) -> None:
        super().__init__()
        self.analysis_thread: QThread | None = None
        self.analysis_worker: AudioWorker | None = None
        self.settings = QSettings("EcoAcoustic", "Sentinel")
        self.export_options = ExportOptions(formats=["csv", "table"], include_zone_metadata=False)
        self.log_records: list[tuple[str, str]] = []
        self.total_files = 0
        self.processed_files = 0
        self.elapsed_timer = QElapsedTimer()
        self.elapsed_tick = QTimer(self)
        self.elapsed_tick.timeout.connect(self._update_elapsed)
        self.result_rows: list[dict] = []
        self.session_detection_counts: Counter[str] = Counter()
        self.analyzed_files: list[str] = []
        self.reference_table: ReferenceTable | None = None
        self.validation_results: list[tuple[ValidationResult, str | None]] = []

        # La validacion de la ruta recorre carpetas: se espera a que el usuario deje de escribir.
        self._input_check_timer = QTimer(self)
        self._input_check_timer.setSingleShot(True)
        self._input_check_timer.setInterval(350)
        self._input_check_timer.timeout.connect(self._validate_audio_inputs)

        self.profile_manager = ProfileManager(Path.cwd() / "profiles")
        self.audio_analyzer = AudioAnalyzer(
            profile_manager=self.profile_manager,
            similarity_threshold=0.75,
            overlap_seconds=self.DEFAULTS["overlap"],
            threads=self.DEFAULTS["threads"],
        )

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.75)
        self._temp_playback_path: Path | None = None

        self._init_ui()
        self._load_settings()
        self._refresh_export_summary()
        self._validate_audio_inputs()
        self._validate_zone_inputs()
        self._refresh_profile_library()
        QTimer.singleShot(400, self._load_saved_validation_table)

    def _init_ui(self) -> None:
        self.setWindowTitle(APP_TITLE)
        icon_path = _resource_path("assets/app.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setMinimumSize(1300, 820)
        self.setStyleSheet(build_stylesheet(_resource_path("assets/ui")))

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header = self._build_header()
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([520, 780])
        root_layout.addWidget(splitter, 1)

        self._configure_tab_order()

    def _build_header(self) -> QWidget:
        card = QFrame()
        card.setObjectName("HeaderCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title = QLabel(APP_TITLE)
        title.setObjectName("TitleLabel")
        subtitle = QLabel("Monitoreo bioacústico · BirdNET V2.4 y clasificador local CONDESAN")
        subtitle.setObjectName("SubtitleLabel")
        text_col.addWidget(title)
        text_col.addWidget(subtitle)
        layout.addLayout(text_col, 1)

        self.status_chip = QLabel("Listo")
        self.status_chip.setObjectName("StatusChipReady")
        layout.addWidget(self.status_chip, 0, Qt.AlignVCenter)
        return card

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        flow_hint = QLabel("Configuración por pasos")
        flow_hint.setObjectName("HintLabel")
        layout.addWidget(flow_hint)

        self.steps_tabs = QTabWidget()
        self.steps_tabs.addTab(self._build_step_audio(), "1. Audio")
        self.steps_tabs.addTab(self._build_step_zone(), "2. Zona")
        self.steps_tabs.addTab(self._build_step_params(), "3. Parámetros")
        self.steps_tabs.addTab(self._build_step_export(), "4. Exportación")
        layout.addWidget(self.steps_tabs, 1)

        actions = QHBoxLayout()
        self.run_btn = QPushButton("Iniciar análisis")
        self.run_btn.setObjectName("PrimaryAction")
        self.run_btn.clicked.connect(self._start_analysis)

        self.stop_btn = QPushButton("Detener")
        self.stop_btn.setObjectName("DangerAction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_analysis)

        self.open_output_btn = QPushButton("Abrir salida")
        self.open_output_btn.clicked.connect(self._open_output_folder)

        actions.addWidget(self.run_btn)
        actions.addWidget(self.stop_btn)
        actions.addWidget(self.open_output_btn)
        layout.addLayout(actions)
        return panel

    def _build_step_audio(self) -> QWidget:
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Archivo de audio o carpeta")
        self.input_path_edit.textChanged.connect(self._schedule_input_validation)

        pick_file_btn = QPushButton("Archivo…")
        pick_file_btn.clicked.connect(self._browse_input_file)
        pick_folder_btn = QPushButton("Carpeta…")
        pick_folder_btn.clicked.connect(self._browse_input_folder)

        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(str(Path.cwd() / "output"))
        self.output_path_edit.textChanged.connect(self._schedule_input_validation)
        pick_out_btn = QPushButton("Salida…")
        pick_out_btn.clicked.connect(self._browse_output_folder)

        self.input_validation_label = QLabel("")
        self.input_validation_label.setWordWrap(True)

        grid.addWidget(QLabel("Entrada"), 0, 0)
        grid.addWidget(self.input_path_edit, 0, 1)
        grid.addWidget(pick_file_btn, 0, 2)
        grid.addWidget(pick_folder_btn, 0, 3)

        grid.addWidget(QLabel("Carpeta de salida"), 1, 0)
        grid.addWidget(self.output_path_edit, 1, 1)
        grid.addWidget(pick_out_btn, 1, 2)

        tip = QLabel("Se comprueba que la ruta exista y contenga audios compatibles (WAV, FLAC, MP3, OGG, M4A, WMA, AIFF).")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        grid.addWidget(tip, 2, 0, 1, 4)
        grid.addWidget(self.input_validation_label, 3, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(4, 1)
        return tab

    def _build_step_zone(self) -> QWidget:
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)

        zone_mode_box = QGroupBox("¿Conoces la ubicación de la grabación?")
        zone_mode_layout = QVBoxLayout(zone_mode_box)
        zone_mode_layout.setContentsMargins(10, 8, 10, 8)
        zone_mode_layout.setSpacing(6)

        self.radio_zone_no = QRadioButton("🌍 No, no sé dónde fue grabada (modo global)")
        self.radio_zone_no.setChecked(True)
        self.radio_zone_no.setToolTip("No se filtrarán especies por ubicación.")
        self.radio_zone_yes = QRadioButton("📍 Sí, indicar zona")
        self.radio_zone_yes.setToolTip("Filtra las especies probables según las coordenadas.")
        self.radio_zone_yes.toggled.connect(self._on_zone_mode_toggled)
        self.radio_zone_no.toggled.connect(self._validate_zone_inputs)

        zone_mode_layout.addWidget(self.radio_zone_no)
        zone_mode_layout.addWidget(self.radio_zone_yes)

        self.country_edit = QLineEdit()
        self.country_edit.setPlaceholderText("País")
        self.country_edit.textChanged.connect(self._validate_zone_inputs)

        self.province_edit = QLineEdit()
        self.province_edit.setPlaceholderText("Provincia/Estado")
        self.province_edit.textChanged.connect(self._validate_zone_inputs)

        self.canton_edit = QLineEdit()
        self.canton_edit.setPlaceholderText("Cantón/Ciudad")
        self.canton_edit.textChanged.connect(self._validate_zone_inputs)

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setDecimals(6)
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setSingleStep(0.001)
        self.lat_spin.valueChanged.connect(self._validate_zone_inputs)

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setDecimals(6)
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setSingleStep(0.001)
        self.lon_spin.valueChanged.connect(self._validate_zone_inputs)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setDecimals(1)
        self.radius_spin.setRange(1.0, 200.0)
        self.radius_spin.setSingleStep(1.0)
        self.radius_spin.setValue(25.0)
        self.radius_spin.setSuffix(" km")

        self.zone_validation_label = QLabel("")
        self.zone_validation_label.setWordWrap(True)

        self.zone_inputs_container = QWidget()
        zone_grid = QGridLayout(self.zone_inputs_container)
        zone_grid.setContentsMargins(0, 0, 0, 0)
        zone_grid.setHorizontalSpacing(8)
        zone_grid.setVerticalSpacing(10)
        zone_grid.addWidget(QLabel("País"), 0, 0)
        zone_grid.addWidget(self.country_edit, 0, 1)
        zone_grid.addWidget(QLabel("Provincia"), 0, 2)
        zone_grid.addWidget(self.province_edit, 0, 3)
        zone_grid.addWidget(QLabel("Cantón/Ciudad"), 1, 0)
        zone_grid.addWidget(self.canton_edit, 1, 1)
        zone_grid.addWidget(QLabel("Latitud"), 1, 2)
        zone_grid.addWidget(self.lat_spin, 1, 3)
        zone_grid.addWidget(QLabel("Longitud"), 2, 0)
        zone_grid.addWidget(self.lon_spin, 2, 1)
        zone_grid.addWidget(QLabel("Radio"), 2, 2)
        zone_grid.addWidget(self.radius_spin, 2, 3)

        grid.addWidget(zone_mode_box, 0, 0, 1, 4)
        grid.addWidget(self.zone_inputs_container, 1, 0, 1, 4)
        grid.addWidget(self.zone_validation_label, 2, 0, 1, 4)

        tip = QLabel("BirdNET usa la latitud y la longitud para filtrar especies; país, provincia y cantón se guardan como contexto.")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        grid.addWidget(tip, 3, 0, 1, 4)
        grid.setRowStretch(4, 1)
        self._on_zone_mode_toggled(self.radio_zone_yes.isChecked())
        return tab

    def _build_step_params(self) -> QWidget:
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._apply_preset)

        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setRange(0.0, 2.9)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setSuffix(" s")
        self.overlap_spin.setToolTip("Solape entre ventanas de 3 segundos.")

        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setDecimals(2)
        self.sensitivity_spin.setRange(0.5, 1.5)
        self.sensitivity_spin.setSingleStep(0.05)
        self.sensitivity_spin.setToolTip(
            "Sensibilidad de BirdNET en el modo descubrimiento (0,5–1,5). Valores altos detectan más, "
            "con más falsos positivos. No afecta al clasificador local, que usa umbrales calibrados."
        )

        self.min_conf_spin = QDoubleSpinBox()
        self.min_conf_spin.setDecimals(2)
        self.min_conf_spin.setRange(0.01, 0.99)
        self.min_conf_spin.setSingleStep(0.01)
        self.min_conf_spin.setValue(0.75)
        self.min_conf_spin.setToolTip("Similitud coseno mínima para confirmar una especie de la biblioteca de perfiles.")

        self.anomaly_threshold_spin = QDoubleSpinBox()
        self.anomaly_threshold_spin.setDecimals(2)
        self.anomaly_threshold_spin.setRange(0.05, 0.99)
        self.anomaly_threshold_spin.setSingleStep(0.01)
        self.anomaly_threshold_spin.setValue(0.45)
        self.anomaly_threshold_spin.setToolTip(
            "Confianza mínima de BirdNET para marcar un evento como anomalía acústica."
        )

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, max(2, os.cpu_count() or 8))
        self.threads_spin.setToolTip("Hilos de CPU para el análisis.")

        self.multi_vector_check = QCheckBox("Alta precisión (varios prototipos por especie)")
        self.multi_vector_check.setChecked(True)
        self.multi_vector_check.setToolTip("Compara cada segmento con varios prototipos de cada especie.")

        self.noise_reduction_check = QCheckBox("Reducción de ruido (DSP)")
        self.noise_reduction_check.setChecked(False)
        self.noise_reduction_check.setToolTip(
            "Aplica noisereduce antes de extraer embeddings y en el espectrograma y la reproducción."
        )

        self.smoothing_window_spin = QSpinBox()
        self.smoothing_window_spin.setRange(1, 11)
        self.smoothing_window_spin.setSingleStep(1)
        self.smoothing_window_spin.setValue(3)
        self.smoothing_window_spin.setSuffix(" segmentos")
        self.smoothing_window_spin.setToolTip("Ventana del suavizado temporal de las detecciones de perfiles.")

        self.discovery_mode_check = QCheckBox("Modo descubrimiento (BirdNET global, 6 522 especies)")
        self.discovery_mode_check.setChecked(False)
        self.discovery_mode_check.setToolTip(
            "Reporta especies conocidas por BirdNET aunque no estén en tu biblioteca local."
        )

        self.local_classifier_check = QCheckBox("Usar clasificador local CONDESAN (BirdNET calibrado)")
        self.local_classifier_check.setChecked(False)
        if self.audio_analyzer.local_classifier_available():
            self.local_classifier_check.setToolTip(
                "Detecta las especies del bosque seco validadas por CONDESAN usando la capa de BirdNET "
                "con umbrales calibrados por especie. No requiere perfiles en la Biblioteca de Especies."
            )
        else:
            self.local_classifier_check.setEnabled(False)
            self.local_classifier_check.setToolTip("Modelo no encontrado en local_models/condesan.")

        restore_btn = QPushButton("Restaurar valores")
        restore_btn.clicked.connect(self._restore_defaults)

        grid.addWidget(QLabel("Preset"), 0, 0)
        grid.addWidget(self.preset_combo, 0, 1)
        grid.addWidget(restore_btn, 0, 2, 1, 2)

        grid.addWidget(QLabel("Solape"), 1, 0)
        grid.addWidget(self.overlap_spin, 1, 1)
        grid.addWidget(QLabel("Sensibilidad"), 1, 2)
        grid.addWidget(self.sensitivity_spin, 1, 3)

        grid.addWidget(QLabel("Umbral de similitud"), 2, 0)
        grid.addWidget(self.min_conf_spin, 2, 1)
        grid.addWidget(QLabel("Hilos"), 2, 2)
        grid.addWidget(self.threads_spin, 2, 3)

        grid.addWidget(self.multi_vector_check, 3, 0, 1, 2)
        grid.addWidget(QLabel("Ventana de suavizado"), 3, 2)
        grid.addWidget(self.smoothing_window_spin, 3, 3)
        grid.addWidget(self.noise_reduction_check, 4, 0, 1, 2)
        grid.addWidget(QLabel("Umbral de anomalía"), 4, 2)
        grid.addWidget(self.anomaly_threshold_spin, 4, 3)
        grid.addWidget(self.discovery_mode_check, 5, 0, 1, 4)
        grid.addWidget(self.local_classifier_check, 6, 0, 1, 4)

        tip = QLabel(
            "Consejo: para fauna densa usa «Alta sensibilidad»; para priorizar la precisión, «Precisión alta». "
            "Para comparar con TablaFinal en igualdad de condiciones usa solape 0 s."
        )
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        grid.addWidget(tip, 7, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(8, 1)
        return tab

    def _build_step_export(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.export_summary = QLabel("")
        self.export_summary.setWordWrap(True)
        self.export_summary.setObjectName("HintLabel")

        export_btn = QPushButton("Formatos de exportación…")
        export_btn.clicked.connect(self._open_export_dialog)

        meta_hint = QLabel("Define los formatos de salida y si se incluyen los metadatos de zona.")
        meta_hint.setObjectName("HintLabel")
        meta_hint.setWordWrap(True)

        layout.addWidget(self.export_summary)
        layout.addWidget(export_btn, 0, Qt.AlignLeft)
        layout.addWidget(meta_hint)
        layout.addStretch(1)
        return tab

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        runtime = QGroupBox("Ejecución")
        runtime_grid = QGridLayout(runtime)
        runtime_grid.setHorizontalSpacing(10)
        runtime_grid.setVerticalSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Listo")
        self.progress_label = QLabel("Segmentos: 0/0")
        self.elapsed_label = QLabel("Tiempo: 00:00:00")

        runtime_grid.addWidget(self.progress_bar, 0, 0, 1, 4)
        runtime_grid.addWidget(self.progress_label, 1, 0, 1, 2)
        runtime_grid.addWidget(self.elapsed_label, 1, 2, 1, 2)

        self.metric_files = QLabel("0/0")
        self.metric_detections = QLabel("0")
        self.metric_species = QLabel("0")
        self.metric_conf = QLabel("—")

        runtime_grid.addWidget(self._metric_card("Segmentos analizados", self.metric_files), 2, 0)
        runtime_grid.addWidget(self._metric_card("Detecciones", self.metric_detections), 2, 1)
        runtime_grid.addWidget(self._metric_card("Especies", self.metric_species), 2, 2)
        runtime_grid.addWidget(self._metric_card("Confianza media", self.metric_conf), 2, 3)
        layout.addWidget(runtime)

        self.result_tabs = QTabWidget()
        self.summary_tab = self._build_summary_tab()
        self.detections_tab = self._build_detections_tab()
        self.validation_tab = self._build_validation_tab()
        self.library_tab = self._build_species_library_tab()
        self.logs_tab = self._build_logs_tab()
        self.result_tabs.addTab(self.summary_tab, "Resumen")
        self.result_tabs.addTab(self.detections_tab, "Detecciones")
        self.result_tabs.addTab(self.validation_tab, "Validación")
        self.result_tabs.addTab(self.library_tab, "Biblioteca de especies")
        self.result_tabs.addTab(self.logs_tab, "Registro")
        layout.addWidget(self.result_tabs, 1)
        return panel

    def _metric_card(self, label: str, value_label: QLabel, caption: QLabel | None = None) -> QWidget:
        card = QFrame()
        card.setObjectName("MetricCard")
        box = QVBoxLayout(card)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(2)

        title = QLabel(label)
        title.setObjectName("MetricTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        value_label.setObjectName("MetricValue")
        value_label.setAlignment(Qt.AlignCenter)
        box.addWidget(title)
        box.addWidget(value_label)
        if caption is not None:
            caption.setObjectName("MetricCaption")
            caption.setAlignment(Qt.AlignCenter)
            caption.setWordWrap(True)
            box.addWidget(caption)
        return card

    def _build_summary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.summary_headline = QLabel("Sin resultados aún.")
        self.summary_headline.setObjectName("SectionLabel")
        self.summary_headline.setWordWrap(True)
        layout.addWidget(self.summary_headline)

        self.species_chart = QChart()
        self.species_chart.setTitle("Especies más detectadas")
        self.species_chart.legend().setVisible(False)
        self.species_chart.setAnimationOptions(QChart.NoAnimation)
        self.species_chart.setBackgroundRoundness(0)
        self.species_chart_view = QChartView(self.species_chart)
        self.species_chart_view.setRenderHint(QPainter.Antialiasing)
        self.species_chart_view.setMinimumHeight(280)
        layout.addWidget(self.species_chart_view, 3)

        self.top_species_table = QTableWidget(0, 3)
        self.top_species_table.setHorizontalHeaderLabels(["Especie", "Detecciones", "Confianza media"])
        self.top_species_table.verticalHeader().setVisible(False)
        self.top_species_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.top_species_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.top_species_table.setItemDelegateForColumn(2, ConfidenceDelegate(self.top_species_table))
        header = self.top_species_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.top_species_table.setColumnWidth(2, 140)
        layout.addWidget(self.top_species_table, 2)
        return tab

    def _build_detections_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        self.play_btn = QPushButton("▶  Reproducir selección")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_selected)
        hint = QLabel("Doble clic en una fila para escucharla · clic derecho para añadirla a un perfil")
        hint.setObjectName("HintLabel")
        toolbar.addWidget(self.play_btn)
        toolbar.addWidget(hint, 1)
        layout.addLayout(toolbar)

        self.detections_table = QTableWidget(0, 7)
        self.detections_table.setHorizontalHeaderLabels(
            ["Inicio (s)", "Fin (s)", "Especie", "Nombre científico", "Confianza", "Fuente", "Archivo"]
        )
        self.detections_table.verticalHeader().setVisible(False)
        self.detections_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.detections_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.detections_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.detections_table.setAlternatingRowColors(True)
        self.detections_table.setItemDelegateForColumn(4, ConfidenceDelegate(self.detections_table))
        header = self.detections_table.horizontalHeader()
        for col, mode in enumerate(
            (
                QHeaderView.ResizeToContents,
                QHeaderView.ResizeToContents,
                QHeaderView.Stretch,
                QHeaderView.Stretch,
                QHeaderView.Fixed,
                QHeaderView.ResizeToContents,
                QHeaderView.Interactive,
            )
        ):
            header.setSectionResizeMode(col, mode)
        self.detections_table.setColumnWidth(4, 120)
        self.detections_table.setColumnWidth(6, 200)
        self.detections_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.detections_table.customContextMenuRequested.connect(self._show_detections_context_menu)
        self.detections_table.itemSelectionChanged.connect(self._on_detection_selection_changed)
        self.detections_table.cellDoubleClicked.connect(lambda _row, _col: self._play_selected())

        self.spectrogram_widget = SpectrogramWidget()
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.detections_table)
        splitter.addWidget(self.spectrogram_widget)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([320, 260])
        layout.addWidget(splitter)
        return tab

    def _build_validation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        top = QHBoxLayout()
        self.load_validation_btn = QPushButton("Cargar tabla de validaciones…")
        self.load_validation_btn.setToolTip("TablaFinal.xlsx del proyecto CONDESAN, u otra tabla con las mismas columnas.")
        self.load_validation_btn.clicked.connect(self._choose_validation_table)
        self.validation_source_label = QLabel("Sin tabla cargada.")
        self.validation_source_label.setObjectName("HintLabel")
        self.validation_source_label.setWordWrap(True)
        top.addWidget(self.load_validation_btn)
        top.addWidget(self.validation_source_label, 1)
        layout.addLayout(top)

        selector = QHBoxLayout()
        selector.addWidget(QLabel("Grabación"))
        self.validation_combo = QComboBox()
        self.validation_combo.currentIndexChanged.connect(self._on_validation_selection)
        selector.addWidget(self.validation_combo, 1)
        layout.addLayout(selector)

        kpis = QHBoxLayout()
        kpis.setSpacing(8)
        self.kpi_recall, self.kpi_recall_caption = QLabel("—"), QLabel("")
        self.kpi_confirmed, self.kpi_confirmed_caption = QLabel("—"), QLabel("")
        self.kpi_species, self.kpi_species_caption = QLabel("—"), QLabel("")
        self.kpi_conf, self.kpi_conf_caption = QLabel("—"), QLabel("")
        kpis.addWidget(self._metric_card("Coincidencia con expertos", self.kpi_recall, self.kpi_recall_caption))
        kpis.addWidget(self._metric_card("Detecciones confirmadas", self.kpi_confirmed, self.kpi_confirmed_caption))
        kpis.addWidget(self._metric_card("Especies validadas encontradas", self.kpi_species, self.kpi_species_caption))
        kpis.addWidget(self._metric_card("Confianza media de la app", self.kpi_conf, self.kpi_conf_caption))
        layout.addLayout(kpis)

        self.validation_figure = Figure(figsize=(7, 4.4), layout="constrained")
        self.validation_canvas = FigureCanvasQTAgg(self.validation_figure)
        self.validation_canvas.setMinimumHeight(260)

        self.validation_species_table = QTableWidget(0, 6)
        self.validation_species_table.setHorizontalHeaderLabels(
            ["Especie", "Validadas", "Detectadas", "Coinciden", "Coincidencia", "Confianza"]
        )
        self.validation_species_table.verticalHeader().setVisible(False)
        self.validation_species_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.validation_species_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.validation_species_table.setAlternatingRowColors(True)
        self.validation_species_table.setItemDelegateForColumn(4, ConfidenceDelegate(self.validation_species_table))
        self.validation_species_table.setItemDelegateForColumn(5, ConfidenceDelegate(self.validation_species_table))
        vheader = self.validation_species_table.horizontalHeader()
        vheader.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            vheader.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        for col in (4, 5):
            vheader.setSectionResizeMode(col, QHeaderView.Fixed)
            self.validation_species_table.setColumnWidth(col, 120)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.validation_canvas)
        splitter.addWidget(self.validation_species_table)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([440, 150])
        layout.addWidget(splitter, 1)

        note = QLabel(
            "«Coincidencia con expertos»: detecciones validadas por el equipo que la app también encuentra "
            "(misma especie, ≥ 0,5 s de solape). «Sin validar» no significa error: el equipo solo revisó lo que BirdNET propuso."
        )
        note.setObjectName("HintLabel")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._show_validation_message("Carga TablaFinal.xlsx y analiza una grabación de CONDESAN para comparar.")
        return tab

    def _build_species_library_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        add_btn = QPushButton("Agregar especie")
        add_btn.clicked.connect(self._add_reference_species)
        refresh_btn = QPushButton("Recargar")
        refresh_btn.clicked.connect(self._refresh_profile_library)
        remove_btn = QPushButton("Eliminar seleccionada")
        remove_btn.setObjectName("DangerAction")
        remove_btn.clicked.connect(self._remove_selected_profile)

        controls.addWidget(add_btn)
        controls.addWidget(refresh_btn)
        controls.addWidget(remove_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.profile_table = QTableWidget(0, 3)
        self.profile_table.setHorizontalHeaderLabels(["Especie", "Audio de referencia", "Vectores"])
        self.profile_table.verticalHeader().setVisible(False)
        self.profile_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.profile_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.profile_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.profile_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.profile_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.profile_table.itemSelectionChanged.connect(self._on_profile_selected)
        layout.addWidget(self.profile_table)

        self.profile_stats_label = QLabel("Selecciona una especie para ver sus detecciones en esta sesión.")
        self.profile_stats_label.setObjectName("HintLabel")
        self.profile_stats_label.setWordWrap(True)
        layout.addWidget(self.profile_stats_label)
        return tab

    def _build_logs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.logs_group = QGroupBox("Registro del proceso")
        self.logs_group.setCheckable(True)
        self.logs_group.setChecked(True)
        group_layout = QVBoxLayout(self.logs_group)
        group_layout.setSpacing(8)

        self.logs_body = QWidget()
        body_layout = QVBoxLayout(self.logs_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Filtro"))
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItems(["Todos", "Info", "Warning", "Error"])
        self.log_filter_combo.currentTextChanged.connect(self._refresh_log_view)
        controls.addWidget(self.log_filter_combo)

        copy_btn = QPushButton("Copiar")
        copy_btn.clicked.connect(self._copy_logs)
        clear_btn = QPushButton("Limpiar")
        clear_btn.clicked.connect(self._clear_logs)
        controls.addStretch(1)
        controls.addWidget(copy_btn)
        controls.addWidget(clear_btn)
        body_layout.addLayout(controls)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self.log_view.setFont(mono)
        body_layout.addWidget(self.log_view)

        group_layout.addWidget(self.logs_body)
        self.logs_group.toggled.connect(self.logs_body.setVisible)
        layout.addWidget(self.logs_group)
        return tab

    def _configure_tab_order(self) -> None:
        QWidget.setTabOrder(self.input_path_edit, self.output_path_edit)
        QWidget.setTabOrder(self.output_path_edit, self.radio_zone_no)
        QWidget.setTabOrder(self.radio_zone_no, self.radio_zone_yes)
        QWidget.setTabOrder(self.radio_zone_yes, self.country_edit)
        QWidget.setTabOrder(self.country_edit, self.province_edit)
        QWidget.setTabOrder(self.province_edit, self.canton_edit)
        QWidget.setTabOrder(self.canton_edit, self.lat_spin)
        QWidget.setTabOrder(self.lat_spin, self.lon_spin)
        QWidget.setTabOrder(self.lon_spin, self.radius_spin)
        QWidget.setTabOrder(self.radius_spin, self.preset_combo)
        QWidget.setTabOrder(self.preset_combo, self.overlap_spin)
        QWidget.setTabOrder(self.overlap_spin, self.sensitivity_spin)
        QWidget.setTabOrder(self.sensitivity_spin, self.min_conf_spin)
        QWidget.setTabOrder(self.min_conf_spin, self.threads_spin)
        QWidget.setTabOrder(self.threads_spin, self.multi_vector_check)
        QWidget.setTabOrder(self.multi_vector_check, self.noise_reduction_check)
        QWidget.setTabOrder(self.noise_reduction_check, self.smoothing_window_spin)
        QWidget.setTabOrder(self.smoothing_window_spin, self.anomaly_threshold_spin)
        QWidget.setTabOrder(self.anomaly_threshold_spin, self.discovery_mode_check)
        QWidget.setTabOrder(self.discovery_mode_check, self.local_classifier_check)
        QWidget.setTabOrder(self.local_classifier_check, self.run_btn)

    def _browse_input_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo de audio",
            str(Path.home()),
            "Audio (*.wav *.flac *.mp3 *.ogg *.m4a *.wma *.aiff *.aif);;Todos (*.*)",
        )
        if path:
            self.input_path_edit.setText(path)

    def _browse_input_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de audios", str(Path.home()))
        if path:
            self.input_path_edit.setText(path)

    def _browse_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida", self.output_path_edit.text().strip() or str(Path.cwd()))
        if path:
            self.output_path_edit.setText(path)

    def _open_output_folder(self) -> None:
        output = self.output_path_edit.text().strip()
        if not output:
            QMessageBox.warning(self, "Salida vacía", "Define primero la carpeta de salida.")
            return
        Path(output).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(output))

    def _validate_audio_inputs(self) -> bool:
        input_path = self.input_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()

        if not input_path:
            self.input_validation_label.setObjectName("ErrorLabel")
            self.input_validation_label.setText("Selecciona un archivo de audio o carpeta.")
            self._repolish(self.input_validation_label)
            return False

        p = Path(input_path)
        if not p.exists():
            self.input_validation_label.setObjectName("ErrorLabel")
            self.input_validation_label.setText("La ruta de entrada no existe.")
            self._repolish(self.input_validation_label)
            return False

        if p.is_file() and p.suffix.lower() not in AUDIO_EXTENSIONS:
            self.input_validation_label.setObjectName("ErrorLabel")
            self.input_validation_label.setText("Extensión de audio no soportada.")
            self._repolish(self.input_validation_label)
            return False

        total = self._estimate_total_files(input_path)
        if total == 0:
            self.input_validation_label.setObjectName("ErrorLabel")
            self.input_validation_label.setText("No se encontraron audios compatibles en la ruta.")
            self._repolish(self.input_validation_label)
            return False

        if not output_path:
            self.input_validation_label.setObjectName("ErrorLabel")
            self.input_validation_label.setText("Define una carpeta de salida.")
            self._repolish(self.input_validation_label)
            return False

        self.input_validation_label.setObjectName("OkLabel")
        self.input_validation_label.setText(f"Entrada válida. Audios detectados: {total}")
        self._repolish(self.input_validation_label)
        return True

    def _schedule_input_validation(self) -> None:
        if hasattr(self, "_input_check_timer"):
            self._input_check_timer.start()

    def _on_zone_mode_toggled(self, enabled: bool) -> None:
        if hasattr(self, "zone_inputs_container"):
            self.zone_inputs_container.setEnabled(bool(enabled))
        self._validate_zone_inputs()

    def _validate_zone_inputs(self) -> bool:
        if self.radio_zone_no.isChecked():
            self.zone_validation_label.setObjectName("OkLabel")
            self.zone_validation_label.setText("Modo global: no se aplicará filtro geográfico.")
            self._repolish(self.zone_validation_label)
            return True

        lat = self.lat_spin.value()
        lon = self.lon_spin.value()

        if lat == 0.0 and lon == 0.0:
            self.zone_validation_label.setObjectName("ErrorLabel")
            self.zone_validation_label.setText("Activaste el filtro de zona: define una latitud y longitud válidas (distintas de 0,0).")
            self._repolish(self.zone_validation_label)
            return False

        self.zone_validation_label.setObjectName("OkLabel")
        self.zone_validation_label.setText("Zona activa para filtro de especies.")
        self._repolish(self.zone_validation_label)
        return True

    def _open_export_dialog(self) -> None:
        dialog = ExportOptionsDialog(self.export_options, self)
        if dialog.exec() == QDialog.Accepted:
            self.export_options = ExportOptions(
                formats=dialog.selected_formats(),
                include_zone_metadata=dialog.include_zone_metadata(),
            )
            self._refresh_export_summary()

    def _refresh_export_summary(self) -> None:
        labels = []
        for fmt in self.export_options.formats:
            if fmt == "table":
                labels.append("Raven")
            elif fmt == "csv":
                labels.append("CSV")
            elif fmt == "audacity":
                labels.append("Audacity")
            elif fmt == "kaleidoscope":
                labels.append("Kaleidoscope")
        formats_txt = ", ".join(labels) if labels else "Ninguno"
        zone_txt = "Si" if self.export_options.include_zone_metadata else "No"
        self.export_summary.setText(f"Formatos: {formats_txt}\nIncluir metadatos de zona en CSV: {zone_txt}")

    def _apply_preset(self, preset_name: str) -> None:
        cfg = self.PRESETS.get(preset_name)
        if not cfg:
            return
        self.overlap_spin.setValue(cfg["overlap"])
        self.sensitivity_spin.setValue(cfg["sensitivity"])
        self.min_conf_spin.setValue(cfg["min_conf"])
        self.threads_spin.setValue(cfg["threads"])

    def _restore_defaults(self) -> None:
        self.overlap_spin.setValue(self.DEFAULTS["overlap"])
        self.sensitivity_spin.setValue(self.DEFAULTS["sensitivity"])
        self.min_conf_spin.setValue(self.DEFAULTS["min_conf"])
        self.threads_spin.setValue(self.DEFAULTS["threads"])
        self.multi_vector_check.setChecked(bool(self.DEFAULTS["use_multi_vector"]))
        self.noise_reduction_check.setChecked(bool(self.DEFAULTS["use_noise_reduction"]))
        self.anomaly_threshold_spin.setValue(float(self.DEFAULTS["anomaly_threshold"]))
        self.smoothing_window_spin.setValue(int(self.DEFAULTS["smoothing_window"]))
        self.discovery_mode_check.setChecked(bool(self.DEFAULTS["use_global_discovery"]))
        self.local_classifier_check.setChecked(bool(self.DEFAULTS["use_local_classifier"]))
        self.preset_combo.setCurrentText("Balanceado")

    def _refresh_profile_library(self) -> None:
        if not hasattr(self, "profile_table"):
            return
        profiles = self.profile_manager.list_profiles()
        self.profile_table.setRowCount(0)
        for idx, profile in enumerate(profiles):
            self.profile_table.insertRow(idx)
            self.profile_table.setItem(idx, 0, QTableWidgetItem(profile.name))
            self.profile_table.setItem(idx, 1, QTableWidgetItem(Path(profile.source_audio).name))
            self.profile_table.setItem(idx, 2, QTableWidgetItem(f"{profile.num_vectors}x{profile.vector_dim}"))

        if profiles:
            self.profile_stats_label.setText(
                f"Perfiles cargados: {len(profiles)}. Selecciona una especie para ver sus detecciones en esta sesión."
            )
        else:
            self.profile_stats_label.setText(
                "No hay perfiles de referencia. Agrega una especie con un clip .wav/.flac para habilitar el motor por similitud."
            )

    def _add_reference_species(self) -> None:
        name, ok = QInputDialog.getText(self, "Nueva especie", "Nombre de especie personalizada:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Nombre requerido", "Debes ingresar un nombre para la especie.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar audio de referencia",
            str(Path.home()),
            "Audio (*.wav *.flac *.mp3 *.ogg *.m4a *.wma *.aiff *.aif);;Todos (*.*)",
        )
        if not file_path:
            return

        try:
            self._add_log("Info", f"Generando embedding de referencia para '{name}' ...")
            profile, chunks = self.audio_analyzer.add_reference_profile(
                name,
                file_path,
                use_multi_vector=self.multi_vector_check.isChecked(),
            )
            self._add_log("Info", f"Perfil guardado: {profile.name} ({chunks} chunks usados).")
            self._refresh_profile_library()
        except Exception as ex:
            QMessageBox.critical(self, "Error al crear perfil", str(ex))
            self._add_log("Error", f"No se pudo crear perfil '{name}': {ex}")

    def _remove_selected_profile(self) -> None:
        row = self.profile_table.currentRow() if hasattr(self, "profile_table") else -1
        if row < 0:
            QMessageBox.information(self, "Selección requerida", "Selecciona una especie para eliminar.")
            return
        name_item = self.profile_table.item(row, 0)
        if not name_item:
            return
        profile_name = name_item.text().strip()
        if not profile_name:
            return

        ans = QMessageBox.question(
            self,
            "Eliminar perfil",
            f"Se eliminará el perfil '{profile_name}'.\n¿Deseas continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return

        if self.profile_manager.remove_profile(profile_name):
            self._add_log("Warning", f"Perfil eliminado: {profile_name}")
            self._refresh_profile_library()
        else:
            QMessageBox.warning(self, "No encontrado", "No se pudo eliminar el perfil seleccionado.")

    def _on_profile_selected(self) -> None:
        row = self.profile_table.currentRow() if hasattr(self, "profile_table") else -1
        if row < 0:
            return
        name_item = self.profile_table.item(row, 0)
        if not name_item:
            return
        name = name_item.text().strip()
        count = self.session_detection_counts.get(name, 0)
        self.profile_stats_label.setText(f"Detecciones de '{name}' en la sesión actual: {count}")

    def _estimate_total_files(self, input_path: str) -> int:
        p = Path(input_path)
        if not p.exists():
            return 0
        if p.is_file():
            return 1 if p.suffix.lower() in AUDIO_EXTENSIONS else 0
        # Sin distinguir mayusculas: AudioMoth guarda los archivos como .WAV.
        return sum(1 for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file())

    def _start_analysis(self) -> None:
        if not self._validate_audio_inputs():
            self.steps_tabs.setCurrentIndex(0)
            return
        if not self._validate_zone_inputs():
            self.steps_tabs.setCurrentIndex(1)
            return
        if not self.export_options.formats:
            self.steps_tabs.setCurrentIndex(3)
            QMessageBox.warning(self, "Exportación", "Selecciona al menos un formato de salida.")
            return
        profiles = self.profile_manager.list_profiles()
        use_local = self.local_classifier_check.isChecked() and self.local_classifier_check.isEnabled()
        if not profiles and not self.discovery_mode_check.isChecked() and not use_local:
            self.result_tabs.setCurrentWidget(self.library_tab)
            QMessageBox.warning(
                self,
                "Perfiles requeridos",
                "Debes agregar al menos una especie en 'Biblioteca de Especies', activar Modo Descubrimiento "
                "o activar el clasificador local CONDESAN.",
            )
            return

        output_dir = Path(self.output_path_edit.text().strip())
        output_dir.mkdir(parents=True, exist_ok=True)

        self.result_rows = []
        self.session_detection_counts.clear()
        self.analyzed_files = [str(p) for p in self.audio_analyzer._collect_audio_files(self.input_path_edit.text().strip())]
        self._clear_results_view()
        self._clear_logs()
        self._set_running_state(True)
        self._set_status("running")
        if use_local:
            self._add_log("Info", "Clasificador local CONDESAN activado (especies locales con umbrales calibrados).")
        if not profiles and self.discovery_mode_check.isChecked():
            self._add_log("Warning", "Sin perfiles locales: se usará solo BirdNET global.")
        elif self.discovery_mode_check.isChecked():
            self._add_log("Info", "Iniciando análisis híbrido (perfiles + BirdNET global)…")
        else:
            self._add_log("Info", "Iniciando análisis…")

        self.audio_analyzer.overlap_seconds = self.overlap_spin.value()
        self.audio_analyzer.threads = self.threads_spin.value()
        self.audio_analyzer.similarity_threshold = self.min_conf_spin.value()
        self.audio_analyzer.anomaly_threshold = self.anomaly_threshold_spin.value()
        self.audio_analyzer.use_noise_reduction = self.noise_reduction_check.isChecked()

        self.total_files = 1
        self.processed_files = 0
        self.metric_files.setText("0/0")
        self.progress_label.setText("Segmentos: 0/0")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Analizando…")

        self.elapsed_timer.start()
        self.elapsed_tick.start(1000)

        zone_meta = None
        if self.export_options.include_zone_metadata and self.radio_zone_yes.isChecked():
            zone_meta = {
                "country": self.country_edit.text().strip(),
                "province": self.province_edit.text().strip(),
                "canton": self.canton_edit.text().strip(),
                "lat": f"{self.lat_spin.value():.6f}",
                "lon": f"{self.lon_spin.value():.6f}",
                "radius_km": f"{self.radius_spin.value():.1f}",
            }

        self.analysis_thread = QThread(self)
        self.analysis_worker = AudioWorker(
            self.audio_analyzer,
            self.input_path_edit.text().strip(),
            str(output_dir),
            self.export_options.formats,
            zone_metadata=zone_meta,
            similarity_threshold=self.min_conf_spin.value(),
            anomaly_threshold=self.anomaly_threshold_spin.value(),
            overlap_seconds=self.overlap_spin.value(),
            threads=self.threads_spin.value(),
            use_noise_reduction=self.noise_reduction_check.isChecked(),
            use_multi_vector=self.multi_vector_check.isChecked(),
            smoothing_window=self.smoothing_window_spin.value(),
            use_global_discovery=self.discovery_mode_check.isChecked(),
            latitude=self.lat_spin.value() if self.radio_zone_yes.isChecked() else None,
            longitude=self.lon_spin.value() if self.radio_zone_yes.isChecked() else None,
            use_local_classifier=use_local,
            sensitivity=self.sensitivity_spin.value(),
        )
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self._on_worker_progress)
        self.analysis_worker.log.connect(self._add_log)
        self.analysis_worker.finished.connect(self._on_worker_finished)
        self.analysis_worker.failed.connect(self._on_worker_failed)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.failed.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
        self.analysis_thread.start()

    def _stop_analysis(self) -> None:
        if self.analysis_worker is not None:
            self._add_log("Warning", "Solicitud de detención enviada.")
            self.analysis_worker.request_stop()

    def _on_worker_progress(self, done: int, total: int, current_file: str, detections: int, avg_conf: float) -> None:
        self.processed_files = done
        self.total_files = total
        self.metric_files.setText(f"{done}/{total}")
        self.progress_label.setText(f"Segmentos: {done}/{total} · {Path(current_file).name}")
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(min(done, total))
        self.progress_bar.setFormat("%p %")
        self.metric_detections.setText(str(detections))
        self.metric_conf.setText(f"{avg_conf * 100:.0f} %" if detections else "—")

    def _on_worker_finished(self, detections: list[dict], summary: dict, exported_files: list[str]) -> None:
        self.elapsed_tick.stop()
        self._set_running_state(False)
        self._set_status("ready")
        self.result_rows = detections
        self.session_detection_counts = Counter(row.get("common", "N/A") for row in detections)
        self._populate_detections_table(detections)
        self._populate_summary(detections)
        self._refresh_profile_library()

        self.metric_files.setText(f"{summary.get('processed_segments', 0)}/{summary.get('total_segments', 0)}")
        self.progress_label.setText(
            f"Segmentos: {summary.get('processed_segments', 0)}/{summary.get('total_segments', 0)}"
        )
        self.progress_bar.setFormat("Completado")
        self.progress_bar.setRange(0, max(1, summary.get("total_segments", 1)))
        self.progress_bar.setValue(summary.get("processed_segments", 0))

        if summary.get("stopped"):
            self._add_log("Warning", "Análisis detenido por el usuario.")
        else:
            self._add_log("Info", f"Análisis completado. Detecciones: {summary.get('detections', 0)}.")

        self._refresh_validation()
        if self.validation_results:
            self._add_log("Info", f"Validación: {len(self.validation_results)} resultado(s) comparados con la tabla de expertos.")
            self.result_tabs.setCurrentWidget(self.validation_tab)

        for path in exported_files:
            self._add_log("Info", f"Exportado: {path}")

        self.analysis_worker = None
        self.analysis_thread = None

    def _on_worker_failed(self, message: str) -> None:
        self.elapsed_tick.stop()
        self._set_running_state(False)
        self._set_status("error")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Error")
        self._add_log("Error", f"Falló el análisis: {message}")
        self.analysis_worker = None
        self.analysis_thread = None
        QMessageBox.critical(self, "Error en el análisis", message)

    def _update_elapsed(self) -> None:
        if not self.elapsed_timer.isValid():
            return
        ms = self.elapsed_timer.elapsed()
        secs = ms // 1000
        hh = secs // 3600
        mm = (secs % 3600) // 60
        ss = secs % 60
        self.elapsed_label.setText(f"Tiempo: {hh:02}:{mm:02}:{ss:02}")

    def _row_payload_from_table(self, row_index: int) -> dict | None:
        if row_index < 0:
            return None
        first_item = self.detections_table.item(row_index, 0)
        payload = first_item.data(Qt.UserRole) if first_item is not None else None
        return dict(payload) if isinstance(payload, dict) else None

    def _on_detection_selection_changed(self) -> None:
        if not hasattr(self, "spectrogram_widget"):
            return
        payload = self._row_payload_from_table(self.detections_table.currentRow())
        self.play_btn.setEnabled(payload is not None)
        if not payload:
            self.spectrogram_widget.clear_plot()
            return
        self._update_spectrogram_for_payload(payload)

    def _play_selected(self) -> None:
        payload = self._row_payload_from_table(self.detections_table.currentRow())
        if payload:
            self._play_segment(payload)

    def _update_spectrogram_for_payload(self, row_payload: dict) -> None:
        audio_path = str(row_payload.get("file", "")).strip()
        if not audio_path or not Path(audio_path).exists():
            self.spectrogram_widget.clear_plot("Audio no disponible para el espectrograma.")
            return

        start = max(0.0, float(row_payload.get("start", 0.0)))
        end = max(start + 0.15, float(row_payload.get("end", start + 0.25)))
        common = str(row_payload.get("common", "")).strip() or "—"
        scientific = str(row_payload.get("scientific", "")).strip()
        source = SOURCE_LABELS.get(str(row_payload.get("source", "custom")).lower(), "")
        conf = float(row_payload.get("confidence", 0.0))
        title = f"{common} ({scientific}) · {conf * 100:.0f} % · {source}"
        try:
            self.spectrogram_widget.plot_segment(
                audio_path,
                start_sec=start,
                end_sec=end,
                title=title,
                use_noise_reduction=self.noise_reduction_check.isChecked(),
            )
        except Exception as ex:
            self.spectrogram_widget.clear_plot("No se pudo generar el espectrograma.")
            self._add_log("Warning", f"Error al dibujar el espectrograma: {ex}")

    def _populate_detections_table(self, rows: list[dict]) -> None:
        table = self.detections_table
        table.setSortingEnabled(False)
        table.setRowCount(0)
        table.setRowCount(len(rows))
        italic = QFont()
        italic.setItalic(True)
        for idx, row in enumerate(rows):
            source = str(row.get("source", "custom")).strip().lower()
            conf = float(row.get("confidence", 0.0))
            file_path = str(row.get("file", ""))
            items = [
                NumericItem(float(row["start"]), f"{float(row['start']):.2f}"),
                NumericItem(float(row["end"]), f"{float(row['end']):.2f}"),
                QTableWidgetItem(str(row.get("common", ""))),
                QTableWidgetItem(str(row.get("scientific", ""))),
                NumericItem(conf, f"{conf * 100:.0f} %"),
                QTableWidgetItem(SOURCE_LABELS.get(source, source)),
                QTableWidgetItem(Path(file_path).name if file_path else ""),
            ]
            items[3].setFont(italic)
            items[4].setData(CONF_ROLE, conf)
            fg, bg = SOURCE_STYLES.get(source, ("#334155", "#EEF2F1"))
            items[5].setForeground(QColor(fg))
            items[5].setBackground(QColor(bg))
            items[5].setToolTip(SOURCE_TIPS.get(source, ""))
            items[6].setToolTip(file_path)
            items[0].setData(Qt.UserRole, dict(row))
            for col, item in enumerate(items):
                table.setItem(idx, col, item)
        table.setSortingEnabled(True)
        table.sortByColumn(0, Qt.AscendingOrder)
        # La columna "Archivo" solo aporta cuando se analizaron varios audios.
        table.setColumnHidden(6, len({str(r.get("file", "")) for r in rows}) <= 1)
        self.play_btn.setEnabled(bool(rows))

        if rows:
            table.selectRow(0)
            self._on_detection_selection_changed()
        elif hasattr(self, "spectrogram_widget"):
            self.spectrogram_widget.clear_plot()

    def _show_detections_context_menu(self, pos) -> None:
        row = self.detections_table.rowAt(pos.y())
        if row < 0:
            return
        self.detections_table.selectRow(row)
        row_payload = self._row_payload_from_table(row)
        if not row_payload:
            return

        menu = QMenu(self)
        source = str(row_payload.get("source", "custom")).lower()
        if source == "anomaly":
            refine_action = menu.addAction("Crear o actualizar un perfil desde esta anomalía")
        else:
            refine_action = menu.addAction("Añadir este audio al perfil de la especie")
        play_action = menu.addAction("Reproducir")
        chosen = menu.exec(self.detections_table.viewport().mapToGlobal(pos))
        if chosen == refine_action:
            self._refine_profile_from_detection(row, row_payload=row_payload)
        elif chosen == play_action:
            self._play_segment(row_payload)

    def _refine_profile_from_detection(self, row: int, *, row_payload: dict | None = None) -> None:
        payload = row_payload or self._row_payload_from_table(row) or {}
        source = str(payload.get("source", "custom")).lower()
        species = str(payload.get("common") or payload.get("scientific") or "").strip()
        audio_file = str(payload.get("file", "")).strip()
        start_sec = float(payload.get("start", 0.0))
        end_sec = float(payload.get("end", 0.0))

        if source == "anomaly":
            suggested = str(payload.get("birdnet_common", "")).strip() or str(payload.get("birdnet_scientific", "")).strip()
            species_input, ok = QInputDialog.getText(
                self,
                "Asignar nombre a la anomalía",
                "Nombre de la especie o perfil para esta anomalía:",
                text=suggested,
            )
            if not ok:
                return
            species = species_input.strip()

        if not species:
            QMessageBox.warning(self, "Especie no válida", "La fila seleccionada no tiene especie.")
            return
        if not audio_file or not Path(audio_file).exists():
            QMessageBox.warning(self, "Audio no disponible", "No se encontró el archivo de audio de la detección.")
            return

        try:
            profiles = self.profile_manager.list_profiles()
            profile_names = {p.name.strip().lower() for p in profiles}
            species_exists = species.strip().lower() in profile_names

            if species_exists:
                embedding = self.audio_analyzer.extract_segment_embedding(audio_file, start_sec, end_sec)
                profile, total_vectors = self.audio_analyzer.refine_profile_with_embedding(
                    species,
                    embedding,
                    use_multi_vector=self.multi_vector_check.isChecked(),
                )
                self._add_log("Info", f"Perfil '{species}' refinado con muestra en {start_sec:.2f}-{end_sec:.2f}s.")
                QMessageBox.information(
                    self,
                    "Perfil actualizado",
                    f"Perfil de {profile.name} refinado con nueva muestra. Vectores acumulados: {total_vectors}.",
                )
            else:
                answer = QMessageBox.question(
                    self,
                    "Crear nuevo perfil",
                    f"La especie '{species}' no existe. ¿Deseas crear un nuevo perfil usando este audio?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer != QMessageBox.Yes:
                    return

                embedding = self.audio_analyzer.extract_segment_embedding(audio_file, start_sec, end_sec)
                profile = self.profile_manager.save_profile(species, audio_file, embedding)
                self._add_log("Info", f"Nueva especie creada desde una detección: {profile.name}")
                QMessageBox.information(
                    self,
                    "Perfil creado",
                    f"Perfil '{profile.name}' creado con la detección seleccionada.",
                )
        except Exception as ex:
            QMessageBox.critical(self, "Refinamiento fallido", str(ex))
            self._add_log("Error", f"No se pudo refinar perfil '{species}': {ex}")
        finally:
            self._refresh_profile_library()

    def _cleanup_temp_playback(self) -> None:
        if self._temp_playback_path is None:
            return
        try:
            if self._temp_playback_path.exists():
                self._temp_playback_path.unlink()
        except Exception:
            pass
        self._temp_playback_path = None

    def _play_segment(self, row: dict) -> None:
        """Reproduce exactamente la ventana detectada.

        El segmento se escribe en un WAV temporal y se reproduce desde el inicio: buscar
        una posicion antes de que QMediaPlayer cargue el archivo no es fiable (sonaba
        desde el comienzo de la grabacion) y los temporizadores de parada se acumulaban.
        """
        audio_file = str(row.get("file", ""))
        if not audio_file or not Path(audio_file).exists():
            QMessageBox.warning(self, "Audio no disponible", "No existe el archivo para reproducir.")
            return

        start = max(0.0, float(row.get("start", 0.0)))
        end = max(start + 0.5, float(row.get("end", start + 3.0)))
        self._update_spectrogram_for_payload(row)
        try:
            signal, sample_rate = librosa.load(audio_file, sr=None, mono=True, offset=start, duration=end - start)
        except Exception as ex:
            self._add_log("Warning", f"No se pudo leer el segmento para reproducir: {ex}")
            return
        if signal.size == 0:
            self._add_log("Warning", "El segmento seleccionado está vacío.")
            return
        if self.noise_reduction_check.isChecked() and nr is not None:
            try:
                signal = nr.reduce_noise(y=signal, sr=sample_rate, stationary=True, prop_decrease=0.9)
            except Exception as ex:
                self._add_log("Warning", f"No se pudo aplicar la reducción de ruido en la reproducción: {ex}")

        self.player.stop()
        self.player.setSource(QUrl())  # libera el temporal anterior (en Windows queda bloqueado)
        self._cleanup_temp_playback()
        fd, tmp_name = tempfile.mkstemp(prefix="ecoacoustic_play_", suffix=".wav")
        os.close(fd)
        self._temp_playback_path = Path(tmp_name)
        sf.write(tmp_name, np.clip(np.asarray(signal, dtype=np.float32), -1.0, 1.0), sample_rate, subtype="PCM_16")
        self.player.setSource(QUrl.fromLocalFile(tmp_name))
        self.player.play()

    def _populate_summary(self, rows: list[dict]) -> None:
        if not rows:
            self.summary_headline.setText("Sin detecciones para resumir.")
            self.metric_detections.setText("0")
            self.metric_species.setText("0")
            self.metric_conf.setText("—")
            self._update_species_chart([])
            self.top_species_table.setRowCount(0)
            return

        confidences = [float(r.get("confidence", 0.0)) for r in rows]
        avg_conf = mean(confidences)
        self.metric_detections.setText(str(len(rows)))
        self.metric_conf.setText(f"{avg_conf * 100:.0f} %")

        groups: dict[str, list[float]] = defaultdict(list)
        scientific_of: dict[str, str] = {}
        for r in rows:
            name = str(r.get("common") or r.get("scientific") or "N/A")
            groups[name].append(float(r.get("confidence", 0.0)))
            scientific_of.setdefault(name, str(r.get("scientific", "")))
        unique_species = len(groups)
        self.metric_species.setText(str(unique_species))
        top_items = sorted(((name, len(v), mean(v)) for name, v in groups.items()), key=lambda x: (-x[1], -x[2]))[:12]

        self.summary_headline.setText(
            f"{len(rows)} detecciones de {unique_species} especies · confianza media {avg_conf * 100:.0f} %"
        )
        self.top_species_table.setRowCount(len(top_items))
        for idx, (name, count, conf) in enumerate(top_items):
            sci = scientific_of.get(name, "")
            label = f"{name} ({sci})" if sci and sci != name else name
            self.top_species_table.setItem(idx, 0, QTableWidgetItem(label))
            self.top_species_table.setItem(idx, 1, NumericItem(count, str(count)))
            conf_item = NumericItem(conf, f"{conf * 100:.0f} %")
            conf_item.setData(CONF_ROLE, conf)
            self.top_species_table.setItem(idx, 2, conf_item)
        self._update_species_chart([(name, count) for name, count, _ in top_items])

    def _update_species_chart(self, top_items: list[tuple[str, int]]) -> None:
        self.species_chart.removeAllSeries()
        for axis in list(self.species_chart.axes()):
            self.species_chart.removeAxis(axis)
        if not top_items:
            self.species_chart.setTitle("Especies más detectadas (sin datos)")
            return

        items = list(reversed(top_items))  # la mas frecuente arriba
        bar_set = QBarSet("Detecciones")
        bar_set.setColor(QColor(PALETTE.primary))
        bar_set.setBorderColor(QColor(PALETTE.primary))
        for _, value in items:
            bar_set.append(value)
        series = QHorizontalBarSeries()
        series.append(bar_set)
        series.setLabelsVisible(True)
        series.setLabelsFormat("@value")
        self.species_chart.addSeries(series)

        axis_y = QBarCategoryAxis()
        axis_y.append([name if len(name) <= 34 else name[:33] + "…" for name, _ in items])
        self.species_chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        axis_x = QValueAxis()
        axis_x.setRange(0, max(v for _, v in items) * 1.15 + 1)
        axis_x.setLabelFormat("%d")
        axis_x.setTitleText("Detecciones")
        self.species_chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        self.species_chart.setTitle("Especies más detectadas")

    def _clear_results_view(self) -> None:
        self.summary_headline.setText("Sin resultados aún.")
        self.detections_table.setRowCount(0)
        if hasattr(self, "play_btn"):
            self.play_btn.setEnabled(False)
        if hasattr(self, "spectrogram_widget"):
            self.spectrogram_widget.clear_plot()
        self.top_species_table.setRowCount(0)
        self.metric_detections.setText("0")
        self.metric_species.setText("0")
        self.metric_conf.setText("—")
        self._update_species_chart([])
        self.session_detection_counts.clear()
        self.validation_results = []
        if hasattr(self, "validation_combo"):
            self.validation_combo.blockSignals(True)
            self.validation_combo.clear()
            self.validation_combo.blockSignals(False)
            self._show_validation_message("Analizando… la comparación con la tabla de expertos aparecerá al terminar.")
        if hasattr(self, "profile_stats_label"):
            self.profile_stats_label.setText("Selecciona una especie para ver sus detecciones en esta sesión.")

    def _choose_validation_table(self) -> None:
        current = str(self.settings.value("validation_table", "") or "")
        start_dir = str(Path(current).parent) if current else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar tabla de validaciones",
            start_dir,
            "Tablas (*.xlsx *.xls *.csv *.txt);;Todos (*.*)",
        )
        if path:
            self._load_validation_table(path, interactive=True)

    def _load_saved_validation_table(self) -> None:
        path = str(self.settings.value("validation_table", "") or "")
        if path and Path(path).exists():
            self._load_validation_table(path, interactive=False)

    def _load_validation_table(self, path: str, *, interactive: bool) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            table = load_reference_table(path)
        except Exception as ex:
            QApplication.restoreOverrideCursor()
            self._add_log("Error", f"No se pudo cargar la tabla de validaciones: {ex}")
            self.validation_source_label.setText(f"No se pudo cargar {Path(path).name}: {ex}")
            if interactive:
                QMessageBox.critical(self, "Tabla de validaciones", str(ex))
            return
        QApplication.restoreOverrideCursor()
        self.reference_table = table
        self.settings.setValue("validation_table", path)
        self.validation_source_label.setText(table.describe())
        self._add_log("Info", f"Tabla de validaciones cargada: {table.describe()}")
        self._refresh_validation()

    def _show_validation_message(self, message: str) -> None:
        for label in (self.kpi_recall, self.kpi_confirmed, self.kpi_species, self.kpi_conf):
            label.setText("—")
        for label in (self.kpi_recall_caption, self.kpi_confirmed_caption, self.kpi_species_caption, self.kpi_conf_caption):
            label.setText("")
        self.validation_species_table.setRowCount(0)
        self.validation_figure.clear()
        ax = self.validation_figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10, color=PALETTE.text_muted, transform=ax.transAxes)
        ax.set_axis_off()
        self.validation_canvas.draw_idle()

    def _refresh_validation(self) -> None:
        self.validation_results = []
        combo = self.validation_combo
        combo.blockSignals(True)
        combo.clear()
        combo.blockSignals(False)
        if self.reference_table is None:
            self._show_validation_message("Carga TablaFinal.xlsx (botón superior) para comparar con las validaciones del equipo.")
            return
        if not self.analyzed_files:
            self._show_validation_message("Analiza una grabación de CONDESAN para compararla con las validaciones del equipo.")
            return

        rows_by_file: dict[str, list[dict]] = defaultdict(list)
        for row in self.result_rows:
            rows_by_file[str(row.get("file", ""))].append(row)
        reasons: list[str] = []
        for audio in self.analyzed_files:
            reference, message = self.reference_table.lookup(audio)
            if reference is None:
                reasons.append(message)
                continue
            result = compare_recording(rows_by_file.get(audio, []), reference, recording=message)
            self.validation_results.append((result, audio))
        if not self.validation_results:
            detail = reasons[0] if reasons else ""
            self._show_validation_message(
                "Ninguna grabación analizada tiene validaciones en la tabla cargada.\n" + detail
            )
            return
        if len(self.validation_results) > 1:
            total = combine_results([res for res, _ in self.validation_results])
            self.validation_results.insert(0, (total, None))

        combo.blockSignals(True)
        for res, _ in self.validation_results:
            recall = "—" if res.recall is None else f"{res.recall * 100:.0f} %"
            combo.addItem(f"{res.recording}  ·  coincidencia {recall}")
        combo.blockSignals(False)
        combo.setCurrentIndex(0)
        self._on_validation_selection(0)

    def _on_validation_selection(self, index: int) -> None:
        if 0 <= index < len(self.validation_results):
            result, audio = self.validation_results[index]
            self._show_validation_result(result, audio)

    def _show_validation_result(self, res: ValidationResult, audio: str | None) -> None:
        def pct(value: float | None) -> str:
            return "—" if value is None else f"{value * 100:.0f} %"

        self.kpi_recall.setText(pct(res.recall))
        self.kpi_recall_caption.setText(f"{res.matched_validated} de {res.n_validated} detecciones validadas")
        self.kpi_confirmed.setText(pct(res.confirmed_fraction))
        caption = f"{res.confirmed} de {res.n_app} detecciones de la app"
        if res.matched_rejected:
            caption += f" · {res.matched_rejected} coinciden con rechazos"
        self.kpi_confirmed_caption.setText(caption)
        found, total = res.species_found
        self.kpi_species.setText(f"{found}/{total}")
        self.kpi_species_caption.setText("especies validadas que la app detecta")
        if res.app_rows:
            confs = [float(r.get("confidence", 0.0)) for r in res.app_rows]
        else:
            confs = [s.mean_confidence for s in res.species for _ in range(s.detected) if s.detected]
        self.kpi_conf.setText(pct(mean(confs)) if confs else "—")
        self.kpi_conf_caption.setText(f"{res.n_app} detecciones")

        try:
            if audio is None:
                plot_species_recall(self.validation_figure, res)
            else:
                plot_validation(self.validation_figure, res, audio_path=audio)
        except Exception as ex:
            self._add_log("Warning", f"No se pudo dibujar la validación: {ex}")
        self.validation_canvas.draw_idle()

        table = self.validation_species_table
        table.setRowCount(len(res.species))
        for idx, sp in enumerate(res.species):
            name = QTableWidgetItem(sp.species)
            if sp.common and sp.common != sp.species:
                name.setToolTip(sp.common)
            table.setItem(idx, 0, name)
            table.setItem(idx, 1, NumericItem(sp.validated, str(sp.validated)))
            table.setItem(idx, 2, NumericItem(sp.detected, str(sp.detected)))
            table.setItem(idx, 3, NumericItem(sp.matched_validated, str(sp.matched_validated)))
            recall_item = NumericItem(sp.recall if sp.recall is not None else -1, pct(sp.recall))
            if sp.recall is not None:
                recall_item.setData(CONF_ROLE, sp.recall)
            table.setItem(idx, 4, recall_item)
            conf_ok = sp.detected > 0 and not np.isnan(sp.mean_confidence)
            conf_item = NumericItem(sp.mean_confidence if conf_ok else -1, pct(sp.mean_confidence) if conf_ok else "—")
            if conf_ok:
                conf_item.setData(CONF_ROLE, sp.mean_confidence)
            table.setItem(idx, 5, conf_item)

    def _add_log(self, level: str, message: str) -> None:
        self.log_records.append((level, message))
        self._refresh_log_view()

    def _refresh_log_view(self) -> None:
        selected = self.log_filter_combo.currentText() if hasattr(self, "log_filter_combo") else "Todos"
        lines = []
        for level, message in self.log_records:
            if selected != "Todos" and level != selected:
                continue
            lines.append(f"[{level}] {message}")
        self.log_view.setPlainText("\n".join(lines))
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self.log_view.toPlainText())

    def _clear_logs(self) -> None:
        self.log_records.clear()
        self.log_view.clear()

    def _set_status(self, status: str) -> None:
        if status == "running":
            self.status_chip.setText("Analizando")
            self.status_chip.setObjectName("StatusChipRunning")
        elif status == "error":
            self.status_chip.setText("Error")
            self.status_chip.setObjectName("StatusChipError")
        else:
            self.status_chip.setText("Listo")
            self.status_chip.setObjectName("StatusChipReady")
        self._repolish(self.status_chip)

    def _set_running_state(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.open_output_btn.setEnabled(not running)
        self.input_path_edit.setEnabled(not running)
        self.output_path_edit.setEnabled(not running)
        self.steps_tabs.setEnabled(not running)

    def _repolish(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _load_settings(self) -> None:
        self.input_path_edit.setText(self.settings.value("input_path", self.input_path_edit.text()))
        self.output_path_edit.setText(self.settings.value("output_path", self.output_path_edit.text()))

        use_zone = self.settings.value("use_zone", False, type=bool)
        self.radio_zone_yes.setChecked(use_zone)
        self.radio_zone_no.setChecked(not use_zone)
        self._on_zone_mode_toggled(use_zone)
        self.country_edit.setText(self.settings.value("country", ""))
        self.province_edit.setText(self.settings.value("province", ""))
        self.canton_edit.setText(self.settings.value("canton", ""))
        self.lat_spin.setValue(float(self.settings.value("lat", 0.0)))
        self.lon_spin.setValue(float(self.settings.value("lon", 0.0)))
        self.radius_spin.setValue(float(self.settings.value("radius", 25.0)))

        self.overlap_spin.setValue(float(self.settings.value("overlap", self.DEFAULTS["overlap"])))
        self.sensitivity_spin.setValue(float(self.settings.value("sensitivity", self.DEFAULTS["sensitivity"])))
        self.min_conf_spin.setValue(float(self.settings.value("min_conf", self.DEFAULTS["min_conf"])))
        self.anomaly_threshold_spin.setValue(
            float(self.settings.value("anomaly_threshold", self.DEFAULTS["anomaly_threshold"]))
        )
        self.threads_spin.setValue(int(self.settings.value("threads", self.DEFAULTS["threads"])))
        self.multi_vector_check.setChecked(self.settings.value("use_multi_vector", self.DEFAULTS["use_multi_vector"], type=bool))
        self.noise_reduction_check.setChecked(
            self.settings.value("use_noise_reduction", self.DEFAULTS["use_noise_reduction"], type=bool)
        )
        self.smoothing_window_spin.setValue(
            int(self.settings.value("smoothing_window", self.DEFAULTS["smoothing_window"]))
        )
        self.discovery_mode_check.setChecked(
            self.settings.value("use_global_discovery", self.DEFAULTS["use_global_discovery"], type=bool)
        )
        if self.local_classifier_check.isEnabled():
            self.local_classifier_check.setChecked(
                self.settings.value("use_local_classifier", self.DEFAULTS["use_local_classifier"], type=bool)
            )

        export_formats = self.settings.value("export_formats", self.export_options.formats)
        if isinstance(export_formats, str):
            export_formats = [x for x in export_formats.split(",") if x]
        include_zone_meta = self.settings.value("include_zone_metadata", False, type=bool)
        self.export_options = ExportOptions(formats=list(export_formats), include_zone_metadata=include_zone_meta)
        if not self.export_options.formats:
            self.export_options.formats = ["csv", "table"]

    def closeEvent(self, event) -> None:
        # Detener un analisis en curso antes de destruir la ventana (si no, el hilo queda colgado).
        if self.analysis_worker is not None and self.analysis_thread is not None:
            self.analysis_worker.request_stop()
            self.analysis_thread.quit()
            self.analysis_thread.wait(15000)
        self.player.stop()
        self.player.setSource(QUrl())
        self.settings.setValue("input_path", self.input_path_edit.text().strip())
        self.settings.setValue("output_path", self.output_path_edit.text().strip())
        self.settings.setValue("use_zone", self.radio_zone_yes.isChecked())
        self.settings.setValue("country", self.country_edit.text().strip())
        self.settings.setValue("province", self.province_edit.text().strip())
        self.settings.setValue("canton", self.canton_edit.text().strip())
        self.settings.setValue("lat", self.lat_spin.value())
        self.settings.setValue("lon", self.lon_spin.value())
        self.settings.setValue("radius", self.radius_spin.value())
        self.settings.setValue("overlap", self.overlap_spin.value())
        self.settings.setValue("sensitivity", self.sensitivity_spin.value())
        self.settings.setValue("min_conf", self.min_conf_spin.value())
        self.settings.setValue("anomaly_threshold", self.anomaly_threshold_spin.value())
        self.settings.setValue("threads", self.threads_spin.value())
        self.settings.setValue("use_multi_vector", self.multi_vector_check.isChecked())
        self.settings.setValue("use_noise_reduction", self.noise_reduction_check.isChecked())
        self.settings.setValue("smoothing_window", self.smoothing_window_spin.value())
        self.settings.setValue("use_global_discovery", self.discovery_mode_check.isChecked())
        self.settings.setValue("use_local_classifier", self.local_classifier_check.isChecked())
        self.settings.setValue("export_formats", ",".join(self.export_options.formats))
        self.settings.setValue("include_zone_metadata", self.export_options.include_zone_metadata)
        self._cleanup_temp_playback()
        super().closeEvent(event)


def main() -> int:
    multiprocessing.freeze_support()  # necesario en el ejecutable empaquetado (PyInstaller)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    base_font = QFont("Segoe UI")
    base_font.setPointSize(10)
    app.setFont(base_font)
    icon_path = _resource_path("assets/app.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = AudioAnalysisWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
