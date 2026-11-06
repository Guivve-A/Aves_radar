import csv
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from statistics import mean

import librosa
import librosa.display
import numpy as np
import soundfile as sf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QValueAxis
from PySide6.QtCore import QElapsedTimer, QSettings, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from audio_analyzer import AudioAnalyzer, AudioWorker, ProfileManager
from ui_theme import build_stylesheet

try:
    import noisereduce as nr
except Exception:  # pragma: no cover - optional dependency fallback
    nr = None


APP_TITLE = "EcoAcoustic Sentinel"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aiff", ".aif"}


def _resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative_path
    return Path(__file__).resolve().parent / relative_path


@dataclass
class ExportOptions:
    formats: list[str]
    include_zone_metadata: bool = False


class ExportOptionsDialog(QDialog):
    def __init__(self, options: ExportOptions, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configurar exportacion")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hint = QLabel("Selecciona formatos para generar en cada ejecucion.")
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
            self.error_label.setText("Debes seleccionar al menos un formato de exportacion.")
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

        self.figure = Figure(figsize=(6, 2.8), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._colorbar = None
        layout.addWidget(self.canvas)
        self.clear_plot()

    def clear_plot(self, message: str = "Selecciona una deteccion para ver su espectrograma.") -> None:
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
    ) -> None:
        duration = max(0.15, float(end_sec) - float(start_sec))
        signal, sample_rate = librosa.load(
            audio_path,
            sr=None,
            mono=True,
            offset=max(0.0, float(start_sec)),
            duration=duration,
        )
        if signal.size == 0:
            self.clear_plot("No se pudo extraer audio para espectrograma.")
            return

        if use_noise_reduction and nr is not None:
            signal = nr.reduce_noise(y=signal, sr=sample_rate, stationary=True, prop_decrease=0.9)

        signal = signal.astype("float32", copy=False)
        n_fft = 2048
        hop_length = 256
        stft_matrix = librosa.stft(
            y=signal,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
        )
        magnitude = np.abs(stft_matrix)
        ref_amp = float(magnitude.max()) if magnitude.size else 1.0
        if ref_amp <= 0.0:
            ref_amp = 1e-6
        spec_db = librosa.amplitude_to_db(magnitude, ref=ref_amp)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = librosa.display.specshow(
            spec_db,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="hz",
            cmap="inferno",
            ax=ax,
        )
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")
        max_plot_hz = min(12000, sample_rate // 2)
        ax.set_ylim([0, max_plot_hz])
        ax.set_title(title or "Espectrograma (STFT)")
        self._colorbar = self.figure.colorbar(img, ax=ax, pad=0.01, format="%+2.0f dB")
        self._colorbar.set_label("Intensidad (dB)")
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
        "Precision alta": {"overlap": 0.20, "sensitivity": 0.90, "min_conf": 0.82, "threads": max(1, (os.cpu_count() or 4) // 2)},
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

    def _init_ui(self) -> None:
        self.setWindowTitle(APP_TITLE)
        icon_path = _resource_path("assets/app.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setMinimumSize(1300, 820)
        self.setStyleSheet(build_stylesheet())

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
        layout.setContentsMargins(14, 12, 14, 12)

        text_col = QVBoxLayout()
        title = QLabel("EcoAcoustic Sentinel")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("Flujo guiado: Audio + Zona -> Analizar -> Ver resultados")
        subtitle.setObjectName("SubtitleLabel")
        text_col.addWidget(title)
        text_col.addWidget(subtitle)

        layout.addLayout(text_col, 1)

        self.status_chip = QLabel("Listo")
        self.status_chip.setObjectName("StatusChipReady")
        layout.addWidget(self.status_chip, 0, Qt.AlignTop)
        return card

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        flow_hint = QLabel("Configuracion por pasos")
        flow_hint.setObjectName("HintLabel")
        layout.addWidget(flow_hint)

        self.steps_tabs = QTabWidget()
        self.steps_tabs.addTab(self._build_step_audio(), "1. Audio")
        self.steps_tabs.addTab(self._build_step_zone(), "2. Zona")
        self.steps_tabs.addTab(self._build_step_params(), "3. Parametros")
        self.steps_tabs.addTab(self._build_step_export(), "4. Exportacion")
        layout.addWidget(self.steps_tabs, 1)

        actions = QHBoxLayout()
        self.run_btn = QPushButton("Iniciar analisis")
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
        self.input_path_edit.textChanged.connect(self._validate_audio_inputs)

        pick_file_btn = QPushButton("Archivo...")
        pick_file_btn.clicked.connect(self._browse_input_file)
        pick_folder_btn = QPushButton("Carpeta...")
        pick_folder_btn.clicked.connect(self._browse_input_folder)

        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(str(Path.cwd() / "output"))
        self.output_path_edit.textChanged.connect(self._validate_audio_inputs)
        pick_out_btn = QPushButton("Salida...")
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

        tip = QLabel("Validacion inmediata: ruta existente, extension soportada y conteo de archivos.")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        grid.addWidget(tip, 2, 0, 1, 4)
        grid.addWidget(self.input_validation_label, 3, 0, 1, 4)
        return tab

    def _build_step_zone(self) -> QWidget:
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)

        zone_mode_box = QGroupBox("¿Conoces la ubicacion de la grabacion?")
        zone_mode_layout = QVBoxLayout(zone_mode_box)
        zone_mode_layout.setContentsMargins(10, 8, 10, 8)
        zone_mode_layout.setSpacing(6)

        self.radio_zone_no = QRadioButton("🌍 No, no se donde fue grabada (Modo Global)")
        self.radio_zone_no.setChecked(True)
        self.radio_zone_no.setToolTip("El sistema no filtrara especies por ubicacion.")
        self.radio_zone_yes = QRadioButton("📍 Si, indicar zona")
        self.radio_zone_yes.setToolTip("Filtra especies probables segun coordenadas.")
        self.radio_zone_yes.toggled.connect(self._on_zone_mode_toggled)
        self.radio_zone_no.toggled.connect(self._validate_zone_inputs)

        zone_mode_layout.addWidget(self.radio_zone_no)
        zone_mode_layout.addWidget(self.radio_zone_yes)

        self.country_edit = QLineEdit()
        self.country_edit.setPlaceholderText("Pais")
        self.country_edit.textChanged.connect(self._validate_zone_inputs)

        self.province_edit = QLineEdit()
        self.province_edit.setPlaceholderText("Provincia/Estado")
        self.province_edit.textChanged.connect(self._validate_zone_inputs)

        self.canton_edit = QLineEdit()
        self.canton_edit.setPlaceholderText("Canton/Ciudad")
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
        zone_grid.addWidget(QLabel("Pais"), 0, 0)
        zone_grid.addWidget(self.country_edit, 0, 1)
        zone_grid.addWidget(QLabel("Provincia"), 0, 2)
        zone_grid.addWidget(self.province_edit, 0, 3)
        zone_grid.addWidget(QLabel("Canton/Ciudad"), 1, 0)
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

        tip = QLabel("BirdNET usa lat/lon para filtro local; pais/provincia/canton se guardan como contexto en UI.")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        grid.addWidget(tip, 3, 0, 1, 4)
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
        self.sensitivity_spin.setToolTip("Mayor valor aumenta sensibilidad de deteccion.")

        self.min_conf_spin = QDoubleSpinBox()
        self.min_conf_spin.setDecimals(2)
        self.min_conf_spin.setRange(0.01, 0.99)
        self.min_conf_spin.setSingleStep(0.01)
        self.min_conf_spin.setValue(0.75)
        self.min_conf_spin.setToolTip("Umbral de similitud coseno para confirmar especie personalizada.")

        self.anomaly_threshold_spin = QDoubleSpinBox()
        self.anomaly_threshold_spin.setDecimals(2)
        self.anomaly_threshold_spin.setRange(0.05, 0.99)
        self.anomaly_threshold_spin.setSingleStep(0.01)
        self.anomaly_threshold_spin.setValue(0.45)
        self.anomaly_threshold_spin.setToolTip(
            "Umbral de confianza BirdNET para marcar eventos como anomalia acustica."
        )

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, max(2, os.cpu_count() or 8))
        self.threads_spin.setToolTip("Hilos de CPU para analisis.")

        self.multi_vector_check = QCheckBox("Modo Alta Precision (Multi-Vector)")
        self.multi_vector_check.setChecked(True)
        self.multi_vector_check.setToolTip("Compara cada chunk contra multiples prototipos por especie.")

        self.noise_reduction_check = QCheckBox("Activar Reduccion de Ruido (DSP)")
        self.noise_reduction_check.setChecked(False)
        self.noise_reduction_check.setToolTip(
            "Aplica noisereduce antes de extraer embeddings y en espectrograma/reproduccion."
        )

        self.smoothing_window_spin = QSpinBox()
        self.smoothing_window_spin.setRange(1, 11)
        self.smoothing_window_spin.setSingleStep(1)
        self.smoothing_window_spin.setValue(3)
        self.smoothing_window_spin.setSuffix(" frames")
        self.smoothing_window_spin.setToolTip("Ventana para suavizado temporal de detecciones.")

        self.discovery_mode_check = QCheckBox("Activar Modo Descubrimiento (Base de datos Global)")
        self.discovery_mode_check.setChecked(False)
        self.discovery_mode_check.setToolTip(
            "Reportara especies conocidas por BirdNET aunque no esten en tu biblioteca local."
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

        restore_btn = QPushButton("Restaurar defaults")
        restore_btn.clicked.connect(self._restore_defaults)

        grid.addWidget(QLabel("Preset"), 0, 0)
        grid.addWidget(self.preset_combo, 0, 1)
        grid.addWidget(restore_btn, 0, 2, 1, 2)

        grid.addWidget(QLabel("Overlap"), 1, 0)
        grid.addWidget(self.overlap_spin, 1, 1)
        grid.addWidget(QLabel("Sensibilidad"), 1, 2)
        grid.addWidget(self.sensitivity_spin, 1, 3)

        grid.addWidget(QLabel("Umbral similitud"), 2, 0)
        grid.addWidget(self.min_conf_spin, 2, 1)
        grid.addWidget(QLabel("Threads"), 2, 2)
        grid.addWidget(self.threads_spin, 2, 3)

        grid.addWidget(self.multi_vector_check, 3, 0, 1, 2)
        grid.addWidget(QLabel("Ventana de Suavizado"), 3, 2)
        grid.addWidget(self.smoothing_window_spin, 3, 3)
        grid.addWidget(self.noise_reduction_check, 4, 0, 1, 2)
        grid.addWidget(QLabel("Umbral anomalia"), 4, 2)
        grid.addWidget(self.anomaly_threshold_spin, 4, 3)
        grid.addWidget(self.discovery_mode_check, 5, 0, 1, 4)
        grid.addWidget(self.local_classifier_check, 6, 0, 1, 4)

        tip = QLabel("Tip: para fauna densa usa 'Alta sensibilidad'; para precision usa 'Precision alta'.")
        tip.setObjectName("HintLabel")
        tip.setWordWrap(True)
        grid.addWidget(tip, 7, 0, 1, 4)
        return tab

    def _build_step_export(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.export_summary = QLabel("")
        self.export_summary.setWordWrap(True)
        self.export_summary.setObjectName("HintLabel")

        export_btn = QPushButton("Exportar...")
        export_btn.clicked.connect(self._open_export_dialog)

        meta_hint = QLabel("Usa este panel para definir formatos de salida y metadatos de zona.")
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

        runtime = QGroupBox("Ejecucion")
        runtime_grid = QGridLayout(runtime)
        runtime_grid.setHorizontalSpacing(10)
        runtime_grid.setVerticalSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Listo")
        self.progress_label = QLabel("Archivos: 0/0")
        self.elapsed_label = QLabel("Tiempo: 00:00:00")

        runtime_grid.addWidget(self.progress_bar, 0, 0, 1, 4)
        runtime_grid.addWidget(self.progress_label, 1, 0, 1, 2)
        runtime_grid.addWidget(self.elapsed_label, 1, 2, 1, 2)

        self.metric_files = QLabel("0/0")
        self.metric_detections = QLabel("0")
        self.metric_species = QLabel("0")
        self.metric_conf = QLabel("0.00")

        runtime_grid.addWidget(self._metric_card("Archivos", self.metric_files), 2, 0)
        runtime_grid.addWidget(self._metric_card("Detecciones", self.metric_detections), 2, 1)
        runtime_grid.addWidget(self._metric_card("Especies", self.metric_species), 2, 2)
        runtime_grid.addWidget(self._metric_card("Conf. media", self.metric_conf), 2, 3)

        layout.addWidget(runtime)

        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self._build_summary_tab(), "Resumen")
        self.result_tabs.addTab(self._build_detections_tab(), "Detecciones")
        self.result_tabs.addTab(self._build_species_library_tab(), "Biblioteca de Especies")
        self.result_tabs.addTab(self._build_logs_tab(), "Logs")
        layout.addWidget(self.result_tabs, 1)

        return panel

    def _metric_card(self, label: str, value_label: QLabel) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        box = QVBoxLayout(card)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(3)

        title = QLabel(label)
        title.setObjectName("HintLabel")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("font-size: 20px; font-weight: 700;")

        box.addWidget(title, 0, Qt.AlignCenter)
        box.addWidget(value_label)
        return card

    def _build_summary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.summary_headline = QLabel("Sin resultados aun.")
        self.summary_headline.setObjectName("HintLabel")
        self.summary_headline.setWordWrap(True)
        layout.addWidget(self.summary_headline)

        self.species_chart = QChart()
        self.species_chart.setTitle("Top especies detectadas")
        self.species_chart.legend().setVisible(False)
        self.species_chart_view = QChartView(self.species_chart)
        self.species_chart_view.setMinimumHeight(260)
        layout.addWidget(self.species_chart_view)

        self.top_species_table = QTableWidget(0, 2)
        self.top_species_table.setHorizontalHeaderLabels(["Especie", "Conteo"])
        self.top_species_table.verticalHeader().setVisible(False)
        self.top_species_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.top_species_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.top_species_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.top_species_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.top_species_table)
        return tab

    def _build_detections_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.detections_table = QTableWidget(0, 7)
        self.detections_table.setHorizontalHeaderLabels(
            ["Inicio (s)", "Fin (s)", "Especie comun", "Especie cientifica", "Confianza", "Archivo", "Accion"]
        )
        self.detections_table.verticalHeader().setVisible(False)
        self.detections_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.detections_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.detections_table.setAlternatingRowColors(True)
        self.detections_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.detections_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.detections_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.detections_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.detections_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.detections_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.detections_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.detections_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.detections_table.customContextMenuRequested.connect(self._show_detections_context_menu)
        self.detections_table.itemSelectionChanged.connect(self._on_detection_selection_changed)

        self.spectrogram_widget = SpectrogramWidget()
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.detections_table)
        splitter.addWidget(self.spectrogram_widget)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([300, 220])

        layout.addWidget(splitter)
        return tab

    def _build_species_library_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        add_btn = QPushButton("Agregar Nueva Especie")
        add_btn.clicked.connect(self._add_reference_species)
        refresh_btn = QPushButton("Recargar")
        refresh_btn.clicked.connect(self._refresh_profile_library)
        remove_btn = QPushButton("Eliminar Seleccionada")
        remove_btn.setObjectName("DangerAction")
        remove_btn.clicked.connect(self._remove_selected_profile)

        controls.addWidget(add_btn)
        controls.addWidget(refresh_btn)
        controls.addWidget(remove_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.profile_table = QTableWidget(0, 3)
        self.profile_table.setHorizontalHeaderLabels(["Especie", "Referencia", "Dim"])
        self.profile_table.verticalHeader().setVisible(False)
        self.profile_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.profile_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.profile_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.profile_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.profile_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.profile_table.itemSelectionChanged.connect(self._on_profile_selected)
        layout.addWidget(self.profile_table)

        self.profile_stats_label = QLabel("Selecciona una especie para ver sus detecciones en esta sesion.")
        self.profile_stats_label.setObjectName("HintLabel")
        self.profile_stats_label.setWordWrap(True)
        layout.addWidget(self.profile_stats_label)
        return tab

    def _build_logs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.logs_group = QGroupBox("Logs del proceso")
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
        self.log_view.setFont(QFont("Consolas", 10))
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
            QMessageBox.warning(self, "Salida vacia", "Define primero la carpeta de salida.")
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
            self.input_validation_label.setText("Extension no soportada para analisis.")
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
        self.input_validation_label.setText(f"Entrada valida. Archivos detectados: {total}")
        self._repolish(self.input_validation_label)
        return True

    def _on_zone_mode_toggled(self, enabled: bool) -> None:
        if hasattr(self, "zone_inputs_container"):
            self.zone_inputs_container.setEnabled(bool(enabled))
        self._validate_zone_inputs()

    def _validate_zone_inputs(self) -> bool:
        if self.radio_zone_no.isChecked():
            self.zone_validation_label.setObjectName("OkLabel")
            self.zone_validation_label.setText("Modo Global activo. No se aplicara filtro geoespacial.")
            self._repolish(self.zone_validation_label)
            return True

        lat = self.lat_spin.value()
        lon = self.lon_spin.value()

        if lat == 0.0 and lon == 0.0:
            self.zone_validation_label.setObjectName("ErrorLabel")
            self.zone_validation_label.setText("Activaste filtro de zona: define latitud/longitud valida (no 0,0).")
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
                f"Perfiles cargados: {len(profiles)}. Selecciona una especie para ver detecciones de esta sesion."
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
            QMessageBox.information(self, "Seleccion requerida", "Selecciona una especie para eliminar.")
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
            f"Se eliminara el perfil '{profile_name}'.\n¿Deseas continuar?",
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
        self.profile_stats_label.setText(f"Detecciones en sesion actual para '{name}': {count}")

    def _command_args(self) -> list[str]:
        args = [
            "-m",
            "birdnet_analyzer.analyze",
            self.input_path_edit.text().strip(),
            "-o",
            self.output_path_edit.text().strip(),
            "--overlap",
            f"{self.overlap_spin.value():.2f}",
            "--sensitivity",
            f"{self.sensitivity_spin.value():.2f}",
            "--min_conf",
            f"{self.min_conf_spin.value():.2f}",
            "-t",
            str(self.threads_spin.value()),
            "--rtype",
        ]
        args.extend(self.export_options.formats)

        if self.radio_zone_yes.isChecked():
            args.extend(
                [
                    "--lat",
                    f"{self.lat_spin.value():.6f}",
                    "--lon",
                    f"{self.lon_spin.value():.6f}",
                ]
            )

        if self.export_options.include_zone_metadata and "csv" in self.export_options.formats and self.radio_zone_yes.isChecked():
            args.extend(["--additional_columns", "lat", "lon"])
        return args

    def _estimate_total_files(self, input_path: str) -> int:
        p = Path(input_path)
        if not p.exists():
            return 0
        if p.is_file():
            return 1 if p.suffix.lower() in AUDIO_EXTENSIONS else 0
        count = 0
        for ext in AUDIO_EXTENSIONS:
            count += len(list(p.rglob(f"*{ext}")))
        return count

    def _start_analysis(self) -> None:
        if not self._validate_audio_inputs():
            self.steps_tabs.setCurrentIndex(0)
            return
        if not self._validate_zone_inputs():
            self.steps_tabs.setCurrentIndex(1)
            return
        if not self.export_options.formats:
            self.steps_tabs.setCurrentIndex(3)
            QMessageBox.warning(self, "Exportacion", "Selecciona al menos un formato de salida.")
            return
        profiles = self.profile_manager.list_profiles()
        use_local = self.local_classifier_check.isChecked() and self.local_classifier_check.isEnabled()
        if not profiles and not self.discovery_mode_check.isChecked() and not use_local:
            self.result_tabs.setCurrentIndex(2)
            QMessageBox.warning(
                self,
                "Perfiles requeridos",
                "Debes agregar al menos una especie en 'Biblioteca de Especies', activar Modo Descubrimiento "
                "o activar el clasificador local CONDESAN.",
            )
            return

        output_dir = Path(self.output_path_edit.text().strip())
        output_dir.mkdir(parents=True, exist_ok=True)

        self.result_rows.clear()
        self.session_detection_counts.clear()
        self._clear_results_view()
        self._clear_logs()
        self._set_running_state(True)
        self._set_status("running")
        if use_local:
            self._add_log("Info", "Clasificador local CONDESAN activado (especies locales con umbrales calibrados).")
        if not profiles and self.discovery_mode_check.isChecked():
            self._add_log("Warning", "Sin perfiles locales: se usara solo BirdNET global.")
        elif self.discovery_mode_check.isChecked():
            self._add_log("Info", "Iniciando analisis hibrido (custom + BirdNET global)...")
        else:
            self._add_log("Info", "Iniciando analisis por similitud de embeddings...")

        self.audio_analyzer.overlap_seconds = self.overlap_spin.value()
        self.audio_analyzer.threads = self.threads_spin.value()
        self.audio_analyzer.similarity_threshold = self.min_conf_spin.value()
        self.audio_analyzer.anomaly_threshold = self.anomaly_threshold_spin.value()
        self.audio_analyzer.use_noise_reduction = self.noise_reduction_check.isChecked()

        self.total_files = 1
        self.processed_files = 0
        self.metric_files.setText("0/0")
        self.progress_label.setText("Chunks: 0/0")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Analizando...")

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
            self._add_log("Warning", "Solicitud de detencion enviada.")
            self.analysis_worker.request_stop()

    def _on_worker_progress(self, done: int, total: int, current_file: str, detections: int, avg_conf: float) -> None:
        self.processed_files = done
        self.total_files = total
        self.metric_files.setText(f"{done}/{total}")
        self.progress_label.setText(f"Chunks: {done}/{total} | {Path(current_file).name}")
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(min(done, total))
        self.metric_detections.setText(str(detections))
        self.metric_conf.setText(f"{avg_conf:.2f}")

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
            f"Chunks: {summary.get('processed_segments', 0)}/{summary.get('total_segments', 0)}"
        )
        self.progress_bar.setRange(0, max(1, summary.get("total_segments", 1)))
        self.progress_bar.setValue(summary.get("processed_segments", 0))

        if summary.get("stopped"):
            self._add_log("Warning", "Analisis detenido por usuario.")
        else:
            self._add_log("Info", f"Analisis completado. Detecciones: {summary.get('detections', 0)}.")

        for path in exported_files:
            self._add_log("Info", f"Exportado: {path}")

        self.analysis_worker = None
        self.analysis_thread = None

    def _on_worker_failed(self, message: str) -> None:
        self.elapsed_tick.stop()
        self._set_running_state(False)
        self._set_status("error")
        self._add_log("Error", f"Fallo en analisis por embeddings: {message}")
        self.analysis_worker = None
        self.analysis_thread = None

    def _update_elapsed(self) -> None:
        if not self.elapsed_timer.isValid():
            return
        ms = self.elapsed_timer.elapsed()
        secs = ms // 1000
        hh = secs // 3600
        mm = (secs % 3600) // 60
        ss = secs % 60
        self.elapsed_label.setText(f"Tiempo: {hh:02}:{mm:02}:{ss:02}")

    def _load_results_from_output(self) -> None:
        output = Path(self.output_path_edit.text().strip())
        if not output.exists():
            return

        csv_files = sorted(output.rglob("*.BirdNET.results.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not csv_files:
            self.summary_headline.setText("Analisis finalizado sin CSV disponible para tabla/graficos.")
            return

        rows: list[dict] = []
        for csv_path in csv_files:
            rows.extend(self._read_csv_rows(csv_path))
        self.result_rows = rows

        self._populate_detections_table(rows)
        self._populate_summary(rows)

    def _read_csv_rows(self, csv_path: Path) -> list[dict]:
        parsed: list[dict] = []
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    start = self._safe_float(self._get_field(row, "start (s)", "start"))
                    end = self._safe_float(self._get_field(row, "end (s)", "end"))
                    scientific = self._get_field(row, "scientific name", "scientific")
                    common = self._get_field(row, "common name", "common")
                    confidence = self._safe_float(self._get_field(row, "confidence"))
                    file_path = self._get_field(row, "file", "audio file")
                    source = self._get_field(row, "source") or "custom"
                    if not file_path:
                        file_path = ""
                    parsed.append(
                        {
                            "start": start,
                            "end": end,
                            "scientific": scientific,
                            "common": common,
                            "confidence": confidence,
                            "file": file_path,
                            "source": source,
                        }
                    )
        except Exception as ex:
            self._add_log("Error", f"No se pudo leer CSV {csv_path}: {ex}")
        return parsed

    def _get_field(self, row: dict, *candidates: str) -> str:
        normalized = {k.strip().lower(): v for k, v in row.items()}
        for c in candidates:
            if c in normalized:
                return normalized[c]
        return ""

    def _safe_float(self, value: str) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _row_payload_from_table(self, row_index: int) -> dict | None:
        if row_index < 0:
            return None
        first_item = self.detections_table.item(row_index, 0)
        if first_item is not None:
            payload = first_item.data(Qt.UserRole)
            if isinstance(payload, dict):
                return dict(payload)

        common_item = self.detections_table.item(row_index, 2)
        scientific_item = self.detections_table.item(row_index, 3)
        file_item = self.detections_table.item(row_index, 5)
        return {
            "start": self._safe_float(self.detections_table.item(row_index, 0).text() if self.detections_table.item(row_index, 0) else "0"),
            "end": self._safe_float(self.detections_table.item(row_index, 1).text() if self.detections_table.item(row_index, 1) else "0"),
            "common": common_item.text() if common_item else "N/A",
            "scientific": scientific_item.text() if scientific_item else "N/A",
            "confidence": self._safe_float(self.detections_table.item(row_index, 4).text() if self.detections_table.item(row_index, 4) else "0"),
            "file": file_item.text() if file_item else "",
            "source": "custom",
        }

    def _on_detection_selection_changed(self) -> None:
        if not hasattr(self, "spectrogram_widget"):
            return
        row = self.detections_table.currentRow()
        payload = self._row_payload_from_table(row)
        if not payload:
            self.spectrogram_widget.clear_plot()
            return
        self._update_spectrogram_for_payload(payload)

    def _update_spectrogram_for_payload(self, row_payload: dict) -> None:
        audio_path = str(row_payload.get("file", "")).strip()
        if not audio_path or not Path(audio_path).exists():
            self.spectrogram_widget.clear_plot("Audio no disponible para espectrograma.")
            return

        start = max(0.0, float(row_payload.get("start", 0.0)))
        end = max(start + 0.15, float(row_payload.get("end", start + 0.25)))
        common = str(row_payload.get("common", "N/A"))
        scientific = str(row_payload.get("scientific", "N/A"))
        source = str(row_payload.get("source", "custom")).lower()
        title = f"{common} | {scientific} | {source}"
        try:
            self.spectrogram_widget.plot_segment(
                audio_path,
                start_sec=start,
                end_sec=end,
                title=title,
                use_noise_reduction=self.noise_reduction_check.isChecked(),
            )
        except Exception as ex:
            self.spectrogram_widget.clear_plot("No se pudo generar espectrograma.")
            self._add_log("Warning", f"Error al renderizar espectrograma: {ex}")

    def _populate_detections_table(self, rows: list[dict]) -> None:
        self.detections_table.setRowCount(0)
        for idx, row in enumerate(rows):
            self.detections_table.insertRow(idx)
            source = str(row.get("source", "custom")).strip().lower()
            is_global = source == "birdnet_global"
            is_anomaly = source == "anomaly"
            is_local = source == "local"
            common_label = row["common"]
            if is_global:
                common_label = f"{common_label} (Global)"

            row_items = [
                QTableWidgetItem(f"{row['start']:.2f}"),
                QTableWidgetItem(f"{row['end']:.2f}"),
                QTableWidgetItem(common_label),
                QTableWidgetItem(row["scientific"]),
                QTableWidgetItem(f"{row['confidence']:.3f}"),
                QTableWidgetItem(row["file"]),
            ]
            for col, item in enumerate(row_items):
                if is_global:
                    item.setForeground(QColor("#1D4ED8"))
                    item.setBackground(QColor("#EEF4FF"))
                    item.setToolTip("Deteccion obtenida por BirdNET global.")
                elif is_anomaly:
                    item.setForeground(QColor("#92400E"))
                    item.setBackground(QColor("#FFF7E6"))
                    item.setToolTip("Evento biologico no reconocido por perfiles locales (anomalia).")
                elif is_local:
                    item.setForeground(QColor("#0F5257"))
                    item.setBackground(QColor("#E6F6F7"))
                    item.setToolTip("Deteccion del clasificador local CONDESAN (BirdNET con umbral calibrado por especie).")
                else:
                    item.setForeground(QColor("#14532D"))
                    item.setBackground(QColor("#ECFDF3"))
                    item.setToolTip("Especie conocida por perfiles locales.")
                if col == 0:
                    item.setData(Qt.UserRole, dict(row))
                self.detections_table.setItem(idx, col, item)

            play_btn = QPushButton("Reproducir")
            exists = Path(row["file"]).exists() if row["file"] else False
            play_btn.setEnabled(exists)
            play_btn.clicked.connect(partial(self._play_segment, dict(row)))
            self.detections_table.setCellWidget(idx, 6, play_btn)

        if rows:
            self.detections_table.selectRow(0)
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
            refine_action = menu.addAction("Crear/actualizar perfil desde anomalia")
        else:
            refine_action = menu.addAction("Anadir este audio al perfil de la especie")
        chosen = menu.exec(self.detections_table.viewport().mapToGlobal(pos))
        if chosen == refine_action:
            self._refine_profile_from_detection(row, row_payload=row_payload)

    def _refine_profile_from_detection(self, row: int, *, row_payload: dict | None = None) -> None:
        species_item = self.detections_table.item(row, 2)
        scientific_item = self.detections_table.item(row, 3)
        file_item = self.detections_table.item(row, 5)
        start_item = self.detections_table.item(row, 0)
        end_item = self.detections_table.item(row, 1)

        payload = row_payload or self._row_payload_from_table(row) or {}
        source = str(payload.get("source", "custom")).lower()
        species = species_item.text().strip() if species_item else ""
        if not species and scientific_item:
            species = scientific_item.text().strip()
        species = species.replace(" (Global)", "").strip()
        audio_file = str(payload.get("file", "")).strip() or (file_item.text().strip() if file_item else "")
        start_sec = float(payload.get("start", 0.0)) if payload else (self._safe_float(start_item.text().strip()) if start_item else 0.0)
        end_sec = float(payload.get("end", 0.0)) if payload else (self._safe_float(end_item.text().strip()) if end_item else 0.0)

        if source == "anomaly":
            suggested = str(payload.get("birdnet_common", "")).strip() or str(payload.get("birdnet_scientific", "")).strip()
            species_input, ok = QInputDialog.getText(
                self,
                "Asignar nombre para anomalia",
                "Nombre de especie/perfil para esta anomalia:",
                text=suggested,
            )
            if not ok:
                return
            species = species_input.strip()

        if not species:
            QMessageBox.warning(self, "Especie no valida", "La fila seleccionada no tiene especie.")
            return
        if not audio_file or not Path(audio_file).exists():
            QMessageBox.warning(self, "Audio no disponible", "No se encontro el archivo de audio de la deteccion.")
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
                self._add_log("Info", f"Nueva especie creada desde deteccion global: {profile.name}")
                QMessageBox.information(
                    self,
                    "Perfil creado",
                    f"Perfil '{profile.name}' creado correctamente con la deteccion seleccionada.",
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
        audio_file = row.get("file", "")
        if not audio_file or not Path(audio_file).exists():
            QMessageBox.warning(self, "Audio no disponible", "No existe el archivo para reproduccion.")
            return

        start = max(0.0, float(row.get("start", 0.0)))
        end = max(start + 0.5, float(row.get("end", start + 1.0)))
        play_source = str(audio_file)
        play_start = start
        play_duration = max(0.2, end - start)

        self._update_spectrogram_for_payload(row)

        if self.noise_reduction_check.isChecked() and nr is not None:
            try:
                signal, sample_rate = librosa.load(
                    str(audio_file),
                    sr=None,
                    mono=True,
                    offset=start,
                    duration=play_duration,
                )
                if signal.size > 0:
                    cleaned = nr.reduce_noise(y=signal, sr=sample_rate, stationary=True, prop_decrease=0.9)
                    self._cleanup_temp_playback()
                    fd, tmp_name = tempfile.mkstemp(prefix="ecoacoustic_play_", suffix=".wav")
                    os.close(fd)
                    self._temp_playback_path = Path(tmp_name)
                    sf.write(str(self._temp_playback_path), cleaned.astype("float32"), sample_rate)
                    play_source = str(self._temp_playback_path)
                    play_start = 0.0
            except Exception as ex:
                self._add_log("Warning", f"No se pudo aplicar reduccion de ruido en reproduccion: {ex}")
        else:
            self._cleanup_temp_playback()

        duration_ms = int(play_duration * 1000)

        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(play_source))
        self.player.setPosition(int(play_start * 1000))
        self.player.play()
        QTimer.singleShot(duration_ms, self.player.stop)

    def _populate_summary(self, rows: list[dict]) -> None:
        if not rows:
            self.summary_headline.setText("Sin detecciones para resumir.")
            self.metric_detections.setText("0")
            self.metric_species.setText("0")
            self.metric_conf.setText("0.00")
            self._update_species_chart([])
            self.top_species_table.setRowCount(0)
            return

        self.metric_detections.setText(str(len(rows)))
        avg_conf = mean([r["confidence"] for r in rows]) if rows else 0.0
        self.metric_conf.setText(f"{avg_conf:.2f}")

        species_keys = []
        for r in rows:
            common = r["common"] or "N/A"
            scientific = r["scientific"] or "N/A"
            species_keys.append(f"{common} ({scientific})")

        counts = Counter(species_keys)
        unique_species = len(counts)
        self.metric_species.setText(str(unique_species))

        top_items = counts.most_common(10)
        self.summary_headline.setText(
            f"Detecciones: {len(rows)} | Especies unicas: {unique_species} | Confianza media: {avg_conf:.2f}"
        )

        self.top_species_table.setRowCount(0)
        for idx, (name, count) in enumerate(top_items):
            self.top_species_table.insertRow(idx)
            self.top_species_table.setItem(idx, 0, QTableWidgetItem(name))
            self.top_species_table.setItem(idx, 1, QTableWidgetItem(str(count)))

        self._update_species_chart(top_items)

    def _update_species_chart(self, top_items: list[tuple[str, int]]) -> None:
        self.species_chart.removeAllSeries()
        for axis in list(self.species_chart.axes()):
            self.species_chart.removeAxis(axis)

        if not top_items:
            self.species_chart.setTitle("Top especies detectadas (sin datos)")
            return

        categories = [item[0] for item in top_items]
        values = [item[1] for item in top_items]

        bar_set = QBarSet("Detecciones")
        for v in values:
            bar_set.append(v)

        series = QBarSeries()
        series.append(bar_set)
        self.species_chart.addSeries(series)

        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setLabelsAngle(-35)
        self.species_chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, max(values) + 1)
        axis_y.setLabelFormat("%d")
        self.species_chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        self.species_chart.setTitle("Top especies detectadas")

    def _clear_results_view(self) -> None:
        self.summary_headline.setText("Sin resultados aun.")
        self.detections_table.setRowCount(0)
        if hasattr(self, "spectrogram_widget"):
            self.spectrogram_widget.clear_plot()
        self.top_species_table.setRowCount(0)
        self.metric_detections.setText("0")
        self.metric_species.setText("0")
        self.metric_conf.setText("0.00")
        self._update_species_chart([])
        self.session_detection_counts.clear()
        if hasattr(self, "profile_stats_label"):
            self.profile_stats_label.setText("Selecciona una especie para ver sus detecciones en esta sesion.")

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
    app = QApplication(sys.argv)
    icon_path = _resource_path("assets/app.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = AudioAnalysisWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
