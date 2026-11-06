"""Validación de las detecciones de la app frente a validaciones expertas.

Se carga una tabla de referencia (``TablaFinal.xlsx`` del proyecto CONDESAN o
un CSV/XLSX con columnas equivalentes) y, para cada grabación analizada, se
calcula:

* **Coincidencia con expertos** (exhaustividad): fracción de las detecciones
  validadas por el equipo (``Valid = 1``) que la app también detecta.
* **Detecciones confirmadas**: fracción de las detecciones de la app que
  coinciden con una detección validada. Una detección *sin validar* no es
  necesariamente un error: el equipo solo revisó las propuestas de BirdNET.
* **Coincide con un rechazo**: la app detecta algo que el equipo marcó como
  incorrecto (``Valid = 0``).

Dos detecciones coinciden si son de la misma especie y se solapan en el tiempo
al menos ``min_overlap`` segundos.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FNAME_RE = re.compile(r"(?:^|_)(\d{8})_(\d{6})\.wav$", re.IGNORECASE)
DEVICE_RE = re.compile(r"\bAM\d{3}\b", re.IGNORECASE)

# Nombres de columna aceptados para cada campo (TablaFinal de CONDESAN y equivalentes).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "species": ("Nombre científico", "Nombre cientifico", "Scientific name", "Especie", "species"),
    "common_es": ("Nombre español", "Nombre espanol", "Nombre común", "Common name", "common"),
    "device": ("Dispositivo", "Device", "Grabadora", "device"),
    "day": ("Día", "Dia", "Fecha", "Date", "day"),
    "hour": ("Hora", "Time", "hour"),
    "start": ("Inicio (s)", "Begin Time (s)", "Start (s)", "start"),
    "end": ("Final (s)", "End Time (s)", "End (s)", "end"),
    "valid": ("Valid", "Válido", "Valido", "valid"),
    "confidence": ("Confidence Score", "Confidence", "Confianza", "confidence"),
}
REQUIRED = ("species", "device", "day", "hour", "start", "end", "valid")

# Nombres que BirdNET usa de forma distinta a la tabla del proyecto.
SPECIES_SYNONYMS = {
    "burhinus superciliaris": "hesperoburhinus superciliaris",
    "tyto alba": "tyto furcata",
    "xenops minutus": "xenops mexicanus",
}

STATUS_CONFIRMED = "confirmada"
STATUS_REJECTED = "rechazada por expertos"
STATUS_UNVALIDATED = "sin validar"


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def species_key(name: str) -> str:
    key = _norm_text(name)
    return SPECIES_SYNONYMS.get(key, key)


def canonical_fname(name: str) -> str | None:
    """``AAAAMMDD_HHMMSS.WAV`` a partir del nombre de un archivo AudioMoth (con o sin número de serie)."""
    match = FNAME_RE.search(Path(name).name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}.WAV"


def device_from_path(path: str | Path) -> str | None:
    """Primer identificador ``AMxxx`` que aparezca en la ruta (carpeta de la grabadora)."""
    for part in reversed(Path(path).parts):
        match = DEVICE_RE.search(part)
        if match:
            return match.group(0).upper()
    return None


def _pick_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    by_norm = {_norm_text(c): c for c in columns}
    for alias in aliases:
        hit = by_norm.get(_norm_text(alias))
        if hit is not None:
            return hit
    return None


def _hour_digits(value) -> str:
    text = str(value).strip()
    if hasattr(value, "strftime"):
        text = value.strftime("%H:%M:%S")
    digits = re.sub(r"\D", "", text.split(".")[0])
    return digits[:6].zfill(6) if digits else ""


@dataclass
class ReferenceTable:
    """Validaciones expertas normalizadas (una fila por detección revisada)."""

    rows: pd.DataFrame
    source: str

    @property
    def n_rows(self) -> int:
        return int(len(self.rows))

    @property
    def n_recordings(self) -> int:
        return int(self.rows[["device", "fname"]].drop_duplicates().shape[0])

    def describe(self) -> str:
        valid = int(self.rows["valid"].sum())
        return (
            f"{Path(self.source).name}: {self.n_rows:,} detecciones revisadas ({valid:,} válidas) "
            f"en {self.n_recordings:,} grabaciones".replace(",", " ")
        )

    def lookup(self, audio_path: str | Path) -> tuple[pd.DataFrame | None, str]:
        """Filas de la grabación y un mensaje; ``None`` si no hay validaciones o es ambigua."""
        fname = canonical_fname(str(audio_path))
        if fname is None:
            return None, "El nombre del archivo no sigue el formato AudioMoth (AAAAMMDD_HHMMSS.WAV)."
        candidates = self.rows[self.rows["fname"] == fname]
        if candidates.empty:
            return None, f"No hay validaciones para {fname} en la tabla cargada."
        device = device_from_path(audio_path)
        if device is not None:
            subset = candidates[candidates["device"] == device]
            if subset.empty:
                return None, f"No hay validaciones para {device} / {fname}."
            return subset.reset_index(drop=True), f"{device} / {fname}"
        devices = sorted(candidates["device"].unique())
        if len(devices) > 1:
            return None, (
                f"{fname} existe en varias grabadoras ({', '.join(devices)}). "
                "Coloca el audio dentro de la carpeta de su grabadora (p. ej. AM005/...)."
            )
        return candidates.reset_index(drop=True), f"{devices[0]} / {fname}"


def load_reference_table(path: str | Path) -> ReferenceTable:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, sep=None, engine="python")
    raw.columns = [str(c).strip() for c in raw.columns]
    columns = list(raw.columns)
    mapping = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        col = _pick_column(columns, aliases)
        if col is not None:
            mapping[field_name] = col
    missing = [f for f in REQUIRED if f not in mapping]
    if missing:
        raise ValueError(
            "La tabla no tiene las columnas necesarias: " + ", ".join(missing)
            + ". Se esperan las columnas de TablaFinal (Nombre científico, Dispositivo, Día, Hora, Inicio (s), Final (s), Valid)."
        )

    out = pd.DataFrame(
        {
            "species": raw[mapping["species"]].astype(str).str.strip(),
            "common_es": raw[mapping["common_es"]].astype(str).str.strip() if "common_es" in mapping else "",
            "device": raw[mapping["device"]].astype(str).str.strip().str.upper(),
            "start": pd.to_numeric(raw[mapping["start"]], errors="coerce"),
            "end": pd.to_numeric(raw[mapping["end"]], errors="coerce"),
            "valid": pd.to_numeric(raw[mapping["valid"]], errors="coerce").fillna(0).astype(int),
            "confidence": pd.to_numeric(raw[mapping["confidence"]], errors="coerce") if "confidence" in mapping else np.nan,
        }
    )
    dates = pd.to_datetime(raw[mapping["day"]], errors="coerce").dt.strftime("%Y%m%d")
    hours = raw[mapping["hour"]].map(_hour_digits)
    out["fname"] = dates + "_" + hours + ".WAV"
    out["species_key"] = out["species"].map(species_key)
    out = out.dropna(subset=["start", "end"])
    out = out[out["fname"].str.len() == len("20250101_000000.WAV")].drop_duplicates()
    return ReferenceTable(rows=out.reset_index(drop=True), source=str(path))


@dataclass
class SpeciesSummary:
    species: str
    common: str
    validated: int
    detected: int
    matched_validated: int
    confirmed_detections: int
    mean_confidence: float

    @property
    def recall(self) -> float | None:
        return self.matched_validated / self.validated if self.validated else None


@dataclass
class ValidationResult:
    recording: str
    n_validated: int
    n_rejected: int
    n_app: int
    matched_validated: int
    confirmed: int
    matched_rejected: int
    species: list[SpeciesSummary] = field(default_factory=list)
    app_status: list[str] = field(default_factory=list)
    reference: pd.DataFrame | None = None
    app_rows: list[dict] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        return self.matched_validated / self.n_validated if self.n_validated else None

    @property
    def confirmed_fraction(self) -> float | None:
        return self.confirmed / self.n_app if self.n_app else None

    @property
    def species_found(self) -> tuple[int, int]:
        validated = [s for s in self.species if s.validated > 0]
        return sum(1 for s in validated if s.matched_validated > 0), len(validated)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def compare_recording(
    app_rows: list[dict],
    reference: pd.DataFrame,
    *,
    recording: str = "",
    min_overlap: float = 0.5,
) -> ValidationResult:
    """Compara las detecciones de la app de UNA grabación con sus validaciones expertas."""
    ref = reference.reset_index(drop=True)
    validated = ref[ref["valid"] == 1]
    rejected = ref[ref["valid"] == 0]
    app = [dict(r) for r in app_rows]
    app_keys = [species_key(r.get("scientific", "")) for r in app]

    def matches(row: dict, key: str, table: pd.DataFrame) -> np.ndarray:
        if table.empty:
            return np.zeros(0, dtype=bool)
        same = (table["species_key"] == key).to_numpy()
        ov = np.array([_overlap(float(row["start"]), float(row["end"]), s, e) for s, e in zip(table["start"], table["end"])])
        return same & (ov >= min_overlap)

    status: list[str] = []
    validated_hit = np.zeros(len(validated), dtype=bool)
    for row, key in zip(app, app_keys):
        hit_valid = matches(row, key, validated)
        if hit_valid.any():
            validated_hit |= hit_valid
            status.append(STATUS_CONFIRMED)
        elif matches(row, key, rejected).any():
            status.append(STATUS_REJECTED)
        else:
            status.append(STATUS_UNVALIDATED)

    # Resumen por especie (unión de especies validadas y detectadas por la app).
    names: dict[str, tuple[str, str]] = {}
    for _, r in validated.iterrows():
        names.setdefault(r["species_key"], (r["species"], r.get("common_es", "") or ""))
    for row, key in zip(app, app_keys):
        names.setdefault(key, (row.get("scientific", ""), row.get("common", "")))
    species = []
    for key, (sci, common) in names.items():
        v_mask = (validated["species_key"] == key).to_numpy()
        idx = [i for i, k in enumerate(app_keys) if k == key]
        confs = [float(app[i].get("confidence", 0.0)) for i in idx]
        species.append(
            SpeciesSummary(
                species=sci,
                common=common,
                validated=int(v_mask.sum()),
                detected=len(idx),
                matched_validated=int((validated_hit & v_mask).sum()),
                confirmed_detections=sum(1 for i in idx if status[i] == STATUS_CONFIRMED),
                mean_confidence=float(np.mean(confs)) if confs else float("nan"),
            )
        )
    species.sort(key=lambda s: (-s.validated, -s.detected, s.species))

    return ValidationResult(
        recording=recording,
        n_validated=int(len(validated)),
        n_rejected=int(len(rejected)),
        n_app=len(app),
        matched_validated=int(validated_hit.sum()),
        confirmed=sum(1 for s in status if s == STATUS_CONFIRMED),
        matched_rejected=sum(1 for s in status if s == STATUS_REJECTED),
        species=species,
        app_status=status,
        reference=ref,
        app_rows=app,
    )


def combine_results(results: list[ValidationResult]) -> ValidationResult:
    """Agrega varias grabaciones en un único resultado (sumas por especie)."""
    total = ValidationResult(recording="Todas las grabaciones con validaciones", n_validated=0, n_rejected=0, n_app=0,
                             matched_validated=0, confirmed=0, matched_rejected=0)
    by_species: dict[str, SpeciesSummary] = {}
    conf_acc: dict[str, list[float]] = {}
    for res in results:
        total.n_validated += res.n_validated
        total.n_rejected += res.n_rejected
        total.n_app += res.n_app
        total.matched_validated += res.matched_validated
        total.confirmed += res.confirmed
        total.matched_rejected += res.matched_rejected
        for s in res.species:
            key = species_key(s.species)
            agg = by_species.setdefault(key, SpeciesSummary(s.species, s.common, 0, 0, 0, 0, float("nan")))
            agg.validated += s.validated
            agg.detected += s.detected
            agg.matched_validated += s.matched_validated
            agg.confirmed_detections += s.confirmed_detections
            if s.detected and not np.isnan(s.mean_confidence):
                conf_acc.setdefault(key, []).extend([s.mean_confidence] * s.detected)
    for key, agg in by_species.items():
        vals = conf_acc.get(key)
        agg.mean_confidence = float(np.mean(vals)) if vals else float("nan")
    total.species = sorted(by_species.values(), key=lambda s: (-s.validated, -s.detected, s.species))
    return total


# --------------------------------------------------------------------------- #
# Gráfico
# --------------------------------------------------------------------------- #
STATUS_COLORS = {STATUS_CONFIRMED: "#0F766E", STATUS_UNVALIDATED: "#64748B", STATUS_REJECTED: "#B42318"}
EXPERT_COLOR = "#16A34A"


def plot_validation(figure, result: ValidationResult, *, audio_path: str | None = None, max_seconds: float = 120.0) -> None:
    """Espectrograma de la grabación (arriba) y línea de tiempo por especie (abajo)."""
    figure.clear()
    has_audio = bool(audio_path) and Path(str(audio_path)).exists()
    if has_audio:
        gs = figure.add_gridspec(2, 1, height_ratios=[0.75, 1.6])
        ax_spec = figure.add_subplot(gs[0])
        ax = figure.add_subplot(gs[1], sharex=ax_spec)
    else:
        ax_spec = None
        ax = figure.add_subplot(111)

    duration = 0.0
    if has_audio:
        try:
            import librosa

            signal, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=max_seconds)
            duration = len(signal) / float(sr)
            hop = max(512, int(len(signal) / 3000))
            spec = np.abs(librosa.stft(signal, n_fft=2048, hop_length=hop))
            spec_db = librosa.amplitude_to_db(spec, ref=float(spec.max()) if spec.size and spec.max() > 0 else 1.0)
            freqs = np.linspace(0, sr / 2, spec.shape[0])
            ax_spec.imshow(
                spec_db,
                origin="lower",
                aspect="auto",
                cmap="magma",
                extent=[0, duration, 0, freqs[-1] / 1000.0],
                vmin=-80,
                vmax=0,
            )
            ax_spec.set_ylim(0, min(12.0, freqs[-1] / 1000.0))
            ax_spec.set_ylabel("kHz")
            ax_spec.tick_params(labelbottom=False)
            for row, status in zip(result.app_rows, result.app_status):
                ax_spec.axvspan(float(row["start"]), float(row["end"]), ymin=0.0, ymax=0.06, color=STATUS_COLORS[status], alpha=0.9, lw=0)
            ax_spec.set_title(result.recording, fontsize=10, loc="left")
        except Exception:
            ax_spec.text(0.5, 0.5, "No se pudo calcular el espectrograma", ha="center", va="center", transform=ax_spec.transAxes)

    ref = result.reference if result.reference is not None else pd.DataFrame(columns=["species_key", "start", "end", "valid", "species"])
    validated = ref[ref["valid"] == 1] if not ref.empty else ref
    keys: list[str] = []
    labels: list[str] = []
    for s in result.species:
        k = species_key(s.species)
        if k not in keys:
            keys.append(k)
            labels.append(s.species)
    if not keys:
        ax.text(0.5, 0.5, "Sin detecciones ni validaciones para esta grabación", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        return

    y_of = {k: len(keys) - 1 - i for i, k in enumerate(keys)}
    for _, r in validated.iterrows():
        y = y_of.get(r["species_key"])
        if y is not None:
            ax.barh(y + 0.2, float(r["end"]) - float(r["start"]), left=float(r["start"]), height=0.34, color=EXPERT_COLOR, alpha=0.85, lw=0)
    for row, status in zip(result.app_rows, result.app_status):
        y = y_of.get(species_key(row.get("scientific", "")))
        if y is None:
            continue
        conf = float(row.get("confidence", 0.0))
        start, end = float(row["start"]), float(row["end"])
        ax.barh(y - 0.2, end - start, left=start, height=0.34, color=STATUS_COLORS[status], alpha=0.35 + 0.65 * min(1.0, conf), lw=0)
        if end - start >= 2.0:
            ax.text((start + end) / 2, y - 0.2, f"{conf * 100:.0f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    duration = max(duration, float(ref["end"].max()) if not ref.empty else 0.0, max((float(r["end"]) for r in result.app_rows), default=0.0))
    ax.set_xlim(0, max(duration, 3.0))
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=8, style="italic")
    ax.set_xlabel("Tiempo (s)")
    ax.grid(axis="x", color="#E2E8F0", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    from matplotlib.patches import Patch

    handles = [
        Patch(color=EXPERT_COLOR, label="Validada por expertos"),
        Patch(color=STATUS_COLORS[STATUS_CONFIRMED], label="App: confirmada"),
        Patch(color=STATUS_COLORS[STATUS_UNVALIDATED], label="App: sin validar"),
        Patch(color=STATUS_COLORS[STATUS_REJECTED], label="App: coincide con un rechazo"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False)


def plot_species_recall(figure, result: ValidationResult) -> None:
    """Barras horizontales: % de las detecciones validadas que la app recupera, por especie."""
    figure.clear()
    ax = figure.add_subplot(111)
    rows = [s for s in result.species if s.validated > 0]
    if not rows:
        ax.text(0.5, 0.5, "Sin detecciones validadas en las grabaciones analizadas", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    rows = sorted(rows, key=lambda s: s.validated)[-25:]
    y = np.arange(len(rows))
    recall = np.array([s.matched_validated / s.validated for s in rows])
    ax.barh(y, np.ones_like(recall) * 100, color="#E2E8F0", height=0.62, lw=0)
    ax.barh(y, recall * 100, color=["#0F766E" if r >= 0.75 else "#D97706" if r >= 0.4 else "#B42318" for r in recall], height=0.62, lw=0)
    for yi, s, r in zip(y, rows, recall):
        ax.text(101, yi, f"{r * 100:.0f} %  ({s.matched_validated}/{s.validated})", va="center", fontsize=8, color="#334155")
    ax.set_yticks(y)
    ax.set_yticklabels([s.species for s in rows], fontsize=8, style="italic")
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Detecciones validadas que la app también detecta (%)")
    ax.set_title(result.recording, fontsize=10, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
