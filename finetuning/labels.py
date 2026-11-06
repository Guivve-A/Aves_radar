"""Lectura de las validaciones expertas (TablaFinal.xlsx) y ventanas multietiqueta.

Cada fila de TablaFinal es una deteccion de BirdNET (ventana de 3 s de una
grabacion AudioMoth de 55 s) revisada por un ornitologo:

    Valid = 1  -> la especie esta presente en la ventana (verdadero positivo).
    Valid = 0  -> BirdNET se equivoco (falso positivo, "negativo duro").

Una misma ventana puede contener varias especies, por lo que el problema se
modela como clasificacion multietiqueta.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

COLUMN_MAP = {
    "Nombre científico": "species",
    "Nombre inglés": "common_en",
    "Nombre español": "common_es",
    "IUCN": "iucn",
    "Gremio alimenticio": "guild",
    "Dispositivo": "device",
    "Día": "day",
    "Periodo": "period",
    "Hora": "hour",
    "Inicio (s)": "start",
    "Final (s)": "end",
    "Duración (s)": "duration",
    "Low Freq (Hz)": "low_freq",
    "High Freq (Hz)": "high_freq",
    "Peak Freq (Hz)": "peak_freq",
    "Valid": "valid",
    "Confidence Score": "birdnet_conf",
}


def load_detections(path: Path | str = config.TABLA_FINAL) -> pd.DataFrame:
    """Carga TablaFinal.xlsx con columnas normalizadas y sin duplicados exactos."""
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns=COLUMN_MAP)
    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en {path}: {missing}")

    df["species"] = df["species"].astype(str).str.strip()
    df["device"] = df["device"].astype(str).str.strip()
    df["period"] = df["period"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["day"]).dt.strftime("%Y%m%d")
    df["hour"] = df["hour"].astype(str).str.strip()
    df["fname"] = df["date"] + "_" + df["hour"].str.replace(":", "", regex=False) + ".WAV"
    df["session"] = df["device"] + "_" + df["date"] + "_" + df["period"]
    df["file_key"] = df["device"] + "/" + df["period"] + "/" + df["fname"]
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["valid"] = df["valid"].astype(int)

    key = ["file_key", "start", "species", "valid"]
    df = df.drop_duplicates(subset=key).reset_index(drop=True)
    return df


def species_table(det: pd.DataFrame) -> pd.DataFrame:
    """Resumen por especie: positivos, negativos duros y metadatos."""
    meta = det.groupby("species")[["common_en", "common_es", "iucn", "guild"]].first()
    counts = det.groupby("species")["valid"].agg(n="size", positives="sum")
    counts["hard_negatives"] = counts["n"] - counts["positives"]
    files = det[det["valid"] == 1].groupby("species")["file_key"].nunique().rename("files")
    sessions = det[det["valid"] == 1].groupby("species")["session"].nunique().rename("sessions")
    out = meta.join(counts).join(files).join(sessions).fillna(0)
    out[["files", "sessions"]] = out[["files", "sessions"]].astype(int)
    return out.sort_values("positives", ascending=False)


def target_species(det: pd.DataFrame, min_positives: int = config.MIN_POSITIVES_TARGET) -> list[str]:
    table = species_table(det)
    return table.index[table["positives"] >= min_positives].tolist()


@dataclass(frozen=True)
class WindowKey:
    file_key: str
    start: float


def build_windows(det: pd.DataFrame) -> pd.DataFrame:
    """Agrupa detecciones por ventana (archivo, inicio) en etiquetas multietiqueta.

    Columnas resultantes:
        file_key, device, date, period, session, fname, start, end,
        pos_species (lista), neg_species (lista), max_conf
    """
    rows = []
    for (file_key, start), grp in det.groupby(["file_key", "start"], sort=True):
        first = grp.iloc[0]
        pos = sorted(set(grp.loc[grp["valid"] == 1, "species"]))
        neg = sorted(set(grp.loc[grp["valid"] == 0, "species"]) - set(pos))
        rows.append(
            {
                "file_key": file_key,
                "device": first["device"],
                "date": first["date"],
                "period": first["period"],
                "session": first["session"],
                "fname": first["fname"],
                "start": float(start),
                "end": float(grp["end"].max()),
                "pos_species": pos,
                "neg_species": neg,
                "max_conf": float(grp["birdnet_conf"].max()),
            }
        )
    return pd.DataFrame(rows)
