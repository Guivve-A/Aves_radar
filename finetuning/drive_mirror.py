"""Construccion de la copia local (espejo) de los datos de Google Drive.

La carpeta original "AudioMoth - CONDESAN" se trata como SOLO LECTURA: este
modulo nunca modifica ni elimina nada en Drive. Solo escribe archivos nuevos
dentro de la copia local (config.DATA_ROOT), replicando la estructura:

    <DATA_ROOT>/<Grabadora>/<YYYYMMDD>/<Carpeta de audio>/<YYYYMMDD_HHMMSS>.WAV

Dos modos de uso:

1) Desde descargas del conector de Google Drive (entorno sin acceso directo a
   drive.google.com). Cada descarga queda en un JSON {id, title, mimeType,
   content(base64)}; un manifiesto "fileId carpeta nombre" indica a que
   grabadora/fecha/periodo pertenece cada archivo:

       python -m finetuning.drive_mirror decode \
           --tool-results <dir_con_json> --manifest data_cache/work/drive_manifest.txt

2) Con la carpeta de Drive sincronizada localmente (Google Drive para
   escritorio): no hace falta copiar nada, basta con apuntar ECO_DATA_ROOT a la
   carpeta "AudioMoth - CONDESAN" y construir el indice:

       python -m finetuning.drive_mirror index
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import wave
from pathlib import Path

import pandas as pd

from . import config

FOLDERS_JSON = config.WORK_DIR / "drive_folders.json"
INDEX_CSV = config.WORK_DIR / "mirror_index.csv"
AUDIO_SUFFIXES = {".wav"}
# Nombre AudioMoth: [SERIAL_]YYYYMMDD_HHMMSS.WAV (algunas grabadoras anteponen su numero de serie).
_FNAME_RE = re.compile(r"(?:^|_)(\d{8})_(\d{6})\.wav$", re.IGNORECASE)


def canonical_fname(name: str) -> str | None:
    """Devuelve YYYYMMDD_HHMMSS.WAV a partir del nombre original (con o sin prefijo de serie)."""
    match = _FNAME_RE.search(name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}.WAV"


def _period_from_parts(parts: list[str], fname: str) -> str:
    for part in reversed(parts):
        low = part.lower()
        if any(k in low for k in ("sunrise", "morning", "mañana", "manana", "amanecer")):
            return "Sunrise"
        if any(k in low for k in ("sunset", "tarde", "evening", "afternoon", "atardecer", "noche")):
            return "Sunset"
    match = _FNAME_RE.search(fname)
    if match:
        return "Sunrise" if int(match.group(2)[:2]) < 12 else "Sunset"
    return "Unknown"


def _wav_info(raw: bytes) -> tuple[int, int, int, float]:
    with wave.open(io.BytesIO(raw)) as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return w.getnchannels(), rate, w.getsampwidth(), frames / float(rate)


def load_manifest(path: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        file_id, akey, fname = line.split()
        out[file_id] = (akey, fname)
    return out


def decode_tool_results(tool_results: Path, manifest_path: Path, *, overwrite: bool = False) -> pd.DataFrame:
    """Decodifica las descargas del conector y las coloca en el espejo local."""
    folders = json.loads(FOLDERS_JSON.read_text(encoding="utf-8"))
    manifest = load_manifest(manifest_path)
    rows = []
    for json_path in sorted(Path(tool_results).glob("mcp-Google_Drive-download_file_content-*.txt")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        file_id = payload.get("id", "")
        title = payload.get("title", "")
        if file_id not in manifest:
            continue
        akey, expected = manifest[file_id]
        if canonical_fname(title) != expected:
            raise ValueError(f"Titulo inesperado para {file_id}: {title} != {expected}")
        device, date, period = akey.split("/")
        folder_title = folders["audio"][akey]["title"]
        dest = config.DATA_ROOT / device / date / folder_title / title
        raw = base64.b64decode(payload["content"])
        channels, rate, width, duration = _wav_info(raw)
        if dest.exists() and dest.stat().st_size == len(raw) and not overwrite:
            status = "exists"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(raw)
            tmp.replace(dest)
            status = "written"
        rows.append(
            {
                "file_id": file_id,
                "device": device,
                "date": date,
                "period": period,
                "fname": expected,
                "original_name": title,
                "rel_path": str(dest.relative_to(config.DATA_ROOT)),
                "bytes": len(raw),
                "channels": channels,
                "sample_rate": rate,
                "sample_width": width,
                "duration_s": round(duration, 3),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def build_index(root: Path = config.DATA_ROOT) -> pd.DataFrame:
    """Indexa los WAV del espejo (o de la carpeta de Drive sincronizada)."""
    rows = []
    root = Path(root)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        rel = path.relative_to(root)
        parts = list(rel.parts)
        device = next((p for p in parts if re.fullmatch(r"AM\d{3}", p)), "")
        fname = canonical_fname(path.name)
        if not device or fname is None:
            continue  # p. ej. clips de Segment_Analyzer o archivos ajenos
        if any(p.lower().startswith(("results", "segment")) for p in parts[:-1]):
            continue
        period = _period_from_parts(parts[:-1], path.name)
        rows.append(
            {
                "device": device,
                "period": period,
                "fname": fname,
                "file_key": f"{device}/{period}/{fname}",
                "rel_path": str(rel),
                "bytes": path.stat().st_size,
            }
        )
    index = pd.DataFrame(rows).drop_duplicates("file_key") if rows else pd.DataFrame(
        columns=["device", "period", "fname", "file_key", "rel_path", "bytes"]
    )
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(INDEX_CSV, index=False)
    return index


def labeled_scope(index: pd.DataFrame) -> pd.DataFrame:
    """Solo las grabaciones de sesiones (grabadora x dia x periodo) revisadas en TablaFinal.

    Fuera de esas sesiones no hay etiquetas: sus ventanas contarian como negativas
    sin serlo, asi que no se usan para entrenar ni para la muestra de anotacion.
    """
    from .labels import load_detections

    sessions = set(load_detections()["session"])
    date = index["fname"].astype(str).str.slice(0, 8)
    key = index["device"].astype(str) + "_" + date + "_" + index["period"].astype(str)
    out = index[key.isin(sessions)].reset_index(drop=True)
    print(f"Alcance etiquetado: {len(out)} de {len(index)} grabaciones ({len(sessions)} sesiones de TablaFinal)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    dec = sub.add_parser("decode", help="Decodifica descargas del conector al espejo local.")
    dec.add_argument("--tool-results", required=True, type=Path)
    dec.add_argument("--manifest", default=config.WORK_DIR / "drive_manifest.txt", type=Path)
    dec.add_argument("--overwrite", action="store_true")
    sub.add_parser("index", help="Indexa los WAV disponibles en DATA_ROOT.")
    args = parser.parse_args()

    if args.cmd == "decode":
        report = decode_tool_results(args.tool_results, args.manifest, overwrite=args.overwrite)
        log_path = config.WORK_DIR / "mirror_decode_log.csv"
        if log_path.exists():
            prev = pd.read_csv(log_path, dtype=str)
            report = pd.concat([prev, report.astype(str)]).drop_duplicates("file_id", keep="last")
        report.to_csv(log_path, index=False)
        print(report["status"].value_counts().to_dict() if len(report) else "sin descargas")
        bad = report[(report["sample_rate"].astype(str) != "48000") | (report["channels"].astype(str) != "1")]
        if len(bad):
            print("ADVERTENCIA formato inesperado:\n", bad.to_string())
    index = build_index()
    print(f"Indice: {len(index)} grabaciones en {config.DATA_ROOT}")


if __name__ == "__main__":
    main()
