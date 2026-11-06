"""Muestra aleatoria de ventanas para anotacion exhaustiva (conjunto de prueba sin sesgo).

Las ventanas se eligen al azar, SIN mirar las puntuaciones de BirdNET, de modo que
el conjunto sirve para medir lo que BirdNET (y cualquier otro modelo) no detecta.
El muestreo se estratifica por grabadora x periodo (amanecer/atardecer) y las
ventanas siguen la rejilla de 3 s de BirdNET (inicio 0, 3, 6, ... s), por lo que
coinciden exactamente con los embeddings calculados por ``finetuning.embeddings``.

Salida (en ``--out``, una carpeta NUEVA; los audios originales no se modifican):

    anotacion.xlsx   hoja "Anotacion" para rellenar + hoja "Especies" de referencia
    anotacion.csv    la misma tabla en CSV
    clips/           un WAV de 5 s por ventana (1 s de contexto a cada lado)

Instrucciones para el anotador: escuchar el clip y anotar TODAS las especies que se
oyen entre los segundos 1 y 4 del clip (la ventana de 3 s), aunque sean lejanas.

Uso:
    python -m finetuning.sample_annotation --n 300 --out /ruta/anotacion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CONTEXT_S = 1.0
ANNOTATION_COLUMNS = [
    "especies_presentes",  # nombres cientificos separados por ';' (vacio si no hay aves)
    "sin_aves",  # 'si' si no se oye ningun ave
    "dudosas",  # especies con identificacion insegura, separadas por ';'
    "ruido",  # lluvia, viento, insectos, motor, etc.
    "anotador",
    "notas",
]


def _excluded_sessions() -> set[str]:
    qc = config.RESULTS_DIR / "qc_sessions.csv"
    if not qc.exists():
        return set()
    df = pd.read_csv(qc)
    return set(df.loc[df["excluded"].astype(bool), "session"])


def sample_windows(
    index: pd.DataFrame,
    n: int = 300,
    *,
    seed: int = config.RANDOM_SEED,
    window_s: float = config.SIG_LENGTH,
    recording_s: float = config.RECORDING_LENGTH,
    exclude_sessions: set[str] | None = None,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Elige ``n`` ventanas al azar, repartidas por igual entre estratos grabadora x periodo.

    Con ``data_root`` se lee la duracion real de cada grabacion elegida (no todas las
    grabadoras graban 55 s); las ilegibles o mas cortas que una ventana se descartan.
    """
    idx = index.copy()
    idx["date"] = idx["fname"].str.slice(0, 8)
    idx["session"] = idx["device"] + "_" + idx["date"] + "_" + idx["period"]
    if exclude_sessions:
        idx = idx[~idx["session"].isin(exclude_sessions)]
    if idx.empty:
        raise ValueError("No hay grabaciones para muestrear.")

    rng = np.random.default_rng(seed)
    strata = sorted(idx.groupby(["device", "period"]).groups)
    base, extra = divmod(n, len(strata))
    durations: dict[str, float] = {}

    def grid_for(rec) -> np.ndarray:
        if data_root is None:
            dur = recording_s
        else:
            if rec["file_key"] not in durations:
                import soundfile as sf

                try:
                    durations[rec["file_key"]] = float(sf.info(str(Path(data_root) / rec["rel_path"])).duration)
                except Exception:
                    durations[rec["file_key"]] = 0.0
            dur = durations[rec["file_key"]]
        return np.arange(0.0, dur - window_s + 1e-6, window_s)

    rows = []
    for j, key in enumerate(rng.permutation(len(strata))):
        dev, per = strata[key]
        quota = base + (1 if j < extra else 0)
        pool = idx[(idx["device"] == dev) & (idx["period"] == per)]
        # dict (no set): conserva el orden de extraccion, asi la muestra es reproducible
        # aunque Python aleatorice el hash de las cadenas en cada proceso.
        chosen: dict[tuple[str, float], None] = {}
        attempts = 0
        while len(chosen) < quota and attempts < 50 * quota:
            attempts += 1
            rec = pool.iloc[int(rng.integers(len(pool)))]
            grid = grid_for(rec)
            if len(grid) == 0:
                continue  # grabacion ilegible o demasiado corta
            chosen[(rec["file_key"], float(rng.choice(grid)))] = None
        for file_key, start in chosen:
            rec = pool[pool["file_key"] == file_key].iloc[0]
            rows.append(
                {
                    "device": dev,
                    "period": per,
                    "date": f"{rec['date'][:4]}-{rec['date'][4:6]}-{rec['date'][6:8]}",
                    "time": f"{rec['fname'][9:11]}:{rec['fname'][11:13]}:{rec['fname'][13:15]}",
                    "file_key": file_key,
                    "rel_path": rec["rel_path"],
                    "start_s": start,
                    "end_s": start + window_s,
                }
            )
    out = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.insert(0, "id", [f"W{i + 1:04d}" for i in range(len(out))])
    return out


def write_clips(sample: pd.DataFrame, data_root: Path, out_dir: Path, context_s: float = CONTEXT_S) -> list[str]:
    """Escribe un WAV por ventana con contexto; devuelve los nombres (vacio si fallo)."""
    import soundfile as sf

    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for row in sample.itertuples(index=False):
        stamp = f"{row.date.replace('-', '')}_{row.time.replace(':', '')}"
        name = f"{row.id}_{row.device}_{stamp}_{int(row.start_s):02d}s.wav"
        path = Path(data_root) / row.rel_path
        try:
            try:
                with sf.SoundFile(str(path)) as f:
                    sr = f.samplerate
                    a = max(0, int((row.start_s - context_s) * sr))
                    b = min(f.frames, int((row.end_s + context_s) * sr))
                    f.seek(a)
                    audio = f.read(b - a, dtype="int16", always_2d=False)
            except Exception:
                # Algunos WAV (cabecera con tamano erroneo o en Drive) no admiten seek:
                # se lee de forma secuencial y se recorta en memoria.
                audio, sr = sf.read(str(path), dtype="int16", always_2d=False)
                a = max(0, int((row.start_s - context_s) * sr))
                audio = audio[a : int((row.end_s + context_s) * sr)]
            if len(audio) == 0:
                raise ValueError("la ventana esta fuera del audio (archivo truncado)")
            sf.write(str(clip_dir / name), audio, sr, subtype="PCM_16")
            names.append(name)
        except Exception as ex:  # archivo corrupto o inaccesible
            print(f"  aviso: no se pudo cortar {row.rel_path}: {ex}")
            names.append("")
    return names


def species_reference() -> pd.DataFrame:
    try:
        from .labels import load_detections, species_table

        return species_table(load_detections()).reset_index()[["species", "common_es", "common_en", "positives"]]
    except Exception:
        return pd.DataFrame(columns=["species", "common_es", "common_en", "positives"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="Numero de ventanas a anotar.")
    parser.add_argument("--out", type=Path, required=True, help="Carpeta de salida (se crea; no debe ser la de los audios).")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--no-clips", action="store_true", help="Solo la hoja, sin cortar clips.")
    parser.add_argument("--sobrescribir", action="store_true", help="Permite reemplazar una hoja existente (se pierde lo anotado).")
    args = parser.parse_args()

    out = args.out.resolve()
    if out == config.DATA_ROOT.resolve() or config.DATA_ROOT.resolve() in out.parents:
        raise SystemExit("La carpeta de salida no puede estar dentro de la carpeta de audios original.")
    if (out / "anotacion.xlsx").exists() and not args.sobrescribir:
        raise SystemExit(
            f"Ya existe {out / 'anotacion.xlsx'}: no se sobrescribe para no perder anotaciones. "
            "Usa otra carpeta o --sobrescribir."
        )
    out.mkdir(parents=True, exist_ok=True)

    from .drive_mirror import INDEX_CSV, build_index, labeled_scope

    index = labeled_scope(pd.read_csv(INDEX_CSV) if INDEX_CSV.exists() else build_index())
    sample = sample_windows(index, args.n, seed=args.seed, exclude_sessions=_excluded_sessions(), data_root=config.DATA_ROOT)
    sample["clip"] = "" if args.no_clips else write_clips(sample, config.DATA_ROOT, out)
    for col in ANNOTATION_COLUMNS:
        sample[col] = ""

    sample.to_csv(out / "anotacion.csv", index=False)
    with pd.ExcelWriter(out / "anotacion.xlsx") as xw:
        sample.to_excel(xw, sheet_name="Anotacion", index=False)
        species_reference().to_excel(xw, sheet_name="Especies", index=False)
        pd.DataFrame(
            {
                "Instrucciones": [
                    "Escuche cada clip (5 s). La ventana a anotar es la de los segundos 1 a 4.",
                    "Anote en 'especies_presentes' TODAS las especies audibles, con nombre cientifico, separadas por ';'.",
                    "Si no hay aves escriba 'si' en 'sin_aves'. Especies inseguras van en 'dudosas'.",
                    "No use las detecciones de BirdNET para decidir: el objetivo es medir lo que el detector no ve.",
                ]
            }
        ).to_excel(xw, sheet_name="Instrucciones", index=False)
    counts = sample.groupby(["device", "period"]).size()
    print(f"{len(sample)} ventanas muestreadas en {len(counts)} estratos -> {out}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
