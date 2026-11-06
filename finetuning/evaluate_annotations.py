"""Evaluacion sin sesgo con la muestra aleatoria anotada exhaustivamente.

Compara cada metodo con la anotacion experta de ventanas elegidas al azar
(``finetuning.sample_annotation``). Como esas ventanas no se eligieron con
BirdNET, la anotacion incluye los cantos que BirdNET no detecta, y la
comparacion deja de estar sesgada a su favor.

Las puntuaciones son las predicciones fuera de muestra de la validacion cruzada
(``oof_scores.npz`` de ``finetuning.train``): el modelo que puntua cada ventana
no se entreno con su sesion, y la anotacion nunca se usa para entrenar.

Uso:
    python -m finetuning.evaluate_annotations --anotacion /ruta/anotacion.xlsx

Las filas sin anotar (``especies_presentes`` vacio y ``sin_aves`` distinto de
'si') se ignoran, de modo que se puede evaluar con la anotacion a medias.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config
from .train import METHOD_LABELS, OOF_NPZ

BIRDNET_THRESHOLDS = (0.1, 0.25, 0.5)
MIN_POS_SPECIES = 3


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def load_annotations(path: Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, sheet_name="Anotacion") if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    for col in ("especies_presentes", "sin_aves", "dudosas"):
        df[col] = df[col].fillna("").astype(str) if col in df.columns else ""
    done = (df["especies_presentes"].str.strip() != "") | df["sin_aves"].map(_norm).isin(["si", "s", "yes", "x", "1"])
    return df[done].reset_index(drop=True)


def name_index(species: list[str], info: pd.DataFrame | None) -> dict[str, str]:
    """Nombre normalizado (cientifico o comun) -> nombre cientifico de la lista objetivo."""
    idx = {_norm(sp): sp for sp in species}
    if info is not None:
        for sp, row in info.iterrows():
            for col in ("common_es", "common_en"):
                if col in info.columns and isinstance(row[col], str) and row[col]:
                    idx.setdefault(_norm(row[col]), sp)
    return idx


def parse_species(cell: str, names: dict[str, str]) -> tuple[set[str], set[str]]:
    found, unknown = set(), set()
    for raw in re.split(r"[;,\n]", str(cell)):
        key = _norm(raw)
        if not key:
            continue
        if key in names:
            found.add(names[key])
        else:
            unknown.add(raw.strip())
    return found, unknown


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        yy = y[i].ravel()
        if yy.sum() == 0:
            continue
        deltas.append(average_precision_score(yy, a[i].ravel()) - average_precision_score(yy, b[i].ravel()))
    d = np.array(deltas)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def evaluate(ann: pd.DataFrame, oof_path: Path = OOF_NPZ, info: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    data = np.load(oof_path, allow_pickle=False)
    if "file_key" not in data.files:
        raise SystemExit("oof_scores.npz no tiene file_key/start: vuelve a ejecutar python -m finetuning.train.")
    species = [str(s) for s in data["species"]]
    pos = {(str(k), round(float(t), 2)): i for i, (k, t) in enumerate(zip(data["file_key"], data["start"]))}
    names = name_index(species, info)

    rows_idx, y_true, uncertain_names, unknown_names, dudosas = [], [], [], set(), 0
    missing = []
    for r in ann.itertuples(index=False):
        key = (str(r.file_key), round(float(r.start_s), 2))
        if key not in pos:
            missing.append(r.id)
            continue
        present, unknown = parse_species(r.especies_presentes, names)
        unsure, _ = parse_species(getattr(r, "dudosas", ""), names)
        unknown_names |= unknown
        dudosas += len(unsure)
        row = np.zeros(len(species), dtype=np.float32)
        for sp in present:
            row[species.index(sp)] = 1.0
        rows_idx.append(pos[key])
        y_true.append(row)
        uncertain_names.append(unsure)
    if not rows_idx:
        raise SystemExit(
            f"Ninguna de las {len(ann)} ventanas anotadas tiene puntuaciones fuera de muestra: "
            "comprueba que la hoja y el entrenamiento correspondan a los mismos datos."
        )

    rows_idx = np.array(rows_idx)
    Y = np.vstack(y_true)
    Y_tabla = data["Y"][rows_idx]
    methods = [k[4:] for k in data.files if k.startswith("oof_")]
    scores = {m: data[f"oof_{m}"][rows_idx] for m in methods}
    n_pos = int(Y.sum())

    out: dict = {
        "windows_annotated": int(len(ann)),
        "windows_evaluated": int(len(rows_idx)),
        "windows_without_scores": missing,
        "windows_with_target_species": int((Y.sum(axis=1) > 0).sum()),
        "presences_target_species": n_pos,
        "uncertain_mentions": int(dudosas),
        "unknown_or_non_target_names": sorted(unknown_names),
    }
    if n_pos == 0:
        out["note"] = "Aun no hay presencias de especies objetivo anotadas."
        return out, pd.DataFrame()

    # 1) Cuanto ve TablaFinal (detecciones de BirdNET revisadas) de lo que realmente suena.
    tp = float((Y_tabla * Y).sum())
    out["tablafinal_recall"] = tp / n_pos
    out["tablafinal_precision"] = tp / max(float(Y_tabla.sum()), 1.0)

    # 2) BirdNET a umbrales fijos.
    sig = scores.get("birdnet")  # ya es la probabilidad sigmoide de BirdNET
    if sig is not None:
        out["birdnet_at_threshold"] = {
            str(t): {
                "recall": float(((sig >= t) * Y).sum() / n_pos),
                "precision": float(((sig >= t) * Y).sum() / max(float((sig >= t).sum()), 1.0)),
            }
            for t in BIRDNET_THRESHOLDS
        }

    # 3) Metricas por metodo (agregadas sobre todos los pares ventana-especie).
    per_method = {}
    for m in methods:
        s = scores[m]
        keep_sp = Y.sum(axis=0) >= MIN_POS_SPECIES
        macro = [average_precision_score(Y[:, k], s[:, k]) for k in np.where(keep_sp)[0]]
        per_method[m] = {
            "label": METHOD_LABELS.get(m, m),
            "micro_ap": float(average_precision_score(Y.ravel(), s.ravel())),
            "macro_ap": float(np.mean(macro)) if macro else None,
            "species_in_macro": int(keep_sp.sum()),
        }
    out["methods"] = per_method

    # 4) Diferencias con IC 95 % (bootstrap por ventanas).
    comps = [("ft_birdnet", "birdnet"), ("lineal_l2", "birdnet"), ("lineal_l2", "prototipos")]
    out["bootstrap_micro_ap"] = []
    for a, b in comps:
        if a in scores and b in scores:
            mean, lo, hi = _bootstrap_delta(Y, scores[a], scores[b])
            out["bootstrap_micro_ap"].append({"a": a, "b": b, "delta_mean": mean, "ci_low": lo, "ci_high": hi})

    # 5) Tabla por especie.
    rows = []
    for k, sp in enumerate(species):
        npos = int(Y[:, k].sum())
        if npos == 0 and Y_tabla[:, k].sum() == 0:
            continue
        row = {"species": sp, "presences": npos, "in_tablafinal": int((Y_tabla[:, k] * Y[:, k]).sum())}
        for m in methods:
            row[f"ap_{m}"] = float(average_precision_score(Y[:, k], scores[m][:, k])) if 0 < npos < len(Y) else np.nan
        rows.append(row)
    return out, pd.DataFrame(rows).sort_values("presences", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anotacion", type=Path, required=True, help="anotacion.xlsx o anotacion.csv rellenada.")
    parser.add_argument("--oof", type=Path, default=OOF_NPZ)
    args = parser.parse_args()

    ann = load_annotations(args.anotacion)
    if ann.empty:
        print(
            "La hoja todavia no tiene ventanas anotadas (filas con 'especies_presentes' o con "
            "'sin_aves' = si). No hay nada que evaluar; vuelve a ejecutar este paso cuando el equipo anote."
        )
        return
    info_path = config.RESULTS_DIR / "species_info.csv"
    info = pd.read_csv(info_path, index_col=0) if info_path.exists() else None
    summary, per_species = evaluate(ann, args.oof, info)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "annotated_eval.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not per_species.empty:
        per_species.to_csv(config.RESULTS_DIR / "annotated_eval_species.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
