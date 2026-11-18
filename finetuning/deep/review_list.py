"""Ventanas para escuchar: donde el modelo afinado, el verificador y BirdNET discrepan (aprendizaje activo).

Las etiquetas actuales solo cubren lo que BirdNET propuso y alguien reviso. Lo que mas informacion aporta ahora
es escuchar justo las ventanas en disputa. La lista usa predicciones fuera de muestra: cada ventana la puntua un
modelo que no vio su sesion.

Grupos (hasta ``--per-species`` ventanas por especie y grupo, y como mucho dos por grabacion):
    nuevas      BirdNET no propuso la especie (< 0,25) y el modelo afinado si (>= 0,25); de mayor a menor puntuacion
    dudosas     propuestas de BirdNET que nadie reviso y que el verificador local puntua por debajo de su corte
    probables   propuestas de BirdNET que nadie reviso y que el verificador puntua mas alto

La lista contiene detecciones individuales: se escribe en la carpeta de trabajo (no versionada) y no se publica.
En ``finetuning/results/deep`` solo queda un resumen con conteos por grupo.

Uso:
    python -m finetuning.deep.review_list --model b43
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import config
from ..embeddings import flat_sigmoid
from . import data as D
from . import probes as PB
from . import protocol as PR

RESULTS = config.RESULTS_DIR / "deep"
OUT = D.DEEP_DIR / "ventanas_para_revisar.xlsx"
WINDOW_S = 3.0


def pick(rows: np.ndarray, order_key: np.ndarray, rec: np.ndarray, limit: int, per_rec: int = 2) -> np.ndarray:
    """Primeras ``limit`` filas segun ``order_key`` (ascendente), con un maximo por grabacion."""
    seen: dict = {}
    out = []
    for i in np.argsort(order_key, kind="stable"):
        r = rec[rows[i]]
        if seen.get(r, 0) >= per_rec:
            continue
        seen[r] = seen.get(r, 0) + 1
        out.append(rows[i])
        if len(out) >= limit:
            break
    return np.asarray(out, dtype=np.int64)


def build(data: D.DeepData, cat: np.ndarray, model: np.ndarray, verifier: np.ndarray, cutoff: np.ndarray, per_species: int) -> tuple[pd.DataFrame, dict]:
    s0 = flat_sigmoid(data.logit)
    w = data.weight.astype(np.float64)
    rec = data.meta["rec"].to_numpy()
    frames, summary = [], {"nuevas": {}, "dudosas": {}, "probables": {}}
    for k, sp in enumerate(data.species):
        unseen = np.isin(cat[:, k], (PR.E, PR.G))
        new = np.flatnonzero(unseen & (model[:, k] >= PR.GATE))
        unreviewed = np.flatnonzero(cat[:, k] == PR.U)
        doubtful = unreviewed[verifier[unreviewed, k] < cutoff[k]] if cutoff[k] > 0 else unreviewed[:0]
        likely = unreviewed[verifier[unreviewed, k] >= cutoff[k]] if cutoff[k] > 0 else unreviewed[:0]
        groups = (
            ("nuevas", new, -model[new, k]),
            ("dudosas", doubtful, verifier[doubtful, k]),
            ("probables", likely, -verifier[likely, k]),
        )
        for name, rows, key in groups:
            summary[name][sp] = {"pares": int(len(rows)), "estimacion_conjunto_completo": float(w[rows].sum())}
            chosen = pick(rows, key, rec, per_species)
            if not len(chosen):
                continue
            meta = data.meta.iloc[chosen]
            frames.append(
                pd.DataFrame(
                    {
                        "grupo": name,
                        "especie": sp,
                        "grabadora": meta["device"].to_numpy(),
                        "archivo": meta["file_key"].to_numpy(),
                        "inicio_s": meta["start"].to_numpy(dtype=float),
                        "fin_s": meta["start"].to_numpy(dtype=float) + WINDOW_S,
                        "birdnet": np.round(s0[chosen, k], 3),
                        "modelo_afinado": np.round(model[chosen, k], 3),
                        "verificador": np.round(verifier[chosen, k], 3),
                        "presente (si / no / duda)": "",
                        "notas": "",
                    }
                )
            )
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="b43", help="Modelo afinado cuyas propuestas nuevas se listan (clave de oof_ra.npz).")
    parser.add_argument("--per-species", type=int, default=6)
    args = parser.parse_args()
    data = D.load(maps=True)
    cat = PR.categories(data)
    model = np.load(D.DEEP_DIR / "oof_ra.npz")[f"oof_{args.model}"]
    verifier = np.load(PB.OOF_NPZ)["oof_lineal"]
    cutoff, _ = PB.verification_cutoffs(data, cat, verifier)
    table, summary = build(data, cat, model, verifier, cutoff, args.per_species)
    table.to_excel(OUT, index=False)
    totals = {g: {"pares": int(sum(v["pares"] for v in d.values())), "estimacion_conjunto_completo": float(sum(v["estimacion_conjunto_completo"] for v in d.values())), "en_la_lista": int((table["grupo"] == g).sum()) if len(table) else 0} for g, d in summary.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "review_list_summary.json").write_text(json.dumps({"modelo": args.model, "por_especie_y_grupo": args.per_species, "totales": totals, "por_especie": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(totals, indent=2, ensure_ascii=False))
    print(f"{len(table)} ventanas en {OUT}")


if __name__ == "__main__":
    main()
