"""Analisis complementarios que no dependen de etiquetas nuevas.

``new_detections``: detecciones que un modelo propone y que BirdNET no propuso (< 0,25),
con su "corroboracion temporal": que fraccion cae junto a una deteccion VALIDADA de la
misma especie en la misma grabacion. Las aves suelen vocalizar varias veces seguidas, asi
que una deteccion nueva pegada a una validada es probablemente real; se compara con la
tasa base de una ventana cualquiera. No es una verdad de campo, pero es una evidencia
independiente (los modelos por ventana no ven a las vecinas) y no cuesta anotacion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..embeddings import flat_sigmoid

REVIEW_THRESHOLD = 0.25


def _neighbor_positive(data, radius: int) -> np.ndarray:
    """(N, S) bool: hay un positivo validado de la especie a <= ``radius`` ventanas en la misma grabacion (sin contar la propia)."""
    rec = data.meta["rec"].to_numpy()
    pos = data.meta["pos"].to_numpy()
    order = np.lexsort((pos, rec))
    Y = data.Y[order] > 0
    r, p = rec[order], pos[order]
    out = np.zeros_like(Y)
    for d in range(1, radius + 1):
        same = (r[d:] == r[:-d]) & (p[d:] - p[:-d] == d)
        out[d:][same] |= Y[:-d][same]
        out[:-d][same] |= Y[d:][same]
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    return out[back]


def new_detections(data, scores: dict[str, np.ndarray], recall_target: float = 0.9, radius: int = 2, min_pos: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resumen por metodo y detalle por especie de las detecciones nuevas respecto a BirdNET.

    Punto de operacion por especie: el umbral del modelo que conserva ``recall_target`` de las
    detecciones validadas (ponderado). Una deteccion "nueva" supera ese umbral en una ventana
    donde BirdNET dio < 0,25 (nadie la reviso).
    """
    bn = flat_sigmoid(data.logit)
    near = _neighbor_positive(data, radius)
    w = data.weight.astype(np.float64)
    unreviewed = (bn < REVIEW_THRESHOLD) & (data.Y == 0) & (~data.hard_neg)
    rows = []
    for m, sc in scores.items():
        for k, sp in enumerate(data.species):
            pos = data.Y[:, k] > 0
            if pos.sum() < min_pos:
                continue
            ps, pw = sc[pos, k], w[pos]
            order = np.argsort(ps)
            cw = np.cumsum(pw[order])
            thr = float(ps[order][np.searchsorted(cw, (1 - recall_target) * cw[-1])])
            pool = unreviewed[:, k]
            new = pool & (sc[:, k] >= thr)
            rows.append(
                {
                    "method": m,
                    "species": sp,
                    "threshold": thr,
                    "n_new": int(new.sum()),
                    "w_new": float(w[new].sum()),
                    "w_validated": float(w[pos].sum()),
                    "adjacent_new": float((w[new] * near[new, k]).sum()),
                    "base_rate": float((w[pool] * near[pool, k]).sum() / max(w[pool].sum(), 1e-12)),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail
    detail["frac_adjacent"] = detail["adjacent_new"] / detail["w_new"].replace(0, np.nan)
    summary = (
        detail.groupby("method", sort=False)
        .apply(
            lambda g: pd.Series(
                {
                    "new_detections": int(g["n_new"].sum()),
                    "new_weighted": float(g["w_new"].sum()),
                    "new_per_100_validated": float(100 * g["w_new"].sum() / g["w_validated"].sum()),
                    "frac_adjacent": float(g["adjacent_new"].sum() / max(g["w_new"].sum(), 1e-12)),
                    "base_rate": float((g["base_rate"] * g["w_new"]).sum() / max(g["w_new"].sum(), 1e-12)),
                    "species": int(len(g)),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    summary["lift"] = summary["frac_adjacent"] / summary["base_rate"].replace(0, np.nan)
    return summary, detail
