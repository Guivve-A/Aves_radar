"""Prueba de robustez con verdad conocida: deteccion de cantos atenuados.

Positivos: ventanas con una especie validada, originales (0 dB) y atenuadas 6, 12, 18 y
24 dB sobre un fondo de la misma sesion (``mixtures``). Negativos: ventanas de fondo
originales y mezclas fondo + fondo. Todas las puntuaciones son fuera de muestra (el
modelo que puntua una ventana no vio su sesion al entrenar).

Para comparar modelos con escalas distintas se fija, por especie, el umbral que deja
pasar una fraccion ``fpr`` de los negativos, y se mide la exhaustividad a ese umbral.
Tambien se informa el AUC (independiente del umbral).

Nota: los negativos se eligieron como ventanas donde BirdNET da < 0,1 a todas las especies
objetivo, de modo que el umbral de BirdNET resulta muy bajo. La prueba es, por tanto,
conservadora para cualquier modelo distinto de BirdNET.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import mixtures as MX
from .metrics import weighted_auc

LEVELS = (0.0, 6.0, 12.0, 18.0, 24.0)


class SNRBenchmark:
    def __init__(self, data, plan: pd.DataFrame, fpr: float = 0.005, min_pos: int = 10):
        self.data, self.fpr, self.min_pos = data, fpr, min_pos
        self.src = plan["src"].to_numpy()
        kind = plan["kind"].to_numpy()
        att = plan["atten_db"].to_numpy()
        self.bg_clean = np.flatnonzero(MX.background_mask(data))
        self.bg_mix = np.flatnonzero(kind == "bg_test")
        self.test_rows = {a: np.flatnonzero((kind == "test") & np.isclose(att, a)) for a in MX.TEST_ATTENUATIONS}
        self.sources = np.unique(self.src[kind == "test"])  # las mismas ventanas de origen en todos los niveles
        self.w = data.weight.astype(np.float64)
        self.uniq, self.inv = np.unique(data.groups, return_inverse=True)
        # filas (del conjunto original) que aportan el peso de cada negativo
        self.neg_owner = np.r_[self.bg_clean, self.src[self.bg_mix]]

    def prepare(self, clean: np.ndarray, mixed: np.ndarray) -> list[dict]:
        """Ordena una vez los negativos de cada especie y reune los positivos por nivel."""
        out = []
        for k, sp in enumerate(self.data.species):
            src0 = self.sources[self.data.Y[self.sources, k] > 0]
            if len(src0) < self.min_pos:
                continue
            neg = np.r_[clean[self.bg_clean, k], mixed[self.bg_mix, k]].astype(np.float64)
            order = np.argsort(neg, kind="stable")
            levels = {0.0: (clean[src0, k].astype(np.float64), src0)}
            for a, rows in self.test_rows.items():
                sel = rows[self.data.Y[self.src[rows], k] > 0]
                levels[a] = (mixed[sel, k].astype(np.float64), self.src[sel])
            out.append({"species": sp, "neg_sorted": neg[order], "neg_owner": self.neg_owner[order], "levels": levels})
        return out

    def recalls(self, prep: list[dict], mult: np.ndarray | None = None) -> np.ndarray:
        """Exhaustividad (especies, niveles) al umbral de falsos positivos fijado."""
        w = self.w if mult is None else self.w * mult
        res = np.full((len(prep), len(LEVELS)), np.nan)
        for i, p in enumerate(prep):
            cw = np.cumsum(w[p["neg_owner"]])
            if cw[-1] <= 0:
                continue
            thr = p["neg_sorted"][min(int(np.searchsorted(cw, (1.0 - self.fpr) * cw[-1], side="left")), len(cw) - 1)]
            for j, a in enumerate(LEVELS):
                ps, owner = p["levels"][a]
                pw = w[owner]
                if pw.sum() > 0:
                    res[i, j] = float((pw * (ps > thr)).sum() / pw.sum())
        return res

    def species_table(self, prep: list[dict]) -> pd.DataFrame:
        rec = self.recalls(prep)
        rows = []
        for i, p in enumerate(prep):
            neg_w = self.w[p["neg_owner"]]
            for j, a in enumerate(LEVELS):
                ps, owner = p["levels"][a]
                y = np.r_[np.ones(len(ps)), np.zeros(len(neg_w))]
                rows.append({"species": p["species"], "atten_db": a, "n_pos": int(len(ps)), "recall": rec[i, j], "auc": weighted_auc(y, np.r_[ps, p["neg_sorted"]], np.r_[self.w[owner], neg_w])})
        return pd.DataFrame(rows)

    def summary(self, scores: dict[str, tuple[np.ndarray, np.ndarray]], reps: int = 300, seed: int = 0, baseline: str = "birdnet") -> pd.DataFrame:
        """Media macro por nivel de atenuacion, IC por bootstrap de sesiones y diferencia frente a ``baseline``."""
        rng = np.random.default_rng(seed)
        mults = [np.bincount(rng.integers(0, len(self.uniq), len(self.uniq)), minlength=len(self.uniq)).astype(np.float64)[self.inv] for _ in range(reps)]
        preps = {m: self.prepare(c, x) for m, (c, x) in scores.items()}
        point = {m: np.nanmean(self.recalls(p), axis=0) for m, p in preps.items()}
        aucs = {m: self.species_table(p).groupby("atten_db")["auc"].mean() for m, p in preps.items()}
        boots = {m: np.stack([np.nanmean(self.recalls(p, mult), axis=0) for mult in mults]) for m, p in preps.items()}
        rows = []
        for m in scores:
            for j, a in enumerate(LEVELS):
                row = {
                    "method": m,
                    "atten_db": a,
                    "recall": float(point[m][j]),
                    "recall_lo": float(np.nanpercentile(boots[m][:, j], 2.5)),
                    "recall_hi": float(np.nanpercentile(boots[m][:, j], 97.5)),
                    "auc": float(aucs[m].get(a, np.nan)),
                    "species": int(len(preps[m])),
                }
                if baseline in boots and m != baseline:
                    d = boots[m][:, j] - boots[baseline][:, j]
                    row |= {"delta_recall": float(point[m][j] - point[baseline][j]), "delta_lo": float(np.nanpercentile(d, 2.5)), "delta_hi": float(np.nanpercentile(d, 97.5))}
                rows.append(row)
        return pd.DataFrame(rows)
