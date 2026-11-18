"""Precision promedio ponderada.

La AP acepta pesos de muestreo (1/p de la grabacion), de modo que estima el valor que se obtendria sobre
el conjunto completo. Usa la misma definicion que scikit-learn (suma de precision x incremento de
exhaustividad, con empates agrupados) pero sobre un orden precalculado, lo que hace barato repetirla
en el bootstrap.
"""

from __future__ import annotations

import numpy as np


class RankedPair:
    """Orden fijo de una especie para un metodo: permite AP ponderada en O(N)."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        order = np.argsort(-s, kind="stable")
        ss = s[order]
        self.order = order
        self.y = y[order].astype(np.float64)
        self.ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])

    def ap(self, w: np.ndarray) -> float:
        ww = w[self.order]
        tp = np.cumsum(ww * self.y)[self.ends]
        fp = np.cumsum(ww * (1.0 - self.y))[self.ends]
        if tp.size == 0 or tp[-1] <= 0:
            return float("nan")
        denom = tp + fp
        prec = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
        rec = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))

    def curve(self, w: np.ndarray):
        ww = w[self.order]
        tp = np.cumsum(ww * self.y)[self.ends]
        fp = np.cumsum(ww * (1.0 - self.y))[self.ends]
        denom = tp + fp
        prec = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
        rec = tp / tp[-1] if tp.size and tp[-1] > 0 else np.zeros_like(tp)
        return prec, rec


def weighted_ap(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    return RankedPair(np.asarray(y), np.asarray(s, dtype=np.float64)).ap(w)
