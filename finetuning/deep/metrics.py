"""Metricas ponderadas y bootstrap por sesion.

Todas las metricas aceptan pesos de muestreo (1/p de la grabacion), de modo que
estiman el valor que se obtendria sobre el conjunto completo. La precision
promedio (AP) se calcula con la misma definicion que scikit-learn (suma de
precision x incremento de exhaustividad, con empates agrupados) pero sobre un
orden precalculado, lo que hace barato repetirla en el bootstrap.

Regimenes de evaluacion (igual que ``finetuning.train``):
    estricto     todo par (ventana, especie) no validado cuenta como negativo
    enmascarado  se excluyen los pares inciertos (BirdNET >= 0,5 sin validar)
    candidatos   solo ventanas donde BirdNET da >= 0,1 a la especie
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..embeddings import flat_sigmoid

CANDIDATE_THRESHOLD = 0.1
REGIMES = ("estricto", "enmascarado", "candidatos")


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


def weighted_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Area bajo la curva ROC ponderada (probabilidad de ordenar bien un par positivo-negativo)."""
    y = np.asarray(y) > 0
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    wp, wn = w[y].sum(), w[~y].sum()
    if wp <= 0 or wn <= 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ss, yy, ww = np.asarray(s)[order], y[order], w[order]
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    neg_cum = np.cumsum(ww * (~yy))
    neg_below = np.r_[0.0, neg_cum[ends[:-1]]]            # negativos con score estrictamente menor
    neg_tie = neg_cum[ends] - neg_below                    # negativos empatados
    pos_group = np.add.reduceat(ww * yy, starts)
    return float(np.sum(pos_group * (neg_below + 0.5 * neg_tie)) / (wp * wn))


def keep_mask(data, k: int, regime: str) -> np.ndarray | None:
    if regime == "estricto":
        return None
    if regime == "enmascarado":
        return ~data.uncertain[:, k]
    if regime == "candidatos":
        return flat_sigmoid(data.logit[:, k]) >= CANDIDATE_THRESHOLD
    raise ValueError(regime)


def species_metrics(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> dict:
    n_pos = int((y > 0).sum())
    out = {"n_pos": n_pos, "n": int(len(y)), "w_pos": float(w[y > 0].sum())}
    if n_pos == 0 or n_pos == len(y):
        return out | {"ap": np.nan, "auc": np.nan, "f1": np.nan, "rec_at_p90": np.nan}
    rp = RankedPair(y, np.asarray(s, dtype=np.float64))
    prec, rec = rp.curve(w.astype(np.float64))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return out | {
        "ap": float(np.sum(np.diff(np.r_[0.0, rec]) * prec)),
        "auc": weighted_auc(y, s, w),
        "f1": float(f1.max()),
        "rec_at_p90": float(rec[prec >= 0.9].max()) if np.any(prec >= 0.9) else 0.0,
    }


def metrics_table(data, scores: dict[str, np.ndarray], regime: str = "estricto", weighted: bool = True, rows: np.ndarray | None = None) -> pd.DataFrame:
    out = []
    w_all = data.weight.astype(np.float64) if weighted else np.ones(len(data.Y))
    for m, sc in scores.items():
        for k, sp in enumerate(data.species):
            keep = keep_mask(data, k, regime)
            if rows is not None:
                sel = np.zeros(len(data.Y), bool)
                sel[rows] = True
                keep = sel if keep is None else keep & sel
            y, s, w = data.Y[:, k], sc[:, k], w_all
            if keep is not None:
                y, s, w = y[keep], s[keep], w[keep]
            out.append({"method": m, "species": sp, "regime": regime} | species_metrics(y, s, w))
    return pd.DataFrame(out)


def summarize(table: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for m, grp in table.groupby("method", sort=False):
        rows.append(
            {
                "method": m,
                "label": (labels or {}).get(m, m),
                "macro_ap": grp["ap"].mean(),
                "median_ap": grp["ap"].median(),
                "macro_auc": grp["auc"].mean(),
                "macro_f1": grp["f1"].mean(),
                "macro_rec_at_p90": grp["rec_at_p90"].mean(),
                "species_evaluated": int(grp["ap"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def macro_ap(data, scores: np.ndarray, regime: str = "estricto", weighted: bool = True, rows: np.ndarray | None = None) -> float:
    return float(metrics_table(data, {"m": scores}, regime, weighted, rows)["ap"].mean())


class SessionBootstrap:
    """Bootstrap pareado por sesiones de la mAP macro (ponderada)."""

    def __init__(self, data, regime: str = "estricto", reps: int = 500, seed: int = 0, weighted: bool = True, species_mask: np.ndarray | None = None):
        self.data, self.regime, self.reps = data, regime, reps
        groups = data.groups
        self.uniq, self.inv = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.uniq), size=(reps, len(self.uniq)))
        self.mult = np.stack([np.bincount(d, minlength=len(self.uniq)) for d in draws]).astype(np.float64)
        self.w = data.weight.astype(np.float64) if weighted else np.ones(len(groups))
        self.keeps = [keep_mask(data, k, regime) for k in range(len(data.species))]
        self.species_mask = np.ones(len(data.species), bool) if species_mask is None else species_mask
        self._cache: dict[str, np.ndarray] = {}

    def replicate_map(self, name: str, scores: np.ndarray) -> np.ndarray:
        """mAP macro en cada replica (reps,)."""
        if name in self._cache:
            return self._cache[name]
        aps = np.full((self.reps, len(self.data.species)), np.nan)
        for k in range(len(self.data.species)):
            if not self.species_mask[k]:
                continue
            keep = self.keeps[k]
            idx = np.arange(len(self.w)) if keep is None else np.flatnonzero(keep)
            y, s = self.data.Y[idx, k], scores[idx, k].astype(np.float64)
            if y.sum() == 0:
                continue
            rp = RankedPair(y, s)
            base, inv = self.w[idx], self.inv[idx]
            for r in range(self.reps):
                aps[r, k] = rp.ap(base * self.mult[r, inv])
        out = np.nanmean(aps, axis=1)
        self._cache[name] = out
        return out

    def compare(self, a: str, sa: np.ndarray, b: str, sb: np.ndarray) -> dict:
        da = self.replicate_map(a, sa) - self.replicate_map(b, sb)
        return {
            "a": a,
            "b": b,
            "regime": self.regime,
            "delta_mean": float(np.mean(da)),
            "ci_low": float(np.percentile(da, 2.5)),
            "ci_high": float(np.percentile(da, 97.5)),
            "p_delta_le_0": float(np.mean(da <= 0)),
            "reps": int(self.reps),
        }

    def interval(self, name: str, scores: np.ndarray) -> dict:
        v = self.replicate_map(name, scores)
        return {"method": name, "regime": self.regime, "mean": float(np.mean(v)), "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5))}


def reviewed_only(data, scores: dict[str, np.ndarray], reps: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Discriminacion entre detecciones validadas y rechazadas (ambas revisadas por expertos).

    No depende de que detecciones decidio revisar el equipo, asi que no esta afectada
    por el sesgo de verificacion. Dos indices:
        auc_pooled   AUC sobre todos los pares revisados (mezcla especies)
        concordance  fraccion de pares (validada, rechazada) de la MISMA especie bien ordenados
    Intervalos por bootstrap de sesiones. n pequeno (decenas de rechazos): leer con cautela.
    """
    pos_i, pos_k = np.nonzero(data.Y > 0)
    neg_i, neg_k = np.nonzero(data.hard_neg)
    groups = data.groups
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    mult = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(reps)]).astype(np.float64)
    species_with_neg = np.unique(neg_k)

    def stats(sc: np.ndarray, wpos: np.ndarray, wneg: np.ndarray) -> tuple[float, float]:
        sp, sn = sc[pos_i, pos_k], sc[neg_i, neg_k]
        y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
        pooled = weighted_auc(y, np.r_[sp, sn], np.r_[wpos, wneg])
        num = den = 0.0
        for k in species_with_neg:
            a, b = pos_k == k, neg_k == k
            if not a.any():
                continue
            ps, ns = sp[a], sn[b]
            pw, nw = wpos[a], wneg[b]
            cmp = (ps[:, None] > ns[None, :]) + 0.5 * (ps[:, None] == ns[None, :])
            num += float(pw @ cmp @ nw)
            den += float(pw.sum() * nw.sum())
        return pooled, (num / den if den > 0 else float("nan"))

    rows = []
    ones_p, ones_n = np.ones(len(pos_i)), np.ones(len(neg_i))
    for m, sc in scores.items():
        pooled, conc = stats(sc, ones_p, ones_n)
        boot = np.array([stats(sc, mult[r, inv[pos_i]], mult[r, inv[neg_i]]) for r in range(reps)])
        rows.append(
            {
                "method": m,
                "n_valid": int(len(pos_i)),
                "n_rejected": int(len(neg_i)),
                "species_with_rejections": int(len(species_with_neg)),
                "auc_pooled": pooled,
                "auc_pooled_lo": float(np.nanpercentile(boot[:, 0], 2.5)),
                "auc_pooled_hi": float(np.nanpercentile(boot[:, 0], 97.5)),
                "concordance": conc,
                "concordance_lo": float(np.nanpercentile(boot[:, 1], 2.5)),
                "concordance_hi": float(np.nanpercentile(boot[:, 1], 97.5)),
            }
        )
    return pd.DataFrame(rows)
