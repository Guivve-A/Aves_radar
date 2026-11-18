"""Protocolo de evaluacion consciente del proceso de etiquetado.

Como se generaron las etiquetas (comprobado sobre la muestra, ver ``review_cells``):

1. BirdNET propone una deteccion si su probabilidad es >= 0,25 (``GATE``).
2. El equipo reviso las propuestas por "celdas" sesion x especie, de forma casi todo-o-nada:
   en 604 de 708 celdas con propuestas estan listadas >= 90 % de ellas; en 92 no hay ninguna.
3. De lo revisado, el experto valido (15 913) o rechazo (79).

Categorias de cada par (ventana, especie):

    P   validada por un experto                                   -> positivo
    H   rechazada por un experto                                  -> negativo fiable (72 utiles)
    E   BirdNET < 0,05 (``SILENT``), nadie la reviso              -> negativo "facil"
    G   0,05 <= BirdNET < 0,25, nadie la reviso                   -> desconocido (zona gris)
    U   BirdNET >= 0,25 sin listar, celda no revisada o parcial   -> desconocido
    R0  BirdNET >= 0,25 sin listar en una celda revisada          -> negativo (casi no hay)
    X   listada pero con BirdNET recalculado < 0,25               -> audio y etiqueta no coinciden: se excluye

Consecuencia: todo positivo tiene BirdNET >= 0,25, asi que la precision promedio de BirdNET
solo baja de 1 por las propuestas sin revisar (U) y por los 72 rechazos. Un modelo solo puede
"ganarle" en mAP adivinando que celdas se revisaron. Por eso la mAP se usa aqui como control
de no inferioridad y las mejoras se miden donde son medibles:

    M1  mAP "limpia" (P frente a H, R0 y E; U, G y X fuera)   control: no perder mas de 0,01
    SLI indice de fuga de seleccion: 0,5 = la correccion aprendida no distingue P de U
    M2  colocacion de los rechazos entre las validadas de su especie y grabadora (AUC estratificada)
    M3  deteccion de cantos validados atenuados 6-24 dB, con el mismo volumen de alertas que BirdNET

Reglas de decision fijadas antes de entrenar:
    D1 seguro:  el limite inferior del IC de M1(modelo) - M1(BirdNET) esta por encima de -0,01
    D2 mejor descartando falsos positivos: el IC pareado de M2 esta por encima de 0
    D3 mas robusto a la perdida de senal:  el IC pareado de M3 esta por encima de 0
    D4 exportable: D1 y (D2 o D3), sin perdida significativa en la otra; a igualdad, el mas simple
Ninguna regla usa la mAP estricta ni la enmascarada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..embeddings import flat_sigmoid
from .metrics import RankedPair, weighted_ap

GATE = 0.25
SILENT = 0.05
KAPPA = 0.9
E, G, U, R0, P, H, X = range(7)
CAT_NAMES = {E: "E", G: "G", U: "U", R0: "R0", P: "P", H: "H", X: "X"}
NONINFERIORITY = 0.01
LEVELS = (0.0, 6.0, 12.0, 18.0, 24.0)
# Correccion acotada: con |correccion| <= 0,9 logits, un par donde BirdNET calla (< 0,05, logit < -2,94) no puede
# quedar por encima de uno validado (>= 0,25, logit >= -1,10), porque 0,9 < (2,944 - 1,099) / 2. La mAP limpia solo
# puede cambiar por el orden entre validadas y rechazadas: el control de seguridad se cumple por construccion.
BOUND = 0.9


# --------------------------------------------------------------------------- #
# Categorias y celdas de revision
# --------------------------------------------------------------------------- #
def review_cells(data) -> pd.DataFrame:
    """Una fila por celda sesion x especie con propuestas de BirdNET (>= 0,25), ponderada por 1/p."""
    s0 = flat_sigmoid(data.logit)
    w = data.weight.astype(np.float64)[:, None]
    det = s0 >= GATE
    listed = (data.Y > 0) | data.hard_neg
    sess, si = np.unique(data.groups, return_inverse=True)
    n_det = np.zeros((len(sess), data.Y.shape[1]))
    n_listed = np.zeros_like(n_det)
    np.add.at(n_det, si, w * det)
    np.add.at(n_listed, si, w * (det & listed))
    rows = []
    for i, s in enumerate(sess):
        for k in np.flatnonzero(n_det[i] > 0):
            kappa = n_listed[i, k] / n_det[i, k]
            rows.append((s, data.species[k], n_det[i, k], n_listed[i, k], kappa, "revisada" if kappa >= KAPPA else ("nunca" if n_listed[i, k] == 0 else "parcial")))
    return pd.DataFrame(rows, columns=["session", "species", "detections", "listed", "kappa", "status"])


def categories(data) -> np.ndarray:
    """Matriz (N, S) int8 con la categoria de cada par."""
    s0 = flat_sigmoid(data.logit)
    w = data.weight.astype(np.float64)[:, None]
    det = s0 >= GATE
    listed = (data.Y > 0) | data.hard_neg
    _, si = np.unique(data.groups, return_inverse=True)
    n_sess = si.max() + 1
    n_det = np.zeros((n_sess, data.Y.shape[1]))
    n_listed = np.zeros_like(n_det)
    np.add.at(n_det, si, w * det)
    np.add.at(n_listed, si, w * (det & listed))
    reviewed_cell = (n_listed >= KAPPA * np.maximum(n_det, 1e-12)) & (n_det > 0)
    cat = np.full(data.Y.shape, E, dtype=np.int8)
    cat[(s0 >= SILENT) & (s0 < GATE)] = G
    unlisted = det & ~listed
    cat[unlisted] = U
    cat[unlisted & reviewed_cell[si]] = R0
    cat[(data.Y > 0)] = P
    cat[data.hard_neg] = H
    cat[listed & ~det] = X
    return cat


def category_counts(data, cat: np.ndarray) -> dict:
    w = data.weight.astype(np.float64)
    return {CAT_NAMES[c]: {"pairs": int((cat == c).sum()), "weighted": float((w[:, None] * (cat == c)).sum())} for c in CAT_NAMES}


# --------------------------------------------------------------------------- #
# Pliegues
# --------------------------------------------------------------------------- #
def make_folds(data, cat: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Pliegues por sesion equilibrados en ventanas, rechazos y positivos de cada especie.

    Asignacion voraz y determinista: las sesiones con mas "masa rara" (rechazos y positivos de
    especies escasas) se reparten primero; cada una va al pliegue que deja las cargas mas parejas.
    Solo usa etiquetas y conteos, nunca predicciones de modelos entrenados.
    """
    sess, si = np.unique(data.groups, return_inverse=True)
    S = data.Y.shape[1]
    feats = np.zeros((len(sess), S + 2))
    np.add.at(feats[:, :S], si, (cat == P).astype(np.float64))
    np.add.at(feats[:, S], si, (cat == H).sum(axis=1).astype(np.float64))
    np.add.at(feats[:, S + 1], si, np.ones(len(si)))
    tot = np.maximum(feats.sum(axis=0), 1.0)
    share = feats / tot                                   # fraccion del total que aporta cada sesion
    importance = share[:, :S].sum(axis=1) + 3.0 * share[:, S] + share[:, S + 1]
    weights = np.r_[np.ones(S), 3.0, 3.0]                 # rechazos y ventanas pesan mas que una especie
    load = np.zeros((n_splits, S + 2))
    fold_of = np.zeros(len(sess), dtype=np.int64)
    for i in np.lexsort((sess, -importance)):
        cost = [float((weights * (load[f] + share[i]) ** 2).sum() + (weights * np.delete(load, f, axis=0) ** 2).sum()) for f in range(n_splits)]
        f = int(np.argmin(cost))
        fold_of[i] = f
        load[f] += share[i]
    return fold_of[si]


def load_folds(data, cat: np.ndarray, path) -> np.ndarray:
    """Lee los pliegues guardados o los crea (son deterministas) si faltan o no corresponden a los datos."""
    if path.exists():
        folds = np.load(path)
        if len(folds) == len(data.Y):
            return folds
    folds = make_folds(data, cat)
    np.save(path, folds)
    return folds


# --------------------------------------------------------------------------- #
# Bootstrap de sesiones dentro de cada grabadora
# --------------------------------------------------------------------------- #
class Resampler:
    """Multiplicidades de bootstrap: se remuestrean sesiones con reemplazo dentro de cada grabadora."""

    def __init__(self, data, reps: int = 1000, seed: int = 0):
        self.sessions, self.inv = np.unique(data.groups, return_inverse=True)
        dev = pd.Series(data.meta["device"].to_numpy()).groupby(self.inv).first().to_numpy()
        rng = np.random.default_rng(seed)
        mult = np.zeros((reps, len(self.sessions)))
        for d in np.unique(dev):
            idx = np.flatnonzero(dev == d)
            draws = rng.integers(0, len(idx), size=(reps, len(idx)))
            for r in range(reps):
                mult[r, idx] = np.bincount(draws[r], minlength=len(idx))
        self.mult = mult
        self.reps = reps

    def rows(self, r: int) -> np.ndarray:
        return self.mult[r, self.inv]


def paired(a: np.ndarray, b: np.ndarray, point_a: float, point_b: float) -> dict:
    d = a - b
    return {"delta": float(point_a - point_b), "ci_low": float(np.nanpercentile(d, 2.5)), "ci_high": float(np.nanpercentile(d, 97.5)), "p_le_0": float(np.nanmean(d <= 0))}


def ci(v: np.ndarray, point: float) -> dict:
    return {"value": float(point), "ci_low": float(np.nanpercentile(v, 2.5)), "ci_high": float(np.nanpercentile(v, 97.5))}


# --------------------------------------------------------------------------- #
# M1: mAP por regimen
# --------------------------------------------------------------------------- #
def regime_mask(data, cat: np.ndarray, k: int, regime: str) -> np.ndarray:
    c = cat[:, k]
    if regime == "limpio":
        return np.isin(c, (P, H, R0, E))
    if regime == "estricto":
        return c != X
    if regime == "enmascarado":
        return (c != X) & ~((data.Y[:, k] == 0) & (flat_sigmoid(data.logit[:, k]) >= 0.5))
    raise ValueError(regime)


class APEvaluator:
    """mAP macro ponderada por regimen, con replicas de bootstrap baratas (orden precalculado)."""

    def __init__(self, data, cat: np.ndarray, resampler: Resampler):
        self.data, self.cat, self.rs = data, cat, resampler
        self.w = data.weight.astype(np.float64)
        self.masks = {r: [np.flatnonzero(regime_mask(data, cat, k, r)) for k in range(data.Y.shape[1])] for r in ("limpio", "estricto", "enmascarado")}

    def evaluate(self, scores: np.ndarray, regime: str, boot: bool = True):
        """Devuelve (mAP, AP por especie, mAP por replica o None)."""
        S = self.data.Y.shape[1]
        aps = np.full(S, np.nan)
        reps = np.full((self.rs.reps, S), np.nan) if boot else None
        for k in range(S):
            idx = self.masks[regime][k]
            y = (self.cat[idx, k] == P).astype(np.float64)
            if y.sum() == 0:
                continue
            rp = RankedPair(y, scores[idx, k].astype(np.float64))
            base = self.w[idx]
            aps[k] = rp.ap(base)
            if boot:
                inv = self.rs.inv[idx]
                for r in range(self.rs.reps):
                    reps[r, k] = rp.ap(base * self.rs.mult[r, inv])
        return float(np.nanmean(aps)), aps, (np.nanmean(reps, axis=1) if boot else None)


def clean_map(data, cat: np.ndarray, scores: np.ndarray, rows: np.ndarray) -> float:
    """mAP limpia ponderada sobre ``rows`` (``scores`` alineado con ``rows``); solo especies con algun positivo ahi."""
    w = data.weight[rows].astype(np.float64)
    aps = []
    for k in range(cat.shape[1]):
        c = cat[rows, k]
        keep = np.isin(c, (P, H, R0, E))
        y = (c[keep] == P).astype(np.float64)
        if y.sum() > 0:
            aps.append(weighted_ap(y, scores[keep, k].astype(np.float64), w[keep]))
    return float(np.mean(aps)) if aps else float("nan")


def clean_map_by_fold(data, cat: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> float:
    """mAP limpia calculada dentro de cada pliegue de prueba y promediada (analisis secundario).

    La medida principal junta todas las ventanas: una falsa alarma cuenta aunque en su pliegue no haya ningun
    positivo de la especie. Dentro de un pliegue esa especie no se evalua, y la falsa alarma no se ve.
    """
    out = []
    for f in sorted(np.unique(folds)):
        rows = np.flatnonzero(folds == f)
        out.append(clean_map(data, cat, scores[rows], rows))
    return float(np.mean(out))


# --------------------------------------------------------------------------- #
# SLI: indice de fuga de seleccion
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def bounded(scores: np.ndarray, base_logit: np.ndarray, bound: float = BOUND) -> np.ndarray:
    """sigma(logit de BirdNET + correccion recortada a +-bound). Conserva los NaN (mezclas que no son de prueba)."""
    z0 = np.clip(np.asarray(base_logit, dtype=np.float64), -20.0, 20.0)
    delta = np.clip(_logit(scores) - z0, -bound, bound)
    return flat_sigmoid(z0 + delta).astype(np.float32)


def selection_leak_index(data, cat: np.ndarray, scores: np.ndarray, bins: int = 5, min_each: int = 5) -> dict:
    """AUC de (logit del modelo - logit de BirdNET) entre pares validados (P) y sin revisar (U),
    dentro de cada especie y de cada quintil del score de BirdNET. 0,5 = la correccion aprendida
    no distingue lo revisado de lo no revisado; > 0,5 = el modelo esta aprendiendo el patron de revision."""
    delta = _logit(scores) - np.clip(data.logit.astype(np.float64), -16, 16)
    s0 = flat_sigmoid(data.logit)
    num = den = 0.0
    per_species = {}
    for k, sp in enumerate(data.species):
        p_rows, u_rows = np.flatnonzero(cat[:, k] == P), np.flatnonzero(cat[:, k] == U)
        if len(p_rows) < min_each or len(u_rows) < min_each:
            continue
        edges = np.quantile(s0[np.r_[p_rows, u_rows], k], np.linspace(0, 1, bins + 1))
        edges[-1] += 1e-9
        s_num = s_den = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            pp = p_rows[(s0[p_rows, k] >= a) & (s0[p_rows, k] < b)]
            uu = u_rows[(s0[u_rows, k] >= a) & (s0[u_rows, k] < b)]
            if len(pp) == 0 or len(uu) == 0:
                continue
            dp, du = delta[pp, k], delta[uu, k]
            auc = ((dp[:, None] > du[None, :]).mean() + 0.5 * (dp[:, None] == du[None, :]).mean())
            s_num += auc * len(uu)
            s_den += len(uu)
        if s_den > 0:
            per_species[sp] = s_num / s_den
            num += s_num
            den += s_den
    return {"sli": float(num / den) if den > 0 else float("nan"), "species": per_species, "pairs_u": int(den)}


# --------------------------------------------------------------------------- #
# M2: colocacion de los rechazos
# --------------------------------------------------------------------------- #
class RejectionPlacement:
    """Para cada rechazo, fraccion ponderada de detecciones validadas de la misma especie y grabadora
    que quedan por encima (empates 1/2). Media sobre rechazos = AUC estratificada. Ambas clases fueron
    revisadas, asi que la seleccion de que revisar se cancela."""

    def __init__(self, data, cat: np.ndarray, resampler: Resampler, min_same_device: int = 5):
        self.data, self.cat, self.rs = data, cat, resampler
        dev = data.meta["device"].to_numpy()
        self.w = data.weight.astype(np.float64)
        self.items = []
        for j, k in zip(*np.nonzero(cat == H)):
            pos = np.flatnonzero(cat[:, k] == P)
            same = pos[dev[pos] == dev[j]]
            ref = same if len(same) >= min_same_device else pos
            if len(ref) == 0:
                continue
            self.items.append((int(j), int(k), ref, len(same) >= min_same_device))
        self.n = len(self.items)

    def evaluate(self, scores: np.ndarray, boot: bool = True):
        pv = np.zeros(self.n)
        reps = np.zeros((self.rs.reps, 2)) if boot else None      # numerador y denominador por replica
        cmp_cache = []
        for i, (j, k, ref, _) in enumerate(self.items):
            c = (scores[ref, k] > scores[j, k]) + 0.5 * (scores[ref, k] == scores[j, k])
            pv[i] = float((self.w[ref] * c).sum() / self.w[ref].sum())
            cmp_cache.append(c)
        point = float(pv.mean()) if self.n else float("nan")
        boots = None
        if boot and self.n:
            boots = np.zeros(self.rs.reps)
            for r in range(self.rs.reps):
                m = self.rs.rows(r)
                num = den = 0.0
                for (j, k, ref, _), c in zip(self.items, cmp_cache):
                    wr = self.w[ref] * m[ref]
                    if m[j] > 0 and wr.sum() > 0:
                        num += m[j] * float((wr * c).sum() / wr.sum())
                        den += m[j]
                boots[r] = num / den if den > 0 else np.nan
        return point, pv, boots

    def removed_at_retention(self, scores: np.ndarray, retention: float = 0.95) -> dict:
        """Rechazos que quedan por debajo del umbral que conserva ``retention`` de las validadas de su especie."""
        removed = 0
        for j, k, _, _ in self.items:
            pos = np.flatnonzero(self.cat[:, k] == P)
            ps, pw = scores[pos, k], self.w[pos]
            order = np.argsort(ps)
            cw = np.cumsum(pw[order])
            thr = ps[order][min(int(np.searchsorted(cw, (1 - retention) * cw[-1])), len(ps) - 1)]
            removed += int(scores[j, k] < thr)
        return {"removed": removed, "of": self.n}


# --------------------------------------------------------------------------- #
# M3: cantos atenuados con el mismo volumen de alertas que BirdNET
# --------------------------------------------------------------------------- #
class AttenuationBenchmark:
    """Exhaustividad sobre cantos validados atenuados (mezclas ``test``), fuera de muestra.

    Punto de operacion principal ("presupuesto de alertas"): por especie, el umbral del modelo que
    marca en el audio real el mismo volumen ponderado de ventanas que BirdNET con 0,25. No usa
    etiquetas y da a todos los modelos la misma carga de revision. Secundario: 1 % de falsas alarmas
    sobre ventanas donde BirdNET calla para esa especie (E) y mezclas fondo + fondo.
    """

    def __init__(self, data, cat: np.ndarray, mix, resampler: Resampler, rows: np.ndarray | None = None, min_pos: int = 10, fa_rate: float = 0.01):
        self.data, self.cat, self.rs, self.min_pos, self.fa_rate = data, cat, resampler, min_pos, fa_rate
        N = len(data.Y)
        inside = np.ones(N, bool)
        if rows is not None:
            inside[:] = False
            inside[rows] = True
        self.real = np.flatnonzero(inside)
        self.w = data.weight.astype(np.float64)
        self.s0 = flat_sigmoid(data.logit)
        src, kind, att = mix.src, mix.kind, mix.plan["atten_db"].to_numpy()
        self.src = src
        self.test = {a: np.flatnonzero((kind == "test") & np.isclose(att, a) & inside[src]) for a in LEVELS[1:]}
        self.bg = np.flatnonzero((kind == "bg_test") & inside[src])
        self.bg_silent = flat_sigmoid(mix.logit[self.bg]) < SILENT
        self.mix_rows = np.sort(np.r_[self.bg, *self.test.values()])
        self.sources = np.unique(src[(kind == "test") & inside[src]])
        self.species = [k for k in range(data.Y.shape[1]) if (cat[self.sources, k] == P).sum() >= min_pos]

    def prepare(self, clean: np.ndarray, mixed: np.ndarray) -> list[dict]:
        out = []
        for k in self.species:
            real_scores = clean[self.real, k].astype(np.float64)
            order = np.argsort(real_scores, kind="stable")
            null_real = self.real[self.cat[self.real, k] == E]
            null_scores = np.r_[clean[null_real, k], mixed[self.bg[self.bg_silent[:, k]], k]].astype(np.float64)
            null_owner = np.r_[null_real, self.src[self.bg[self.bg_silent[:, k]]]]
            n_order = np.argsort(null_scores, kind="stable")
            src0 = self.sources[self.cat[self.sources, k] == P]
            levels = {0.0: (clean[src0, k].astype(np.float64), src0)}
            for a, rows in self.test.items():
                sel = rows[self.cat[self.src[rows], k] == P]
                levels[a] = (mixed[sel, k].astype(np.float64), self.src[sel])
            out.append(
                {
                    "k": k,
                    "real_sorted": real_scores[order],
                    "real_rows": self.real[order],
                    "alert": (self.s0[self.real, k] >= GATE)[order],
                    "null_sorted": null_scores[n_order],
                    "null_owner": null_owner[n_order],
                    "levels": levels,
                }
            )
        return out

    def recalls(self, prep: list[dict], mult: np.ndarray | None = None, mode: str = "presupuesto") -> np.ndarray:
        """(especies, niveles). ``mult``: multiplicidad de bootstrap por fila del conjunto.

        En el modo "presupuesto" el volumen de alertas del modelo iguala exactamente al de BirdNET. El volumen solo
        puede cambiar de ventana en ventana, asi que se interpola entre el umbral mas bajo que no supera el presupuesto
        y el siguiente (equivale a incluir al azar la ventana del borde con la probabilidad que cuadra el volumen).
        Las ventanas con peso cero (sesiones que no entran en la replica) no pueden fijar el umbral.
        """
        w = self.w if mult is None else self.w * mult
        res = np.full((len(prep), len(LEVELS)), np.nan)
        for i, p in enumerate(prep):
            if mode == "presupuesto":
                wr = w[p["real_rows"]]
                budget = float((wr * p["alert"]).sum())            # volumen de alertas de BirdNET a 0,25
                if wr.sum() <= 0 or budget <= 0:
                    continue
                keep = wr > 0
                ws, ss = wr[keep], p["real_sorted"][keep]
                first = np.r_[True, ss[1:] != ss[:-1]]              # un nivel por score distinto: los empates entran juntos
                levels = ss[first]
                vol = np.cumsum(ws[::-1])[::-1][first]              # peso con score >= cada nivel (decreciente)
                j = int(np.searchsorted(-vol, -budget * (1 + 1e-9), side="left"))    # primer nivel cuyo volumen cabe
                t_hi, v_hi = (levels[j], float(vol[j])) if j < len(levels) else (np.inf, 0.0)
                lam = 0.0
                if j > 0:
                    lam = (budget - v_hi) / (float(vol[j - 1]) - v_hi)
                    lam = 0.0 if lam < 1e-9 else min(lam, 1.0)
                cuts = [(t_hi, 1.0 - lam)] + ([(levels[j - 1], lam)] if lam > 0 else [])
                strict = False                                      # alerta si score >= umbral
            else:
                wn = w[p["null_owner"]]
                cw = np.cumsum(wn)
                if cw[-1] <= 0:
                    continue
                cuts = [(p["null_sorted"][min(int(np.searchsorted(cw, (1 - self.fa_rate) * cw[-1], side="left")), len(cw) - 1)], 1.0)]
                strict = True
            for jx, a in enumerate(LEVELS):
                ps, owner = p["levels"][a]
                pw = w[owner]
                if pw.sum() > 0:
                    res[i, jx] = sum(share * float((pw * ((ps > thr) if strict else (ps >= thr))).sum() / pw.sum()) for thr, share in cuts)
        return res

    def evaluate(self, clean: np.ndarray, mixed: np.ndarray, mode: str = "presupuesto", boot: bool = True):
        """Devuelve (curva macro por nivel, resumen S = media de 6-24 dB, replicas de la curva, tabla por especie)."""
        prep = self.prepare(clean, mixed)
        point = self.recalls(prep, None, mode)
        curve = np.nanmean(point, axis=0)
        reps = None
        if boot:
            reps = np.stack([np.nanmean(self.recalls(prep, self.rs.rows(r), mode), axis=0) for r in range(self.rs.reps)])
        table = pd.DataFrame(point, columns=[f"{a:g}" for a in LEVELS])
        table.insert(0, "species", [self.data.species[p["k"]] for p in prep])
        table.insert(1, "n_pos", [len(p["levels"][0.0][0]) for p in prep])
        return curve, float(np.nanmean(curve[1:])), reps, table

    def summary_score(self, clean: np.ndarray, mixed: np.ndarray, mode: str = "presupuesto") -> float:
        """S: exhaustividad media sobre 6, 12, 18 y 24 dB (macro por especie). Para la seleccion anidada."""
        curve = np.nanmean(self.recalls(self.prepare(clean, mixed), None, mode), axis=0)
        return float(np.nanmean(curve[1:]))
