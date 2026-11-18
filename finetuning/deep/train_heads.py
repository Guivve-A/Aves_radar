"""Validacion cruzada de las cabezas sobre embeddings (muestra estratificada, metricas ponderadas).

Protocolo
---------
* 5 pliegues agrupados por sesion (grabadora x dia x periodo). Dentro de cada pliegue, el
  15 % de las sesiones de entrenamiento se reserva para la parada temprana y para elegir
  entre configuraciones candidatas: los pliegues de prueba no intervienen en ninguna decision.
* Metricas ponderadas por 1/p (estiman las del conjunto completo) en tres regimenes
  (estricto, enmascarado, candidatos), bootstrap pareado por sesiones frente a BirdNET,
  discriminacion entre detecciones validadas y rechazadas, y prueba de cantos atenuados.

Uso:
    python -m finetuning.deep.train_heads                       # metodos principales
    python -m finetuning.deep.train_heads --methods lineal,residual
    python -m finetuning.deep.train_heads --lodo                # ademas, dejando una grabadora fuera
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

from .. import config
from ..embeddings import flat_sigmoid
from . import data as D
from . import heads as H
from . import metrics as M
from . import mixtures as MX
from . import tflite_tools as T
from .snr import SNRBenchmark

RESULTS = config.RESULTS_DIR / "deep"
OOF_NPZ = D.DEEP_DIR / "oof_heads.npz"

LABELS = {
    "birdnet": "BirdNET V2.4 (sin ajuste)",
    "prototipos": "Prototipos coseno (app actual)",
    "lineal_v1": "Sonda lineal, receta anterior",
    "ft_v1": "Ultima capa afinada (L2-SP), receta anterior",
    "lineal": "Sonda lineal, receta nueva",
    "mlp": "MLP 2x512, receta nueva",
    "lineal_mix": "Sonda lineal + mezclas atenuadas",
    "residual_estricto": "Adaptador residual (candidatos como negativos)",
    "residual": "Adaptador residual anclado a BirdNET",
    "residual_mix": "Adaptador residual + mezclas atenuadas",
    "contexto": "Adaptador residual con contexto temporal (GRU)",
    "birdnet_x_lineal": "BirdNET x sonda local (media geometrica)",
    "birdnet_x_lineal_mix": "BirdNET x sonda local con mezclas (media geometrica)",
}

# Configuraciones fijadas de antemano. Cuando hay varias, se elige por pliegue con la validacion interna.
_SCRATCH = H.HeadConfig(kind="linear", lr=2e-3, weight_decay=1e-2, epochs=60, patience=60, candidates="ignore")
_RES = H.HeadConfig(kind="residual", hidden=512, depth=2, dropout=0.3, lr=5e-4, weight_decay=5e-2, epochs=40, patience=8)
METHODS: dict[str, list[H.HeadConfig]] = {
    "lineal": [replace(_SCRATCH, l2_input=l2) for l2 in (False, True)],
    "mlp": [replace(_SCRATCH, kind="mlp", hidden=512, depth=2, dropout=0.3, lr=1e-3, weight_decay=5e-2, mixup=0.2, epochs=40, patience=40, batch_size=1024)],
    "lineal_mix": [replace(_SCRATCH, mixture_weight=0.5, l2_input=l2) for l2 in (False, True)],
    "residual_estricto": [replace(_RES, candidates="negative", anchor=a) for a in (0.02, 0.2)],
    "residual": [replace(_RES, anchor=a) for a in (0.02, 0.2)],
    "residual_mix": [replace(_RES, anchor=a, mixture_weight=0.5) for a in (0.02, 0.2)],
    "contexto": [replace(_RES, kind="context", anchor=a, dropout=0.3) for a in (0.02, 0.2)],
}
LEGACY = ("prototipos", "lineal_v1", "ft_v1")
DEFAULT = ["birdnet", "prototipos", "lineal_v1", "ft_v1", "lineal", "mlp", "residual_estricto", "residual", "contexto"]
MIX_METHODS = ["lineal_mix", "residual_mix"]


def birdnet_dense(data: D.DeepData) -> tuple[np.ndarray, np.ndarray]:
    """Columnas de la capa de clasificacion de BirdNET para las especies objetivo: W (1024, S), b (S)."""
    model, graph, tensors, ops = T.graph_ops()
    fc = next(op for op in reversed(ops) if op["op"] == "FULLY_CONNECTED")
    W = T.constant(model, graph, fc["inputs"][1])[data.class_index]
    b = T.constant(model, graph, fc["inputs"][2])[data.class_index]
    return W.T.astype(np.float32).copy(), b.astype(np.float32)


def subset(data: D.DeepData, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "emb": data.emb[rows],
        "logit": data.logit[rows],
        "Y": data.Y[rows],
        "hard_neg": data.hard_neg[rows],
        "weight": data.weight[rows],
        "rec": data.meta["rec"].to_numpy()[rows],
        "pos": data.meta["pos"].to_numpy()[rows],
    }


class Mixtures:
    """Mezclas atenuadas: caracteristicas, etiquetas heredadas de la ventana de origen y candidatos."""

    def __init__(self, data: D.DeepData, threshold: float = 0.10):
        self.plan = MX.load_plan()
        feats = MX.load_features(maps=True)
        done = np.load(MX.MIX_DIR / "done.npy")
        if not done.all():
            raise RuntimeError("Faltan mezclas por extraer (python -m finetuning.deep.mixtures --extract).")
        self.emb = np.asarray(feats["emb"], dtype=np.float32)
        self.logit = np.asarray(feats["logit"], dtype=np.float32)
        self.src = self.plan["src"].to_numpy()
        self.kind = self.plan["kind"].to_numpy()
        is_pos = np.isin(self.kind, ("test", "train"))
        self.Y = np.where(is_pos[:, None], data.Y[self.src], 0.0).astype(np.float32)
        src_cand = H.candidate_mask(data.Y, data.logit, data.hard_neg, threshold)[self.src] | data.hard_neg[self.src]
        bg_cand = H.candidate_mask(data.Y, data.logit, data.hard_neg, threshold)[self.plan["bg"].to_numpy()]
        mix_cand = (flat_sigmoid(self.logit) >= threshold) & (self.Y == 0)
        self.candidate = (src_cand | bg_cand | mix_cand) & (self.Y == 0)
        self.weight = data.weight[self.src]

    def training_rows(self, fit_rows: np.ndarray) -> dict[str, np.ndarray]:
        sel = np.flatnonzero(np.isin(self.kind, ("train", "bg_train")) & np.isin(self.src, fit_rows))
        return {
            "emb": self.emb[sel],
            "logit": self.logit[sel],
            "Y": self.Y[sel],
            "hard_neg": np.zeros(self.Y[sel].shape, bool),
            "weight": self.weight[sel],
            "candidate": self.candidate[sel],
        }

    def test_rows(self, test_rows: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(self.kind, ("test", "bg_test")) & np.isin(self.src, test_rows))


def val_metric_fn(data: D.DeepData, rows: np.ndarray):
    """mAP macro ponderada en regimen enmascarado sobre las sesiones de validacion interna."""
    Y, w, unc = data.Y[rows], data.weight[rows].astype(np.float64), data.uncertain[rows]
    usable = [k for k in range(Y.shape[1]) if Y[:, k].sum() > 0]

    def metric(prob: np.ndarray) -> float:
        aps = []
        for k in usable:
            keep = ~unc[:, k]
            aps.append(M.weighted_ap(Y[keep, k], prob[keep, k].astype(np.float64), w[keep]))
        return float(np.nanmean(aps)) if aps else float("nan")

    return metric


MAP_TOLERANCE = 0.01


class RobustSelector:
    """Criterio de seleccion para los modelos entrenados con mezclas atenuadas.

    La mAP de validacion premia parecerse a BirdNET (las etiquetas salen de BirdNET), asi que
    no sirve para elegir un modelo mas sensible. Regla fijada de antemano: maximizar la
    exhaustividad sobre los cantos atenuados de las sesiones de validacion interna, sujeto a que
    la mAP enmascarada no baje mas de ``MAP_TOLERANCE`` respecto del punto de partida (BirdNET).
    Todo se calcula con sesiones de entrenamiento: el pliegue de prueba no interviene.
    """

    def __init__(self, data: D.DeepData, val_rows: np.ndarray, mix: Mixtures, eval_rows: np.ndarray | None = None, weights: np.ndarray | None = None):
        self.data, self.val_rows = data, val_rows
        self.rows = val_rows if eval_rows is None else eval_rows          # filas donde se predice (puede ser un subconjunto)
        self.bench = SNRBenchmark(data, mix.plan, min_pos=5, rows=self.rows)
        self.mix_rows = self.bench.mix_rows
        Y, unc = data.Y[self.rows], data.uncertain[self.rows]
        w = (data.weight[self.rows] if weights is None else weights).astype(np.float64)
        usable = [k for k in range(Y.shape[1]) if Y[:, k].sum() > 0]
        self._map = lambda prob: float(np.nanmean([M.weighted_ap(Y[~unc[:, k], k], prob[~unc[:, k], k].astype(np.float64), w[~unc[:, k]]) for k in usable]))
        self.map0: float | None = None
        self.last: dict = {}

    def __call__(self, prob: np.ndarray, prob_mix: np.ndarray) -> float:
        m = self._map(prob)
        clean = np.zeros_like(self.data.Y)
        clean[self.rows] = prob
        mixed = np.zeros((len(self.bench.src), self.data.Y.shape[1]), np.float32)
        mixed[self.mix_rows] = prob_mix
        r = self.bench.mean_recall(clean, mixed)
        if self.map0 is None:
            self.map0 = m                                            # primera llamada: punto de partida
        self.last = {"map": m, "recall": r}
        return r if m >= self.map0 - MAP_TOLERANCE else m - 10.0


def fit_predict(name: str, data: D.DeepData, fit_rows: np.ndarray, val_rows: np.ndarray, test_rows: np.ndarray, mix: Mixtures | None, dense=None):
    """Devuelve (probabilidades de prueba, probabilidades de las mezclas de prueba o None, informacion)."""
    if name == "birdnet":
        out_mix = None
        if mix is not None:
            out_mix = (mix.test_rows(test_rows), flat_sigmoid(mix.logit[mix.test_rows(test_rows)]))
        return flat_sigmoid(data.logit[test_rows]), out_mix, {}
    if name in LEGACY:
        from .. import heads as V1

        Xf, Yf = data.emb[fit_rows], data.Y[fit_rows]
        if name == "prototipos":
            both = np.r_[fit_rows, val_rows]
            head = V1.PrototypeHead().fit(data.emb[both], data.Y[both])
            return head.decision_function(data.emb[test_rows]), None, {}
        if name == "lineal_v1":
            head = V1.MLPHead(V1.HeadConfig(hidden=0, dropout=0.0, lr=1e-3, weight_decay=1e-3, epochs=200, patience=25, l2_input=True))
        else:
            head = V1.BirdNETFineTuneHead(dense[0], dense[1], V1.FTConfig(l2sp=0.3, lr=3e-4))
        head.fit(Xf, Yf, X_val=data.emb[val_rows], Y_val=data.Y[val_rows])
        return head.predict_proba(data.emb[test_rows]), None, {"best_epoch": int(getattr(head, "best_epoch", -1))}

    train, val = subset(data, fit_rows), subset(data, val_rows)
    best = None
    for i, cfg in enumerate(METHODS[name]):
        use_mix = mix is not None and cfg.mixture_weight > 0
        extra = mix.training_rows(fit_rows) if use_mix else None
        if use_mix and cfg.kind == "residual":
            metric = RobustSelector(data, val_rows, mix)
            vdict = dict(val, mix_emb=mix.emb[metric.mix_rows], mix_logit=mix.logit[metric.mix_rows])
        else:
            metric, vdict = val_metric_fn(data, val_rows), val
        res = H.fit(cfg, train, vdict, metric, extra=extra)
        score = max(h["val"] for h in res.history if h["epoch"] == res.best_epoch)
        if best is None or score > best[0]:
            best = (score, i, res)
    score, i, res = best
    te = subset(data, test_rows)
    prob = H.predict(res.model, te["emb"], te["logit"], te["rec"], te["pos"])
    out_mix = None
    if mix is not None and METHODS[name][i].kind != "context":
        mrows = mix.test_rows(test_rows)
        out_mix = (mrows, H.predict(res.model, mix.emb[mrows], mix.logit[mrows]))
    return prob, out_mix, {"best_epoch": int(res.best_epoch), "config": i, "val": float(score), "val_start": float(res.history[0]["val"])}


def cross_validate(data: D.DeepData, methods: list[str], mix: Mixtures | None, folds: np.ndarray, verbose: bool = True):
    dense = birdnet_dense(data) if "ft_v1" in methods else None
    oof = {m: np.zeros_like(data.Y) for m in methods}
    oof_mix = {m: np.full((len(mix.plan), data.Y.shape[1]), np.nan, np.float32) for m in methods} if mix is not None else {}
    info: dict[str, list] = {m: [] for m in methods}
    for f in sorted(np.unique(folds)):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        is_val = D.inner_split(data.groups[tr])
        fit_rows, val_rows = tr[~is_val], tr[is_val]
        for m in methods:
            t0 = time.time()
            prob, pm, inf = fit_predict(m, data, fit_rows, val_rows, te, mix, dense)
            oof[m][te] = prob
            if pm is not None:
                oof_mix[m][pm[0]] = pm[1]
            info[m].append(inf)
            if verbose:
                print(f"  pliegue {int(f) + 1} {m:18s} {time.time() - t0:6.1f} s {inf}", flush=True)
    return oof, oof_mix, info


def evaluate(data: D.DeepData, oof: dict[str, np.ndarray], oof_mix: dict, mix: Mixtures | None, reps: int = 500, tag: str = "heads") -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tables, summaries = [], []
    for regime in M.REGIMES:
        t = M.metrics_table(data, oof, regime)
        tables.append(t)
        s = M.summarize(t, LABELS)
        s.insert(1, "regime", regime)
        summaries.append(s)
    pd.concat(tables).to_csv(RESULTS / f"cv_metrics_{tag}.csv", index=False)
    summ = pd.concat(summaries)
    summ.to_csv(RESULTS / f"cv_summary_{tag}.csv", index=False)
    print(summ.pivot(index="method", columns="regime", values="macro_ap").reindex(list(oof)).round(4).to_string(), flush=True)

    boots = []
    for regime in ("estricto", "enmascarado"):
        sb = M.SessionBootstrap(data, regime=regime, reps=reps)
        for m in oof:
            boots.append(sb.interval(m, oof[m]))
            if m != "birdnet" and "birdnet" in oof:
                boots.append(sb.compare(m, oof[m], "birdnet", oof["birdnet"]))
        for a, b in (("lineal", "lineal_v1"), ("lineal", "prototipos"), ("residual", "residual_estricto"), ("contexto", "residual"), ("residual_mix", "residual")):
            if a in oof and b in oof:
                boots.append(sb.compare(a, oof[a], b, oof[b]))
    (RESULTS / f"bootstrap_{tag}.json").write_text(json.dumps(boots, indent=2), encoding="utf-8")

    rev = M.reviewed_only(data, oof)
    rev.to_csv(RESULTS / f"reviewed_only_{tag}.csv", index=False)
    print(rev[["method", "n_rejected", "auc_pooled", "auc_pooled_lo", "auc_pooled_hi", "concordance"]].round(3).to_string(index=False), flush=True)

    out = {"summary": summ, "bootstrap": boots, "reviewed": rev}
    if mix is not None:
        usable = {m: (oof[m], oof_mix[m]) for m in oof if m in oof_mix and not np.isnan(oof_mix[m][mix.test_rows(np.arange(len(data.Y)))]).any()}
        if usable:
            bench = SNRBenchmark(data, mix.plan)
            snr = bench.summary(usable)
            snr.to_csv(RESULTS / f"snr_{tag}.csv", index=False)
            pd.concat([bench.species_table(bench.prepare(c, x)).assign(method=m) for m, (c, x) in usable.items()]).to_csv(RESULTS / f"snr_species_{tag}.csv", index=False)
            print(snr.pivot(index="method", columns="atten_db", values="recall").round(3).to_string(), flush=True)
            out["snr"] = snr
    return out


def leave_one_device_out(data: D.DeepData, methods: list[str], mix: Mixtures | None) -> pd.DataFrame:
    dev = data.meta["device"].to_numpy()
    dense = birdnet_dense(data) if "ft_v1" in methods else None
    rows = []
    for d in sorted(np.unique(dev)):
        te, tr = np.flatnonzero(dev == d), np.flatnonzero(dev != d)
        is_val = D.inner_split(data.groups[tr])
        fit_rows, val_rows = tr[~is_val], tr[is_val]
        for m in methods:
            prob, _, _ = fit_predict(m, data, fit_rows, val_rows, te, None, dense)
            full = np.zeros_like(data.Y)
            full[te] = prob
            for regime in ("estricto", "enmascarado"):
                t = M.metrics_table(data, {m: full}, regime, rows=te)
                ok = t["n_pos"] >= 3
                rows.append({"device": d, "method": m, "regime": regime, "macro_ap": float(t.loc[ok, "ap"].mean()), "species_evaluated": int(ok.sum()), "windows": int(len(te))})
        print(f"  LODO {d} listo", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default=",".join(DEFAULT))
    parser.add_argument("--mixtures", action="store_true", help="Incluye los metodos con mezclas y la prueba de cantos atenuados.")
    parser.add_argument("--lodo", action="store_true")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--tag", default="heads")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    data = D.load(maps=False)
    methods = [m for m in args.methods.split(",") if m]
    mix = None
    if args.mixtures:
        mix = Mixtures(data)
        methods += [m for m in MIX_METHODS if m not in methods]
    folds = D.group_folds(data.groups)
    print(f"{len(data.Y)} ventanas, {data.meta['session'].nunique()} sesiones, {int(data.Y.sum())} pares positivos; metodos: {methods}", flush=True)
    oof, oof_mix, info = cross_validate(data, methods, mix, folds)
    # Conjunto sin entrenamiento adicional: BirdNET propone y la sonda local verifica (media geometrica).
    for probe in ("lineal", "lineal_mix"):
        if "birdnet" in oof and probe in oof:
            name = f"birdnet_x_{probe}"
            oof[name] = np.sqrt(oof["birdnet"] * oof[probe])
            if probe in oof_mix and "birdnet" in oof_mix:
                oof_mix[name] = np.sqrt(oof_mix["birdnet"] * oof_mix[probe])
    np.savez_compressed(D.DEEP_DIR / f"oof_{args.tag}.npz", folds=folds, **{f"oof_{m}": v for m, v in oof.items()}, **{f"mix_{m}": v for m, v in oof_mix.items()})
    evaluate(data, oof, oof_mix, mix, reps=args.reps, tag=args.tag)
    if args.lodo:
        lodo = leave_one_device_out(data, [m for m in methods if m in ("birdnet", "prototipos", "lineal", "residual", "contexto")], None)
        lodo.to_csv(RESULTS / f"lodo_{args.tag}.csv", index=False)
        print(lodo[lodo.regime == "estricto"].pivot(index="device", columns="method", values="macro_ap").round(3).to_string(), flush=True)
    run = {
        "methods": methods,
        "configs": {m: [asdict(c) for c in METHODS[m]] for m in methods if m in METHODS},
        "fold_info": info,
        "windows": int(len(data.Y)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (RESULTS / f"run_info_{args.tag}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"Listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
