"""Sondas entrenadas desde cero sobre el embedding de BirdNET y verificador de dos etapas.

A diferencia de los modelos de ``train_ra``, estas sondas no parten del logit de BirdNET: aprenden como suena
cada especie en estas grabaciones a partir de las detecciones validadas. Tratamiento de cada par ventana-especie:

    positivo            validado por un experto
    negativo explicito  rechazado por un experto (pocos y valiosos: peso alto)
    candidato           BirdNET >= 0,10 sin validar: lo mas probable es que nadie lo revisara; no entra en la perdida
    negativo            el resto

Como detector independiente una sonda no alcanza a BirdNET, pero ordena las propuestas de BirdNET mejor que el
propio BirdNET. De ahi el esquema de dos etapas que evalua ``evaluate_ra``: BirdNET propone y la sonda verifica.

Metodos:
    prototipos   prototipos coseno de la aplicacion (primera fase)
    lineal_v1    sonda lineal de la primera fase (NumPy; lo no validado es negativo)
    lineal       sonda lineal con el tratamiento de etiquetas de arriba (PyTorch)

Uso:
    python -m finetuning.deep.probes                 # predicciones fuera de muestra -> oof_scratch.npz
    python -m finetuning.deep.probes --export DIR    # ademas ajusta la sonda con toda la muestra y guarda el verificador
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config
from ..embeddings import flat_sigmoid
from . import data as D
from . import protocol as PR
from .metrics import weighted_ap

RESULTS = config.RESULTS_DIR / "deep"
OOF_NPZ = D.DEEP_DIR / "oof_scratch.npz"
VERIFIER_FILE = "verificador_local.npz"
METHODS = ("prototipos", "lineal_v1", "lineal")
RETENTION = 0.95          # el verificador marca como dudosa una deteccion por debajo del percentil 5 de las validadas
MIN_POSITIVES = 20        # con menos validadas el percentil no es estable: esa especie no se marca


@dataclass
class ProbeConfig:
    lr: float = 2e-3
    weight_decay: float = 1e-2
    epochs: int = 60
    batch_size: int = 512
    warmup: float = 0.05
    ema: float = 0.998
    pos_weight_power: float = 0.5
    pos_weight_cap: float = 30.0
    candidates: str = "ignore"        # ignore | negative
    candidate_threshold: float = 0.10
    hard_negative_weight: float = 20.0
    use_sampling_weights: bool = True
    patience: int = 60
    seed: int = 0
    l2_input: bool = False            # normaliza el embedding a norma 1 antes de estandarizar


# Dos candidatas fijadas de antemano; cada pliegue elige con sus sesiones de validacion interna.
CONFIGS = [ProbeConfig(l2_input=False), ProbeConfig(l2_input=True)]


# --------------------------------------------------------------------------- #
# Modelo y perdida
# --------------------------------------------------------------------------- #
class LinearProbe(nn.Module):
    def __init__(self, n_species: int, mean: np.ndarray, std: np.ndarray, l2_input: bool):
        super().__init__()
        self.l2_input = l2_input
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))
        self.net = nn.Linear(1024, n_species)

    def init_bias(self, prior: np.ndarray) -> None:
        """Sesgo de salida = logit de la prevalencia: evita gastar el entrenamiento en aprender que casi todo es negativo."""
        p = np.clip(prior, 1e-5, 1 - 1e-5)
        with torch.no_grad():
            self.net.bias.copy_(torch.from_numpy(np.log(p / (1 - p)).astype(np.float32)))
            self.net.weight.mul_(0.1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if self.l2_input:
            emb = F.normalize(emb, dim=-1)
        return self.net((emb - self.mean) / self.std)

    def raw_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """(W, b) tales que logit = x @ W + b, con x el embedding (ya normalizado si ``l2_input``)."""
        w = self.net.weight.detach().numpy().astype(np.float64)        # (S, 1024)
        mean, std = self.mean.numpy().astype(np.float64), self.std.numpy().astype(np.float64)
        W = (w / std).T
        b = self.net.bias.detach().numpy().astype(np.float64) - (mean / std) @ w.T
        return W.astype(np.float32), b.astype(np.float32)


def candidate_mask(Y: np.ndarray, logit: np.ndarray, hard_neg: np.ndarray, threshold: float) -> np.ndarray:
    """Pares que BirdNET propone (probabilidad >= umbral) pero que nadie valido ni rechazo."""
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    return (prob >= threshold) & (Y == 0) & (~hard_neg)


def pair_weights(Y: np.ndarray, logit: np.ndarray, hard_neg: np.ndarray, sample_weight: np.ndarray, cfg: ProbeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Peso de cada par (ventana, especie) en la perdida y peso por clase de los positivos."""
    w = np.repeat((sample_weight / float(sample_weight.mean()))[:, None], Y.shape[1], axis=1).astype(np.float32) if cfg.use_sampling_weights else np.ones_like(Y, dtype=np.float32)
    if cfg.candidates == "ignore":
        w[candidate_mask(Y, logit, hard_neg, cfg.candidate_threshold) & (Y == 0)] = 0.0
    pos = (w * (Y > 0)).sum(axis=0)
    neg = (w * (Y == 0)).sum(axis=0)
    pos_weight = np.clip((neg / np.maximum(pos, 1e-6)) ** cfg.pos_weight_power, 1.0, cfg.pos_weight_cap).astype(np.float32)
    w = w * np.where(Y > 0, pos_weight[None, :], 1.0)
    w[hard_neg] *= cfg.hard_negative_weight
    return w.astype(np.float32), pos_weight


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def state(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def predict(model: nn.Module, emb: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Probabilidades (N, S)."""
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(emb), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(np.asarray(emb[a : a + batch], dtype=np.float32)))).numpy())
    return np.concatenate(out)


class FitResult:
    def __init__(self, model: nn.Module, best_epoch: int, history: list[dict]):
        self.model, self.best_epoch, self.history = model, best_epoch, history


def fit(cfg: ProbeConfig, train: dict[str, np.ndarray], val: dict[str, np.ndarray] | None = None, val_metric=None) -> FitResult:
    """Entrena la sonda lineal. ``train`` / ``val``: emb, logit, Y, hard_neg, weight.

    ``val_metric(prob) -> float`` elige la epoca (mayor es mejor); sin validacion se queda con la ultima.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    base = train["emb"]
    if cfg.l2_input:
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    model = LinearProbe(train["Y"].shape[1], base.mean(axis=0), base.std(axis=0) + 1e-3, cfg.l2_input)
    W, _ = pair_weights(train["Y"], train["logit"], train["hard_neg"], train["weight"], cfg)
    sw = (train["weight"] / train["weight"].mean())[:, None] if cfg.use_sampling_weights else np.ones((len(train["Y"]), 1), np.float32)
    model.init_bias((sw * train["Y"]).sum(axis=0) / sw.sum())
    emb_t, Y_t, W_t = (torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in (train["emb"], train["Y"], W))
    n_items, bs = len(emb_t), cfg.batch_size
    steps_per_epoch = max(1, math.ceil(n_items / bs))
    total = steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total, int(cfg.warmup * total)))
    ema = EMA(model, cfg.ema)

    def evaluate(state: dict) -> float:
        if val is None or val_metric is None:
            return float("nan")
        probe = copy.deepcopy(model)
        probe.load_state_dict(state)
        return float(val_metric(predict(probe, val["emb"])))

    best_state, best_score, best_epoch = ema.state(), evaluate(ema.state()), 0
    history = [{"epoch": 0, "val": best_score}]
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = rng.permutation(n_items)
        run = 0.0
        for a in range(0, n_items, bs):
            bt = torch.from_numpy(perm[a : a + bs])
            loss = F.binary_cross_entropy_with_logits(model(emb_t[bt]), Y_t[bt], reduction="none")
            loss = (loss * W_t[bt]).sum() / W_t[bt].sum().clamp(min=1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update(model, step)
            step += 1
            run += float(loss.detach())
        score = evaluate(ema.state())
        history.append({"epoch": epoch, "loss": run / steps_per_epoch, "val": score})
        if val is None or val_metric is None or score > best_score + 1e-5:
            best_state, best_score, best_epoch = ema.state(), score, epoch
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best_epoch, history)


# --------------------------------------------------------------------------- #
# Validacion cruzada
# --------------------------------------------------------------------------- #
def subset(data: D.DeepData, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {"emb": data.emb[rows], "logit": data.logit[rows], "Y": data.Y[rows], "hard_neg": data.hard_neg[rows], "weight": data.weight[rows]}


def val_metric_fn(data: D.DeepData, rows: np.ndarray):
    """mAP macro ponderada sobre las sesiones de validacion interna, sin los pares inciertos (BirdNET >= 0,5 sin validar)."""
    Y, w, unc = data.Y[rows], data.weight[rows].astype(np.float64), data.uncertain[rows]
    usable = [k for k in range(Y.shape[1]) if Y[:, k].sum() > 0]

    def metric(prob: np.ndarray) -> float:
        aps = [weighted_ap(Y[~unc[:, k], k], prob[~unc[:, k], k].astype(np.float64), w[~unc[:, k]]) for k in usable]
        return float(np.nanmean(aps)) if aps else float("nan")

    return metric


def fit_linear(data: D.DeepData, fit_rows: np.ndarray, val_rows: np.ndarray) -> tuple[LinearProbe, dict]:
    """Entrena las dos candidatas y devuelve la mejor en la validacion interna."""
    train, val, metric = subset(data, fit_rows), subset(data, val_rows), val_metric_fn(data, val_rows)
    best = None
    for i, cfg in enumerate(CONFIGS):
        res = fit(cfg, train, val, metric)
        score = max(h["val"] for h in res.history if h["epoch"] == res.best_epoch)
        if best is None or score > best[0]:
            best = (score, i, res)
    score, i, res = best
    return res.model, {"best_epoch": int(res.best_epoch), "config": i, "l2_input": bool(CONFIGS[i].l2_input), "val": float(score), "val_start": float(res.history[0]["val"])}


def fit_predict(name: str, data: D.DeepData, fit_rows: np.ndarray, val_rows: np.ndarray, test_rows: np.ndarray) -> tuple[np.ndarray, dict]:
    if name == "lineal":
        model, info = fit_linear(data, fit_rows, val_rows)
        return predict(model, data.emb[test_rows]), info
    from .. import heads as V1                                    # metodos de la primera fase

    if name == "prototipos":
        both = np.r_[fit_rows, val_rows]
        head = V1.PrototypeHead().fit(data.emb[both], data.Y[both])
        return head.decision_function(data.emb[test_rows]), {}
    if name == "lineal_v1":
        head = V1.MLPHead(V1.HeadConfig(hidden=0, dropout=0.0, lr=1e-3, weight_decay=1e-3, epochs=200, patience=25, l2_input=True))
        head.fit(data.emb[fit_rows], data.Y[fit_rows], X_val=data.emb[val_rows], Y_val=data.Y[val_rows])
        return head.predict_proba(data.emb[test_rows]), {"best_epoch": int(getattr(head, "best_epoch", -1))}
    raise ValueError(name)


def cross_validate(data: D.DeepData, methods: list[str], folds: np.ndarray, verbose: bool = True) -> tuple[dict, dict]:
    oof = {m: np.zeros_like(data.Y) for m in methods}
    info: dict[str, list] = {m: [] for m in methods}
    for f in sorted(np.unique(folds)):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        is_val = D.inner_split(data.groups[tr])
        fit_rows, val_rows = tr[~is_val], tr[is_val]
        for m in methods:
            t0 = time.time()
            oof[m][te], inf = fit_predict(m, data, fit_rows, val_rows, te)
            info[m].append(inf)
            if verbose:
                print(f"  pliegue {int(f) + 1} {m:12s} {time.time() - t0:6.1f} s {inf}", flush=True)
    return oof, info


# --------------------------------------------------------------------------- #
# Verificador para la aplicacion
# --------------------------------------------------------------------------- #
def verification_cutoffs(data: D.DeepData, cat: np.ndarray, oof: np.ndarray, retention: float = RETENTION, min_pos: int = MIN_POSITIVES) -> tuple[np.ndarray, np.ndarray]:
    """Por especie, el score de la sonda (fuera de muestra) que conserva ``retention`` de las detecciones validadas."""
    w = data.weight.astype(np.float64)
    cut = np.zeros(cat.shape[1], np.float32)
    n_pos = np.zeros(cat.shape[1], np.int64)
    for k in range(cat.shape[1]):
        pos = np.flatnonzero(cat[:, k] == PR.P)
        n_pos[k] = len(pos)
        if len(pos) < min_pos:
            continue
        order = np.argsort(oof[pos, k])
        cw = np.cumsum(w[pos][order])
        cut[k] = oof[pos, k][order][min(int(np.searchsorted(cw, (1 - retention) * cw[-1])), len(pos) - 1)]
    return cut, n_pos


def flagged_summary(data: D.DeepData, cat: np.ndarray, oof: np.ndarray, cut: np.ndarray) -> dict:
    """Que marca el verificador como dudoso, medido con predicciones fuera de muestra."""
    w = data.weight.astype(np.float64)[:, None]
    low = oof < cut[None, :]
    out = {}
    for name, c in (("validadas", PR.P), ("rechazadas", PR.H), ("sin_revisar", PR.U)):
        sel = cat == c
        out[name] = {"pares": int(sel.sum()), "marcadas": int((sel & low).sum()), "fraccion_ponderada": float((w * (sel & low)).sum() / max((w * sel).sum(), 1e-12))}
    return out


def export_verifier(data: D.DeepData, cat: np.ndarray, oof: np.ndarray, out_dir: Path) -> dict:
    """Ajusta la sonda con todas las sesiones y guarda pesos, cortes por especie y metadatos en un solo archivo.

    Los pesos son los de la sonda (1 024 x especies), no los de BirdNET. El archivo no se versiona.
    """
    is_val = D.inner_split(data.groups)
    rows = np.arange(len(data.Y))
    model, info = fit_linear(data, rows[~is_val], rows[is_val])
    W, b = model.raw_weights()
    cut, n_pos = verification_cutoffs(data, cat, oof)
    summary = flagged_summary(data, cat, oof, cut)
    # Los cortes salen de las predicciones fuera de muestra de la validacion cruzada, y el modelo que se distribuye es otro
    # (ajustado con todas las sesiones). Se comprueba cuanto marca ese modelo en sus sesiones de validacion interna, que no
    # ajustaron sus pesos (solo eligieron epoca y configuracion).
    held = np.zeros(cat.shape, bool)
    held[rows[is_val]] = True
    shipped = np.zeros(cat.shape, np.float32)
    shipped[rows[is_val]] = predict(model, data.emb[rows[is_val]])
    w = data.weight.astype(np.float64)[:, None]
    sel = held & (cat == PR.P) & (cut[None, :] > 0)
    low = sel & (shipped < cut[None, :])
    summary["modelo_exportado_en_validacion_interna"] = {"validadas": int(sel.sum()), "marcadas": int(low.sum()), "fraccion_ponderada": float((w * low).sum() / max((w * sel).sum(), 1e-12))}
    meta = {
        "name": "Verificador local CONDESAN (sonda lineal sobre el embedding de BirdNET V2.4)",
        "species": list(data.species),
        "l2_input": bool(model.l2_input),
        "retention": RETENTION,
        "min_positives": MIN_POSITIVES,
        "train_positives": [int(v) for v in n_pos],
        "training": info | {"recordings": int(data.info.get("recordings", 0)), "windows": int(len(data.Y))},
        "out_of_sample": summary,
        "license_note": "Sonda entrenada sobre embeddings de BirdNET V2.4 (CC BY-NC-SA 4.0): uso no comercial, atribucion y compartir igual.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / VERIFIER_FILE, W=W, b=b, cutoff=cut, meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--export", default=None, help="Carpeta donde guardar el verificador (p. ej. local_models/condesan).")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    data = D.load(maps=False)
    cat = PR.categories(data)
    folds = PR.load_folds(data, cat, D.DEEP_DIR / "folds.npy")
    methods = [m for m in args.methods.split(",") if m]
    print(f"{len(data.Y)} ventanas, {data.meta['session'].nunique()} sesiones; sondas: {methods}", flush=True)
    oof, info = cross_validate(data, methods, folds)
    store = dict(np.load(OOF_NPZ)) if OOF_NPZ.exists() else {}
    store |= {f"oof_{m}": v for m, v in oof.items()}
    np.savez_compressed(OOF_NPZ, **store)
    RESULTS.mkdir(parents=True, exist_ok=True)
    run = {"methods": methods, "configs": [asdict(c) for c in CONFIGS], "fold_info": info, "windows": int(len(data.Y)), "elapsed_s": round(time.time() - t0, 1)}
    if args.export:
        if "lineal" not in oof:
            raise SystemExit("El verificador necesita la sonda 'lineal'.")
        run["verifier"] = export_verifier(data, cat, oof["lineal"], Path(args.export))
        print("Verificador:", json.dumps(run["verifier"]["out_of_sample"], ensure_ascii=False), flush=True)
    (RESULTS / "run_info_probes.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
