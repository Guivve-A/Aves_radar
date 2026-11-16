"""Ajuste fino de las capas convolucionales profundas de BirdNET V2.4.

Las capas inferiores quedan congeladas y sus activaciones estan guardadas en disco
(``f3``: entrada del bloque 4; ``f2``: entrada del bloque 3). Se entrena la replica en
PyTorch (``backbone.BirdNETTop``) desde esas activaciones hasta los logits de las
especies objetivo. El modelo arranca siendo exactamente BirdNET.

Regularizacion y aumento de datos:
    L2-SP           penaliza ||w - w_BirdNET||^2 (Li et al., 2018): el punto de partida es el ancla
    ancla funcional penaliza alejarse del logit de BirdNET en los pares no positivos
    tasas de aprendizaje discriminativas por bloque, calentamiento + coseno, media movil (EMA)
    profundidad estocastica y dropout antes de la capa de clasificacion
    desplazamiento temporal circular de las activaciones
    mezclas atenuadas (positivo atenuado + fondo de la misma sesion), calculadas sobre el audio

Las etiquetas se tratan como en ``heads``: positivos validados, rechazos expertos con peso
alto y candidatos de BirdNET sin revisar fuera de la perdida.

Uso:
    python -m finetuning.deep.finetune                 # bloque 4 + POST + clasificador (corte f3)
    python -m finetuning.deep.finetune --cut f2        # tambien el bloque 3
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..embeddings import flat_sigmoid
from . import data as D
from . import heads as H
from . import metrics as M
from . import mixtures as MX
from .backbone import BirdNETTop
from .train_heads import LABELS, RESULTS, Mixtures, evaluate, val_metric_fn


@dataclass
class FTConfig:
    cut: str = "f3"
    train_groups: tuple[str, ...] = ("block4", "post", "classifier")
    lr: float = 1e-4
    lr_mult: dict = field(default_factory=lambda: {"block3": 0.3, "block4": 0.5, "post": 1.0, "classifier": 3.0})
    weight_decay: float = 0.0
    l2sp: float = 1e-2
    l2sp_mult: dict = field(default_factory=lambda: {"block3": 3.0, "block4": 1.0, "post": 1.0, "classifier": 0.3})
    anchor: float = 0.05
    epochs: int = 6
    batch_size: int = 64
    background_per_epoch: int = 4000
    mixtures_per_epoch: int = 6000
    mixture_weight: float = 0.5
    drop_path: float = 0.1
    head_dropout: float = 0.2
    time_roll: int = 2
    ema: float = 0.998
    warmup: float = 0.06
    patience: int = 3
    val_background: int = 3000
    seed: int = 0
    loss: H.HeadConfig = field(default_factory=lambda: H.HeadConfig(candidates="ignore", candidate_threshold=0.10, hard_negative_weight=20.0))


def _as_tensor(feat, rows: np.ndarray) -> torch.Tensor:
    rows = np.asarray(rows)
    order = np.argsort(rows)
    block = np.asarray(feat[rows[order]], dtype=np.float32)
    out = np.empty_like(block)
    out[order] = block
    return torch.from_numpy(out)


def predict(model: BirdNETTop, feat, rows: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(rows), batch):
            out.append(torch.sigmoid(model(_as_tensor(feat, rows[a : a + batch]))).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(model.class_indices)), np.float32)


def fit(cfg: FTConfig, data: D.DeepData, fit_rows: np.ndarray, val_rows: np.ndarray, mix: Mixtures | None, mix_feat=None, verbose: bool = True):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    feat = data.f3 if cfg.cut == "f3" else data.f2
    model = BirdNETTop(cfg.cut, class_indices=data.class_index)
    model.drop_path, model.head_dropout = cfg.drop_path, cfg.head_dropout
    model.set_anchor()
    groups = model.groups()
    param_groups = []
    for g, keys in groups.items():
        train_it = g in cfg.train_groups
        for k in keys:
            model.params[k].requires_grad_(train_it)
        if train_it and keys:
            param_groups.append({"params": [model.params[k] for k in keys], "lr": cfg.lr * cfg.lr_mult.get(g, 1.0)})
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Pesos por par (ventana, especie) sobre las filas de entrenamiento.
    W, pos_weight = H.pair_weights(data.Y[fit_rows], data.logit[fit_rows], data.hard_neg[fit_rows], data.weight[fit_rows], cfg.loss)
    cand = H.candidate_mask(data.Y[fit_rows], data.logit[fit_rows], data.hard_neg[fit_rows], cfg.loss.candidate_threshold)
    priority = (data.Y[fit_rows].sum(axis=1) > 0) | data.hard_neg[fit_rows].any(axis=1) | cand.any(axis=1)
    pri_local, bg_local = np.flatnonzero(priority), np.flatnonzero(~priority)
    scale = float(data.weight[fit_rows].mean())
    if mix is not None and cfg.mixture_weight > 0:
        mrows = np.flatnonzero(np.isin(mix.kind, ("train", "bg_train")) & np.isin(mix.src, fit_rows))
        Wm, _ = H.pair_weights(mix.Y[mrows], mix.logit[mrows], np.zeros(mix.Y[mrows].shape, bool), mix.weight[mrows], cfg.loss, pos_weight, candidate=mix.candidate[mrows], weight_scale=scale)
    else:
        mrows, Wm = np.zeros(0, np.int64), None

    n_bg = min(cfg.background_per_epoch, len(bg_local))
    n_mix = min(cfg.mixtures_per_epoch, len(mrows))
    steps_per_epoch = int(np.ceil((len(pri_local) + n_bg + n_mix) / cfg.batch_size))
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: H.cosine_lr(s, total, int(cfg.warmup * total)))
    ema = H.EMA(model, cfg.ema)

    # Validacion interna: filas prioritarias + una muestra fija de fondo (con el peso reescalado).
    vY, vcand = data.Y[val_rows], H.candidate_mask(data.Y[val_rows], data.logit[val_rows], data.hard_neg[val_rows], cfg.loss.candidate_threshold)
    vpri = (vY.sum(axis=1) > 0) | data.hard_neg[val_rows].any(axis=1) | vcand.any(axis=1)
    vbg = np.flatnonzero(~vpri)
    vbg_take = rng.choice(vbg, size=min(cfg.val_background, len(vbg)), replace=False) if len(vbg) else vbg
    val_sel = np.sort(np.r_[np.flatnonzero(vpri), vbg_take])
    val_w = data.weight[val_rows].astype(np.float64).copy()
    if len(vbg_take):
        val_w[vbg_take] *= len(vbg) / len(vbg_take)
    sub_rows = val_rows[val_sel]
    vy, vw, vunc = data.Y[sub_rows], val_w[val_sel], data.uncertain[sub_rows]
    usable = [k for k in range(vy.shape[1]) if vy[:, k].sum() > 0]

    def val_score(state: dict) -> float:
        probe = copy.deepcopy(model)
        probe.load_state_dict(state)
        prob = predict(probe, feat, sub_rows)
        aps = [M.weighted_ap(vy[~vunc[:, k], k], prob[~vunc[:, k], k].astype(np.float64), vw[~vunc[:, k]]) for k in usable]
        return float(np.nanmean(aps))

    best_state, best_score, best_epoch = ema.state(), val_score(ema.state()), 0
    history = [{"epoch": 0, "val": best_score}]
    if verbose:
        print(f"    inicio (BirdNET) val {best_score:.4f}; {len(pri_local)} filas prioritarias + {n_bg} de fondo + {n_mix} mezclas por epoca", flush=True)
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        bg_take = rng.choice(bg_local, size=n_bg, replace=False) if n_bg else np.zeros(0, np.int64)
        mix_take = rng.choice(len(mrows), size=n_mix, replace=False) if n_mix else np.zeros(0, np.int64)
        # (origen, indice): 0 = ventana original, 1 = mezcla
        items = np.r_[np.c_[np.zeros(len(pri_local), np.int64), pri_local], np.c_[np.zeros(n_bg, np.int64), bg_take], np.c_[np.ones(n_mix, np.int64), mix_take]]
        bg_scale = len(bg_local) / max(n_bg, 1)
        mix_scale = cfg.mixture_weight * len(mrows) / max(n_mix, 1)
        is_bg = np.zeros(len(fit_rows), bool)
        is_bg[bg_take] = True
        rng.shuffle(items)
        run = 0.0
        for a in range(0, len(items), cfg.batch_size):
            it = items[a : a + cfg.batch_size]
            c, m = it[it[:, 0] == 0, 1], it[it[:, 0] == 1, 1]
            xs, ys, ws, bs = [], [], [], []
            if len(c):
                xs.append(_as_tensor(feat, fit_rows[c]))
                ys.append(data.Y[fit_rows[c]])
                ws.append(W[c] * np.where(is_bg[c], bg_scale, 1.0)[:, None])
                bs.append(data.logit[fit_rows[c]])
            if len(m):
                xs.append(_as_tensor(mix_feat, mrows[m]))
                ys.append(mix.Y[mrows[m]])
                ws.append(Wm[m] * mix_scale)
                bs.append(mix.logit[mrows[m]])
            x = torch.cat(xs)
            y, w, base = (torch.from_numpy(np.concatenate(v).astype(np.float32)) for v in (ys, ws, bs))
            if cfg.time_roll > 0:
                x = torch.roll(x, int(rng.integers(-cfg.time_roll, cfg.time_roll + 1)), dims=2)  # eje temporal (NHWC)
            z = model(x)
            loss = H.pair_loss(z, y, w, cfg.loss)
            if cfg.anchor > 0:
                loss = loss + cfg.anchor * H.anchor_loss(z, base, y)
            if cfg.l2sp > 0:
                loss = loss + 0.5 * cfg.l2sp * model.l2sp(cfg.l2sp_mult)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            sched.step()
            ema.update(model, step)
            step += 1
            run += float(loss.detach())
        score = val_score(ema.state())
        history.append({"epoch": epoch, "loss": run / steps_per_epoch, "val": score, "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"    epoca {epoch} perdida {run / steps_per_epoch:.4f} val {score:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if score > best_score + 1e-5:
            best_state, best_score, best_epoch = ema.state(), score, epoch
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        drift = {g: float(sum((model.params[k] - model._anchor[k]).pow(2).sum() for k in keys) ** 0.5 / max(sum(model._anchor[k].pow(2).sum() for k in keys) ** 0.5, 1e-12)) for g, keys in groups.items() if keys}
    return model, {"best_epoch": best_epoch, "val_start": history[0]["val"], "val_best": best_score, "history": history, "relative_drift": drift}


def cross_validate(cfg: FTConfig, data: D.DeepData, mix: Mixtures | None, name: str, folds: np.ndarray):
    feat = data.f3 if cfg.cut == "f3" else data.f2
    mix_feat = None
    if mix is not None:
        feats = MX.load_features(maps=True)
        mix_feat = feats["f3"] if cfg.cut == "f3" else feats.get("f2")
        if mix_feat is None:
            raise RuntimeError("Las mezclas no tienen el mapa f2 (extrae con --f2).")
    oof = np.zeros_like(data.Y)
    oof_mix = np.full((len(mix.plan), data.Y.shape[1]), np.nan, np.float32) if mix is not None else None
    infos = []
    for f in sorted(np.unique(folds)):
        t0 = time.time()
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        is_val = D.inner_split(data.groups[tr])
        fit_rows, val_rows = tr[~is_val], tr[is_val]
        print(f"  pliegue {int(f) + 1}: {len(fit_rows)} ventanas de ajuste, {len(val_rows)} de validacion, {len(te)} de prueba", flush=True)
        model, info = fit(cfg, data, fit_rows, val_rows, mix, mix_feat)
        oof[te] = predict(model, feat, te)
        if mix is not None:
            mrows = mix.test_rows(te)
            oof_mix[mrows] = predict(model, mix_feat, mrows)
        info["seconds"] = round(time.time() - t0, 1)
        infos.append(info)
        print(f"  pliegue {int(f) + 1} {name}: mejor epoca {info['best_epoch']}, val {info['val_start']:.4f} -> {info['val_best']:.4f}, deriva {info['relative_drift']} ({info['seconds']:.0f} s)", flush=True)
        np.savez_compressed(D.DEEP_DIR / f"oof_{name}_partial.npz", oof=oof, done_folds=np.array(sorted(np.unique(folds))[: len(infos)]))
    return oof, oof_mix, infos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cut", default="f3", choices=("f3", "f2"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2sp", type=float, default=1e-2)
    parser.add_argument("--no-mixtures", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reps", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    data = D.load(maps=True)
    cfg = FTConfig(cut=args.cut, epochs=args.epochs, lr=args.lr, l2sp=args.l2sp)
    if args.cut == "f2":
        cfg.train_groups = ("block3", "block4", "post", "classifier")
    if args.no_mixtures:
        cfg.mixture_weight = 0.0
    name = args.name or ("ft_bloque4" if args.cut == "f3" else "ft_bloques34") + ("" if not args.no_mixtures else "_sin_mezclas")
    LABELS.setdefault(name, {"f3": "BirdNET con bloque 4 afinado", "f2": "BirdNET con bloques 3 y 4 afinados"}[args.cut] + (" (sin mezclas)" if args.no_mixtures else ""))
    mix = None if args.no_mixtures else Mixtures(data)
    folds = D.group_folds(data.groups)
    print(f"Ajuste fino '{name}': corte {cfg.cut}, grupos {cfg.train_groups}, {len(data.Y)} ventanas", flush=True)
    oof, oof_mix, infos = cross_validate(cfg, data, mix, name, folds)
    np.savez_compressed(D.DEEP_DIR / f"oof_{name}.npz", folds=folds, oof=oof, **({"mix": oof_mix} if oof_mix is not None else {}))
    scores = {"birdnet": flat_sigmoid(data.logit), name: oof}
    scores_mix = {}
    if mix is not None:
        all_test = mix.test_rows(np.arange(len(data.Y)))
        bn_mix = np.full_like(oof_mix, np.nan)
        bn_mix[all_test] = flat_sigmoid(mix.logit[all_test])
        scores_mix = {"birdnet": bn_mix, name: oof_mix}
    evaluate(data, scores, scores_mix, mix, reps=args.reps, tag=name)
    cfg_dict = asdict(cfg)
    (RESULTS / f"run_info_{name}.json").write_text(json.dumps({"config": cfg_dict, "folds": infos, "elapsed_s": round(time.time() - t0, 1)}, indent=2, default=str), encoding="utf-8")
    print(f"Listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
