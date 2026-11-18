"""Entrenamiento "consciente de la revision": una sola perdida y un solo muestreador para toda la
escalera de modelos, desde una correccion lineal sobre el embedding hasta las capas convolucionales.

Todos los modelos son BirdNET mas una correccion que empieza en cero:

    lineal   logit = logit BirdNET + Lineal(embedding)                         (referencia)
    mlp      logit = logit BirdNET + MLP residual(embedding)                    (modelo principal)
    post     BN final + POST_CONV 3x3 + filas del clasificador, desde la salida del bloque 4
    b43      ademas los sub-bloques 4-3 y 4-4, desde la salida del sub-bloque 4-2
    b4       el bloque 4 completo, desde la salida del bloque 3

Perdida (ver ``protocol`` para las categorias de cada par ventana-especie):

    L = L_sup + lambda_E * L_facil
    L_sup    entropia cruzada sobre lo que un experto decidio: validadas (1) y rechazadas (0), mas las
             mezclas atenuadas de una validada (1 para su especie: el canto sigue ahi, mas debil)
    L_facil  destilacion hacia BirdNET, KL(Bern(s0) || Bern(sigma(z))), solo donde BirdNET calla (s0 < 0,05)
    Peso 0   propuestas sin revisar (U), zona gris (G) y pares con audio y etiqueta discordantes (X):
             no se sabe que son, y darles un objetivo distinto del de las validadas ensenaria a
             adivinar que se reviso.

Sin parada temprana: numero fijo de pasos, calentamiento + coseno y pesos EMA del ultimo paso. El unico
ajuste que se elige (cuanto puede alejarse de BirdNET: ``HEAD_GRID`` en la cabeza, ``DEEP_GRID`` en las capas
convolucionales) lo decide cada pliegue con un pliegue interno: la configuracion con mas exhaustividad sobre
cantos atenuados entre las que no pierden mAP limpia; el pliegue de prueba no interviene.

Uso:
    python -m finetuning.deep.train_ra --rungs lineal,mlp --ablations     # cabezas + ablaciones
    python -m finetuning.deep.train_ra --rungs post,b43                    # capas convolucionales
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config
from ..embeddings import flat_sigmoid
from . import data as D
from . import protocol as PR
from .backbone import CUT_TENSORS, BirdNETTop
from .mixtures import Mixtures

RESULTS = config.RESULTS_DIR / "deep"
CACHE_DIR = D.DEEP_DIR / "maps"
HEAD_RUNGS = ("lineal", "mlp")
DEEP_RUNGS = {"post": ("g44", ("post", "classifier")), "b43": ("g42", ("b43", "b44", "post", "classifier")), "b4": ("f3", ("b41", "b42", "b43", "b44", "post", "classifier"))}
LAYER_RATE = {"post": 1.0, "b44": 0.7, "b43": 0.49, "b42": 0.34, "b41": 0.24}
# Rejillas fijadas antes de ver ningun pliegue de prueba; cada pliegue elige con su pliegue interno.
# Cabezas: tasa de aprendizaje, decaimiento de la capa de salida (region de confianza) y peso de la destilacion.
HEAD_GRID = {
    "libre": {"lr": 1e-3, "head_decay": 0.05, "lam_easy": 3.0},
    "d10": {"lr": 2e-4, "head_decay": 10.0, "lam_easy": 1.0},
    "d10_l3": {"lr": 2e-4, "head_decay": 10.0, "lam_easy": 3.0},
    "d30": {"lr": 2e-4, "head_decay": 30.0, "lam_easy": 1.0},
    "d100": {"lr": 2e-4, "head_decay": 100.0, "lam_easy": 1.0},
}
# Capas convolucionales: paso relativo y decaimiento desacoplado hacia los pesos de BirdNET.
# La rejilla inicial tenia las tres primeras. En la primera pasada de POST_CONV el pliegue interno eligio "agresivo"
# en 4 de 5 pliegues (el borde de la rejilla), asi que se amplio con dos configuraciones mas libres. La decision se tomo
# solo con los pliegues internos: las predicciones de esa pasada sobre los pliegues de prueba se descartaron sin evaluarse.
DEEP_GRID = {
    "muy_prudente": {"rho": 1e-4, "decay": 30.0},
    "prudente": {"rho": 3e-4, "decay": 10.0},
    "agresivo": {"rho": 1e-3, "decay": 3.0},
    "muy_agresivo": {"rho": 3e-3, "decay": 1.0},
    "libre": {"rho": 1e-2, "decay": 0.3},
}
DEEP_STEPS = 500                                 # los mismos pasos en el pliegue interno y en el ajuste final
TRAIN_CAP_STRONG, TRAIN_CAP_WEAK = 12.0, 6.0     # atenuacion maxima de una mezcla usada como positivo al entrenar


@dataclass
class RAConfig:
    rung: str = "mlp"
    steps: int = 1500
    batch: tuple = (64, 4, 36, 8, 80)      # validadas, rechazos, mezclas positivas, mezclas de fondo, ventanas por diseno
    lr: float = 1e-3
    weight_decay: float = 0.05
    head_decay: float = 0.05               # decaimiento desacoplado de la capa de salida de la cabeza: region de confianza en torno a BirdNET
    warmup: int = 100
    ema: float = 0.998
    lam_easy: float = 1.0
    labels: str = "review"                 # review | strict (ablacion: lo no validado es negativo)
    crossfades: bool = True                # False = ablacion sin mezclas
    mix_negatives: bool = False            # True = variante: falsas alarmas certificadas de las mezclas como negativos
    rho: float = 3e-4                      # paso relativo de las capas convolucionales
    decay: float = 10.0                    # decaimiento desacoplado hacia BirdNET (capas convolucionales)
    seed: int = 42


# --------------------------------------------------------------------------- #
# Modelos de cabeza
# --------------------------------------------------------------------------- #
class DeltaHead(nn.Module):
    """logit = logit de BirdNET + correccion(embedding). La ultima capa empieza en cero."""

    def __init__(self, kind: str, n_species: int, mean: np.ndarray, std: np.ndarray, width: int = 256, dropout: float = 0.2):
        super().__init__()
        self.kind = kind
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))
        if kind == "mlp":
            self.inp = nn.Sequential(nn.Linear(1024, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))
            self.block = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))
            self.out = nn.Linear(width, n_species)
        else:
            self.out = nn.Linear(1024, n_species)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, emb: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        x = ((emb - self.mean) / self.std).clamp(-8.0, 8.0)
        if self.kind == "mlp":
            h = self.inp(x)
            x = h + self.block(h)
        return z0.clamp(-20.0, 20.0) + self.out(x)


# --------------------------------------------------------------------------- #
# Objetivos, muestreador y perdida
# --------------------------------------------------------------------------- #
class Targets:
    """Objetivo y mascaras por par, para ventanas reales y para mezclas."""

    def __init__(self, data: D.DeepData, cat: np.ndarray, mix: Mixtures | None, labels: str = "review", mix_negatives: bool = False):
        s0 = flat_sigmoid(data.logit).astype(np.float32)
        self.T = np.where(cat == PR.P, 1.0, np.where(cat == PR.H, 0.0, s0)).astype(np.float32)
        self.sup = (cat == PR.P) | (cat == PR.H)
        self.easy = cat == PR.E
        if labels == "strict":                       # ablacion: todo lo no validado es negativo
            hard0 = np.isin(cat, (PR.U, PR.R0, PR.G))
            self.T[hard0] = 0.0
            self.sup = self.sup | hard0
        if mix is not None:
            sm = flat_sigmoid(mix.logit).astype(np.float32)
            src, bg = mix.src, mix.plan["bg"].to_numpy()
            att = mix.plan["atten_db"].to_numpy()[:, None]
            is_pos = np.isin(mix.kind, ("train", "test"))[:, None]
            cap = np.where(s0[src] >= 0.5, TRAIN_CAP_STRONG, TRAIN_CAP_WEAK)
            self.sup_mix = is_pos & (cat[src] == PR.P) & (att <= cap)
            self.easy_mix = (sm < PR.SILENT) & (cat[src] == PR.E) & (cat[bg] == PR.E)
            self.T_mix = np.where(self.sup_mix, 1.0, sm).astype(np.float32)
            # La especie calla en los dos ingredientes y aun asi BirdNET la puntua >= 0,05 en la mezcla: sumar dos audios
            # no crea un canto, asi que es una falsa alarma tan fiable como un negativo facil. Son los unicos negativos
            # que existen en la zona gris. Solo se usan en la variante ``mix_negatives``.
            self.neg_mix = (sm >= PR.SILENT) & (cat[src] == PR.E) & (cat[bg] == PR.E)
            if mix_negatives:
                self.T_mix[self.neg_mix] = 0.0
                self.sup_all_mix = self.sup_mix | self.neg_mix
            else:
                self.sup_all_mix = self.sup_mix


class Sampler:
    """Composicion fija del lote: validadas (equilibradas por especie), rechazos, mezclas y ventanas por diseno."""

    def __init__(self, data: D.DeepData, cat: np.ndarray, targets: Targets, mix: Mixtures | None, fit_rows: np.ndarray, cfg: RAConfig):
        self.rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        inside = np.zeros(len(data.Y), bool)
        inside[fit_rows] = True
        self.fit_rows = fit_rows
        self.by_species = [np.flatnonzero((cat[:, k] == PR.P) & inside) for k in range(cat.shape[1])]
        n = np.array([len(v) for v in self.by_species], dtype=np.float64)
        self.p_species = np.sqrt(n) / np.sqrt(n).sum()
        self.rej = np.flatnonzero((cat == PR.H).any(axis=1) & inside)
        w = data.weight[fit_rows].astype(np.float64)
        self.p_design = w / w.sum()
        self.use_mix = mix is not None and cfg.crossfades
        if self.use_mix:
            ok = (mix.kind == "train") & inside[mix.src]
            self.xpos = [np.flatnonzero(ok & targets.sup_mix[:, k]) for k in range(cat.shape[1])]
            m = np.array([len(v) for v in self.xpos], dtype=np.float64)
            self.p_xpos = np.sqrt(m) / max(np.sqrt(m).sum(), 1e-12)
            self.xneg = np.flatnonzero((mix.kind == "bg_train") & inside[mix.src])
            self.xcert = np.flatnonzero(np.isin(mix.kind, ("train", "bg_train")) & inside[mix.src] & targets.neg_mix.any(axis=1)) if cfg.mix_negatives else np.zeros(0, np.int64)

    def batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Indices de ventanas reales y de mezclas del lote."""
        n_lab, n_rej, n_xp, n_xn, n_des = self.cfg.batch
        rng = self.rng
        if not self.use_mix:
            n_des, n_xp, n_xn = n_des + n_xp + n_xn, 0, 0
        ks = rng.choice(len(self.by_species), size=n_lab, p=self.p_species)
        real = [rng.choice(self.by_species[k]) for k in ks]
        if len(self.rej):
            real += list(rng.choice(self.rej, size=n_rej))
        real += list(self.fit_rows[rng.choice(len(self.fit_rows), size=n_des, p=self.p_design)])
        mixes: list[int] = []
        if n_xp and self.p_xpos.sum() > 0:
            mixes += [int(rng.choice(self.xpos[k])) for k in rng.choice(len(self.xpos), size=n_xp, p=self.p_xpos)]
        if n_xn and len(self.xneg):
            mixes += list(rng.choice(self.xneg, size=n_xn))
        if n_xn and len(self.xcert):                 # variante: tantas mezclas con una falsa alarma certificada como mezclas de fondo
            mixes += list(rng.choice(self.xcert, size=n_xn))
        return np.asarray(real, dtype=np.int64), np.asarray(mixes, dtype=np.int64)


def ra_loss(z: torch.Tensor, T: torch.Tensor, sup: torch.Tensor, easy: torch.Tensor, lam: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(z, T, reduction="none")
    l_sup = (bce * sup).sum() / sup.sum().clamp(min=1.0)
    Tc = T.clamp(1e-7, 1 - 1e-7)
    entropy = -(Tc * Tc.log() + (1 - Tc) * (1 - Tc).log())
    l_easy = ((bce - entropy) * easy).sum() / z.shape[0]          # KL(Bern(T) || Bern(sigma(z)))
    return l_sup + lam * l_easy


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)


def _rows(arr, idx: np.ndarray) -> np.ndarray:
    """Lectura por filas de un mapa de memoria (indices ordenados para leer en disco de forma secuencial)."""
    idx = np.asarray(idx)
    order = np.argsort(idx, kind="stable")
    block = np.asarray(arr[idx[order]], dtype=np.float32)
    out = np.empty_like(block)
    out[order] = block
    return out


# --------------------------------------------------------------------------- #
# Caracteristicas por escalon
# --------------------------------------------------------------------------- #
class Features:
    """Entradas de cada escalon para ventanas reales y mezclas."""

    def __init__(self, data: D.DeepData, mix: Mixtures | None, rung: str):
        self.rung = rung
        self.z0 = data.logit
        self.z0_mix = mix.logit if mix is not None else None
        if rung in HEAD_RUNGS:
            self.x, self.x_mix = data.emb, (mix.emb if mix is not None else None)
        else:
            cut = DEEP_RUNGS[rung][0]
            if cut == "f3":
                from . import mixtures as MX

                self.x = data.f3
                self.x_mix = MX.load_features(maps=True)["f3"] if mix is not None else None
            else:
                self.x = np.load(CACHE_DIR / f"{cut}.npy", mmap_mode="r")
                self.x_mix = np.load(CACHE_DIR / f"{cut}_mix.npy", mmap_mode="r") if mix is not None else None

    def real(self, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(_rows(self.x, rows))

    def mixed(self, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(_rows(self.x_mix, rows))


def build_map_cache(data: D.DeepData, threads: int = 4) -> None:
    """Guarda la salida de los sub-bloques 4-2 y 4-4 (g42, g44) para ventanas reales y mezclas, desde f3 congelado."""
    from . import mixtures as MX

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    top = BirdNETTop("f3", class_indices=data.class_index).eval()
    sources = [("", data.f3, data.logit)]
    if (MX.MIX_DIR / "done.npy").exists() and np.load(MX.MIX_DIR / "done.npy").all():
        feats = MX.load_features(maps=True)
        sources.append(("_mix", feats["f3"], np.asarray(feats["logit"], dtype=np.float32)))
    for suffix, f3, ref in sources:
        n = len(f3)
        if all((CACHE_DIR / f"{c}{suffix}.npy").exists() for c in ("g42", "g44")):
            continue
        outs = {c: np.lib.format.open_memmap(CACHE_DIR / f"{c}{suffix}.tmp.npy", mode="w+", dtype=np.float16, shape=(n, 3, 8, 192)) for c in ("g42", "g44")}
        worst = 0.0
        t0 = time.time()
        with torch.no_grad():
            for a in range(0, n, 512):
                x = torch.from_numpy(np.asarray(f3[a : a + 512], dtype=np.float32))
                g42 = top(x, stop_at=CUT_TENSORS["g42"])
                outs["g42"][a : a + 512] = g42.numpy().astype(np.float16)
                if a % (512 * 20) == 0:                      # control: la replica reproduce los logits guardados
                    worst = max(worst, float((top(x).numpy() - ref[a : a + 512]).__abs__().max()))
                outs["g44"][a : a + 512] = top(x, stop_at=CUT_TENSORS["g44"]).numpy().astype(np.float16)
        for c, arr in outs.items():
            arr.flush()
            del arr
            (CACHE_DIR / f"{c}{suffix}.tmp.npy").replace(CACHE_DIR / f"{c}{suffix}.npy")
        print(f"  mapas g42/g44{suffix}: {n} ventanas en {time.time() - t0:.0f} s; error maximo de logit de la replica {worst:.4f}", flush=True)


# --------------------------------------------------------------------------- #
# Entrenamiento de un modelo
# --------------------------------------------------------------------------- #
def make_model(cfg: RAConfig, data: D.DeepData, fit_rows: np.ndarray):
    if cfg.rung in HEAD_RUNGS:
        emb = data.emb[fit_rows]
        model = DeltaHead(cfg.rung, data.Y.shape[1], emb.mean(axis=0), emb.std(axis=0) + 1e-3)
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if name.startswith("out."):
                continue
            (decay if p.dim() > 1 else no_decay).append(p)
        groups = [{"params": [model.out.weight, model.out.bias], "weight_decay": cfg.head_decay}, {"params": no_decay, "weight_decay": 0.0}]
        if decay:
            groups.append({"params": decay, "weight_decay": cfg.weight_decay})
        return model, torch.optim.AdamW(groups, lr=cfg.lr)
    cut, groups = DEEP_RUNGS[cfg.rung]
    model = BirdNETTop(cut, class_indices=data.class_index)
    model.make_relative(groups)
    by_group = model.groups()
    param_groups = []
    for g in groups:
        params = [model.params[k] for k in by_group[g] if k in model.relative]
        if not params:
            continue
        if g == "classifier":                       # las filas del clasificador siguen la misma region de confianza
            param_groups.append({"params": params, "lr": cfg.rho, "weight_decay": cfg.decay})
        else:
            param_groups.append({"params": params, "lr": cfg.rho * LAYER_RATE[g], "weight_decay": cfg.decay})
    return model, torch.optim.AdamW(param_groups, lr=cfg.rho)


def predict(model, feats: Features, rows: np.ndarray, mixed: bool = False, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    head = feats.rung in HEAD_RUNGS
    z0 = feats.z0_mix if mixed else feats.z0
    with torch.no_grad():
        for a in range(0, len(rows), batch):
            r = rows[a : a + batch]
            x = feats.mixed(r) if mixed else feats.real(r)
            z = model(x, torch.from_numpy(np.asarray(z0[r], dtype=np.float32))) if head else model(x)
            out.append(torch.sigmoid(z).numpy())
    return np.concatenate(out) if out else np.zeros((0, feats.z0.shape[1]), np.float32)


def train(cfg: RAConfig, data: D.DeepData, cat: np.ndarray, mix: Mixtures | None, feats: Features, fit_rows: np.ndarray, targets: Targets | None = None):
    torch.manual_seed(cfg.seed)
    targets = targets or Targets(data, cat, mix, cfg.labels, cfg.mix_negatives)
    sampler = Sampler(data, cat, targets, mix, fit_rows, cfg)
    model, opt = make_model(cfg, data, fit_rows)
    head = cfg.rung in HEAD_RUNGS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / cfg.warmup if s < cfg.warmup else 0.5 * (1 + math.cos(math.pi * min((s - cfg.warmup) / max(cfg.steps - cfg.warmup, 1), 1.0))))
    ema = EMA(model, cfg.ema)
    roll_rng = np.random.default_rng(cfg.seed + 1)
    losses = []
    model.train()
    for step in range(cfg.steps):
        real, mixes = sampler.batch()
        xs = [feats.real(real)]
        z0s = [data.logit[real]]
        T, sup, easy = [targets.T[real]], [targets.sup[real]], [targets.easy[real]]
        if len(mixes):
            xs.append(feats.mixed(mixes))
            z0s.append(mix.logit[mixes])
            T.append(targets.T_mix[mixes])
            sup.append(targets.sup_all_mix[mixes])
            easy.append(targets.easy_mix[mixes])
        x = torch.cat(xs)
        Tt = torch.from_numpy(np.concatenate(T))
        supt = torch.from_numpy(np.concatenate(sup).astype(np.float32))
        easyt = torch.from_numpy(np.concatenate(easy).astype(np.float32))
        if head:
            z = model(x, torch.from_numpy(np.concatenate(z0s).astype(np.float32)))
        else:
            shift = int(roll_rng.integers(-1, 2))
            z = model(torch.roll(x, shift, dims=2) if shift else x)      # desplazamiento temporal circular de un paso
        loss = ra_loss(z, Tt, supt, easyt, cfg.lam_easy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        losses.append(float(loss.detach()))
    model.load_state_dict(ema.shadow)
    model.eval()
    info = {"loss_start": float(np.mean(losses[:20])), "loss_end": float(np.mean(losses[-50:]))}
    if not head:
        info["relative_drift"] = model.relative_drift()
    return model, info


# --------------------------------------------------------------------------- #
# Seleccion anidada y validacion cruzada
# --------------------------------------------------------------------------- #
clean_map = PR.clean_map


class InnerJudge:
    """Puntua una configuracion en el pliegue interno: S de cantos atenuados, sujeto a no perder mAP limpia."""

    def __init__(self, data, cat, mix, rows: np.ndarray):
        self.data, self.cat, self.mix, self.rows = data, cat, mix, rows
        self.bench = PR.AttenuationBenchmark(data, cat, mix, PR.Resampler(data, reps=1), rows=rows, min_pos=5)
        bn = flat_sigmoid(data.logit[rows])
        self.map0 = clean_map(data, cat, bn, rows)
        self.s0 = self._s(bn, flat_sigmoid(mix.logit[self.bench.mix_rows]))

    def _s(self, prob: np.ndarray, prob_mix: np.ndarray) -> float:
        clean = np.zeros(self.data.Y.shape, np.float32)
        clean[self.rows] = prob
        mixed = np.zeros(self.mix.logit.shape, np.float32)
        mixed[self.bench.mix_rows] = prob_mix
        return self.bench.summary_score(clean, mixed)

    def score(self, model, feats: Features) -> dict:
        prob = predict(model, feats, self.rows)
        s = self._s(prob, predict(model, feats, self.bench.mix_rows, mixed=True))
        m = clean_map(self.data, self.cat, prob, self.rows)
        return {"S": s, "S_birdnet": self.s0, "clean_map": m, "clean_map_birdnet": self.map0, "feasible": bool(m >= self.map0 - PR.NONINFERIORITY)}


def choose(results: dict) -> str:
    """Mayor S entre las configuraciones que no pierden mAP limpia; si ninguna cumple, la de mayor mAP limpia."""
    feasible = {k: v for k, v in results.items() if v["feasible"]}
    if feasible:
        return max(feasible, key=lambda k: feasible[k]["S"])
    return max(results, key=lambda k: results[k]["clean_map"])


def cross_validate(rung: str, data, cat, mix, folds: np.ndarray, picked_by_fold: dict | None = None, variant: dict | None = None, name: str | None = None, verbose: bool = True):
    """Devuelve (oof, oof_mix, info).

    Sin ``picked_by_fold`` cada pliegue elige su configuracion de la rejilla con el pliegue interno. Con
    ``picked_by_fold`` (ablaciones) se reutiliza la elegida por el modelo de referencia y no hay seleccion.
    ``variant``: cambios de configuracion de la ablacion.
    """
    name = name or rung
    feats = Features(data, mix, rung)
    head = rung in HEAD_RUNGS
    grid = HEAD_GRID if head else DEEP_GRID
    base = RAConfig(rung=rung) if head else RAConfig(rung=rung, steps=DEEP_STEPS, batch=(21, 2, 12, 3, 26), warmup=50, ema=0.99)
    if variant:
        base = replace(base, **variant)
    targets = Targets(data, cat, mix, base.labels, base.mix_negatives)
    oof = np.zeros(data.Y.shape, np.float32)
    oof_mix = np.full(mix.logit.shape, np.nan, np.float32) if mix is not None else None
    info = {"folds": []}
    all_folds = sorted(np.unique(folds))
    for f in all_folds:
        t0 = time.time()
        test = np.flatnonzero(folds == f)
        fold_info: dict = {"fold": int(f)}
        if picked_by_fold is not None:
            pick = picked_by_fold[int(f)]
        else:
            v = all_folds[(all_folds.index(f) + 1) % len(all_folds)]
            inner_fit, inner_val = np.flatnonzero((folds != f) & (folds != v)), np.flatnonzero(folds == v)
            judge = InnerJudge(data, cat, mix, inner_val)
            results = {}
            for label, params in grid.items():
                m, _ = train(replace(base, **params), data, cat, mix, feats, inner_fit, targets)
                results[label] = judge.score(m, feats)
            pick = choose(results)
            fold_info["inner"] = results
        cfg = replace(base, **grid[pick])
        fold_info["picked"] = pick
        fit_rows = np.flatnonzero(folds != f)
        model, tinfo = train(cfg, data, cat, mix, feats, fit_rows, targets)
        oof[test] = predict(model, feats, test)
        if mix is not None:
            mrows = np.flatnonzero(np.isin(mix.kind, ("test", "bg_test")) & np.isin(mix.src, test))
            oof_mix[mrows] = predict(model, feats, mrows, mixed=True)
        fold_info |= tinfo | {"seconds": round(time.time() - t0, 1)}
        info["folds"].append(fold_info)
        if verbose:
            print(f"  {name} pliegue {int(f) + 1}: {json.dumps({k: v for k, v in fold_info.items() if k != 'inner'}, default=float)}", flush=True)
            if "inner" in fold_info:
                print("     interno:", {k: (round(v["S"], 4), round(v["clean_map"], 4), v["feasible"]) for k, v in fold_info["inner"].items()}, "BirdNET:", (round(judge.s0, 4), round(judge.map0, 4)), flush=True)
    info["config"] = asdict(base)
    if mix is not None:
        info["mixture_negatives_available"] = int(targets.neg_mix[np.isin(mix.kind, ("train", "bg_train"))].sum())
    return oof, oof_mix, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rungs", default="lineal,mlp")
    parser.add_argument("--ablations", nargs="?", const="mlp_estricto,mlp_sin_mezclas,mlp_semilla2", default=None,
                        help="Variantes del modelo principal con la configuracion que eligio cada pliegue: etiquetas estrictas, sin mezclas, otra semilla "
                             "(por defecto las tres) o mlp_neg_mezcla (negativos certificados de las mezclas).")
    parser.add_argument("--cache", action="store_true", help="Solo calcula los mapas g42/g44.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)      # la correccion empieza en cero: sin esto los gradientes minusculos frenan la CPU
    data = D.load(maps=True)
    if args.cache:
        build_map_cache(data, args.threads)
        return
    cat = PR.categories(data)
    folds = PR.load_folds(data, cat, D.DEEP_DIR / "folds.npy")
    mix = Mixtures()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = D.DEEP_DIR / "oof_ra.npz"
    store = dict(np.load(out_path)) if out_path.exists() else {}
    info_path = RESULTS / "run_info_ra.json"
    infos = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    def save(name, oof, oof_mix, info):
        store[f"oof_{name}"] = oof
        store[f"mix_{name}"] = oof_mix
        infos[name] = info
        np.savez_compressed(out_path, **store)
        info_path.write_text(json.dumps(infos, indent=2, default=float), encoding="utf-8")

    rungs = [r for r in args.rungs.split(",") if r]
    for rung in rungs:
        t0 = time.time()
        oof, oof_mix, info = cross_validate(rung, data, cat, mix, folds)
        save(rung, oof, oof_mix, info)
        print(f"{rung} listo en {time.time() - t0:.0f} s; elegidas: {[fi['picked'] for fi in info['folds']]}", flush=True)
    if args.ablations:
        if "mlp" not in infos:
            raise SystemExit("Las ablaciones necesitan el modelo principal (mlp) ya entrenado.")
        picked = {int(fi["fold"]): fi["picked"] for fi in infos["mlp"]["folds"]}
        variants = {"mlp_estricto": {"labels": "strict"}, "mlp_sin_mezclas": {"crossfades": False}, "mlp_semilla2": {"seed": 43}, "mlp_neg_mezcla": {"mix_negatives": True}}
        for name in [a for a in args.ablations.split(",") if a]:
            t0 = time.time()
            oof, oof_mix, info = cross_validate("mlp", data, cat, mix, folds, picked_by_fold=picked, variant=variants[name], name=name)
            save(name, oof, oof_mix, info)
            print(f"{name} listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
