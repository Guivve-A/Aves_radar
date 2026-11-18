"""Cabezas profundas sobre los embeddings de BirdNET (PyTorch) y su entrenamiento.

Modelos (salida: un logit por especie objetivo):
    linear    sonda lineal desde cero
    mlp       perceptron multicapa desde cero
    residual  logit = logit de BirdNET + correccion g(embedding); la ultima capa de g
              empieza en cero, asi que el modelo arranca siendo exactamente BirdNET
    context   como ``residual`` pero la correccion ve la secuencia de ventanas de la
              grabacion (GRU bidireccional): usa el contexto temporal

Tratamiento de las etiquetas (sesgo de verificacion). Cada par (ventana, especie) es:
    positivo            validado por un experto
    negativo explicito  rechazado por un experto (pocos y valiosos: peso alto)
    candidato           BirdNET >= ``candidate_threshold`` sin validar: lo mas probable es que
                        nadie lo revisara; por defecto NO entra en la perdida ("ignore")
    negativo            el resto
La perdida usa los pesos de muestreo de la muestra estratificada, un peso por clase
para los positivos y, opcionalmente, un ancla hacia BirdNET en el espacio de logits.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGIT_CLIP = 20.0
LOGIT_SCALE = 5.0


@dataclass
class HeadConfig:
    kind: str = "residual"            # linear | mlp | residual | context
    hidden: int = 512
    depth: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 40
    batch_size: int = 512
    warmup: float = 0.05
    ema: float = 0.998
    pos_weight_power: float = 0.5
    pos_weight_cap: float = 30.0
    loss: str = "bce"                 # bce | asl
    asl_gamma_neg: float = 2.0
    asl_margin: float = 0.05
    candidates: str = "ignore"        # ignore | negative
    candidate_threshold: float = 0.10
    hard_negative_weight: float = 20.0
    anchor: float = 0.0               # peso de ||logit - logit_BirdNET||^2 (pares no positivos)
    mixup: float = 0.0
    use_sampling_weights: bool = True
    mixture_weight: float = 0.0       # > 0: usa las mezclas de entrenamiento con este peso relativo
    patience: int = 8
    seed: int = 0
    gru_hidden: int = 128
    proj: int = 256
    l2_input: bool = False            # normaliza el embedding a norma 1 antes de estandarizar (linear | mlp)
    extra: dict = field(default_factory=dict)


def _mlp(inp: int, hidden: int, depth: int, out: int, dropout: float, zero_last: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = inp
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
        d = hidden
    last = nn.Linear(d, out)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class WindowHead(nn.Module):
    """Cabezas que puntuan cada ventana por separado (linear, mlp, residual)."""

    def __init__(self, cfg: HeadConfig, n_species: int, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))
        if cfg.kind == "linear":
            self.net = nn.Linear(1024, n_species)
        elif cfg.kind == "mlp":
            self.net = _mlp(1024, cfg.hidden, cfg.depth, n_species, cfg.dropout, zero_last=False)
        elif cfg.kind == "residual":
            self.net = _mlp(1024 + n_species, cfg.hidden, cfg.depth, n_species, cfg.dropout, zero_last=True)
        else:
            raise ValueError(cfg.kind)

    def init_bias(self, prior: np.ndarray) -> None:
        """Sesgo de salida = logit de la prevalencia: evita gastar el entrenamiento en aprender que casi todo es negativo."""
        if self.cfg.kind == "residual":
            return
        last = self.net if isinstance(self.net, nn.Linear) else self.net[-1]
        p = np.clip(prior, 1e-5, 1 - 1e-5)
        with torch.no_grad():
            last.bias.copy_(torch.from_numpy(np.log(p / (1 - p)).astype(np.float32)))
            if self.cfg.kind == "linear":
                last.weight.mul_(0.1)

    def forward(self, emb: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
        if self.cfg.l2_input and self.cfg.kind != "residual":
            emb = F.normalize(emb, dim=-1)
        x = (emb - self.mean) / self.std
        if self.cfg.kind == "residual":
            l = logit.clamp(-LOGIT_CLIP, LOGIT_CLIP)
            return l + self.net(torch.cat([x, l / LOGIT_SCALE], dim=1))
        return self.net(x)


class ContextHead(nn.Module):
    """Correccion residual con contexto temporal: GRU bidireccional sobre las ventanas de la grabacion."""

    def __init__(self, cfg: HeadConfig, n_species: int, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))
        self.proj = nn.Sequential(nn.Linear(1024, cfg.proj), nn.GELU(), nn.Dropout(cfg.dropout))
        self.gru = nn.GRU(cfg.proj + n_species, cfg.gru_hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(2 * cfg.gru_hidden + cfg.proj + n_species, n_species)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, emb: torch.Tensor, logit: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``emb``: (B, T, 1024); ``logit``: (B, T, S). Devuelve (B, T, S)."""
        l = logit.clamp(-LOGIT_CLIP, LOGIT_CLIP)
        h = torch.cat([self.proj((emb - self.mean) / self.std), l / LOGIT_SCALE], dim=-1)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            ctx, _ = self.gru(packed)
            ctx, _ = nn.utils.rnn.pad_packed_sequence(ctx, batch_first=True, total_length=h.shape[1])
        else:
            ctx, _ = self.gru(h)
        return l + self.out(self.drop(torch.cat([ctx, h], dim=-1)))


# --------------------------------------------------------------------------- #
# Perdida
# --------------------------------------------------------------------------- #
def candidate_mask(Y: np.ndarray, logit: np.ndarray, hard_neg: np.ndarray, threshold: float) -> np.ndarray:
    """Pares que BirdNET propone (probabilidad >= umbral) pero que nadie valido ni rechazo."""
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    return (prob >= threshold) & (Y == 0) & (~hard_neg)


def pair_weights(
    Y: np.ndarray,
    logit: np.ndarray,
    hard_neg: np.ndarray,
    sample_weight: np.ndarray,
    cfg: HeadConfig,
    pos_weight: np.ndarray | None = None,
    candidate: np.ndarray | None = None,
    weight_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Peso de cada par (ventana, especie) en la perdida y peso por clase de los positivos.

    ``candidate``: mascara explicita de candidatos sin revisar (por defecto se deduce del logit).
    ``weight_scale``: media de los pesos de muestreo usada para normalizar (la del conjunto de entrenamiento).
    """
    scale = float(sample_weight.mean()) if weight_scale is None else float(weight_scale)
    w = np.repeat((sample_weight / scale)[:, None], Y.shape[1], axis=1).astype(np.float32) if cfg.use_sampling_weights else np.ones_like(Y, dtype=np.float32)
    if candidate is None:
        candidate = candidate_mask(Y, logit, hard_neg, cfg.candidate_threshold)
    if cfg.candidates == "ignore":
        w[candidate & (Y == 0)] = 0.0
    if pos_weight is None:
        pos = (w * (Y > 0)).sum(axis=0)
        neg = (w * (Y == 0)).sum(axis=0)
        pos_weight = np.clip((neg / np.maximum(pos, 1e-6)) ** cfg.pos_weight_power, 1.0, cfg.pos_weight_cap).astype(np.float32)
    w = w * np.where(Y > 0, pos_weight[None, :], 1.0)
    w[hard_neg] *= cfg.hard_negative_weight
    return w.astype(np.float32), pos_weight


def pair_loss(z: torch.Tensor, y: torch.Tensor, w: torch.Tensor, cfg: HeadConfig) -> torch.Tensor:
    if cfg.loss == "asl":
        p = torch.sigmoid(z)
        pm = (p - cfg.asl_margin).clamp(min=0)
        loss = -(y * F.logsigmoid(z) + (1 - y) * pm.pow(cfg.asl_gamma_neg) * torch.log((1 - pm).clamp(min=1e-8)))
    else:
        loss = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    return (loss * w).sum() / w.sum().clamp(min=1e-8)


def anchor_loss(z: torch.Tensor, base: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Distancia al logit de BirdNET en los pares no positivos (ancla en el espacio de funciones)."""
    d = (z - base.clamp(-LOGIT_CLIP, LOGIT_CLIP)) * (1 - y)
    return d.pow(2).mean()


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
                else:
                    self.shadow[k].copy_(v)

    def state(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def standardizer(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = emb.mean(axis=0)
    std = emb.std(axis=0) + 1e-3
    return mean.astype(np.float32), std.astype(np.float32)


def _sequences(rec: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices por grabacion (B, T) rellenados con -1 y longitudes."""
    order = np.lexsort((pos, rec))
    uniq, start, count = np.unique(rec[order], return_index=True, return_counts=True)
    T = int(count.max())
    seq = np.full((len(uniq), T), -1, dtype=np.int64)
    for b, (s, c) in enumerate(zip(start, count)):
        seq[b, :c] = order[s : s + c]
    return seq, count.astype(np.int64)


class FitResult:
    def __init__(self, model: nn.Module, best_epoch: int, history: list[dict]):
        self.model, self.best_epoch, self.history = model, best_epoch, history


def predict(model: nn.Module, emb: np.ndarray, logit: np.ndarray, rec: np.ndarray | None = None, pos: np.ndarray | None = None, batch: int = 4096) -> np.ndarray:
    """Probabilidades (N, S)."""
    model.eval()
    out = np.zeros(logit.shape, dtype=np.float32)
    with torch.no_grad():
        if isinstance(model, ContextHead):
            seq, lengths = _sequences(rec, pos)
            for a in range(0, len(seq), 256):
                idx = seq[a : a + 256]
                safe = np.where(idx < 0, 0, idx)
                z = model(torch.from_numpy(np.asarray(emb[safe], dtype=np.float32)), torch.from_numpy(np.asarray(logit[safe], dtype=np.float32)), torch.from_numpy(lengths[a : a + 256]))
                valid = idx >= 0
                out[idx[valid]] = torch.sigmoid(z).numpy()[valid]
        else:
            for a in range(0, len(emb), batch):
                z = model(torch.from_numpy(np.asarray(emb[a : a + batch], dtype=np.float32)), torch.from_numpy(np.asarray(logit[a : a + batch], dtype=np.float32)))
                out[a : a + batch] = torch.sigmoid(z).numpy()
    return out


def fit(
    cfg: HeadConfig,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray] | None,
    val_metric=None,
    extra: dict[str, np.ndarray] | None = None,
    verbose: bool = False,
) -> FitResult:
    """Entrena una cabeza.

    ``train`` / ``val``: diccionarios con emb, logit, Y, hard_neg, weight y (para ``context``) rec, pos.
    ``extra``: filas adicionales de entrenamiento (mezclas atenuadas) con las mismas claves.
    ``val_metric(prob[, prob_mezclas]) -> float``: criterio de parada temprana (mayor es mejor). Si ``val``
    trae ``mix_emb`` / ``mix_logit`` (mezclas de las sesiones de validacion) tambien se puntuan.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_species = train["Y"].shape[1]
    context = cfg.kind == "context"
    base_emb = train["emb"]
    if cfg.l2_input and cfg.kind in ("linear", "mlp"):
        base_emb = base_emb / np.maximum(np.linalg.norm(base_emb, axis=1, keepdims=True), 1e-12)
    mean, std = standardizer(base_emb)
    model: nn.Module = ContextHead(cfg, n_species, mean, std) if context else WindowHead(cfg, n_species, mean, std)

    W, pos_weight = pair_weights(train["Y"], train["logit"], train["hard_neg"], train["weight"], cfg, candidate=train.get("candidate"))
    if not context:
        sw = (train["weight"] / train["weight"].mean())[:, None] if cfg.use_sampling_weights else np.ones((len(train["Y"]), 1), np.float32)
        model.init_bias((sw * train["Y"]).sum(axis=0) / sw.sum())
    emb, logit, Y = train["emb"], train["logit"], train["Y"]
    if extra is not None and cfg.mixture_weight > 0 and not context:
        # Mezclas atenuadas: la especie validada sigue presente (positivo); los candidatos sin revisar
        # de la ventana de origen siguen siendo desconocidos y se pasan en extra["candidate"].
        We, _ = pair_weights(extra["Y"], extra["logit"], extra["hard_neg"], extra["weight"], cfg, pos_weight, candidate=extra.get("candidate"), weight_scale=float(train["weight"].mean()))
        emb = np.concatenate([emb, extra["emb"]])
        logit = np.concatenate([logit, extra["logit"]])
        Y = np.concatenate([Y, extra["Y"]])
        W = np.concatenate([W, cfg.mixture_weight * We])
    emb_t, logit_t, Y_t, W_t = (torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in (emb, logit, Y, W))

    if context:
        seq, lengths = _sequences(train["rec"], train["pos"])
        n_items, bs = len(seq), max(8, cfg.batch_size // 16)
    else:
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
        prob = predict(probe, val["emb"], val["logit"], val.get("rec"), val.get("pos"))
        if "mix_emb" in val and not context:
            return float(val_metric(prob, predict(probe, val["mix_emb"], val["mix_logit"])))
        return float(val_metric(prob))

    best_state, best_score, best_epoch = ema.state(), evaluate(ema.state()), 0
    history = [{"epoch": 0, "val": best_score}]
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = rng.permutation(n_items)
        run = 0.0
        for a in range(0, n_items, bs):
            b = perm[a : a + bs]
            if context:
                idx = seq[b]
                valid = torch.from_numpy(idx >= 0)
                safe = torch.from_numpy(np.where(idx < 0, 0, idx))
                z = model(emb_t[safe], logit_t[safe], torch.from_numpy(lengths[b]))
                y, w, base = Y_t[safe], W_t[safe] * valid[..., None], logit_t[safe]
            else:
                bt = torch.from_numpy(b)
                x, base, y, w = emb_t[bt], logit_t[bt], Y_t[bt], W_t[bt]
                if cfg.mixup > 0 and cfg.kind in ("linear", "mlp"):
                    lam = float(rng.beta(cfg.mixup, cfg.mixup))
                    lam = max(lam, 1 - lam)
                    p2 = torch.randperm(len(bt))
                    x = lam * x + (1 - lam) * x[p2]
                    y = torch.maximum(y, y[p2]) if lam < 0.75 else y
                    w = torch.maximum(w, w[p2])
                z = model(x, base)
            loss = pair_loss(z, y, w, cfg)
            if cfg.anchor > 0 and cfg.kind in ("residual", "context"):
                loss = loss + cfg.anchor * anchor_loss(z, base, y)
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
        if verbose:
            print(f"    epoca {epoch:3d} perdida {run / steps_per_epoch:.4f} val {score:.4f}", flush=True)
        if val is None or val_metric is None or score > best_score + 1e-5:
            best_state, best_score, best_epoch = ema.state(), score, epoch
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best_epoch, history)
