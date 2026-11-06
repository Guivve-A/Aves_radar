"""Cabezas de clasificacion sobre embeddings de BirdNET (NumPy puro).

* ``PrototypeHead``: replica el metodo few-shot actual de la app (prototipos
  K-means por especie y similitud coseno maxima).
* ``MLPHead``: cabeza entrenable (``hidden=0`` -> sonda lineal / regresion
  logistica multietiqueta; ``hidden>0`` -> perceptron multicapa de una capa
  oculta). Se entrena con entropia cruzada binaria ponderada, AdamW,
  dropout, suavizado de etiquetas, mixup opcional y parada temprana.

Las cabezas entrenables guardan su propio preprocesamiento (normalizacion L2
opcional + estandarizacion) para que la inferencia en la app sea exacta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def macro_ap(y: np.ndarray, scores: np.ndarray, mask: np.ndarray | None = None) -> float:
    aps = []
    for s in range(y.shape[1]):
        keep = np.ones(len(y), dtype=bool) if mask is None else ~mask[:, s]
        ys = y[keep, s]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        aps.append(average_precision_score(ys, scores[keep, s]))
    return float(np.mean(aps)) if aps else float("nan")


# --------------------------------------------------------------------------- #
# Prototipos (metodo actual de la app)
# --------------------------------------------------------------------------- #
class PrototypeHead:
    """Few-shot por prototipos: score(x, s) = max_k cos(x, c_{s,k})."""

    def __init__(self, max_prototypes: int = 5, min_chunks_for_cluster: int = 10, seed: int = 42) -> None:
        self.max_prototypes = max_prototypes
        self.min_chunks_for_cluster = min_chunks_for_cluster
        self.seed = seed
        self.prototypes: list[np.ndarray] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PrototypeHead":
        Xn = l2_normalize(X)
        self.prototypes = []
        for s in range(Y.shape[1]):
            pos = Xn[Y[:, s] > 0.5]
            if len(pos) == 0:
                self.prototypes.append(np.zeros((1, X.shape[1]), np.float32))
                continue
            if len(pos) <= self.min_chunks_for_cluster:
                centers = pos.mean(axis=0, keepdims=True)
            else:
                k = min(self.max_prototypes, len(pos))
                km = KMeans(n_clusters=k, random_state=self.seed, n_init="auto").fit(pos)
                centers = km.cluster_centers_
            self.prototypes.append(l2_normalize(np.asarray(centers, np.float32)))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xn = l2_normalize(X)
        return np.stack([np.max(Xn @ P.T, axis=1) for P in self.prototypes], axis=1)


# --------------------------------------------------------------------------- #
# Cabeza entrenable (lineal o MLP)
# --------------------------------------------------------------------------- #
@dataclass
class HeadConfig:
    hidden: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 150
    batch_size: int = 256
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    max_pos_weight: float = 20.0
    l2_input: bool = False
    patience: int = 20
    seed: int = 42

    def tag(self) -> str:
        kind = "lineal" if self.hidden == 0 else f"mlp{self.hidden}"
        return f"{kind}{'_l2' if self.l2_input else ''}{'_mix' if self.mixup_alpha > 0 else ''}"


@dataclass
class MLPHead:
    config: HeadConfig = field(default_factory=HeadConfig)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    params: dict[str, np.ndarray] = field(default_factory=dict)
    best_epoch: int = 0
    history: list[float] = field(default_factory=list)

    # ---- preprocesamiento ---------------------------------------------------
    def _prep(self, X: np.ndarray) -> np.ndarray:
        Z = l2_normalize(X) if self.config.l2_input else X.astype(np.float32)
        return ((Z - self.mean) / self.std).astype(np.float32)

    def _fit_scaler(self, X: np.ndarray) -> None:
        Z = l2_normalize(X) if self.config.l2_input else X.astype(np.float32)
        self.mean = Z.mean(axis=0).astype(np.float32)
        self.std = (Z.std(axis=0) + 1e-6).astype(np.float32)

    # ---- red ------------------------------------------------------------------
    def _init(self, d_in: int, d_out: int, prior: np.ndarray, rng: np.random.Generator) -> None:
        h = self.config.hidden
        bias_out = np.log(np.clip(prior, 1e-4, 1 - 1e-4) / (1 - np.clip(prior, 1e-4, 1 - 1e-4))).astype(np.float32)
        if h > 0:
            self.params = {
                "W1": (rng.standard_normal((d_in, h)) * np.sqrt(2.0 / d_in)).astype(np.float32),
                "b1": np.zeros(h, np.float32),
                "W2": (rng.standard_normal((h, d_out)) * np.sqrt(1.0 / h)).astype(np.float32),
                "b2": bias_out,
            }
        else:
            self.params = {
                "W2": (rng.standard_normal((d_in, d_out)) * 0.01).astype(np.float32),
                "b2": bias_out,
            }

    def _forward(self, Z: np.ndarray, *, train: bool, rng: np.random.Generator | None = None):
        p = self.params
        cache = {}
        if self.config.hidden > 0:
            pre = Z @ p["W1"] + p["b1"]
            H = np.maximum(pre, 0.0)
            if train and self.config.dropout > 0:
                keep = 1.0 - self.config.dropout
                mask = (rng.random(H.shape) < keep).astype(np.float32) / keep
                H = H * mask
                cache["mask"] = mask
            cache.update(pre=pre, H=H)
            logits = H @ p["W2"] + p["b2"]
        else:
            logits = Z @ p["W2"] + p["b2"]
        return logits, cache

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(self._prep(X), train=False)
        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    # ---- entrenamiento --------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        ignore: np.ndarray | None = None,
        fixed_epochs: int | None = None,
    ) -> "MLPHead":
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        self._fit_scaler(X)
        Z = self._prep(X)
        Yt = Y.astype(np.float32)
        W_ign = np.ones_like(Yt) if ignore is None else (~ignore).astype(np.float32)
        n, d = Z.shape
        n_s = Yt.shape[1]
        pos = Yt.sum(axis=0)
        prior = (pos + 1.0) / (n + 2.0)
        pos_weight = np.clip((n - pos) / np.maximum(pos, 1.0), 1.0, cfg.max_pos_weight) ** 0.5
        self._init(d, n_s, prior, rng)

        adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        best_score, best_params, wait = -np.inf, None, 0
        epochs = fixed_epochs if fixed_epochs is not None else cfg.epochs
        Zv = self._prep(X_val) if X_val is not None else None
        self.history = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                zb, yb, wb = Z[idx], Yt[idx], W_ign[idx]
                if cfg.mixup_alpha > 0:
                    lam = rng.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                    perm = rng.permutation(len(idx))
                    zb = lam * zb + (1 - lam) * zb[perm]
                    yb = lam * yb + (1 - lam) * yb[perm]
                    wb = np.minimum(wb, wb[perm])
                if cfg.label_smoothing > 0:
                    yb = yb * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
                logits, cache = self._forward(zb, train=True, rng=rng)
                prob = sigmoid(logits)
                # BCE ponderada: w_pos para y=1, 1 para y=0 (interpolado si y es suave).
                w = wb * (yb * pos_weight + (1 - yb))
                dlogits = (w * (prob - yb)) / len(idx)
                grads = {}
                if cfg.hidden > 0:
                    grads["W2"] = cache["H"].T @ dlogits
                    grads["b2"] = dlogits.sum(axis=0)
                    dH = dlogits @ self.params["W2"].T
                    if "mask" in cache:
                        dH = dH * cache["mask"]
                    dpre = dH * (cache["pre"] > 0)
                    grads["W1"] = zb.T @ dpre
                    grads["b1"] = dpre.sum(axis=0)
                else:
                    grads["W2"] = zb.T @ dlogits
                    grads["b2"] = dlogits.sum(axis=0)
                step += 1
                for k, g in grads.items():
                    adam_m[k] = b1 * adam_m[k] + (1 - b1) * g
                    adam_v[k] = b2 * adam_v[k] + (1 - b2) * g * g
                    mhat = adam_m[k] / (1 - b1**step)
                    vhat = adam_v[k] / (1 - b2**step)
                    decay = cfg.weight_decay * self.params[k] if k.startswith("W") else 0.0
                    self.params[k] = (self.params[k] - cfg.lr * (mhat / (np.sqrt(vhat) + eps) + decay)).astype(np.float32)

            if Zv is not None and fixed_epochs is None:
                score = macro_ap(Y_val, self._forward(Zv, train=False)[0])
                self.history.append(score)
                if score > best_score + 1e-4:
                    best_score, wait = score, 0
                    best_params = {k: v.copy() for k, v in self.params.items()}
                    self.best_epoch = epoch
                else:
                    wait += 1
                    if wait >= cfg.patience:
                        break
        if best_params is not None:
            self.params = best_params
        elif fixed_epochs is not None:
            self.best_epoch = fixed_epochs
        return self

    # ---- exportacion ------------------------------------------------------------
    def to_arrays(self) -> dict[str, np.ndarray]:
        out = {"mean": self.mean, "std": self.std, "l2_input": np.array(int(self.config.l2_input))}
        out.update({k: v for k, v in self.params.items()})
        out["hidden"] = np.array(self.config.hidden)
        return out


# --------------------------------------------------------------------------- #
# Fine-tuning de la capa de clasificacion de BirdNET (L2-SP)
# --------------------------------------------------------------------------- #
@dataclass
class FTConfig:
    l2sp: float = 0.1            # peso de la penalizacion ||W - W0||^2 + ||b - b0||^2
    lr: float = 3e-4
    epochs: int = 120
    batch_size: int = 256
    pos_weight: float = 1.0      # BCE sin reponderar: conserva la calibracion de BirdNET
    patience: int = 15
    seed: int = 42

    def tag(self) -> str:
        return f"ft_birdnet_l{self.l2sp:g}"


@dataclass
class BirdNETFineTuneHead:
    """Afina la capa densa final de BirdNET para las especies locales.

    logit_s(e) = e . w_s + b_s, con (w_s, b_s) inicializados en los pesos
    originales de BirdNET (``W0``, ``b0``) y entrenados sobre los embeddings
    crudos (sin normalizar, igual que BirdNET) minimizando

        L = BCE(y, sigma(logit)) + lambda * (||W - W0||^2 + ||b - b0||^2)

    (regularizacion L2-SP, "L2 hacia el punto de partida"). La parada temprana
    evalua tambien la epoca 0, por lo que si el ajuste no mejora la validacion
    se conserva la capa original de BirdNET.
    """

    W0: np.ndarray
    b0: np.ndarray
    config: FTConfig = field(default_factory=FTConfig)
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    best_epoch: int = 0
    history: list[float] = field(default_factory=list)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32) @ self.W + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def fit(self, X, Y, *, X_val=None, Y_val=None, ignore=None, fixed_epochs=None) -> "BirdNETFineTuneHead":
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        X = X.astype(np.float32)
        Yt = Y.astype(np.float32)
        Wgt = np.ones_like(Yt) if ignore is None else (~ignore).astype(np.float32)
        self.W, self.b = self.W0.copy(), self.b0.copy()
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
        epochs = fixed_epochs if fixed_epochs is not None else cfg.epochs
        best_params = (self.W.copy(), self.b.copy())
        best_score = macro_ap(Y_val, self.decision_function(X_val)) if X_val is not None and fixed_epochs is None else -np.inf
        self.history = [best_score]
        self.best_epoch, wait = 0, 0
        n = len(X)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                xb, yb, wb = X[idx], Yt[idx], Wgt[idx]
                prob = sigmoid(xb @ self.W + self.b)
                w = wb * (yb * cfg.pos_weight + (1 - yb))
                dlogit = w * (prob - yb) / len(idx)
                gW = xb.T @ dlogit + 2 * cfg.l2sp * (self.W - self.W0)
                gb = dlogit.sum(axis=0) + 2 * cfg.l2sp * (self.b - self.b0)
                step += 1
                mW = beta1 * mW + (1 - beta1) * gW
                vW = beta2 * vW + (1 - beta2) * gW * gW
                mb = beta1 * mb + (1 - beta1) * gb
                vb = beta2 * vb + (1 - beta2) * gb * gb
                self.W -= cfg.lr * (mW / (1 - beta1**step)) / (np.sqrt(vW / (1 - beta2**step)) + eps)
                self.b -= cfg.lr * (mb / (1 - beta1**step)) / (np.sqrt(vb / (1 - beta2**step)) + eps)
            if X_val is not None and fixed_epochs is None:
                score = macro_ap(Y_val, self.decision_function(X_val))
                self.history.append(score)
                if score > best_score + 1e-4:
                    best_score, wait, self.best_epoch = score, 0, epoch
                    best_params = (self.W.copy(), self.b.copy())
                else:
                    wait += 1
                    if wait >= cfg.patience:
                        break
        if fixed_epochs is None:
            self.W, self.b = best_params
        else:
            self.best_epoch = fixed_epochs
        return self

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b, "W0": self.W0, "b0": self.b0, "l2_input": np.array(0), "hidden": np.array(-1)}
