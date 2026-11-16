"""Mezclas de audio con relacion senal-ruido controlada.

Las etiquetas solo existen donde BirdNET ya detecto la especie (sesgo de
verificacion), asi que no permiten medir si un modelo encuentra cantos mas
debiles. Aqui se construye una prueba con verdad conocida: se toma una ventana
con una especie validada, se atenua ``A`` dB y se le suma una ventana de fondo
de la MISMA sesion a su nivel natural:

    x_mezcla = 10^(-A/20) * x_positiva + x_fondo

El canto sigue presente (por construccion) pero mas debil, como si el ave
estuviera mas lejos. Al mezclar solo dentro de una sesion, cada mezcla pertenece
a un unico pliegue de la validacion cruzada (los pliegues se agrupan por sesion)
y no hay fuga de informacion entre entrenamiento y prueba.

Dos familias, con ventanas de fondo distintas:
    test    cada positivo x atenuaciones fijas (0 = original, 6, 12, 18, 24 dB)
    train   cada positivo x ``k`` atenuaciones aleatorias en [3, 24] dB (aumento de datos)
Ademas, mezclas fondo + fondo como negativos procesados igual que los positivos.

Uso:
    python -m finetuning.deep.mixtures --plan     # escribe el plan (reproducible)
    python -m finetuning.deep.mixtures --extract  # calcula embeddings, logits y f3 de las mezclas
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..embeddings import flat_sigmoid
from . import data as D
from .extract import CHUNK, read_audio, split_signal
from .tflite_tools import MultiOutputBirdNET

MIX_DIR = D.DEEP_DIR / "mix"
TEST_ATTENUATIONS = (6.0, 12.0, 18.0, 24.0)
TRAIN_RANGE = (3.0, 24.0)
BACKGROUND_MAX_PROB = 0.10   # una ventana de fondo no debe tener ninguna especie objetivo con BirdNET >= 0,1
SEED = 2026


def background_mask(data: D.DeepData) -> np.ndarray:
    """Ventanas sin especie validada ni rechazada y sin candidatos de BirdNET entre las especies objetivo."""
    quiet = flat_sigmoid(data.logit).max(axis=1) < BACKGROUND_MAX_PROB
    return quiet & (data.Y.sum(axis=1) == 0) & (~data.hard_neg.any(axis=1))


def make_plan(data: D.DeepData, k_train: int = 2, bg_pairs_per_session: int = 40, seed: int = SEED) -> pd.DataFrame:
    """Plan reproducible: una fila por mezcla (indices de ventana en el conjunto consolidado)."""
    rng = np.random.default_rng(seed)
    bg = background_mask(data)
    sessions = data.meta["session"].to_numpy()
    full = data.meta["start"].to_numpy() < 54.0 - 1e-6  # se evita la ultima ventana (1 s de audio + ceros)
    pos_rows = np.flatnonzero((data.Y.sum(axis=1) > 0) & full)
    rows = []
    bg_by_session = {s: np.flatnonzero(bg & full & (sessions == s)) for s in np.unique(sessions)}
    for i in pos_rows:
        pool = bg_by_session[sessions[i]]
        if len(pool) == 0:
            continue
        picks = rng.choice(pool, size=len(TEST_ATTENUATIONS) + k_train, replace=len(pool) < len(TEST_ATTENUATIONS) + k_train)
        for a, j in zip(TEST_ATTENUATIONS, picks[: len(TEST_ATTENUATIONS)]):
            rows.append(("test", int(i), int(j), float(a)))
        for j in picks[len(TEST_ATTENUATIONS) :]:
            rows.append(("train", int(i), int(j), float(rng.uniform(*TRAIN_RANGE))))
    for s, pool in bg_by_session.items():
        if len(pool) < 2:
            continue
        n = min(bg_pairs_per_session, len(pool))
        a_idx = rng.choice(pool, size=n, replace=False)
        b_idx = rng.choice(pool, size=n, replace=True)
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            kind = "bg_test" if t % 2 == 0 else "bg_train"
            rows.append((kind, int(i), int(j), float(rng.choice(TEST_ATTENUATIONS) if kind == "bg_test" else rng.uniform(*TRAIN_RANGE))))
    plan = pd.DataFrame(rows, columns=["kind", "src", "bg", "atten_db"])
    plan["session"] = sessions[plan["src"].to_numpy()]
    return plan


class _AudioCache:
    """Lectura de ventanas de audio por (grabacion, posicion) con una cache pequena de archivos."""

    def __init__(self, data: D.DeepData, capacity: int = 64):
        man = pd.read_csv(D.MANIFEST, dtype=str).set_index("file_key")["rel_path"]
        self.rel = man
        self.meta = data.meta
        self.capacity = capacity
        self._files: dict[str, np.ndarray] = {}

    def window(self, row: int) -> np.ndarray:
        fk = self.meta.at[row, "file_key"]
        if fk not in self._files:
            if len(self._files) >= self.capacity:
                self._files.pop(next(iter(self._files)))
            self._files[fk] = split_signal(read_audio(config.DATA_ROOT / self.rel[fk]))
        return self._files[fk][int(self.meta.at[row, "pos"])]


def mix(pos: np.ndarray, bg: np.ndarray, atten_db: float) -> np.ndarray:
    return (10.0 ** (-atten_db / 20.0)) * pos + bg


def extract(plan: pd.DataFrame, data: D.DeepData, threads: int = 4, batch: int = 32, keep_f2: bool = False) -> None:
    """Calcula las salidas de BirdNET para cada mezcla del plan (reanudable por bloques)."""
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    n = len(plan)
    cols = data.class_index
    emb = np.lib.format.open_memmap(MIX_DIR / "emb.npy", mode="r+" if (MIX_DIR / "emb.npy").exists() else "w+", dtype=np.float32, shape=(n, 1024))
    logit = np.lib.format.open_memmap(MIX_DIR / "logit.npy", mode="r+" if (MIX_DIR / "logit.npy").exists() else "w+", dtype=np.float32, shape=(n, len(cols)))
    f3 = np.lib.format.open_memmap(MIX_DIR / "f3.npy", mode="r+" if (MIX_DIR / "f3.npy").exists() else "w+", dtype=np.float16, shape=(n, 6, 16, 108))
    f2 = None
    if keep_f2:
        f2 = np.lib.format.open_memmap(MIX_DIR / "f2.npy", mode="r+" if (MIX_DIR / "f2.npy").exists() else "w+", dtype=np.float16, shape=(n, 12, 32, 72))
    done_path = MIX_DIR / "done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(n, bool)
    if len(done) != n:
        raise ValueError("El plan cambio de tamano: borra la carpeta de mezclas para regenerarla.")
    net = MultiOutputBirdNET(threads=threads)
    audio = _AudioCache(data)
    # Se procesa ordenado por grabacion de origen para aprovechar la cache de audio.
    order = np.argsort(data.meta["rec"].to_numpy()[plan["src"].to_numpy()], kind="stable")
    order = order[~done[order]]
    t0 = time.time()
    for a in range(0, len(order), batch):
        idx = order[a : a + batch]
        x = np.stack([mix(audio.window(int(plan.at[i, "src"])), audio.window(int(plan.at[i, "bg"])), float(plan.at[i, "atten_db"])) for i in idx]).astype(np.float32)
        out = net(x)
        emb[idx], logit[idx], f3[idx] = out["emb"], out["logits"][:, cols], out["f3"].astype(np.float16)
        if f2 is not None:
            f2[idx] = out["f2"].astype(np.float16)
        done[idx] = True
        if (a // batch) % 50 == 0:
            for arr in (emb, logit, f3):
                arr.flush()
            np.save(done_path, done)
            print(f"  mezclas {int(done.sum())}/{n} ({time.time() - t0:.0f} s)", flush=True)
    for arr in (emb, logit, f3):
        arr.flush()
    np.save(done_path, done)
    print(f"Mezclas listas: {int(done.sum())}/{n} ({time.time() - t0:.0f} s)", flush=True)


def load_plan() -> pd.DataFrame:
    return pd.read_csv(MIX_DIR / "plan.csv")


def load_features(maps: bool = True) -> dict[str, np.ndarray]:
    mode = "r" if maps else None
    out = {"emb": np.load(MIX_DIR / "emb.npy", mmap_mode=mode), "logit": np.load(MIX_DIR / "logit.npy"), "f3": np.load(MIX_DIR / "f3.npy", mmap_mode=mode)}
    if (MIX_DIR / "f2.npy").exists():
        out["f2"] = np.load(MIX_DIR / "f2.npy", mmap_mode=mode)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--k-train", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--f2", action="store_true", help="Guarda tambien el mapa f2 (para afinar el bloque 3).")
    args = parser.parse_args()
    data = D.load(maps=False)
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    if args.plan or not (MIX_DIR / "plan.csv").exists():
        plan = make_plan(data, k_train=args.k_train)
        plan.to_csv(MIX_DIR / "plan.csv", index=False)
        print(plan["kind"].value_counts().to_dict(), "sesiones", plan["session"].nunique())
    if args.extract:
        extract(load_plan(), data, threads=args.threads, keep_f2=args.f2)


if __name__ == "__main__":
    main()
