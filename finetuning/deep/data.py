"""Conjunto de ventanas para el aprendizaje profundo (muestra estratificada con pesos).

El conjunto completo (16 629 grabaciones, 3,6 GB de embeddings) no se puede
copiar a este entorno, asi que se trabaja con una muestra probabilistica de
grabaciones con probabilidad de inclusion conocida:

    A  grabaciones con algun positivo de una especie objetivo "rara" (<= 450
       positivos) o con algun rechazo experto (Valid = 0)            p = 1
    B  resto de grabaciones con detecciones revisadas                p = 0,30
    C  grabaciones de sesiones revisadas sin detecciones revisadas   p = 0,10

Cada ventana hereda el peso 1/p de su grabacion; con esos pesos las metricas
estiman las del conjunto completo (estimador de Horvitz-Thompson).

``build`` consolida las caracteristicas por grabacion (``extract``) en matrices
contiguas dentro de ``<WORK_DIR>/deep`` y ``load`` las abre (las grandes, como
mapas de memoria).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, labels
from ..dataset import BIRDNET_SYNONYMS, birdnet_label_index
from ..embeddings import flat_sigmoid
from . import extract
from .tflite_tools import load_labels

DEEP_DIR = config.WORK_DIR / "deep"
MANIFEST = config.WORK_DIR / "wav_manifest.csv"
FRAME = config.WORK_DIR / "sampling_frame.csv"
UNCERTAIN_THRESHOLD = 0.5
QC_TOLERANCE = 0.02
QC_MIN_AGREEMENT = 0.5


@dataclass
class DeepData:
    emb: np.ndarray            # (N, 1024) float32
    logit: np.ndarray          # (N, S) float32, logits BirdNET de las especies objetivo
    Y: np.ndarray              # (N, S) float32
    hard_neg: np.ndarray       # (N, S) bool, rechazos expertos (Valid = 0)
    uncertain: np.ndarray      # (N, S) bool, BirdNET >= 0,5 sin validar
    weight: np.ndarray         # (N,) float32, 1 / probabilidad de inclusion de la grabacion
    meta: pd.DataFrame         # file_key, session, device, date, period, start, rec, pos, stratum
    species: list[str]
    birdnet_labels: list[str]
    class_index: list[int]     # columnas de las especies objetivo en la salida de BirdNET
    f3: np.ndarray | None = None       # (N, 6, 16, 108) float16 (mapa de memoria)
    f2: np.ndarray | None = None       # (N, 12, 32, 72) float16 (mapa de memoria)
    logits_all: np.ndarray | None = None  # (N, 6522) float16 (mapa de memoria)
    info: dict = field(default_factory=dict)

    @property
    def groups(self) -> np.ndarray:
        return self.meta["session"].to_numpy()

    @property
    def prob(self) -> np.ndarray:
        return flat_sigmoid(self.logit)


def _session_parts(file_key: pd.Series) -> pd.DataFrame:
    parts = file_key.str.split("/", expand=True)
    date = parts[2].str.slice(0, 8)
    return pd.DataFrame({"device": parts[0], "date": date, "period": parts[1], "session": parts[0] + "_" + date + "_" + parts[1]})


def consolidate(consume: bool = False, verbose: bool = True) -> int:
    """Une las caracteristicas por grabacion en matrices contiguas (mapas de memoria).

    Con ``consume=True`` cada archivo por grabacion se borra tras copiarse, para no
    duplicar el espacio en disco (las caracteristicas se pueden regenerar desde el audio).
    """
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MANIFEST, dtype={"file_key": str, "session": str, "stratum": str}).fillna("")
    man = man[[extract.cache_path(k).exists() for k in man.file_key]].reset_index(drop=True)
    if man.empty:
        raise RuntimeError("No hay caracteristicas extraidas (ejecuta finetuning.deep.extract).")
    n_classes = len(load_labels())
    counts = []
    for k in man.file_key:
        with np.load(extract.cache_path(k)) as d:
            counts.append(int(d["starts"].shape[0]))
    n = int(np.sum(counts))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    emb = np.lib.format.open_memmap(DEEP_DIR / "emb.npy", mode="w+", dtype=np.float32, shape=(n, 1024))
    lall = np.lib.format.open_memmap(DEEP_DIR / "logits_all.npy", mode="w+", dtype=np.float16, shape=(n, n_classes))
    f3 = np.lib.format.open_memmap(DEEP_DIR / "f3.npy", mode="w+", dtype=np.float16, shape=(n, 6, 16, 108))
    f2 = np.lib.format.open_memmap(DEEP_DIR / "f2.npy", mode="w+", dtype=np.float16, shape=(n, 12, 32, 72))
    rows = []
    for r, (rec, a, b) in enumerate(zip(man.itertuples(), offsets[:-1], offsets[1:])):
        path = extract.cache_path(rec.file_key)
        with np.load(path) as d:
            emb[a:b], lall[a:b], f3[a:b], f2[a:b] = d["emb"], d["logits"], d["f3"], d["f2"]
            rows.extend((rec.file_key, float(st), r, j) for j, st in enumerate(d["starts"]))
        if consume:
            path.unlink()
        if verbose and (r + 1) % 500 == 0:
            print(f"  consolidadas {r + 1}/{len(man)} grabaciones", flush=True)
    for arr in (emb, lall, f3, f2):
        arr.flush()
    pd.DataFrame(rows, columns=["file_key", "start", "rec", "pos"]).to_csv(DEEP_DIR / "index.csv", index=False)
    if verbose:
        print(f"Consolidado: {len(man)} grabaciones, {n} ventanas en {DEEP_DIR}", flush=True)
    return n


def build(verbose: bool = True) -> dict:
    """Etiquetas, pesos y metadatos sobre las matrices consolidadas (rapido; se puede repetir)."""
    man = pd.read_csv(MANIFEST, dtype={"file_key": str, "session": str, "stratum": str}).fillna("")
    meta = pd.read_csv(DEEP_DIR / "index.csv")
    meta = meta.merge(man[["file_key", "stratum", "weight"]], on="file_key", how="left", validate="many_to_one")
    if meta.weight.isna().any():
        raise ValueError("Hay grabaciones consolidadas que no estan en el manifiesto.")
    meta = pd.concat([meta, _session_parts(meta["file_key"])], axis=1)
    n = len(meta)
    det = labels.load_detections()
    bn_labels = load_labels()
    bn_index = birdnet_label_index(bn_labels)
    lall = np.load(DEEP_DIR / "logits_all.npy", mmap_mode="r")

    # Control de calidad: la confianza registrada en TablaFinal debe coincidir con la recalculada.
    key = {(fk, round(st, 2)): i for i, (fk, st) in enumerate(zip(meta.file_key, meta.start))}
    qc = []
    for d in det.itertuples(index=False):
        i = key.get((d.file_key, round(float(d.start), 2)))
        col = bn_index.get(BIRDNET_SYNONYMS.get(d.species, d.species))
        if i is None or col is None:
            continue
        p = float(flat_sigmoid(np.float32(lall[i, col])))
        qc.append((d.session, abs(p - float(d.birdnet_conf)) <= QC_TOLERANCE))
    qc = pd.DataFrame(qc, columns=["session", "match"]).groupby("session")["match"].agg(rows="size", agreement="mean").reset_index()
    bad = set(qc.loc[qc.agreement < QC_MIN_AGREEMENT, "session"])
    qc.assign(excluded=qc.session.isin(bad)).to_csv(DEEP_DIR / "qc_sessions.csv", index=False)
    if bad:
        det = det[~det.session.isin(bad)]

    # Especies objetivo: >= 15 positivos en TablaFinal y presentes en >= 2 sesiones de la muestra.
    windows = labels.build_windows(det)
    wkey = {(fk, round(float(st), 2)): i for i, (fk, st) in enumerate(zip(windows.file_key, windows.start))}
    pos_lists, neg_lists = [], []
    for fk, st in zip(meta.file_key, meta.start):
        j = wkey.get((fk, round(st, 2)))
        pos_lists.append(windows.at[j, "pos_species"] if j is not None else [])
        neg_lists.append(windows.at[j, "neg_species"] if j is not None else [])
    sess_of: dict[str, set] = {}
    cnt: dict[str, int] = {}
    for lst, s in zip(pos_lists, meta.session):
        for sp in lst:
            cnt[sp] = cnt.get(sp, 0) + 1
            sess_of.setdefault(sp, set()).add(s)
    species = [s for s in labels.target_species(det) if cnt.get(s, 0) >= 10 and len(sess_of.get(s, ())) >= 2]
    sidx = {s: k for k, s in enumerate(species)}
    Y = np.zeros((n, len(species)), np.float32)
    hard = np.zeros((n, len(species)), bool)
    for i, (pos, neg) in enumerate(zip(pos_lists, neg_lists)):
        for s in pos:
            if s in sidx:
                Y[i, sidx[s]] = 1.0
        for s in neg:
            if s in sidx:
                hard[i, sidx[s]] = True
    cols = [bn_index[BIRDNET_SYNONYMS.get(s, s)] for s in species]
    logit = np.asarray(lall[:, cols], dtype=np.float32)
    keep = ~meta.session.isin(bad).to_numpy()
    np.save(DEEP_DIR / "Y.npy", Y)
    np.save(DEEP_DIR / "hard_neg.npy", hard)
    np.save(DEEP_DIR / "logit.npy", logit)
    np.save(DEEP_DIR / "keep.npy", keep)
    meta.to_csv(DEEP_DIR / "meta.csv", index=False)
    frame = pd.read_csv(FRAME)
    info = {
        "species": species,
        "birdnet_labels": [bn_labels[c] for c in cols],
        "class_index": [int(c) for c in cols],
        "recordings": int(meta.file_key[keep].nunique()),
        "windows": int(keep.sum()),
        "sessions": int(meta.session[keep].nunique()),
        "excluded_sessions": sorted(bad),
        "positive_pairs": int(Y[keep].sum()),
        "hard_negative_pairs": int(hard[keep].sum()),
        "strata_sample": meta.loc[keep].drop_duplicates("file_key").stratum.value_counts().to_dict(),
        "strata_universe": frame.stratum.value_counts().to_dict(),
        "estimated_universe_windows": float(meta.weight[keep].sum()),
        "estimated_universe_positive_pairs": float((Y[keep] * meta.weight.to_numpy()[keep, None]).sum()),
        "qc_agreement_kept": float(qc.loc[~qc.session.isin(bad)].eval("rows * agreement").sum() / max(qc.loc[~qc.session.isin(bad), "rows"].sum(), 1)),
    }
    (DEEP_DIR / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(json.dumps({k: v for k, v in info.items() if k not in ("species", "birdnet_labels", "class_index")}, indent=2, ensure_ascii=False))
    return info


def load(maps: bool = True) -> DeepData:
    """Abre el conjunto consolidado (sin las sesiones excluidas por el control de calidad)."""
    info = json.loads((DEEP_DIR / "info.json").read_text(encoding="utf-8"))
    keep = np.load(DEEP_DIR / "keep.npy")
    rows = np.flatnonzero(keep)
    full = bool(keep.all())
    meta = pd.read_csv(DEEP_DIR / "meta.csv", dtype={"date": str})
    emb = np.load(DEEP_DIR / "emb.npy", mmap_mode="r")
    logit = np.load(DEEP_DIR / "logit.npy")
    Y = np.load(DEEP_DIR / "Y.npy")
    hard = np.load(DEEP_DIR / "hard_neg.npy")

    def big(name):
        if not maps or not (DEEP_DIR / name).exists():
            return None
        arr = np.load(DEEP_DIR / name, mmap_mode="r")
        return arr if full else _RowView(arr, rows)

    sel = (lambda a: a) if full else (lambda a: a[rows])
    logit_s, Y_s = sel(logit), sel(Y)
    return DeepData(
        emb=np.asarray(emb if full else emb[rows], dtype=np.float32),
        logit=logit_s,
        Y=Y_s,
        hard_neg=sel(hard),
        uncertain=(flat_sigmoid(logit_s) >= UNCERTAIN_THRESHOLD) & (Y_s == 0),
        weight=meta.weight.to_numpy(dtype=np.float32)[rows],
        meta=meta.iloc[rows].reset_index(drop=True),
        species=info["species"],
        birdnet_labels=info["birdnet_labels"],
        class_index=info["class_index"],
        f3=big("f3.npy"),
        f2=big("f2.npy"),
        logits_all=big("logits_all.npy"),
        info=info,
    )


class _RowView:
    """Vista por filas de un mapa de memoria (para excluir sesiones sin copiar la matriz)."""

    def __init__(self, base: np.ndarray, rows: np.ndarray):
        self.base, self.rows = base, rows
        self.shape = (len(rows), *base.shape[1:])
        self.dtype = base.dtype

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.base[self.rows[idx[0]]][(slice(None), *idx[1:])]
        return self.base[self.rows[idx]]


def inner_split(groups: np.ndarray, frac: float = 0.15, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Marca como validacion interna un ``frac`` de las sesiones (parada temprana y seleccion)."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * len(uniq))))
    val = set(rng.choice(uniq, size=n_val, replace=False))
    return np.array([g in val for g in groups])


if __name__ == "__main__":
    import sys

    if "--consolidar" in sys.argv:
        consolidate(consume="--consumir" in sys.argv)
    build()
