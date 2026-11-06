"""Ensamblado del conjunto de entrenamiento multietiqueta (X, Y, grupos).

Cada fila es una ventana de 3 s (rejilla 0, 3, ..., 54 s) de una grabacion
copiada. Etiquetas por especie objetivo s:

    Y[i, s] = 1  si un experto valido la deteccion de s en la ventana (Valid=1)
    Y[i, s] = 0  en otro caso (incluye ventanas sin ninguna deteccion validada,
                 "fondo", y ventanas con otras especies)

``hard_neg[i, s]`` marca los negativos explicitos (Valid=0: BirdNET predijo s
y el experto lo rechazo) y ``uncertain[i, s]`` las ventanas donde BirdNET
asigna a s una puntuacion alta (>= ``uncertain_threshold``) pero no hay
validacion: pueden ser falsos positivos rechazados o detecciones que no se
revisaron. El analisis reporta metricas con y sin enmascararlas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, labels
from .embeddings import ALL_NPZ, flat_sigmoid

# Nombres cientificos de TablaFinal que difieren de la taxonomia de BirdNET V2.4.
BIRDNET_SYNONYMS = {
    "Hesperoburhinus superciliaris": "Burhinus superciliaris",
    "Tyto furcata": "Tyto alba",
    "Xenops mexicanus": "Xenops minutus",
}


@dataclass
class Dataset:
    X: np.ndarray                 # (N, 1024) embeddings BirdNET sin normalizar
    Y: np.ndarray                 # (N, S) etiquetas 0/1
    hard_neg: np.ndarray          # (N, S) negativos explicitos
    uncertain: np.ndarray         # (N, S) no validadas con score BirdNET alto
    birdnet_logit: np.ndarray     # (N, S) logit BirdNET de cada especie objetivo
    birdnet_top1: np.ndarray      # (N,) indice de la clase BirdNET mas probable
    any_positive: np.ndarray      # (N,) la ventana tiene alguna especie validada
    meta: pd.DataFrame            # file_key, device, date, period, session, start
    species: list[str]
    birdnet_labels: list[str]
    species_info: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def groups(self) -> np.ndarray:
        return self.meta["session"].to_numpy()


def birdnet_label_index(birdnet_labels: list[str]) -> dict[str, int]:
    return {lab.split("_")[0]: i for i, lab in enumerate(birdnet_labels)}


def session_consistency(det: pd.DataFrame, data, *, tolerance: float = 0.02, logits: np.ndarray | None = None) -> pd.DataFrame:
    """Control de calidad: concordancia entre la confianza registrada en
    TablaFinal y la recalculada por este pipeline, por sesion.

    Si el audio copiado es el mismo que analizo el equipo, BirdNET debe
    reproducir su confianza casi exactamente (|diferencia| < tolerance). Una
    concordancia baja indica que los archivos de esa carpeta no corresponden
    a las etiquetas (p. ej. archivos mezclados o renombrados).
    """
    bn_index = birdnet_label_index([str(x) for x in data["labels"]])
    keys = {(fk, st): i for i, (fk, st) in enumerate(zip(data["file_key"].astype(str), np.round(data["start"].astype(float), 2)))}
    logits = data["logits"] if logits is None else logits
    rows = []
    for r in det.itertuples(index=False):
        i = keys.get((r.file_key, round(float(r.start), 2)))
        sp = BIRDNET_SYNONYMS.get(r.species, r.species)
        if i is None or sp not in bn_index:
            continue
        ours = float(flat_sigmoid(np.float32(logits[i, bn_index[sp]])))
        rows.append((r.session, abs(ours - float(r.birdnet_conf)) < tolerance))
    qc = pd.DataFrame(rows, columns=["session", "match"])
    return qc.groupby("session")["match"].agg(rows="size", agreement="mean").reset_index()


def session_from_file_key(file_key: pd.Series) -> pd.DataFrame:
    """device, date, period y session a partir de 'AM001/Sunrise/20250724_060000.WAV'."""
    parts = file_key.str.split("/", expand=True)
    device, period, fname = parts[0], parts[1], parts[2]
    date = fname.str.slice(0, 8)
    return pd.DataFrame({"device": device, "date": date, "period": period, "session": device + "_" + date + "_" + period})


def build_dataset(
    *,
    min_positives_subset: int = 10,
    min_sessions: int = 2,
    uncertain_threshold: float = 0.5,
    min_session_agreement: float = 0.5,
    det: pd.DataFrame | None = None,
) -> Dataset:
    data = np.load(ALL_NPZ, allow_pickle=False)
    det = labels.load_detections() if det is None else det
    # Con el conjunto completo los logits ocupan varios GB (ventanas x 6 522 clases):
    # se cargan una sola vez en float16 y solo se convierten las columnas necesarias.
    logits16 = data["logits"]

    qc = session_consistency(det, data, logits=logits16)
    bad_sessions = set(qc.loc[qc["agreement"] < min_session_agreement, "session"])
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    qc.assign(excluded=qc["session"].isin(bad_sessions)).to_csv(config.RESULTS_DIR / "qc_sessions.csv", index=False)

    file_key_all = data["file_key"].astype(str)
    # La sesion se deduce del propio file_key (grabadora/periodo/AAAAMMDD_HHMMSS.WAV):
    # en el conjunto completo la mayoria de grabaciones no tiene ninguna deteccion.
    meta_all = session_from_file_key(pd.Series(file_key_all))
    keep = ~meta_all["session"].isin(bad_sessions).to_numpy()
    if bad_sessions:
        det = det[~det["session"].isin(bad_sessions)]

    rows_kept = np.flatnonzero(keep)
    file_key = file_key_all[rows_kept]
    start = np.round(data["start"].astype(float), 2)[rows_kept]
    emb = data["emb"]
    emb = (emb if len(rows_kept) == len(emb) else emb[rows_kept]).astype(np.float32, copy=False)
    bn_labels = [str(x) for x in data["labels"]]
    bn_index = birdnet_label_index(bn_labels)
    windows = labels.build_windows(det)
    win_key = {(fk, round(float(st), 2)): i for i, (fk, st) in enumerate(zip(windows["file_key"], windows["start"]))}

    meta = pd.concat([pd.DataFrame({"file_key": file_key, "start": start}), meta_all.iloc[rows_kept].reset_index(drop=True)], axis=1)
    unknown = ~meta["session"].isin(set(det["session"]))
    if unknown.any():
        # Sesiones que el equipo no reviso: no tienen etiquetas y contarian como negativos.
        raise ValueError(f"Grabaciones fuera de las sesiones de TablaFinal: {meta.loc[unknown, 'file_key'].unique()[:5]}")

    pos_lists, neg_lists = [], []
    for fk, st in zip(file_key, start):
        j = win_key.get((fk, float(st)))
        if j is None:
            pos_lists.append([])
            neg_lists.append([])
        else:
            pos_lists.append(windows.at[j, "pos_species"])
            neg_lists.append(windows.at[j, "neg_species"])

    # Especies objetivo: >= 15 positivos en TablaFinal y suficientes en el subconjunto copiado.
    candidates = labels.target_species(det)
    counts = pd.Series([s for lst in pos_lists for s in lst]).value_counts()
    sessions_per_species: dict[str, set[str]] = {}
    for lst, sess in zip(pos_lists, meta["session"]):
        for s in lst:
            sessions_per_species.setdefault(s, set()).add(sess)
    species = [
        s
        for s in candidates
        if counts.get(s, 0) >= min_positives_subset and len(sessions_per_species.get(s, ())) >= min_sessions
    ]
    s_index = {s: k for k, s in enumerate(species)}

    n, n_s = len(file_key), len(species)
    Y = np.zeros((n, n_s), dtype=np.float32)
    hard_neg = np.zeros((n, n_s), dtype=bool)
    for i, (pos, neg) in enumerate(zip(pos_lists, neg_lists)):
        for s in pos:
            if s in s_index:
                Y[i, s_index[s]] = 1.0
        for s in neg:
            if s in s_index:
                hard_neg[i, s_index[s]] = True

    cols = []
    used_labels = []
    for s in species:
        key = BIRDNET_SYNONYMS.get(s, s)
        if key not in bn_index:
            raise KeyError(f"Especie sin etiqueta BirdNET: {s}")
        cols.append(bn_index[key])
        used_labels.append(bn_labels[bn_index[key]])
    birdnet_logit = logits16[:, cols][rows_kept].astype(np.float32)
    top1 = np.empty(len(rows_kept), dtype=np.int32)
    for a in range(0, len(rows_kept), 20_000):
        top1[a : a + 20_000] = np.argmax(logits16[rows_kept[a : a + 20_000]], axis=1)
    del logits16
    uncertain = (flat_sigmoid(birdnet_logit) >= uncertain_threshold) & (Y == 0)

    info = labels.species_table(det).reindex(species)
    info["subset_positives"] = Y.sum(axis=0).astype(int)
    info["subset_sessions"] = [len(sessions_per_species.get(s, ())) for s in species]
    info["subset_hard_negatives"] = hard_neg.sum(axis=0).astype(int)
    info["birdnet_label"] = used_labels

    return Dataset(
        X=emb,
        Y=Y,
        hard_neg=hard_neg,
        uncertain=uncertain,
        birdnet_logit=birdnet_logit,
        birdnet_top1=top1,
        any_positive=np.array([len(p) > 0 for p in pos_lists]),
        meta=meta,
        species=species,
        birdnet_labels=used_labels,
        species_info=info,
    )


def summarize(ds: Dataset) -> dict:
    return {
        "windows": int(len(ds.X)),
        "recordings": int(ds.meta["file_key"].nunique()),
        "sessions": int(ds.meta["session"].nunique()),
        "devices": sorted(ds.meta["device"].unique().tolist()),
        "species": len(ds.species),
        "positive_pairs": int(ds.Y.sum()),
        "windows_with_target_positive": int((ds.Y.sum(axis=1) > 0).sum()),
        "windows_with_any_validated_species": int(ds.any_positive.sum()),
        "background_windows": int((~ds.any_positive).sum()),
        "hard_negative_pairs": int(ds.hard_neg.sum()),
        "uncertain_pairs": int(ds.uncertain.sum()),
        "embedding_dim": int(ds.X.shape[1]),
    }
