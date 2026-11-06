"""Evaluacion (validacion cruzada agrupada), fine-tuning, calibracion y exportacion.

Protocolo
---------
1. Validacion cruzada de 5 pliegues agrupada por *sesion* (grabadora x dia x
   periodo): ninguna sesion aparece a la vez en entrenamiento y prueba, lo que
   evita la fuga de informacion por autocorrelacion temporal.
2. Dentro de cada pliegue, el 15 % de las sesiones de entrenamiento se reserva
   para la parada temprana (validacion interna).
3. Metodos comparados:
     birdnet       capa de clasificacion original de BirdNET V2.4 (sin ajuste)
     ft_birdnet    misma capa afinada con datos locales (regularizacion L2-SP)
     prototipos    few-shot por prototipos coseno (metodo actual de la app)
     lineal(_l2)   sonda lineal entrenada desde cero (entrada cruda / L2)
     mlp(_l2)      MLP de una capa oculta entrenado desde cero
4. Tres regimenes de evaluacion: estricto (todo lo no validado es negativo),
   enmascarado (se excluyen pares inciertos) y candidatos (solo ventanas con
   score BirdNET >= 0,1 para la especie: verificacion de detecciones).
5. Bootstrap pareado por sesiones, validacion dejando una grabadora fuera
   (LODO), curva de aprendizaje y calibracion de umbrales por especie.

Uso:
    python -m finetuning.train
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import config
from . import dataset as D
from .embeddings import flat_sigmoid
from .heads import BirdNETFineTuneHead, FTConfig, HeadConfig, MLPHead, PrototypeHead

OOF_NPZ = config.WORK_DIR / "oof_scores.npz"
DENSE_NPZ = config.WORK_DIR / "birdnet_dense.npz"
CANDIDATE_THRESHOLD = 0.1

SCRATCH = {
    "lineal": HeadConfig(hidden=0, dropout=0.0, lr=1e-3, weight_decay=1e-3, epochs=200, patience=25),
    "lineal_l2": HeadConfig(hidden=0, dropout=0.0, lr=1e-3, weight_decay=1e-3, epochs=200, patience=25, l2_input=True),
    "mlp": HeadConfig(hidden=256, dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=150, patience=20),
    "mlp_l2": HeadConfig(hidden=256, dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=150, patience=20, l2_input=True),
    "mlp_l2_mixup": HeadConfig(hidden=256, dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=150, patience=20, l2_input=True, mixup_alpha=0.4),
}
FINETUNE = {
    "ft_birdnet": FTConfig(l2sp=0.3, lr=3e-4),
    "ft_birdnet_l0.03": FTConfig(l2sp=0.03, lr=3e-4),
    "ft_birdnet_l0.003": FTConfig(l2sp=0.003, lr=3e-4),
}
METHOD_LABELS = {
    "birdnet": "BirdNET V2.4 (sin ajuste)",
    "ft_birdnet": "BirdNET afinado (L2-SP, $\\lambda$=0,3)",
    "ft_birdnet_l0.03": "BirdNET afinado ($\\lambda$=0,03)",
    "ft_birdnet_l0.003": "BirdNET afinado ($\\lambda$=0,003)",
    "prototipos": "Prototipos coseno (app actual)",
    "lineal": "Sonda lineal",
    "lineal_l2": "Sonda lineal (L2)",
    "mlp": "MLP 256",
    "mlp_l2": "MLP 256 (L2)",
    "mlp_l2_mixup": "MLP 256 (L2 + mixup)",
}
MAIN_METHODS = ["birdnet", "ft_birdnet", "prototipos", "lineal_l2", "mlp_l2"]


# --------------------------------------------------------------------------- #
# Particiones agrupadas
# --------------------------------------------------------------------------- #
def group_folds(groups: np.ndarray, n_splits: int = 5, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Asigna sesiones a pliegues equilibrando el numero de ventanas."""
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(uniq)), -counts))
    fold_of_group: dict[str, int] = {}
    load = np.zeros(n_splits)
    for gi in order:
        f = int(np.argmin(load))
        fold_of_group[uniq[gi]] = f
        load[f] += counts[gi]
    return np.array([fold_of_group[g] for g in groups])


def inner_split(groups: np.ndarray, frac: float = 0.15, seed: int = config.RANDOM_SEED) -> np.ndarray:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * len(uniq))))
    val_groups = set(rng.choice(uniq, size=n_val, replace=False))
    return np.array([g in val_groups for g in groups])


def birdnet_dense_for(ds: D.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if not DENSE_NPZ.exists():
        from .embeddings import birdnet_classifier_weights

        W, b, labels = birdnet_classifier_weights()
        np.savez_compressed(DENSE_NPZ, W=W, b=b, labels=np.array(labels))
    dense = np.load(DENSE_NPZ)
    labels = [str(x) for x in dense["labels"]]
    cols = [labels.index(lab) for lab in ds.birdnet_labels]
    return dense["W"][:, cols].astype(np.float32), dense["b"][cols].astype(np.float32)


# --------------------------------------------------------------------------- #
# Metodos
# --------------------------------------------------------------------------- #
def fit_predict(method: str, ds: D.Dataset, tr: np.ndarray, te: np.ndarray, *, ignore: np.ndarray | None = None):
    if method == "birdnet":
        return flat_sigmoid(ds.birdnet_logit[te]), None
    if method == "prototipos":
        Ytr = ds.Y[tr].copy()
        if ignore is not None:
            Ytr[ignore[tr]] = 0.0
        head = PrototypeHead().fit(ds.X[tr], Ytr)
        return head.decision_function(ds.X[te]), head
    is_val = inner_split(ds.groups[tr])
    fit_idx, val_idx = tr[~is_val], tr[is_val]
    if method in FINETUNE:
        W0, b0 = birdnet_dense_for(ds)
        head = BirdNETFineTuneHead(W0, b0, FTConfig(**vars(FINETUNE[method])))
    else:
        head = MLPHead(HeadConfig(**vars(SCRATCH[method])))
    head.fit(
        ds.X[fit_idx],
        ds.Y[fit_idx],
        X_val=ds.X[val_idx],
        Y_val=ds.Y[val_idx],
        ignore=None if ignore is None else ignore[fit_idx],
    )
    return head.predict_proba(ds.X[te]), head


def cross_validate(ds: D.Dataset, methods: list[str], *, n_splits: int = 5, ignore: np.ndarray | None = None, verbose: bool = True):
    folds = group_folds(ds.groups, n_splits)
    oof = {m: np.zeros_like(ds.Y) for m in methods}
    epochs = {m: [] for m in methods}
    for f in range(n_splits):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        for m in methods:
            t0 = time.time()
            scores, head = fit_predict(m, ds, tr, te, ignore=ignore)
            oof[m][te] = scores
            if hasattr(head, "best_epoch"):
                epochs[m].append(int(head.best_epoch))
            if verbose:
                print(f"  pliegue {f + 1}/{n_splits} {m:18s} {time.time() - t0:5.1f} s", flush=True)
    return oof, folds, epochs


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
def species_metrics(y: np.ndarray, s: np.ndarray, keep: np.ndarray | None = None) -> dict:
    if keep is not None:
        y, s = y[keep], s[keep]
    n_pos = int(y.sum())
    out = {"n_pos": n_pos, "n": int(len(y))}
    if n_pos == 0 or n_pos == len(y):
        return out | {"ap": np.nan, "auc": np.nan, "f1": np.nan, "thr": np.nan, "prec": np.nan, "rec": np.nan, "rec_at_p90": np.nan}
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    rec_at_p90 = float(np.max(rec[prec >= 0.9])) if np.any(prec >= 0.9) else 0.0
    return out | {
        "ap": float(average_precision_score(y, s)),
        "auc": float(roc_auc_score(y, s)),
        "f1": float(f1[best]),
        "thr": float(thr[best]) if len(thr) else np.nan,
        "prec": float(prec[best]),
        "rec": float(rec[best]),
        "rec_at_p90": rec_at_p90,
    }


def keep_mask(ds: D.Dataset, k: int, regime: str) -> np.ndarray | None:
    if regime == "estricto":
        return None
    if regime == "enmascarado":
        return ~ds.uncertain[:, k]
    if regime == "candidatos":
        return flat_sigmoid(ds.birdnet_logit[:, k]) >= CANDIDATE_THRESHOLD
    raise ValueError(regime)


def metrics_table(ds: D.Dataset, oof: dict[str, np.ndarray], regime: str) -> pd.DataFrame:
    rows = []
    for m, scores in oof.items():
        for k, sp in enumerate(ds.species):
            rows.append({"method": m, "species": sp, "regime": regime} | species_metrics(ds.Y[:, k], scores[:, k], keep_mask(ds, k, regime)))
    return pd.DataFrame(rows)


def summarize_methods(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, grp in table.groupby("method", sort=False):
        rows.append(
            {
                "method": m,
                "label": METHOD_LABELS.get(m, m),
                "macro_ap": grp["ap"].mean(),
                "median_ap": grp["ap"].median(),
                "macro_auc": grp["auc"].mean(),
                "macro_f1": grp["f1"].mean(),
                "macro_rec_at_p90": grp["rec_at_p90"].mean(),
                "species_evaluated": int(grp["ap"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def session_bootstrap(ds: D.Dataset, oof: dict[str, np.ndarray], a: str, b: str, *, reps: int = 500, seed: int = 0) -> dict:
    """IC 95 % de mAP(a) - mAP(b) remuestreando sesiones con reemplazo (regimen estricto)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(ds.groups)
    idx_by = {g: np.where(ds.groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(reps):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        y = ds.Y[idx]
        aps_a, aps_b = [], []
        for k in range(y.shape[1]):
            if 0 < y[:, k].sum() < len(y):
                aps_a.append(average_precision_score(y[:, k], oof[a][idx, k]))
                aps_b.append(average_precision_score(y[:, k], oof[b][idx, k]))
        deltas.append(np.mean(aps_a) - np.mean(aps_b))
    deltas = np.array(deltas)
    return {
        "a": a,
        "b": b,
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_delta_le_0": float((deltas <= 0).mean()),
        "reps": reps,
    }


def leave_one_device_out(ds: D.Dataset, methods: list[str], *, min_pos: int = 3) -> pd.DataFrame:
    rows = []
    dev_arr = ds.meta["device"].to_numpy()
    for dev in sorted(np.unique(dev_arr)):
        te, tr = np.where(dev_arr == dev)[0], np.where(dev_arr != dev)[0]
        for m in methods:
            scores, _ = fit_predict(m, ds, tr, te)
            aps = [
                average_precision_score(ds.Y[te, k], scores[:, k])
                for k in range(len(ds.species))
                if min_pos <= ds.Y[te, k].sum() < len(te) and ds.Y[tr, k].sum() > 0
            ]
            rows.append({"device": dev, "method": m, "macro_ap": float(np.mean(aps)) if aps else np.nan, "species_evaluated": len(aps), "windows": len(te)})
        print(f"  LODO {dev} listo", flush=True)
    return pd.DataFrame(rows)


def learning_curve(ds: D.Dataset, methods: list[str], ks=(3, 5, 10, 20, 40), seed: int = 0) -> pd.DataFrame:
    """mAP (estricto) en funcion del maximo de positivos por especie usados al entrenar."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        ignore = np.zeros_like(ds.Y, dtype=bool)
        for s in range(ds.Y.shape[1]):
            pos = np.where(ds.Y[:, s] > 0)[0]
            if len(pos) > k:
                drop = rng.choice(pos, size=len(pos) - k, replace=False)
                ignore[drop, s] = True
        oof, _, _ = cross_validate(ds, methods, ignore=ignore, verbose=False)
        for m in methods:
            t = metrics_table(ds, {m: oof[m]}, "estricto")
            rows.append({"k": k, "method": m, "macro_ap": float(t["ap"].mean())})
        print(f"  curva de aprendizaje k={k} lista", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Calibracion de BirdNET y especies no locales
# --------------------------------------------------------------------------- #
def birdnet_calibration(ds: D.Dataset, grid=(0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
    """Precision observada de BirdNET (fraccion validada) por especie y umbral."""
    sig = flat_sigmoid(ds.birdnet_logit)
    rows = []
    for k, sp in enumerate(ds.species):
        row = {"species": sp, "n_pos": int(ds.Y[:, k].sum())}
        for t in grid:
            det = sig[:, k] >= t
            row[f"n@{t:g}"] = int(det.sum())
            row[f"prec@{t:g}"] = float(ds.Y[det, k].mean()) if det.any() else np.nan
            row[f"rec@{t:g}"] = float((det & (ds.Y[:, k] > 0)).sum() / max(ds.Y[:, k].sum(), 1))
        # umbral minimo que alcanza precision >= 0.9 / 0.95 (maxima exhaustividad)
        for target in (0.9, 0.95):
            prec, rec, thr = precision_recall_curve(ds.Y[:, k], sig[:, k])
            ok = np.where(prec[:-1] >= target)[0]
            row[f"thr_p{int(target * 100)}"] = float(thr[ok[0]]) if len(ok) else np.nan
            row[f"rec_p{int(target * 100)}"] = float(rec[ok[0]]) if len(ok) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


REVIEW_THRESHOLD = 0.25


def review_protocol_evidence(calib: pd.DataFrame) -> dict:
    """Evidencia del umbral de revision usado por el equipo.

    Si todas las detecciones >= t fueron revisadas, la fraccion validada debe ser
    ~1 por encima de t y caer por debajo (detecciones nunca revisadas).
    """
    med = {f"median_prec@{t}": float(calib[f"prec@{t}"].median()) for t in ("0.1", "0.25", "0.5")}
    capped = calib.loc[calib["prec@0.5"] < 0.8, "species"].tolist()
    return med | {"species_with_partial_review": capped}


def tablafinal_calibration(det: pd.DataFrame, species: list[str], targets=(0.9, 0.95), floor: float = REVIEW_THRESHOLD) -> pd.DataFrame:
    """Umbrales por especie con el metodo de Wood & Kahl (2024).

    Se usan solo detecciones revisadas por expertos (TablaFinal: Valid 0/1). Si
    la especie tiene >= 2 rechazos, se ajusta una regresion logistica
    P(valida | logit(score)) y se toma el score minimo con P >= objetivo. Si no
    hay rechazos registrados, la precision observada es 1 y el umbral es el de
    revision (no hay evidencia por debajo de el). El umbral nunca baja de
    ``floor``.
    """
    from sklearn.linear_model import LogisticRegression

    rows = []
    for sp in species:
        d = det[det["species"] == sp]
        conf = d["birdnet_conf"].clip(1e-4, 1 - 1e-4).to_numpy()
        valid = d["valid"].to_numpy()
        n_inv = int((valid == 0).sum())
        row = {"species": sp, "reviewed": int(len(d)), "rejected": n_inv, "precision_reviewed": float(valid.mean()) if len(d) else np.nan}
        if n_inv >= 2 and n_inv < len(d):
            x = np.log(conf / (1 - conf)).reshape(-1, 1)
            lr = LogisticRegression(C=10.0).fit(x, valid)
            grid = np.linspace(0.01, 0.99, 197)
            pg = lr.predict_proba(np.log(grid / (1 - grid)).reshape(-1, 1))[:, 1]
            row["method"] = "logistica"
            for t in targets:
                ok = np.where(pg >= t)[0]
                row[f"thr_p{int(t * 100)}"] = float(max(grid[ok[0]], floor)) if len(ok) else np.nan
        else:
            row["method"] = "sin rechazos" if n_inv == 0 else "pocos rechazos"
            for t in targets:
                row[f"thr_p{int(t * 100)}"] = floor
        rows.append(row)
    return pd.DataFrame(rows)


def nonlocal_detections(ds: D.Dataset, local_species: set[str], thresholds=(0.25, 0.5, 0.75)) -> dict:
    """Cuantas detecciones de BirdNET global corresponden a especies ajenas a la lista local."""
    data = np.load(config.WORK_DIR / "embeddings_all.npz")
    labels = [str(x) for x in data["labels"]]
    keep_rows = set(zip(ds.meta["file_key"], ds.meta["start"]))
    mask = np.array([(fk, round(float(st), 2)) in keep_rows for fk, st in zip(data["file_key"].astype(str), data["start"])])
    sig = flat_sigmoid(data["logits"][mask].astype(np.float32))
    local_set = {D.BIRDNET_SYNONYMS.get(s, s) for s in local_species}
    is_local = np.array([lab.split("_")[0] in local_set for lab in labels])
    out = {"windows": int(mask.sum()), "local_species_in_list": int(is_local.sum())}
    for t in thresholds:
        det = sig >= t
        out[f"det_total@{t:g}"] = int(det.sum())
        out[f"det_nonlocal@{t:g}"] = int(det[:, ~is_local].sum())
        out[f"windows_with_nonlocal@{t:g}"] = int(det[:, ~is_local].any(axis=1).sum())
    top = pd.Series(np.array(labels)[~is_local][np.argsort(-(sig[:, ~is_local] >= 0.5).sum(axis=0))[:10]])
    counts = (sig[:, ~is_local] >= 0.5).sum(axis=0)
    order = np.argsort(-counts)[:10]
    out["top_nonlocal@0.5"] = [{"label": str(np.array(labels)[~is_local][i]), "detections": int(counts[i])} for i in order if counts[i] > 0]
    del top
    return out


# --------------------------------------------------------------------------- #
# Exportacion del modelo local para la app
# --------------------------------------------------------------------------- #
def export_local_model(ds: D.Dataset, calib: pd.DataFrame, cv_strict: pd.DataFrame, chosen: str, head, summary: dict, tf_calib: pd.DataFrame | None = None) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    info = ds.species_info
    use_custom_weights = chosen != "birdnet" and head is not None
    if use_custom_weights:
        np.savez_compressed(config.MODELS_DIR / "condesan_local.npz", W=head.W.astype(np.float32), b=head.b.astype(np.float32))
    ap = cv_strict[cv_strict["method"] == chosen].set_index("species")["ap"]
    species = []
    tfc = tf_calib.set_index("species") if tf_calib is not None else None
    for k, sp in enumerate(ds.species):
        c = calib.iloc[k]
        if tfc is not None:
            thr = float(tfc.at[sp, "thr_p90"]) if np.isfinite(tfc.at[sp, "thr_p90"]) else 0.5
            thr95 = float(tfc.at[sp, "thr_p95"]) if np.isfinite(tfc.at[sp, "thr_p95"]) else None
            how = str(tfc.at[sp, "method"])
        else:
            thr = float(c["thr_p90"]) if np.isfinite(c["thr_p90"]) else 0.5
            thr95 = None if not np.isfinite(c["thr_p95"]) else float(c["thr_p95"])
            how = "subconjunto"
        species.append(
            {
                "scientific": sp,
                "common_en": str(info.at[sp, "common_en"]),
                "common_es": str(info.at[sp, "common_es"]),
                "birdnet_label": ds.birdnet_labels[k],
                "threshold": float(np.clip(thr, 0.1, 0.95)),
                "threshold_p95": thr95,
                "threshold_method": how,
                "train_positives": int(ds.Y[:, k].sum()),
                "cv_ap": None if np.isnan(ap.get(sp, np.nan)) else float(ap.get(sp)),
            }
        )
    meta = {
        "name": "EcoAcoustic Sentinel - clasificador local CONDESAN",
        "version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": "condesan_local.npz" if use_custom_weights else "birdnet_original",
        "selected_method": chosen,
        "selection_rule": "metodo con mayor mAP estricto en validacion cruzada agrupada por sesion",
        "embedding": {
            "model": "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
            "layer": "GLOBAL_AVG_POOL (1024) -> CLASS_DENSE_LAYER",
            "input": "embedding crudo (sin normalizar)",
            "window_s": config.SIG_LENGTH,
            "sample_rate": config.SAMPLE_RATE,
        },
        "threshold_rule": (
            "Wood & Kahl (2024): regresion logistica P(valida | logit(score)) sobre las detecciones revisadas "
            "de TablaFinal; umbral = score minimo con P >= 0,90, nunca inferior al umbral de revision (0,25)"
        ),
        "species": species,
        "training_data": summary,
        "license_note": "Basado en BirdNET V2.4 (Cornell Lab / TU Chemnitz, CC BY-NC-SA 4.0): uso no comercial, atribucion y compartir igual.",
    }
    (config.MODELS_DIR / "condesan_local.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
RAPID_METHODS = ["birdnet", "ft_birdnet", "prototipos", "lineal_l2", "mlp_l2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validacion cruzada, analisis y exportacion del modelo local.")
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Solo los metodos principales, bootstrap de 200 replicas y curva de aprendizaje con k = 3, 10, 40 "
        "(recomendado para el conjunto completo en Colab).",
    )
    args = parser.parse_args()
    config.ensure_dirs()
    t0 = time.time()
    from . import labels as L

    det = L.load_detections()
    ds = D.build_dataset(det=det)
    summary = D.summarize(ds)
    print(json.dumps(summary, indent=2))
    (config.RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ds.species_info.to_csv(config.RESULTS_DIR / "species_info.csv")

    methods = RAPID_METHODS if args.rapido else ["birdnet", *FINETUNE.keys(), "prototipos", *SCRATCH.keys()]
    reps = 200 if args.rapido else 500
    ks = (3, 10, 40) if args.rapido else (3, 5, 10, 20, 40)
    print("Validacion cruzada agrupada por sesion...")
    oof, folds, epochs = cross_validate(ds, methods)
    tables = {r: metrics_table(ds, oof, r) for r in ("estricto", "enmascarado", "candidatos")}
    pd.concat(tables.values()).to_csv(config.RESULTS_DIR / "cv_metrics.csv", index=False)
    summaries = []
    for r, t in tables.items():
        s = summarize_methods(t)
        s.insert(1, "regime", r)
        summaries.append(s)
    summ = pd.concat(summaries)
    summ.to_csv(config.RESULTS_DIR / "cv_summary.csv", index=False)
    print(summ.pivot(index="method", columns="regime", values="macro_ap").round(4).to_string())

    strict = summ[summ["regime"] == "estricto"].set_index("method")["macro_ap"]
    deployable = [m for m in ["birdnet", *FINETUNE.keys()] if m in strict.index]
    chosen = max(deployable, key=lambda m: strict[m])
    best_scratch = max([m for m in SCRATCH.keys() if m in strict.index], key=lambda m: strict[m])
    print(f"Modelo local elegido: {chosen}; mejor cabeza desde cero: {best_scratch}")

    print("Ablacion: ignorar pares inciertos al entrenar...")
    oof_ign, _, _ = cross_validate(ds, ["ft_birdnet", best_scratch], ignore=ds.uncertain, verbose=False)
    abl = summarize_methods(metrics_table(ds, oof_ign, "estricto"))
    abl["method"] = abl["method"] + "_ignora_inciertos"
    abl.to_csv(config.RESULTS_DIR / "ablation_ignore_uncertain.csv", index=False)

    print("Bootstrap pareado por sesiones...")
    boot = [
        session_bootstrap(ds, oof, "ft_birdnet", "birdnet", reps=reps),
        session_bootstrap(ds, oof, best_scratch, "prototipos", reps=reps),
        session_bootstrap(ds, oof, "birdnet", best_scratch, reps=reps),
    ]
    (config.RESULTS_DIR / "bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    print("Validacion dejando una grabadora fuera (LODO)...")
    leave_one_device_out(ds, ["birdnet", "ft_birdnet", "prototipos", best_scratch]).to_csv(config.RESULTS_DIR / "lodo.csv", index=False)

    print("Curva de aprendizaje...")
    learning_curve(ds, ["prototipos", best_scratch, "ft_birdnet"], ks=ks).to_csv(config.RESULTS_DIR / "learning_curve.csv", index=False)

    print("Calibracion de BirdNET y especies no locales...")
    calib = birdnet_calibration(ds)
    calib.to_csv(config.RESULTS_DIR / "birdnet_calibration.csv", index=False)
    (config.RESULTS_DIR / "review_protocol.json").write_text(json.dumps(review_protocol_evidence(calib), indent=2), encoding="utf-8")
    tf_calib = tablafinal_calibration(det[~det["session"].isin(set(pd.read_csv(config.RESULTS_DIR / "qc_sessions.csv").query("excluded")["session"]))], ds.species)
    tf_calib.to_csv(config.RESULTS_DIR / "thresholds_tablafinal.csv", index=False)
    nonlocal_info = nonlocal_detections(ds, set(det["species"].unique()))
    (config.RESULTS_DIR / "nonlocal_detections.json").write_text(json.dumps(nonlocal_info, indent=2), encoding="utf-8")

    print(f"Exportando modelo local ({chosen})...")
    head = None
    if chosen != "birdnet":
        _, head = fit_predict(chosen, ds, np.arange(len(ds.X)), np.arange(1))
    export_local_model(ds, calib, tables["estricto"], chosen, head, summary, tf_calib)

    np.savez_compressed(
        OOF_NPZ,
        folds=folds,
        Y=ds.Y,
        uncertain=ds.uncertain,
        species=np.array(ds.species),
        file_key=np.asarray(ds.meta["file_key"].astype(str).tolist(), dtype=str),
        start=ds.meta["start"].to_numpy(dtype=float),
        session=np.asarray(ds.meta["session"].astype(str).tolist(), dtype=str),
        **{f"oof_{m}": v for m, v in oof.items()},
    )
    run = {
        "chosen_local_model": chosen,
        "best_scratch_head": best_scratch,
        "cv_best_epochs": {m: e for m, e in epochs.items() if e},
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "mode": "rapido" if args.rapido else "completo",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (config.RESULTS_DIR / "run_info.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
