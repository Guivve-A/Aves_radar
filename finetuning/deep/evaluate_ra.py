"""Evaluacion final con el protocolo consciente de la revision (``protocol``).

Para cada modelo, sobre predicciones fuera de muestra y con bootstrap de sesiones dentro de cada
grabadora, pareado contra BirdNET:

    M1  mAP limpia (control de no inferioridad) y, por continuidad, mAP estricta y enmascarada
    SLI indice de fuga de seleccion
    M2  colocacion de los rechazos expertos (AUC estratificada por especie y grabadora)
    M3  exhaustividad sobre cantos atenuados con el mismo volumen de alertas que BirdNET
    D1-D3  reglas de decision fijadas de antemano

Incluye el verificador de baja capacidad (5 variables) para el problema de los rechazos.

Uso:
    python -m finetuning.deep.evaluate_ra
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..embeddings import flat_sigmoid
from . import data as D
from . import protocol as PR
from .train_heads import RESULTS, Mixtures

ORDER = ["birdnet", "verificador", "lineal", "mlp", "mlp_sin_mezclas", "mlp_estricto", "mlp_semilla2", "post", "b43", "b4", "prototipos", "lineal_v1", "lineal_desde_cero"]
LABELS = {
    "birdnet": "BirdNET V2.4 (sin ajuste)",
    "verificador": "BirdNET + verificador (5 variables)",
    "lineal": "Correccion lineal sobre el embedding",
    "mlp": "Correccion MLP residual (modelo principal)",
    "mlp_sin_mezclas": "Ablacion: MLP sin mezclas atenuadas",
    "mlp_estricto": "Ablacion: MLP con etiquetas estrictas",
    "mlp_semilla2": "MLP, otra semilla",
    "post": "Capas convolucionales: POST_CONV",
    "b43": "Capas convolucionales: sub-bloques 4-3 y 4-4 + POST_CONV",
    "b4": "Capas convolucionales: bloque 4 completo",
    "prototipos": "Desde cero: prototipos coseno (app)",
    "lineal_v1": "Desde cero: sonda lineal, receta anterior",
    "lineal_desde_cero": "Desde cero: sonda lineal, receta nueva",
}


def verifier_scores(data: D.DeepData, cat: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, dict]:
    """Regresion logistica P(valida | revisada) comun a todas las especies, con 5 variables derivadas del audio.

    f1 logit de BirdNET; f2 margen sobre la mejor clase competidora de las 6 522; f3 maximo logit de la misma
    especie en otras ventanas de la grabacion; f4 log de la norma del embedding; f5 numero de especies objetivo
    con BirdNET >= 0,25 en la ventana. Se ajusta con los pares revisados (validados y rechazados) de los
    pliegues de entrenamiento y se aplica al pliegue restante.
    """
    from sklearn.linear_model import LogisticRegression

    rows, cols = np.nonzero((cat == PR.P) | (cat == PR.H))
    y = (cat[rows, cols] == PR.P).astype(int)
    z0 = data.logit[rows, cols].astype(np.float64)
    all_logits = data.logits_all
    order = np.argsort(rows, kind="stable")
    comp = np.empty(len(rows))
    class_index = np.asarray(data.class_index)
    for a in range(0, len(order), 512):
        idx = order[a : a + 512]
        block = np.asarray(all_logits[rows[idx]], dtype=np.float32)
        own = class_index[cols[idx]]
        block[np.arange(len(idx)), own] = -np.inf
        comp[idx] = block.max(axis=1)
    f2 = z0 - comp
    rec = data.meta["rec"].to_numpy()
    f3 = np.full(len(rows), -20.0)
    by_rec = pd.Series(np.arange(len(rec))).groupby(rec).apply(lambda s: s.to_numpy())
    for i, (r, k) in enumerate(zip(rows, cols)):
        others = by_rec[rec[r]]
        others = others[others != r]
        if len(others):
            f3[i] = float(data.logit[others, k].max())
    f4 = np.log(np.linalg.norm(data.emb[rows], axis=1) + 1e-9)
    f5 = (flat_sigmoid(data.logit[rows]) >= PR.GATE).sum(axis=1).astype(np.float64)
    Xf = np.c_[z0, f2, f3, f4, f5]
    prob = flat_sigmoid(data.logit).astype(np.float32).copy()
    coefs = []
    for f in sorted(np.unique(folds)):
        tr, te = folds[rows] != f, folds[rows] == f
        mu, sd = Xf[tr].mean(axis=0), Xf[tr].std(axis=0) + 1e-9
        n_pos, n_neg = int(y[tr].sum()), int((1 - y[tr]).sum())
        w = np.where(y[tr] == 1, 0.8 / max(n_pos, 1), 0.2 / max(n_neg, 1)) * len(y[tr])   # los rechazos pesan el 20 %
        clf = LogisticRegression(C=1.0, max_iter=1000).fit((Xf[tr] - mu) / sd, y[tr], sample_weight=w)
        prob[rows[te], cols[te]] = clf.predict_proba((Xf[te] - mu) / sd)[:, 1]
        coefs.append(clf.coef_[0].tolist())
    return prob, {"features": ["logit_birdnet", "margen_competidor", "max_misma_especie_otras_ventanas", "log_norma_embedding", "especies_sobre_0.25"], "mean_coef": np.mean(coefs, axis=0).round(3).tolist()}


def collect(data: D.DeepData, mix: Mixtures):
    """Puntuaciones fuera de muestra: {nombre: (reales, mezclas o None)}."""
    all_test = np.flatnonzero(np.isin(mix.kind, ("test", "bg_test")))
    bn_mix = np.full(mix.logit.shape, np.nan, np.float32)
    bn_mix[all_test] = flat_sigmoid(mix.logit[all_test])
    scores = {"birdnet": (flat_sigmoid(data.logit).astype(np.float32), bn_mix)}
    path = D.DEEP_DIR / "oof_ra.npz"
    if path.exists():
        z = np.load(path)
        for key in z.files:
            if key.startswith("oof_"):
                name = key[4:]
                m = z[f"mix_{name}"] if f"mix_{name}" in z.files else None
                scores[name] = (z[key], m)
    path = D.DEEP_DIR / "oof_scratch.npz"
    if path.exists():
        z = np.load(path)
        rename = {"lineal": "lineal_desde_cero"}
        for key in z.files:
            if key.startswith("oof_") and key[4:] != "birdnet":
                scores[rename.get(key[4:], key[4:])] = (z[key], None)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = D.load(maps=True)
    cat = PR.categories(data)
    folds = np.load(D.DEEP_DIR / "folds.npy")
    mix = Mixtures(data)
    rs = PR.Resampler(data, reps=args.reps)
    apev = PR.APEvaluator(data, cat, rs)
    rej = PR.RejectionPlacement(data, cat, rs)
    att = PR.AttenuationBenchmark(data, cat, mix, rs)
    scores = collect(data, mix)
    vprob, vinfo = verifier_scores(data, cat, folds)
    scores["verificador"] = (vprob, None)
    names = [m for m in ORDER if m in scores] + [m for m in scores if m not in ORDER]

    res: dict[str, dict] = {}
    boots: dict[str, dict] = {}
    species_rows, m3_rows = [], []
    for m in names:
        clean, mixed = scores[m]
        r: dict = {"label": LABELS.get(m, m)}
        b: dict = {}
        for regime in ("limpio", "estricto", "enmascarado"):
            point, aps, reps = apev.evaluate(clean, regime)
            r[f"map_{regime}"] = point
            b[f"map_{regime}"] = (point, reps)
            for sp, ap in zip(data.species, aps):
                species_rows.append({"method": m, "species": sp, "regime": regime, "ap": ap})
        r["sli"] = PR.selection_leak_index(data, cat, clean)["sli"] if m != "birdnet" else 0.5
        point, _, reps = rej.evaluate(clean)
        r["m2"] = point
        b["m2"] = (point, reps)
        r["m2_removed95"] = rej.removed_at_retention(clean)["removed"]
        if mixed is not None:
            for mode in ("presupuesto", "falsas_alarmas"):
                curve, s, reps, table = att.evaluate(clean, mixed, mode)
                r[f"m3_{mode}"] = {f"{a:g}": float(c) for a, c in zip(PR.LEVELS, curve)}
                r[f"m3_{mode}_S"] = s
                b[f"m3_{mode}"] = (curve, reps)
                if mode == "presupuesto":
                    m3_rows.append(table.assign(method=m))
        res[m], boots[m] = r, b
        print(f"  {m:18s} limpia {r['map_limpio']:.4f} estricta {r['map_estricto']:.4f} enmasc. {r['map_enmascarado']:.4f} SLI {r['sli']:.3f} M2 {r['m2']:.3f} M3 {r.get('m3_presupuesto_S', float('nan')):.4f}", flush=True)

    base = boots["birdnet"]
    for m in names:
        r, b = res[m], boots[m]
        for key in ("map_limpio", "map_estricto", "map_enmascarado", "m2"):
            r[f"{key}_ci"] = PR.ci(b[key][1], b[key][0])
            if m != "birdnet":
                r[f"{key}_delta"] = PR.paired(b[key][1], base[key][1], b[key][0], base[key][0])
        for mode in ("presupuesto", "falsas_alarmas"):
            key = f"m3_{mode}"
            if key not in b:
                continue
            curve, reps = b[key]
            r[f"{key}_ci"] = {f"{a:g}": PR.ci(reps[:, j], curve[j]) for j, a in enumerate(PR.LEVELS)}
            s_reps = np.nanmean(reps[:, 1:], axis=1)
            r[f"{key}_S_ci"] = PR.ci(s_reps, float(np.nanmean(curve[1:])))
            if m != "birdnet":
                bc, br = base[key]
                r[f"{key}_S_delta"] = PR.paired(s_reps, np.nanmean(br[:, 1:], axis=1), float(np.nanmean(curve[1:])), float(np.nanmean(bc[1:])))
                r[f"{key}_delta"] = {f"{a:g}": PR.paired(reps[:, j], br[:, j], curve[j], bc[j]) for j, a in enumerate(PR.LEVELS)}
        if m != "birdnet":
            r["D1_seguro"] = bool(r["map_limpio_delta"]["ci_low"] > -PR.NONINFERIORITY)
            r["D2_rechazos"] = bool(r["m2_delta"]["ci_low"] > 0)
            r["D3_atenuados"] = bool(r.get("m3_presupuesto_S_delta", {}).get("ci_low", -1) > 0) if "m3_presupuesto" in b else None

    # comparaciones pareadas entre escalones (misma semilla, pliegues y banco de mezclas)
    pairs = [("mlp", "lineal"), ("mlp", "mlp_sin_mezclas"), ("mlp", "mlp_estricto"), ("mlp", "mlp_semilla2"), ("post", "mlp"), ("b43", "post"), ("b4", "b43"), ("b43", "mlp"), ("b4", "mlp")]
    ladder = []
    for a, bname in pairs:
        if a in boots and bname in boots and "m3_presupuesto" in boots[a] and "m3_presupuesto" in boots[bname]:
            ca, ra = boots[a]["m3_presupuesto"]
            cb, rb = boots[bname]["m3_presupuesto"]
            ladder.append({"a": a, "b": bname, "m3_S": PR.paired(np.nanmean(ra[:, 1:], axis=1), np.nanmean(rb[:, 1:], axis=1), float(np.nanmean(ca[1:])), float(np.nanmean(cb[1:]))), "m2": PR.paired(boots[a]["m2"][1], boots[bname]["m2"][1], boots[a]["m2"][0], boots[bname]["m2"][0]), "map_limpio": PR.paired(boots[a]["map_limpio"][1], boots[bname]["map_limpio"][1], boots[a]["map_limpio"][0], boots[bname]["map_limpio"][0])})

    cells = PR.review_cells(data)
    info = dict(data.info)
    info |= {
        "categories": PR.category_counts(data, cat),
        "review_cells": cells.status.value_counts().to_dict(),
        "unlisted_by_cell": {s: float(cells.loc[cells.status == s, "detections"].sum() - cells.loc[cells.status == s, "listed"].sum()) for s in ("revisada", "nunca", "parcial")},
        "folds": [{"fold": int(f), "windows": int((folds == f).sum()), "sessions": int(len(set(data.groups[folds == f]))), "rejections": int((cat[folds == f] == PR.H).sum()), "positives": int((cat[folds == f] == PR.P).sum())} for f in sorted(np.unique(folds))],
        "m3_species": int(len(att.species)),
        "m2_rejections": int(rej.n),
        "mixtures": {k: int(v) for k, v in pd.Series(mix.kind).value_counts().items()},
        "verifier": vinfo,
        "bootstrap_reps": int(args.reps),
    }
    (RESULTS / "eval.json").write_text(json.dumps({"methods": res, "ladder": ladder, "dataset": info}, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(species_rows).to_csv(RESULTS / "eval_species.csv", index=False)
    if m3_rows:
        pd.concat(m3_rows).to_csv(RESULTS / "eval_m3_species.csv", index=False)
    flat = []
    for m in names:
        r = res[m]
        flat.append(
            {
                "method": m,
                "label": r["label"],
                "map_limpio": r["map_limpio"],
                "map_limpio_delta": r.get("map_limpio_delta", {}).get("delta"),
                "map_limpio_delta_lo": r.get("map_limpio_delta", {}).get("ci_low"),
                "map_limpio_delta_hi": r.get("map_limpio_delta", {}).get("ci_high"),
                "map_estricto": r["map_estricto"],
                "map_enmascarado": r["map_enmascarado"],
                "sli": r["sli"],
                "m2": r["m2"],
                "m2_lo": r["m2_ci"]["ci_low"],
                "m2_hi": r["m2_ci"]["ci_high"],
                "m2_delta": r.get("m2_delta", {}).get("delta"),
                "m2_delta_lo": r.get("m2_delta", {}).get("ci_low"),
                "m2_delta_hi": r.get("m2_delta", {}).get("ci_high"),
                "m2_removed95": r["m2_removed95"],
                "m3_S": r.get("m3_presupuesto_S"),
                "m3_S_delta": r.get("m3_presupuesto_S_delta", {}).get("delta"),
                "m3_S_delta_lo": r.get("m3_presupuesto_S_delta", {}).get("ci_low"),
                "m3_S_delta_hi": r.get("m3_presupuesto_S_delta", {}).get("ci_high"),
                **{f"m3_{a:g}dB": r.get("m3_presupuesto", {}).get(f"{a:g}") for a in PR.LEVELS},
                "m3_fa_S": r.get("m3_falsas_alarmas_S"),
                "D1": r.get("D1_seguro"),
                "D2": r.get("D2_rechazos"),
                "D3": r.get("D3_atenuados"),
            }
        )
    table = pd.DataFrame(flat)
    table.to_csv(RESULTS / "eval_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[["method", "map_limpio", "map_limpio_delta_lo", "sli", "m2", "m2_delta", "m2_delta_lo", "m2_delta_hi", "m3_S", "m3_S_delta", "m3_S_delta_lo", "m3_S_delta_hi", "D1", "D2", "D3"]].round(4).to_string(index=False))
        print(table[["method"] + [f"m3_{a:g}dB" for a in PR.LEVELS] + ["m3_fa_S", "m2_removed95"]].round(3).to_string(index=False))
    for row in ladder:
        print(f"  {row['a']} - {row['b']}: M3 {row['m3_S']['delta']:+.4f} [{row['m3_S']['ci_low']:+.4f}; {row['m3_S']['ci_high']:+.4f}]  M2 {row['m2']['delta']:+.3f} [{row['m2']['ci_low']:+.3f}; {row['m2']['ci_high']:+.3f}]")
    print(f"Evaluacion lista en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
