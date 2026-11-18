"""Evaluacion conjunta, tablas LaTeX y figuras de los experimentos de aprendizaje profundo.

Reune las predicciones fuera de muestra de todas las corridas (cabezas, cabezas con
mezclas y ajuste fino de capas convolucionales), las evalua con el mismo protocolo y
escribe solo resultados agregados (metricas, intervalos, curvas): nada de audio ni de
detecciones individuales.

Uso:
    python -m finetuning.deep.report            # evaluacion + tablas + figuras
    python -m finetuning.deep.report --no-eval  # solo tablas y figuras a partir de los CSV
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from ..embeddings import flat_sigmoid
from ..report import GRID, INK, INK_2, MUTED  # mismo estilo que el resto del informe
from . import analysis as A
from . import data as D
from . import metrics as M
from .train_heads import LABELS, RESULTS, Mixtures, evaluate

TAB = config.TABLES_DIR
FIG = config.FIGURES_DIR

# Orden fijo de presentacion y nombres cortos.
ORDER = [
    "birdnet",
    "prototipos",
    "lineal_v1",
    "lineal",
    "mlp",
    "lineal_mix",
    "ft_v1",
    "residual_estricto",
    "residual",
    "contexto",
    "residual_mix",
    "birdnet_x_lineal",
    "birdnet_x_lineal_mix",
    "ft_bloque4",
    "ft_bloques34",
]
SHORT = {
    "birdnet": "BirdNET V2.4 (sin ajuste)",
    "prototipos": "Prototipos coseno (app)",
    "lineal_v1": "Sonda lineal, receta anterior",
    "lineal": "Sonda lineal, receta nueva",
    "mlp": "MLP 2$\\times$512, receta nueva",
    "lineal_mix": "Sonda lineal + mezclas",
    "ft_v1": "\\'Ultima capa afinada (L2-SP), anterior",
    "residual_estricto": "Adaptador residual (candidatos = negativos)",
    "residual": "Adaptador residual",
    "contexto": "Adaptador residual + contexto (GRU)",
    "residual_mix": "Adaptador residual + mezclas",
    "birdnet_x_lineal": "BirdNET $\\times$ sonda local",
    "birdnet_x_lineal_mix": "BirdNET $\\times$ sonda local con mezclas",
    "ft_bloque4": "Bloque 4 afinado (capas conv.)",
    "ft_bloques34": "Bloques 3 y 4 afinados (capas conv.)",
}
PLAIN = {k: v.replace("$\\times$", "x").replace("\\'U", "Ú") for k, v in SHORT.items()}
FAMILY = {
    "birdnet": "Referencia",
    "prototipos": "Desde cero",
    "lineal_v1": "Desde cero",
    "lineal": "Desde cero",
    "mlp": "Desde cero",
    "lineal_mix": "Desde cero",
    "ft_v1": "Anclados a BirdNET",
    "residual_estricto": "Anclados a BirdNET",
    "residual": "Anclados a BirdNET",
    "contexto": "Anclados a BirdNET",
    "residual_mix": "Anclados a BirdNET",
    "birdnet_x_lineal": "Conjuntos",
    "birdnet_x_lineal_mix": "Conjuntos",
    "ft_bloque4": "Capas convolucionales",
    "ft_bloques34": "Capas convolucionales",
}
# Paleta categorica validada (primeras ranuras) + gris de referencia para BirdNET.
C_REF, C1, C2, C3 = INK_2, "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#ffffff"


def num(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}".replace(".", "{,}")


def signed(x, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.{nd}f}".replace(".", "{,}")


def interval(lo, hi, nd: int = 3) -> str:
    return f"[{signed(lo, nd)}; {signed(hi, nd)}]"


# --------------------------------------------------------------------------- #
# Evaluacion conjunta
# --------------------------------------------------------------------------- #
def collect(data: D.DeepData, mix: Mixtures | None):
    """Predicciones fuera de muestra de todas las corridas disponibles."""
    oof: dict[str, np.ndarray] = {}
    oof_mix: dict[str, np.ndarray] = {}
    for tag in ("heads", "mix"):
        path = D.DEEP_DIR / f"oof_{tag}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        for key in z.files:
            if key.startswith("oof_"):
                oof[key[4:]] = z[key]          # la corrida con mezclas prevalece (mismo modelo, con puntuaciones de mezclas)
            elif key.startswith("mix_"):
                oof_mix[key[4:]] = z[key]
    for path in sorted(D.DEEP_DIR.glob("oof_ft_*.npz")):
        if path.stem.endswith("_partial"):
            continue
        z = np.load(path)
        name = path.stem[4:]
        oof[name] = z["oof"]
        if "mix" in z.files:
            oof_mix[name] = z["mix"]
    if mix is not None and "birdnet" not in oof_mix:
        rows = mix.test_rows(np.arange(len(data.Y)))
        bn = np.full((len(mix.plan), data.Y.shape[1]), np.nan, np.float32)
        bn[rows] = flat_sigmoid(mix.logit[rows])
        oof_mix["birdnet"] = bn
    oof.setdefault("birdnet", flat_sigmoid(data.logit))
    ordered = {m: oof[m] for m in ORDER if m in oof} | {m: v for m, v in oof.items() if m not in ORDER}
    return ordered, {m: oof_mix[m] for m in ordered if m in oof_mix}


def final_evaluation(reps: int = 500) -> None:
    data = D.load(maps=False)
    mix = Mixtures(data) if (D.DEEP_DIR / "mix" / "done.npy").exists() else None
    oof, oof_mix = collect(data, mix)
    for m in oof:
        LABELS.setdefault(m, PLAIN.get(m, m))
    evaluate(data, oof, oof_mix, mix, reps=reps, tag="final")
    summary, detail = A.new_detections(data, oof)
    summary.to_csv(RESULTS / "new_detections_final.csv", index=False)
    detail.to_csv(RESULTS / "new_detections_species_final.csv", index=False)
    info = dict(data.info)
    info["birdnet_weighted_map"] = {r: M.macro_ap(data, oof["birdnet"], r) for r in M.REGIMES}
    info["birdnet_unweighted_map"] = {r: M.macro_ap(data, oof["birdnet"], r, weighted=False) for r in M.REGIMES}
    info["uncertain_pairs"] = int(data.uncertain.sum())
    info["positives_per_species"] = {s: int(v) for s, v in zip(data.species, data.Y.sum(axis=0))}
    (RESULTS / "dataset_deep.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Tablas
# --------------------------------------------------------------------------- #
def _load():
    summ = pd.read_csv(RESULTS / "cv_summary_final.csv")
    boot = json.loads((RESULTS / "bootstrap_final.json").read_text(encoding="utf-8"))
    rev = pd.read_csv(RESULTS / "reviewed_only_final.csv")
    snr = pd.read_csv(RESULTS / "snr_final.csv") if (RESULTS / "snr_final.csv").exists() else None
    return summ, boot, rev, snr


def _delta(boot: list[dict], m: str, regime: str, base: str = "birdnet"):
    for b in boot:
        if b.get("a") == m and b.get("b") == base and b.get("regime") == regime:
            return b
    return None


def table_methods(summ, boot, rev) -> None:
    piv = summ.pivot(index="method", columns="regime", values="macro_ap")
    rec = summ[summ.regime == "estricto"].set_index("method")["macro_rec_at_p90"]
    revs = rev.set_index("method")
    methods = [m for m in ORDER if m in piv.index]
    best = {r: piv.loc[[m for m in methods if m != "birdnet"], r].max() for r in ("estricto", "enmascarado")}
    lines = [
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{mAP ponderada} & \\multicolumn{2}{c}{Diferencia frente a BirdNET (estricta)} & & \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Familia & M\\'etodo & estricta & enmasc. & $\\Delta$ & IC 95\\,\\% & Rec@P90 & AUC revisadas\\\\",
        "\\midrule",
    ]
    last = None
    for m in methods:
        fam = FAMILY.get(m, "")
        d = _delta(boot, m, "estricto")
        cells = [
            fam if fam != last else "",
            SHORT.get(m, m),
            num(piv.at[m, "estricto"]),
            num(piv.at[m, "enmascarado"]),
            "--" if d is None else signed(d["delta_mean"]),
            "--" if d is None else interval(d["ci_low"], d["ci_high"]),
            num(rec.get(m, np.nan)),
            num(revs.at[m, "auc_pooled"], 2) if m in revs.index else "--",
        ]
        if last is not None and fam != last:
            lines.append("\\addlinespace")
        last = fam
        lines.append(" & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_methods.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_reviewed(rev) -> None:
    lines = ["\\begin{tabular}{lcccc}", "\\toprule", "M\\'etodo & AUC global & IC 95\\,\\% & Concordancia por especie & IC 95\\,\\%\\\\", "\\midrule"]
    r = rev.set_index("method")
    for m in [m for m in ORDER if m in r.index]:
        lines.append(
            f"{SHORT.get(m, m)} & {num(r.at[m, 'auc_pooled'], 2)} & [{num(r.at[m, 'auc_pooled_lo'], 2)}; {num(r.at[m, 'auc_pooled_hi'], 2)}] & "
            f"{num(r.at[m, 'concordance'], 2)} & [{num(r.at[m, 'concordance_lo'], 2)}; {num(r.at[m, 'concordance_hi'], 2)}]\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_reviewed.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_snr(snr) -> None:
    if snr is None:
        return
    levels = sorted(snr.atten_db.unique())
    lines = ["\\begin{tabular}{l" + "c" * len(levels) + "cc}", "\\toprule"]
    lines.append(" & \\multicolumn{" + str(len(levels)) + "}{c}{Exhaustividad seg\\'un la atenuaci\\'on del canto} & \\multicolumn{2}{c}{Diferencia a 12\\,dB}\\\\")
    lines.append("\\cmidrule(lr){2-" + str(len(levels) + 1) + "}\\cmidrule(lr){" + str(len(levels) + 2) + "-" + str(len(levels) + 3) + "}")
    lines.append("M\\'etodo & " + " & ".join("original" if a == 0 else f"$-${a:g}\\,dB" for a in levels) + " & $\\Delta$ & IC 95\\,\\%\\\\")
    lines.append("\\midrule")
    for m in [m for m in ORDER if m in set(snr.method)]:
        g = snr[snr.method == m].set_index("atten_db")
        row = [SHORT.get(m, m)] + [num(g.at[a, "recall"]) for a in levels]
        if "delta_recall" in g.columns and np.isfinite(g.at[12.0, "delta_recall"] if 12.0 in g.index else np.nan):
            row += [signed(g.at[12.0, "delta_recall"]), interval(g.at[12.0, "delta_lo"], g.at[12.0, "delta_hi"])]
        else:
            row += ["--", "--"]
        lines.append(" & ".join(row) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_snr.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_new() -> None:
    path = RESULTS / "new_detections_final.csv"
    if not path.exists():
        return
    t = pd.read_csv(path).set_index("method")
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "M\\'etodo & Detecciones nuevas por cada 100 validadas & Junto a una validada & Tasa base & Raz\\'on\\\\",
        "\\midrule",
    ]
    for m in [m for m in ORDER if m in t.index and m != "birdnet"]:
        if t.at[m, "new_weighted"] <= 0:
            continue
        lines.append(f"{SHORT.get(m, m)} & {num(t.at[m, 'new_per_100_validated'], 1)} & {num(100 * t.at[m, 'frac_adjacent'], 1)}\\,\\% & {num(100 * t.at[m, 'base_rate'], 1)}\\,\\% & {num(t.at[m, 'lift'], 1)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_new.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_lodo() -> None:
    path = RESULTS / "lodo_heads.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    t = t[t.regime == "estricto"]
    methods = [m for m in ORDER if m in set(t.method)]
    piv = t.pivot(index="device", columns="method", values="macro_ap")
    n = t.drop_duplicates("device").set_index("device")
    lines = ["\\begin{tabular}{lrr" + "c" * len(methods) + "}", "\\toprule", "Grabadora & ventanas & especies & " + " & ".join(SHORT.get(m, m) for m in methods) + "\\\\", "\\midrule"]
    for dev in piv.index:
        lines.append(f"{dev} & {int(n.at[dev, 'windows']):,}".replace(",", "\\,") + f" & {int(n.at[dev, 'species_evaluated'])} & " + " & ".join(num(piv.at[dev, m]) for m in methods) + "\\\\")
    lines.append("\\midrule")
    lines.append("Media & & & " + " & ".join(num(piv[m].mean()) for m in methods) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_lodo.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_sample() -> None:
    info = json.loads((RESULTS / "dataset_deep.json").read_text(encoding="utf-8"))
    su, ss = info["strata_universe"], info["strata_sample"]
    rows = [
        ("A", "Con alg\\'un positivo de especie rara ($\\leq$ 450) o alg\\'un rechazo", su["A"], "1", ss["A"]),
        ("B", "Resto de grabaciones con detecciones revisadas", su["B"], "0{,}30", ss["B"]),
        ("C", "Sesiones revisadas, sin detecciones revisadas", su["C"], "0{,}10", ss["C"]),
    ]
    thin = lambda v: f"{int(v):,}".replace(",", "\\,")
    lines = ["\\begin{tabular}{clrcr}", "\\toprule", "Estrato & Grabaciones & Universo & Prob. & Muestra\\\\", "\\midrule"]
    lines += [f"{a} & {b} & {thin(c)} & {d} & {thin(e)}\\\\" for a, b, c, d, e in rows]
    lines += ["\\midrule", f"& Total & {thin(sum(su.values()))} & & {thin(sum(ss.values()))}\\\\", "\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_sample.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def macros(summ, boot, rev, snr) -> None:
    info = json.loads((RESULTS / "dataset_deep.json").read_text(encoding="utf-8"))
    piv = summ.pivot(index="method", columns="regime", values="macro_ap")
    names = {"birdnet": "Birdnet", "prototipos": "Proto", "lineal_v1": "LinealOld", "lineal": "Lineal", "mlp": "Mlp", "ft_v1": "FtOld", "residual": "Residual", "residual_estricto": "ResidualStrict", "contexto": "Contexto", "residual_mix": "ResidualMix", "lineal_mix": "LinealMix", "birdnet_x_lineal": "Ensemble", "ft_bloque4": "FtConv", "ft_bloques34": "FtConvDeep"}
    thin = lambda v: f"{int(round(v)):,}".replace(",", "\\,")
    out = ["% Generado por finetuning/deep/report.py; no editar a mano."]
    out.append(f"\\newcommand{{\\deepRecordings}}{{{thin(info['recordings'])}}}")
    out.append(f"\\newcommand{{\\deepWindows}}{{{thin(info['windows'])}}}")
    out.append(f"\\newcommand{{\\deepPositives}}{{{thin(info['positive_pairs'])}}}")
    out.append(f"\\newcommand{{\\deepRejected}}{{{info['hard_negative_pairs']}}}")
    out.append(f"\\newcommand{{\\deepUniverseWindows}}{{{thin(info['estimated_universe_windows'])}}}")
    out.append(f"\\newcommand{{\\deepUniversePositives}}{{{thin(info['estimated_universe_positive_pairs'])}}}")
    out.append(f"\\newcommand{{\\deepQc}}{{{num(100 * info['qc_agreement_kept'], 1)}}}")
    out.append(f"\\newcommand{{\\deepBirdnetUnweighted}}{{{num(info['birdnet_unweighted_map']['estricto'])}}}")
    for m, tag in names.items():
        if m in piv.index:
            out.append(f"\\newcommand{{\\deepMap{tag}}}{{{num(piv.at[m, 'estricto'])}}}")
            out.append(f"\\newcommand{{\\deepMapMasked{tag}}}{{{num(piv.at[m, 'enmascarado'])}}}")
            d = _delta(boot, m, "estricto")
            if d is not None:
                out.append(f"\\newcommand{{\\deepDelta{tag}}}{{{signed(d['delta_mean'])}}}")
                out.append(f"\\newcommand{{\\deepCi{tag}}}{{{interval(d['ci_low'], d['ci_high'])}}}")
    r = rev.set_index("method")
    for m, tag in names.items():
        if m in r.index:
            out.append(f"\\newcommand{{\\deepRev{tag}}}{{{num(r.at[m, 'auc_pooled'], 2)}}}")
    if snr is not None:
        for m, tag in names.items():
            g = snr[snr.method == m].set_index("atten_db")
            for a, lab in ((0.0, "Zero"), (6.0, "Six"), (12.0, "Twelve"), (18.0, "Eighteen"), (24.0, "TwentyFour")):
                if a in g.index:
                    out.append(f"\\newcommand{{\\deepSnr{tag}{lab}}}{{{num(100 * g.at[a, 'recall'], 1)}}}")
    (TAB / "macros_deep.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def figure_methods(summ, boot) -> None:
    """Barras horizontales de mAP por metodo (un panel por regimen); BirdNET en gris de referencia."""
    piv = summ.pivot(index="method", columns="regime", values="macro_ap")
    methods = [m for m in ORDER if m in piv.index]
    ci = {(b["method"], b["regime"]): (b["ci_low"], b["ci_high"]) for b in boot if "method" in b}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.33 * len(methods) + 1.0), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, regime, title in zip(axes, ("estricto", "enmascarado"), ("Régimen estricto", "Régimen enmascarado")):
        vals = piv.loc[methods, regime].to_numpy()
        colors = [C_REF if m == "birdnet" else C1 for m in methods]
        ax.barh(y, vals, height=0.56, color=colors, edgecolor=SURFACE, linewidth=0.8, zorder=2)
        for yi, m, v in zip(y, methods, vals):
            if (m, regime) in ci:
                lo, hi = ci[(m, regime)]
                ax.plot([lo, hi], [yi, yi], color=INK, linewidth=0.9, zorder=3, solid_capstyle="butt")
            ax.text(min(max(v, ci.get((m, regime), (v, v))[1]) + 0.012, 1.0), yi, f"{v:.3f}".replace(".", ","), va="center", ha="left", fontsize=7.5, color=INK)
        ax.axvline(piv.at["birdnet", regime], color=MUTED, linewidth=0.8, zorder=1)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0,25", "0,5", "0,75", "1"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("mAP macro ponderada")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([PLAIN.get(m, m) for m in methods])
    fig.tight_layout()
    _save(fig, "fig_deep_methods")


def figure_snr(snr, focus: list[str] | None = None) -> None:
    """Exhaustividad frente a atenuacion: BirdNET (referencia) y hasta tres modelos."""
    if snr is None:
        return
    avail = [m for m in ORDER if m in set(snr.method)]
    if focus is None:
        cands = [m for m in ("ft_bloque4", "residual_mix", "lineal_mix", "birdnet_x_lineal_mix", "residual", "lineal") if m in avail]
        focus = ["birdnet"] + cands[:3]
    styles = {0: (C_REF, "--", "o"), 1: (C1, "-", "D"), 2: (C2, "-", "s"), 3: (C3, "-", "^")}
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ends = []
    for i, m in enumerate(focus):
        g = snr[snr.method == m].sort_values("atten_db")
        color, ls, mk = styles[i]
        x = g.atten_db.to_numpy()
        ax.fill_between(x, 100 * g.recall_lo, 100 * g.recall_hi, color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(x, 100 * g.recall, color=color, linestyle=ls, linewidth=2.0, marker=mk, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.4, label=PLAIN.get(m, m), zorder=3)
        ends.append((float(100 * g.recall.iloc[-1]), m))
    # etiquetas directas al final de cada linea, separadas si se pisan
    ends.sort()
    placed = []
    for v, m in ends:
        yv = v if not placed or v - placed[-1] >= 4.5 else placed[-1] + 4.5
        placed.append(yv)
        ax.annotate(f"{v:.0f} %", xy=(24, v), xytext=(25.0, yv), fontsize=7.5, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6) if abs(yv - v) > 0.5 else None)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["original", "−6 dB", "−12 dB", "−18 dB", "−24 dB"])
    ax.set_xlim(-1, 28)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exhaustividad (%) con 0,5 % de falsos positivos")
    ax.set_xlabel("Atenuación del canto sobre un fondo de la misma sesión")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_deep_snr")


def figure_snr_species(method: str | None = None, level: float = 12.0) -> None:
    """Por especie: exhaustividad a ``level`` dB de atenuacion, BirdNET frente al modelo afinado (mancuernas)."""
    path = RESULTS / "snr_species_final.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    avail = set(t.method)
    if method is None:
        method = next((m for m in ("ft_bloque4", "residual_mix", "lineal_mix") if m in avail), None)
    if method is None or "birdnet" not in avail:
        return
    a = t[(t.method == "birdnet") & np.isclose(t.atten_db, level)].set_index("species")["recall"]
    b = t[(t.method == method) & np.isclose(t.atten_db, level)].set_index("species")["recall"]
    df = pd.DataFrame({"birdnet": a, "modelo": b}).dropna()
    df = df.assign(delta=df.modelo - df.birdnet).sort_values("delta")
    fig, ax = plt.subplots(figsize=(6.0, 0.2 * len(df) + 1.1))
    y = np.arange(len(df))
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.plot([100 * r.birdnet, 100 * r.modelo], [yi, yi], color=GRID if abs(r.delta) < 1e-9 else MUTED, linewidth=1.2, zorder=1)
    ax.scatter(100 * df.birdnet, y, s=34, color=C_REF, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=PLAIN["birdnet"], marker="o")
    ax.scatter(100 * df.modelo, y, s=34, color=C1, edgecolor=SURFACE, linewidth=1.2, zorder=4, label=PLAIN.get(method, method), marker="D")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontstyle="italic", fontsize=7.5)
    ax.set_xlim(-3, 103)
    ax.set_xlabel(f"Exhaustividad (%) con el canto atenuado {level:g} dB")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_deep_snr_species")


def figure_species(focus: tuple[str, ...] = ("prototipos", "lineal_v1", "lineal")) -> None:
    """AP por especie de las cabezas entrenadas desde cero (receta anterior frente a nueva)."""
    t = pd.read_csv(RESULTS / "cv_metrics_final.csv")
    t = t[t.regime == "estricto"]
    focus = tuple(m for m in focus if m in set(t.method))
    if len(focus) < 2:
        return
    piv = t.pivot(index="species", columns="method", values="ap")
    order = piv[focus[-1]].sort_values().index
    styles = [(C2, "s"), (C3, "^"), (C1, "D")]
    fig, ax = plt.subplots(figsize=(6.0, 0.2 * len(order) + 1.1))
    y = np.arange(len(order))
    ax.hlines(y, piv.loc[order, list(focus)].min(axis=1), piv.loc[order, list(focus)].max(axis=1), color=GRID, linewidth=1.2, zorder=1)
    if "birdnet" in piv.columns:
        ax.scatter(piv.loc[order, "birdnet"], y, s=22, facecolor="none", edgecolor=C_REF, linewidth=1.0, zorder=2, label=PLAIN["birdnet"])
    for m, (c, mk) in zip(focus, styles[-len(focus) :]):
        ax.scatter(piv.loc[order, m], y, s=30, color=c, marker=mk, edgecolor=SURFACE, linewidth=1.0, zorder=3, label=PLAIN.get(m, m))
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontstyle="italic", fontsize=7.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0,25", "0,5", "0,75", "1"])
    ax.set_xlabel("Precisión promedio (AP) ponderada, régimen estricto")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_deep_species")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--reps", type=int, default=500)
    args = parser.parse_args()
    if not args.no_eval:
        final_evaluation(reps=args.reps)
    summ, boot, rev, snr = _load()
    TAB.mkdir(parents=True, exist_ok=True)
    table_methods(summ, boot, rev)
    table_reviewed(rev)
    table_snr(snr)
    table_new()
    table_lodo()
    table_sample()
    macros(summ, boot, rev, snr)
    figure_methods(summ, boot)
    figure_snr(snr)
    figure_snr_species()
    figure_species()
    print("Tablas y figuras escritas en", TAB, "y", FIG)


if __name__ == "__main__":
    main()
