"""Figuras y tablas LaTeX del informe a partir de los resultados del pipeline.

Todo lo que se escribe aqui es agregado (metricas, conteos, curvas), nunca
audio ni detecciones individuales, por lo que puede versionarse.

Uso:
    python -m finetuning.report            # todo
    python -m finetuning.report --no-tsne  # sin el mapa de embeddings (no requiere datos)
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import config
from .train import METHOD_LABELS, OOF_NPZ

# Tinta y superficie (texto siempre en tinta neutra, nunca en color de serie).
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    }
)

# Paleta categorica de referencia en orden fijo; el color sigue a la entidad
# (cada metodo conserva su color en todas las figuras). BirdNET sin ajuste es la
# referencia y va en gris neutro con linea discontinua; los tres metodos
# restantes usan las tres primeras ranuras (validadas para todos los pares).
# Ademas del color se codifica con marcador y estilo de linea (lectura en B/N).
COLORS = {"birdnet": INK_2, "ft_birdnet": "#2a78d6", "prototipos": "#eb6834", "lineal_l2": "#1baf7a"}
MARKERS = {"birdnet": "o", "ft_birdnet": "D", "prototipos": "s", "lineal_l2": "^"}
LINESTYLES = {"birdnet": "--", "ft_birdnet": "-", "prototipos": "-.", "lineal_l2": ":"}
FOCUS = ["birdnet", "ft_birdnet", "lineal_l2", "prototipos"]
REGIMES = ["estricto", "enmascarado", "candidatos"]
REGIME_LABELS = {"estricto": "Estricto", "enmascarado": "Enmascarado", "candidatos": "Candidatos (BirdNET $\\geq$ 0,1)"}
REGIME_COLORS = {"estricto": "#2a78d6", "enmascarado": "#eb6834", "candidatos": "#1baf7a"}
METHOD_ORDER = [
    "birdnet",
    "ft_birdnet",
    "ft_birdnet_l0.03",
    "ft_birdnet_l0.003",
    "prototipos",
    "lineal",
    "lineal_l2",
    "mlp",
    "mlp_l2",
    "mlp_l2_mixup",
]
SHORT = {
    "birdnet": "BirdNET",
    "ft_birdnet": "BirdNET afinado",
    "prototipos": "Prototipos (app)",
    "lineal_l2": "Sonda lineal L2",
    "lineal": "Sonda lineal",
    "mlp": "MLP",
    "mlp_l2": "MLP L2",
    "mlp_l2_mixup": "MLP L2+mixup",
}

R = config.RESULTS_DIR
FIG = config.FIGURES_DIR
TAB = config.TABLES_DIR


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def _tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _fmt(x, nd=3) -> str:
    if x is None:
        return "--"
    try:
        if np.isnan(float(x)):
            return "--"
    except (TypeError, ValueError):
        return str(x)
    return f"{float(x):.{nd}f}".replace(".", "{,}")


def _dec(x: float, nd: int = 3, sign: bool = False) -> str:
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace(".", "{,}")


def _int(x) -> str:
    return f"{int(x):,}".replace(",", r"\,")


def _it(sp: str) -> str:
    return f"$\\it{{{sp.replace(' ', '~')}}}$"


def _write(name: str, body: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")


def _label(m: str) -> str:
    return _tex_escape(METHOD_LABELS.get(m, m)).replace(r"\textbackslash{}", "\\")


# --------------------------------------------------------------------------- #
# Tablas
# --------------------------------------------------------------------------- #
def table_dataset(summary: dict, qc: pd.DataFrame, subset: dict) -> None:
    excluded = qc[qc["excluded"]]
    rows = [
        ("Grabaciones seleccionadas y copiadas de Drive", f"{subset.get('copied', subset.get('files', '--'))}"),
        ("Grabaciones usadas tras control de calidad (55 s, 48 kHz)", f"{summary['recordings']}"),
        ("Sesiones (grabadora $\\times$ día $\\times$ periodo)", f"{summary['sessions']}"),
        ("Grabadoras", ", ".join(summary["devices"])),
        ("Ventanas de 3 s", _int(summary["windows"])),
        ("Especies objetivo", f"{summary['species']}"),
        ("Pares ventana--especie positivos", _int(summary["positive_pairs"])),
        ("Ventanas con al menos un positivo objetivo", _int(summary["windows_with_target_positive"])),
        ("Ventanas de fondo (sin especie validada)", _int(summary["background_windows"])),
        ("Negativos explícitos (\\texttt{Valid}=0)", f"{summary['hard_negative_pairs']}"),
        ("Pares inciertos (BirdNET $\\geq 0{,}5$ sin validar)", f"{summary['uncertain_pairs']}"),
        (
            "Sesiones excluidas por control de calidad",
            f"{len(excluded)} ({', '.join(_tex_escape(s) for s in excluded['session'])})" if len(excluded) else "0",
        ),
    ]
    body = "\\begin{tabular}{lr}\n\\toprule\nConcepto & Valor\\\\\n\\midrule\n"
    body += "".join(f"{a} & {b}\\\\\n" for a, b in rows)
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_dataset", body)


def table_methods(summ: pd.DataFrame) -> None:
    piv = {r: summ[summ["regime"] == r].set_index("method") for r in REGIMES}
    strict = piv["estricto"]
    methods = [m for m in METHOD_ORDER if m in strict.index]
    best = {
        "macro_ap": strict["macro_ap"].idxmax(),
        "macro_f1": strict["macro_f1"].idxmax(),
        "masked": piv["enmascarado"]["macro_ap"].idxmax(),
        "cand": piv["candidatos"]["macro_ap"].idxmax(),
    }

    def cell(v, is_best):
        s = _fmt(v)
        return f"\\textbf{{{s}}}" if is_best else s

    body = (
        "\\begin{tabular}{lccccccc}\n\\toprule\n"
        " & \\multicolumn{5}{c}{Régimen estricto} & Enmasc. & Candidatos\\\\\n"
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-8}\n"
        "Método & mAP & AP mediana & AUC & F1 & Rec@P90 & mAP & mAP\\\\\n\\midrule\n"
    )
    groups = {"birdnet": 0, "ft_birdnet": 1, "ft_birdnet_l0.03": 1, "ft_birdnet_l0.003": 1}
    prev = None
    for m in methods:
        g = groups.get(m, 2)
        if prev is not None and g != prev:
            body += "\\midrule\n"
        prev = g
        r = strict.loc[m]
        body += (
            f"{_label(m)} & {cell(r.macro_ap, m == best['macro_ap'])} & {_fmt(r.median_ap)} & {_fmt(r.macro_auc)} & "
            f"{cell(r.macro_f1, m == best['macro_f1'])} & {_fmt(r.macro_rec_at_p90)} & "
            f"{cell(piv['enmascarado'].at[m, 'macro_ap'], m == best['masked'])} & {cell(piv['candidatos'].at[m, 'macro_ap'], m == best['cand'])}\\\\\n"
        )
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_methods", body)


def table_species(per: pd.DataFrame, info: pd.DataFrame, local_meta: dict, tf_thr: pd.DataFrame) -> None:
    strict = per[per["regime"] == "estricto"]
    piv = strict.pivot(index="species", columns="method", values="ap")
    thr = {s["scientific"]: s["threshold"] for s in local_meta.get("species", [])}
    rev = tf_thr.set_index("species")
    order = info.sort_values("subset_positives", ascending=False).index
    cols = ["birdnet", "ft_birdnet", "lineal_l2", "prototipos"]
    head = (
        "Especie & $n$ & ses. & revis. & BirdNET & afinado & sonda L2 & prototipos & umbral\\\\\n"
    )
    body = (
        "\\begingroup\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{longtable}{lrrrccccc}\n\\caption{Precisión promedio (AP) por especie en validación cruzada agrupada por sesión "
        "(régimen estricto). $n$: ventanas positivas en el subconjunto; ses.: sesiones con la especie; revis.: detecciones revisadas "
        "por el equipo en TablaFinal (todas las grabaciones); umbral: umbral exportado para la app. En negrita, el mejor AP de la fila.}"
        "\\label{tab:species}\\\\\n\\toprule\n" + head + "\\midrule\n\\endfirsthead\n"
        "\\toprule\n" + head + "\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n"
    )
    for sp in order:
        vals = [piv.at[sp, c] if c in piv.columns else np.nan for c in cols]
        best_val = np.nanmax(vals)
        cells = [f"\\textbf{{{_fmt(v)}}}" if np.isclose(v, best_val) else _fmt(v) for v in vals]
        body += (
            f"\\textit{{{_tex_escape(sp)}}} & {int(info.at[sp, 'subset_positives'])} & {int(info.at[sp, 'subset_sessions'])} & "
            f"{_int(rev.at[sp, 'reviewed'])} & " + " & ".join(cells) + f" & {_fmt(thr.get(sp), 2)}\\\\\n"
        )
    body += "\\end{longtable}\n\\endgroup\n"
    _write("tab_species", body)


def table_lodo(lodo: pd.DataFrame) -> None:
    piv = lodo.pivot(index="device", columns="method", values="macro_ap")
    methods = [m for m in FOCUS if m in piv.columns] + [m for m in piv.columns if m not in FOCUS]
    n_sp = lodo.groupby("device")["species_evaluated"].max()
    n_win = lodo.groupby("device")["windows"].max()
    head = " & ".join(SHORT.get(m, _label(m)) for m in methods)
    body = f"\\begin{{tabular}}{{l{'c' * len(methods)}rr}}\n\\toprule\nGrabadora retenida & {head} & especies & ventanas\\\\\n\\midrule\n"
    for dev, row in piv.iterrows():
        body += f"{dev} & " + " & ".join(_fmt(row[m]) for m in methods) + f" & {int(n_sp[dev])} & {_int(n_win[dev])}\\\\\n"
    body += "\\midrule\nPromedio & " + " & ".join(_fmt(piv[m].mean()) for m in methods) + " & & \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_lodo", body)


def table_bootstrap(boot: list[dict]) -> None:
    body = "\\begin{tabular}{lccc}\n\\toprule\nComparación ($A - B$) & $\\Delta$mAP & IC 95\\,\\% & $P(\\Delta\\leq 0)$\\\\\n\\midrule\n"
    for b in boot:
        name = f"{SHORT.get(b['a'], b['a'])} $-$ {SHORT.get(b['b'], b['b'])}"
        body += f"{name} & {_dec(b['delta_mean'], sign=True)} & [{_dec(b['ci_low'], sign=True)}; {_dec(b['ci_high'], sign=True)}] & {_dec(b['p_delta_le_0'])}\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_bootstrap", body)


def table_learning_curve(lc: pd.DataFrame, birdnet_map: float) -> None:
    piv = lc.pivot(index="k", columns="method", values="macro_ap")
    methods = [m for m in ("prototipos", "lineal_l2", "ft_birdnet") if m in piv.columns]
    body = (
        f"\\begin{{tabular}}{{r{'c' * len(methods)}}}\n\\toprule\n"
        "$k$ (positivos por especie) & " + " & ".join(SHORT[m] for m in methods) + "\\\\\n\\midrule\n"
    )
    for k, row in piv.iterrows():
        body += f"{int(k)} & " + " & ".join(_fmt(row[m]) for m in methods) + "\\\\\n"
    body += f"\\midrule\n\\multicolumn{{{len(methods) + 1}}}{{l}}{{Referencia: BirdNET sin ajuste (no usa los datos), mAP = {_fmt(birdnet_map)}}}\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_learning_curve", body)


def table_ablation(summ: pd.DataFrame, abl: pd.DataFrame, run: dict) -> None:
    strict = summ[summ["regime"] == "estricto"].set_index("method")
    rows = []
    for m in ("ft_birdnet", "ft_birdnet_l0.03", "ft_birdnet_l0.003"):
        if m not in strict.index:
            continue
        ep = run.get("cv_best_epochs", {}).get(m, [])
        rows.append((_label(m), "pérdida BCE estándar", strict.at[m, "macro_ap"], strict.at[m, "macro_f1"], ", ".join(str(e) for e in ep)))
    for r in abl.itertuples():
        base = r.method.replace("_ignora_inciertos", "")
        ep = run.get("cv_best_epochs", {}).get(base, [])
        rows.append((_label(base), "ignora pares inciertos", r.macro_ap, r.macro_f1, ", ".join(str(e) for e in ep)))
    body = "\\begin{tabular}{llccl}\n\\toprule\nVariante & Pérdida & mAP & F1 & Mejor época por pliegue\\\\\n\\midrule\n"
    for a, b, c, d, e in rows:
        body += f"{a} & {b} & {_fmt(c)} & {_fmt(d)} & {e}\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_ablation", body)


def table_review(calib: pd.DataFrame, tf_thr: pd.DataFrame, n_rows: int = 12) -> None:
    m = calib.merge(tf_thr, on="species")
    m = m.sort_values("prec@0.25").head(n_rows)
    body = (
        "\\begin{tabular}{lrrcccrr}\n\\toprule\n"
        " & \\multicolumn{5}{c}{Subconjunto (BirdNET recalculado)} & \\multicolumn{2}{c}{TablaFinal}\\\\\n"
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}\n"
        "Especie & positivos & $\\geq$0,25 & valid. $\\geq$0,25 & valid. $\\geq$0,5 & valid. $\\geq$0,75 & revisadas & rechazadas\\\\\n\\midrule\n"
    )
    for _, r in m.iterrows():
        body += (
            f"\\textit{{{_tex_escape(r['species'])}}} & {int(r['n_pos'])} & {int(r['n@0.25'])} & {_fmt(r['prec@0.25'], 2)} & "
            f"{_fmt(r['prec@0.5'], 2)} & {_fmt(r['prec@0.75'], 2)} & {_int(r['reviewed'])} & {int(r['rejected'])}\\\\\n"
        )
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_review", body)


def table_nonlocal(nl: dict) -> None:
    body = "\\begin{tabular}{lccc}\n\\toprule\nUmbral BirdNET & Detecciones & No locales & \\% no locales\\\\\n\\midrule\n"
    for t in ("0.25", "0.5", "0.75"):
        tot, non = nl[f"det_total@{t}"], nl[f"det_nonlocal@{t}"]
        body += f"{t.replace('.', ',')} & {_int(tot)} & {_int(non)} & {100 * non / max(tot, 1):.1f}\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_nonlocal", body)
    body = "\\begin{tabular}{llr}\n\\toprule\nEspecie (BirdNET) & Nombre común & Detecciones $\\geq$0,5\\\\\n\\midrule\n"
    for item in nl["top_nonlocal@0.5"][:8]:
        sci, _, common = item["label"].partition("_")
        body += f"\\textit{{{_tex_escape(sci)}}} & {_tex_escape(common)} & {item['detections']}\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    _write("tab_nonlocal_top", body)


def write_macros(summary: dict, summ: pd.DataFrame, boot: list[dict], nl: dict, review: dict, qc: pd.DataFrame, lodo: pd.DataFrame) -> None:
    """Numeros clave como macros LaTeX para que el texto no quede desfasado."""
    s = summ[summ["regime"] == "estricto"].set_index("method")["macro_ap"]
    mk = summ[summ["regime"] == "enmascarado"].set_index("method")["macro_ap"]
    cd = summ[summ["regime"] == "candidatos"].set_index("method")["macro_ap"]
    bmap = {(b["a"], b["b"]): b for b in boot}
    lodo_mean = lodo.groupby("method")["macro_ap"].mean()
    ok = qc[~qc["excluded"]]

    def num(x, nd=3):
        return f"{x:.{nd}f}".replace(".", "{,}")

    macros = {
        "nWindows": _int(summary["windows"]),
        "nRecordings": str(summary["recordings"]),
        "nSessions": str(summary["sessions"]),
        "nSpecies": str(summary["species"]),
        "nPositives": _int(summary["positive_pairs"]),
        "nBackground": _int(summary["background_windows"]),
        "mapBirdnet": num(s["birdnet"]),
        "mapFT": num(s["ft_birdnet"]),
        "mapProto": num(s["prototipos"]),
        "mapLinear": num(s["lineal_l2"]),
        "mapMLP": num(s["mlp_l2"]),
        "mapBirdnetMasked": num(mk["birdnet"]),
        "mapBirdnetCand": num(cd["birdnet"]),
        "mapProtoCand": num(cd["prototipos"]),
        "mapLinearCand": num(cd["lineal_l2"]),
        "lodoBirdnet": num(lodo_mean.get("birdnet", np.nan)),
        "lodoProto": num(lodo_mean.get("prototipos", np.nan)),
        "lodoLinear": num(lodo_mean.get("lineal_l2", np.nan)),
        "nonlocalHalf": f"{100 * nl['det_nonlocal@0.5'] / nl['det_total@0.5']:.0f}",
        "nonlocalQuarter": f"{100 * nl['det_nonlocal@0.25'] / nl['det_total@0.25']:.0f}",
        "reviewMedianQuarter": num(review["median_prec@0.25"], 2),
        "reviewMedianHalf": num(review["median_prec@0.5"], 2),
        "nPartialReview": str(len(review["species_with_partial_review"])),
        "qcAgreement": f"{100 * (qc['rows'] * qc['agreement']).sum() / max(qc['rows'].sum(), 1):.1f}".replace(".", "{,}"),
        "qcAgreementKept": f"{100 * (ok['rows'] * ok['agreement']).sum() / max(ok['rows'].sum(), 1):.1f}".replace(".", "{,}"),
        "qcExcludedAgreement": f"{100 * qc.loc[qc['excluded'], 'agreement'].mean():.1f}".replace(".", "{,}") if qc["excluded"].any() else "--",
    }
    for key, (a, b) in {"FTvsBirdnet": ("ft_birdnet", "birdnet"), "LinearVsProto": ("lineal_l2", "prototipos"), "BirdnetVsLinear": ("birdnet", "lineal_l2")}.items():
        if (a, b) in bmap:
            d = bmap[(a, b)]
            macros[f"delta{key}"] = f"{d['delta_mean']:+.3f}".replace(".", "{,}")
            macros[f"ci{key}"] = f"[{d['ci_low']:+.3f}; {d['ci_high']:+.3f}]".replace(".", "{,}")
    body = "% Generado por finetuning/report.py; no editar a mano.\n"
    body += "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    _write("macros", body)


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def fig_species_counts(info: pd.DataFrame) -> None:
    order = info.sort_values("positives")
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    y = np.arange(len(order))
    ax.barh(y + 0.2, order["positives"], height=0.38, color="#9ec5f4", edgecolor="white", linewidth=0.6, label="TablaFinal (todas las grabaciones)")
    ax.barh(y - 0.2, order["subset_positives"], height=0.38, color="#2a78d6", edgecolor="white", linewidth=0.6, label="Subconjunto copiado y analizado")
    ax.set_yticks(y)
    ax.set_yticklabels([_it(s) for s in order.index])
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Detecciones validadas por especie (escala logarítmica)")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "fig_species_counts")


def fig_ap_by_species(per: pd.DataFrame) -> None:
    piv = per[per["regime"] == "estricto"].pivot(index="species", columns="method", values="ap")
    piv = piv.sort_values("lineal_l2")
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    y = np.arange(len(piv))
    for i, (_, row) in enumerate(piv.iterrows()):
        vals = row[FOCUS]
        ax.plot([vals.min(), vals.max()], [i, i], color=GRID, lw=1.2, zorder=1)
    for m in FOCUS:
        ax.scatter(
            piv[m], y, label=SHORT[m], marker=MARKERS[m], s=26, color=COLORS[m], edgecolors="white", linewidths=0.6, zorder=3
        )
    ax.set_yticks(y)
    ax.set_yticklabels([_it(s) for s in piv.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Precisión promedio (AP), validación cruzada agrupada por sesión")
    ax.set_xlim(0, 1.03)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=4)
    _save(fig, "fig_ap_by_species")


def fig_method_summary(summ: pd.DataFrame) -> None:
    methods = ["birdnet", "ft_birdnet", "lineal_l2", "mlp_l2", "prototipos"]
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    x = np.arange(len(methods))
    w = 0.26
    for j, reg in enumerate(REGIMES):
        sub = summ[summ["regime"] == reg].set_index("method")
        vals = [sub.at[m, "macro_ap"] for m in methods]
        bars = ax.bar(x + (j - 1) * w, vals, width=w - 0.02, color=REGIME_COLORS[reg], edgecolor="white", linewidth=0.6, label=REGIME_LABELS[reg])
        if reg == "estricto":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in methods], fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("mAP (macro)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    _save(fig, "fig_method_summary")


def fig_pr_curves(oof: dict, Y: np.ndarray, species: list[str], info: pd.DataFrame) -> None:
    ranked = info.sort_values("subset_positives", ascending=False).index.tolist()
    pick = [ranked[0], ranked[len(ranked) // 6], ranked[len(ranked) // 3], ranked[len(ranked) // 2], ranked[2 * len(ranked) // 3], ranked[-3]]
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.4), sharex=True, sharey=True)
    for ax, sp in zip(axes.ravel(), pick):
        k = species.index(sp)
        for m in FOCUS:
            if m not in oof:
                continue
            p, r, _ = precision_recall_curve(Y[:, k], oof[m][:, k])
            ax.plot(r, p, color=COLORS[m], ls=LINESTYLES[m], lw=1.4, label=SHORT[m])
        genus, epithet = sp.split()[:2]
        ax.set_title(f"{_it(genus[0] + '. ' + epithet)} ($n$={int(Y[:, k].sum())})", fontsize=8)
        ax.set_ylim(0, 1.03)
        ax.set_xlim(0, 1.0)
    for ax in axes[1]:
        ax.set_xlabel("Exhaustividad (recall)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precisión")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=7.5)
    _save(fig, "fig_pr_curves")


def fig_lodo(lodo: pd.DataFrame) -> None:
    piv = lodo.pivot(index="device", columns="method", values="macro_ap")
    methods = [m for m in FOCUS if m in piv.columns]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    x = np.arange(len(piv))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        ax.bar(x + (j - (len(methods) - 1) / 2) * w, piv[m], width=w - 0.02, color=COLORS[m], edgecolor="white", linewidth=0.6, label=SHORT[m])
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("mAP en grabadora no vista")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper center")
    _save(fig, "fig_lodo")


def fig_learning_curve(lc: pd.DataFrame, birdnet_map: float) -> None:
    piv = lc.pivot(index="k", columns="method", values="macro_ap")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.axhline(birdnet_map, color=COLORS["birdnet"], ls=LINESTYLES["birdnet"], lw=1.2, label="BirdNET sin ajuste")
    for m in ("ft_birdnet", "lineal_l2", "prototipos"):
        if m in piv.columns:
            ax.plot(piv.index, piv[m], color=COLORS[m], ls=LINESTYLES[m], marker=MARKERS[m], ms=5, mec="white", mew=0.6, label=SHORT[m])
    ax.set_xscale("log")
    ax.set_xticks(piv.index)
    ax.set_xticklabels([str(int(k)) for k in piv.index])
    ax.minorticks_off()
    ax.set_xlabel("Positivos de entrenamiento por especie ($k$)")
    ax.set_ylabel("mAP (macro, estricto)")
    ax.set_ylim(0.3, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, "fig_learning_curve")


def fig_score_distributions(oof: dict, Y: np.ndarray) -> None:
    methods = [m for m in ("birdnet", "lineal_l2", "prototipos") if m in oof]
    fig, axes = plt.subplots(1, len(methods), figsize=(6.4, 2.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        s = oof[m]
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        bins = np.linspace(lo, hi, 41)
        ax.hist(s[Y == 0], bins=bins, color="#c9c8c3", density=True, label="Negativos")
        ax.hist(s[Y == 1], bins=bins, color=COLORS[m], density=True, alpha=0.8, label="Positivos")
        ax.set_yscale("log")
        ax.set_title(SHORT[m])
        ax.set_xlabel("Puntuación" if m != "prototipos" else "Similitud coseno")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Densidad (log)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    _save(fig, "fig_score_distributions")


def fig_review_coverage(calib: pd.DataFrame, tf_thr: pd.DataFrame) -> None:
    """Fraccion validada de las detecciones BirdNET por umbral: evidencia de revision parcial."""
    m = calib.merge(tf_thr, on="species")
    ts = [0.1, 0.25, 0.5, 0.75, 0.9]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    curves = m[[f"prec@{t}" for t in ts]].to_numpy(dtype=float)
    for row in curves:
        ax.plot(ts, row, color="#c9c8c3", lw=0.8, zorder=1)
    ax.plot(ts, np.nanmedian(curves, axis=0), color="#2a78d6", lw=2, marker="o", ms=4, label="Mediana de especies", zorder=3)
    worst = m.sort_values("prec@0.25").head(2)
    for (_, r), ls in zip(worst.iterrows(), ("-.", ":")):
        ax.plot(ts, [r[f"prec@{t}"] for t in ts], color=INK, lw=1.2, ls=ls, label=r["species"], zorder=2)
    ax.axvline(0.25, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Umbral de BirdNET")
    ax.set_ylabel("Fracción validada")
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("Detecciones del subconjunto", fontsize=8)
    ax = axes[1]
    ax.scatter(m["reviewed"], m["prec@0.25"], s=22, color="#2a78d6", edgecolors="white", linewidths=0.6, zorder=3)
    for _, r in m.sort_values("prec@0.25").head(3).iterrows():
        g, e = r["species"].split()[:2]
        ax.annotate(f"{g[0]}. {e}", (r["reviewed"], r["prec@0.25"]), fontsize=6.5, color=INK_2, xytext=(4, -2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Detecciones revisadas en TablaFinal")
    ax.set_ylabel("Fracción validada ($\\geq$0,25)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cobertura frente a abundancia", fontsize=8)
    _save(fig, "fig_review_coverage")


def fig_embedding_map(n_species: int = 6, max_per_species: int = 80, seed: int = 0) -> None:
    """t-SNE de embeddings BirdNET; un panel por especie (pequenos multiplos)."""
    from sklearn.manifold import TSNE

    from . import dataset as D
    from .heads import l2_normalize

    ds = D.build_dataset()
    single = ds.Y.sum(axis=1) == 1
    rng = np.random.default_rng(seed)
    top = list(ds.species_info.sort_values("subset_positives", ascending=False).index[: 2 * n_species])
    idx, lab = [], []
    for sp in top:
        k = ds.species.index(sp)
        cand = np.where(single & (ds.Y[:, k] == 1))[0]
        cand = rng.choice(cand, size=min(max_per_species, len(cand)), replace=False)
        idx.extend(cand)
        lab.extend([sp] * len(cand))
    bg = rng.choice(np.where(~ds.any_positive)[0], size=300, replace=False)
    idx.extend(bg)
    lab.extend(["Fondo"] * len(bg))
    lab = np.array(lab)
    emb = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca").fit_transform(l2_normalize(ds.X[np.array(idx)]))
    show = top[:n_species]
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.4))
    for ax, sp in zip(axes.ravel(), show):
        ax.scatter(emb[lab == "Fondo", 0], emb[lab == "Fondo", 1], s=4, color="#e4e3df", linewidths=0, label="Fondo")
        other = (lab != "Fondo") & (lab != sp)
        ax.scatter(emb[other, 0], emb[other, 1], s=4, color="#b4b3ad", linewidths=0, label="Otras especies")
        sel = lab == sp
        ax.scatter(emb[sel, 0], emb[sel, 1], s=8, color="#2a78d6", linewidths=0, label="Especie del panel")
        ax.set_title(_it(sp), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=7.5, markerscale=2)
    _save(fig, "fig_embedding_tsne")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-tsne", action="store_true", help="omite el mapa t-SNE (requiere embeddings locales)")
    args = parser.parse_args()

    config.ensure_dirs()
    summary = json.loads((R / "dataset_summary.json").read_text(encoding="utf-8"))
    run = json.loads((R / "run_info.json").read_text(encoding="utf-8"))
    info = pd.read_csv(R / "species_info.csv", index_col=0)
    qc = pd.read_csv(R / "qc_sessions.csv")
    summ = pd.read_csv(R / "cv_summary.csv")
    per = pd.read_csv(R / "cv_metrics.csv")
    lodo = pd.read_csv(R / "lodo.csv")
    lc = pd.read_csv(R / "learning_curve.csv")
    abl = pd.read_csv(R / "ablation_ignore_uncertain.csv")
    boot = json.loads((R / "bootstrap.json").read_text(encoding="utf-8"))
    calib = pd.read_csv(R / "birdnet_calibration.csv")
    tf_thr = pd.read_csv(R / "thresholds_tablafinal.csv")
    nl = json.loads((R / "nonlocal_detections.json").read_text(encoding="utf-8"))
    review = json.loads((R / "review_protocol.json").read_text(encoding="utf-8"))
    subset = json.loads((R / "subset_summary.json").read_text(encoding="utf-8")) if (R / "subset_summary.json").exists() else {}
    mirror_index = config.WORK_DIR / "mirror_index.csv"
    if mirror_index.exists():
        subset["copied"] = len(pd.read_csv(mirror_index))
        (R / "subset_summary.json").write_text(json.dumps(subset, indent=2), encoding="utf-8")
    meta_path = config.MODELS_DIR / "condesan_local.json"
    local_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    birdnet_map = float(summ.query("regime == 'estricto' and method == 'birdnet'")["macro_ap"].iloc[0])

    table_dataset(summary, qc, subset)
    table_methods(summ)
    table_species(per, info, local_meta, tf_thr)
    table_lodo(lodo)
    table_bootstrap(boot)
    table_learning_curve(lc, birdnet_map)
    table_ablation(summ, abl, run)
    table_review(calib, tf_thr)
    table_nonlocal(nl)
    write_macros(summary, summ, boot, nl, review, qc, lodo)

    fig_species_counts(info)
    fig_ap_by_species(per)
    fig_method_summary(summ)
    fig_lodo(lodo)
    fig_learning_curve(lc, birdnet_map)
    fig_review_coverage(calib, tf_thr)
    if OOF_NPZ.exists():
        data = np.load(OOF_NPZ, allow_pickle=False)
        species = [str(s) for s in data["species"]]
        oof = {k[4:]: data[k] for k in data.files if k.startswith("oof_")}
        fig_pr_curves(oof, data["Y"], species, info)
        fig_score_distributions(oof, data["Y"])
    if not args.no_tsne:
        fig_embedding_map()
    print("Figuras y tablas generadas en", R)


if __name__ == "__main__":
    main()
