"""Tablas LaTeX y figuras de la segunda fase (aprendizaje profundo) a partir de ``evaluate_ra``.

Solo se escriben resultados agregados (metricas, intervalos, curvas): nada de audio ni de
detecciones individuales.

Uso:
    python -m finetuning.deep.report
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from ..report import GRID, INK, INK_2, MUTED  # mismo estilo que el resto del informe
from . import protocol as PR

RESULTS = config.RESULTS_DIR / "deep"
TAB = config.TABLES_DIR
FIG = config.FIGURES_DIR

NAMES = {
    "birdnet": "BirdNET V2.4 (sin ajuste)",
    "dos_etapas": "BirdNET propone, sonda local verifica",
    "verificador": "BirdNET propone, verificador de 5 variables ordena",
    "lineal": "Correcci\\'on lineal del embedding",
    "mlp": "Correcci\\'on MLP residual del embedding",
    "mlp_sin_mezclas": "\\quad sin mezclas atenuadas",
    "mlp_estricto": "\\quad con etiquetas estrictas",
    "mlp_semilla2": "\\quad otra semilla",
    "mlp_neg_mezcla": "\\quad con negativos de mezcla",
    "post": "POST\\_CONV afinada",
    "b43": "Sub-bloques 4-3 y 4-4 + POST\\_CONV",
    "b4": "Bloque 4 completo",
    "lineal_acotado": "Correcci\\'on lineal",
    "mlp_acotado": "Correcci\\'on MLP residual",
    "post_acotado": "POST\\_CONV afinada",
    "b43_acotado": "Sub-bloques 4-3 y 4-4 + POST\\_CONV",
    "b4_acotado": "Bloque 4 completo",
    "prototipos": "Prototipos coseno (app)",
    "lineal_v1": "Sonda lineal, receta anterior",
    "lineal_desde_cero": "Sonda lineal, receta nueva",
}
PLAIN = {
    "birdnet": "BirdNET V2.4 (sin ajuste)",
    "dos_etapas": "BirdNET propone, sonda local verifica",
    "verificador": "BirdNET propone, verificador de 5 variables",
    "lineal": "Lineal",
    "mlp": "MLP",
    "mlp_sin_mezclas": "MLP sin mezclas",
    "mlp_estricto": "MLP, etiquetas estrictas",
    "mlp_semilla2": "MLP, otra semilla",
    "mlp_neg_mezcla": "MLP con negativos de mezcla",
    "post": "POST_CONV",
    "b43": "Sub-bloques 4-3/4-4",
    "b4": "Bloque 4",
    "lineal_acotado": "Lineal acotada",
    "mlp_acotado": "MLP acotada",
    "post_acotado": "POST_CONV acotada",
    "b43_acotado": "Sub-bloques 4-3/4-4 acotada",
    "b4_acotado": "Bloque 4 acotada",
    "prototipos": "Prototipos coseno (app)",
    "lineal_v1": "Sonda lineal, receta anterior",
    "lineal_desde_cero": "Sonda lineal, receta nueva",
}
MACRO = {
    "birdnet": "Birdnet", "dos_etapas": "TwoStage", "verificador": "Verifier", "lineal": "Linear", "mlp": "Mlp", "mlp_sin_mezclas": "MlpNoMix",
    "mlp_estricto": "MlpStrict", "mlp_semilla2": "MlpSeed", "mlp_neg_mezcla": "MlpNeg", "post": "Post", "b43": "Deep", "b4": "Block", "lineal_acotado": "LinearB",
    "mlp_acotado": "MlpB", "post_acotado": "PostB", "b43_acotado": "DeepB", "b4_acotado": "BlockB", "prototipos": "Proto",
    "lineal_v1": "ProbeOld", "lineal_desde_cero": "ProbeNew",
}
FREE = ["lineal", "mlp", "mlp_sin_mezclas", "mlp_estricto", "mlp_semilla2", "mlp_neg_mezcla", "post", "b43", "b4"]
BOUNDED = ["lineal_acotado", "mlp_acotado", "post_acotado", "b43_acotado", "b4_acotado"]
VERIFIERS = ["birdnet", "dos_etapas", "verificador", "prototipos", "lineal_v1", "lineal_desde_cero"]
GRID_NAMES = {"libre": "libre", "d10": "d10", "d10_l3": "d10+", "d30": "d30", "d100": "d100", "muy_prudente": "muy prudente", "prudente": "prudente", "agresivo": "agresiva", "muy_agresivo": "muy agresiva"}
C_REF, C1, C2, C3 = INK_2, "#2a78d6", "#eb6834", "#1baf7a"   # gris de referencia + tres primeras ranuras validadas
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


def thin(v) -> str:
    return f"{int(round(v)):,}".replace(",", "\\,")


def mark(flag) -> str:
    return "--" if flag is None else ("s\\'i" if flag else "no")


def load():
    return json.loads((RESULTS / "eval.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Tablas
# --------------------------------------------------------------------------- #
def table_sample(ds: dict) -> None:
    su, ss = ds["strata_universe"], ds["strata_sample"]
    rows = [
        ("A", "Con alg\\'un positivo de una especie rara ($\\leq$ 450) o alg\\'un rechazo", su["A"], "1", ss["A"]),
        ("B", "Resto de grabaciones con detecciones revisadas", su["B"], "0{,}30", ss["B"]),
        ("C", "De sesiones revisadas, sin detecciones revisadas", su["C"], "0{,}10", ss["C"]),
    ]
    lines = ["\\begin{tabular}{clrcr}", "\\toprule", "Estrato & Grabaciones & Universo & Prob. & Muestra\\\\", "\\midrule"]
    lines += [f"{a} & {b} & {thin(c)} & {d} & {thin(e)}\\\\" for a, b, c, d, e in rows]
    lines += ["\\midrule", f"& Total & {thin(sum(su.values()))} & & {thin(sum(ss.values()))}\\\\", "\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_sample.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_cells(ds: dict) -> None:
    cats = ds["categories"]
    desc = [
        ("P", "Validada por un experto", "positivo"),
        ("H", "Rechazada por un experto", "negativo fiable"),
        ("E", "BirdNET $<$ 0{,}05; nadie la revis\\'o", "negativo f\\'acil"),
        ("G", "0{,}05 $\\leq$ BirdNET $<$ 0{,}25; nadie la revis\\'o", "desconocido"),
        ("U", "BirdNET $\\geq$ 0{,}25 sin listar, celda no revisada o parcial", "desconocido"),
        ("R0", "BirdNET $\\geq$ 0{,}25 sin listar, celda revisada", "negativo (solo al evaluar)"),
        ("X", "Listada, pero BirdNET recalculado $<$ 0{,}25", "se excluye"),
    ]
    lines = ["\\begin{tabular}{clrl}", "\\toprule", "Cat. & Par (ventana, especie) & Pares & Uso\\\\", "\\midrule"]
    lines += [f"{c} & {d} & {thin(cats[c]['pairs'])} & {u}\\\\" for c, d, u in desc]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_cells.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_never(ds: dict, top: int = 8) -> None:
    """Especies con mas propuestas de BirdNET en celdas que nadie reviso (para preguntar al equipo de validacion)."""
    rows = ds.get("never_reviewed_by_species", [])
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Especie & Celdas sin revisar (de las que tienen propuestas) & Propuestas sin revisar (estimaci\\'on)\\\\", "\\midrule"]
    for r in rows[:top]:
        lines.append(f"\\textit{{{r['species']}}} & {r['cells']} de {r['cells_total']} & {thin(r['detections'])}\\\\")
    rest = rows[top:]
    if rest:
        lines.append(f"Otras {len(rest)} especies & {sum(r['cells'] for r in rest)} & {thin(sum(r['detections'] for r in rest))}\\\\")
    lines += ["\\midrule", f"Total & {sum(r['cells'] for r in rows)} & {thin(sum(r['detections'] for r in rows))}\\\\", "\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_never.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_full(methods: dict) -> None:
    """Subconjunto de la primera fase frente al conjunto completo (Colab) y a la muestra ponderada."""
    sub = pd.read_csv(config.RESULTS_DIR / "cv_summary.csv")
    full = pd.read_csv(config.RESULTS_DIR / "completo" / "cv_summary.csv")
    rows = [("birdnet", "BirdNET V2.4 (sin ajuste)"), ("ft_birdnet", "\\'Ultima capa afinada (L2-SP)"), ("lineal_l2", "Sonda lineal desde cero"), ("mlp_l2", "MLP desde cero"), ("prototipos", "Prototipos coseno (app)")]

    def get(df, m, regime):
        v = df[(df.method == m) & (df.regime == regime)]["macro_ap"]
        return float(v.iloc[0]) if len(v) else float("nan")

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & Subconjunto (257 grab.) & \\multicolumn{2}{c}{Conjunto completo (16\\,629 grab.)} & Muestra ponderada\\\\",
        "\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}",
        "M\\'etodo & estricta & estricta & enmascarada & estricta\\\\",
        "\\midrule",
    ]
    sample = {"birdnet": methods["birdnet"]["map_estricto"], "lineal_l2": methods.get("lineal_v1", {}).get("map_estricto"), "prototipos": methods.get("prototipos", {}).get("map_estricto")}
    for m, label in rows:
        lines.append(f"{label} & {num(get(sub, m, 'estricto'))} & {num(get(full, m, 'estricto'))} & {num(get(full, m, 'enmascarado'))} & {num(sample.get(m))}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ladder_rows(methods: dict, names: list[str]) -> list[str]:
    lines = []
    for m in [x for x in names if x in methods]:
        r = methods[m]
        d1, d3 = r["map_limpio_delta"], r.get("m3_presupuesto_S_delta")
        orig = r.get("m3_presupuesto", {}).get("0")
        lines.append(
            f"{NAMES[m]} & {num(r['map_limpio'])} & {signed(d1['delta'])} {interval(d1['ci_low'], d1['ci_high'])} & {num(r['sli'], 2)} & "
            f"{'--' if orig is None else num(100 * orig, 1)} & {'--' if d3 is None else num(100 * r['m3_presupuesto_S'], 1)} & "
            f"{'--' if d3 is None else signed(100 * d3['delta'], 1) + ' ' + interval(100 * d3['ci_low'], 100 * d3['ci_high'], 1)} & "
            f"{mark(r.get('D1_seguro'))} & {mark(r.get('D3_atenuados'))}\\\\"
        )
    return lines


def table_results(methods: dict) -> None:
    """Escalera de modelos anclados: control de seguridad (M1), fuga (SLI) y cantos atenuados (M3)."""
    b = methods["birdnet"]
    lines = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{M1: mAP limpia} & & \\multicolumn{3}{c}{M3: exhaustividad (\\%)} & \\multicolumn{2}{c}{Reglas}\\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
        "Modelo & valor & $\\Delta$ [IC 95\\,\\%] & SLI & original & $S$ & $\\Delta S$ [IC 95\\,\\%] & D1 & D3\\\\",
        "\\midrule",
        f"{NAMES['birdnet']} & {num(b['map_limpio'])} & -- & {num(b['sli'], 2)} & {num(100 * b['m3_presupuesto']['0'], 1)} & {num(100 * b['m3_presupuesto_S'], 1)} & -- & -- & --\\\\",
        "\\addlinespace",
        "\\multicolumn{9}{l}{\\emph{Correcci\\'on libre}}\\\\",
    ]
    lines += _ladder_rows(methods, FREE)
    bounded = _ladder_rows(methods, BOUNDED)
    if bounded:
        lines += ["\\addlinespace", "\\multicolumn{9}{l}{\\emph{Correcci\\'on acotada a $\\pm 0{,}9$ logits (a\\~nadida despu\\'es de ver los resultados de las cabezas)}}\\\\"] + bounded
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_m3(methods: dict) -> None:
    levels = [f"{a:g}" for a in PR.LEVELS]
    head = " & ".join("original" if a == "0" else f"$-${a}\\,dB" for a in levels)
    lines = ["\\begin{tabular}{l" + "c" * len(levels) + "c}", "\\toprule", f"Modelo & {head} & 1\\,\\% f.\\,a.\\\\", "\\midrule"]
    groups = [["birdnet"], [m for m in FREE if not m.startswith("mlp_")], BOUNDED]
    for gi, group in enumerate(groups):
        rows = [m for m in group if m in methods and "m3_presupuesto" in methods[m]]
        if not rows:
            continue
        if gi == 2:
            lines.append("\\multicolumn{7}{l}{\\emph{Correcci\\'on acotada}}\\\\")
        for m in rows:
            r = methods[m]
            lines.append(f"{NAMES[m]} & " + " & ".join(num(100 * r["m3_presupuesto"][a], 1) for a in levels) + f" & {num(100 * r['m3_falsas_alarmas_S'], 1)}\\\\")
        if gi < 2:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_m3.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_verifiers(methods: dict, n_rej: int) -> None:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        f"Modelo & M2 [IC 95\\,\\%] & $\\Delta$ frente a BirdNET [IC 95\\,\\%] & Retirados (de {n_rej}) & SLI & D1 & D2\\\\",
        "\\midrule",
    ]
    for m in [x for x in VERIFIERS + ["lineal", "mlp", "post", "b43", "b4"] + BOUNDED if x in methods]:
        r = methods[m]
        c = r["m2_ci"]
        d = r.get("m2_delta")
        label = NAMES[m] + (" (acotada)" if m in BOUNDED else "")
        lines.append(
            f"{label} & {num(r['m2'], 2)} [{num(c['ci_low'], 2)}; {num(c['ci_high'], 2)}] & {'--' if d is None else signed(d['delta'], 2) + ' ' + interval(d['ci_low'], d['ci_high'], 2)} & "
            f"{r['m2_removed95']} & {num(r['sli'], 2)} & {mark(r.get('D1_seguro'))} & {mark(r.get('D2_rechazos'))}\\\\"
        )
        if m in ("birdnet", "lineal_desde_cero"):
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_verifiers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_ladder(ladder: list[dict]) -> None:
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "Comparaci\\'on (A $-$ B) & $\\Delta S$ de M3, puntos [IC 95\\,\\%] & $\\Delta$ M1 (mAP limpia) [IC 95\\,\\%] & $\\Delta$ M2 [IC 95\\,\\%]\\\\", "\\midrule"]
    short = {"lineal": "lineal", "mlp": "MLP", "mlp_sin_mezclas": "MLP sin mezclas", "mlp_estricto": "MLP con etiquetas estrictas", "mlp_semilla2": "MLP con otra semilla", "mlp_neg_mezcla": "MLP con negativos de mezcla", "post": "POST\\_CONV", "b43": "sub-bloques 4-3/4-4", "b4": "bloque 4"}
    for row in ladder:
        a, b = short.get(row["a"], row["a"]), short.get(row["b"], row["b"])
        m3 = row["m3_S"]
        cells = [f"{signed(100 * m3['delta'], 1)} {interval(100 * m3['ci_low'], 100 * m3['ci_high'], 1)}"]
        cells += [f"{signed(row[k]['delta'], nd)} {interval(row[k]['ci_low'], row[k]['ci_high'], nd)}" for k, nd in (("map_limpio", 3), ("m2", 2))]
        lines.append(f"{a} $-$ {b} & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_ladder.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_grid(run: dict) -> None:
    """Configuracion elegida en cada pliegue con el pliegue interno, por escalon."""
    rows = [("lineal", "Lineal"), ("mlp", "MLP"), ("post", "POST\\_CONV"), ("b43", "Sub-bloques 4-3 y 4-4"), ("b4", "Bloque 4 completo")]
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule", "Escal\\'on & Pliegue 1 & Pliegue 2 & Pliegue 3 & Pliegue 4 & Pliegue 5 & Deriva relativa\\\\", "\\midrule"]
    for m, label in rows:
        if m not in run:
            continue
        folds = run[m]["folds"]
        picks = [GRID_NAMES.get(f["picked"], f["picked"]) + ("" if any(v["feasible"] for v in f["inner"].values()) else "$^{*}$") for f in folds]
        drift = [f["relative_drift"] for f in folds if "relative_drift" in f]
        if drift:
            conv = [np.mean([v for k, v in d.items() if k != "classifier"]) for d in drift]
            cell = f"{num(100 * float(np.mean(conv)), 1)}\\,\\%"
        else:
            cell = "--"
        lines.append(f"{label} & " + " & ".join(picks) + f" & {cell}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_deep_grid.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def macros(doc: dict, run: dict) -> None:
    methods, ds = doc["methods"], doc["dataset"]
    out = ["% Generado por finetuning/deep/report.py; no editar a mano."]
    add = lambda name, value: out.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    add("deepRecordings", thin(ds["recordings"]))
    add("deepWindows", thin(ds["windows"]))
    add("deepPositives", thin(ds["positive_pairs"]))
    add("deepUniverseWindows", thin(ds["estimated_universe_windows"]))
    add("deepUniversePositives", thin(ds["estimated_universe_positive_pairs"]))
    add("deepQc", num(100 * ds["qc_agreement_kept"], 1))
    add("deepRejections", str(ds["m2_rejections"]))
    add("deepSpeciesAtt", str(ds["m3_species"]))
    add("deepReps", thin(ds["bootstrap_reps"]))
    cells = ds["review_cells"]
    add("deepCellsReviewed", str(cells.get("revisada", 0)))
    add("deepCellsNever", str(cells.get("nunca", 0)))
    add("deepCellsPartial", str(cells.get("parcial", 0)))
    add("deepCellsTotal", str(sum(cells.values())))
    for c in ("P", "H", "U", "R0", "G", "X"):
        add(f"deepCat{c.replace('0', 'Zero')}", thin(ds["categories"][c]["pairs"]))
    add("deepCatUWeighted", thin(ds["categories"]["U"]["weighted"]))
    add("deepCatGWeighted", thin(ds["categories"]["G"]["weighted"]))
    add("deepCatPWeighted", thin(ds["categories"]["P"]["weighted"]))
    mix = ds["mixtures"]
    add("deepMixTotal", thin(sum(mix.values())))
    add("deepMixTest", thin(mix.get("test", 0)))
    add("deepMixTrain", thin(mix.get("train", 0)))
    add("deepMixSources", thin(mix.get("test", 0) / (len(PR.LEVELS) - 1)))
    for m, tag in MACRO.items():
        if m not in methods:
            continue
        r = methods[m]
        add(f"deepClean{tag}", num(r["map_limpio"]))
        add(f"deepCleanFold{tag}", num(r["map_limpio_pliegues"]))
        add(f"deepStrict{tag}", num(r["map_estricto"]))
        add(f"deepMasked{tag}", num(r["map_enmascarado"]))
        add(f"deepSli{tag}", num(r["sli"], 2))
        add(f"deepRej{tag}", num(r["m2"], 2))
        add(f"deepRejPct{tag}", num(100 * r["m2"], 0))
        add(f"deepRejRemoved{tag}", str(r["m2_removed95"]))
        if "map_limpio_delta" in r:
            add(f"deepCleanDelta{tag}", signed(r["map_limpio_delta"]["delta"]))
            add(f"deepCleanCi{tag}", interval(r["map_limpio_delta"]["ci_low"], r["map_limpio_delta"]["ci_high"]))
            add(f"deepCleanFoldDelta{tag}", signed(r["map_limpio_pliegues"] - methods["birdnet"]["map_limpio_pliegues"]))
            add(f"deepRejDelta{tag}", signed(r["m2_delta"]["delta"], 2))
            add(f"deepRejCi{tag}", interval(r["m2_delta"]["ci_low"], r["m2_delta"]["ci_high"], 2))
        g = r.get("gate_changes")
        if g and m != "birdnet":
            add(f"deepGrayUp{tag}", num(100 * g["nuevas"]["G"]["fraccion_ponderada"], 0))
            add(f"deepSilentUp{tag}", thin(g["nuevas"]["E"]["pares"]))
            add(f"deepValidLost{tag}", thin(g["perdidas"]["P"]["pares"]))
        if "m3_presupuesto" in r:
            add(f"deepAtt{tag}", num(100 * r["m3_presupuesto_S"], 1))
            add(f"deepAttFa{tag}", num(100 * r["m3_falsas_alarmas_S"], 1))
            for a, lab in (("0", "Zero"), ("6", "Six"), ("12", "Twelve"), ("18", "Eighteen"), ("24", "TwentyFour")):
                add(f"deepAtt{tag}{lab}", num(100 * r["m3_presupuesto"][a], 1))
            if "m3_presupuesto_S_delta" in r:
                d = r["m3_presupuesto_S_delta"]
                add(f"deepAttDelta{tag}", signed(100 * d["delta"], 1))
                add(f"deepAttCi{tag}", interval(100 * d["ci_low"], 100 * d["ci_high"], 1))
                d6 = r["m3_presupuesto_delta"]["6"]
                add(f"deepAttSixDelta{tag}", signed(100 * d6["delta"], 1))
                add(f"deepAttSixCi{tag}", interval(100 * d6["ci_low"], 100 * d6["ci_high"], 1))
    if "mixture_negatives_available" in run.get("mlp_neg_mezcla", {}):
        add("deepMixNegatives", thin(run["mlp_neg_mezcla"]["mixture_negatives_available"]))
    verifier = run.get("_probes", {}).get("verifier", {}).get("out_of_sample")
    if verifier:
        add("deepFlagValid", num(100 * verifier["validadas"]["fraccion_ponderada"], 1))
        add("deepFlagRejected", str(verifier["rechazadas"]["marcadas"]))
        add("deepFlagRejectedOf", str(verifier["rechazadas"]["pares"]))
        add("deepFlagUnreviewed", num(100 * verifier["sin_revisar"]["fraccion_ponderada"], 1))
        held = verifier.get("modelo_exportado_en_validacion_interna")
        if held:
            add("deepFlagHeldValid", num(100 * held["fraccion_ponderada"], 1))
            add("deepFlagHeldMarked", str(held["marcadas"]))
            add("deepFlagHeldOf", thin(held["validadas"]))
    (TAB / "macros_deep.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def figure_m3(methods: dict, focus: list[str]) -> None:
    """Cambio de exhaustividad respecto de BirdNET en cada nivel de atenuacion, con IC 95 %, para hasta tres modelos."""
    styles = [(C1, "D"), (C2, "s"), (C3, "^")]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(PR.LEVELS), dtype=float)
    step = 0.22
    offsets = (np.arange(len(focus)) - (len(focus) - 1) / 2) * step
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    for (color, mk), m, off in zip(styles, focus, offsets):
        d = methods[m]["m3_presupuesto_delta"]
        y = np.array([100 * d[f"{a:g}"]["delta"] for a in PR.LEVELS])
        lo = np.array([100 * d[f"{a:g}"]["ci_low"] for a in PR.LEVELS])
        hi = np.array([100 * d[f"{a:g}"]["ci_high"] for a in PR.LEVELS])
        ax.vlines(x + off, lo, hi, color=color, linewidth=1.6, alpha=0.55, zorder=2)
        ax.scatter(x + off, y, s=46, color=color, marker=mk, edgecolor=SURFACE, linewidth=1.2, label=PLAIN[m], zorder=3)
    base = methods["birdnet"]["m3_presupuesto"]
    ax.set_xticks(x)
    ax.set_xticklabels([("original" if a == 0 else f"−{a:g} dB") + f"\nBirdNET: {100 * base[f'{a:g}']:.0f} %" for a in PR.LEVELS])
    ax.set_xlim(-0.5, len(PR.LEVELS) - 0.5)
    ax.set_ylabel("Cambio de exhaustividad frente a BirdNET\n(puntos, mismo volumen de alertas)")
    ax.set_xlabel("Atenuación del canto validado sobre un fondo de su misma sesión")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_deep_m3")


def figure_tradeoff(methods: dict, free: list[str], bounded: list[str]) -> None:
    """Dos paneles con una fila por modelo: cambio de mAP limpia (seguridad) y cambio de exhaustividad atenuada, con IC 95 %."""
    names = free + bounded
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 0.33 * len(names) + 1.25), sharey=True, gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.06})
    y = np.arange(len(names))[::-1]
    lows = [methods[m]["map_limpio_delta"]["ci_low"] for m in names]
    highs = [methods[m]["map_limpio_delta"]["ci_high"] for m in names]
    left = min(min(lows), -0.02) - 0.006
    ax1.axvspan(left, -PR.NONINFERIORITY, color="#e34948", alpha=0.07, linewidth=0, zorder=0)
    ax1.axvline(-PR.NONINFERIORITY, color=MUTED, linewidth=0.8, zorder=1)
    ax1.axvline(0, color=GRID, linewidth=0.8, zorder=1)
    ax2.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    for yi, m in zip(y, names):
        r = methods[m]
        color, mk = (C3, "o") if m in bounded else (C1, "D")
        d1, d3 = r["map_limpio_delta"], r["m3_presupuesto_S_delta"]
        ax1.plot([d1["ci_low"], d1["ci_high"]], [yi, yi], color=color, linewidth=1.5, alpha=0.55, zorder=2)
        ax1.scatter([d1["delta"]], [yi], s=40, color=color, marker=mk, edgecolor=SURFACE, linewidth=1.1, zorder=3)
        ax2.plot([100 * d3["ci_low"], 100 * d3["ci_high"]], [yi, yi], color=color, linewidth=1.5, alpha=0.55, zorder=2)
        ax2.scatter([100 * d3["delta"]], [yi], s=40, color=color, marker=mk, edgecolor=SURFACE, linewidth=1.1, zorder=3)
    if bounded and free:
        for ax in (ax1, ax2):
            ax.axhline(len(bounded) - 0.5, color=GRID, linewidth=0.8, zorder=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels([PLAIN[m] for m in names])
    ax1.set_ylim(-0.6, len(names) - 0.4)
    ax1.set_xlim(left, max(max(highs), 0.0) + 0.006)
    ax1.text(left + 0.002, len(names) - 0.55, "no pasa el control", fontsize=7, color=INK_2, va="top")
    ax1.set_xlabel("Cambio de mAP limpia (seguridad: > −0,01)")
    ax2.set_xlabel("Cambio de exhaustividad en\ncantos atenuados (puntos)")
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    ax2.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save(fig, "fig_deep_tradeoff")


def figure_m2(methods: dict, names: list[str]) -> None:
    """Colocacion de los rechazos (M2) por modelo, con IC; linea de referencia en BirdNET."""
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(names) + 0.9))
    y = np.arange(len(names))[::-1]
    base = methods["birdnet"]["m2"]
    ax.axvline(base, color=MUTED, linewidth=0.8, zorder=1)
    ax.axvline(0.5, color=GRID, linewidth=0.8, zorder=1)
    for yi, m in zip(y, names):
        r = methods[m]
        c = r["m2_ci"]
        ax.plot([c["ci_low"], c["ci_high"]], [yi, yi], color=MUTED, linewidth=1.1, zorder=2)
        ax.scatter([r["m2"]], [yi], s=44, color=C_REF if m == "birdnet" else C1, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.text(min(c["ci_high"] + 0.012, 0.975), yi, f"{r['m2']:.2f}".replace(".", ","), va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([PLAIN[m] for m in names])
    ax.set_xlim(0.4, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0,5", "0,6", "0,7", "0,8", "0,9", "1"])
    ax.set_xlabel("M2: fracción de detecciones validadas que quedan por encima de cada rechazo")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig_deep_m2")


def main() -> None:
    doc = load()
    methods, ds = doc["methods"], doc["dataset"]
    run_path = RESULTS / "run_info_ra.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    probes_path = RESULTS / "run_info_probes.json"
    if probes_path.exists():
        run["_probes"] = json.loads(probes_path.read_text(encoding="utf-8"))
    TAB.mkdir(parents=True, exist_ok=True)
    table_sample(ds)
    table_cells(ds)
    table_never(ds)
    table_full(methods)
    table_results(methods)
    table_m3(methods)
    table_verifiers(methods, ds["m2_rejections"])
    table_ladder(doc["ladder"])
    table_grid(run)
    macros(doc, run)
    has = lambda m: m in methods and "m3_presupuesto_S_delta" in methods[m]
    free = [m for m in FREE if has(m)]
    bounded = [m for m in BOUNDED if has(m)]
    if free:
        deep = [m for m in ("b4", "b43", "post") if has(m)]
        top = deep[0] if deep else "mlp"
        focus = [m for m in ("mlp", top, f"{top}_acotado") if has(m)]
        figure_m3(methods, list(dict.fromkeys(focus)))
        figure_tradeoff(methods, free, bounded)
    figure_m2(methods, [m for m in ("birdnet", "dos_etapas", "lineal_desde_cero", "lineal_v1", "prototipos", "verificador", "mlp", "post", "b43", "b4") if m in methods])
    print("Tablas y figuras escritas en", TAB, "y", FIG)


if __name__ == "__main__":
    main()
