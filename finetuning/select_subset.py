"""Seleccion estratificada de grabaciones bajo un presupuesto de descarga.

Cuando no es posible copiar todo el conjunto de audios (p. ej. ancho de banda
limitado), se elige un subconjunto de grabaciones de 55 s que maximiza la
cobertura de positivos validados por especie, con una cuota maxima por especie
para que las especies dominantes (p. ej. Mimus longicaudatus) no acaparen el
presupuesto, y con un pequeno bono por diversidad de sesiones (grabadora x dia x
periodo) para permitir validacion cruzada agrupada.

Algoritmo (voraz, tipo cobertura de conjuntos ponderada):
    ganancia(f) = sum_s w_s * min(necesidad_s, c_{f,s})
                  + bono_negativo_duro * [f contiene Valid=0]
                  + bono_sesion * [la sesion de f aun no esta representada]
con w_s = 1/sqrt(total_s) y necesidad_s = min(total_s, cuota) - ya_cubierto_s.

Uso:
    python -m finetuning.select_subset --budget 220 --quota 60
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config, labels


def greedy_select(
    det: pd.DataFrame,
    *,
    budget: int,
    quota: int = 60,
    min_positives: int = config.MIN_POSITIVES_TARGET,
    hard_negative_bonus: float = 0.5,
    session_bonus: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = labels.target_species(det, min_positives)
    pos = det[(det["valid"] == 1) & det["species"].isin(targets)]
    counts = pos.groupby(["file_key", "species"])["start"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=targets, fill_value=0)

    all_files = det.drop_duplicates("file_key").set_index("file_key")[["device", "date", "period", "session", "fname"]]
    counts = counts.reindex(all_files.index, fill_value=0)

    hard_neg_files = set(det.loc[det["valid"] == 0, "file_key"])
    has_hard_neg = np.array([fk in hard_neg_files for fk in counts.index], dtype=float)

    c = counts.to_numpy(dtype=float)
    total = c.sum(axis=0)
    need = np.minimum(total, quota)
    weights = 1.0 / np.sqrt(np.maximum(total, 1.0))
    sessions = all_files["session"].to_numpy()

    chosen: list[int] = []
    chosen_mask = np.zeros(len(counts), dtype=bool)
    seen_sessions: set[str] = set()
    for _ in range(min(budget, len(counts))):
        gain = (np.minimum(c, need) * weights).sum(axis=1)
        gain += hard_negative_bonus * weights.mean() * has_hard_neg
        new_session = np.array([s not in seen_sessions for s in sessions], dtype=float)
        gain += session_bonus * weights.mean() * new_session
        gain[chosen_mask] = -np.inf
        best = int(np.argmax(gain))
        if not np.isfinite(gain[best]) or gain[best] <= 0:
            break
        chosen.append(best)
        chosen_mask[best] = True
        seen_sessions.add(sessions[best])
        need = np.maximum(need - c[best], 0.0)
        has_hard_neg[best] = 0.0

    selected = all_files.iloc[chosen].copy()
    selected["order"] = np.arange(len(chosen))
    selected["target_positive_windows"] = c[chosen].sum(axis=1).astype(int)

    covered = c[chosen].sum(axis=0)
    coverage = pd.DataFrame(
        {
            "species": targets,
            "total_positive_windows": total.astype(int),
            "quota": np.minimum(total, quota).astype(int),
            "selected_positive_windows": covered.astype(int),
        }
    )
    coverage["coverage_of_quota"] = (coverage["selected_positive_windows"].clip(upper=coverage["quota"]) / coverage["quota"]).round(3)
    return selected.reset_index(), coverage


def topup_select(
    det: pd.DataFrame,
    have: set[str],
    *,
    budget: int,
    min_sessions: int = 4,
    quota: int = 60,
    min_positives: int = config.MIN_POSITIVES_TARGET,
    exclude: set[str] | None = None,
    cheap_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Seleccion complementaria a partir de lo ya copiado (``have``).

    Prioriza (1) que cada especie objetivo aparezca en al menos ``min_sessions``
    sesiones distintas -necesario para validar agrupando por sesion- y
    (2) completar una cuota mayor de positivos. ``cheap_keys`` (carpetas
    grabadora/fecha/periodo ya localizadas en Drive) recibe un pequeno bono
    porque su copia es mas barata.
    """
    exclude = exclude or set()
    cheap_keys = cheap_keys or set()
    targets = labels.target_species(det, min_positives)
    pos = det[(det["valid"] == 1) & det["species"].isin(targets)]
    files = det.drop_duplicates("file_key").set_index("file_key")[["device", "date", "period", "session", "fname"]]

    covered_sessions = {s: set(pos.loc[pos["file_key"].isin(have) & (pos["species"] == s), "session"]) for s in targets}
    available_sessions = pos.groupby("species")["session"].nunique().to_dict()
    counts = pos.groupby(["file_key", "species"])["start"].nunique().unstack(fill_value=0).reindex(columns=targets, fill_value=0)
    have_counts = counts.reindex(sorted(have & set(counts.index))).sum(axis=0)
    need = np.maximum(np.minimum(pos.groupby("species").size().reindex(targets).to_numpy(), quota) - have_counts.to_numpy(), 0.0)
    weights = 1.0 / np.sqrt(np.maximum(pos.groupby("species").size().reindex(targets).to_numpy(), 1.0))

    candidates = [fk for fk in counts.index if fk not in have and fk not in exclude]
    chosen: list[str] = []
    for _ in range(budget):
        best_key, best_gain = None, 0.0
        for fk in candidates:
            if fk in chosen:
                continue
            row = counts.loc[fk].to_numpy(dtype=float)
            sess = files.at[fk, "session"]
            gain = float((np.minimum(row, need) * weights).sum())
            for j, s in enumerate(targets):
                if row[j] > 0 and sess not in covered_sessions[s] and len(covered_sessions[s]) < min(min_sessions, available_sessions.get(s, 0)):
                    gain += 3.0 * weights[j] * np.sqrt(row[j])
            key = f"{files.at[fk, 'device']}/{files.at[fk, 'date']}/{files.at[fk, 'period']}"
            if key in cheap_keys:
                gain *= 1.15
            if gain > best_gain:
                best_key, best_gain = fk, gain
        if best_key is None:
            break
        chosen.append(best_key)
        row = counts.loc[best_key].to_numpy(dtype=float)
        need = np.maximum(need - row, 0.0)
        sess = files.at[best_key, "session"]
        for j, s in enumerate(targets):
            if row[j] > 0:
                covered_sessions[s].add(sess)
    out = files.loc[chosen].reset_index()
    out["order"] = np.arange(len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, default=220, help="Numero maximo de grabaciones a copiar.")
    parser.add_argument("--quota", type=int, default=60, help="Cuota de ventanas positivas por especie.")
    parser.add_argument("--curve", action="store_true", help="Imprime la curva cobertura vs presupuesto.")
    args = parser.parse_args()

    config.ensure_dirs()
    det = labels.load_detections()

    if args.curve:
        for budget in (50, 100, 150, 200, 250, 300, 400, 600):
            sel, cov = greedy_select(det, budget=budget, quota=args.quota)
            print(
                f"budget={budget:4d} files={len(sel):4d} sessions={sel['session'].nunique():3d} "
                f"devices={sel['device'].nunique()} mean_cov={cov['coverage_of_quota'].mean():.3f} "
                f"min_cov={cov['coverage_of_quota'].min():.3f} pos_windows={int(cov['selected_positive_windows'].sum())}"
            )
        return

    selected, coverage = greedy_select(det, budget=args.budget, quota=args.quota)
    out_csv = config.WORK_DIR / "subset_files.csv"
    selected.to_csv(out_csv, index=False)
    coverage.to_csv(config.RESULTS_DIR / "subset_coverage.csv", index=False)
    summary = {
        "budget": args.budget,
        "quota": args.quota,
        "files": int(len(selected)),
        "sessions": int(selected["session"].nunique()),
        "devices": sorted(selected["device"].unique().tolist()),
        "mean_coverage_of_quota": float(coverage["coverage_of_quota"].mean()),
        "selected_positive_windows_targets": int(coverage["selected_positive_windows"].sum()),
    }
    (config.RESULTS_DIR / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
