"""Pruebas de las metricas ponderadas y utilidades del paquete finetuning.deep (sin datos reales)."""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

from finetuning.deep import metrics as M


def toy_data(n=600, s=4, seed=0):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, s)) < 0.08).astype(np.float32)
    logit = rng.normal(-4, 2, (n, s)).astype(np.float32) + 5 * Y
    groups = np.array([f"S{g:02d}" for g in rng.integers(0, 12, n)])
    weight = rng.choice([1.0, 10 / 3, 10.0], n).astype(np.float32)
    hard = (rng.random((n, s)) < 0.02) & (Y == 0)
    prob = 1 / (1 + np.exp(-logit))
    data = SimpleNamespace(Y=Y, logit=logit, weight=weight, groups=groups, species=[f"sp{k}" for k in range(s)], uncertain=(prob >= 0.5) & (Y == 0), hard_neg=hard)
    return data, prob


@unittest.skipUnless(HAVE_SKLEARN, "requiere scikit-learn")
class WeightedMetricsTest(unittest.TestCase):
    def test_ap_matches_sklearn_with_weights_and_ties(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            n = int(rng.integers(20, 400))
            y = (rng.random(n) < 0.2).astype(float)
            if y.sum() == 0:
                y[0] = 1
            s = np.round(rng.normal(size=n) + y, 1)  # redondeo -> empates
            w = rng.choice([1.0, 3.0, 10.0], n)
            self.assertAlmostEqual(M.weighted_ap(y, s, w), average_precision_score(y, s, sample_weight=w), places=10)
            self.assertAlmostEqual(M.weighted_ap(y, s), average_precision_score(y, s), places=10)

    def test_auc_matches_sklearn_with_weights_and_ties(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            n = int(rng.integers(20, 400))
            y = (rng.random(n) < 0.3).astype(float)
            y[:2] = [0, 1]
            s = np.round(rng.normal(size=n) + y, 1)
            w = rng.choice([1.0, 3.0, 10.0], n)
            self.assertAlmostEqual(M.weighted_auc(y, s, w), roc_auc_score(y, s, sample_weight=w), places=10)

    def test_metrics_table_regimes(self):
        data, prob = toy_data()
        strict = M.metrics_table(data, {"m": prob}, "estricto")
        masked = M.metrics_table(data, {"m": prob}, "enmascarado")
        self.assertEqual(len(strict), len(data.species))
        # enmascarar pares inciertos solo quita negativos con score alto: la AP no puede bajar
        self.assertTrue(np.all(masked["ap"].to_numpy() >= strict["ap"].to_numpy() - 1e-12))
        k = 0
        self.assertAlmostEqual(strict["ap"].iloc[k], average_precision_score(data.Y[:, k], prob[:, k], sample_weight=data.weight), places=10)

    def test_session_bootstrap_identical_methods_give_zero_difference(self):
        data, prob = toy_data()
        boot = M.SessionBootstrap(data, reps=50, seed=3)
        res = boot.compare("a", prob, "b", prob.copy())
        self.assertEqual(res["delta_mean"], 0.0)
        better = boot.compare("oracle", data.Y + 0.01 * prob, "a", prob)
        self.assertGreater(better["ci_low"], 0.0)

    def test_reviewed_only_perfect_and_inverted(self):
        data, _ = toy_data()
        perfect = data.Y - data.hard_neg.astype(float)
        table = M.reviewed_only(data, {"perfecto": perfect, "invertido": -perfect}, reps=20)
        self.assertAlmostEqual(table.loc[0, "auc_pooled"], 1.0)
        self.assertAlmostEqual(table.loc[1, "auc_pooled"], 0.0)
        self.assertEqual(int(table.loc[0, "n_rejected"]), int(data.hard_neg.sum()))


class SplitAndFoldsTest(unittest.TestCase):
    def test_split_signal_matches_birdnet_convention(self):
        from finetuning.deep.extract import split_signal

        chunks = split_signal(np.ones(55 * 48_000, dtype=np.float32))
        self.assertEqual(chunks.shape, (19, 144_000))          # 18 ventanas completas + 1 s rellenado con ceros
        self.assertEqual(float(chunks[-1][:48_000].min()), 1.0)
        self.assertEqual(float(np.abs(chunks[-1][48_000:]).max()), 0.0)
        self.assertEqual(len(split_signal(np.ones(int(54.5 * 48_000), dtype=np.float32))), 18)  # resto < 1 s: se descarta
        self.assertEqual(len(split_signal(np.ones(72_000, dtype=np.float32))), 1)   # 1,5 s: una ventana rellenada
        self.assertEqual(len(split_signal(np.ones(24_000, dtype=np.float32))), 0)   # menos de 1 s: sin ventanas

    def test_group_folds_keep_sessions_together(self):
        from finetuning.deep.data import group_folds, inner_split

        rng = np.random.default_rng(0)
        groups = np.array([f"S{g:02d}" for g in rng.integers(0, 20, 2000)])
        folds = group_folds(groups)
        for g in np.unique(groups):
            self.assertEqual(len(set(folds[groups == g])), 1)
        self.assertTrue(np.array_equal(folds, group_folds(groups)))
        self.assertEqual(sorted(set(folds)), [0, 1, 2, 3, 4])
        val = inner_split(groups)
        for g in np.unique(groups):
            self.assertEqual(len(set(val[groups == g])), 1)


if __name__ == "__main__":
    unittest.main()
