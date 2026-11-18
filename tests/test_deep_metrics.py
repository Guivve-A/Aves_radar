"""Pruebas de la precision promedio ponderada y de utilidades del paquete finetuning.deep (sin datos reales)."""

import unittest

import numpy as np

try:
    from sklearn.metrics import average_precision_score

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

from finetuning.deep import metrics as M


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

    def test_ranked_pair_reuses_the_order_for_bootstrap_weights(self):
        rng = np.random.default_rng(2)
        y = (rng.random(300) < 0.25).astype(float)
        s = np.round(rng.normal(size=300) + y, 1)
        rp = M.RankedPair(y, s)
        for _ in range(5):
            w = rng.integers(0, 4, 300).astype(float)       # multiplicidades de bootstrap, con ceros
            keep = w > 0
            self.assertAlmostEqual(rp.ap(w), average_precision_score(y[keep], s[keep], sample_weight=w[keep]), places=10)
        self.assertTrue(np.isnan(rp.ap(np.where(y > 0, 0.0, 1.0))))   # sin positivos no hay AP


class SplitTest(unittest.TestCase):
    def test_split_signal_matches_birdnet_convention(self):
        from finetuning.deep.extract import split_signal

        chunks = split_signal(np.ones(55 * 48_000, dtype=np.float32))
        self.assertEqual(chunks.shape, (19, 144_000))          # 18 ventanas completas + 1 s rellenado con ceros
        self.assertEqual(float(chunks[-1][:48_000].min()), 1.0)
        self.assertEqual(float(np.abs(chunks[-1][48_000:]).max()), 0.0)
        self.assertEqual(len(split_signal(np.ones(int(54.5 * 48_000), dtype=np.float32))), 18)  # resto < 1 s: se descarta
        self.assertEqual(len(split_signal(np.ones(72_000, dtype=np.float32))), 1)   # 1,5 s: una ventana rellenada
        self.assertEqual(len(split_signal(np.ones(24_000, dtype=np.float32))), 0)   # menos de 1 s: sin ventanas

    def test_inner_split_keeps_sessions_together(self):
        from finetuning.deep.data import inner_split

        rng = np.random.default_rng(0)
        groups = np.array([f"S{g:02d}" for g in rng.integers(0, 20, 2000)])
        val = inner_split(groups)
        for g in np.unique(groups):
            self.assertEqual(len(set(val[groups == g])), 1)
        self.assertEqual(len(set(groups[val])), 3)             # 15 % de 20 sesiones
        self.assertTrue(np.array_equal(val, inner_split(groups)))


if __name__ == "__main__":
    unittest.main()
