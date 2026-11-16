"""Pruebas de las cabezas profundas con datos sinteticos (se omiten si PyTorch no esta instalado)."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    from finetuning.deep import heads as H

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def synthetic(n_rec=120, t=10, s=3, seed=0):
    """Embeddings con una direccion por especie; el 'logit de BirdNET' es una version ruidosa de la verdad."""
    rng = np.random.default_rng(seed)
    n = n_rec * t
    dirs = rng.normal(size=(s, 1024)).astype(np.float32)
    present = rng.random((n, s)) < 0.15
    emb = (rng.normal(0, 1, (n, 1024)) + 0.75 * present.astype(np.float64) @ dirs).astype(np.float32)
    logit = (-6 + 9 * present + rng.normal(0, 2.5, (n, s))).astype(np.float32)
    prob = 1 / (1 + np.exp(-logit))
    Y = (present & (prob >= 0.25)).astype(np.float32)          # solo se "valida" lo que BirdNET propone
    hard = (~present) & (prob >= 0.6) & (rng.random((n, s)) < 0.5)
    return {
        "emb": emb,
        "logit": logit,
        "Y": Y,
        "hard_neg": hard,
        "weight": rng.choice([1.0, 10.0], n).astype(np.float32),
        "rec": np.repeat(np.arange(n_rec), t),
        "pos": np.tile(np.arange(t), n_rec),
    }, present


@unittest.skipUnless(HAVE_TORCH, "requiere PyTorch")
class DeepHeadsTest(unittest.TestCase):
    def test_residual_starts_exactly_at_birdnet(self):
        d, _ = synthetic()
        cfg = H.HeadConfig(kind="residual", epochs=0)
        res = H.fit(cfg, d, None)
        prob = H.predict(res.model, d["emb"], d["logit"])
        np.testing.assert_allclose(prob, 1 / (1 + np.exp(-d["logit"])), atol=1e-5)

    def test_context_starts_exactly_at_birdnet(self):
        d, _ = synthetic()
        res = H.fit(H.HeadConfig(kind="context", epochs=0), d, None)
        prob = H.predict(res.model, d["emb"], d["logit"], d["rec"], d["pos"])
        np.testing.assert_allclose(prob, 1 / (1 + np.exp(-d["logit"])), atol=1e-5)

    def test_candidates_are_ignored_and_hard_negatives_upweighted(self):
        d, _ = synthetic()
        cfg = H.HeadConfig(candidates="ignore", candidate_threshold=0.1, hard_negative_weight=20.0, use_sampling_weights=False)
        W, pw = H.pair_weights(d["Y"], d["logit"], d["hard_neg"], d["weight"], cfg)
        prob = 1 / (1 + np.exp(-d["logit"]))
        cand = (prob >= 0.1) & (d["Y"] == 0) & (~d["hard_neg"])
        self.assertTrue(np.all(W[cand] == 0))
        self.assertTrue(np.all(W[d["hard_neg"]] == 20.0))
        self.assertTrue(np.all(pw >= 1.0))
        strict, _ = H.pair_weights(d["Y"], d["logit"], d["hard_neg"], d["weight"], H.HeadConfig(candidates="negative", use_sampling_weights=False))
        self.assertTrue(np.all(strict[cand] == 1.0))

    def test_linear_probe_learns_and_training_is_deterministic(self):
        d, present = synthetic()
        half = len(d["Y"]) // 2
        tr = {k: v[:half] for k, v in d.items()}
        te = {k: v[half:] for k, v in d.items()}
        cfg = H.HeadConfig(kind="linear", epochs=15, lr=2e-3, patience=50)
        p1 = H.predict(H.fit(cfg, tr, None).model, te["emb"], te["logit"])
        p2 = H.predict(H.fit(cfg, tr, None).model, te["emb"], te["logit"])
        np.testing.assert_allclose(p1, p2, atol=1e-6)
        truth = present[half:]
        for k in range(truth.shape[1]):
            self.assertGreater(p1[truth[:, k], k].mean(), p1[~truth[:, k], k].mean() + 0.2)

    def test_early_stopping_keeps_start_when_nothing_improves(self):
        d, _ = synthetic()
        res = H.fit(H.HeadConfig(kind="residual", epochs=3, patience=1), d, d, val_metric=lambda prob: 0.0)
        self.assertEqual(res.best_epoch, 0)
        prob = H.predict(res.model, d["emb"], d["logit"])
        np.testing.assert_allclose(prob, 1 / (1 + np.exp(-d["logit"])), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
