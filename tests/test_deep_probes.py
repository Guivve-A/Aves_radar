"""Pruebas de las sondas desde cero y del verificador (datos sinteticos; se omiten si PyTorch no esta instalado)."""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch  # noqa: F401

    from finetuning.deep import probes as PB
    from finetuning.deep import protocol as PR

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def synthetic(n=1200, s=3, seed=0):
    """Embeddings con una direccion por especie; el 'logit de BirdNET' es una version ruidosa de la verdad."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(s, 1024)).astype(np.float32)
    present = rng.random((n, s)) < 0.15
    emb = (rng.normal(0, 1, (n, 1024)) + 0.75 * present.astype(np.float64) @ dirs).astype(np.float32)
    logit = (-6 + 9 * present + rng.normal(0, 2.5, (n, s))).astype(np.float32)
    prob = 1 / (1 + np.exp(-logit))
    Y = (present & (prob >= 0.25)).astype(np.float32)          # solo se "valida" lo que BirdNET propone
    hard = (~present) & (prob >= 0.6) & (rng.random((n, s)) < 0.5)
    return {"emb": emb, "logit": logit, "Y": Y, "hard_neg": hard, "weight": rng.choice([1.0, 10.0], n).astype(np.float32)}, present


@unittest.skipUnless(HAVE_TORCH, "requiere PyTorch")
class ProbesTest(unittest.TestCase):
    def test_candidates_are_ignored_and_rejections_upweighted(self):
        d, _ = synthetic()
        cfg = PB.ProbeConfig(candidates="ignore", candidate_threshold=0.1, hard_negative_weight=20.0, use_sampling_weights=False)
        W, pw = PB.pair_weights(d["Y"], d["logit"], d["hard_neg"], d["weight"], cfg)
        prob = 1 / (1 + np.exp(-d["logit"]))
        cand = (prob >= 0.1) & (d["Y"] == 0) & (~d["hard_neg"])
        self.assertTrue(np.all(W[cand] == 0))
        self.assertTrue(np.all(W[d["hard_neg"]] == 20.0))
        self.assertTrue(np.all(pw >= 1.0))
        strict, _ = PB.pair_weights(d["Y"], d["logit"], d["hard_neg"], d["weight"], PB.ProbeConfig(candidates="negative", use_sampling_weights=False))
        self.assertTrue(np.all(strict[cand] == 1.0))

    def test_probe_learns_and_training_is_deterministic(self):
        d, present = synthetic()
        half = len(d["Y"]) // 2
        tr = {k: v[:half] for k, v in d.items()}
        cfg = PB.ProbeConfig(epochs=15)
        p1 = PB.predict(PB.fit(cfg, tr).model, d["emb"][half:])
        p2 = PB.predict(PB.fit(cfg, tr).model, d["emb"][half:])
        np.testing.assert_allclose(p1, p2, atol=1e-6)
        truth = present[half:]
        for k in range(truth.shape[1]):
            self.assertGreater(p1[truth[:, k], k].mean(), p1[~truth[:, k], k].mean() + 0.2)

    def test_validation_metric_picks_the_epoch(self):
        d, _ = synthetic(n=400)
        res = PB.fit(PB.ProbeConfig(epochs=3, patience=1), d, d, val_metric=lambda prob: 0.0)
        self.assertEqual(res.best_epoch, 0)                     # nada mejora: se queda con el punto de partida
        self.assertEqual(len(res.history), 2)                   # y para tras agotar la paciencia

    def test_exported_weights_reproduce_the_probe_on_raw_embeddings(self):
        d, _ = synthetic(n=600)
        for l2 in (False, True):
            model = PB.fit(PB.ProbeConfig(epochs=4, l2_input=l2), d).model
            W, b = model.raw_weights()
            x = d["emb"].astype(np.float64)
            if l2:
                x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            np.testing.assert_allclose(1 / (1 + np.exp(-(x @ W + b))), PB.predict(model, d["emb"]), atol=2e-4)

    def test_cutoffs_keep_the_requested_share_of_validated_detections(self):
        rng = np.random.default_rng(3)
        n, s = 4000, 3
        cat = np.full((n, s), PR.E, np.int8)
        cat[:1000, 0] = PR.P
        cat[1000:1040, 0] = PR.H
        cat[:10, 1] = PR.P                                      # muy pocas validadas: esta especie no se marca
        cat[2000:2600, 2] = PR.P
        cat[2600:2700, 2] = PR.U
        scores = rng.random((n, s)).astype(np.float32)
        scores[1000:1040, 0] *= 0.05                             # los rechazos puntuan bajo
        data = SimpleNamespace(weight=np.ones(n, np.float32))
        cut, n_pos = PB.verification_cutoffs(data, cat, scores)
        self.assertEqual(list(n_pos), [1000, 10, 600])
        self.assertEqual(float(cut[1]), 0.0)
        self.assertAlmostEqual(float((scores[:1000, 0] >= cut[0]).mean()), 0.95, places=2)
        summary = PB.flagged_summary(data, cat, scores, cut)
        self.assertAlmostEqual(summary["validadas"]["fraccion_ponderada"], 0.05 * 1600 / 1610, places=2)
        self.assertGreater(summary["rechazadas"]["marcadas"], 30)
        self.assertEqual(summary["sin_revisar"]["pares"], 100)


if __name__ == "__main__":
    unittest.main()
