"""Pruebas del protocolo de evaluacion consciente de la revision (datos sinteticos, sin PyTorch)."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from finetuning.deep import protocol as PR


def logit(p):
    return float(np.log(p / (1 - p)))


def toy(n_sessions=20, per=80, s=3, seed=0, reviewed_share=0.8):
    """Sesiones de 4 grabadoras. BirdNET propone ~12 % de los pares; cada celda sesion x especie se revisa entera o nada."""
    rng = np.random.default_rng(seed)
    n = n_sessions * per
    session = np.repeat([f"AM{1 + i % 4:03d}_202507{10 + i:02d}_Audio_Sunrise" for i in range(n_sessions)], per)
    z = rng.normal(-6.0, 1.0, (n, s))
    z = np.minimum(z, logit(0.04))                                   # el fondo queda por debajo de 0,05
    gray = rng.random((n, s)) < 0.05
    z[gray] = rng.uniform(logit(0.06), logit(0.24), int(gray.sum()))
    proposed = (rng.random((n, s)) < 0.12) & ~gray
    z[proposed] = rng.uniform(logit(0.26), 6.0, int(proposed.sum()))
    Y = np.zeros((n, s), np.float32)
    hard = np.zeros((n, s), bool)
    reviewed = {}
    for sess in np.unique(session):
        rows = np.flatnonzero(session == sess)
        for k in range(s):
            reviewed[(sess, k)] = bool(rng.random() < reviewed_share)
            if reviewed[(sess, k)]:
                prop = rows[proposed[rows, k]]
                rejected = rng.random(len(prop)) < 0.08
                Y[prop[~rejected], k] = 1.0
                hard[prop[rejected], k] = True
    meta = pd.DataFrame({"session": session, "device": [x.split("_")[0] for x in session], "rec": np.arange(n) // 10})
    data = SimpleNamespace(
        Y=Y,
        hard_neg=hard,
        logit=z.astype(np.float32),
        weight=rng.choice([1.0, 10 / 3, 10.0], n).astype(np.float32),
        groups=session,
        meta=meta,
        species=[f"sp{k}" for k in range(s)],
    )
    return data, proposed, reviewed


def toy_mixtures(data, cat, drop_per_db=0.25, seed=1):
    """Una mezcla de prueba por canto validado y nivel: el logit de BirdNET cae con la atenuacion."""
    rng = np.random.default_rng(seed)
    sources = np.flatnonzero((cat == PR.P).any(axis=1))
    src, kind, att, z = [], [], [], []
    for a in PR.LEVELS[1:]:
        src.append(sources)
        kind += ["test"] * len(sources)
        att += [a] * len(sources)
        z.append(data.logit[sources] - drop_per_db * a + rng.normal(0, 0.3, (len(sources), data.Y.shape[1])))
    quiet = np.flatnonzero((cat == PR.E).all(axis=1))[:50]
    src.append(quiet)
    kind += ["bg_test"] * len(quiet)
    att += [0.0] * len(quiet)
    z.append(data.logit[quiet])
    src = np.concatenate(src)
    return SimpleNamespace(src=src, kind=np.array(kind), plan=pd.DataFrame({"atten_db": att, "bg": src}), logit=np.concatenate(z).astype(np.float32))


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


class CategoriesTest(unittest.TestCase):
    def setUp(self):
        self.data, self.proposed, self.reviewed = toy()
        self.cat = PR.categories(self.data)

    def test_every_pair_gets_the_category_of_what_is_known_about_it(self):
        data, cat = self.data, self.cat
        s0 = sigmoid(data.logit)
        self.assertTrue(np.array_equal(cat == PR.P, data.Y > 0))
        self.assertTrue(np.array_equal(cat == PR.H, data.hard_neg))
        self.assertTrue(np.all(s0[cat == PR.E] < PR.SILENT))
        self.assertTrue(np.all((s0[cat == PR.G] >= PR.SILENT) & (s0[cat == PR.G] < PR.GATE)))
        self.assertTrue(np.all(s0[np.isin(cat, (PR.P, PR.H, PR.U, PR.R0))] >= PR.GATE))
        self.assertEqual(int((cat == PR.X).sum()), 0)

    def test_unlisted_proposals_are_unknown_only_in_cells_nobody_reviewed(self):
        data, cat = self.data, self.cat
        for k in range(data.Y.shape[1]):
            for sess in np.unique(data.groups):
                rows = np.flatnonzero(data.groups == sess)
                unlisted = rows[self.proposed[rows, k] & (data.Y[rows, k] == 0) & ~data.hard_neg[rows, k]]
                expected = PR.R0 if self.reviewed[(sess, k)] else PR.U
                self.assertTrue(np.all(cat[unlisted, k] == expected))
        cells = PR.review_cells(data)
        self.assertEqual(set(cells.status), {"revisada", "nunca"})

    def test_listed_pair_below_the_gate_is_excluded(self):
        data = self.data
        j, k = map(int, np.argwhere(data.Y > 0)[0])
        data.logit[j, k] = logit(0.10)
        self.assertEqual(int(PR.categories(data)[j, k]), PR.X)

    def test_folds_keep_sessions_together_and_spread_rejections(self):
        data, cat = self.data, self.cat
        folds = PR.make_folds(data, cat)
        self.assertEqual(sorted(set(folds)), [0, 1, 2, 3, 4])
        for sess in np.unique(data.groups):
            self.assertEqual(len(set(folds[data.groups == sess])), 1)
        self.assertTrue(np.array_equal(folds, PR.make_folds(data, cat)))
        rejections = np.array([(cat[folds == f] == PR.H).sum() for f in range(5)])
        self.assertGreater(rejections.min(), 0)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.data, _, _ = toy(seed=3)
        self.cat = PR.categories(self.data)
        self.rs = PR.Resampler(self.data, reps=30, seed=1)
        self.s0 = sigmoid(self.data.logit)

    def test_resampler_draws_sessions_within_each_recorder(self):
        dev = pd.Series(self.data.meta["device"].to_numpy()).groupby(self.rs.inv).first().to_numpy()
        for d in np.unique(dev):
            self.assertTrue(np.all(self.rs.mult[:, dev == d].sum(axis=1) == (dev == d).sum()))

    def test_clean_map_ignores_unknown_pairs(self):
        apev = PR.APEvaluator(self.data, self.cat, self.rs)
        base, _, _ = apev.evaluate(self.s0, "limpio", boot=False)
        moved = self.s0.copy()
        moved[np.isin(self.cat, (PR.U, PR.G))] = 0.999             # cambiar lo desconocido no altera la mAP limpia
        same, _, _ = apev.evaluate(moved, "limpio", boot=False)
        self.assertAlmostEqual(base, same, places=12)
        strict, _, _ = apev.evaluate(moved, "estricto", boot=False)
        self.assertLess(strict, base)

    def test_selection_leak_index_is_half_without_a_learned_correction(self):
        # sin correccion, la diferencia de logits es ruido de redondeo: el indice queda en 0,5 salvo azar
        self.assertAlmostEqual(PR.selection_leak_index(self.data, self.cat, self.s0)["sli"], 0.5, delta=0.03)
        leaky = self.s0.copy()
        leaky[self.cat == PR.P] = np.minimum(leaky[self.cat == PR.P] * 1.5, 0.999)     # sube solo lo revisado
        self.assertGreater(PR.selection_leak_index(self.data, self.cat, leaky)["sli"], 0.95)

    def test_rejection_placement_extremes_and_ties(self):
        rej = PR.RejectionPlacement(self.data, self.cat, self.rs)
        self.assertEqual(rej.n, int((self.cat == PR.H).sum()))
        good = np.where(self.cat == PR.H, 0.0, 1.0).astype(np.float32)
        point, _, boots = rej.evaluate(good)
        self.assertEqual(point, 1.0)
        self.assertTrue(np.all(boots[np.isfinite(boots)] == 1.0))
        self.assertEqual(rej.evaluate(1.0 - good, boot=False)[0], 0.0)
        self.assertEqual(rej.evaluate(np.ones_like(good), boot=False)[0], 0.5)
        self.assertEqual(rej.removed_at_retention(good)["removed"], rej.n)
        self.assertEqual(rej.removed_at_retention(np.ones_like(good))["removed"], 0)

    def test_attenuation_benchmark_gives_every_model_the_alert_volume_of_birdnet(self):
        data, cat = self.data, self.cat
        mix = toy_mixtures(data, cat)
        bench = PR.AttenuationBenchmark(data, cat, mix, self.rs, min_pos=5)
        bn_mix = sigmoid(mix.logit)
        curve, s, reps, table = bench.evaluate(self.s0, bn_mix)
        self.assertEqual(curve[0], 1.0)                              # todo canto validado supera 0,25 por construccion
        self.assertTrue(np.all(np.diff(curve) <= 1e-9))              # menos senal, menos deteccion
        self.assertEqual(reps.shape, (30, len(PR.LEVELS)))
        self.assertEqual(len(table), len(bench.species))
        # una transformacion monotona del score no cambia nada: el umbral se fija por volumen de alertas
        curve2, s2, _, _ = bench.evaluate(self.s0**2, bn_mix**2, boot=False)
        self.assertTrue(np.allclose(curve, curve2))
        # un modelo que reconoce mejor las mezclas gana exhaustividad sin emitir mas alertas en el audio real
        _, better, _, _ = bench.evaluate(self.s0, sigmoid(mix.logit + 2.0), boot=False)
        self.assertGreater(better, s)
        # un modelo que ademas dispara en ventanas reales pierde: su umbral sube para respetar el presupuesto
        noisy = self.s0.copy()
        raised = np.flatnonzero((cat == PR.E).all(axis=1))[::3]
        noisy[raised] = 0.99
        curve3, _, _, _ = bench.evaluate(noisy, bn_mix, boot=False)
        self.assertLess(curve3[0], 1.0)

    def test_bootstrap_threshold_ignores_windows_outside_the_replicate(self):
        data, cat = self.data, self.cat
        mix = toy_mixtures(data, cat)
        bench = PR.AttenuationBenchmark(data, cat, mix, self.rs, min_pos=5)
        bn_mix = sigmoid(mix.logit)
        prep = bench.prepare(self.s0, bn_mix)
        for r in range(self.rs.reps):
            m = self.rs.rows(r)
            got = bench.recalls(prep, m)
            for i, p in enumerate(prep):
                for j, a in enumerate(PR.LEVELS):
                    ps, owner = p["levels"][a]
                    pw = bench.w[owner] * m[owner]
                    if pw.sum() > 0 and np.isfinite(got[i, j]):
                        # el umbral de BirdNET nunca baja de 0,25: su exhaustividad no supera la del umbral fijo
                        self.assertLessEqual(got[i, j], float((pw * (ps >= PR.GATE)).sum() / pw.sum()) + 1e-12)

    def test_alert_volume_is_matched_exactly_by_interpolation(self):
        bench = PR.AttenuationBenchmark.__new__(PR.AttenuationBenchmark)
        bench.fa_rate = 0.01
        levels = {a: (np.array([0.35, 0.45]), np.array([3, 4])) for a in PR.LEVELS}

        def prep(alert):
            return [{"k": 0, "real_sorted": np.array([0.1, 0.2, 0.3, 0.4, 0.5]), "real_rows": np.arange(5), "alert": np.array(alert), "levels": levels}]

        # BirdNET marca peso 7. El modelo solo puede marcar 2 (>= 0,4) o 12 (>= 0,3): se interpola a partes iguales.
        bench.w = np.array([1.0, 5.0, 10.0, 1.0, 1.0])
        got = bench.recalls(prep([False, True, False, True, True]))
        self.assertTrue(np.allclose(got, 0.5 * 0.5 + 0.5 * 1.0))
        # Si el volumen cuadra con un umbral, no hay interpolacion.
        got = bench.recalls(prep([False, False, False, True, True]))
        self.assertTrue(np.allclose(got, 0.5))
        # Una ventana de peso cero justo por debajo del corte no puede fijar el umbral (0,3 y no 0,2).
        bench.w = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
        mult = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
        low = {a: (np.array([0.25, 0.45]), np.array([3, 4])) for a in PR.LEVELS}
        p = prep([False, False, True, True, True])
        p[0]["levels"] = low
        self.assertTrue(np.allclose(bench.recalls(p, mult), 0.5))
        # Si ni la ventana mas alta cabe en el presupuesto, se marca con la probabilidad que cuadra el volumen.
        bench.w = np.array([1.0, 1.0, 1.0, 1.0, 4.0])
        top = {a: (np.array([0.9]), np.array([4])) for a in PR.LEVELS}
        p = prep([True, False, False, False, False])
        p[0]["levels"] = top
        self.assertTrue(np.allclose(bench.recalls(p), 0.25))

    def test_bounded_correction_cannot_swap_silent_and_validated_pairs(self):
        data, cat = self.data, self.cat
        rng = np.random.default_rng(5)
        wild = sigmoid(rng.normal(0, 6, data.logit.shape))          # modelo arbitrario
        safe = PR.bounded(wild, data.logit)
        self.assertLess(float(safe[cat == PR.E].max()), float(safe[cat == PR.P].min()))
        self.assertLessEqual(float(np.abs(PR._logit(safe) - np.clip(data.logit, -16, 16)).max()), PR.BOUND + 1e-4)
        self.assertTrue(np.allclose(PR.bounded(self.s0, data.logit), self.s0, atol=1e-6))
        apev = PR.APEvaluator(data, cat, self.rs)
        base, _, _ = apev.evaluate(self.s0, "limpio", boot=False)
        bounded_map, _, _ = apev.evaluate(safe, "limpio", boot=False)
        wild_map, _, _ = apev.evaluate(wild, "limpio", boot=False)
        self.assertGreater(bounded_map, wild_map)
        self.assertGreater(bounded_map, base - 0.1)                  # solo puede cambiar el orden entre validadas y rechazadas
        with_nan = np.full((3, data.Y.shape[1]), np.nan, np.float32)
        self.assertTrue(np.isnan(PR.bounded(with_nan, data.logit[:3])).all())


if __name__ == "__main__":
    unittest.main()
