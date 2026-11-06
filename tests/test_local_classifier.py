"""Pruebas del clasificador local CONDESAN y de la calibracion de umbrales.

Usan pesos sinteticos, por lo que no requieren BirdNET ni datos privados:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_classifier import META_FILE, WEIGHTS_FILE, LocalSpeciesClassifier  # noqa: E402


def _write_model(folder: Path, W: np.ndarray, b: np.ndarray, thresholds: list[float]) -> None:
    species = [
        {
            "scientific": f"Genus species{k}",
            "common_es": f"Especie {k}",
            "common_en": f"Species {k}",
            "birdnet_label": f"Genus species{k}_Species {k}",
            "threshold": t,
        }
        for k, t in enumerate(thresholds)
    ]
    meta = {"version": "test", "weights": WEIGHTS_FILE, "species": species}
    (folder / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    np.savez(folder / WEIGHTS_FILE, W=W, b=b)


class LocalClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.W = rng.normal(scale=0.05, size=(1024, 3)).astype(np.float32)
        self.b = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        _write_model(self.folder, self.W, self.b, [0.5, 0.5, 0.95])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_available_and_load(self) -> None:
        self.assertTrue(LocalSpeciesClassifier.available(self.folder))
        self.assertFalse(LocalSpeciesClassifier.available(self.folder / "missing"))
        clf = LocalSpeciesClassifier.load(self.folder)
        self.assertEqual(len(clf.species), 3)
        self.assertIn("3 especies", clf.describe())

    def test_probabilities_match_sigmoid_of_linear_layer(self) -> None:
        clf = LocalSpeciesClassifier.load(self.folder)
        x = np.random.default_rng(1).normal(size=(4, 1024)).astype(np.float32)
        expected = 1.0 / (1.0 + np.exp(-(x @ self.W + self.b)))
        np.testing.assert_allclose(clf.probabilities(x), expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(clf.probabilities(x[0]).shape, (1, 3))

    def test_detections_respect_per_species_thresholds(self) -> None:
        clf = LocalSpeciesClassifier.load(self.folder)
        x = np.zeros(1024, dtype=np.float32)  # probabilidades = sigmoid(b) = 0.12, 0.5, 0.88
        rows = clf.detections(x, 3.0, 6.0, "a.wav")
        self.assertEqual([r["scientific"] for r in rows], ["Genus species1"])
        self.assertEqual(rows[0]["source"], "local")
        self.assertEqual(rows[0]["common"], "Especie 1")
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (3.0, 6.0))
        # Con umbrales a la mitad entra tambien la especie 2 (0.88 >= 0.475).
        rows = clf.detections(x, 3.0, 6.0, "a.wav", threshold_scale=0.5)
        self.assertEqual({r["scientific"] for r in rows}, {"Genus species1", "Genus species2"})

    def test_rejects_incompatible_weights(self) -> None:
        _write_model(self.folder, self.W[:, :2], self.b[:2], [0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            LocalSpeciesClassifier.load(self.folder)

    def test_exported_model_metadata_is_consistent(self) -> None:
        meta_path = Path(__file__).resolve().parents[1] / "local_models" / "condesan" / META_FILE
        if not meta_path.exists():
            self.skipTest("modelo exportado no presente")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertGreater(len(meta["species"]), 0)
        for sp in meta["species"]:
            self.assertTrue(0.1 <= sp["threshold"] <= 0.95, sp["scientific"])
            self.assertIn("_", sp["birdnet_label"])


class ThresholdCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import pandas  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("requiere pandas y scikit-learn")

    def test_wood_kahl_thresholds(self) -> None:
        import pandas as pd

        from finetuning.train import REVIEW_THRESHOLD, tablafinal_calibration

        rng = np.random.default_rng(0)
        conf = rng.uniform(0.25, 1.0, size=400)
        # Especie A: la validez crece con la confianza (rechazos frecuentes a baja confianza).
        valid_a = (rng.random(400) < 1 / (1 + np.exp(-12 * (conf - 0.45)))).astype(int)
        det = pd.concat(
            [
                pd.DataFrame({"species": "A", "birdnet_conf": conf, "valid": valid_a}),
                pd.DataFrame({"species": "B", "birdnet_conf": conf[:50], "valid": 1}),
            ]
        )
        out = tablafinal_calibration(det, ["A", "B"]).set_index("species")
        self.assertEqual(out.at["A", "method"], "logistica")
        self.assertGreater(out.at["A", "thr_p90"], 0.5)
        self.assertGreaterEqual(out.at["A", "thr_p95"], out.at["A", "thr_p90"])
        # Sin rechazos no hay evidencia por debajo del umbral de revision.
        self.assertEqual(out.at["B", "method"], "sin rechazos")
        self.assertEqual(out.at["B", "thr_p90"], REVIEW_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
