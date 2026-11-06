"""Pruebas de la validación de detecciones contra la tabla de expertos.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas as pd

    from validation import (
        STATUS_CONFIRMED,
        STATUS_REJECTED,
        STATUS_UNVALIDATED,
        canonical_fname,
        combine_results,
        compare_recording,
        device_from_path,
        load_reference_table,
        species_key,
    )
except ImportError:  # pragma: no cover
    pd = None


def _tabla() -> "pd.DataFrame":
    base = {"Nombre español": "x", "Día": "2025-07-25", "Hora": "06:01:00", "Confidence Score": 0.6}
    rows = [
        {"Nombre científico": "Icterus mesomelas", "Dispositivo": "AM005", "Inicio (s)": 42.0, "Final (s)": 45.0, "Valid": 1},
        {"Nombre científico": "Icterus mesomelas", "Dispositivo": "AM005", "Inicio (s)": 51.0, "Final (s)": 54.0, "Valid": 1},
        {"Nombre científico": "Zenaida meloda", "Dispositivo": "AM005", "Inicio (s)": 9.0, "Final (s)": 12.0, "Valid": 1},
        {"Nombre científico": "Leptotila verreauxi", "Dispositivo": "AM005", "Inicio (s)": 30.0, "Final (s)": 33.0, "Valid": 0},
        {"Nombre científico": "Zenaida meloda", "Dispositivo": "AM002", "Inicio (s)": 0.0, "Final (s)": 3.0, "Valid": 1},
    ]
    return pd.DataFrame([base | r for r in rows])


@unittest.skipIf(pd is None, "requiere pandas")
class ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tabla.csv"
        _tabla().to_csv(self.path, index=False)
        self.table = load_reference_table(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_file_identification(self) -> None:
        self.assertEqual(canonical_fname("24A04F085CC5BF65_20250725_060100.WAV"), "20250725_060100.WAV")
        self.assertEqual(canonical_fname("20250725_060100.wav"), "20250725_060100.WAV")
        self.assertIsNone(canonical_fname("grabacion.wav"))
        self.assertEqual(device_from_path("/datos/AM005/20250725/Audio_sunrise/x.WAV"), "AM005")
        self.assertIsNone(device_from_path("/datos/x.WAV"))
        self.assertEqual(species_key("Burhinus superciliaris"), species_key("Hesperoburhinus superciliaris"))

    def test_lookup_needs_device_when_ambiguous(self) -> None:
        rows, msg = self.table.lookup("/d/AM005/20250725/24A04F085CC5BF65_20250725_060100.WAV")
        self.assertEqual(len(rows), 4)
        self.assertIn("AM005", msg)
        rows, msg = self.table.lookup("/d/20250725_060100.WAV")
        self.assertIsNone(rows)
        self.assertIn("varias grabadoras", msg)
        rows, msg = self.table.lookup("/d/AM005/20250801_060100.WAV")
        self.assertIsNone(rows)

    def test_compare_counts_and_status(self) -> None:
        ref, _ = self.table.lookup("/d/AM005/20250725_060100.WAV")
        app = [
            {"start": 41.7, "end": 44.7, "scientific": "Icterus mesomelas", "common": "Bolsero", "confidence": 0.69},
            {"start": 29.7, "end": 32.7, "scientific": "Leptotila verreauxi", "common": "Paloma", "confidence": 0.40},
            {"start": 18.0, "end": 21.0, "scientific": "Mimus longicaudatus", "common": "Sinsonte", "confidence": 0.30},
            {"start": 44.6, "end": 47.6, "scientific": "Icterus mesomelas", "common": "Bolsero", "confidence": 0.50},
        ]
        res = compare_recording(app, ref, recording="AM005 / 20250725_060100.WAV")
        self.assertEqual(res.n_validated, 3)
        self.assertEqual(res.n_rejected, 1)
        self.assertEqual(res.matched_validated, 1)  # 42-45 sí; 51-54 y Zenaida no
        self.assertAlmostEqual(res.recall, 1 / 3)
        # 44.6-47.6 solo se solapa 0.4 s con 42-45 (< 0.5 s): sin validar
        self.assertEqual(res.app_status, [STATUS_CONFIRMED, STATUS_REJECTED, STATUS_UNVALIDATED, STATUS_UNVALIDATED])
        self.assertAlmostEqual(res.confirmed_fraction, 1 / 4)
        self.assertEqual(res.species_found, (1, 2))
        icterus = next(s for s in res.species if s.species == "Icterus mesomelas")
        self.assertEqual((icterus.validated, icterus.detected, icterus.matched_validated), (2, 2, 1))

        total = combine_results([res, res])
        self.assertEqual((total.n_validated, total.matched_validated, total.n_app), (6, 2, 8))

    def test_plot_runs_without_audio(self) -> None:
        try:
            from matplotlib.figure import Figure
        except ImportError:  # pragma: no cover
            self.skipTest("requiere matplotlib")
        from validation import plot_validation

        ref, _ = self.table.lookup("/d/AM005/20250725_060100.WAV")
        res = compare_recording([{"start": 42, "end": 45, "scientific": "Icterus mesomelas", "common": "", "confidence": 0.7}], ref)
        fig = Figure()
        plot_validation(fig, res)
        self.assertGreaterEqual(len(fig.axes), 1)


if __name__ == "__main__":
    unittest.main()
