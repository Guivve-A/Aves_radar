"""Pruebas del muestreo aleatorio para anotacion y del lector de anotaciones.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas as pd

    from finetuning.evaluate_annotations import name_index, parse_species
    from finetuning.sample_annotation import sample_windows
except ImportError:  # pragma: no cover
    pd = None


def _index() -> "pd.DataFrame":
    rows = []
    for dev in ("AM001", "AM002"):
        for per, hour in (("Sunrise", "06"), ("Sunset", "18")):
            for day in range(24, 30):
                fname = f"202507{day}_{hour}0000.WAV"
                rows.append(
                    {
                        "device": dev,
                        "period": per,
                        "fname": fname,
                        "file_key": f"{dev}/{per}/{fname}",
                        "rel_path": f"{dev}/202507{day}/{per}/{fname}",
                    }
                )
    return pd.DataFrame(rows)


@unittest.skipIf(pd is None, "requiere pandas")
class SampleWindowsTest(unittest.TestCase):
    def test_balanced_grid_aligned_and_reproducible(self) -> None:
        a = sample_windows(_index(), 40, seed=3)
        b = sample_windows(_index(), 40, seed=3)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 40)
        self.assertEqual(a.groupby(["device", "period"]).size().tolist(), [10, 10, 10, 10])
        self.assertTrue(((a["start_s"] % 3) == 0).all())
        self.assertTrue((a["end_s"] <= 55).all())
        self.assertFalse(a.duplicated(["file_key", "start_s"]).any())
        self.assertEqual(a["id"].iloc[0], "W0001")

    def test_excluded_sessions_are_never_sampled(self) -> None:
        out = sample_windows(_index(), 30, seed=0, exclude_sessions={"AM001_20250724_Sunrise"})
        sessions = out["device"] + "_" + out["date"].str.replace("-", "") + "_" + out["period"]
        self.assertNotIn("AM001_20250724_Sunrise", set(sessions))


@unittest.skipIf(pd is None, "requiere pandas")
class ParseSpeciesTest(unittest.TestCase):
    def test_scientific_and_common_names_with_accents_and_case(self) -> None:
        info = pd.DataFrame(
            {"common_es": ["Sinsonte colilargo", "Tórtola melódica"], "common_en": ["Long-tailed Mockingbird", "West Peruvian Dove"]},
            index=["Mimus longicaudatus", "Zenaida meloda"],
        )
        names = name_index(list(info.index), info)
        found, unknown = parse_species("mimus  LONGICAUDATUS; Tortola melodica, Ave rara", names)
        self.assertEqual(found, {"Mimus longicaudatus", "Zenaida meloda"})
        self.assertEqual(unknown, {"Ave rara"})
        self.assertEqual(parse_species("", names), (set(), set()))


if __name__ == "__main__":
    unittest.main()
