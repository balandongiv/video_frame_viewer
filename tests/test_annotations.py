import unittest
from pathlib import Path
import tempfile

from viewer.annotations import AnnotationEntry, clamp_entry, load_annotations, save_annotations, validate_annotations


class AnnotationIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.tmp_path = Path(self._temp_dir.name)

    def test_load_annotations_missing_file(self) -> None:
        csv_path = self.tmp_path / "ear_eog.csv"
        self.assertEqual(load_annotations(csv_path), [])

    def test_save_and_load_roundtrip(self) -> None:
        csv_path = self.tmp_path / "ear_eog.csv"
        entries = [
            AnnotationEntry(onset=1.0, duration=0.5, description="HB_CL"),
            AnnotationEntry(onset=2.0, duration=0.75, description="BLINK"),
        ]

        save_annotations(csv_path, entries)
        reloaded = load_annotations(csv_path)

        self.assertEqual(reloaded, entries)


class AnnotationValidationTests(unittest.TestCase):
    def test_validate_annotations_enforces_bounds(self) -> None:
        entries = [AnnotationEntry(onset=0.0, duration=0.2, description="OK")]
        self.assertEqual(validate_annotations(entries, max_time=2.0), entries)

        with self.assertRaises(ValueError):
            validate_annotations(
                [AnnotationEntry(onset=-1, duration=0.2, description="bad")],
                max_time=2.0,
            )

        with self.assertRaises(ValueError):
            validate_annotations(
                [AnnotationEntry(onset=1.0, duration=-0.1, description="bad")],
                max_time=2.0,
            )

        with self.assertRaises(ValueError):
            validate_annotations(
                [AnnotationEntry(onset=1.9, duration=0.2, description="bad")],
                max_time=2.0,
            )

    def test_clamp_entry_limits_to_bounds(self) -> None:
        entry = AnnotationEntry(onset=-1.0, duration=5.0, description="wide")
        clamped = clamp_entry(entry, max_time=1.5)

        self.assertEqual(clamped.onset, 0.0)
        self.assertAlmostEqual(clamped.end, 1.5, places=6)
        self.assertGreater(clamped.duration, 0)


if __name__ == "__main__":
    unittest.main()
