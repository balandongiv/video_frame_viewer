import tempfile
import unittest
from pathlib import Path

from viewer.time_series import (
    AnnotationSegment,
    load_annotations_from_csv,
    sanitize_annotation,
    save_annotations_to_csv,
)


class AnnotationIoTests(unittest.TestCase):
    def test_sanitize_annotation_clamps_bounds(self) -> None:
        onset, duration = sanitize_annotation(-1.0, -0.5, max_end=2.0)
        self.assertGreaterEqual(onset, 0.0)
        self.assertGreater(duration, 0.0)
        self.assertLessEqual(onset + duration, 2.0)

    def test_round_trip_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ear_eog.csv"
            segments = [
                AnnotationSegment(onset=1.5, duration=0.25, description="HB_CL"),
                AnnotationSegment(onset=3.0, duration=0.5, description="BLINK"),
            ]

            save_annotations_to_csv(path, segments)
            loaded = load_annotations_from_csv(path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].description, "HB_CL")
            self.assertAlmostEqual(loaded[1].duration, 0.5)


if __name__ == "__main__":
    unittest.main()
