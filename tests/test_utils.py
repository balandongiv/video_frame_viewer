"""Tests for viewer utility helpers."""
import tempfile
import unittest
from pathlib import Path

from src.utils import find_md_mff_videos, is_md_mff_video, seconds_to_frame_index


class SecondsToFrameIndexTests(unittest.TestCase):
    def test_positive_seconds(self) -> None:
        self.assertEqual(seconds_to_frame_index(1.0, 30.0), 30)
        self.assertEqual(seconds_to_frame_index(2.5, 30.0), 75)

    def test_fractional_seconds(self) -> None:
        self.assertEqual(seconds_to_frame_index(0.1, 30.0), 3)
        self.assertEqual(seconds_to_frame_index(0.0333, 30.0), 0)

    def test_negative_seconds(self) -> None:
        self.assertEqual(seconds_to_frame_index(-1.0, 30.0), -30)


class MdMffVideoTests(unittest.TestCase):
    def test_accepts_matching_mov(self) -> None:
        self.assertTrue(is_md_mff_video(Path("MD.mff.Sample.mov")))
        self.assertTrue(is_md_mff_video(Path("C:/data/MD.MFF.video.MOV")))

    def test_rejects_non_matching(self) -> None:
        self.assertFalse(is_md_mff_video(Path("other.mov")))
        self.assertFalse(is_md_mff_video(Path("MD.mff.sample.mp4")))

    def test_finds_md_mff_videos_with_mixed_extension_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            matching_lower = root / "MD.mff.sample.mov"
            matching_upper = root / "MD.mff.Sample.MOV"
            non_matching = root / "other.mov"
            nested_dir = root / "nested"
            nested_dir.mkdir()
            nested_match = nested_dir / "MD.mff.nested.mov"
            md_dir = root / "MD.mff"
            md_dir.mkdir()
            md_dir_match = md_dir / "S01_20170519_043933.mov"

            for path in [
                matching_lower,
                matching_upper,
                non_matching,
                nested_match,
                md_dir_match,
            ]:
                path.touch()

            results = find_md_mff_videos(root)

            self.assertIn(matching_lower, results)
            self.assertIn(matching_upper, results)
            self.assertIn(nested_match, results)
            self.assertIn(md_dir_match, results)
            self.assertNotIn(non_matching, results)


if __name__ == "__main__":
    unittest.main()
