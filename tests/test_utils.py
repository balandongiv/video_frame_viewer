"""Tests for viewer utility helpers."""
import unittest

from viewer.utils import seconds_to_frame_index


class SecondsToFrameIndexTests(unittest.TestCase):
    def test_positive_seconds(self) -> None:
        self.assertEqual(seconds_to_frame_index(1.0, 30.0), 30)
        self.assertEqual(seconds_to_frame_index(2.5, 30.0), 75)

    def test_fractional_seconds(self) -> None:
        self.assertEqual(seconds_to_frame_index(0.1, 30.0), 3)
        self.assertEqual(seconds_to_frame_index(0.0333, 30.0), 0)

    def test_negative_seconds(self) -> None:
        self.assertEqual(seconds_to_frame_index(-1.0, 30.0), -30)


if __name__ == "__main__":
    unittest.main()
