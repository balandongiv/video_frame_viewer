"""Unit tests for VideoHandler utilities."""
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from video_handler import VideoHandler


def create_temp_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "sample.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(video_path), fourcc, 1.0, (16, 16))
    for i in range(3):
        frame = np.full((16, 16, 3), i * 80, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class VideoHandlerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_load_and_frame_count(self) -> None:
        video_path = create_temp_video(self.tmp_dir)
        handler = VideoHandler()

        self.assertTrue(handler.load(video_path))
        self.assertEqual(handler.frame_count, 3)
        self.assertAlmostEqual(handler.fps, 1.0, places=2)

        handler.release()
        self.assertIsNone(handler.capture)
        self.assertEqual(handler.frame_count, 0)

    def test_clamp_and_read_frame(self) -> None:
        video_path = create_temp_video(self.tmp_dir)
        handler = VideoHandler()
        self.assertTrue(handler.load(video_path))

        self.assertEqual(handler.clamp_index(-5), 0)
        self.assertEqual(handler.clamp_index(10), 2)

        frame = handler.read_frame(1)
        self.assertIsNotNone(frame)
        assert frame is not None  # For type checkers
        self.assertEqual(frame.shape[0], 16)
        self.assertEqual(frame.shape[1], 16)

        out_of_range_frame = handler.read_frame(10)
        self.assertIsNotNone(out_of_range_frame)

    def test_release_resets_fps(self) -> None:
        video_path = create_temp_video(self.tmp_dir)
        handler = VideoHandler()
        self.assertTrue(handler.load(video_path))
        self.assertGreater(handler.fps, 0)

        handler.release()
        self.assertEqual(handler.fps, 0.0)


if __name__ == "__main__":
    unittest.main()
