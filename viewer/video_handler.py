"""Video handling utilities for reading frames."""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class VideoHandler:
    """Manage loading and frame retrieval for videos."""

    def __init__(self) -> None:
        self.capture: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.video_path: Optional[Path] = None

    def load(self, path: Path) -> bool:
        """Load a video file for frame access."""
        self.release()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return False

        self.capture = capture
        self.video_path = path
        self.frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        return True

    def release(self) -> None:
        """Release the currently loaded video capture."""
        if self.capture:
            self.capture.release()
        self.capture = None
        self.video_path = None
        self.frame_count = 0

    def clamp_index(self, index: int) -> int:
        """Clamp a frame index to the valid range of the loaded video."""
        if self.frame_count <= 0:
            return 0
        return max(0, min(index, self.frame_count - 1))

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """Read a frame by index, returning None when unavailable."""
        if not self.capture or self.frame_count <= 0:
            return None

        clamped_index = self.clamp_index(index)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, clamped_index)
        success, frame = self.capture.read()
        if not success:
            return None
        return frame

    def read_frames(self, indices: List[int]) -> List[Tuple[int, Optional[np.ndarray]]]:
        """Read multiple frames and return them alongside their indices."""
        frames: List[Tuple[int, Optional[np.ndarray]]] = []
        for index in indices:
            frames.append((index, self.read_frame(index)))
        return frames
