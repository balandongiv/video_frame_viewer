"""Utility helpers for the video viewer application."""
import os
from pathlib import Path
from typing import List, Optional

import cv2
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap


def frame_to_pixmap(frame, target_size: Optional[QSize] = None) -> Optional[QPixmap]:
    """Convert a BGR OpenCV frame to a QPixmap, optionally scaled."""
    if frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    image = QImage(
        rgb_frame.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    )
    pixmap = QPixmap.fromImage(image)
    if target_size:
        return pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap


def placeholder_pixmap(target_size: QSize) -> QPixmap:
    """Create a simple placeholder pixmap with a neutral background."""
    pixmap = QPixmap(target_size)
    pixmap.fill(Qt.lightGray)
    return pixmap


def seconds_to_frame_index(seconds: float, fps: float = 30.0) -> int:
    """Convert seconds to a frame index using the provided sampling rate."""
    return int(seconds * fps)


def is_md_mff_video(path: Path) -> bool:
    """Return True if the path points to an MD.mff .mov video file.

    Some datasets store videos inside a directory named ``MD.mff`` without
    repeating the token in the filename itself. We therefore check every path
    component for ``MD.mff`` in addition to the filename.
    """

    if path.suffix.lower() != ".mov":
        return False

    return any("md.mff" in part.lower() for part in path.parts)


def find_md_mff_videos(root: Path) -> List[Path]:
    """Return all MD.mff .mov videos under the given root, any extension case.

    Uses ``os.walk`` to avoid platform-specific glob quirks and guarantees we
    only return files that match the MD.mff pattern.
    """

    videos: List[Path] = []

    for dirpath, _, filenames in os.walk(root):
        dirpath_path = Path(dirpath)
        for name in filenames:
            candidate = dirpath_path / name
            if is_md_mff_video(candidate):
                videos.append(candidate)

    return videos
