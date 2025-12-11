"""Utility helpers for the video viewer application."""
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
    """Return True if the path points to an MD.mff .mov video file."""
    return path.suffix.lower() == ".mov" and "md.mff" in path.name.lower()


def find_md_mff_videos(root: Path) -> List[Path]:
    """Return all MD.mff .mov videos under the given root, any extension case."""
    return [path for path in root.rglob("*") if is_md_mff_video(path)]
