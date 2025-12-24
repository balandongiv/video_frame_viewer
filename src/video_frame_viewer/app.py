"""Application entry point helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QApplication

from video_frame_viewer.config import AppConfig
from video_frame_viewer.gui import VideoFrameViewer


def launch_app(
    config: AppConfig,
    *,
    config_path: Optional[Path] = None,
    config_source: str = "",
    argv: Optional[list[str]] = None,
) -> int:
    """Launch the Qt application."""

    app = QApplication(argv or sys.argv)
    window = VideoFrameViewer(config, config_path=config_path, config_source=config_source)
    window.show()
    return app.exec_()
