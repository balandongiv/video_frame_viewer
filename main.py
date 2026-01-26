"""Entry point for running the Video Frame Viewer from a source checkout.
g"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from video_frame_viewer.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
