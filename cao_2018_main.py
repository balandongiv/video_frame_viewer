"""Entry point for the Cao 2018 sustained-attention driving annotation reviewer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from cao_2018_gui import (  # noqa: E402
    DEFAULT_CAO_ROOT,
    SESSION_FILENAME,
    Cao2018Viewer,
    ensure_cao_session_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review Cao 2018 sustained-attention driving annotations initialized from PyBlinker output.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_CAO_ROOT,
        help=rf"Cao 2018 dataset root. Default: {DEFAULT_CAO_ROOT}",
    )
    return parser


def _ensure_session_logs(dataset_root: Path) -> int:
    """Create missing review YAML logs for Cao 2018 session folders."""
    if not dataset_root.exists():
        return 0

    created = 0
    for subject_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            has_recording = any(session_dir.glob("*.fif"))
            if not has_recording:
                continue
            session_path = session_dir / SESSION_FILENAME
            if session_path.exists():
                continue
            try:
                ensure_cao_session_file(session_dir)
            except Exception:
                continue
            else:
                created += 1
    return created


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _ensure_session_logs(args.dataset_root)

    app = QApplication(argv or sys.argv)
    window = Cao2018Viewer(dataset_root=args.dataset_root)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
