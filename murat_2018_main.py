"""Entry point for the Murat 2018 EDF annotation reviewer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from murat_2018_gui import (  # noqa: E402
    DEFAULT_MURAT_ROOT,
    SESSION_FILENAME,
    Murat2018Viewer,
    ensure_murat_session_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review Murat 2018 EDF annotations initialized from Blinker output.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_MURAT_ROOT,
        help=rf"Murat 2018 dataset root. Default: {DEFAULT_MURAT_ROOT}",
    )
    return parser


def _ensure_session_logs(dataset_root: Path) -> int:
    """Create missing review YAML logs for Murat recording folders."""

    if not dataset_root.exists():
        return 0

    created = 0
    for folder in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        has_recording = any(folder.glob("*.edf")) or any(folder.glob("*.fif"))
        if not has_recording:
            continue
        session_path = folder / SESSION_FILENAME
        if session_path.exists():
            continue
        try:
            ensure_murat_session_file(folder)
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
    window = Murat2018Viewer(dataset_root=args.dataset_root)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
