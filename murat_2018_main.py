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

from murat_2018_gui import DEFAULT_MURAT_ROOT, Murat2018Viewer  # noqa: E402


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    app = QApplication(argv or sys.argv)
    window = Murat2018Viewer(dataset_root=args.dataset_root)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
