"""Console entrypoint for the Video Frame Viewer."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app import launch_app
from config import (
    AppConfig,
    ConfigNotFoundError,
    ConfigResolution,
    resolve_config,
    save_config,
)
from version import __version__

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for EEG/EOG/EAR data synchronized to video.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML configuration file. Overrides environment variables.",
    )
    parser.add_argument(
        "--set-root",
        type=Path,
        help="Update the stored dataset root before launching the viewer.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Fail instead of prompting when no configuration is available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )


def _apply_root_override(
    resolution: ConfigResolution, requested_root: Path, requested_path: Path | None
) -> ConfigResolution:
    updated = AppConfig(
        dataset_root=requested_root.expanduser(),
        mov_dir=resolution.config.mov_dir,
        csv_dir=resolution.config.csv_dir,
        fif_dir=resolution.config.fif_dir,
        ui=dict(resolution.config.ui),
    )
    saved_path = save_config(updated, requested_path or resolution.path)
    LOGGER.info("Updated dataset root to %s", updated.dataset_root)
    return ConfigResolution(updated, saved_path, "cli_set_root")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.version:
        print(__version__)
        return 0

    try:
        resolution = resolve_config(cli_config=args.config, allow_prompt=not args.no_prompt)
    except ConfigNotFoundError as exc:
        LOGGER.error(str(exc))
        return 1
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Failed to load configuration: %s", exc)
        return 1

    if args.set_root:
        resolution = _apply_root_override(resolution, args.set_root, args.config)

    LOGGER.info(
        "Using dataset root %s (source: %s%s)",
        resolution.config.dataset_root,
        resolution.source,
        f", path={resolution.path}" if resolution.path else "",
    )
    return launch_app(
        resolution.config,
        config_path=resolution.path,
        config_source=resolution.source,
        argv=argv,
    )


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    sys.exit(main())
