"""Helpers for mapping dataset files to video recordings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Optional


@dataclass(frozen=True)
class RecordingLocator:
    """Derived identifiers for locating related files for a video."""

    subject_folder: str
    base_identifier: str

    def candidate_dir(self, root: Path) -> Path:
        return root / self.subject_folder / self.base_identifier

    def fallback_dir(self, root: Path) -> Path:
        return root / self.subject_folder

    def preferred_path(self, root: Path, filename: str) -> Path:
        candidate = self.candidate_dir(root) / filename
        return candidate


def _infer_recording_locator(video_path: Path) -> RecordingLocator:
    raw_path = str(video_path)
    normalized = Path(raw_path.replace("\\", "/"))

    parts = normalized.parts
    if len(parts) == 1 and "\\" in raw_path:
        parts = PureWindowsPath(raw_path).parts

    subject_folder = next(
        (part for part in reversed(parts) if part.upper().startswith("S") and part[1:].isdigit()),
        None,
    )
    if subject_folder is None:
        raise ValueError(f"Could not determine subject folder from {video_path}")

    stem = normalized.stem
    lower_stem = stem.lower()
    prefix = "md.mff."

    base_identifier = stem
    prefix_index = lower_stem.find(prefix)
    if prefix_index != -1:
        base_identifier = stem[prefix_index + len(prefix) :]

    return RecordingLocator(subject_folder=subject_folder, base_identifier=base_identifier)


def derive_time_series_path(video_path: Path, processed_root: Path) -> Path:
    """Return the expected time series path for a given video file."""

    locator = _infer_recording_locator(video_path)
    return locator.preferred_path(processed_root, "ear_eog.fif")


def derive_annotation_path(
    video_path: Path, processed_root: Path, csv_root: Optional[Path] = None
) -> Path:
    """Return the expected annotation CSV path for a given video file."""

    locator = _infer_recording_locator(video_path)
    base_root = csv_root or processed_root

    if csv_root is None:
        ts_path = locator.preferred_path(processed_root, "ear_eog.fif")
        return ts_path.with_suffix(".csv")

    csv_candidate = locator.candidate_dir(base_root) / "ear_eog.csv"
    return csv_candidate
