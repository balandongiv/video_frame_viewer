"""Helpers for loading, validating, and persisting annotation data."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class AnnotationEntry:
    """A simple annotation segment described by onset, duration, and label."""

    onset: float
    duration: float
    description: str

    @property
    def end(self) -> float:
        return self.onset + self.duration


REQUIRED_COLUMNS = ("onset", "duration", "description")
MIN_DURATION = 1e-3


def _ensure_required_columns(fieldnames: Iterable[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"Annotation file missing columns: {', '.join(missing)}")


def load_annotations(csv_path: Path) -> List[AnnotationEntry]:
    """Load annotations from a CSV file if it exists.

    The CSV is expected to contain the ``onset``, ``duration``, and
    ``description`` columns. Missing files return an empty list to allow new
    annotations to be created from scratch.
    """

    if not csv_path.exists():
        return []

    entries: List[AnnotationEntry] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Annotation file is missing a header row.")
        _ensure_required_columns(reader.fieldnames)

        for row in reader:
            try:
                onset = float(row["onset"])
                duration = float(row["duration"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Annotations must contain numeric onset and duration values.") from exc

            description = str(row.get("description", "")).strip()
            entries.append(AnnotationEntry(onset=onset, duration=duration, description=description))

    return entries


def clamp_entry(entry: AnnotationEntry, max_time: float) -> AnnotationEntry:
    """Return an entry clamped to a valid time range with positive duration."""

    bounded_onset = max(0.0, min(entry.onset, max_time))
    bounded_end = max(bounded_onset + MIN_DURATION, min(entry.end, max_time))
    bounded_duration = bounded_end - bounded_onset
    if bounded_duration <= 0:
        raise ValueError("Annotation duration must be greater than zero.")

    return AnnotationEntry(onset=bounded_onset, duration=bounded_duration, description=entry.description)


def validate_annotations(entries: Iterable[AnnotationEntry], max_time: float) -> List[AnnotationEntry]:
    """Validate and clamp annotations to a recording boundary."""

    validated: List[AnnotationEntry] = []
    for entry in entries:
        if entry.duration <= 0:
            raise ValueError("Annotation duration must be greater than zero.")
        if entry.onset < 0:
            raise ValueError("Annotation onset must be non-negative.")
        if entry.end > max_time:
            raise ValueError("Annotation end time exceeds the recording length.")
        validated.append(entry)
    return validated


def save_annotations(csv_path: Path, entries: Iterable[AnnotationEntry]) -> None:
    """Persist annotations to CSV in the expected schema."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "onset": f"{entry.onset:.9f}",
                    "duration": f"{entry.duration:.9f}",
                    "description": entry.description,
                }
            )

