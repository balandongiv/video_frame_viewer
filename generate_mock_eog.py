"""Generate deterministic mock EOG annotation/data files for tests."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import mne
import numpy as np

SEED = 12345
REPO_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = REPO_ROOT / "test_data" / "file_example_MOV_480_700kB.mov"
REFERENCE_CSV = REPO_ROOT / "reference" / "ear_eog.csv"
REFERENCE_FIF = REPO_ROOT / "reference" / "ear_eog.fif"
OUTPUT_CSV = REPO_ROOT / "test_data" / "ear_eog.csv"
OUTPUT_FIF = REPO_ROOT / "test_data" / "ear_eog.fif"

NumericFormatter = Callable[[float], str]


@dataclass
class CsvTemplate:
    """Reference metadata for generating mock annotations."""

    header: List[str]
    dialect: csv.Dialect
    formatters: Dict[int, NumericFormatter]
    categorical_values: Dict[int, List[str]]
    numeric_columns: Sequence[int]


@dataclass
class FifTemplate:
    """Reference metadata for writing mock FIF files."""

    sfreq: float
    channel_name: str
    channel_type: str
    highpass: float | None
    lowpass: float | None


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")


def _sniff_csv(path: Path) -> Tuple[csv.Dialect, bool]:
    with path.open(newline="") as handle:
        sample = handle.read(2048)
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")
    try:
        has_header = sniffer.has_header(sample)
    except csv.Error:
        has_header = True
    return dialect, has_header


def _infer_numeric_formatter(example: str) -> NumericFormatter:
    if "." in example:
        decimals = len(example.split(".")[1])
        fmt = f"{{:.{decimals}f}}"
        return lambda value: fmt.format(value)
    return lambda value: str(int(round(value)))


def _load_csv_template(path: Path) -> CsvTemplate:
    _ensure_exists(path, "Reference CSV")
    dialect, has_header = _sniff_csv(path)

    with path.open(newline="") as handle:
        reader = csv.reader(handle, dialect)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Reference CSV at {path} is empty.")

    header = rows[0] if has_header else [f"col_{index}" for index in range(len(rows[0]))]
    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        raise ValueError(f"Reference CSV at {path} contains no data rows.")

    numeric_columns: List[int] = []
    formatters: Dict[int, NumericFormatter] = {}
    categorical_values: Dict[int, List[str]] = {}

    for index, value in enumerate(data_rows[0]):
        try:
            float(value)
        except ValueError:
            categorical_values[index] = list(
                dict.fromkeys(row[index] for row in data_rows if len(row) > index)
            )
            continue
        numeric_columns.append(index)
        formatters[index] = _infer_numeric_formatter(value)

    return CsvTemplate(
        header=header,
        dialect=dialect,
        formatters=formatters,
        categorical_values=categorical_values,
        numeric_columns=numeric_columns,
    )


def _load_fif_template(path: Path) -> FifTemplate:
    _ensure_exists(path, "Reference FIF")
    raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
    channel_types = raw.get_channel_types()
    channel_name = raw.ch_names[0]
    channel_type = channel_types[0]
    for name, ch_type in zip(raw.ch_names, channel_types):
        if ch_type == "eog":
            channel_name = name
            channel_type = ch_type
            break
    template = FifTemplate(
        sfreq=raw.info["sfreq"],
        channel_name=channel_name,
        channel_type=channel_type,
        highpass=raw.info.get("highpass"),
        lowpass=raw.info.get("lowpass"),
    )
    raw.close()
    return template


def _video_duration_seconds(path: Path) -> float:
    _ensure_exists(path, "Video file")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video at {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()

    if fps <= 0.0:
        raise ValueError(f"Could not determine FPS for {path}")
    if frame_count <= 0.0:
        raise ValueError(f"Could not determine frame count for {path}")

    return frame_count / fps


def _choose_numeric_value(column_name: str, onset: float, duration: float) -> float:
    lower = column_name.lower()
    if "dur" in lower:
        return duration
    if "onset" in lower or "start" in lower or "time" in lower:
        return onset
    return onset


def _generate_annotations(
    duration_seconds: float, template: CsvTemplate, rng: np.random.Generator
) -> List[List[str]]:
    event_count = int(rng.integers(8, 16))
    max_event_duration = max(0.1, min(1.0, duration_seconds / 5))
    event_durations = rng.uniform(0.1, max_event_duration, size=event_count)

    base_positions = np.linspace(0.0, duration_seconds, event_count + 2)[1:-1]
    jitter = rng.uniform(-0.2, 0.2, size=event_count)
    onset_times = np.clip(
        base_positions + jitter, 0.0, np.maximum(0.0, duration_seconds - event_durations)
    )
    sorted_indices = np.argsort(onset_times)

    annotations: List[List[str]] = []
    for index in sorted_indices:
        onset = float(onset_times[index])
        event_duration = float(event_durations[index])
        row: List[str] = []
        for col_index, column_name in enumerate(template.header):
            if col_index in template.numeric_columns:
                value = _choose_numeric_value(column_name, onset, event_duration)
                formatter = template.formatters[col_index]
                row.append(formatter(value))
            else:
                choices = template.categorical_values.get(col_index) or ["Mock"]
                choice_index = int(rng.integers(0, len(choices)))
                row.append(choices[choice_index])
        annotations.append(row)

    return annotations


def _write_annotations(path: Path, template: CsvTemplate, rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, template.dialect)
        writer.writerow(template.header)
        for row in rows:
            writer.writerow(row)


def _write_mock_fif(
    path: Path, template: FifTemplate, duration_seconds: float, rng: np.random.Generator
) -> int:
    n_samples = int(round(duration_seconds * template.sfreq))
    n_samples = max(n_samples, 1)
    data = rng.normal(loc=0.0, scale=1e-6, size=(1, n_samples))

    info = mne.create_info(
        ch_names=[template.channel_name], sfreq=template.sfreq, ch_types=[template.channel_type]
    )
    with info._unlock():
        info["highpass"] = template.highpass
        info["lowpass"] = template.lowpass

    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_meas_date(0)
    file_id = {"version": 2, "machid": np.array([0, 0], dtype=">i4"), "secs": 0, "usecs": 0}
    with raw.info._unlock():
        raw.info["file_id"] = file_id
        raw.info["meas_id"] = file_id.copy()
    raw.save(path, overwrite=True, verbose="ERROR")
    raw.close()
    return n_samples


def regenerate_mock_eog() -> None:
    """Regenerate deterministic mock annotation and FIF test data."""

    csv_template = _load_csv_template(REFERENCE_CSV)
    fif_template = _load_fif_template(REFERENCE_FIF)
    duration_seconds = _video_duration_seconds(VIDEO_PATH)

    rng = np.random.default_rng(SEED)
    annotations = _generate_annotations(duration_seconds, csv_template, rng)
    _write_annotations(OUTPUT_CSV, csv_template, annotations)
    sample_count = _write_mock_fif(OUTPUT_FIF, fif_template, duration_seconds, rng)

    print(
        f"Generated {len(annotations)} annotations "
        f"over {duration_seconds:.3f}s video duration."
    )
    print(
        f"FIF sampling frequency: {fif_template.sfreq} Hz, "
        f"samples written: {sample_count}"
    )
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"FIF output: {OUTPUT_FIF}")


if __name__ == "__main__":
    regenerate_mock_eog()
