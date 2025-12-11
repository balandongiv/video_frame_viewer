"""Time series loading and visualization helpers."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import List, Optional

import mne
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

PROCESSED_ROOT = Path(r"D:\dataset\drowsy_driving_raja_processed")


def derive_time_series_path(video_path: Path, processed_root: Path = PROCESSED_ROOT) -> Path:
    """Return the expected time series path for a given video file.

    The path is built by matching the subject folder from anywhere in the video
    path and using the base recording identifier (portion after ``MD.mff.``
    without any trailing numeric suffix) to locate ``ear_eog.fif`` in the
    processed dataset root.
    """

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

    parts = base_identifier.split("_")
    if parts and parts[-1].isdigit() and len(parts[-1]) <= 2:
        base_identifier = "_".join(parts[:-1]) or base_identifier

    return processed_root / subject_folder / base_identifier / "ear_eog.fif"


class TimeSeriesViewer(QWidget):
    """Widget that renders time series data alongside the video frames."""

    def __init__(self, max_points: int = 10000, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.max_points = max_points
        self.raw: Optional[mne.io.BaseRaw] = None
        self._plotted_curves: List[pg.PlotDataItem] = []
        self._times: Optional[np.ndarray] = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Channels")

        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
        self.plot_widget.addItem(self.cursor_line)
        self.cursor_line.hide()

        self.status_label = QLabel("Load a video to view synchronized time series data.")

        layout.addWidget(self.plot_widget)
        layout.addWidget(self.status_label)

    def load_for_video(self, video_path: Optional[Path]) -> None:
        """Load and plot the time series associated with the provided video."""

        self._clear_plot()
        self._times = None
        self.raw = None

        if video_path is None:
            self.status_label.setText("Select a video to view its time series.")
            self.cursor_line.hide()
            return

        try:
            ts_path = derive_time_series_path(video_path)
        except ValueError as exc:  # pragma: no cover - guardrails for unexpected paths
            self.status_label.setText(str(exc))
            self.cursor_line.hide()
            return
        if not ts_path.exists():
            self.status_label.setText(
                f"Time series file not found at {ts_path}."
            )
            self.cursor_line.hide()
            return

        self.status_label.setText(f"Loaded time series from {ts_path}.")
        self.raw = mne.io.read_raw_fif(str(ts_path), preload=True, verbose="ERROR")
        self._times = self.raw.times
        self._plot_data()

    def update_cursor_time(self, seconds: float) -> None:
        """Move the vertical cursor line to the provided timestamp."""

        if self._times is None or seconds < 0:
            self.cursor_line.hide()
            return

        self.cursor_line.setPos(seconds)
        self.cursor_line.show()

    def _plot_data(self) -> None:
        if self.raw is None:
            return

        data = self.raw.get_data()
        times = self._times
        if times is None:
            return

        decimation = max(1, data.shape[1] // self.max_points)
        if decimation > 1:
            data = data[:, ::decimation]
            times = times[::decimation]

        peak = np.nanmax(np.abs(data)) or 1.0
        spacing = peak * 2
        offsets = np.arange(data.shape[0]) * spacing

        self.plot_widget.enableAutoRange()
        for idx, channel in enumerate(data):
            curve = self.plot_widget.plot(times, channel + offsets[idx], pen=pg.mkPen(width=1))
            curve.setDownsampling(auto=True, method="peak")
            self._plotted_curves.append(curve)

        self.cursor_line.show()

    def _clear_plot(self) -> None:
        for curve in self._plotted_curves:
            self.plot_widget.removeItem(curve)
        self._plotted_curves.clear()
        self.plot_widget.enableAutoRange()
        self.cursor_line.hide()
