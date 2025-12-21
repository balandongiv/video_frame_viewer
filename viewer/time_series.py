"""Time series loading and visualization helpers."""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import List, Optional, Set

import mne
import numpy as np
import pyqtgraph as pg
from mne.io.constants import FIFF
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

PROCESSED_ROOT = Path(r"D:\dataset\drowsy_driving_raja_processed")
PRIMARY_CHANNEL = "EEG-E8"
EAR_AVG_CHANNEL = "EAR-avg_ear"
DEFAULT_VISIBLE_CHANNELS = {PRIMARY_CHANNEL, EAR_AVG_CHANNEL}
TARGET_PLOT_UNIT = "µV"
UNIT_SCALE_FACTORS = {
    "v": 1e6,
    "mv": 1e3,
    "uv": 1.0,
    "µv": 1.0,
    "nv": 1e-3,
}
PAIR_GAP_MULTIPLIER = 0.5
CHANNEL_PALETTE = [
    "#d32f2f",  # primary red
    "#c62828",
    "#ef5350",
    "#b71c1c",
    "#f44336",
]
ANNOTATION_PALETTE = [
    "#1e88e5",
    "#43a047",
    "#fb8c00",
    "#8e24aa",
    "#6d4c41",
    "#00897b",
    "#f4511e",
]
MIN_ANNOTATION_DURATION = 0.1

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Annotation:
    onset: float
    duration: float
    description: str


@dataclass
class AnnotationItem:
    annotation: Annotation
    region: pg.LinearRegionItem


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

    candidate = processed_root / subject_folder / base_identifier / "ear_eog.fif"
    fallback = processed_root / subject_folder / "ear_eog.fif"

    if not candidate.exists() and fallback.exists():
        return fallback

    return candidate


class TimeSeriesViewer(QWidget):
    """Widget that renders time series data alongside the video frames."""

    annotation_jump_requested = pyqtSignal(float)
    FILTER_ALL = "__all__"
    FILTER_NONE = "__none__"

    def __init__(self, max_points: int = 10000, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.max_points = max_points
        self.raw: Optional[mne.io.BaseRaw] = None
        self._plotted_curves: List[pg.PlotDataItem] = []
        self._times: Optional[np.ndarray] = None
        self._selected_channels: Set[str] = set()
        self._last_cursor_time: float = 0.0
        self.default_view_span_seconds: float = 5.0
        self.view_span_seconds: float = self.default_view_span_seconds
        self.min_span_seconds: float = 0.1
        self._last_ts_path: Optional[Path] = None
        self.processed_root: Path = PROCESSED_ROOT
        self._annotation_items: List[AnnotationItem] = []
        self._annotation_by_region: dict[pg.LinearRegionItem, AnnotationItem] = {}
        self._annotation_colors: dict[str, pg.Color] = {}
        self._annotations: List[Annotation] = []
        self._annotations_dirty = False
        self._annotation_dragging = False
        self._annotation_drag_start: Optional[float] = None
        self._annotation_drag_preview: Optional[pg.LinearRegionItem] = None
        self._last_annotation_description = ""
        self._annotation_filter_value = self.FILTER_ALL

        self._controls_container = QWidget(self)
        control_layout = QVBoxLayout()

        control_row = QHBoxLayout()
        self.show_all_checkbox = QCheckBox("Show all channels")
        self.show_all_checkbox.setChecked(False)
        self.show_all_checkbox.stateChanged.connect(self._on_show_all_channels)

        self.channel_list = QListWidget()
        self.channel_list.setMaximumHeight(120)
        self.channel_list.itemChanged.connect(self._on_channel_item_changed)

        self.zoom_out_button = QPushButton("Zoom -")
        self.zoom_out_button.clicked.connect(lambda: self._adjust_zoom(1.25))
        self.zoom_in_button = QPushButton("Zoom +")
        self.zoom_in_button.clicked.connect(lambda: self._adjust_zoom(0.8))
        self.zoom_reset_button = QPushButton("Reset Zoom")
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        self.zoom_label = QLabel(self._zoom_label_text())

        control_row.addWidget(self.show_all_checkbox)
        control_row.addWidget(self.zoom_out_button)
        control_row.addWidget(self.zoom_in_button)
        control_row.addWidget(self.zoom_reset_button)
        control_row.addWidget(self.zoom_label)
        control_row.addStretch()

        control_layout.addLayout(control_row)
        control_layout.addWidget(self.channel_list)
        self._controls_container.setLayout(control_layout)

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Channels")
        self.plot_widget.viewport().installEventFilter(self)

        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
        self.plot_widget.addItem(self.cursor_line)
        self.cursor_line.hide()

        annotation_controls = QWidget()
        annotation_layout = QHBoxLayout()
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        self.annotation_mode_checkbox = QCheckBox("Annotation mode (drag to add)")
        self.annotation_mode_checkbox.stateChanged.connect(self._on_annotation_mode_changed)
        annotation_layout.addWidget(self.annotation_mode_checkbox)

        annotation_layout.addWidget(QLabel("Annotations:"))
        self.annotation_filter_combo = QComboBox()
        self.annotation_filter_combo.currentIndexChanged.connect(self._on_annotation_filter_changed)
        annotation_layout.addWidget(self.annotation_filter_combo)
        self.save_annotations_button = QPushButton("Save annotations")
        self.save_annotations_button.clicked.connect(self._save_annotations)
        self.save_annotations_button.setEnabled(False)
        self.annotations_dirty_label = QLabel("")
        self.annotations_dirty_label.setStyleSheet("color: #d32f2f;")
        annotation_layout.addWidget(self.save_annotations_button)
        annotation_layout.addWidget(self.annotations_dirty_label)
        annotation_layout.addWidget(QLabel("Shortcuts: [ or P = prev, ] or N = next"))
        annotation_layout.addStretch()
        annotation_controls.setLayout(annotation_layout)

        self.status_label = QLabel("Load a video to view synchronized time series data.")

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self.plot_widget)
        layout.addWidget(annotation_controls)
        layout.addWidget(self.status_label)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)

        self._update_annotation_filter_options(force_all=True)

    def channel_controls(self) -> QWidget:
        """Expose the channel selection controls for external layouts."""

        return self._controls_container

    def set_processed_root(self, processed_root: Path) -> None:
        """Override the processed dataset root for time series data."""

        self.processed_root = processed_root

    def load_for_video(self, video_path: Optional[Path]) -> None:
        """Load and plot the time series associated with the provided video."""

        self._clear_plot()
        self._clear_annotations()
        self._times = None
        self.raw = None
        self._selected_channels.clear()
        self.channel_list.clear()
        self._reset_zoom()
        self._last_ts_path = None
        self._annotations = []
        self._set_annotations_dirty(False)
        self._update_annotation_filter_options(force_all=True)

        if video_path is None:
            self.status_label.setText("Select a video to view its time series.")
            self.cursor_line.hide()
            return

        try:
            ts_path = derive_time_series_path(video_path, processed_root=self.processed_root)
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
        self._last_ts_path = ts_path
        self.raw = mne.io.read_raw_fif(str(ts_path), preload=True, verbose="ERROR")
        self._times = self.raw.times
        self._populate_channel_list()
        self._plot_data()
        self._add_annotations_for_path(ts_path)
        self._ensure_view_range(0.0)

    def update_cursor_time(self, seconds: float) -> None:
        """Keep the current time centered under a fixed cursor."""

        if self._times is None or self._times.size == 0:
            self.cursor_line.hide()
            return

        self._ensure_view_range(max(0.0, seconds))

    def _plot_data(self) -> None:
        if self.raw is None:
            return

        picks = self._channel_indices()
        if not picks:
            self.status_label.setText("Select at least one channel to display.")
            return

        data = self.raw.get_data(picks=picks)
        times = self._times
        if times is None:
            return

        decimation = max(1, data.shape[1] // self.max_points)
        if decimation > 1:
            data = data[:, ::decimation]
            times = times[::decimation]

        channel_names = [self.raw.ch_names[index] for index in picks]
        channel_types = self.raw.get_channel_types(picks=picks)
        data, picks, channel_names, channel_types = self._order_channels_for_display(
            data, picks, channel_names, channel_types
        )
        data, normalized_unit = self._normalize_channel_units(
            data, picks, channel_names, channel_types
        )
        if normalized_unit:
            self.plot_widget.setLabel("left", "Channels", units=normalized_unit)
        else:
            self.plot_widget.setLabel("left", "Channels")

        peak = np.nanmax(np.abs(data)) or 1.0
        spacing = peak * 2
        offsets = self._channel_offsets(channel_names, spacing)

        self.plot_widget.enableAutoRange()
        for idx, (channel, channel_name) in enumerate(zip(data, channel_names)):
            curve = self.plot_widget.plot(
                times, channel + offsets[idx], pen=self._pen_for_channel(channel_name, idx)
            )
            curve.setDownsampling(auto=True, method="peak")
            self._plotted_curves.append(curve)

        self.cursor_line.show()
        total_channels = len(self.raw.ch_names) if self.raw else len(picks)
        self.status_label.setText(
            f"Displaying {len(picks)} of {total_channels} channel(s) from {self._last_ts_path}"
        )

    def _normalize_channel_units(
        self,
        data: np.ndarray,
        picks: List[int],
        channel_names: List[str],
        channel_types: List[str],
    ) -> tuple[np.ndarray, Optional[str]]:
        if self.raw is None:
            return data, None

        orig_units = getattr(self.raw, "_orig_units", {}) or {}
        scales = np.ones(len(channel_names), dtype=float)
        normalize_any = False

        for idx, (pick, name, channel_type) in enumerate(
            zip(picks, channel_names, channel_types)
        ):
            if not self._should_normalize_channel(name, channel_type):
                continue
            normalize_any = True
            scales[idx] = self._scale_for_channel(name, pick, orig_units)

        if not normalize_any:
            return data, None

        return data * scales[:, np.newaxis], TARGET_PLOT_UNIT

    def _order_channels_for_display(
        self,
        data: np.ndarray,
        picks: List[int],
        channel_names: List[str],
        channel_types: List[str],
    ) -> tuple[np.ndarray, List[int], List[str], List[str]]:
        if PRIMARY_CHANNEL not in channel_names or EAR_AVG_CHANNEL not in channel_names:
            return data, picks, channel_names, channel_types

        indices = list(range(len(channel_names)))
        eeg_index = channel_names.index(PRIMARY_CHANNEL)
        ear_index = channel_names.index(EAR_AVG_CHANNEL)
        insert_at = min(eeg_index, ear_index)

        for idx in sorted({eeg_index, ear_index}, reverse=True):
            indices.pop(idx)

        ordered_pair = [eeg_index, ear_index]
        indices[insert_at:insert_at] = ordered_pair

        ordered_data = data[indices]
        ordered_picks = [picks[idx] for idx in indices]
        ordered_names = [channel_names[idx] for idx in indices]
        ordered_types = [channel_types[idx] for idx in indices]
        return ordered_data, ordered_picks, ordered_names, ordered_types

    def _channel_offsets(self, channel_names: List[str], spacing: float) -> np.ndarray:
        offsets = np.zeros(len(channel_names), dtype=float)
        pair = {PRIMARY_CHANNEL, EAR_AVG_CHANNEL}
        for idx in range(1, len(channel_names)):
            prev_name = channel_names[idx - 1]
            curr_name = channel_names[idx]
            gap = spacing * PAIR_GAP_MULTIPLIER if {prev_name, curr_name} == pair else spacing
            offsets[idx] = offsets[idx - 1] + gap
        return offsets

    def _should_normalize_channel(self, name: str, channel_type: str) -> bool:
        if channel_type in {"eeg", "eog"}:
            return True
        return name.upper().startswith("EAR-")

    def _scale_for_channel(self, name: str, pick: int, orig_units: dict[str, str]) -> float:
        if self.raw is None:
            return UNIT_SCALE_FACTORS["v"]
        try:
            ch_info = self.raw.info["chs"][pick]
        except (IndexError, KeyError, TypeError):
            ch_info = {}
        unit_value = ch_info.get("unit")
        unit_mul = ch_info.get("unit_mul", 0)
        if unit_value == FIFF.FIFF_UNIT_V:
            return UNIT_SCALE_FACTORS["v"] * (10 ** unit_mul)
        if isinstance(unit_value, str):
            return self._scale_to_target(unit_value)
        unit = orig_units.get(name)
        if unit:
            return self._scale_to_target(unit)
        return UNIT_SCALE_FACTORS["v"]

    def _scale_to_target(self, unit: str) -> float:
        if not unit:
            return UNIT_SCALE_FACTORS["v"]
        normalized = unit.strip().lower().replace("μ", "µ")
        return UNIT_SCALE_FACTORS.get(normalized, UNIT_SCALE_FACTORS["v"])

    def _add_annotations_for_path(self, ts_path: Path) -> None:
        if self._times is None or self._times.size == 0:
            return

        annotations = self._load_annotations(ts_path)
        if not annotations:
            return

        self._annotations = list(annotations)
        for annotation in annotations:
            self._render_annotation(annotation)
        self._update_annotation_filter_options()
        self._set_annotations_dirty(False)

    def _load_annotations(self, ts_path: Path) -> List[Annotation]:
        csv_path = ts_path.with_suffix(".csv")
        if not csv_path.exists():
            LOGGER.info("No annotation CSV found at %s", csv_path)
            return []

        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, skipinitialspace=True)
                if reader.fieldnames is None:
                    LOGGER.warning("Annotation CSV missing headers: %s", csv_path)
                    return []

                normalized = {name.strip(): name for name in reader.fieldnames}
                required = {"onset", "duration", "description"}
                if not required.issubset(normalized):
                    LOGGER.warning("Annotation CSV missing required columns: %s", csv_path)
                    return []

                annotations: List[Annotation] = []
                for row in reader:
                    onset = self._parse_float(row.get(normalized["onset"]))
                    duration = self._parse_float(row.get(normalized["duration"]))
                    description = (row.get(normalized["description"]) or "").strip()

                    if onset is None or duration is None or duration <= 0 or not description:
                        continue

                    annotations.append(
                        Annotation(onset=onset, duration=duration, description=description)
                    )

                return annotations
        except (OSError, csv.Error) as exc:
            LOGGER.warning("Failed to read annotations from %s: %s", csv_path, exc)
            return []

    def _parse_float(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    def _populate_channel_list(self) -> None:
        if self.raw is None:
            return

        self.channel_list.blockSignals(True)
        self.show_all_checkbox.blockSignals(True)
        self.show_all_checkbox.setChecked(False)
        self.channel_list.clear()

        defaults_present: Set[str] = set()
        for name in self.raw.ch_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            is_default = name in DEFAULT_VISIBLE_CHANNELS
            if is_default:
                defaults_present.add(name)
            item.setCheckState(Qt.Checked if is_default else Qt.Unchecked)
            self.channel_list.addItem(item)

        self.channel_list.blockSignals(False)
        self.show_all_checkbox.blockSignals(False)
        self._selected_channels = defaults_present

    def _channel_indices(self) -> List[int]:
        if self.raw is None:
            return []

        if self.show_all_checkbox.isChecked() or not self._selected_channels:
            return list(range(len(self.raw.ch_names)))

        return [idx for idx, name in enumerate(self.raw.ch_names) if name in self._selected_channels]

    def _on_show_all_channels(self, state: int) -> None:
        if state == Qt.Checked:
            self._selected_channels.clear()
        else:
            self._selected_channels = self._checked_channel_names()
        self._replot()

    def _on_channel_item_changed(self, item: QListWidgetItem) -> None:
        name = item.text()
        if item.checkState() == Qt.Checked:
            self._selected_channels.add(name)
        else:
            self._selected_channels.discard(name)
        if not self.show_all_checkbox.isChecked():
            self._replot()

    def _checked_channel_names(self) -> Set[str]:
        names: Set[str] = set()
        for index in range(self.channel_list.count()):
            item = self.channel_list.item(index)
            if item.checkState() == Qt.Checked:
                names.add(item.text())
        return names

    def _replot(self) -> None:
        self._clear_plot(hide_cursor=False)
        self._plot_data()
        self._ensure_view_range(self._last_cursor_time)

    def _adjust_zoom(self, multiplier: float, anchor_time: Optional[float] = None) -> None:
        self.view_span_seconds = max(
            self.min_span_seconds, min(self.view_span_seconds * multiplier, self._max_span_seconds())
        )
        if anchor_time is not None:
            self._last_cursor_time = max(0.0, anchor_time)
        self.zoom_label.setText(self._zoom_label_text())
        self._ensure_view_range(self._last_cursor_time)

    def _reset_zoom(self) -> None:
        self.view_span_seconds = min(self.default_view_span_seconds, self._max_span_seconds())
        self.zoom_label.setText(self._zoom_label_text())
        self._ensure_view_range(self._last_cursor_time)

    def _zoom_label_text(self) -> str:
        return f"Zoom window: {self.view_span_seconds:.2f}s"

    def _ensure_view_range(self, target_time: float) -> None:
        if self._times is None or self._times.size == 0:
            return

        self._last_cursor_time = max(0.0, min(target_time, float(self._times[-1])))
        half_span = self.view_span_seconds / 2
        data_end = float(self._times[-1])

        x_min = self._last_cursor_time - half_span
        x_max = self._last_cursor_time + half_span

        if x_min < 0:
            x_min = 0.0
            x_max = min(self.view_span_seconds, data_end)

        if x_max > data_end:
            x_max = data_end
            x_min = max(0.0, data_end - self.view_span_seconds)

        self.plot_widget.setXRange(x_min, x_max, padding=0)
        center = (x_min + x_max) / 2 if x_max > x_min else self._last_cursor_time
        self.cursor_line.setPos(center)
        self.cursor_line.show()
        self.plot_widget.enableAutoRange(y=True)

    def _max_span_seconds(self) -> float:
        if self._times is None or self._times.size == 0:
            return max(self.view_span_seconds, self.min_span_seconds)
        duration = float(self._times[-1])
        return max(self.min_span_seconds, duration)

    def _clear_plot(self, hide_cursor: bool = True) -> None:
        for curve in self._plotted_curves:
            self.plot_widget.removeItem(curve)
        self._plotted_curves.clear()
        self.plot_widget.enableAutoRange()
        if hide_cursor:
            self.cursor_line.hide()

    def _clear_annotations(self) -> None:
        for item in self._annotation_items:
            self.plot_widget.removeItem(item.region)
        self._annotation_items.clear()
        self._annotation_by_region.clear()
        if self._annotation_drag_preview is not None:
            self.plot_widget.removeItem(self._annotation_drag_preview)
            self._annotation_drag_preview = None

    def _color_for_description(self, description: str) -> pg.Color:
        if description not in self._annotation_colors:
            palette_index = len(self._annotation_colors) % len(ANNOTATION_PALETTE)
            self._annotation_colors[description] = pg.mkColor(ANNOTATION_PALETTE[palette_index])
        return self._annotation_colors[description]

    def _pen_for_channel(self, channel_name: str, index: int) -> pg.Pen:
        if channel_name == PRIMARY_CHANNEL:
            return pg.mkPen(CHANNEL_PALETTE[0], width=1.5)

        palette_index = (index + 1) % len(CHANNEL_PALETTE)
        return pg.mkPen(CHANNEL_PALETTE[palette_index], width=1)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if (
            obj is self.plot_widget.viewport()
            and event.type() == QEvent.Wheel
            and event.modifiers() & Qt.ControlModifier
        ):
            delta = event.angleDelta().y()
            multiplier = 0.8 if delta > 0 else 1.25
            anchor_time = self._time_at_position(event.pos())
            self._adjust_zoom(multiplier, anchor_time=anchor_time)
            return True
        if obj is self.plot_widget.viewport():
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.RightButton:
                    self._handle_annotation_context_menu(event.pos())
                    return True
                if event.button() == Qt.LeftButton and self.annotation_mode_checkbox.isChecked():
                    self._start_annotation_drag(event.pos())
                    return True
            if event.type() == QEvent.MouseMove and self._annotation_dragging:
                self._update_annotation_drag(event.pos())
                return True
            if event.type() == QEvent.MouseButtonRelease and self._annotation_dragging:
                if event.button() == Qt.LeftButton:
                    self._finalize_annotation_drag(event.pos())
                    return True

        return super().eventFilter(obj, event)

    def _time_at_position(self, pos) -> Optional[float]:
        view_box = self.plot_widget.getPlotItem().getViewBox()
        if view_box is None:
            return None

        scene_pos = self.plot_widget.mapToScene(pos)
        view_pos = view_box.mapSceneToView(scene_pos)
        return view_pos.x()

    def _render_annotation(self, annotation: Annotation) -> None:
        if self._times is None or self._times.size == 0:
            return

        data_end = float(self._times[-1])
        start = max(0.0, annotation.onset)
        end = annotation.onset + annotation.duration
        if not math.isfinite(end):
            return
        end = min(end, data_end)

        if end <= start or start >= data_end:
            return

        base_color = self._color_for_description(annotation.description)
        brush_color = pg.mkColor(base_color)
        brush_color.setAlpha(60)
        pen_color = pg.mkColor(base_color)
        pen_color.setAlpha(160)

        region = pg.LinearRegionItem(
            values=[start, end],
            brush=pg.mkBrush(brush_color),
            pen=pg.mkPen(pen_color, width=1),
            movable=False,
        )
        region.setZValue(10 + len(self._annotation_items))
        region.setToolTip(annotation.description)
        region.setVisible(self._annotation_visible(annotation))
        self.plot_widget.addItem(region)
        item = AnnotationItem(annotation=annotation, region=region)
        self._annotation_items.append(item)
        self._annotation_by_region[region] = item

    def _set_annotations_dirty(self, dirty: bool) -> None:
        self._annotations_dirty = dirty
        if dirty:
            self.annotations_dirty_label.setText("Unsaved changes")
        else:
            self.annotations_dirty_label.setText("")
        self.save_annotations_button.setEnabled(dirty and self._last_ts_path is not None)

    def _save_annotations(self) -> None:
        if self._last_ts_path is None:
            self.status_label.setText("No time series loaded to save annotations.")
            return

        csv_path = self._last_ts_path.with_suffix(".csv")
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["onset", "duration", "description"])
                writer.writeheader()
                for annotation in sorted(self._annotations, key=lambda entry: entry.onset):
                    writer.writerow(
                        {
                            "onset": f"{annotation.onset:.6f}",
                            "duration": f"{annotation.duration:.6f}",
                            "description": annotation.description,
                        }
                    )
        except OSError as exc:
            self.status_label.setText(f"Failed to save annotations: {exc}")
            return

        self.status_label.setText(f"Saved annotations to {csv_path}.")
        self._set_annotations_dirty(False)

    def _handle_annotation_context_menu(self, pos) -> None:
        annotation_item = self._annotation_item_at(pos)
        if annotation_item is None:
            return

        menu = QMenu(self)
        delete_action = menu.addAction("Delete annotation")
        selected_action = menu.exec_(self.plot_widget.viewport().mapToGlobal(pos))
        if selected_action == delete_action:
            self._delete_annotation(annotation_item)

    def _annotation_item_at(self, pos) -> Optional[AnnotationItem]:
        scene_pos = self.plot_widget.mapToScene(pos)
        for item in self.plot_widget.scene().items(scene_pos):
            annotation_item = self._annotation_by_region.get(item)
            if annotation_item is not None:
                return annotation_item
        return None

    def _delete_annotation(self, annotation_item: AnnotationItem) -> None:
        if annotation_item.annotation in self._annotations:
            self._annotations.remove(annotation_item.annotation)
        if annotation_item in self._annotation_items:
            self._annotation_items.remove(annotation_item)
        self._annotation_by_region.pop(annotation_item.region, None)
        self.plot_widget.removeItem(annotation_item.region)
        self._set_annotations_dirty(True)
        self._update_annotation_filter_options()
        self.status_label.setText("Annotation deleted.")

    def _start_annotation_drag(self, pos) -> None:
        time_value = self._time_at_position(pos)
        if time_value is None or self._times is None or self._times.size == 0:
            return

        clamped_time = max(0.0, min(time_value, float(self._times[-1])))
        self._annotation_dragging = True
        self._annotation_drag_start = clamped_time

        if self._annotation_drag_preview is None:
            brush_color = pg.mkColor("#90caf9")
            brush_color.setAlpha(80)
            pen_color = pg.mkColor("#42a5f5")
            pen_color.setAlpha(180)
            self._annotation_drag_preview = pg.LinearRegionItem(
                values=[clamped_time, clamped_time],
                brush=pg.mkBrush(brush_color),
                pen=pg.mkPen(pen_color, width=1, style=Qt.DashLine),
                movable=False,
            )
            self._annotation_drag_preview.setZValue(20)
            self.plot_widget.addItem(self._annotation_drag_preview)
        else:
            self._annotation_drag_preview.setRegion([clamped_time, clamped_time])

    def _update_annotation_drag(self, pos) -> None:
        if self._annotation_drag_start is None or self._annotation_drag_preview is None:
            return

        time_value = self._time_at_position(pos)
        if time_value is None or self._times is None:
            return

        clamped_time = max(0.0, min(time_value, float(self._times[-1])))
        start = self._annotation_drag_start
        self._annotation_drag_preview.setRegion([start, clamped_time])

    def _finalize_annotation_drag(self, pos) -> None:
        if self._annotation_drag_start is None or self._times is None:
            self._reset_annotation_drag()
            return

        time_value = self._time_at_position(pos)
        if time_value is None:
            time_value = self._annotation_drag_start

        data_end = float(self._times[-1])
        start = max(0.0, min(self._annotation_drag_start, data_end))
        end = max(0.0, min(time_value, data_end))
        onset = min(start, end)
        duration = abs(end - start)

        if duration < MIN_ANNOTATION_DURATION:
            onset = min(onset, data_end)
            duration = MIN_ANNOTATION_DURATION
            if onset + duration > data_end:
                onset = max(0.0, data_end - duration)

        description = self._prompt_for_description()
        if description:
            annotation = Annotation(onset=onset, duration=duration, description=description)
            self._annotations.append(annotation)
            self._render_annotation(annotation)
            self._set_annotations_dirty(True)
            self._update_annotation_filter_options()
            self.status_label.setText("Annotation added.")

        self._reset_annotation_drag()

    def _prompt_for_description(self) -> str:
        default = self._last_annotation_description or ""
        text, ok = QInputDialog.getText(
            self,
            "Add annotation",
            "Description:",
            text=default,
        )
        description = text.strip()
        if ok and description:
            self._last_annotation_description = description
            return description
        return ""

    def _reset_annotation_drag(self) -> None:
        self._annotation_dragging = False
        self._annotation_drag_start = None
        if self._annotation_drag_preview is not None:
            self.plot_widget.removeItem(self._annotation_drag_preview)
            self._annotation_drag_preview = None

    def _on_annotation_mode_changed(self, state: int) -> None:
        if state == Qt.Checked:
            self.plot_widget.setCursor(Qt.CrossCursor)
        else:
            self.plot_widget.setCursor(Qt.ArrowCursor)

    def jump_to_next_annotation(self) -> None:
        """Jump the view to the next annotation onset."""

        self._jump_to_annotation(direction="next")

    def jump_to_previous_annotation(self) -> None:
        """Jump the view to the previous annotation onset."""

        self._jump_to_annotation(direction="previous")

    def _jump_to_annotation(self, direction: str) -> None:
        if self._times is None or not self._annotations:
            self.status_label.setText("No annotations available.")
            return

        annotations = self._filtered_annotations()
        if not annotations:
            self.status_label.setText("No annotations match the current filter.")
            return

        current_position = self._last_cursor_time
        if direction == "next":
            candidates = [annotation for annotation in annotations if annotation.onset > current_position]
            target = min(candidates, key=lambda entry: entry.onset, default=None)
        else:
            candidates = [annotation for annotation in annotations if annotation.onset < current_position]
            target = max(candidates, key=lambda entry: entry.onset, default=None)

        if target is None:
            label = "next" if direction == "next" else "previous"
            self.status_label.setText(f"No {label} annotation.")
            return

        self._ensure_view_range(target.onset)
        self.annotation_jump_requested.emit(target.onset)
        self.status_label.setText(
            f"Jumped to annotation '{target.description}' at {target.onset:.2f}s."
        )

    def _on_annotation_filter_changed(self) -> None:
        data = self.annotation_filter_combo.currentData()
        if data is None:
            data = self.FILTER_ALL
        self._annotation_filter_value = data
        self._apply_annotation_filter()

    def _annotation_visible(self, annotation: Annotation) -> bool:
        if self._annotation_filter_value == self.FILTER_ALL:
            return True
        if self._annotation_filter_value == self.FILTER_NONE:
            return False
        return annotation.description == self._annotation_filter_value

    def _filtered_annotations(self) -> List[Annotation]:
        if self._annotation_filter_value == self.FILTER_ALL:
            return list(self._annotations)
        if self._annotation_filter_value == self.FILTER_NONE:
            return []
        return [
            annotation
            for annotation in self._annotations
            if annotation.description == self._annotation_filter_value
        ]

    def _apply_annotation_filter(self) -> None:
        for item in self._annotation_items:
            item.region.setVisible(self._annotation_visible(item.annotation))

    def _update_annotation_filter_options(self, force_all: bool = False) -> None:
        descriptions = sorted({annotation.description for annotation in self._annotations})
        current_value = self._annotation_filter_value
        if force_all:
            current_value = self.FILTER_ALL

        self.annotation_filter_combo.blockSignals(True)
        self.annotation_filter_combo.clear()
        self.annotation_filter_combo.addItem("All", self.FILTER_ALL)
        for description in descriptions:
            self.annotation_filter_combo.addItem(description, description)
        none_label = "None"
        if any(description == none_label for description in descriptions):
            none_label = "Hide all"
        self.annotation_filter_combo.addItem(none_label, self.FILTER_NONE)

        selected_index = 0
        for index in range(self.annotation_filter_combo.count()):
            if self.annotation_filter_combo.itemData(index) == current_value:
                selected_index = index
                break

        self.annotation_filter_combo.setCurrentIndex(selected_index)
        self.annotation_filter_combo.blockSignals(False)
        self._annotation_filter_value = self.annotation_filter_combo.currentData()
        self._apply_annotation_filter()
