"""Time series loading and visualization helpers."""
from __future__ import annotations

import csv
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import mne
import numpy as np
import pyqtgraph as pg
from mne.io.constants import FIFF
from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.paths import derive_annotation_path, derive_time_series_path

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
MIN_LANE_HEIGHT = 140
LANE_PADDING_RATIO = 0.05
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
ANNOTATION_NUDGE_FPS = 30.0

LOGGER = logging.getLogger(__name__)


@dataclass
class Annotation:
    onset: float
    duration: float
    description: str


@dataclass
class AnnotationItem:
    annotation: Annotation
    region: pg.LinearRegionItem


class TimeSeriesViewer(QWidget):
    """Widget that renders time series data alongside the video frames."""

    annotation_jump_requested = pyqtSignal(float)
    FILTER_ALL = "__all__"
    FILTER_NONE = "__none__"

    def __init__(
        self,
        max_points: int = 10000,
        parent: Optional[QWidget] = None,
        time_series_root: Optional[Path] = None,
        annotation_root: Optional[Path] = None,
        ui_settings: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__(parent)
        self.max_points = max_points
        self.raw: Optional[mne.io.BaseRaw] = None
        self._times: Optional[np.ndarray] = None
        self._selected_channels: Set[str] = set()
        self._last_cursor_time: float = 0.0
        self.default_view_span_seconds: float = 5.0
        self.view_span_seconds: float = self.default_view_span_seconds
        self.min_span_seconds: float = 0.1
        self._last_ts_path: Optional[Path] = None
        self._last_video_path: Optional[Path] = None
        self._last_annotation_path: Optional[Path] = None
        self._expected_ts_path: Optional[Path] = None
        self._expected_annotation_path: Optional[Path] = None
        self.time_series_root: Path = time_series_root or Path.cwd()
        self.annotation_root: Path = annotation_root or self.time_series_root
        self._annotation_by_region: dict[pg.LinearRegionItem, AnnotationItem] = {}
        self._annotation_colors: dict[str, pg.Color] = {}
        self._annotations: List[Annotation] = []
        self._annotations_dirty = False
        self._annotation_dragging = False
        self._annotation_drag_start: Optional[float] = None
        self._annotation_drag_preview: Optional[pg.LinearRegionItem] = None
        self._last_annotation_description = ""
        self._annotation_filter_value = self.FILTER_ALL
        self._selected_annotation: Optional[Annotation] = None
        self._ear_gain_enabled = False
        self._ear_gain_value = 1.0
        self._ear_baseline_value = (
            float(ui_settings.get("ear_baseline", 0.0)) if ui_settings else 0.0
        )
        self._lane_widgets: List[pg.PlotWidget] = []
        self._lane_curves: dict[pg.PlotWidget, List[pg.PlotDataItem]] = {}
        self._lane_cursor_lines: dict[pg.PlotWidget, pg.InfiniteLine] = {}
        self._lane_series: dict[pg.PlotWidget, tuple[np.ndarray, np.ndarray]] = {}
        self._lane_baselines: dict[pg.PlotWidget, pg.InfiniteLine] = {}
        self._lane_baseline_kind: dict[pg.PlotWidget, str] = {}
        self._annotation_items_by_widget: dict[pg.PlotWidget, List[AnnotationItem]] = {}
        self._syncing_annotation_regions = False
        self._annotation_sample_scatter: Optional[pg.ScatterPlotItem] = None
        self._annotation_min_marker: Optional[pg.ScatterPlotItem] = None

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
        gain_row = QHBoxLayout()
        self.ear_gain_checkbox = QCheckBox("Boost EAR-avg_ear")
        self.ear_gain_checkbox.stateChanged.connect(self._on_ear_gain_changed)
        self.ear_gain_spinbox = QDoubleSpinBox()
        self.ear_gain_spinbox.setRange(1.0, 1.0e9)
        self.ear_gain_spinbox.setDecimals(2)
        self.ear_gain_spinbox.setSingleStep(1.0)
        self.ear_gain_spinbox.setKeyboardTracking(True)
        self.ear_gain_spinbox.setValue(self._ear_gain_value)
        self.ear_gain_spinbox.valueChanged.connect(self._on_ear_gain_changed)
        self.ear_gain_label = QLabel(self._ear_gain_label_text())
        self.ear_gain_x2_button = QPushButton("×2")
        self.ear_gain_x2_button.clicked.connect(lambda: self._adjust_ear_gain(2.0))
        self.ear_gain_x5_button = QPushButton("×5")
        self.ear_gain_x5_button.clicked.connect(lambda: self._adjust_ear_gain(5.0))
        self.ear_gain_x10_button = QPushButton("×10")
        self.ear_gain_x10_button.clicked.connect(lambda: self._adjust_ear_gain(10.0))

        gain_row.addWidget(self.ear_gain_checkbox)
        gain_row.addWidget(QLabel("Gain:"))
        gain_row.addWidget(self.ear_gain_spinbox)
        gain_row.addWidget(self.ear_gain_x2_button)
        gain_row.addWidget(self.ear_gain_x5_button)
        gain_row.addWidget(self.ear_gain_x10_button)
        gain_row.addWidget(self.ear_gain_label)
        gain_row.addWidget(QLabel("EAR baseline (red line):"))
        self.ear_baseline_spinbox = QDoubleSpinBox()
        self.ear_baseline_spinbox.setRange(-1.0e9, 1.0e9)
        self.ear_baseline_spinbox.setDecimals(3)
        self.ear_baseline_spinbox.setSingleStep(0.1)
        self.ear_baseline_spinbox.setValue(self._ear_baseline_value)
        self.ear_baseline_spinbox.valueChanged.connect(self._on_ear_baseline_changed)
        gain_row.addWidget(self.ear_baseline_spinbox)
        gain_row.addStretch()
        control_layout.addLayout(gain_row)
        self._controls_container.setLayout(control_layout)

        self.plot_widget = pg.PlotWidget(background="w")
        self._configure_plot_widget(self.plot_widget)
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
        self.plot_widget.addItem(self.cursor_line)
        self.cursor_line.hide()
        self._register_lane_widget(self.plot_widget, self.cursor_line)

        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout()
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.setSpacing(2)
        self._plot_layout.addWidget(self.plot_widget)
        self._plot_container.setLayout(self._plot_layout)
        self._plot_scroll_area = QScrollArea()
        self._plot_scroll_area.setWidgetResizable(True)
        self._plot_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._plot_scroll_area.setWidget(self._plot_container)

        annotation_controls = QWidget()
        annotation_layout = QHBoxLayout()
        annotation_layout.setContentsMargins(0, 0, 0, 0)

        left_controls = QWidget()
        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Annotations:"))
        self.annotation_filter_combo = QComboBox()
        self.annotation_filter_combo.currentIndexChanged.connect(self._on_annotation_filter_changed)
        left_layout.addWidget(self.annotation_filter_combo)
        self.save_annotations_button = QPushButton("Save annotations")
        self.save_annotations_button.clicked.connect(self._save_annotations)
        self.save_annotations_button.setEnabled(False)
        left_layout.addWidget(self.save_annotations_button)
        left_controls.setLayout(left_layout)

        center_controls = QWidget()
        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.annotations_dirty_label = QLabel("")
        self.annotations_dirty_label.setStyleSheet("color: #d32f2f;")
        self.annotations_dirty_label.setAlignment(Qt.AlignCenter)
        reserved_width = (
            self.annotations_dirty_label.fontMetrics().horizontalAdvance("Unsaved changes") + 12
        )
        self.annotations_dirty_label.setFixedWidth(reserved_width)
        center_layout.addWidget(self.annotations_dirty_label)
        center_controls.setLayout(center_layout)

        right_controls = QWidget()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addStretch()
        right_layout.addWidget(self._build_annotation_nudge_group())
        right_layout.addWidget(
            QLabel("Shortcuts: [ or P = prev, ] or N = next, Ctrl+N = next (EAR min)")
        )
        right_controls.setLayout(right_layout)

        annotation_layout.addWidget(left_controls, 1)
        annotation_layout.addWidget(center_controls, 0)
        annotation_layout.addWidget(right_controls, 1)
        annotation_controls.setLayout(annotation_layout)

        self.status_label = QLabel("Load a video to view synchronized time series data.")

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self._plot_scroll_area)
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

        self.time_series_root = processed_root

    def set_annotation_root(self, annotation_root: Path) -> None:
        """Override the root used for annotation CSV files."""

        self.annotation_root = annotation_root

    def expected_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        """Return the expected FIF and CSV paths for the last video selection."""

        return self._expected_ts_path, self._expected_annotation_path

    def last_loaded_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        """Return the last successfully loaded FIF and CSV paths."""

        return self._last_ts_path, self._last_annotation_path

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
        self._last_video_path = None
        self._last_annotation_path = None
        self._expected_ts_path = None
        self._expected_annotation_path = None
        self._annotations = []
        self._set_annotations_dirty(False)
        self._update_annotation_filter_options(force_all=True)
        self._selected_annotation = None

        if video_path is None:
            self.status_label.setText("Select a video to view its time series.")
            self.cursor_line.hide()
            return

        try:
            ts_path = derive_time_series_path(video_path, processed_root=self.time_series_root)
            self._expected_ts_path = ts_path
        except ValueError as exc:  # pragma: no cover - guardrails for unexpected paths
            self.status_label.setText(str(exc))
            self.cursor_line.hide()
            return
        csv_path = derive_annotation_path(
            video_path, processed_root=self.time_series_root, csv_root=self.annotation_root
        )
        self._expected_annotation_path = csv_path
        if not ts_path.exists():
            self.status_label.setText(
                f"Time series file not found at {ts_path}. "
                "Update your configuration or verify the dataset root."
            )
            self.cursor_line.hide()
            return

        self.status_label.setText(f"Loaded time series from {ts_path}.")
        self._last_ts_path = ts_path
        self._last_video_path = video_path
        self._last_annotation_path = csv_path
        self.raw = mne.io.read_raw_fif(str(ts_path), preload=True, verbose="ERROR")
        self._times = self.raw.times
        self._populate_channel_list()
        self._plot_data()
        self._add_annotations_for_path(csv_path)
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

        channel_names = [self.raw.ch_names[index] for index in picks]
        channel_types = self.raw.get_channel_types(picks=picks)
        data, picks, channel_names, channel_types = self._order_channels_for_display(
            data, picks, channel_names, channel_types
        )
        lane_count = len(channel_names)
        data, normalized_unit = self._normalize_channel_units(
            data, picks, channel_names, channel_types
        )
        data, channel_names = self._apply_ear_gain(data, channel_names)

        self._ensure_lane_count(lane_count)
        self._plot_lane_data(
            data,
            times,
            channel_names,
            channel_types,
            normalized_unit,
        )
        self._sync_lane_annotations()

        for cursor_line in self._lane_cursor_lines.values():
            cursor_line.show()
        total_channels = len(self.raw.ch_names) if self.raw else len(picks)
        self.status_label.setText(
            f"Displaying {len(picks)} of {total_channels} channel(s) from {self._last_ts_path}"
        )

    def _plot_lane_data(
        self,
        data: np.ndarray,
        times: np.ndarray,
        channel_names: List[str],
        channel_types: List[str],
        units: Optional[str],
    ) -> None:
        for idx, (channel, channel_name, channel_type) in enumerate(
            zip(data, channel_names, channel_types)
        ):
            widget = self._lane_widgets[idx]
            show_bottom_axis = idx == len(channel_names) - 1
            self._plot_single_lane(
                widget,
                channel,
                times,
                channel_name,
                channel_type,
                units,
                show_bottom_axis=show_bottom_axis,
            )

    def _plot_single_lane(
        self,
        widget: pg.PlotWidget,
        channel: np.ndarray,
        times: np.ndarray,
        channel_name: str,
        channel_type: str,
        units: Optional[str],
        show_bottom_axis: bool,
    ) -> None:
        axis = widget.getPlotItem().getAxis("bottom")
        axis.setStyle(showValues=show_bottom_axis)
        if show_bottom_axis:
            widget.setLabel("bottom", "Time", units="s")
        else:
            widget.setLabel("bottom", "")
        if units:
            widget.setLabel("left", channel_name, units=units)
        else:
            widget.setLabel("left", channel_name)
        curves_to_add: List[pg.GraphicsObject] = []
        if channel_name == EAR_AVG_CHANNEL:
            pen = self._pen_for_channel(channel_name, 0)
            scatter = pg.ScatterPlotItem(
                times,
                channel,
                pen=pen,
                brush=pg.mkBrush(pen.color()),
                size=6,
                symbol="o",
            )
            base_color = pen.color()
            line_color = pg.mkColor(base_color)
            line_color.setAlphaF(0.3)
            line_pen = pg.mkPen(line_color, width=1)
            line_curve = widget.plot(times, channel, pen=line_pen)
            widget.addItem(scatter)
            curves_to_add.extend([line_curve, scatter])
        else:
            curve = widget.plot(times, channel, pen=self._pen_for_channel(channel_name, 0))
            curve.setDownsampling(auto=True, method="peak")
            curves_to_add.append(curve)
        self._lane_curves[widget].extend(curves_to_add)
        self._lane_series[widget] = (times, channel)
        self._add_baseline(widget, channel_name, channel_type)

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

    def _apply_ear_gain(
        self, data: np.ndarray, channel_names: List[str]
    ) -> tuple[np.ndarray, List[str]]:
        if not self._ear_gain_enabled or self._ear_gain_value == 1.0:
            return data, channel_names

        adjusted = data.copy()
        display_names = list(channel_names)
        for idx, name in enumerate(channel_names):
            if name == EAR_AVG_CHANNEL:
                adjusted[idx] *= self._ear_gain_value
                display_names[idx] = f"{name} (×{self._format_gain(self._ear_gain_value)})"
        return adjusted, display_names

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

    def _configure_plot_widget(self, widget: pg.PlotWidget) -> None:
        widget.showGrid(x=True, y=True, alpha=0.3)
        widget.setLabel("bottom", "Time", units="s")
        widget.setLabel("left", "Channels")
        widget.getPlotItem().getAxis("left").setWidth(60)
        widget.viewport().installEventFilter(self)
        widget.setFocusPolicy(Qt.StrongFocus)

    def _ear_gain_label_text(self) -> str:
        status = "on" if self._ear_gain_enabled else "off"
        return f"EAR-avg_ear gain: ×{self._format_gain(self._ear_gain_value)} ({status})"

    def _format_gain(self, value: float) -> str:
        if value.is_integer():
            return f"{value:,.0f}"
        if value >= 1e7:
            return f"{value:,.2f}"
        return f"{value:,.2f}"

    def _adjust_ear_gain(self, multiplier: float) -> None:
        self.ear_gain_spinbox.setValue(self.ear_gain_spinbox.value() * multiplier)

    def _on_ear_gain_changed(self, *_: object) -> None:
        self._ear_gain_enabled = self.ear_gain_checkbox.isChecked()
        self._ear_gain_value = float(self.ear_gain_spinbox.value())
        self.ear_gain_label.setText(self._ear_gain_label_text())
        self._replot()

    def _on_ear_baseline_changed(self, value: float) -> None:
        self._ear_baseline_value = float(value)
        for widget, baseline in self._lane_baselines.items():
            if self._lane_baseline_kind.get(widget) == "ear":
                baseline.setValue(self._ear_baseline_value)

    def _add_annotations_for_path(self, csv_path: Path) -> None:
        if self._times is None or self._times.size == 0:
            return

        annotations = self._load_annotations(csv_path)
        if not annotations:
            return

        self._annotations = list(annotations)
        for annotation in annotations:
            self._render_annotation(annotation)
        self._update_annotation_filter_options()
        self._set_annotations_dirty(False)

    def _load_annotations(self, csv_path: Path) -> List[Annotation]:
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

        return [
            idx
            for idx, name in enumerate(self.raw.ch_names)
            if name in self._selected_channels
        ]

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
            self.min_span_seconds,
            min(self.view_span_seconds * multiplier, self._max_span_seconds()),
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

        for widget in self._lane_widgets:
            widget.setXRange(x_min, x_max, padding=0)
        center = (x_min + x_max) / 2 if x_max > x_min else self._last_cursor_time
        for cursor_line in self._lane_cursor_lines.values():
            cursor_line.setPos(center)
            cursor_line.show()
        self._update_lane_scales(x_min, x_max)

    def _max_span_seconds(self) -> float:
        if self._times is None or self._times.size == 0:
            return max(self.view_span_seconds, self.min_span_seconds)
        duration = float(self._times[-1])
        return max(self.min_span_seconds, duration)

    def _clear_plot(self, hide_cursor: bool = True) -> None:
        for widget, curves in self._lane_curves.items():
            for curve in curves:
                widget.removeItem(curve)
            curves.clear()
        for widget, baseline in self._lane_baselines.items():
            widget.removeItem(baseline)
        self._lane_baselines.clear()
        self._lane_baseline_kind.clear()
        self._lane_series.clear()
        self._clear_annotation_samples()
        if hide_cursor:
            for cursor_line in self._lane_cursor_lines.values():
                cursor_line.hide()

    def _clear_annotations(self) -> None:
        for widget, items in self._annotation_items_by_widget.items():
            for item in items:
                widget.removeItem(item.region)
            items.clear()
        self._annotation_by_region.clear()
        if self._annotation_drag_preview is not None:
            self.plot_widget.removeItem(self._annotation_drag_preview)
            self._annotation_drag_preview = None
        self._selected_annotation = None

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
            obj in self._plot_viewports()
            and event.type() == QEvent.Wheel
            and event.modifiers() & Qt.ControlModifier
        ):
            delta = event.angleDelta().y()
            multiplier = 0.8 if delta > 0 else 1.25
            anchor_time = self._time_at_position(event.pos(), self._widget_for_viewport(obj))
            self._adjust_zoom(multiplier, anchor_time=anchor_time)
            return True
        if obj is self.plot_widget.viewport():
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.RightButton:
                    self._handle_annotation_context_menu(event.pos())
                    return True
                if event.button() == Qt.LeftButton:
                    if event.modifiers() & Qt.ControlModifier:
                        self._start_annotation_drag(event.pos())
                        return True
                    annotation_item = self._annotation_item_at(event.pos())
                    if annotation_item is not None:
                        self._set_selected_annotation(annotation_item.annotation)
            if event.type() == QEvent.MouseMove and self._annotation_dragging:
                self._update_annotation_drag(event.pos())
                return True
            if event.type() == QEvent.MouseButtonRelease and self._annotation_dragging:
                if event.button() == Qt.LeftButton:
                    self._finalize_annotation_drag(event.pos())
                    return True

        return super().eventFilter(obj, event)

    def _plot_viewports(self) -> set[object]:
        return {widget.viewport() for widget in self._lane_widgets}

    def _widget_for_viewport(self, viewport: object) -> pg.PlotWidget:
        for widget in self._lane_widgets:
            if viewport is widget.viewport():
                return widget
        return self.plot_widget

    def _time_at_position(self, pos, widget: Optional[pg.PlotWidget] = None) -> Optional[float]:
        plot_widget = widget or self.plot_widget
        view_box = plot_widget.getPlotItem().getViewBox()
        if view_box is None:
            return None

        scene_pos = plot_widget.mapToScene(pos)
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

        region = self._create_annotation_region(
            annotation,
            start,
            end,
            brush_color,
            pen_color,
            self.plot_widget,
            len(self._annotation_items_by_widget[self.plot_widget]),
        )
        item = AnnotationItem(annotation=annotation, region=region)
        self._annotation_items_by_widget[self.plot_widget].append(item)
        self._annotation_by_region[region] = item
        self._connect_annotation_region(item)
        for widget in self._lane_widgets:
            if widget is self.plot_widget:
                continue
            region = self._create_annotation_region(
                annotation,
                start,
                end,
                brush_color,
                pen_color,
                widget,
                len(self._annotation_items_by_widget[widget]),
            )
            lane_item = AnnotationItem(annotation=annotation, region=region)
            self._annotation_items_by_widget[widget].append(lane_item)
            self._connect_annotation_region(lane_item)

    def _set_annotations_dirty(self, dirty: bool) -> None:
        self._annotations_dirty = dirty
        if dirty:
            self.annotations_dirty_label.setText("Unsaved changes")
        else:
            self.annotations_dirty_label.setText("")
        self.save_annotations_button.setEnabled(dirty and self._annotation_csv_path() is not None)

    def _annotation_csv_path(self) -> Optional[Path]:
        if self._last_video_path is not None:
            try:
                return derive_annotation_path(
                    self._last_video_path,
                    processed_root=self.time_series_root,
                    csv_root=self.annotation_root,
                )
            except ValueError:
                return None
        if self._last_ts_path is not None:
            return self._last_ts_path.with_suffix(".csv")
        return None

    def _save_annotations(self) -> None:
        csv_path = self._annotation_csv_path()
        if csv_path is None:
            self.status_label.setText("No time series loaded to save annotations.")
            self._restore_plot_focus()
            return

        try:
            annotations = self._current_annotations_for_save()
            expected_rows = self._write_annotations_csv(csv_path, annotations)
            actual_rows = self._count_csv_rows(csv_path)
            if actual_rows != expected_rows:
                raise ValueError(
                    "Annotation CSV row count mismatch "
                    f"(expected {expected_rows}, got {actual_rows})."
                )
        except (OSError, csv.Error, ValueError) as exc:
            self.status_label.setText(f"Failed to save annotations: {exc}")
            self._restore_plot_focus()
            return

        self.status_label.setText(f"Saved annotations to {csv_path}.")
        self._set_annotations_dirty(False)
        self._restore_plot_focus()

    def save_annotations(self) -> None:
        """Public handler to save annotations and maintain focus."""
        self._save_annotations()

    def _restore_plot_focus(self) -> None:
        if self.plot_widget is None:
            return

        def focus_plot() -> None:
            self.plot_widget.setFocus(Qt.OtherFocusReason)

        QTimer.singleShot(0, focus_plot)

    def _handle_annotation_context_menu(self, pos) -> None:
        annotation_item = self._annotation_item_at(pos)
        if annotation_item is None:
            return

        self._set_selected_annotation(annotation_item.annotation)
        menu = QMenu(self)
        edit_action = menu.addAction("Edit label")
        delete_action = menu.addAction("Delete annotation")
        selected_action = menu.exec_(self.plot_widget.viewport().mapToGlobal(pos))
        if selected_action == edit_action:
            self._edit_annotation_description(annotation_item)
        if selected_action == delete_action:
            self._delete_annotation(annotation_item)

    def _annotation_item_at(self, pos) -> Optional[AnnotationItem]:
        scene_pos = self.plot_widget.mapToScene(pos)
        for item in self.plot_widget.scene().items(scene_pos):
            annotation_item = self._annotation_by_region.get(item)
            if annotation_item is not None:
                return annotation_item
        return None

    def _create_annotation_region(
        self,
        annotation: Annotation,
        start: float,
        end: float,
        brush_color: pg.Color,
        pen_color: pg.Color,
        widget: pg.PlotWidget,
        index: int,
    ) -> pg.LinearRegionItem:
        region = pg.LinearRegionItem(
            values=[start, end],
            brush=pg.mkBrush(brush_color),
            pen=pg.mkPen(pen_color, width=1),
            movable=True,
        )
        region.setZValue(10 + index)
        region.setToolTip(annotation.description)
        region.setVisible(self._annotation_visible(annotation))
        widget.addItem(region)
        return region

    def _connect_annotation_region(self, annotation_item: AnnotationItem) -> None:
        annotation_item.region.sigRegionChangeFinished.connect(
            lambda *_: self._on_annotation_region_changed(annotation_item)
        )

    def _on_annotation_region_changed(self, annotation_item: AnnotationItem) -> None:
        if self._syncing_annotation_regions:
            return

        region = annotation_item.region
        start, end = region.getRegion()
        onset, duration = self._normalize_region_values(start, end)
        annotation_item.annotation.onset = onset
        annotation_item.annotation.duration = duration
        self._sync_annotation_regions(
            annotation_item.annotation, onset, duration, skip_region=region
        )
        self._set_selected_annotation(annotation_item.annotation, announce=False)
        self._set_annotations_dirty(True)
        self.status_label.setText("Annotation updated.")

    def _normalize_region_values(self, start: float, end: float) -> tuple[float, float]:
        if self._times is not None and self._times.size > 0:
            data_end = float(self._times[-1])
        else:
            data_end = None

        start = max(0.0, start)
        end = max(0.0, end)
        if data_end is not None:
            start = min(start, data_end)
            end = min(end, data_end)

        onset = min(start, end)
        duration = abs(end - start)
        if duration < MIN_ANNOTATION_DURATION:
            duration = MIN_ANNOTATION_DURATION
            if data_end is not None and onset + duration > data_end:
                onset = max(0.0, data_end - duration)
        return onset, duration

    def _sync_annotation_regions(
        self,
        annotation: Annotation,
        onset: float,
        duration: float,
        skip_region: Optional[pg.LinearRegionItem] = None,
    ) -> None:
        self._syncing_annotation_regions = True
        end = onset + duration
        try:
            for items in self._annotation_items_by_widget.values():
                for item in items:
                    if item.annotation != annotation or item.region is skip_region:
                        continue
                    item.region.setRegion([onset, end])
        finally:
            self._syncing_annotation_regions = False

    def _edit_annotation_description(self, annotation_item: AnnotationItem) -> None:
        description = self._prompt_for_description(
            title="Edit annotation label",
            current_value=annotation_item.annotation.description,
        )
        if not description:
            return

        annotation_item.annotation.description = description
        self._refresh_annotation_visuals(annotation_item.annotation)
        self._update_annotation_filter_options()
        self._set_annotations_dirty(True)
        self.status_label.setText("Annotation label updated.")

    def _refresh_annotation_visuals(self, annotation: Annotation) -> None:
        base_color = self._color_for_description(annotation.description)
        brush_color = pg.mkColor(base_color)
        brush_color.setAlpha(60)
        pen_color = pg.mkColor(base_color)
        pen_color.setAlpha(160)

        for items in self._annotation_items_by_widget.values():
            for item in items:
                if item.annotation != annotation:
                    continue
                item.region.setToolTip(annotation.description)
                item.region.setBrush(pg.mkBrush(brush_color))
                item.region.setPen(pg.mkPen(pen_color, width=1))
                item.region.setVisible(self._annotation_visible(annotation))

    def _current_annotations_for_save(self) -> List[Annotation]:
        items = self._annotation_items_by_widget.get(self.plot_widget, [])
        annotations: List[Annotation] = []
        for item in items:
            if item.annotation not in self._annotations:
                continue
            start, end = item.region.getRegion()
            onset, duration = self._normalize_region_values(start, end)
            item.annotation.onset = onset
            item.annotation.duration = duration
            if item.annotation.description:
                annotations.append(item.annotation)
        self._annotations = list(annotations)
        return annotations

    def _write_annotations_csv(self, csv_path: Path, annotations: List[Annotation]) -> int:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_handle = tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=csv_path.parent,
            prefix=f"{csv_path.stem}_",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = Path(tmp_handle.name)
        expected_rows = 1
        try:
            writer = csv.DictWriter(
                tmp_handle,
                fieldnames=["onset", "duration", "description"],
            )
            writer.writeheader()
            for annotation in sorted(annotations, key=lambda entry: entry.onset):
                writer.writerow(
                    {
                        "onset": f"{annotation.onset:.6f}",
                        "duration": f"{annotation.duration:.6f}",
                        "description": annotation.description,
                    }
                )
                expected_rows += 1
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
        except Exception:
            tmp_handle.close()
            tmp_path.unlink(missing_ok=True)
            raise
        tmp_handle.close()

        if csv_path.exists():
            backup_dir = csv_path.parent / "backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{csv_path.stem}_{timestamp}.csv"
            shutil.copy2(csv_path, backup_path)
        os.replace(tmp_path, csv_path)
        if not csv_path.exists():
            raise OSError(f"Annotation CSV was not created at {csv_path}")
        if csv_path.stat().st_size == 0:
            raise OSError(f"Annotation CSV was empty at {csv_path}")
        return expected_rows

    def _count_csv_rows(self, csv_path: Path) -> int:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            return sum(1 for _ in reader)

    def _delete_annotation(self, annotation_item: AnnotationItem) -> None:
        if annotation_item.annotation in self._annotations:
            self._annotations.remove(annotation_item.annotation)
        self._annotation_by_region.pop(annotation_item.region, None)
        self.plot_widget.removeItem(annotation_item.region)
        self._annotation_items_by_widget[self.plot_widget] = [
            item
            for item in self._annotation_items_by_widget[self.plot_widget]
            if item is not annotation_item
        ]
        for widget, items in self._annotation_items_by_widget.items():
            if widget is self.plot_widget:
                continue
            for lane_item in list(items):
                if lane_item.annotation == annotation_item.annotation:
                    widget.removeItem(lane_item.region)
                    items.remove(lane_item)
        self._set_annotations_dirty(True)
        self._update_annotation_filter_options()
        self.status_label.setText("Annotation deleted.")
        if self._selected_annotation == annotation_item.annotation:
            self._selected_annotation = None

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
            self._set_selected_annotation(annotation, announce=False)
            self._set_annotations_dirty(True)
            self._update_annotation_filter_options()
            self.status_label.setText("Annotation added.")

        self._reset_annotation_drag()

    def _prompt_for_description(
        self, title: str = "Add annotation", current_value: str = ""
    ) -> str:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)

        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        label_combo = QComboBox(dialog)
        label_combo.setEditable(True)
        label_combo.setInsertPolicy(QComboBox.NoInsert)

        labels = sorted(
            {
                annotation.description
                for annotation in self._annotations
                if annotation.description
            }
        )
        label_combo.addItems(labels)

        initial_value = current_value or self._last_annotation_description or ""
        if initial_value:
            if initial_value not in labels:
                label_combo.insertItem(0, initial_value)
            label_combo.setCurrentText(initial_value)
        else:
            label_combo.setCurrentText("")

        line_edit = label_combo.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("Select an existing label or type a new one")

        form_layout.addRow("Label:", label_combo)
        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return ""

        description = label_combo.currentText().strip()
        if not description:
            return ""

        self._last_annotation_description = description
        return description

    def _reset_annotation_drag(self) -> None:
        self._annotation_dragging = False
        self._annotation_drag_start = None
        if self._annotation_drag_preview is not None:
            self.plot_widget.removeItem(self._annotation_drag_preview)
            self._annotation_drag_preview = None

    def _target_annotation(self, direction: str) -> Optional[Annotation]:
        if self._times is None or not self._annotations:
            self.status_label.setText("No annotations available.")
            return None

        annotations = self._filtered_annotations()
        if not annotations:
            self.status_label.setText("No annotations match the current filter.")
            return None

        current_position = self._last_cursor_time
        if direction == "next":
            candidates = [
                annotation
                for annotation in annotations
                if annotation.onset > current_position
            ]
            target = min(candidates, key=lambda entry: entry.onset, default=None)
        else:
            candidates = [
                annotation
                for annotation in annotations
                if annotation.onset < current_position
            ]
            target = max(candidates, key=lambda entry: entry.onset, default=None)

        if target is None:
            label = "next" if direction == "next" else "previous"
            self.status_label.setText(f"No {label} annotation.")
            return None

        return target

    def _annotation_minimum_time(
        self, annotation: Annotation
    ) -> tuple[
        Optional[float],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[str],
    ]:
        if self.raw is None or self._times is None or self._times.size == 0:
            return None, None, None, None, "EAR-avg_ear not available; jumped to annotation start."

        try:
            channel_index = self.raw.ch_names.index(EAR_AVG_CHANNEL)
        except ValueError:
            return None, None, None, None, "EAR-avg_ear not available; jumped to annotation start."

        data_end = float(self._times[-1])
        start = max(0.0, annotation.onset)
        end = min(annotation.onset + annotation.duration, data_end)
        if end <= start:
            return (
                None,
                None,
                None,
                None,
                "No EAR-avg_ear samples in annotation; jumped to annotation start.",
            )

        start_sample = int(np.searchsorted(self._times, start, side="left"))
        end_sample = int(np.searchsorted(self._times, end, side="right"))
        if end_sample <= start_sample:
            return (
                None,
                None,
                None,
                None,
                "No EAR-avg_ear samples in annotation; jumped to annotation start.",
            )

        data = self.raw.get_data(
            picks=[channel_index], start=start_sample, stop=end_sample, verbose="ERROR"
        )[0]
        times = self.raw.times[start_sample:end_sample]
        finite_mask = np.isfinite(data) & np.isfinite(times)
        if not np.any(finite_mask):
            return (
                None,
                None,
                None,
                None,
                "No EAR-avg_ear samples in annotation; jumped to annotation start.",
            )

        data = data[finite_mask]
        times = times[finite_mask]
        min_index = int(np.argmin(data))
        min_time = float(times[min_index])
        min_value = float(data[min_index])

        if LOGGER.isEnabledFor(logging.DEBUG):
            samples = ", ".join(
                f"{timestamp:.6f}s:{value:.6f}" for timestamp, value in zip(times, data)
            )
            LOGGER.debug(
                "EAR-avg_ear samples in annotation [%.6f, %.6f]: %s; selected min %.6f at %.6f",
                start,
                end,
                samples,
                min_value,
                min_time,
            )

        return min_time, min_value, times, data, None

    def jump_to_next_annotation(self) -> None:
        """Jump the view to the next annotation onset."""

        self._jump_to_annotation(direction="next")

    def jump_to_previous_annotation(self) -> None:
        """Jump the view to the previous annotation onset."""

        self._jump_to_annotation(direction="previous")

    def _jump_to_annotation(self, direction: str) -> None:
        target = self._target_annotation(direction)
        if target is None:
            return

        self._ensure_view_range(target.onset)
        self.annotation_jump_requested.emit(target.onset)
        self.status_label.setText(
            f"Jumped to annotation '{target.description}' at {target.onset:.2f}s."
        )

    def jump_to_next_annotation_minimum(self) -> None:
        """Jump to the next annotation and center on the EAR-avg_ear minimum."""

        target = self._target_annotation(direction="next")
        if target is None:
            return

        min_time, min_value, times, data, message = self._annotation_minimum_time(target)
        if min_time is None:
            self._clear_annotation_samples()
            self._ensure_view_range(target.onset)
            self.annotation_jump_requested.emit(target.onset)
            fallback = message or (
                f"Jumped to annotation '{target.description}' at {target.onset:.2f}s."
            )
            self.status_label.setText(fallback)
            return

        self._ensure_view_range(min_time)
        self.annotation_jump_requested.emit(min_time)
        self._show_annotation_samples(times, data, min_time, min_value)
        self.status_label.setText(
            "Jumped to EAR-avg_ear minimum "
            f"{min_value:.6f} at {min_time:.2f}s in '{target.description}'."
        )

    def _show_annotation_samples(
        self,
        times: Optional[np.ndarray],
        data: Optional[np.ndarray],
        min_time: Optional[float],
        min_value: Optional[float],
    ) -> None:
        self._clear_annotation_samples()
        if (
            times is None
            or data is None
            or times.size == 0
            or data.size == 0
            or min_time is None
            or min_value is None
        ):
            return

        scatter = pg.ScatterPlotItem(
            times,
            data,
            pen=pg.mkPen("#283593"),
            brush=pg.mkBrush("#5c6bc0"),
            size=8,
            symbol="o",
        )
        scatter.setZValue(60)
        min_marker = pg.ScatterPlotItem(
            [min_time],
            [min_value],
            pen=pg.mkPen("#b71c1c", width=2),
            brush=pg.mkBrush("#e53935"),
            size=12,
            symbol="x",
        )
        min_marker.setZValue(70)
        self.plot_widget.addItem(scatter)
        self.plot_widget.addItem(min_marker)
        self._annotation_sample_scatter = scatter
        self._annotation_min_marker = min_marker

    def _clear_annotation_samples(self) -> None:
        for item in (self._annotation_sample_scatter, self._annotation_min_marker):
            if item is not None:
                self.plot_widget.removeItem(item)
        self._annotation_sample_scatter = None
        self._annotation_min_marker = None

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
        for items in self._annotation_items_by_widget.values():
            for item in items:
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

    def _sync_lane_annotations(self) -> None:
        if not self._annotations:
            return
        for widget, items in self._annotation_items_by_widget.items():
            for item in items:
                widget.removeItem(item.region)
            items.clear()
        self._annotation_by_region.clear()
        for annotation in self._annotations:
            self._render_annotation(annotation)
        self._apply_annotation_filter()

    def _register_lane_widget(self, widget: pg.PlotWidget, cursor_line: pg.InfiniteLine) -> None:
        if widget not in self._lane_widgets:
            self._lane_widgets.append(widget)
        self._lane_curves.setdefault(widget, [])
        self._lane_cursor_lines[widget] = cursor_line
        self._annotation_items_by_widget.setdefault(widget, [])

    def _ensure_lane_count(self, lane_count: int) -> None:
        lane_count = max(1, lane_count)
        while len(self._lane_widgets) < lane_count:
            widget = pg.PlotWidget(background="w")
            self._configure_plot_widget(widget)
            widget.setXLink(self.plot_widget)
            cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
            widget.addItem(cursor_line)
            cursor_line.hide()
            self._plot_layout.addWidget(widget)
            self._register_lane_widget(widget, cursor_line)
        while len(self._lane_widgets) > lane_count:
            widget = self._lane_widgets.pop()
            for item in self._annotation_items_by_widget.get(widget, []):
                widget.removeItem(item.region)
            for curve in self._lane_curves.get(widget, []):
                widget.removeItem(curve)
            cursor_line = self._lane_cursor_lines.pop(widget, None)
            if cursor_line is not None:
                widget.removeItem(cursor_line)
            self._annotation_items_by_widget.pop(widget, None)
            self._lane_curves.pop(widget, None)
            self._lane_series.pop(widget, None)
            self._plot_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._update_lane_layout()

    def _add_baseline(self, widget: pg.PlotWidget, channel_name: str, channel_type: str) -> None:
        baseline_value: Optional[float] = None
        baseline_kind = None
        if channel_type in {"eeg", "eog"}:
            baseline_value = 0.0
            baseline_kind = "zero"
        elif channel_name.upper().startswith("EAR-"):
            baseline_value = self._ear_baseline_value
            baseline_kind = "ear"

        if baseline_value is None or baseline_kind is None:
            return

        baseline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("red", width=1))
        baseline.setValue(baseline_value)
        widget.addItem(baseline)
        self._lane_baselines[widget] = baseline
        self._lane_baseline_kind[widget] = baseline_kind

    def _build_annotation_nudge_group(self) -> QWidget:
        nudge_group = QWidget()
        nudge_layout = QVBoxLayout()
        nudge_layout.setContentsMargins(0, 0, 0, 0)
        nudge_layout.setSpacing(4)
        nudge_layout.addWidget(self._build_annotation_nudge_controls())
        nudge_layout.addWidget(self._build_bulk_annotation_nudge_controls())
        nudge_group.setLayout(nudge_layout)
        return nudge_group

    def _build_annotation_nudge_controls(self) -> QWidget:
        nudge_container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Annotation Nudge"))
        layout.addWidget(QLabel("Step (frames):"))
        self.annotation_nudge_spinbox = QSpinBox()
        self.annotation_nudge_spinbox.setRange(1, 1000000)
        self.annotation_nudge_spinbox.setValue(1)
        self.annotation_nudge_spinbox.setSingleStep(1)
        layout.addWidget(self.annotation_nudge_spinbox)
        self.nudge_back_button = QPushButton("− Nudge")
        self.nudge_back_button.clicked.connect(lambda: self._nudge_selected_annotation(-1))
        self.nudge_forward_button = QPushButton("+ Nudge")
        self.nudge_forward_button.clicked.connect(lambda: self._nudge_selected_annotation(1))
        layout.addWidget(self.nudge_back_button)
        layout.addWidget(self.nudge_forward_button)
        nudge_container.setLayout(layout)
        return nudge_container

    def _build_bulk_annotation_nudge_controls(self) -> QWidget:
        nudge_container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Bulk Nudge (All Annotations)"))
        self.bulk_nudge_back_button = QPushButton("− Bulk Nudge")
        self.bulk_nudge_back_button.clicked.connect(lambda: self._nudge_all_annotations(-1))
        self.bulk_nudge_forward_button = QPushButton("+ Bulk Nudge")
        self.bulk_nudge_forward_button.clicked.connect(lambda: self._nudge_all_annotations(1))
        layout.addWidget(self.bulk_nudge_back_button)
        layout.addWidget(self.bulk_nudge_forward_button)
        nudge_container.setLayout(layout)
        return nudge_container

    def _set_selected_annotation(self, annotation: Annotation, announce: bool = True) -> None:
        self._selected_annotation = annotation
        if announce:
            self.status_label.setText(
                f"Selected annotation '{annotation.description}' at {annotation.onset:.2f}s."
            )

    def _nudge_selected_annotation(self, direction: int) -> None:
        if self._selected_annotation is None:
            self.status_label.setText("Select an annotation to nudge.")
            return
        if direction == 0:
            return
        step_frames = self.annotation_nudge_spinbox.value()
        delta_seconds = (step_frames / ANNOTATION_NUDGE_FPS) * (1 if direction > 0 else -1)
        annotation = self._selected_annotation
        duration = annotation.duration
        new_onset = annotation.onset + delta_seconds
        if self._times is not None and self._times.size > 0:
            data_end = float(self._times[-1])
            max_onset = max(0.0, data_end - duration)
            new_onset = max(0.0, min(new_onset, max_onset))
        else:
            new_onset = max(0.0, new_onset)
        annotation.onset = new_onset
        self._sync_annotation_regions(annotation, new_onset, duration)
        self._set_annotations_dirty(True)
        self.status_label.setText(
            f"Nudged annotation '{annotation.description}' to {annotation.onset:.2f}s."
        )

    def _nudge_all_annotations(self, direction: int) -> None:
        if not self._annotations:
            self.status_label.setText("No annotations available to nudge.")
            return
        if direction == 0:
            return
        step_frames = self.annotation_nudge_spinbox.value()
        delta_seconds = (step_frames / ANNOTATION_NUDGE_FPS) * (1 if direction > 0 else -1)
        data_end = None
        if self._times is not None and self._times.size > 0:
            data_end = float(self._times[-1])

        for annotation in self._annotations:
            duration = annotation.duration
            new_onset = annotation.onset + delta_seconds
            if data_end is not None:
                max_onset = max(0.0, data_end - duration)
                new_onset = max(0.0, min(new_onset, max_onset))
            else:
                new_onset = max(0.0, new_onset)
            annotation.onset = new_onset
            self._sync_annotation_regions(annotation, new_onset, duration)

        self._set_annotations_dirty(True)
        direction_label = "later" if direction > 0 else "earlier"
        self.status_label.setText(
            f"Nudged all annotations {direction_label} by {step_frames} frame(s)."
        )

    def _update_lane_layout(self) -> None:
        for idx, widget in enumerate(self._lane_widgets):
            self._plot_layout.setStretch(idx, 1)
            widget.setMinimumHeight(MIN_LANE_HEIGHT)
        total_height = len(self._lane_widgets) * MIN_LANE_HEIGHT
        if len(self._lane_widgets) > 1:
            total_height += self._plot_layout.spacing() * (len(self._lane_widgets) - 1)
        self._plot_container.setMinimumHeight(total_height)

    def _update_lane_scales(self, x_min: float, x_max: float) -> None:
        for widget, series in self._lane_series.items():
            times, channel = series
            if times.size == 0:
                continue
            start = int(np.searchsorted(times, x_min, side="left"))
            end = int(np.searchsorted(times, x_max, side="right"))
            window = channel[start:end]
            if window.size == 0:
                window = channel
            y_min, y_max = self._robust_range(window)
            widget.setYRange(y_min, y_max, padding=0)

    def _robust_range(self, data: np.ndarray) -> tuple[float, float]:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return -1.0, 1.0
        lower, upper = np.nanpercentile(finite, [1, 99])
        if not math.isfinite(lower) or not math.isfinite(upper):
            lower = float(np.nanmin(finite))
            upper = float(np.nanmax(finite))
        if lower == upper:
            span = max(abs(lower), 1.0)
            return lower - span, upper + span
        padding = (upper - lower) * LANE_PADDING_RATIO
        return lower - padding, upper + padding
