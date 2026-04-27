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
from typing import Any, List, Optional, Set

import mne
import numpy as np
import pyqtgraph as pg
from mne.io.constants import FIFF
from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QApplication,
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

from paths import derive_annotation_path, derive_time_series_path

DEFAULT_PRIMARY_CHANNEL = "EEG-E8"
EOG_REPAIR_CHANNEL = "EOG-EEG-eog_vert_left"
EOG_REPAIR_LEFT_CHANNEL = "eog_vert_left"
EAR_AVG_CHANNEL = "EAR-avg_ear"
TARGET_PLOT_UNIT = "µV"
DEFAULT_REPAIR_THRESHOLD_UNIT = "µV"
REPAIR_THRESHOLD_UNITS = ("V", "mV", "µV", "kµV", "nV")
UNIT_SCALE_FACTORS = {
    "v": 1e6,
    "mv": 1e3,
    "kµv": 1e3,
    "kuv": 1e3,
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


@dataclass(eq=False)
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
    ui_setting_changed = pyqtSignal(str, object)
    FILTER_ALL = "__all__"
    FILTER_NONE = "__none__"

    def __init__(
        self,
        max_points: int = 10000,
        parent: Optional[QWidget] = None,
        time_series_root: Optional[Path] = None,
        annotation_root: Optional[Path] = None,
        ui_settings: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.max_points = max_points
        self.raw: Optional[mne.io.BaseRaw] = None
        self._times: Optional[np.ndarray] = None
        self._selected_channels: Set[str] = set()
        self._primary_channel = DEFAULT_PRIMARY_CHANNEL
        self._repair_threshold = (
            float(ui_settings.get("auto_repair_threshold", 0.0)) if ui_settings else 0.0
        )
        self._repair_threshold_unit = self._coerce_repair_threshold_unit(
            str(ui_settings.get("auto_repair_threshold_unit", DEFAULT_REPAIR_THRESHOLD_UNIT))
            if ui_settings
            else DEFAULT_REPAIR_THRESHOLD_UNIT
        )
        configured_repair_channel = (
            str(ui_settings.get("auto_repair_channel", "")) if ui_settings else ""
        )
        if configured_repair_channel:
            self._primary_channel = configured_repair_channel
        self._last_cursor_time: float = 0.0
        self.default_view_span_seconds: float = 5.0
        self.view_span_seconds: float = self.default_view_span_seconds
        self.min_span_seconds: float = 0.1
        self._last_ts_path: Optional[Path] = None
        self._last_video_path: Optional[Path] = None
        self._last_annotation_path: Optional[Path] = None
        self._expected_ts_path: Optional[Path] = None
        self._expected_annotation_path: Optional[Path] = None
        self._direct_annotation_path: Optional[Path] = None
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
        self._last_create_undo: Optional[callable] = None
        self._auto_repair_original_bounds: dict[int, tuple[float, float]] = {}
        self._auto_repair_notes: dict[int, str] = {}
        self._ear_gain_enabled = False
        self._ear_gain_value = 1.0
        self._ear_baseline_value = (
            float(ui_settings.get("ear_baseline", 0.0)) if ui_settings else 0.0
        )
        self._ear_repair_nth_peak = (
            int(ui_settings.get("ear_repair_nth_peak", 1)) if ui_settings else 1
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
        self.channel_list.currentItemChanged.connect(
            self._on_current_channel_item_changed
        )

        self.zoom_out_button = QPushButton("Zoom -")
        self.zoom_out_button.clicked.connect(lambda: self._adjust_zoom(1.25))
        self.zoom_in_button = QPushButton("Zoom +")
        self.zoom_in_button.clicked.connect(lambda: self._adjust_zoom(0.8))
        self.zoom_reset_button = QPushButton("Reset Zoom")
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        self.zoom_label = QLabel(self._zoom_label_text())
        self.primary_channel_combo = QComboBox()
        self.primary_channel_combo.currentIndexChanged.connect(
            self._on_primary_channel_changed
        )

        control_row.addWidget(self.show_all_checkbox)
        control_row.addWidget(QLabel("Highlighted channel:"))
        control_row.addWidget(self.primary_channel_combo)
        control_row.addWidget(self.zoom_out_button)
        control_row.addWidget(self.zoom_in_button)
        control_row.addWidget(self.zoom_reset_button)
        control_row.addWidget(self.zoom_label)
        control_row.addStretch()

        control_layout.addLayout(control_row)
        control_layout.addWidget(self.channel_list)
        repair_row = QHBoxLayout()
        repair_row.addWidget(QLabel("EOG repair threshold:"))
        self.auto_repair_threshold_spinbox = QDoubleSpinBox()
        self.auto_repair_threshold_spinbox.setRange(-1.0e9, 1.0e9)
        self.auto_repair_threshold_spinbox.setDecimals(3)
        self.auto_repair_threshold_spinbox.setSingleStep(0.1)
        self.auto_repair_threshold_spinbox.setValue(self._repair_threshold)
        self.auto_repair_threshold_spinbox.valueChanged.connect(
            self._on_auto_repair_threshold_changed
        )
        self.auto_repair_threshold_unit_combo = QComboBox()
        self.auto_repair_threshold_unit_combo.addItems(REPAIR_THRESHOLD_UNITS)
        self.auto_repair_threshold_unit_combo.setCurrentText(self._repair_threshold_unit)
        self.auto_repair_threshold_unit_combo.currentTextChanged.connect(
            self._on_auto_repair_threshold_unit_changed
        )
        repair_row.addWidget(self.auto_repair_threshold_spinbox)
        repair_row.addWidget(self.auto_repair_threshold_unit_combo)
        repair_row.addStretch()
        control_layout.addLayout(repair_row)
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
        ear_repair_row = QHBoxLayout()
        ear_repair_row.addWidget(QLabel("EAR repair peak number (1=first, 2=second…):"))
        self.ear_repair_nth_peak_spinbox = QSpinBox()
        self.ear_repair_nth_peak_spinbox.setRange(1, 20)
        self.ear_repair_nth_peak_spinbox.setValue(self._ear_repair_nth_peak)
        self.ear_repair_nth_peak_spinbox.setToolTip(
            "Which local peak to use as the repair boundary when walking outward from "
            "the trough. 1 = first peak found, 2 = second peak, 3 = third peak, etc. "
            "Increase to skip small intermediate peaks and land on a larger, more "
            "prominent peak further from the trough. Default: 1."
        )
        self.ear_repair_nth_peak_spinbox.valueChanged.connect(
            self._on_ear_repair_nth_peak_changed
        )
        ear_repair_row.addWidget(self.ear_repair_nth_peak_spinbox)
        ear_repair_row.addStretch()
        control_layout.addLayout(ear_repair_row)
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
        self.bulk_auto_repair_button = QPushButton("Bulk auto_repair")
        self.bulk_auto_repair_button.clicked.connect(self.bulk_auto_repair_annotations)
        left_layout.addWidget(self.bulk_auto_repair_button)
        self.revert_all_auto_repair_button = QPushButton("Revert all auto_repair")
        self.revert_all_auto_repair_button.clicked.connect(self.revert_all_auto_repair)
        left_layout.addWidget(self.revert_all_auto_repair_button)
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
        self.annotation_count_label = QLabel("")
        self.annotation_count_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.annotation_count_label)
        center_controls.setLayout(center_layout)

        right_controls = QWidget()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addStretch()
        right_layout.addWidget(self._build_annotation_nudge_group())
        right_layout.addWidget(
            QLabel(
                "Shortcuts: [ or B = back, ] or N = next (EAR min), "
                "Space = auto_repair, R = auto_repair_eog, E = auto_repair_ear, "
                "Shift+←/→ = nudge annotation, Ctrl+C = create from EAR"
            )
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
        self._update_annotation_count_label()

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

    def has_unsaved_annotation_changes(self) -> bool:
        """Return whether annotations have unsaved in-memory edits."""

        return self._annotations_dirty

    def current_cursor_time(self) -> float:
        """Return the currently centered cursor time in seconds."""

        return self._last_cursor_time

    def signal_duration_seconds(self) -> Optional[float]:
        """Return the loaded signal duration in seconds."""

        if self._times is None or self._times.size == 0:
            return None
        return float(self._times[-1])

    def seek_time(self, seconds: float) -> None:
        """Move the time-series view to the requested time in seconds."""

        self.update_cursor_time(seconds)

    def load_for_video(self, video_path: Optional[Path]) -> None:
        """Load and plot the time series associated with the provided video."""

        self._reset_loaded_data()

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
        self._load_time_series_paths(ts_path, csv_path, source_video=video_path)

    def load_time_series_file(
        self, time_series_path: Optional[Path], annotation_path: Optional[Path] = None
    ) -> None:
        """Load a standalone EDF/FIF time series with an explicit CSV annotation path."""

        self._reset_loaded_data()
        if time_series_path is None:
            self.status_label.setText("Select a recording to view its time series.")
            self.cursor_line.hide()
            return

        csv_path = annotation_path or time_series_path.with_suffix(".csv")
        self._load_time_series_paths(time_series_path, csv_path)

    def load_annotation_file(self, annotation_path: Path) -> bool:
        """Replace annotations for the current signal with an explicit CSV file."""

        if self.raw is None or self._times is None or self._times.size == 0:
            self.status_label.setText("Load a time series before choosing a CSV file.")
            return False

        if annotation_path.suffix.lower() != ".csv":
            self.status_label.setText(f"Annotation file must be a CSV: {annotation_path}")
            return False

        if not annotation_path.exists():
            self.status_label.setText(f"Annotation CSV not found at {annotation_path}.")
            return False

        self._clear_annotations()
        self._annotations = []
        self._last_annotation_path = annotation_path
        self._direct_annotation_path = annotation_path
        self._update_annotation_filter_options(force_all=True)
        self._add_annotations_for_path(annotation_path)
        self._set_annotations_dirty(False)
        self._ensure_view_range(self._last_cursor_time)
        self.status_label.setText(f"Loaded annotations from {annotation_path}.")
        return True

    def _reset_loaded_data(self) -> None:
        self._clear_plot()
        self._clear_annotations()
        self._times = None
        self.raw = None
        self._selected_channels.clear()
        self._primary_channel = DEFAULT_PRIMARY_CHANNEL
        self.primary_channel_combo.blockSignals(True)
        self.primary_channel_combo.clear()
        self.primary_channel_combo.blockSignals(False)
        self.channel_list.clear()
        self._reset_zoom()
        self._last_ts_path = None
        self._last_video_path = None
        self._last_annotation_path = None
        self._expected_ts_path = None
        self._expected_annotation_path = None
        self._direct_annotation_path = None
        self._annotations = []
        self._set_annotations_dirty(False)
        self._update_annotation_filter_options(force_all=True)
        self._update_annotation_count_label()
        self._selected_annotation = None
        self._auto_repair_original_bounds.clear()
        self._auto_repair_notes.clear()

    def _load_time_series_paths(
        self,
        ts_path: Path,
        csv_path: Path,
        source_video: Optional[Path] = None,
    ) -> None:
        self._expected_ts_path = ts_path
        self._expected_annotation_path = csv_path
        self._direct_annotation_path = csv_path if source_video is None else None
        if not ts_path.exists():
            self.status_label.setText(
                f"Time series file not found at {ts_path}. "
                "Update your configuration or verify the dataset root."
            )
            self.cursor_line.hide()
            return

        self.status_label.setText(f"Loading time series from {ts_path}...")
        self._last_ts_path = ts_path
        self._last_video_path = source_video
        self._last_annotation_path = csv_path
        try:
            self.raw = self._read_raw_time_series(ts_path)
        except Exception as exc:
            self.raw = None
            self._last_ts_path = None
            self.status_label.setText(f"Failed to load time series from {ts_path}: {exc}")
            self.cursor_line.hide()
            return
        self._times = self.raw.times
        self._populate_channel_list()
        self._plot_data()
        self._add_annotations_for_path(csv_path)
        self._ensure_view_range(0.0)
        self._update_annotation_count_label()

    def _read_raw_time_series(self, ts_path: Path) -> mne.io.BaseRaw:
        suffix = ts_path.suffix.lower()
        if suffix == ".fif":
            return mne.io.read_raw_fif(str(ts_path), preload=True, verbose="ERROR")
        if suffix == ".edf":
            return mne.io.read_raw_edf(str(ts_path), preload=True, verbose="ERROR")
        raise ValueError(f"Unsupported time series format: {ts_path.suffix}")

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
        if all(self._is_plot_channel(name) for name in channel_names):
            return data, picks, channel_names, channel_types

        if (
            self._primary_channel not in channel_names
            or EAR_AVG_CHANNEL not in channel_names
        ):
            return data, picks, channel_names, channel_types

        indices = list(range(len(channel_names)))
        eeg_index = channel_names.index(self._primary_channel)
        ear_index = channel_names.index(EAR_AVG_CHANNEL)

        for idx in sorted({eeg_index, ear_index}, reverse=True):
            indices.pop(idx)

        indices = [ear_index] + indices + [eeg_index]

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
        normalized = self._normalize_unit_key(unit)
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

    def _on_ear_repair_nth_peak_changed(self, value: int) -> None:
        self._ear_repair_nth_peak = int(value)
        self.ui_setting_changed.emit("ear_repair_nth_peak", self._ear_repair_nth_peak)

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
        self.primary_channel_combo.blockSignals(True)
        self.show_all_checkbox.setChecked(False)
        self.channel_list.clear()
        self._populate_primary_channel_combo()

        channel_indices = self._channel_indices()
        has_known_channels = any(
            self._is_always_visible_channel(self.raw.ch_names[i]) for i in channel_indices
        )

        defaults_present: Set[str] = set()
        for index in channel_indices:
            name = self.raw.ch_names[index]
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            is_default = self._is_always_visible_channel(name) if has_known_channels else True
            if is_default:
                defaults_present.add(name)
            item.setCheckState(Qt.Checked if is_default else Qt.Unchecked)
            self.channel_list.addItem(item)

        self.channel_list.blockSignals(False)
        self.show_all_checkbox.blockSignals(False)
        self.primary_channel_combo.blockSignals(False)
        self._selected_channels = defaults_present

    def _populate_primary_channel_combo(self) -> None:
        if self.raw is None:
            return

        self.primary_channel_combo.clear()
        available_channels = [
            self.raw.ch_names[index] for index in self._channel_indices()
        ]
        if not available_channels:
            return

        selected_channel = (
            self._primary_channel
            if self._primary_channel in available_channels
            else DEFAULT_PRIMARY_CHANNEL
            if DEFAULT_PRIMARY_CHANNEL in available_channels
            else available_channels[0]
        )
        self._primary_channel = selected_channel

        self.primary_channel_combo.addItems(available_channels)
        self.primary_channel_combo.setCurrentText(selected_channel)

    def _channel_indices(self) -> List[int]:
        if self.raw is None:
            return []

        required = self._required_plot_channel_indices()
        has_known = any(
            index is not None
            and self._is_always_visible_channel(self.raw.ch_names[index])
            for index in required
            if index is not None
        )

        indices: List[int] = []
        for index in required:
            if index is None or index in indices:
                continue
            # In fallback mode, filter by _selected_channels so unchecking works
            if not has_known and self._selected_channels:
                name = self.raw.ch_names[index]
                if name not in self._selected_channels:
                    continue
            indices.append(index)
        return indices

    def _required_plot_channel_indices(self) -> list[Optional[int]]:
        if self.raw is None:
            return []

        specific: list[Optional[int]] = [
            self._first_channel_index_matching(self._is_ear_plot_channel),
            self._first_channel_index_matching(self._is_eog_left_plot_channel),
            self._first_channel_index_matching(self._is_default_eeg_plot_channel),
        ]
        if any(i is not None for i in specific):
            return specific
        # No known channel patterns found — fall back to all channels in the file
        return list(range(len(self.raw.ch_names)))

    def _first_channel_index_matching(self, matcher) -> Optional[int]:
        if self.raw is None:
            return None

        for index, name in enumerate(self.raw.ch_names):
            if matcher(name):
                return index
        return None

    def _on_show_all_channels(self, state: int) -> None:
        if state == Qt.Checked:
            self._selected_channels.clear()
        else:
            self._selected_channels = self._checked_channel_names()
        self._replot()

    def _on_channel_item_changed(self, item: QListWidgetItem) -> None:
        name = item.text()
        if self._is_always_visible_channel(name) and item.checkState() != Qt.Checked:
            self._set_channel_checked(name, True)
            self._selected_channels.add(name)
            return
        if item.checkState() == Qt.Checked:
            self._selected_channels.add(name)
            self._set_repair_channel(name)
        else:
            self._selected_channels.discard(name)
        if not self.show_all_checkbox.isChecked():
            self._replot()

    @classmethod
    def _is_always_visible_channel(cls, channel_name: str) -> bool:
        return cls._is_plot_channel(channel_name)

    @classmethod
    def _is_plot_channel(cls, channel_name: str) -> bool:
        return (
            cls._is_ear_plot_channel(channel_name)
            or cls._is_eog_left_plot_channel(channel_name)
            or cls._is_default_eeg_plot_channel(channel_name)
        )

    @classmethod
    def _is_ear_plot_channel(cls, channel_name: str) -> bool:
        return (
            channel_name.casefold() == "ear"
            or cls._channel_name_matches_token(channel_name, EAR_AVG_CHANNEL)
        )

    @classmethod
    def _is_eog_left_plot_channel(cls, channel_name: str) -> bool:
        return cls._channel_name_matches_token(channel_name, EOG_REPAIR_LEFT_CHANNEL)

    @classmethod
    def _is_default_eeg_plot_channel(cls, channel_name: str) -> bool:
        return cls._channel_name_matches_token(channel_name, DEFAULT_PRIMARY_CHANNEL)

    @staticmethod
    def _channel_name_matches_token(channel_name: str, token: str) -> bool:
        normalized_name = channel_name.casefold()
        normalized_token = token.casefold()
        return (
            normalized_name == normalized_token
            or normalized_name.endswith(normalized_token)
        )

    def _on_current_channel_item_changed(
        self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        self._set_repair_channel(current.text())

    def _on_primary_channel_changed(self, index: int) -> None:
        if self.raw is None or index < 0:
            return

        previous_channel = self._primary_channel
        selected_channel = self.primary_channel_combo.currentText()
        if not selected_channel or selected_channel == previous_channel:
            return

        self._primary_channel = selected_channel
        self.ui_setting_changed.emit("auto_repair_channel", selected_channel)
        if (
            previous_channel in self._selected_channels
            and not self._is_always_visible_channel(previous_channel)
        ):
            self._selected_channels.discard(previous_channel)
            self._set_channel_checked(previous_channel, False)
        self._selected_channels.add(selected_channel)
        self._set_channel_checked(selected_channel, True)
        if not self.show_all_checkbox.isChecked():
            self._replot()

    def _set_repair_channel(self, channel_name: str) -> None:
        if self.raw is None or channel_name == self._primary_channel:
            return
        if channel_name not in self.raw.ch_names:
            return
        self.primary_channel_combo.setCurrentText(channel_name)

    def _on_auto_repair_threshold_changed(self, value: float) -> None:
        self._repair_threshold = float(value)
        self.ui_setting_changed.emit("auto_repair_threshold", self._repair_threshold)

    def _on_auto_repair_threshold_unit_changed(self, unit: str) -> None:
        self._repair_threshold_unit = self._coerce_repair_threshold_unit(unit)
        self.ui_setting_changed.emit(
            "auto_repair_threshold_unit", self._repair_threshold_unit
        )

    @classmethod
    def _coerce_repair_threshold_unit(cls, unit: str) -> str:
        normalized = cls._normalize_unit_key(unit)
        for label in REPAIR_THRESHOLD_UNITS:
            if cls._normalize_unit_key(label) == normalized:
                return label
        return DEFAULT_REPAIR_THRESHOLD_UNIT

    @staticmethod
    def _normalize_unit_key(unit: str) -> str:
        return unit.strip().lower().replace("μ", "µ").replace(" ", "")

    def _set_channel_checked(self, channel_name: str, checked: bool) -> None:
        self.channel_list.blockSignals(True)
        try:
            for index in range(self.channel_list.count()):
                item = self.channel_list.item(index)
                if item.text() == channel_name:
                    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    return
        finally:
            self.channel_list.blockSignals(False)

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
        self._update_annotation_count_label()

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
        self._auto_repair_original_bounds.clear()
        self._auto_repair_notes.clear()
        self._update_annotation_count_label()

    def _annotation_count_text(self) -> str:
        total = len(self._annotations)
        if total == 0:
            return "Blinks: 0"

        current_time = self._last_cursor_time
        previous_or_current = sum(
            1 for annotation in self._annotations if annotation.onset <= current_time
        )
        ahead = total - previous_or_current
        return (
            f"Blinks: {total} total | {previous_or_current} at/before "
            f"{current_time:.2f}s | {ahead} ahead"
        )

    def _update_annotation_count_label(self) -> None:
        if hasattr(self, "annotation_count_label"):
            self.annotation_count_label.setText(self._annotation_count_text())

    def _color_for_description(self, description: str) -> pg.Color:
        if description == "BAD_SIGNAL_BLINK":
            return pg.mkColor("#8b0000")
        if description not in self._annotation_colors:
            palette_index = len(self._annotation_colors) % len(ANNOTATION_PALETTE)
            self._annotation_colors[description] = pg.mkColor(ANNOTATION_PALETTE[palette_index])
        return self._annotation_colors[description]

    def _pen_for_channel(self, channel_name: str, index: int) -> pg.Pen:
        if channel_name == self._primary_channel:
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
        if obj in self._plot_viewports() and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and not (event.modifiers() & Qt.ControlModifier):
                widget = self._widget_for_viewport(obj)
                t = self._time_at_position(event.pos(), widget)
                if t is not None and self._times is not None and self._times.size > 0:
                    self._move_cursor_to(t)

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

    def _move_cursor_to(self, time: float) -> None:
        """Update the cursor line position without re-centering the view."""
        if self._times is None or self._times.size == 0:
            return
        clamped = max(0.0, min(time, float(self._times[-1])))
        self._last_cursor_time = clamped
        for cursor_line in self._lane_cursor_lines.values():
            cursor_line.setPos(clamped)
            cursor_line.show()

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
        self._update_annotation_count_label()

    def _annotation_csv_path(self) -> Optional[Path]:
        if self._direct_annotation_path is not None:
            return self._direct_annotation_path
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
        self._auto_repair_original_bounds.clear()
        self._auto_repair_notes.clear()
        for annotation in self._annotations:
            self._refresh_annotation_visuals(annotation)
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
        auto_repair_action = menu.addAction(f"auto_repair ({DEFAULT_PRIMARY_CHANNEL})")
        auto_repair_eog_action = menu.addAction(
            f"auto_repair_eog ({EOG_REPAIR_CHANNEL})"
        )
        auto_repair_ear_action = menu.addAction(
            f"auto_repair_ear ({EAR_AVG_CHANNEL})"
        )
        revert_auto_repair_action = menu.addAction("Revert auto_repair")
        revert_auto_repair_action.setEnabled(
            id(annotation_item.annotation) in self._auto_repair_original_bounds
        )
        delete_action = menu.addAction("Delete annotation")
        selected_action = menu.exec_(self.plot_widget.viewport().mapToGlobal(pos))
        if selected_action == edit_action:
            self._edit_annotation_description(annotation_item)
        elif selected_action == auto_repair_action:
            self._auto_repair_annotation(annotation_item)
        elif selected_action == auto_repair_eog_action:
            self._auto_repair_annotation_eog(annotation_item.annotation)
        elif selected_action == auto_repair_ear_action:
            self._auto_repair_annotation_ear(annotation_item.annotation)
        elif selected_action == revert_auto_repair_action:
            self._revert_auto_repair(annotation_item.annotation)
        elif selected_action == delete_action:
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
        region.setToolTip(self._annotation_tooltip(annotation))
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
                    if item.annotation is not annotation or item.region is skip_region:
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

    def _auto_repair_annotation(self, annotation_item: AnnotationItem) -> None:
        self._auto_repair_annotation_model(
            annotation_item.annotation,
            channel_name=DEFAULT_PRIMARY_CHANNEL,
            threshold=0.0,
            mode_label="auto_repair",
        )

    def auto_repair_selected_annotation(self) -> None:
        """Repair the currently selected annotation using the selected repair channel."""

        if self._selected_annotation is None:
            self.status_label.setText("Select an annotation before using auto_repair.")
            return
        self._auto_repair_annotation_model(
            self._selected_annotation,
            channel_name=DEFAULT_PRIMARY_CHANNEL,
            threshold=0.0,
            mode_label="auto_repair",
        )

    def auto_repair_selected_annotation_eog(self) -> None:
        """Repair the selected annotation using the selected repair channel."""

        if self._selected_annotation is None:
            self.status_label.setText("Select an annotation before using auto_repair_eog.")
            return
        self._auto_repair_annotation_eog(self._selected_annotation)

    def _auto_repair_annotation_eog(self, annotation: Annotation) -> None:
        result = self._auto_repair_eog_bounds(annotation)
        if result is None:
            return
        self._apply_auto_repair_result(
            annotation, result, channel_name=EOG_REPAIR_CHANNEL, mode_label="auto_repair_eog"
        )

    def auto_repair_selected_annotation_ear(self) -> None:
        """Repair the selected annotation using EAR-avg_ear minimum-to-peak walking."""

        if self._selected_annotation is None:
            self.status_label.setText("Select an annotation before using auto_repair_ear.")
            return
        self._auto_repair_annotation_ear(self._selected_annotation)

    def auto_repair_selected_annotation_ear_with_peak(self, nth_peak: int) -> None:
        """Repair the selected annotation using EAR-avg_ear with a specific nth_peak."""

        if self._selected_annotation is None:
            self.status_label.setText("Select an annotation before using auto_repair_ear.")
            return
        result = self._auto_repair_ear_bounds(self._selected_annotation, nth_peak=nth_peak)
        if result is None:
            return
        self._apply_auto_repair_result(
            self._selected_annotation, result, channel_name=EAR_AVG_CHANNEL, mode_label="auto_repair_ear"
        )

    def auto_repair_selected_annotation_eog_with_peak(self, nth_peak: int) -> None:
        """Repair the selected annotation using EOG (nth_peak reserved for future use)."""

        self.auto_repair_selected_annotation_eog()

    def _auto_repair_annotation_ear(self, annotation: Annotation) -> None:
        result = self._auto_repair_ear_bounds(annotation)
        if result is None:
            return
        self._apply_auto_repair_result(
            annotation, result, channel_name=EAR_AVG_CHANNEL, mode_label="auto_repair_ear"
        )

    def _auto_repair_ear_bounds(
        self, annotation: Annotation, nth_peak: Optional[int] = None
    ) -> Optional[tuple[float, float, float, bool]]:
        if self.raw is None or self._times is None or self._times.size == 0:
            self.status_label.setText("No time series loaded for auto_repair_ear.")
            return None

        ear_index = self._channel_index_by_name(EAR_AVG_CHANNEL)
        if ear_index is None:
            self.status_label.setText(
                f"{EAR_AVG_CHANNEL} not available for auto_repair_ear."
            )
            return None

        channel_data = self._repair_channel_data(ear_index)
        channel_times = self.raw.times
        if channel_data.size != channel_times.size:
            self.status_label.setText("EAR sample/time mismatch; auto_repair_ear skipped.")
            return None

        data_end = float(self._times[-1])
        start = max(0.0, annotation.onset)
        end = min(annotation.onset + annotation.duration, data_end)
        if end <= start:
            self.status_label.setText("Annotation is empty; auto_repair_ear skipped.")
            return None

        repaired = self._repair_ear_bounds_from_trough_peak(
            channel_times,
            channel_data,
            annotation_onset=start,
            annotation_duration=end - start,
            nth_peak=nth_peak if nth_peak is not None else self._ear_repair_nth_peak,
        )
        if repaired is None:
            self.status_label.setText(
                f"No finite minimum found in the annotation window for auto_repair_ear."
            )
            return None

        repaired_start, repaired_end, event_time = repaired
        duration = max(0.0, repaired_end - repaired_start)
        return repaired_start, duration, event_time, False

    def _auto_repair_annotation_model(
        self,
        annotation: Annotation,
        channel_name: str,
        threshold: float,
        mode_label: str,
        threshold_unit: str = TARGET_PLOT_UNIT,
    ) -> None:
        result = self._auto_repair_bounds(
            annotation,
            channel_name=channel_name,
            threshold=threshold,
            threshold_unit=threshold_unit,
        )
        if result is None:
            return

        self._apply_auto_repair_result(
            annotation, result, channel_name=channel_name, mode_label=mode_label
        )

    def _apply_auto_repair_result(
        self,
        annotation: Annotation,
        result: tuple[float, float, float, bool],
        channel_name: str,
        mode_label: str,
    ) -> None:
        pre_onset = annotation.onset
        pre_duration = annotation.duration
        self._last_create_undo = self._make_undo_annotation_repair(
            annotation, pre_onset, pre_duration
        )
        self._store_auto_repair_original_bounds(annotation)
        onset, duration, peak_time, used_split_boundary = result
        unchanged = math.isclose(annotation.onset, onset) and math.isclose(
            annotation.duration, duration
        )
        annotation.onset = onset
        annotation.duration = duration
        self._sync_annotation_regions(annotation, onset, duration)
        self._set_selected_annotation(annotation, announce=False)
        self._set_annotations_dirty(True)
        self._mark_auto_repaired(annotation, used_split_boundary=used_split_boundary)
        if unchanged:
            self.status_label.setText(
                f"{mode_label} checked annotation on {channel_name}; bounds were "
                f"already aligned around event {peak_time:.3f}s."
            )
        else:
            self.status_label.setText(
                f"{mode_label} repaired annotation on {channel_name} "
                f"to {onset:.3f}s-{onset + duration:.3f}s around event {peak_time:.3f}s."
            )

    def bulk_auto_repair_annotations(self) -> None:
        """Repair every annotation in memory using EOG peak-trough without saving the CSV."""

        if not self._annotations:
            self.status_label.setText("No annotations available for bulk auto_repair.")
            return

        total_count = len(self._annotations)
        self.bulk_auto_repair_button.setEnabled(False)
        self.bulk_auto_repair_button.setText("Repairing...")
        self.status_label.setText(f"Bulk auto_repair running: 0/{total_count} checked.")
        QApplication.processEvents()

        repaired_count = 0
        failed_count = 0
        unchanged_count = 0
        last_repaired: Optional[Annotation] = None
        try:
            for checked_count, annotation in enumerate(list(self._annotations), start=1):
                result = self._auto_repair_eog_bounds(annotation)
                if result is None:
                    failed_count += 1
                else:
                    self._store_auto_repair_original_bounds(annotation)
                    onset, duration, _peak_time, used_split_boundary = result
                    unchanged = math.isclose(annotation.onset, onset) and math.isclose(
                        annotation.duration, duration
                    )
                    annotation.onset = onset
                    annotation.duration = duration
                    self._sync_annotation_regions(annotation, onset, duration)
                    self._mark_auto_repaired(
                        annotation, used_split_boundary=used_split_boundary
                    )
                    if unchanged:
                        unchanged_count += 1
                    else:
                        repaired_count += 1
                    last_repaired = annotation

                if checked_count % 10 == 0 or checked_count == total_count:
                    self.status_label.setText(
                        "Bulk auto_repair running: "
                        f"{checked_count}/{total_count} checked, "
                        f"{repaired_count} changed, {unchanged_count} unchanged, "
                        f"{failed_count} failed."
                    )
                    QApplication.processEvents()
        finally:
            self.bulk_auto_repair_button.setEnabled(True)
            self.bulk_auto_repair_button.setText("Bulk auto_repair")

        overlap_count = self._mark_auto_repair_overlaps()
        if repaired_count == 0 and unchanged_count == 0:
            self.status_label.setText(
                f"Bulk auto_repair could not repair any annotations ({failed_count} failed)."
            )
            return

        if last_repaired is not None:
            self._set_selected_annotation(last_repaired, announce=False)
        self._set_annotations_dirty(True)
        self.status_label.setText(
            "Bulk auto_repair finished without saving: "
            f"{repaired_count} changed, {unchanged_count} already aligned, "
            f"{failed_count} failed. "
            f"0 deleted, 0 merged, {overlap_count} overlap-review pair(s). "
            "Annotations were kept separate. "
            "Right-click an annotation and choose Revert auto_repair to restore it."
        )

    def _store_auto_repair_original_bounds(self, annotation: Annotation) -> None:
        self._auto_repair_original_bounds.setdefault(
            id(annotation),
            (annotation.onset, annotation.duration),
        )

    def _revert_auto_repair(self, annotation: Annotation) -> None:
        original = self._auto_repair_original_bounds.pop(id(annotation), None)
        if original is None:
            self.status_label.setText("No auto_repair history for this annotation.")
            return

        onset, duration = original
        annotation.onset = onset
        annotation.duration = duration
        self._auto_repair_notes.pop(id(annotation), None)
        self._sync_annotation_regions(annotation, onset, duration)
        self._refresh_annotation_visuals(annotation)
        self._set_selected_annotation(annotation, announce=False)
        self._set_annotations_dirty(True)
        self.status_label.setText(
            f"Reverted annotation to {onset:.3f}s-{onset + duration:.3f}s."
        )

    def revert_all_auto_repair(self) -> None:
        """Restore all annotations that have auto-repair history."""

        if not self._auto_repair_original_bounds:
            self.status_label.setText("No auto_repair history to revert.")
            return

        reverted_count = 0
        for annotation in list(self._annotations):
            original = self._auto_repair_original_bounds.pop(id(annotation), None)
            if original is None:
                continue
            onset, duration = original
            annotation.onset = onset
            annotation.duration = duration
            self._auto_repair_notes.pop(id(annotation), None)
            self._sync_annotation_regions(annotation, onset, duration)
            self._refresh_annotation_visuals(annotation)
            reverted_count += 1

        self._set_annotations_dirty(True)
        self.status_label.setText(
            f"Reverted {reverted_count} auto_repaired annotation(s) to original bounds."
        )

    def _mark_auto_repaired(
        self, annotation: Annotation, used_split_boundary: bool = False
    ) -> None:
        note = "AUTO_REPAIRED - revert available"
        if used_split_boundary:
            note = f"{note}; SPLIT boundary used"
        self._auto_repair_notes[id(annotation)] = note
        self._refresh_annotation_visuals(annotation)

    def _mark_auto_repair_overlaps(self) -> int:
        for annotation_id, note in list(self._auto_repair_notes.items()):
            if "OVERLAPS" in note:
                self._auto_repair_notes[annotation_id] = note.replace("; OVERLAPS after repair", "")

        overlap_annotations: set[int] = set()
        overlap_count = 0
        annotations = sorted(self._annotations, key=lambda item: item.onset)
        for index, annotation in enumerate(annotations):
            annotation_end = annotation.onset + annotation.duration
            for other in annotations[index + 1 :]:
                if other.onset >= annotation_end:
                    break
                overlap_count += 1
                overlap_annotations.add(id(annotation))
                overlap_annotations.add(id(other))

        for annotation in self._annotations:
            annotation_id = id(annotation)
            if annotation_id not in overlap_annotations:
                continue
            note = self._auto_repair_notes.get(annotation_id, "AUTO_REPAIRED - revert available")
            if "OVERLAPS" not in note:
                note = f"{note}; OVERLAPS after repair"
            self._auto_repair_notes[annotation_id] = note
            self._refresh_annotation_visuals(annotation)

        return overlap_count

    def _annotation_tooltip(self, annotation: Annotation) -> str:
        note = self._auto_repair_notes.get(id(annotation))
        if not note:
            return annotation.description
        return f"{annotation.description}\n{note}"

    def _auto_repair_eog_bounds(
        self, annotation: Annotation
    ) -> Optional[tuple[float, float, float, bool]]:
        if self.raw is None or self._times is None or self._times.size == 0:
            self.status_label.setText("No time series loaded for auto_repair_eog.")
            return None

        channel_data = self._eog_repair_data()
        if channel_data is None:
            self.status_label.setText(
                f"{EOG_REPAIR_CHANNEL} source channels not available for auto_repair_eog."
            )
            return None

        data_end = float(self._times[-1])
        start = max(0.0, annotation.onset)
        end = min(annotation.onset + annotation.duration, data_end)
        if end <= start:
            self.status_label.setText("Annotation is empty; auto_repair_eog skipped.")
            return None

        channel_times = self.raw.times
        if channel_data.size != channel_times.size:
            self.status_label.setText(
                "EOG sample/time mismatch; auto_repair_eog skipped."
            )
            return None

        print(
            f"[EOG_REPAIR] annotation onset={annotation.onset:.6f} "
            f"duration={annotation.duration:.6f}"
        )
        repaired = self._repair_eog_bounds_from_peak_trough(
            channel_times,
            channel_data,
            annotation_onset=start,
            annotation_duration=end - start,
        )
        if repaired is None:
            print("[EOG_REPAIR] repair failed: no finite peak in annotation window")
            self.status_label.setText(
                "No finite peak found in the annotation window for auto_repair_eog."
            )
            return None

        repaired_start, repaired_end, event_time = repaired
        print(
            f"[EOG_REPAIR] left={repaired_start:.4f} right={repaired_end:.4f} "
            f"peak={event_time:.4f}"
        )
        duration = max(0.0, repaired_end - repaired_start)
        return repaired_start, duration, event_time, False

    def _eog_repair_data(self) -> Optional[np.ndarray]:
        left_index = self._channel_index_by_name(EOG_REPAIR_LEFT_CHANNEL)
        if left_index is None:
            return None
        return self._repair_channel_data(left_index)

    def _auto_repair_bounds(
        self,
        annotation: Annotation,
        annotation_bounds: Optional[list[tuple[Annotation, float, float]]] = None,
        channel_name: str = DEFAULT_PRIMARY_CHANNEL,
        threshold: float = 0.0,
        threshold_unit: str = TARGET_PLOT_UNIT,
    ) -> Optional[tuple[float, float, float, bool]]:
        if self.raw is None or self._times is None or self._times.size == 0:
            self.status_label.setText("No time series loaded for auto_repair.")
            return None

        channel_index = self._channel_index_by_name(channel_name)
        if channel_index is None:
            self.status_label.setText(
                f"{channel_name} not available for auto_repair."
            )
            return None

        data_end = float(self._times[-1])
        start = max(0.0, annotation.onset)
        end = min(annotation.onset + annotation.duration, data_end)
        if end <= start:
            self.status_label.setText("Annotation is empty; auto_repair skipped.")
            return None

        start_sample = int(np.searchsorted(self._times, start, side="left"))
        end_sample = int(np.searchsorted(self._times, end, side="right"))
        if end_sample <= start_sample:
            self.status_label.setText("No channel samples in annotation; auto_repair skipped.")
            return None

        channel_data = self._repair_channel_data(channel_index)
        channel_data = self._convert_target_data_to_unit(channel_data, threshold_unit)
        channel_times = self.raw.times
        if channel_data.size != channel_times.size:
            self.status_label.setText("Channel sample/time mismatch; auto_repair skipped.")
            return None

        annotation_data = channel_data[start_sample:end_sample]
        annotation_times = channel_times[start_sample:end_sample]
        finite_mask = np.isfinite(annotation_data) & np.isfinite(annotation_times)
        if not np.any(finite_mask):
            self.status_label.setText(
                "No finite channel samples in annotation; auto_repair skipped."
            )
            return None

        valid_indices = np.nonzero(finite_mask)[0] + start_sample
        valid_data = channel_data[valid_indices]
        if threshold < 0:
            peak_index = int(valid_indices[np.argmin(valid_data)])
        else:
            peak_index = int(valid_indices[np.argmax(valid_data)])
        search_start, search_end = self._auto_repair_search_sample_bounds(
            annotation,
            channel_times,
            peak_index,
            annotation_bounds=annotation_bounds,
        )
        repaired = self._repair_bounds_from_peak(
            channel_times,
            channel_data,
            peak_index,
            search_start_index=search_start,
            search_end_index=search_end,
            allow_split_boundary=True,
            threshold=threshold,
        )
        if repaired is None:
            self.status_label.setText(
                f"Could not repair annotation on {channel_name} inside neighboring "
                f"blink boundaries using threshold {threshold:g} {threshold_unit}."
            )
            return None

        repaired_start, repaired_end, peak_time, used_split_boundary = repaired
        onset, duration = self._normalize_region_values(repaired_start, repaired_end)
        return onset, duration, peak_time, used_split_boundary

    def _channel_index_by_name(self, channel_name: str) -> Optional[int]:
        if self.raw is None:
            return None
        try:
            return self.raw.ch_names.index(channel_name)
        except ValueError:
            pass
        normalized_target = channel_name.casefold()
        for index, name in enumerate(self.raw.ch_names):
            normalized_name = name.casefold()
            if normalized_name == normalized_target or normalized_name.endswith(
                normalized_target
            ):
                return index
        return None

    def _repair_channel_data(self, channel_index: int) -> np.ndarray:
        if self.raw is None:
            return np.array([])

        channel_data = self.raw.get_data(picks=[channel_index], verbose="ERROR")[0]
        channel_name = self.raw.ch_names[channel_index]
        channel_type = self.raw.get_channel_types(picks=[channel_index])[0]
        if not self._should_normalize_channel(channel_name, channel_type):
            return channel_data

        orig_units = getattr(self.raw, "_orig_units", {}) or {}
        scale = self._scale_for_channel(channel_name, channel_index, orig_units)
        return channel_data * scale

    @classmethod
    def _convert_target_data_to_unit(cls, data: np.ndarray, unit: str) -> np.ndarray:
        normalized = cls._normalize_unit_key(unit)
        scale = UNIT_SCALE_FACTORS.get(
            normalized, UNIT_SCALE_FACTORS[cls._normalize_unit_key(TARGET_PLOT_UNIT)]
        )
        if scale == 0:
            return data
        return data / scale

    @staticmethod
    def _eog_find_left_trough_index(
        data: np.ndarray, peak_idx: int, min_separation: int = 4
    ) -> int:
        """Walk left from peak_idx to find the first local minimum.

        A turning point is only accepted once we are at least min_separation samples
        from the peak, preventing noise wiggles 2–3 samples from the peak from being
        mistaken for the blink boundary. If the signal never turns, returns 0.
        """
        i = peak_idx
        while i > 0:
            if data[i - 1] > data[i] and (peak_idx - i) >= min_separation:
                return i
            i -= 1
        return 0

    @staticmethod
    def _eog_find_right_trough_index(
        data: np.ndarray, peak_idx: int, min_separation: int = 4
    ) -> int:
        """Walk right from peak_idx to find the first local minimum.

        A turning point is only accepted once we are at least min_separation samples
        from the peak, preventing noise wiggles 2–3 samples from the peak from being
        mistaken for the blink boundary. If the signal never turns, returns the last index.
        """
        i = peak_idx
        n = len(data)
        while i < n - 1:
            if data[i + 1] > data[i] and (i - peak_idx) >= min_separation:
                return i
            i += 1
        return n - 1

    @staticmethod
    def _ear_find_left_peak_index(
        data: np.ndarray, min_idx: int, nth_peak: int = 1
    ) -> int:
        """Walk left from min_idx and return the index of the Nth true local maximum.

        A true local maximum at position i requires the signal to have been rising
        going leftward (data[i] > data[i+1]) and then falling to the left (data[i-1] < data[i]).
        nth_peak=1 returns the first such peak, 2 the second, etc.
        Returns 0 if fewer than nth_peak peaks are found.
        """
        count = 0
        i = min_idx
        while i > 0:
            if data[i - 1] < data[i] and data[i] > data[i + 1]:
                count += 1
                if count >= nth_peak:
                    return i
            i -= 1
        return 0

    @staticmethod
    def _ear_find_right_peak_index(
        data: np.ndarray, min_idx: int, nth_peak: int = 1
    ) -> int:
        """Walk right from min_idx and return the index of the Nth true local maximum.

        A true local maximum at position i requires the signal to have been rising
        going rightward (data[i] > data[i-1]) and then falling to the right (data[i+1] < data[i]).
        nth_peak=1 returns the first such peak, 2 the second, etc.
        Returns the last index if fewer than nth_peak peaks are found.
        """
        count = 0
        i = min_idx
        n = len(data)
        while i < n - 1:
            if data[i + 1] < data[i] and (i == min_idx or data[i] > data[i - 1]):
                count += 1
                if count >= nth_peak:
                    return i
            i += 1
        return n - 1

    @staticmethod
    def _repair_eog_bounds_from_peak_trough(
        times: np.ndarray,
        data: np.ndarray,
        annotation_onset: float,
        annotation_duration: float,
    ) -> Optional[tuple[float, float, float]]:
        """Find repaired bounds using peak-to-trough walking.

        Finds the maximum amplitude within the annotation window, then walks left
        and right through the full data to find the first local minimum in each
        direction. If no turning point is found within the annotation, the search
        naturally extends beyond the annotation boundaries.

        Returns (left_time, right_time, peak_time) or None if no finite peak exists.
        """
        if times.size == 0 or data.size == 0 or times.size != data.size:
            return None

        annotation_end = annotation_onset + max(0.0, annotation_duration)
        start_idx = int(np.searchsorted(times, annotation_onset, side="left"))
        end_idx = int(np.searchsorted(times, annotation_end, side="right")) - 1
        start_idx = max(0, start_idx)
        end_idx = min(int(times.size) - 1, end_idx)
        if end_idx < start_idx:
            return None

        window_data = data[start_idx : end_idx + 1]
        finite_mask = np.isfinite(window_data)
        if not np.any(finite_mask):
            return None

        peak_offset = int(np.argmax(np.where(finite_mask, window_data, -np.inf)))
        peak_idx = start_idx + peak_offset
        peak_time = float(times[peak_idx])

        left_idx = TimeSeriesViewer._eog_find_left_trough_index(data, peak_idx)
        right_idx = TimeSeriesViewer._eog_find_right_trough_index(data, peak_idx)

        return float(times[left_idx]), float(times[right_idx]), peak_time

    @staticmethod
    def _repair_ear_bounds_from_trough_peak(
        times: np.ndarray,
        data: np.ndarray,
        annotation_onset: float,
        annotation_duration: float,
        nth_peak: int = 1,
    ) -> Optional[tuple[float, float, float]]:
        """Find repaired bounds using trough-to-peak walking for EAR-avg_ear.

        Finds the minimum amplitude within the annotation window, then walks left
        and right through the full data counting local maxima. Returns the position
        of the Nth local maximum in each direction. If fewer than N peaks are found
        the search naturally extends to the data boundary.

        Returns (left_time, right_time, min_time) or None if no finite minimum exists.
        """
        if times.size == 0 or data.size == 0 or times.size != data.size:
            return None

        annotation_end = annotation_onset + max(0.0, annotation_duration)
        start_idx = int(np.searchsorted(times, annotation_onset, side="left"))
        end_idx = int(np.searchsorted(times, annotation_end, side="right")) - 1
        start_idx = max(0, start_idx)
        end_idx = min(int(times.size) - 1, end_idx)
        if end_idx < start_idx:
            return None

        window_data = data[start_idx : end_idx + 1]
        finite_mask = np.isfinite(window_data)
        if not np.any(finite_mask):
            return None

        min_offset = int(np.argmin(np.where(finite_mask, window_data, np.inf)))
        min_idx = start_idx + min_offset
        min_time = float(times[min_idx])

        left_idx = TimeSeriesViewer._ear_find_left_peak_index(data, min_idx, nth_peak)
        right_idx = TimeSeriesViewer._ear_find_right_peak_index(data, min_idx, nth_peak)

        return float(times[left_idx]), float(times[right_idx]), min_time

    @staticmethod
    def _repair_eog_bounds_from_threshold_segment(
        times: np.ndarray,
        data: np.ndarray,
        annotation_onset: float,
        annotation_duration: float,
        threshold: float,
    ) -> Optional[tuple[float, float, float]]:
        if times.size == 0 or data.size == 0 or times.size != data.size:
            return None

        finite_times = times[np.isfinite(times)]
        if finite_times.size == 0:
            return None

        annotation_start = annotation_onset
        annotation_end = annotation_onset + max(0.0, annotation_duration)
        if annotation_end < annotation_start:
            return None

        sample_interval = TimeSeriesViewer._median_sample_interval(times)
        threshold_mask = TimeSeriesViewer._eog_threshold_mask(times, data, threshold)

        peak_index = TimeSeriesViewer._eog_peak_anchor_index_for_annotation(
            times,
            data,
            threshold_mask,
            annotation_start,
            annotation_end,
            sample_interval,
            threshold,
        )
        if peak_index is None:
            return None

        bounds = TimeSeriesViewer._eog_local_event_search_bounds(
            times,
            annotation_start,
            annotation_end,
            sample_interval,
        )
        if bounds is None:
            return None
        start_index, end_index = bounds

        segment = TimeSeriesViewer._threshold_segment_containing_index(
            threshold_mask, peak_index, start_index, end_index
        )
        if segment is None:
            return None

        segment_start, segment_end = segment
        repaired_start = float(times[segment_start])
        repaired_end = float(times[segment_end])
        peak_time = float(times[peak_index])
        if not repaired_start <= peak_time <= repaired_end:
            return None
        return repaired_start, repaired_end, peak_time

    @staticmethod
    def _median_sample_interval(times: np.ndarray) -> float:
        finite_times = times[np.isfinite(times)]
        if finite_times.size < 2:
            return 0.0
        diffs = np.diff(finite_times)
        valid_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if valid_diffs.size == 0:
            return 0.0
        return float(np.median(valid_diffs))

    @staticmethod
    def _eog_threshold_mask(
        times: np.ndarray, data: np.ndarray, threshold: float
    ) -> np.ndarray:
        finite_mask = np.isfinite(times) & np.isfinite(data)
        tolerance = TimeSeriesViewer._eog_threshold_tolerance(threshold)
        if threshold < 0:
            return finite_mask & (data <= threshold + tolerance)
        return finite_mask & (data >= threshold - tolerance)

    @staticmethod
    def _eog_threshold_tolerance(threshold: float) -> float:
        return max(abs(threshold) * 1.0e-4, 1.0e-6)

    @staticmethod
    def _eog_peak_anchor_index_for_annotation(
        times: np.ndarray,
        data: np.ndarray,
        threshold_mask: np.ndarray,
        annotation_start: float,
        annotation_end: float,
        sample_interval: float,
        threshold: float,
    ) -> Optional[int]:
        bounds = TimeSeriesViewer._eog_peak_anchor_search_bounds(
            times, annotation_start, annotation_end, sample_interval
        )
        if bounds is None:
            return None

        start_index, end_index = bounds
        finite_mask = np.isfinite(times[start_index : end_index + 1]) & np.isfinite(
            data[start_index : end_index + 1]
        )
        hit_mask = finite_mask & threshold_mask[start_index : end_index + 1]
        if not np.any(hit_mask):
            return None

        valid_offsets = np.nonzero(hit_mask)[0]
        valid_data = data[start_index : end_index + 1][valid_offsets]
        if threshold < 0:
            return int(start_index + valid_offsets[int(np.argmin(valid_data))])
        return int(start_index + valid_offsets[int(np.argmax(valid_data))])

    @staticmethod
    def _eog_peak_anchor_search_bounds(
        times: np.ndarray,
        annotation_start: float,
        annotation_end: float,
        sample_interval: float,
    ) -> Optional[tuple[int, int]]:
        if times.size == 0:
            return None

        duration = max(0.0, annotation_end - annotation_start)
        tolerance = sample_interval / 2.0 if sample_interval > 0.0 else 0.0
        if duration <= MIN_ANNOTATION_DURATION + sample_interval:
            # Extend left by MIN_ANNOTATION_DURATION so the peak is found even
            # when the annotation onset falls at or after the actual event peak.
            search_start = annotation_start - MIN_ANNOTATION_DURATION - tolerance
            search_end = annotation_end + MIN_ANNOTATION_DURATION + tolerance
        else:
            search_start = annotation_end - MIN_ANNOTATION_DURATION - tolerance
            search_end = annotation_end + MIN_ANNOTATION_DURATION + tolerance

        return TimeSeriesViewer._time_window_to_sample_bounds(times, search_start, search_end)

    @staticmethod
    def _eog_local_event_search_bounds(
        times: np.ndarray,
        annotation_start: float,
        annotation_end: float,
        sample_interval: float,
    ) -> Optional[tuple[int, int]]:
        if times.size == 0:
            return None

        duration = max(0.0, annotation_end - annotation_start)
        tolerance = sample_interval / 2.0 if sample_interval > 0.0 else 0.0
        if duration <= MIN_ANNOTATION_DURATION + sample_interval:
            # Left bound extended so _threshold_segment_containing_index accepts
            # a peak that sits slightly before the annotation onset.
            search_start = annotation_start - MIN_ANNOTATION_DURATION - tolerance
            search_end = annotation_end + tolerance
        else:
            search_start = annotation_end - tolerance
            search_end = annotation_end + MIN_ANNOTATION_DURATION + tolerance

        return TimeSeriesViewer._time_window_to_sample_bounds(times, search_start, search_end)

    @staticmethod
    def _time_window_to_sample_bounds(
        times: np.ndarray, start_time: float, end_time: float
    ) -> Optional[tuple[int, int]]:
        finite_times = times[np.isfinite(times)]
        if finite_times.size == 0:
            return None

        search_start = max(float(finite_times[0]), start_time)
        search_end = min(float(finite_times[-1]), end_time)
        if search_end < search_start:
            return None

        start_index = int(np.searchsorted(times, search_start, side="left"))
        end_index = int(np.searchsorted(times, search_end, side="right")) - 1
        start_index = max(0, min(start_index, times.size - 1))
        end_index = max(start_index, min(end_index, times.size - 1))
        return start_index, end_index

    @staticmethod
    def _threshold_segment_containing_index(
        mask: np.ndarray, index: int, lower_bound: int, upper_bound: int
    ) -> Optional[tuple[int, int]]:
        if (
            mask.size == 0
            or index < lower_bound
            or index > upper_bound
            or index < 0
            or index >= mask.size
            or not bool(mask[index])
        ):
            return None

        segment_start = index
        segment_end = index
        while segment_start > lower_bound and bool(mask[segment_start - 1]):
            segment_start -= 1
        while segment_end < upper_bound and bool(mask[segment_end + 1]):
            segment_end += 1
        return segment_start, segment_end

    def _annotation_bounds_snapshot(self) -> list[tuple[Annotation, float, float]]:
        return [
            (annotation, annotation.onset, annotation.onset + annotation.duration)
            for annotation in self._annotations
        ]

    def _auto_repair_search_sample_bounds(
        self,
        annotation: Annotation,
        times: np.ndarray,
        peak_index: int,
        annotation_bounds: Optional[list[tuple[Annotation, float, float]]] = None,
    ) -> tuple[int, int]:
        bounds = annotation_bounds or self._annotation_bounds_snapshot()
        peak_time = float(times[peak_index])
        left_limit = 0.0
        right_limit = float(times[-1])

        for other, other_start, other_end in bounds:
            if other is annotation:
                continue
            other_center = (other_start + other_end) / 2.0
            if other_end <= peak_time:
                midpoint = (peak_time + other_center) / 2.0
                left_limit = max(left_limit, other_end, midpoint)
            elif other_start >= peak_time:
                midpoint = (peak_time + other_center) / 2.0
                right_limit = min(right_limit, other_start, midpoint)
            elif other_center < peak_time:
                midpoint = (peak_time + other_center) / 2.0
                left_limit = max(left_limit, midpoint)
            else:
                midpoint = (peak_time + other_center) / 2.0
                right_limit = min(right_limit, midpoint)

        search_start = int(np.searchsorted(times, left_limit, side="left"))
        search_end = int(np.searchsorted(times, right_limit, side="right")) - 1
        search_start = max(0, min(search_start, peak_index))
        search_end = min(times.size - 1, max(search_end, peak_index))
        return search_start, search_end

    @staticmethod
    def _repair_bounds_from_samples(
        times: np.ndarray, data: np.ndarray, threshold: float = 0.0
    ) -> Optional[tuple[float, float, float]]:
        if times.size == 0 or data.size == 0 or times.size != data.size:
            return None

        if threshold < 0:
            peak_index = int(np.argmin(data))
        else:
            peak_index = int(np.argmax(data))
        return TimeSeriesViewer._repair_bounds_from_peak(
            times, data, peak_index, threshold=threshold
        )

    @staticmethod
    def _repair_bounds_from_peak(
        times: np.ndarray,
        data: np.ndarray,
        peak_index: int,
        search_start_index: int = 0,
        search_end_index: Optional[int] = None,
        allow_split_boundary: bool = False,
        threshold: float = 0.0,
    ) -> Optional[tuple[float, float, float] | tuple[float, float, float, bool]]:
        if (
            times.size == 0
            or data.size == 0
            or times.size != data.size
            or peak_index < 0
            or peak_index >= data.size
        ):
            return None

        negative_event = threshold < 0
        if negative_event:
            if data[peak_index] >= threshold:
                return None
        elif data[peak_index] <= threshold:
            return None

        if search_end_index is None:
            search_end_index = data.size - 1
        search_start_index = max(0, min(search_start_index, peak_index))
        search_end_index = min(data.size - 1, max(search_end_index, peak_index))

        if negative_event:
            left_crossing = TimeSeriesViewer._nearest_downward_threshold_crossing_left(
                times, data, peak_index, search_start_index, threshold=threshold
            )
            right_crossing = TimeSeriesViewer._nearest_upward_threshold_crossing_right(
                times, data, peak_index, search_end_index, threshold=threshold
            )
        else:
            left_crossing = TimeSeriesViewer._nearest_upward_zero_crossing(
                times, data, peak_index, search_start_index, threshold=threshold
            )
            right_crossing = TimeSeriesViewer._nearest_downward_zero_crossing(
                times, data, peak_index, search_end_index, threshold=threshold
            )
        used_split_boundary = False
        if allow_split_boundary:
            if left_crossing is None:
                left_crossing = float(times[search_start_index])
                used_split_boundary = True
            if right_crossing is None:
                right_crossing = float(times[search_end_index])
                used_split_boundary = True

        if left_crossing is None or right_crossing is None or right_crossing <= left_crossing:
            return None

        if allow_split_boundary:
            return left_crossing, right_crossing, float(times[peak_index]), used_split_boundary
        return left_crossing, right_crossing, float(times[peak_index])

    @staticmethod
    def _nearest_upward_zero_crossing(
        times: np.ndarray,
        data: np.ndarray,
        peak_index: int,
        search_start_index: int = 0,
        threshold: float = 0.0,
    ) -> Optional[float]:
        start = max(0, min(search_start_index, peak_index))
        for index in range(peak_index, max(start, 1) - 1, -1):
            left = float(data[index - 1])
            right = float(data[index])
            if left <= threshold < right:
                return TimeSeriesViewer._interpolated_threshold_time(
                    float(times[index - 1]), left, float(times[index]), right, threshold
                )
        return float(times[start]) if data[start] == threshold else None

    @staticmethod
    def _nearest_downward_zero_crossing(
        times: np.ndarray,
        data: np.ndarray,
        peak_index: int,
        search_end_index: int,
        threshold: float = 0.0,
    ) -> Optional[float]:
        end = min(data.size - 1, max(search_end_index, peak_index))
        for index in range(peak_index + 1, end + 1):
            left = float(data[index - 1])
            right = float(data[index])
            if left > threshold >= right:
                return TimeSeriesViewer._interpolated_threshold_time(
                    float(times[index - 1]), left, float(times[index]), right, threshold
                )
        return float(times[end]) if data[end] == threshold else None

    @staticmethod
    def _nearest_downward_threshold_crossing_left(
        times: np.ndarray,
        data: np.ndarray,
        peak_index: int,
        search_start_index: int = 0,
        threshold: float = 0.0,
    ) -> Optional[float]:
        start = max(0, min(search_start_index, peak_index))
        for index in range(peak_index, max(start, 1) - 1, -1):
            left = float(data[index - 1])
            right = float(data[index])
            if left > threshold >= right:
                return TimeSeriesViewer._interpolated_threshold_time(
                    float(times[index - 1]), left, float(times[index]), right, threshold
                )
        return float(times[start]) if data[start] == threshold else None

    @staticmethod
    def _nearest_upward_threshold_crossing_right(
        times: np.ndarray,
        data: np.ndarray,
        peak_index: int,
        search_end_index: int,
        threshold: float = 0.0,
    ) -> Optional[float]:
        end = min(data.size - 1, max(search_end_index, peak_index))
        for index in range(peak_index + 1, end + 1):
            left = float(data[index - 1])
            right = float(data[index])
            if left <= threshold < right:
                return TimeSeriesViewer._interpolated_threshold_time(
                    float(times[index - 1]), left, float(times[index]), right, threshold
                )
        return float(times[end]) if data[end] == threshold else None

    @staticmethod
    def _interpolated_threshold_time(
        left_time: float,
        left_value: float,
        right_time: float,
        right_value: float,
        threshold: float = 0.0,
    ) -> float:
        if right_value == left_value:
            return left_time
        fraction = (threshold - left_value) / (right_value - left_value)
        fraction = max(0.0, min(1.0, fraction))
        return left_time + fraction * (right_time - left_time)

    def _refresh_annotation_visuals(self, annotation: Annotation) -> None:
        base_color = self._color_for_description(annotation.description)
        brush_color = pg.mkColor(base_color)
        brush_color.setAlpha(60)
        pen_color = pg.mkColor(base_color)
        pen_color.setAlpha(160)
        pen_width = 1

        note = self._auto_repair_notes.get(id(annotation), "")
        if "OVERLAPS" in note:
            brush_color = pg.mkColor("#ffeb3b")
            brush_color.setAlpha(140)
            pen_color = pg.mkColor("#d32f2f")
            pen_color.setAlpha(255)
            pen_width = 3
        elif note:
            brush_color = pg.mkColor("#81d4fa")
            brush_color.setAlpha(120)
            pen_color = pg.mkColor("#0277bd")
            pen_color.setAlpha(255)
            pen_width = 2

        for items in self._annotation_items_by_widget.values():
            for item in items:
                if item.annotation is not annotation:
                    continue
                item.region.setToolTip(self._annotation_tooltip(annotation))
                item.region.setBrush(pg.mkBrush(brush_color))
                self._set_region_pen(item.region, pg.mkPen(pen_color, width=pen_width))
                item.region.setVisible(self._annotation_visible(annotation))

    @staticmethod
    def _set_region_pen(region: pg.LinearRegionItem, pen: pg.Pen) -> None:
        if hasattr(region, "setPen"):
            region.setPen(pen)
            return
        for line in getattr(region, "lines", []):
            if hasattr(line, "setPen"):
                line.setPen(pen)

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

    def _remove_annotation_from_model(self, annotation: Annotation) -> None:
        self._auto_repair_original_bounds.pop(id(annotation), None)
        self._auto_repair_notes.pop(id(annotation), None)
        self._annotations = [item for item in self._annotations if item is not annotation]

    def _delete_annotation(self, annotation_item: AnnotationItem) -> None:
        self._remove_annotation_from_model(annotation_item.annotation)
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
                if lane_item.annotation is annotation_item.annotation:
                    widget.removeItem(lane_item.region)
                    items.remove(lane_item)
        self._set_annotations_dirty(True)
        self._update_annotation_filter_options()
        self.status_label.setText("Annotation deleted.")
        if self._selected_annotation is annotation_item.annotation:
            self._selected_annotation = None

    def delete_selected_annotation(self) -> None:
        """Delete the currently selected annotation, if one is selected."""

        if self._selected_annotation is None:
            self.status_label.setText("Select an annotation before deleting.")
            return

        annotation = self._selected_annotation
        for annotation_item in self._annotation_items_by_widget.get(self.plot_widget, []):
            if annotation_item.annotation is annotation:
                self._delete_annotation(annotation_item)
                return

        self.status_label.setText("Selected annotation is not available to delete.")

    def _remove_annotation_regions(self, annotation: Annotation) -> None:
        for widget, items in self._annotation_items_by_widget.items():
            for item in list(items):
                if item.annotation is not annotation:
                    continue
                self._annotation_by_region.pop(item.region, None)
                widget.removeItem(item.region)
                items.remove(item)

    def _overlapping_annotations(
        self, onset: float, duration: float, description: str
    ) -> List[Annotation]:
        end = onset + duration
        return [
            annotation
            for annotation in self._annotations
            if annotation.description == description
            and onset < annotation.onset + annotation.duration
            and annotation.onset < end
        ]

    def _merge_annotation(self, onset: float, duration: float, description: str) -> Annotation:
        merged_start = onset
        merged_end = onset + duration
        overlaps: List[Annotation] = []

        while True:
            next_overlaps = [
                annotation
                for annotation in self._overlapping_annotations(
                    merged_start,
                    merged_end - merged_start,
                    description,
                )
                if all(annotation is not existing for existing in overlaps)
            ]
            if not next_overlaps:
                break
            overlaps.extend(next_overlaps)
            merged_start = min(
                [merged_start] + [annotation.onset for annotation in next_overlaps]
            )
            merged_end = max(
                [merged_end]
                + [annotation.onset + annotation.duration for annotation in next_overlaps]
            )

        if not overlaps:
            annotation = Annotation(onset=onset, duration=duration, description=description)
            self._annotations.append(annotation)
            self._render_annotation(annotation)
            self.status_label.setText("Annotation added.")
            self._last_create_undo = self._make_undo_new_annotation(annotation)
            return annotation

        merged_annotation = overlaps[0]
        original_onset = merged_annotation.onset
        original_duration = merged_annotation.duration
        absorbed = list(overlaps[1:])
        for annotation in absorbed:
            self._remove_annotation_from_model(annotation)
            self._remove_annotation_regions(annotation)

        merged_annotation.onset = merged_start
        merged_annotation.duration = merged_end - merged_start
        self._sync_annotation_regions(
            merged_annotation,
            merged_annotation.onset,
            merged_annotation.duration,
        )
        self.status_label.setText("Annotation merged with existing label.")
        self._last_create_undo = self._make_undo_merge_annotation(
            merged_annotation, original_onset, original_duration, absorbed
        )
        return merged_annotation

    def _make_undo_new_annotation(self, annotation: Annotation):
        def undo():
            self._remove_annotation_from_model(annotation)
            self._remove_annotation_regions(annotation)
            if self._selected_annotation is annotation:
                self._selected_annotation = None
            self._set_annotations_dirty(True)
            self._update_annotation_filter_options()
            self.status_label.setText("Undo: annotation removed.")
        return undo

    def _make_undo_annotation_repair(
        self, annotation: Annotation, onset: float, duration: float
    ):
        def undo():
            annotation.onset = onset
            annotation.duration = duration
            self._sync_annotation_regions(annotation, onset, duration)
            self._auto_repair_original_bounds.pop(id(annotation), None)
            self._auto_repair_notes.pop(id(annotation), None)
            self._set_selected_annotation(annotation, announce=False)
            self._set_annotations_dirty(True)
            self.status_label.setText("Undo: annotation repair reverted.")
        return undo

    def _make_undo_merge_annotation(
        self,
        merged: Annotation,
        original_onset: float,
        original_duration: float,
        absorbed: List[Annotation],
    ):
        def undo():
            merged.onset = original_onset
            merged.duration = original_duration
            self._sync_annotation_regions(merged, original_onset, original_duration)
            for annotation in absorbed:
                self._annotations.append(annotation)
                self._render_annotation(annotation)
            self._set_annotations_dirty(True)
            self._update_annotation_filter_options()
            self.status_label.setText("Undo: merge reversed.")
        return undo

    def undo_last_annotation(self) -> None:
        if self._last_create_undo is None:
            self.status_label.setText("Nothing to undo.")
            return
        self._last_create_undo()
        self._last_create_undo = None

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
            annotation = self._merge_annotation(onset, duration, description)
            self._set_selected_annotation(annotation, announce=False)
            self._set_annotations_dirty(True)
            self._update_annotation_filter_options()

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
            | {"BAD_SIGNAL_BLINK"}
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
        """Jump to the next annotation and center on the EAR-avg_ear minimum."""

        self.jump_to_next_annotation_minimum()

    def jump_to_previous_annotation(self) -> None:
        """Jump the view to the previous annotation onset."""

        self._jump_to_annotation(direction="previous")

    def _jump_to_annotation(self, direction: str) -> None:
        target = self._target_annotation(direction)
        if target is None:
            return

        self._set_selected_annotation(target, announce=False)
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

        self._set_selected_annotation(target, announce=False)
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
            note = self._auto_repair_notes.get(id(annotation))
            note_text = f" [{note}]" if note else ""
            self.status_label.setText(
                f"Selected annotation '{annotation.description}' at {annotation.onset:.2f}s."
                f"{note_text}"
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

    def nudge_selected_annotation_left(self) -> None:
        """Move the selected annotation left by one nudge step (Shift+Left)."""
        self._nudge_selected_annotation(-1)

    def nudge_selected_annotation_right(self) -> None:
        """Move the selected annotation right by one nudge step (Shift+Right)."""
        self._nudge_selected_annotation(1)

    def _mouse_time_in_plot(self) -> Optional[float]:
        """Return the time under the mouse cursor if it is over any lane widget, else None."""
        global_pos = QCursor.pos()
        for widget in self._lane_widgets:
            local_pos = widget.mapFromGlobal(global_pos)
            rect = widget.rect()
            if rect.contains(local_pos):
                t = self._time_at_position(local_pos, widget)
                if t is not None and self._times is not None and self._times.size > 0:
                    data_end = float(self._times[-1])
                    if 0.0 <= t <= data_end:
                        return t
        return None

    def create_annotation_from_ear(self) -> None:
        """Create a new annotation by finding the EAR trough near the cursor or mouse.

        The search window is one view-span wide, centered on:
          1. The mouse position, if the pointer is currently over a plot lane.
          2. The red cursor line otherwise.

        Finds the minimum EAR-avg_ear amplitude in that window, walks left and
        right to the Nth local peak (ear_repair_nth_peak setting), prompts for
        a label, and creates the annotation.
        """
        if self.raw is None or self._times is None or self._times.size == 0:
            self.status_label.setText("No time series loaded.")
            return

        ear_index = self._channel_index_by_name(EAR_AVG_CHANNEL)
        if ear_index is None:
            self.status_label.setText(f"{EAR_AVG_CHANNEL} not available for annotation creation.")
            return

        channel_data = self._repair_channel_data(ear_index)
        channel_times = self.raw.times
        if channel_data.size != channel_times.size:
            self.status_label.setText("EAR sample/time mismatch; annotation creation skipped.")
            return

        center_time = self._mouse_time_in_plot()
        if center_time is None:
            center_time = self._last_cursor_time

        half_span = self.view_span_seconds / 2.0
        data_end = float(self._times[-1])
        search_start = max(0.0, center_time - half_span)
        search_end = min(data_end, center_time + half_span)
        if search_end <= search_start:
            self.status_label.setText("View window too small; move cursor and try again.")
            return

        repaired = self._repair_ear_bounds_from_trough_peak(
            channel_times,
            channel_data,
            annotation_onset=search_start,
            annotation_duration=search_end - search_start,
            nth_peak=self._ear_repair_nth_peak,
        )
        if repaired is None:
            self.status_label.setText(
                "No finite EAR minimum found near the cursor for annotation creation."
            )
            return

        onset, end_time, min_time = repaired
        duration = max(MIN_ANNOTATION_DURATION, end_time - onset)
        if onset + duration > data_end:
            onset = max(0.0, data_end - duration)

        description = self._prompt_for_description()
        if not description:
            return

        annotation = Annotation(onset=onset, duration=duration, description=description)
        self._annotations.append(annotation)
        self._render_annotation(annotation)
        self._last_create_undo = self._make_undo_new_annotation(annotation)
        self._set_selected_annotation(annotation, announce=False)
        self._set_annotations_dirty(True)
        self._update_annotation_filter_options()
        self.status_label.setText(
            f"Created annotation '{description}' at {onset:.3f}s–{onset + duration:.3f}s "
            f"around EAR trough at {min_time:.3f}s."
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
