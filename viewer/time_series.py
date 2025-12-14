"""Time series loading and visualization helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Set, Tuple

import mne
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)
from viewer.utils import placeholder_pixmap

ANNOTATION_MIN_DURATION = 0.001


@dataclass
class AnnotationSegment:
    """Container for a single annotation interval."""

    onset: float
    duration: float
    description: str

    @property
    def end(self) -> float:
        return self.onset + self.duration


def sanitize_annotation(
    onset: float, duration: float, max_end: Optional[float] = None
) -> Tuple[float, float]:
    """Clamp an annotation interval to valid bounds.

    Ensures onset is not negative, duration stays above the minimum, and the
    annotation remains inside the optional ``max_end`` bound.
    """

    onset = max(0.0, onset)
    duration = max(duration, ANNOTATION_MIN_DURATION)

    if max_end is not None:
        end = min(onset + duration, max_end)
        onset = min(onset, max_end - ANNOTATION_MIN_DURATION)
        duration = max(ANNOTATION_MIN_DURATION, end - onset)

    return onset, duration


def load_annotations_from_csv(csv_path: Path) -> List[AnnotationSegment]:
    """Load annotations from a CSV file with onset, duration, description."""

    if not csv_path.exists():
        return []

    segments: List[AnnotationSegment] = []
    with csv_path.open("r", encoding="utf-8") as csv_file:
        csv_file.readline()
        for line in csv_file:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                onset = float(parts[0])
                duration = float(parts[1])
            except ValueError:
                continue
            description = parts[2]
            segments.append(AnnotationSegment(onset, duration, description))

    return segments


def save_annotations_to_csv(csv_path: Path, annotations: List[AnnotationSegment]) -> None:
    """Persist annotations to disk in MNE-compatible CSV format."""

    with csv_path.open("w", encoding="utf-8") as csv_file:
        csv_file.write("onset,duration,description\n")
        for segment in annotations:
            csv_file.write(
                f"{segment.onset:.9f},{segment.duration:.9f},{segment.description}\n"
            )

PROCESSED_ROOT = Path(r"D:\dataset\drowsy_driving_raja_processed")
PRIMARY_CHANNEL = "EEG-E8"
DEFAULT_VISIBLE_CHANNELS = {PRIMARY_CHANNEL}
CHANNEL_PALETTE = [
    "#d32f2f",  # primary red
    "#c62828",
    "#ef5350",
    "#b71c1c",
    "#f44336",
]
ANNOTATION_PALETTE = [
    "#fff59d",
    "#a5d6a7",
    "#90caf9",
    "#ffcc80",
    "#ce93d8",
    "#b0bec5",
]


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
        self._selected_channels: Set[str] = set()
        self._last_cursor_time: float = 0.0
        self.default_view_span_seconds: float = 5.0
        self.view_span_seconds: float = self.default_view_span_seconds
        self.min_span_seconds: float = 0.1
        self._last_ts_path: Optional[Path] = None
        self.annotations: List[AnnotationSegment] = []
        self.annotation_items: Dict[pg.LinearRegionItem, AnnotationSegment] = {}
        self.annotation_labels: Dict[pg.LinearRegionItem, pg.TextItem] = {}
        self._annotation_path: Optional[Path] = None
        self._annotation_drag_start: Optional[float] = None
        self._preview_region: Optional[pg.LinearRegionItem] = None
        self._selected_region: Optional[pg.LinearRegionItem] = None
        self._last_frame_pixmap = None
        self._last_frame_time: float = 0.0
        self._last_frame_index: Optional[int] = None
        self.default_annotation_label = "HB_CL"
        self._annotation_color_map: Dict[str, str] = {}

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

        self.status_label = QLabel("Load a video to view synchronized time series data.")

        self.annotation_controls = self._build_annotation_controls()
        self.frame_preview = self._build_frame_preview_panel()

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self.plot_widget)
        layout.addWidget(self.annotation_controls)
        layout.addWidget(self.frame_preview)
        layout.addWidget(self.status_label)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)
        layout.setStretch(3, 0)

        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._delete_selected_annotation)
        QShortcut(
            QKeySequence(Qt.Key_Backspace), self, activated=self._delete_selected_annotation
        )
        self._update_annotation_buttons()

    def channel_controls(self) -> QWidget:
        """Expose the channel selection controls for external layouts."""

        return self._controls_container

    def _build_annotation_controls(self) -> QWidget:
        group = QGroupBox("Annotations")
        layout = QHBoxLayout()
        group.setLayout(layout)

        self.annotation_edit_toggle = QCheckBox("Enable editing")
        self.annotation_edit_toggle.setChecked(True)

        layout.addWidget(QLabel("Label:"))
        self.annotation_label_field = QLineEdit(self.default_annotation_label)
        self.annotation_label_field.setMaximumWidth(120)
        self.annotation_label_field.editingFinished.connect(self._update_default_label)
        layout.addWidget(self.annotation_label_field)

        self.annotation_status = QLabel(
            "Click and drag on the plot to add annotations; drag edges to adjust."
        )

        self.delete_annotation_button = QPushButton("Delete selection")
        self.delete_annotation_button.clicked.connect(self._delete_selected_annotation)
        self.save_annotations_button = QPushButton("Save annotations")
        self.save_annotations_button.clicked.connect(self._save_annotations)

        layout.addWidget(self.annotation_edit_toggle)
        layout.addWidget(self.delete_annotation_button)
        layout.addWidget(self.save_annotations_button)
        layout.addWidget(self.annotation_status)
        layout.addStretch()
        return group

    def _build_frame_preview_panel(self) -> QWidget:
        group = QGroupBox("Frame Comparison")
        layout = QHBoxLayout()
        group.setLayout(layout)

        self.frame_preview_label = QLabel()
        self.frame_preview_label.setAlignment(Qt.AlignCenter)
        self.frame_preview_label.setFixedSize(240, 135)
        self.frame_preview_label.setPixmap(
            placeholder_pixmap(self.frame_preview_label.size())
        )

        info_layout = QVBoxLayout()
        self.frame_time_label = QLabel("Time: - s")
        self.frame_index_label = QLabel("Frame: -")
        self.frame_description_label = QLabel(
            "Scrub the time-series to keep this preview synchronized."
        )
        self.frame_description_label.setWordWrap(True)

        info_layout.addWidget(self.frame_time_label)
        info_layout.addWidget(self.frame_index_label)
        info_layout.addWidget(self.frame_description_label)
        info_layout.addStretch()

        layout.addWidget(self.frame_preview_label)
        layout.addLayout(info_layout)
        return group

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
        self._annotation_path = None

        ts_path = self._resolve_time_series_path(video_path)
        if ts_path is None:
            self.status_label.setText(
                "Select a video to view its time series or place ear_eog.fif next to ear_eog.csv."
            )
            self.cursor_line.hide()
            return

        self.status_label.setText(f"Loaded time series from {ts_path}.")
        self._last_ts_path = ts_path
        self._annotation_path = ts_path.with_suffix(".csv")
        self.raw = mne.io.read_raw_fif(str(ts_path), preload=True, verbose="ERROR")
        self._times = self.raw.times
        self._populate_channel_list()
        self._plot_data()
        self._load_annotations()
        self._ensure_view_range(0.0)

    def _resolve_time_series_path(self, video_path: Optional[Path]) -> Optional[Path]:
        """Find a usable ear_eog.fif path and its neighboring CSV annotations."""

        candidates: List[Path] = []

        if video_path is not None:
            try:
                candidates.append(derive_time_series_path(video_path))
            except ValueError:
                pass

            if video_path.is_file():
                candidates.append(video_path.parent / "ear_eog.fif")

        candidates.append(Path.cwd() / "ear_eog.fif")

        for ts_path in candidates:
            if ts_path.exists():
                return ts_path

        return None

    def update_cursor_time(self, seconds: float) -> None:
        """Keep the current time centered under a fixed cursor."""

        if self._times is None or self._times.size == 0:
            self.cursor_line.hide()
            return

        clamped = max(0.0, seconds)
        self._ensure_view_range(clamped)
        self._last_frame_time = clamped
        self._refresh_frame_preview()

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

        peak = np.nanmax(np.abs(data)) or 1.0
        spacing = peak * 2
        offsets = np.arange(data.shape[0]) * spacing

        self.plot_widget.enableAutoRange()
        channel_names = [self.raw.ch_names[index] for index in picks]
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

    def _load_annotations(self) -> None:
        self._clear_annotations()
        if self._annotation_path is None:
            self.annotation_status.setText("Annotations: no file associated.")
            self._update_annotation_buttons()
            return

        max_end = self._recording_end_time()
        loaded = load_annotations_from_csv(self._annotation_path)
        for segment in loaded:
            onset, duration = sanitize_annotation(segment.onset, segment.duration, max_end)
            self.annotations.append(
                AnnotationSegment(onset=onset, duration=duration, description=segment.description)
            )

        for segment in self.annotations:
            self._create_annotation_item(segment)

        if loaded:
            self.annotation_status.setText(
                f"Loaded {len(self.annotations)} annotation(s) from {self._annotation_path.name}."
            )
        else:
            self.annotation_status.setText("No annotations found; draw to add new ones.")
        self._update_annotation_buttons()

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
        for region in list(self.annotation_items.keys()):
            self.plot_widget.removeItem(region)
        for label in list(self.annotation_labels.values()):
            self.plot_widget.removeItem(label)
        if self._preview_region:
            self.plot_widget.removeItem(self._preview_region)
        self.annotations.clear()
        self.annotation_items.clear()
        self.annotation_labels.clear()
        self._preview_region = None
        self._selected_region = None
        self._annotation_color_map.clear()
        self._update_annotation_buttons()

    def _pen_for_channel(self, channel_name: str, index: int) -> pg.Pen:
        if channel_name == PRIMARY_CHANNEL:
            return pg.mkPen(CHANNEL_PALETTE[0], width=1.5)

        palette_index = (index + 1) % len(CHANNEL_PALETTE)
        return pg.mkPen(CHANNEL_PALETTE[palette_index], width=1)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self.plot_widget.viewport():
            if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                multiplier = 0.8 if delta > 0 else 1.25
                anchor_time = self._time_at_position(event.pos())
                self._adjust_zoom(multiplier, anchor_time=anchor_time)
                return True

            if self.annotation_edit_toggle.isChecked() and self.raw is not None:
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    time_point = self._time_at_position(event.pos())
                    if time_point is not None:
                        self._start_annotation_drag(time_point)
                        return True
                if event.type() == QEvent.MouseMove and self._annotation_drag_start is not None:
                    time_point = self._time_at_position(event.pos())
                    if time_point is not None:
                        self._update_preview_region(time_point)
                        return True
                if (
                    event.type() == QEvent.MouseButtonRelease
                    and event.button() == Qt.LeftButton
                    and self._annotation_drag_start is not None
                ):
                    time_point = self._time_at_position(event.pos())
                    if time_point is not None:
                        self._finish_annotation_drag(time_point)
                    self._clear_preview_region()
                    self._annotation_drag_start = None
                    return True

        return super().eventFilter(obj, event)

    def _time_at_position(self, pos) -> Optional[float]:
        view_box = self.plot_widget.getPlotItem().getViewBox()
        if view_box is None:
            return None

        scene_pos = self.plot_widget.mapToScene(pos)
        view_pos = view_box.mapSceneToView(scene_pos)
        return view_pos.x()

    def _start_annotation_drag(self, time_point: float) -> None:
        self._annotation_drag_start = time_point
        if self._preview_region is None:
            self._preview_region = pg.LinearRegionItem(movable=False)
            self._preview_region.setBrush(pg.mkBrush(50, 150, 255, 80))
            self._preview_region.setZValue(5)
            self.plot_widget.addItem(self._preview_region)
        self._update_preview_region(time_point)

    def _update_preview_region(self, current_time: float) -> None:
        if self._preview_region is None or self._annotation_drag_start is None:
            return
        start = self._annotation_drag_start
        self._preview_region.setRegion((min(start, current_time), max(start, current_time)))

    def _finish_annotation_drag(self, end_time: float) -> None:
        if self._annotation_drag_start is None:
            return
        start = self._annotation_drag_start
        if end_time == start:
            return

        onset = min(start, end_time)
        duration = abs(end_time - start)
        max_end = self._recording_end_time()
        onset, duration = sanitize_annotation(onset, duration, max_end)
        description = self._current_annotation_label()
        segment = AnnotationSegment(onset=onset, duration=duration, description=description)
        self.annotations.append(segment)
        self._create_annotation_item(segment)
        self.annotation_status.setText(
            f"Added annotation at {onset:.3f}s for {duration:.3f}s as '{description}'."
        )
        self._update_annotation_buttons()

    def _clear_preview_region(self) -> None:
        if self._preview_region:
            self.plot_widget.removeItem(self._preview_region)
        self._preview_region = None

    def _current_annotation_label(self) -> str:
        text = self.annotation_label_field.text().strip()
        if text:
            self.default_annotation_label = text
        return self.default_annotation_label

    def _annotation_color(self, description: str) -> str:
        if description not in self._annotation_color_map:
            index = len(self._annotation_color_map) % len(ANNOTATION_PALETTE)
            self._annotation_color_map[description] = ANNOTATION_PALETTE[index]
        return self._annotation_color_map[description]

    def _annotation_brush(self, color_name: str) -> pg.Brush:
        color = pg.mkColor(color_name)
        color.setAlpha(80)
        return pg.mkBrush(color)

    def _create_annotation_item(self, segment: AnnotationSegment) -> None:
        region = pg.LinearRegionItem(values=(segment.onset, segment.end), movable=True)
        brush = self._annotation_brush(self._annotation_color(segment.description))
        region.setBrush(brush)
        region.setZValue(10)
        region.sigRegionChangeStarted.connect(lambda: self._select_region(region))
        region.sigRegionChangeFinished.connect(lambda: self._on_region_changed(region))
        region.sigRegionChanged.connect(lambda: self._update_annotation_label_position(region))
        self.annotation_items[region] = segment
        self.plot_widget.addItem(region)

        label = pg.TextItem(segment.description, anchor=(0.5, 1.0), color="k")
        self.annotation_labels[region] = label
        self.plot_widget.addItem(label)
        self._update_annotation_label_position(region)
        self._refresh_region_styles()

    def _select_region(self, region: Optional[pg.LinearRegionItem]) -> None:
        if region is self._selected_region:
            return
        self._selected_region = region
        self._refresh_region_styles()
        self._update_annotation_buttons()

    def _refresh_region_styles(self) -> None:
        for region, segment in self.annotation_items.items():
            color = self._annotation_color(segment.description)
            width = 3 if region is self._selected_region else 1.5
            region.setPen(pg.mkPen(color, width=width))
            region.setBrush(self._annotation_brush(color))
            region.setZValue(15 if region is self._selected_region else 10)

    def _update_annotation_label_position(self, region: pg.LinearRegionItem) -> None:
        label = self.annotation_labels.get(region)
        if label is None:
            return
        min_x, max_x = region.getRegion()
        center_x = (min_x + max_x) / 2
        view_range = self.plot_widget.viewRange()
        if not view_range or len(view_range) < 2:
            return
        _, (y_min, y_max) = view_range
        label_y = y_max - (y_max - y_min) * 0.05
        label.setPos(center_x, label_y)

    def _on_region_changed(self, region: pg.LinearRegionItem) -> None:
        segment = self.annotation_items.get(region)
        if segment is None:
            return
        start, end = region.getRegion()
        onset, duration = sanitize_annotation(start, end - start, self._recording_end_time())
        segment.onset = onset
        segment.duration = duration
        region.setRegion((onset, onset + duration))
        self._update_annotation_label_position(region)
        self.annotation_status.setText(
            f"Updated '{segment.description}' to {onset:.3f}s – {onset + duration:.3f}s."
        )
        self._select_region(region)

    def _delete_selected_annotation(self) -> None:
        if self._selected_region is None:
            return
        segment = self.annotation_items.pop(self._selected_region, None)
        label = self.annotation_labels.pop(self._selected_region, None)
        if segment in self.annotations:
            self.annotations.remove(segment)
        self.plot_widget.removeItem(self._selected_region)
        if label:
            self.plot_widget.removeItem(label)
        self._selected_region = None
        self.annotation_status.setText("Annotation removed.")
        self._update_annotation_buttons()

    def _save_annotations(self) -> None:
        if self._annotation_path is None:
            self.annotation_status.setText("No annotation file available to save.")
            return

        max_end = self._recording_end_time()
        sanitized: List[AnnotationSegment] = []
        for segment in self.annotations:
            onset, duration = sanitize_annotation(segment.onset, segment.duration, max_end)
            segment.onset = onset
            segment.duration = duration
            sanitized.append(
                AnnotationSegment(onset=onset, duration=duration, description=segment.description)
            )

        save_annotations_to_csv(self._annotation_path, sanitized)
        self.annotation_status.setText(
            f"Saved {len(sanitized)} annotation(s) to {self._annotation_path.name}."
        )

    def _update_default_label(self) -> None:
        text = self.annotation_label_field.text().strip()
        if text:
            self.default_annotation_label = text
        else:
            self.annotation_label_field.setText(self.default_annotation_label)

    def _update_annotation_buttons(self) -> None:
        self.delete_annotation_button.setEnabled(self._selected_region is not None)
        self.save_annotations_button.setEnabled(bool(self.annotations))

    def _recording_end_time(self) -> Optional[float]:
        if self._times is None or self._times.size == 0:
            return None
        return float(self._times[-1])

    def update_frame_reference(
        self, seconds: float, pixmap: Optional[QPixmap] = None, frame_index: Optional[int] = None
    ) -> None:
        """Update the frame comparison panel with synchronized context."""

        if pixmap is not None:
            scaled = pixmap.scaled(
                self.frame_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._last_frame_pixmap = scaled
        if frame_index is not None:
            self._last_frame_index = frame_index
        self._last_frame_time = max(0.0, seconds)
        self._refresh_frame_preview()

    def _refresh_frame_preview(self) -> None:
        pixmap = self._last_frame_pixmap or placeholder_pixmap(self.frame_preview_label.size())
        self.frame_preview_label.setPixmap(pixmap)
        self.frame_time_label.setText(f"Time: {self._last_frame_time:.3f} s")
        frame_text = "-" if self._last_frame_index is None else str(self._last_frame_index)
        self.frame_index_label.setText(f"Frame: {frame_text}")
        self.frame_description_label.setText(
            "Use the time-series cursor to keep this preview aligned with annotations."
        )
