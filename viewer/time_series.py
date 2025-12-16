"""Time series loading and visualization helpers."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Set

import mne
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from viewer.annotations import (
    AnnotationEntry,
    clamp_entry,
    load_annotations,
    save_annotations,
    validate_annotations,
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

    expected = processed_root / subject_folder / base_identifier / "ear_eog.fif"
    fallback = processed_root / subject_folder / "ear_eog.fif"
    if not expected.exists() and fallback.exists():
        return fallback

    return expected


class TimeSeriesViewer(QWidget):
    """Widget that renders time series data alongside the video frames."""

    cursor_time_changed = pyqtSignal(float)

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
        self.processed_root: Path = PROCESSED_ROOT
        self._last_ts_path: Optional[Path] = None
        self._annotation_file: Optional[Path] = None
        self._annotation_entries: Dict[str, AnnotationEntry] = {}
        self._annotation_regions: Dict[str, pg.LinearRegionItem] = {}
        self._annotation_labels: Dict[str, pg.TextItem] = {}
        self._next_annotation_id: int = 1
        self._selected_annotation: Optional[str] = None
        self._annotation_drag_start: Optional[float] = None
        self._suppress_cursor_signal: bool = False
        self._default_label = "HB_CL"
        self._last_used_label = self._default_label
        self._min_annotation_duration = 0.05

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

        annotation_row = QHBoxLayout()
        self.annotation_label_input = QLineEdit(self._default_label)
        self.annotation_label_input.setPlaceholderText("Annotation label")
        self.annotation_label_input.setClearButtonEnabled(True)
        self.save_annotations_button = QPushButton("Save Annotations")
        self.save_annotations_button.clicked.connect(self._save_annotations)

        annotation_row.addWidget(QLabel("Label:"))
        annotation_row.addWidget(self.annotation_label_input)
        annotation_row.addWidget(self.save_annotations_button)

        control_layout.addLayout(annotation_row)
        self._controls_container.setLayout(control_layout)

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Channels")
        self.plot_widget.setFocusPolicy(Qt.StrongFocus)
        self.plot_widget.viewport().installEventFilter(self)
        self.plot_widget.installEventFilter(self)
        self.plot_widget.getPlotItem().sigRangeChanged.connect(
            lambda *_: self._refresh_annotation_labels()
        )

        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
        self.plot_widget.addItem(self.cursor_line)
        self.cursor_line.hide()

        self.status_label = QLabel("Load a video to view synchronized time series data.")

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self.plot_widget)
        layout.addWidget(self._build_annotation_hint())
        layout.addWidget(self.status_label)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)

    def channel_controls(self) -> QWidget:
        """Expose the channel selection controls for external layouts."""

        return self._controls_container

    def _build_annotation_hint(self) -> QWidget:
        hint = QLabel(
            "Click and drag to add an annotation. Click to select, press Delete to remove."
        )
        hint.setWordWrap(True)
        return hint

    def load_for_video(self, video_path: Optional[Path]) -> None:
        """Load and plot the time series associated with the provided video."""

        self._clear_plot()
        self._times = None
        self.raw = None
        self._selected_channels.clear()
        self.channel_list.clear()
        self._reset_zoom()
        self._last_ts_path = None
        self._annotation_file = None
        self._clear_annotations()

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
        self._ensure_view_range(0.0)
        self._load_annotations(ts_path.with_name("ear_eog.csv"))

    def set_processed_root(self, root: Path) -> None:
        self.processed_root = root

    def update_cursor_time(self, seconds: float) -> None:
        """Keep the current time centered under a fixed cursor."""

        if self._times is None or self._times.size == 0:
            self.cursor_line.hide()
            return

        self._suppress_cursor_signal = True
        self._ensure_view_range(max(0.0, seconds))
        self._suppress_cursor_signal = False

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

    def _ensure_view_range(self, target_time: float, *, from_user: bool = False) -> None:
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
        if from_user and not self._suppress_cursor_signal:
            self.cursor_time_changed.emit(self._last_cursor_time)

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
        self._refresh_annotation_labels()

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

            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                clicked_time = self._time_at_position(event.pos())
                self._annotation_drag_start = clicked_time
                self.plot_widget.setFocus()
                if clicked_time is not None:
                    self._update_cursor_time_from_user(clicked_time)
                return True

            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                end_time = self._time_at_position(event.pos())
                self._handle_annotation_mouse_release(end_time)
                return True

        if obj is self.plot_widget and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self._delete_selected_annotation()
                return True

        return super().eventFilter(obj, event)

    def _time_at_position(self, pos) -> Optional[float]:
        view_box = self.plot_widget.getPlotItem().getViewBox()
        if view_box is None:
            return None

        scene_pos = self.plot_widget.mapToScene(pos)
        view_pos = view_box.mapSceneToView(scene_pos)
        return view_pos.x()

    # Annotation handling
    def _load_annotations(self, csv_path: Path) -> None:
        self._annotation_file = csv_path
        self._clear_annotations()

        if self._times is None or self._times.size == 0:
            return

        try:
            entries = load_annotations(csv_path)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        max_time = float(self._times[-1])
        clamped_entries = [clamp_entry(entry, max_time) for entry in entries]
        for entry in clamped_entries:
            self._add_annotation(entry)

        if clamped_entries:
            self.status_label.setText(
                f"Loaded {len(clamped_entries)} annotation(s) from {csv_path}."
            )
        else:
            self.status_label.setText("No annotations found. Drag on the plot to add one.")

    def _clear_annotations(self) -> None:
        for region in self._annotation_regions.values():
            self.plot_widget.removeItem(region)
        for label in self._annotation_labels.values():
            self.plot_widget.removeItem(label)
        self._annotation_entries.clear()
        self._annotation_regions.clear()
        self._annotation_labels.clear()
        self._next_annotation_id = 1
        self._selected_annotation = None

    def _add_annotation(self, entry: AnnotationEntry) -> str:
        annotation_id = f"annotation-{self._next_annotation_id}"
        self._next_annotation_id += 1
        self._annotation_entries[annotation_id] = entry

        region = pg.LinearRegionItem(
            values=(entry.onset, entry.end),
            brush=pg.mkBrush(self._annotation_color_for_label(entry.description, alpha=60)),
            movable=True,
        )
        region.setZValue(2)
        region.sigRegionChangeFinished.connect(
            lambda _: self._on_region_change_finished(annotation_id)
        )
        region.sigRegionChangeStarted.connect(lambda _: self._select_annotation(annotation_id))
        self.plot_widget.addItem(region)
        self._annotation_regions[annotation_id] = region

        label_item = pg.TextItem(
            entry.description,
            anchor=(0.5, 1.0),
            fill=pg.mkBrush(255, 255, 255, 180),
        )
        label_item.setZValue(3)
        self.plot_widget.addItem(label_item)
        self._annotation_labels[annotation_id] = label_item
        self._position_annotation_label(annotation_id)
        self._select_annotation(annotation_id)
        return annotation_id

    def _annotation_color_for_label(self, label: str, alpha: int = 120):
        palette = CHANNEL_PALETTE or ["#00838f", "#43a047", "#6a1b9a", "#ffb300"]
        index = abs(hash(label)) % len(palette)
        color = pg.mkColor(palette[index])
        color.setAlpha(alpha)
        return color

    def _position_annotation_label(self, annotation_id: str) -> None:
        label_item = self._annotation_labels.get(annotation_id)
        region = self._annotation_regions.get(annotation_id)
        if not label_item or not region:
            return

        center_time = sum(region.getRegion()) / 2
        y_range = self.plot_widget.viewRange()[1]
        y_pos = y_range[1] if y_range else 0.0
        label_item.setPos(center_time, y_pos)

    def _refresh_annotation_labels(self) -> None:
        for annotation_id in list(self._annotation_labels.keys()):
            self._position_annotation_label(annotation_id)

    def _on_region_change_finished(self, annotation_id: str) -> None:
        if self._times is None or self._times.size == 0:
            return

        entry = self._annotation_entries.get(annotation_id)
        region = self._annotation_regions.get(annotation_id)
        if entry is None or region is None:
            return

        start, end = sorted(region.getRegion())
        max_time = float(self._times[-1])
        bounded_start = max(0.0, min(start, max_time))
        bounded_end = max(bounded_start + self._min_annotation_duration, min(end, max_time))

        region.setRegion((bounded_start, bounded_end))
        self._annotation_entries[annotation_id] = AnnotationEntry(
            onset=bounded_start,
            duration=bounded_end - bounded_start,
            description=entry.description,
        )
        self._position_annotation_label(annotation_id)

    def _select_annotation(self, annotation_id: Optional[str]) -> None:
        for key, region in self._annotation_regions.items():
            width = 2.5 if key == annotation_id else 1.5
            region.setPen(pg.mkPen("k", width=width))
        self._selected_annotation = annotation_id

    def _handle_annotation_mouse_release(self, end_time: Optional[float]) -> None:
        start_time = self._annotation_drag_start
        self._annotation_drag_start = None

        if end_time is None or start_time is None:
            return

        if self._times is None or self._times.size == 0:
            return

        max_time = float(self._times[-1])
        span = abs(end_time - start_time)
        if span < self._min_annotation_duration:
            self._select_annotation_at_time(end_time)
            self._update_cursor_time_from_user(end_time)
            return

        onset = max(0.0, min(start_time, end_time))
        duration = max(self._min_annotation_duration, min(max_time - onset, span))
        new_entry = AnnotationEntry(
            onset=onset,
            duration=duration,
            description=self._current_annotation_label(),
        )
        try:
            clamped = clamp_entry(new_entry, max_time)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        self._add_annotation(clamped)
        self._last_used_label = clamped.description
        self.status_label.setText(
            f"Added annotation '{clamped.description}' at {clamped.onset:.3f}s for {clamped.duration:.3f}s."
        )

    def _current_annotation_label(self) -> str:
        text = self.annotation_label_input.text().strip()
        if text:
            return text
        return self._last_used_label

    def _select_annotation_at_time(self, target_time: float) -> None:
        candidates = []
        for annotation_id, entry in self._annotation_entries.items():
            if entry.onset <= target_time <= entry.end:
                candidates.append((entry.onset, annotation_id))
        if not candidates:
            self._select_annotation(None)
            return

        _, annotation_id = min(candidates, key=lambda item: item[0])
        self._select_annotation(annotation_id)

    def _delete_selected_annotation(self) -> None:
        annotation_id = getattr(self, "_selected_annotation", None)
        if not annotation_id:
            return

        region = self._annotation_regions.pop(annotation_id, None)
        label = self._annotation_labels.pop(annotation_id, None)
        self._annotation_entries.pop(annotation_id, None)
        if region:
            self.plot_widget.removeItem(region)
        if label:
            self.plot_widget.removeItem(label)
        self._selected_annotation = None
        self.status_label.setText("Annotation deleted.")

    def _save_annotations(self) -> None:
        if self._times is None or self._times.size == 0:
            self.status_label.setText("Load a time series before saving annotations.")
            return

        if not self._annotation_file:
            self.status_label.setText("No annotation file target available.")
            return

        max_time = float(self._times[-1])
        entries = list(self._annotation_entries.values())
        try:
            validated = validate_annotations(entries, max_time)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        save_annotations(self._annotation_file, sorted(validated, key=lambda entry: entry.onset))
        self.status_label.setText(f"Saved {len(validated)} annotation(s) to {self._annotation_file}.")

    def _update_cursor_time_from_user(self, seconds: float) -> None:
        clamped = seconds
        if self._times is not None and self._times.size > 0:
            clamped = max(0.0, min(seconds, float(self._times[-1])))
        self._ensure_view_range(clamped, from_user=True)
