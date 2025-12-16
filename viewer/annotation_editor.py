"""Interactive annotation editor for MNE time-series data."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from viewer.utils import frame_to_pixmap
from viewer.video_handler import VideoHandler

REQUIRED_COLUMNS = ("onset", "duration", "description")
DEFAULT_LABEL = "HB_CL"
MIN_DURATION_SECONDS = 0.01


@dataclass
class Annotation:
    """Container for an annotation interval."""

    onset: float
    duration: float
    description: str

    @property
    def end(self) -> float:
        return self.onset + self.duration

    def clamped(self, max_time: float) -> "Annotation":
        """Return a copy that fits within the provided bounds."""
        start = max(0.0, min(self.onset, max_time))
        end = max(start + MIN_DURATION_SECONDS, min(self.end, max_time))
        return Annotation(start, end - start, self.description)


class AnnotationViewBox(pg.ViewBox):
    """Custom ViewBox that turns left-drag gestures into interval selections."""

    sigIntervalSelected = QtCore.pyqtSignal(float, float)
    sigTimeClicked = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent, enableMenu=False)
        self._drag_start_time: Optional[float] = None
        self._pan_last: Optional[QtCore.QPointF] = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.LeftButton:
            pos = self.mapToView(ev.pos())
            if ev.isStart():
                self._drag_start_time = pos.x()
                ev.accept()
            elif ev.isFinish() and self._drag_start_time is not None:
                start = self._drag_start_time
                end = pos.x()
                ev.accept()
                self._drag_start_time = None
                self.sigIntervalSelected.emit(start, end)
            else:
                ev.accept()
        elif ev.button() == Qt.RightButton:
            view_pos = self.mapToView(ev.pos())
            if ev.isStart():
                self._pan_last = view_pos
                ev.accept()
            else:
                if self._pan_last is not None:
                    delta = self._pan_last - view_pos
                    self.translateBy(x=delta.x(), y=delta.y())
                self._pan_last = view_pos
                ev.accept()
            if ev.isFinish():
                self._pan_last = None
        else:
            super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            pos = self.mapToView(ev.pos())
            self.sigTimeClicked.emit(pos.x())
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class AnnotationRegion(pg.LinearRegionItem):
    """Annotation span with clickable selection and labels."""

    clicked = QtCore.pyqtSignal(object)

    def __init__(self, annotation: Annotation, color: str, **kwargs):
        self._base_color = pg.mkColor(color)
        super().__init__(
            values=(annotation.onset, annotation.end),
            brush=pg.mkBrush(color=self._base_color, style=Qt.Dense4Pattern),
            pen=pg.mkPen(self._base_color, width=2),
            **kwargs,
        )
        self.annotation = annotation
        self.setMovable(True)
        self.setZValue(10)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit(self)
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def set_selected(self, selected: bool) -> None:
        pen = pg.mkPen(self._base_color, width=3 if selected else 2)
        self.setPen(pen)


class FramePanel(QWidget):
    """Small video frame viewer synchronized to a provided timestamp."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.video_handler = VideoHandler()
        self.frame_label = QLabel("No video loaded")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(320, 180)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.info_label = QLabel("Time: -s | Frame: -")

        layout = QVBoxLayout()
        layout.addWidget(self.frame_label)
        layout.addWidget(self.info_label)
        self.setLayout(layout)

    def load_video(self, path: Path) -> bool:
        self.video_handler.release()
        if not path.exists():
            self._set_status(f"Video not found: {path.name}")
            return False
        if not self.video_handler.load(path):
            self._set_status(f"Unable to open video: {path.name}")
            return False
        self._set_status(
            f"Loaded {path.name} ({self.video_handler.frame_count} frames @ {self.video_handler.fps:.2f} fps)"
        )
        return True

    def update_time(self, seconds: float) -> None:
        if not self.video_handler.capture or self.video_handler.fps <= 0:
            self.frame_label.setText("No video loaded")
            self._set_status(f"Time: {seconds:.3f}s | Frame: - (no video)")
            return

        frame_index = int(seconds * self.video_handler.fps)
        clamped_index = self.video_handler.clamp_index(frame_index)
        frame = self.video_handler.read_frame(clamped_index)
        if frame is None:
            self.frame_label.setText("Frame unavailable")
            self._set_status(f"Time: {seconds:.3f}s | Frame: unavailable")
            return

        pixmap = frame_to_pixmap(frame)
        if pixmap:
            self.frame_label.setPixmap(pixmap.scaled(self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._set_status(f"Time: {seconds:.3f}s | Frame: {clamped_index}")

    def _set_status(self, message: str) -> None:
        self.info_label.setText(message)


class AnnotationEditorWindow(QMainWindow):
    """Main window for editing time-interval annotations."""

    COLOR_CYCLE = [
        "#0066cc",
        "#009688",
        "#8e24aa",
        "#ef6c00",
        "#d81b60",
        "#3949ab",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MNE Annotation Editor")

        self.raw: Optional[mne.io.BaseRaw] = None
        self.times: Optional[np.ndarray] = None
        self.sample_rate: float = 0.0
        self.duration: float = 0.0
        self.annotations: List[Annotation] = []
        self.regions: Dict[int, Tuple[AnnotationRegion, pg.TextItem]] = {}
        self.selected_region_id: Optional[int] = None
        self.last_label: str = DEFAULT_LABEL
        self._color_index = 0
        self._current_time: float = 0.0
        self.max_points = 12000
        self._plotted_curves: List[pg.PlotDataItem] = []

        self.view_box = AnnotationViewBox()
        self.view_box.sigIntervalSelected.connect(self._handle_interval_drag)
        self.view_box.sigTimeClicked.connect(self._handle_time_clicked)

        plot_item = pg.PlotItem(viewBox=self.view_box)
        plot_item.showGrid(x=True, y=True, alpha=0.4)
        plot_item.setLabel("bottom", "Time", units="s")
        plot_item.setLabel("left", "Channels (offset)")
        self.plot_widget = pg.PlotWidget(plotItem=plot_item, background="w")
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view_box.sigXRangeChanged.connect(self._on_view_range_changed)
        self.view_box.sigYRangeChanged.connect(self._refresh_label_positions)

        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
        self.plot_widget.addItem(self.cursor_line)
        self.cursor_line.hide()

        self.status_label = QLabel("Load an MNE recording and annotations to begin.")
        self.file_status_label = QLabel("")
        self.label_selector = QComboBox()
        self.label_selector.setEditable(True)
        self.label_selector.setInsertPolicy(QComboBox.NoInsert)
        self.label_selector.setEditText(DEFAULT_LABEL)

        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.annotation_list.itemSelectionChanged.connect(self._handle_list_selection)

        self.frame_panel = FramePanel()

        self.raw_path_input = QLineEdit()
        self.annotation_path_input = QLineEdit()
        self.video_path_input = QLineEdit()

        self._build_ui()
        self._seed_default_paths()

    # UI construction
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        layout.addWidget(self._build_path_controls())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        layout.addWidget(self.status_label)
        layout.addWidget(self.file_status_label)

    def _build_path_controls(self) -> QWidget:
        group = QGroupBox("Recording & Annotation Files")
        grid = QGridLayout()

        raw_browse = QPushButton("Browse")
        ann_browse = QPushButton("Browse")
        video_browse = QPushButton("Browse")

        raw_browse.clicked.connect(lambda: self._pick_file(self.raw_path_input, "FIF files (*.fif)"))
        ann_browse.clicked.connect(lambda: self._pick_file(self.annotation_path_input, "CSV files (*.csv)"))
        video_browse.clicked.connect(lambda: self._pick_file(self.video_path_input, "Video files (*.mov *.mp4 *.avi)"))

        load_button = QPushButton("Load")
        load_button.clicked.connect(self._load_sources)

        save_button = QPushButton("Save annotations")
        save_button.clicked.connect(self._save_annotations)

        grid.addWidget(QLabel("Raw FIF:"), 0, 0)
        grid.addWidget(self.raw_path_input, 0, 1)
        grid.addWidget(raw_browse, 0, 2)

        grid.addWidget(QLabel("Annotations CSV:"), 1, 0)
        grid.addWidget(self.annotation_path_input, 1, 1)
        grid.addWidget(ann_browse, 1, 2)

        grid.addWidget(QLabel("Video (optional):"), 2, 0)
        grid.addWidget(self.video_path_input, 2, 1)
        grid.addWidget(video_browse, 2, 2)

        grid.addWidget(load_button, 3, 1)
        grid.addWidget(save_button, 3, 2)

        group.setLayout(grid)
        return group

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        label_group = QGroupBox("Labels")
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Last / default:"))
        label_layout.addWidget(self.label_selector)
        label_group.setLayout(label_layout)

        instructions = QLabel(
            "Left-drag to add an interval.\n"
            "Click an annotation to select, then press Delete/Backspace to remove.\n"
            "Drag annotation edges to resize. Zoom with mouse wheel; pan with right-drag."
        )
        instructions.setWordWrap(True)

        layout.addWidget(label_group)
        layout.addWidget(instructions)

        ann_group = QGroupBox("Current annotations")
        ann_layout = QVBoxLayout()
        ann_layout.addWidget(self.annotation_list)
        ann_group.setLayout(ann_layout)

        layout.addWidget(ann_group)
        layout.addStretch()
        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(self.plot_widget, stretch=3)
        layout.addWidget(self.frame_panel, stretch=1)

        return container

    # Loading helpers
    def _seed_default_paths(self) -> None:
        base = Path.cwd()
        candidates = [base, base / "test_data"]

        for candidate_base in candidates:
            fif = candidate_base / "ear_eog.fif"
            csv_path = candidate_base / "ear_eog.csv"
            mov = candidate_base / "file_example_MOV_480_700kB.mov"
            if fif.exists():
                self.raw_path_input.setText(str(fif))
            if csv_path.exists():
                self.annotation_path_input.setText(str(csv_path))
            if mov.exists():
                self.video_path_input.setText(str(mov))
            if fif.exists() or csv_path.exists() or mov.exists():
                break

    def _pick_file(self, target: QLineEdit, filter_text: str) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Select file", str(Path.cwd()), filter_text)
        if selected:
            target.setText(selected)

    def _load_sources(self) -> None:
        raw_path = Path(self.raw_path_input.text().strip())
        annotations_path = Path(self.annotation_path_input.text().strip())
        video_path_text = self.video_path_input.text().strip()

        if not raw_path.exists():
            self._set_status(f"Raw file not found at {raw_path}")
            return
        if not annotations_path.exists():
            self._set_status(f"Annotation CSV not found at {annotations_path}")
            return

        try:
            self._load_raw(raw_path)
        except Exception as exc:  # pragma: no cover - guardrails
            self._set_status(f"Failed to load raw data: {exc}")
            return

        try:
            self.annotations = self._load_annotation_csv(annotations_path)
            self._apply_loaded_annotations()
            self._set_status(f"Loaded {len(self.annotations)} annotation(s) from {annotations_path.name}")
        except Exception as exc:  # pragma: no cover - guardrails
            self._set_status(f"Failed to load annotations: {exc}")
            return

        if video_path_text:
            self.frame_panel.load_video(Path(video_path_text))
        else:
            self.frame_panel.video_handler.release()
            self.frame_panel.frame_label.setText("No video loaded")
            self.frame_panel._set_status(f"Time: {self._current_time:.3f}s | Frame: - (no video)")

        self.file_status_label.setText(
            f"Recording: {raw_path.name} | Annotations: {annotations_path.name} | Video: {Path(video_path_text).name if video_path_text else 'None'}"
        )

    def _load_raw(self, path: Path) -> None:
        self.raw = mne.io.read_raw_fif(str(path), preload=True, verbose="ERROR")
        self.times = self.raw.times
        self.sample_rate = float(self.raw.info["sfreq"])
        self.duration = float(self.times[-1]) if self.times.size else 0.0
        self._color_index = 0
        self._plot_channels()
        self._clear_regions()
        self._current_time = 0.0
        self._update_cursor()
        self.frame_panel.update_time(self._current_time)
        self.plot_widget.enableAutoRange()
        self.cursor_line.show()

    def _plot_channels(self) -> None:
        if self.raw is None or self.times is None:
            return

        for curve in self._plotted_curves:
            self.plot_widget.removeItem(curve)
        self._plotted_curves.clear()

        data = self.raw.get_data()
        decimation = max(1, data.shape[1] // self.max_points)
        if decimation > 1:
            data = data[:, ::decimation]
            times = self.times[::decimation]
        else:
            times = self.times

        peak = np.nanmax(np.abs(data)) or 1.0
        spacing = peak * 2
        offsets = np.arange(data.shape[0]) * spacing

        for idx, channel in enumerate(data):
            curve = self.plot_widget.plot(
                times, channel + offsets[idx], pen=pg.mkPen(self.COLOR_CYCLE[idx % len(self.COLOR_CYCLE)], width=1)
            )
            curve.setDownsampling(auto=True, method="peak")
            self._plotted_curves.append(curve)

        self.plot_widget.enableAutoRange(y=True)

    def _clear_regions(self) -> None:
        for region, label in self.regions.values():
            self.plot_widget.removeItem(region)
            self.plot_widget.removeItem(label)
        self.regions.clear()
        self.annotation_list.clear()
        self.selected_region_id = None

    def _apply_loaded_annotations(self) -> None:
        self._clear_regions()
        existing_labels = set()
        for ann in self.annotations:
            existing_labels.add(ann.description)
            self._add_region(ann)
        if self.annotations:
            self.last_label = self.annotations[-1].description
        for label in sorted(existing_labels):
            if self.label_selector.findText(label) == -1:
                self.label_selector.addItem(label)
        self.label_selector.setEditText(self.last_label or DEFAULT_LABEL)

    def _load_annotation_csv(self, path: Path) -> List[Annotation]:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
                raise ValueError(f"Expected columns {REQUIRED_COLUMNS}, found {reader.fieldnames}")

            annotations: List[Annotation] = []
            for row in reader:
                onset = float(row["onset"])
                duration = float(row["duration"])
                description = row["description"]
                ann = Annotation(onset, duration, description).clamped(self.duration)
                if ann.duration <= 0:
                    continue
                annotations.append(ann)
            return annotations

    # Annotation handling
    def _add_region(self, annotation: Annotation) -> None:
        color = self.COLOR_CYCLE[self._color_index % len(self.COLOR_CYCLE)]
        self._color_index += 1

        region = AnnotationRegion(annotation, color=color, bounds=(0, self.duration))
        region.sigRegionChangeFinished.connect(lambda: self._handle_region_change(region))
        region.clicked.connect(lambda _: self._select_region_by_item(region))
        self.plot_widget.addItem(region)

        label_item = pg.TextItem(annotation.description, color="k", anchor=(0.5, 1.2))
        label_item.setZValue(11)
        self.plot_widget.addItem(label_item)
        self.regions[id(region)] = (region, label_item)
        self._update_label_position(region, label_item)

        item = QListWidgetItem(self._annotation_text(annotation))
        item.setData(Qt.UserRole, id(region))
        self.annotation_list.addItem(item)

    def _annotation_text(self, annotation: Annotation) -> str:
        return f"{annotation.description}: {annotation.onset:.3f}s → {annotation.end:.3f}s (dur {annotation.duration:.3f}s)"

    def _handle_interval_drag(self, start: float, end: float) -> None:
        if self.raw is None:
            self._set_status("Load data before adding annotations.")
            return

        start_time = max(0.0, min(start, end))
        end_time = max(start_time, max(start, end))
        duration = end_time - start_time

        if duration < MIN_DURATION_SECONDS:
            self._handle_time_clicked(start_time)
            return

        start_time, end_time = self._clamped_interval(start_time, end_time)
        if end_time - start_time < MIN_DURATION_SECONDS:
            self._set_status("Interval is too small after clamping to recording bounds.")
            return

        label, ok = self._prompt_for_label()
        if not ok or not label.strip():
            self._set_status("Annotation creation cancelled.")
            return
        self.last_label = label.strip()
        if self.label_selector.findText(self.last_label) == -1:
            self.label_selector.addItem(self.last_label)
        self.label_selector.setEditText(self.last_label)

        annotation = Annotation(start_time, end_time - start_time, self.last_label)
        self.annotations.append(annotation)
        self._add_region(annotation)
        self._set_status(f"Added annotation {self.last_label} from {start_time:.3f}s to {end_time:.3f}s")

    def _prompt_for_label(self) -> Tuple[str, bool]:
        text = self.label_selector.currentText().strip() or self.last_label or DEFAULT_LABEL
        label, ok = QInputDialog.getText(self, "Annotation label", "Label:", text=text)
        return label.strip(), bool(ok)

    def _handle_time_clicked(self, seconds: float) -> None:
        if self.duration <= 0:
            return
        self._current_time = max(0.0, min(seconds, self.duration))
        self._update_cursor()
        self.frame_panel.update_time(self._current_time)
        self._set_status(f"Cursor at {self._current_time:.3f}s")

    def _on_view_range_changed(self, _view_box, x_range):
        center = sum(x_range) / 2
        self._current_time = max(0.0, min(center, self.duration))
        self._update_cursor_line_position()
        self.frame_panel.update_time(self._current_time)
        self._refresh_label_positions()

    def _update_cursor(self) -> None:
        self._ensure_time_visible(self._current_time)
        self._update_cursor_line_position()

    def _update_cursor_line_position(self) -> None:
        self.cursor_line.setPos(self._current_time)
        self.cursor_line.show()

    def _ensure_time_visible(self, target_time: float) -> None:
        view_range = self.plot_widget.viewRange()[0]
        x_min, x_max = view_range
        if target_time < x_min or target_time > x_max:
            half_span = (x_max - x_min) / 2
            new_min = max(0.0, target_time - half_span)
            new_max = min(self.duration, target_time + half_span)
            self.plot_widget.setXRange(new_min, new_max, padding=0)

    def _handle_region_change(self, region: AnnotationRegion) -> None:
        start, end = region.getRegion()
        start, end = self._clamped_interval(start, end)
        if end - start < MIN_DURATION_SECONDS:
            end = start + MIN_DURATION_SECONDS
        region.setRegion((start, end))
        region.annotation.onset = start
        region.annotation.duration = end - start
        label_item = self.regions[id(region)][1]
        self._update_label_position(region, label_item)
        self._refresh_list_items()
        self._set_status(
            f"Updated {region.annotation.description}: {region.annotation.onset:.3f}s → {region.annotation.end:.3f}s"
        )

    def _update_label_position(self, region: AnnotationRegion, label_item: pg.TextItem) -> None:
        start, end = region.getRegion()
        center = (start + end) / 2
        view_range = self.plot_widget.viewRange()
        y_range = view_range[1] if view_range else (0, 1)
        y_max = y_range[1] if isinstance(y_range, (list, tuple)) else 1
        label_item.setPos(center, y_max)
        label_item.setText(region.annotation.description)

    def _refresh_label_positions(self, *_) -> None:
        for region, label_item in self.regions.values():
            self._update_label_position(region, label_item)

    def _select_region_by_item(self, region: AnnotationRegion) -> None:
        target_id = id(region)
        self.selected_region_id = target_id
        for row in range(self.annotation_list.count()):
            item = self.annotation_list.item(row)
            item_id = item.data(Qt.UserRole)
            item.setSelected(item_id == target_id)
        self._highlight_selected()

    def _handle_list_selection(self) -> None:
        items = self.annotation_list.selectedItems()
        if not items:
            self.selected_region_id = None
        else:
            self.selected_region_id = items[0].data(Qt.UserRole)
        self._highlight_selected()

    def _highlight_selected(self) -> None:
        for region_id, (region, _) in self.regions.items():
            region.set_selected(region_id == self.selected_region_id)

    def _refresh_list_items(self) -> None:
        for idx in range(self.annotation_list.count()):
            item = self.annotation_list.item(idx)
            region_id = item.data(Qt.UserRole)
            region, _ = self.regions.get(region_id, (None, None))  # type: ignore[assignment]
            if region:
                item.setText(self._annotation_text(region.annotation))

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected_region_id:
            self._delete_selected()
        else:
            super().keyPressEvent(event)

    def _delete_selected(self) -> None:
        if self.selected_region_id is None:
            return
        region, label_item = self.regions.pop(self.selected_region_id, (None, None))
        if region:
            self.annotations = [ann for ann in self.annotations if ann is not region.annotation]
            self.plot_widget.removeItem(region)
        if label_item:
            self.plot_widget.removeItem(label_item)
        for row in range(self.annotation_list.count()):
            item = self.annotation_list.item(row)
            if item.data(Qt.UserRole) == self.selected_region_id:
                self.annotation_list.takeItem(row)
                break
        self.selected_region_id = None
        self._set_status("Annotation removed.")

    def _save_annotations(self) -> None:
        path_text = self.annotation_path_input.text().strip()
        if not path_text:
            self._set_status("Provide an annotation CSV path to save.")
            return
        path = Path(path_text)
        cleaned = [ann.clamped(self.duration) for ann in self.annotations if ann.duration > 0]
        cleaned.sort(key=lambda ann: ann.onset)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(REQUIRED_COLUMNS)
            for ann in cleaned:
                writer.writerow([f"{ann.onset:.9f}", f"{ann.duration:.9f}", ann.description])
        self._set_status(f"Saved {len(cleaned)} annotation(s) to {path.name}")

    def _clamped_interval(self, start: float, end: float) -> Tuple[float, float]:
        start_clamped = max(0.0, min(start, self.duration))
        end_clamped = max(0.0, min(end, self.duration))
        if end_clamped < start_clamped:
            start_clamped, end_clamped = end_clamped, start_clamped
        return start_clamped, end_clamped

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)


def launch_annotation_editor() -> None:
    app = QApplication([])
    window = AnnotationEditorWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    launch_annotation_editor()
