"""Murat 2018 EDF annotation review interface."""
from __future__ import annotations

import csv
import math
import os
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import yaml
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from time_series import TimeSeriesViewer

DEFAULT_MURAT_ROOT = Path(r"D:\dataset\murat_2018")
SESSION_FILENAME = "Murat2018Viewer.yaml"
BLINKER_PICKLE = "blinker_results.pkl"
DEFAULT_SESSION_STATE = {
    "stop_position": 0.0,
    "status": "Pending",
    "remark": "",
    "remarks": [],
}


@dataclass(frozen=True)
class MuratRecording:
    """Files associated with one Murat 2018 recording."""

    subject_id: str
    folder: Path
    edf_path: Path

    @property
    def csv_path(self) -> Path:
        return self.edf_path.with_suffix(".csv")

    @property
    def blinker_path(self) -> Path:
        return self.folder / BLINKER_PICKLE

    @property
    def session_path(self) -> Path:
        return self.folder / SESSION_FILENAME


def ensure_murat_session_file(folder: Path) -> Path:
    """Create the Murat review session YAML in a recording folder if missing."""

    session_path = folder / SESSION_FILENAME
    if not session_path.exists():
        with session_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(DEFAULT_SESSION_STATE, handle, sort_keys=False)
    return session_path


class Murat2018Viewer(QMainWindow):
    """Standalone EDF review UI for the Murat 2018 dataset."""

    DEFAULT_STEP_SECONDS = 1.0
    RECORDINGS_PANEL_WIDTH = 280
    RECORDINGS_PANEL_MAX_WIDTH = 320
    DETACHED_PANEL_MAX_WIDTH = 16777215

    def __init__(self, dataset_root: Path = DEFAULT_MURAT_ROOT) -> None:
        super().__init__()
        self.setWindowTitle("Murat 2018 Annotation Reviewer")
        self.dataset_root = dataset_root
        self.recordings: list[MuratRecording] = []
        self.current_recording: Optional[MuratRecording] = None
        self.status_value = "Pending"

        self.time_series_viewer = TimeSeriesViewer()

        self._annotation_play_timer = QTimer(self)
        self._annotation_play_timer.timeout.connect(self._annotation_play_tick)

        self._forward_play_timer = QTimer(self)
        self._forward_play_timer.timeout.connect(self._forward_play_tick)

        self._setup_ui()
        self._setup_shortcuts()
        self._update_review_controls(False)
        self.directory_input.setText(str(self.dataset_root))
        self._scan_directory()

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self._build_directory_controls(main_layout)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        self.side_tabs = self._build_side_tabs()
        self.time_series_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.main_splitter.addWidget(self.side_tabs)
        self.main_splitter.addWidget(self.time_series_viewer)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.main_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        QTimer.singleShot(0, self._apply_side_tab_mode)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_side_tab_mode()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_side_tab_mode)

    def _build_directory_controls(self, parent_layout: QVBoxLayout) -> None:
        directory_group = QGroupBox("Murat 2018 Dataset Directory")
        directory_layout = QHBoxLayout()
        directory_group.setLayout(directory_layout)

        self.directory_input = QLineEdit()
        self.directory_input.setPlaceholderText(r"D:\dataset\murat_2018")
        self.directory_input.returnPressed.connect(self._scan_directory)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_directory)
        scan_button = QPushButton("Rescan")
        scan_button.clicked.connect(self._scan_directory)

        directory_layout.addWidget(QLabel("Root:"))
        directory_layout.addWidget(self.directory_input)
        directory_layout.addWidget(browse_button)
        directory_layout.addWidget(scan_button)
        parent_layout.addWidget(directory_group)

    def _build_side_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        recordings_tab = QWidget()
        recordings_layout = QVBoxLayout()
        recordings_layout.setContentsMargins(0, 0, 0, 0)
        recordings_layout.setSpacing(6)
        recordings_layout.addWidget(self._build_recording_list_panel())
        recordings_layout.addWidget(self._build_review_controls())
        recordings_layout.setStretch(0, 1)
        recordings_layout.setStretch(1, 0)
        recordings_tab.setLayout(recordings_layout)

        channels_tab = QWidget()
        channels_layout = QVBoxLayout()
        channels_layout.setContentsMargins(0, 0, 0, 0)
        channels_layout.setSpacing(6)
        channels_layout.addWidget(self.time_series_viewer.channel_controls())
        channels_layout.addStretch()
        channels_tab.setLayout(channels_layout)

        summary_tab = self._build_summary_tab()

        tabs.addTab(recordings_tab, "Recordings")
        tabs.addTab(channels_tab, "Channels")
        tabs.addTab(summary_tab, "Summary")
        tabs.currentChanged.connect(self._apply_side_tab_mode)
        return tabs

    def _apply_side_tab_mode(self, *_: object) -> None:
        if self.side_tabs.currentIndex() == 0:
            self.time_series_viewer.show()
            self.side_tabs.setMaximumWidth(self.RECORDINGS_PANEL_MAX_WIDTH)
            available_width = max(1, self.main_splitter.width())
            side_width = min(self.RECORDINGS_PANEL_WIDTH, max(220, available_width // 4))
            self.main_splitter.setSizes([side_width, max(1, available_width - side_width)])
            return

        self.side_tabs.setMaximumWidth(self.DETACHED_PANEL_MAX_WIDTH)
        self.time_series_viewer.hide()
        self.main_splitter.setSizes([max(1, self.main_splitter.width()), 0])

    def _build_recording_list_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(QLabel("Discovered EDF Recordings"))
        self.recording_list = QListWidget()
        self.recording_list.itemSelectionChanged.connect(self._load_selected_recording)
        layout.addWidget(self.recording_list)
        return container

    def _build_review_controls(self) -> QGroupBox:
        control_group = QGroupBox("Review Controls")
        layout = QGridLayout()
        control_group.setLayout(layout)

        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("Time in seconds")
        self.time_input.returnPressed.connect(self._search_time)
        self.time_search_button = QPushButton("Go")
        self.time_search_button.clicked.connect(self._search_time)

        self.step_seconds_input = QDoubleSpinBox()
        self.step_seconds_input.setRange(0.01, 3600.0)
        self.step_seconds_input.setDecimals(3)
        self.step_seconds_input.setSingleStep(0.01)
        self.step_seconds_input.setValue(self.DEFAULT_STEP_SECONDS)

        self.left_button = QPushButton("Left")
        self.right_button = QPushButton("Right")
        self.left_button.clicked.connect(lambda: self._step_time(-1))
        self.right_button.clicked.connect(lambda: self._step_time(1))

        self.status_dropdown = QComboBox()
        self.status_dropdown.addItems(["Pending", "Ongoing", "Complete", "Issue"])
        self.status_dropdown.currentTextChanged.connect(self._on_status_changed)

        self.save_annotations_button = QPushButton("Save annotations")
        self.save_annotations_button.clicked.connect(self.time_series_viewer.save_annotations)

        self.current_time_label = QLabel("Current time: -")

        layout.addWidget(QLabel("Time (s):"), 0, 0)
        layout.addWidget(self.time_input, 0, 1)
        layout.addWidget(self.time_search_button, 0, 2)
        layout.addWidget(QLabel("Step (s):"), 1, 0)
        layout.addWidget(self.step_seconds_input, 1, 1, 1, 2)
        layout.addWidget(self.left_button, 2, 0)
        layout.addWidget(self.right_button, 2, 1)
        layout.addWidget(self.save_annotations_button, 2, 2)
        layout.addWidget(QLabel("Status:"), 3, 0)
        layout.addWidget(self.status_dropdown, 3, 1, 1, 2)
        layout.addWidget(self.current_time_label, 4, 0, 1, 3)

        play_layout = QHBoxLayout()
        self.play_button = QPushButton("Play Annotations")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self._toggle_annotation_play)
        self.play_speed_spinbox = QDoubleSpinBox()
        self.play_speed_spinbox.setRange(0, 30.0)
        self.play_speed_spinbox.setValue(2.0)
        self.play_speed_spinbox.setSingleStep(0.1)
        self.play_speed_spinbox.setSuffix(" s")
        self.play_speed_spinbox.setToolTip("Seconds between each annotation jump")
        play_layout.addWidget(self.play_button)
        play_layout.addWidget(QLabel("Speed:"))
        play_layout.addWidget(self.play_speed_spinbox)
        layout.addLayout(play_layout, 5, 0, 1, 3)

        forward_play_layout = QHBoxLayout()
        self.forward_play_button = QPushButton("Forward Play")
        self.forward_play_button.setCheckable(True)
        self.forward_play_button.clicked.connect(self._toggle_forward_play_button)
        self.forward_play_speed_spinbox = QDoubleSpinBox()
        self.forward_play_speed_spinbox.setRange(0.016, 30.0)
        self.forward_play_speed_spinbox.setValue(0.1)
        self.forward_play_speed_spinbox.setSingleStep(0.1)
        self.forward_play_speed_spinbox.setSuffix(" s")
        self.forward_play_speed_spinbox.setToolTip("Seconds between each forward step (F to toggle)")
        forward_play_layout.addWidget(self.forward_play_button)
        forward_play_layout.addWidget(QLabel("Speed:"))
        forward_play_layout.addWidget(self.forward_play_speed_spinbox)
        layout.addLayout(forward_play_layout, 6, 0, 1, 3)

        return control_group

    def _build_summary_tab(self) -> QWidget:
        summary_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.summary_edf_label = QLabel("EDF: (none selected)")
        self.summary_csv_label = QLabel("CSV: (not loaded)")
        self.summary_blinker_label = QLabel("Blinker: (not loaded)")
        for label in (self.summary_edf_label, self.summary_csv_label, self.summary_blinker_label):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setWordWrap(True)

        self.summary_overall_label = QLabel("Dataset summary: (scan dataset to populate)")
        self.summary_overall_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_overall_label.setWordWrap(True)

        self.summary_table = QTableWidget(0, 5)
        self.summary_table.setHorizontalHeaderLabels(
            ["Subject", "Status", "CSV", "Blinker", "EDF"]
        )
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setSelectionMode(QTableWidget.NoSelection)

        remark_group = QGroupBox("Remarks")
        remark_layout = QVBoxLayout()
        remark_group.setLayout(remark_layout)
        self.remark_input = QTextEdit()
        self.remark_input.setPlaceholderText("Enter remarks for this EDF review...")
        self.remark_input.setMinimumHeight(80)
        self.remark_save_button = QPushButton("Save Remark")
        self.remark_save_button.clicked.connect(self._save_remark)
        self.remark_history_label = QLabel("Saved remarks:")
        self.remark_history_list = QListWidget()
        self.remark_history_list.setSelectionMode(QListWidget.NoSelection)
        self.remark_history_list.setMinimumHeight(120)

        remark_button_row = QHBoxLayout()
        remark_button_row.addStretch()
        remark_button_row.addWidget(self.remark_save_button)
        remark_layout.addWidget(self.remark_input)
        remark_layout.addLayout(remark_button_row)
        remark_layout.addWidget(self.remark_history_label)
        remark_layout.addWidget(self.remark_history_list)

        layout.addWidget(self.summary_edf_label)
        layout.addWidget(self.summary_csv_label)
        layout.addWidget(self.summary_blinker_label)
        layout.addWidget(self.summary_overall_label)
        layout.addWidget(self.summary_table)
        layout.addWidget(remark_group)
        layout.addStretch()
        summary_tab.setLayout(layout)
        return summary_tab

    def _browse_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select Murat 2018 dataset root", str(self.dataset_root)
        )
        if selected:
            self.directory_input.setText(selected)
            self._scan_directory()

    def _scan_directory(self) -> None:
        self._save_session_state()
        root_text = self.directory_input.text().strip()
        dataset_root = Path(root_text).expanduser().resolve() if root_text else DEFAULT_MURAT_ROOT
        self.dataset_root = dataset_root
        self.directory_input.setText(str(dataset_root))

        self.recording_list.clear()
        self.recordings = []
        self.current_recording = None
        self.time_series_viewer.load_time_series_file(None)
        self._update_review_controls(False)
        self._update_selected_summary(None)

        if not dataset_root.exists():
            self._set_status(f"Murat 2018 dataset root not found at {dataset_root}.")
            self._refresh_dataset_summary()
            return

        self.recordings = self._discover_recordings(dataset_root)
        for recording in self.recordings:
            csv_state = "CSV" if recording.csv_path.exists() else "No CSV"
            blinker_state = "Blinker" if recording.blinker_path.exists() else "No Blinker"
            item = QListWidgetItem(
                f"{recording.subject_id} | {recording.edf_path.name} | {csv_state} | {blinker_state}"
            )
            item.setData(Qt.UserRole, recording)
            self.recording_list.addItem(item)

        self._refresh_dataset_summary()
        if self.recordings:
            self._set_status(f"Found {len(self.recordings)} Murat EDF recording(s).")
        else:
            self._set_status(f"No EDF recordings found under {dataset_root}.")

    def _discover_recordings(self, dataset_root: Path) -> list[MuratRecording]:
        recordings: list[MuratRecording] = []
        for folder in sorted(
            (path for path in dataset_root.iterdir() if path.is_dir()),
            key=lambda path: self._subject_sort_key(path.name),
        ):
            edf_files = sorted(folder.glob("*.edf"), key=lambda path: path.name.lower())
            if not edf_files:
                continue
            recording = MuratRecording(
                subject_id=folder.name,
                folder=folder,
                edf_path=edf_files[0],
            )
            try:
                ensure_murat_session_file(recording.folder)
            except Exception:
                pass
            recordings.append(recording)
        return recordings

    def _subject_sort_key(self, value: str) -> tuple[int, str]:
        try:
            return (int(value), value)
        except ValueError:
            return (1_000_000, value.lower())

    def _load_selected_recording(self) -> None:
        selected_items = self.recording_list.selectedItems()
        if not selected_items:
            return

        self._save_session_state()
        recording = selected_items[0].data(Qt.UserRole)
        if not isinstance(recording, MuratRecording):
            return

        self.current_recording = recording
        created, message = self._ensure_blinker_annotation_csv(recording)
        self.time_series_viewer.load_time_series_file(recording.edf_path, recording.csv_path)
        loaded_ts, loaded_csv = self.time_series_viewer.last_loaded_paths()
        loaded = loaded_ts is not None
        self._update_review_controls(loaded)
        self._update_selected_summary(recording, loaded_ts, loaded_csv)
        self._load_session_state()
        self._refresh_dataset_summary()

        if created:
            self._set_status(message)
        elif loaded:
            self._set_status(f"Loaded Murat recording {recording.subject_id}.")
        else:
            self._set_status(f"Failed to load Murat recording {recording.subject_id}.")

    def _ensure_blinker_annotation_csv(self, recording: MuratRecording) -> tuple[bool, str]:
        if recording.csv_path.exists():
            return False, f"Using existing annotation CSV at {recording.csv_path}."
        if not recording.blinker_path.exists():
            return False, f"No Blinker pickle found at {recording.blinker_path}."

        try:
            annotations = list(self._annotations_from_blinker(recording.blinker_path))
            self._write_bootstrap_csv(recording.csv_path, annotations)
        except Exception as exc:
            return False, f"Failed to create CSV from Blinker output: {exc}"

        return (
            True,
            f"Created initial annotation CSV with {len(annotations)} blink(s): {recording.csv_path}",
        )

    def _annotations_from_blinker(self, pickle_path: Path) -> Iterable[dict[str, str]]:
        with pickle_path.open("rb") as handle:
            payload = pickle.load(handle)

        frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
        if not isinstance(frames, dict):
            return []

        blink_fits = frames.get("blinkFits")
        if blink_fits is None or not hasattr(blink_fits, "iterrows"):
            return []

        srate = self._blinker_sample_rate(frames)
        rows: list[dict[str, str]] = []
        for _, row in blink_fits.iterrows():
            bounds = self._blinker_bounds(row, srate)
            if bounds is None:
                continue
            left_frame, right_frame = bounds
            onset = max(0.0, left_frame / srate)
            duration = max(0.1, (right_frame - left_frame) / srate)
            rows.append(
                {
                    "onset": f"{onset:.6f}",
                    "duration": f"{duration:.6f}",
                    "description": "blink",
                }
            )
        rows.sort(key=lambda item: float(item["onset"]))
        return rows

    def _blinker_sample_rate(self, frames: dict) -> float:
        for key in ("blinkStats", "params"):
            table = frames.get(key)
            if table is None or not hasattr(table, "columns") or "srate" not in table.columns:
                continue
            if len(table.index) == 0:
                continue
            value = self._numeric_value(table.iloc[0].get("srate"))
            if value is not None and value > 0:
                return value
        return 200.0

    def _blinker_bounds(self, row, srate: float) -> Optional[tuple[float, float]]:
        for left_name, right_name in (
            ("leftBase", "rightBase"),
            ("leftZero", "rightZero"),
            ("leftOuter", "rightOuter"),
        ):
            left = self._numeric_value(row.get(left_name))
            right = self._numeric_value(row.get(right_name))
            if left is not None and right is not None and right > left:
                return left, right

        peak = self._numeric_value(row.get("maxFrame"))
        if peak is None:
            return None
        half_width = 0.15 * srate
        return max(0.0, peak - half_width), peak + half_width

    def _numeric_value(self, value) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    def _write_bootstrap_csv(self, csv_path: Path, rows: list[dict[str, str]]) -> None:
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
        try:
            writer = csv.DictWriter(tmp_handle, fieldnames=["onset", "duration", "description"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
        except Exception:
            tmp_handle.close()
            tmp_path.unlink(missing_ok=True)
            raise
        tmp_handle.close()
        os.replace(tmp_path, csv_path)

    def _search_time(self) -> None:
        text = self.time_input.text().strip()
        if not text:
            self._set_status("Enter a time value in seconds.")
            return
        try:
            seconds = float(text)
        except ValueError:
            self._set_status("Invalid time value. Please enter a number.")
            return
        self._goto_time(seconds)

    def _step_time(self, direction: int) -> None:
        if self.current_recording is None:
            self._set_status("Load a recording before navigating.")
            return
        step = self.step_seconds_input.value() * (1 if direction > 0 else -1)
        self._goto_time(self.time_series_viewer.current_cursor_time() + step)

    def _goto_time(self, seconds: float) -> None:
        if self.current_recording is None:
            self._set_status("Load a recording before navigating.")
            return
        self.time_series_viewer.seek_time(max(0.0, seconds))
        self._update_current_time_label()
        self._set_status(f"Displaying time {self.time_series_viewer.current_cursor_time():.3f}s.")

    def _update_current_time_label(self) -> None:
        current_time = self.time_series_viewer.current_cursor_time()
        duration = self.time_series_viewer.signal_duration_seconds()
        if duration is None or duration <= 0:
            self.current_time_label.setText(f"Current time: {current_time:.3f}s")
            return

        clamped_time = max(0.0, min(current_time, duration))
        remaining = max(0.0, duration - clamped_time)
        percent_complete = (clamped_time / duration) * 100
        self.current_time_label.setText(
            "Current time: "
            f"{clamped_time:.3f}s / {duration:.3f}s "
            f"({percent_complete:.1f}% complete, {remaining:.3f}s left)"
        )

    def _update_review_controls(self, enabled: bool) -> None:
        for widget in (
            self.time_input,
            self.time_search_button,
            self.step_seconds_input,
            self.left_button,
            self.right_button,
            self.save_annotations_button,
            self.status_dropdown,
            self.play_button,
            self.play_speed_spinbox,
            self.forward_play_button,
            self.forward_play_speed_spinbox,
            self.remark_input,
            self.remark_save_button,
            self.remark_history_label,
            self.remark_history_list,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self.current_time_label.setText("Current time: -")

    def _update_selected_summary(
        self,
        recording: Optional[MuratRecording],
        loaded_ts: Optional[Path] = None,
        loaded_csv: Optional[Path] = None,
    ) -> None:
        if recording is None:
            self.summary_edf_label.setText("EDF: (none selected)")
            self.summary_csv_label.setText("CSV: (not loaded)")
            self.summary_blinker_label.setText("Blinker: (not loaded)")
            return

        self.summary_edf_label.setText(
            self._format_path_line("EDF", recording.edf_path, loaded_ts is not None)
        )
        self.summary_csv_label.setText(
            self._format_path_line("CSV", recording.csv_path, loaded_csv is not None)
        )
        self.summary_blinker_label.setText(
            self._format_path_line("Blinker", recording.blinker_path, recording.blinker_path.exists())
        )

    def _format_path_line(self, label: str, path: Path, loaded: bool) -> str:
        exists = "found" if path.exists() else "missing"
        state = "loaded" if loaded else "expected"
        return f"{label}: {path} [{state}, {exists}]"

    def _refresh_dataset_summary(self) -> None:
        totals = {
            "recordings": len(self.recordings),
            "csv": sum(1 for item in self.recordings if item.csv_path.exists()),
            "blinker": sum(1 for item in self.recordings if item.blinker_path.exists()),
            "edf": sum(1 for item in self.recordings if item.edf_path.exists()),
        }
        self.summary_overall_label.setText(
            "Dataset summary: "
            f"{totals['recordings']} recording(s) | "
            f"CSV: {totals['csv']}, "
            f"Blinker: {totals['blinker']}, "
            f"EDF: {totals['edf']}"
        )
        self.summary_table.setRowCount(len(self.recordings))
        for row, recording in enumerate(self.recordings):
            values = [
                recording.subject_id,
                self._status_for_recording(recording),
                "yes" if recording.csv_path.exists() else "missing",
                "yes" if recording.blinker_path.exists() else "missing",
                "yes" if recording.edf_path.exists() else "missing",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.summary_table.setItem(row, col, item)

    def _status_for_recording(self, recording: MuratRecording) -> str:
        if not recording.session_path.exists():
            return "Pending"
        try:
            with recording.session_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:
            return "Pending"
        status = data.get("status", "Pending")
        if status not in {"Pending", "Ongoing", "Complete", "Issue"}:
            return "Pending"
        return status

    def _save_session_state(self) -> None:
        recording = self.current_recording
        if recording is None:
            return

        existing_data = {}
        if recording.session_path.exists():
            try:
                with recording.session_path.open("r", encoding="utf-8") as handle:
                    existing_data = yaml.safe_load(handle) or {}
            except Exception:
                existing_data = {}

        remarks = existing_data.get("remarks")
        if not isinstance(remarks, list):
            remarks = []

        current_position = self.time_series_viewer.current_cursor_time()
        data = {
            "stop_position": current_position,
            "status": self.status_value,
            "remark": self.remark_input.toPlainText().strip(),
            "remarks": remarks,
        }
        try:
            with recording.session_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle)
        except Exception as exc:
            self._set_status(f"Failed to save session state: {exc}")
            return
        self._refresh_dataset_summary()

    def _load_session_state(self) -> None:
        recording = self.current_recording
        if recording is None:
            return
        if not recording.session_path.exists():
            self.status_dropdown.setCurrentText("Pending")
            self.status_value = "Pending"
            self.remark_input.setPlainText("")
            self._refresh_remark_history([])
            self._update_current_time_label()
            return

        try:
            with recording.session_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            status = data.get("status", "Pending")
            self.status_dropdown.setCurrentText(status)
            self.status_value = status
            self.remark_input.setPlainText(str(data.get("remark", "")))
            self._refresh_remark_history(self._extract_remarks(data))
            stop_time = self._numeric_value(data.get("stop_position"))
            if stop_time is None:
                stop_time = self._numeric_value(data.get("stop_time"))
            if stop_time is not None and stop_time > 0:
                self.time_series_viewer.seek_time(stop_time)
            self._update_current_time_label()
        except Exception as exc:
            self._set_status(f"Failed to load session state: {exc}")
            self._refresh_remark_history([])

    def _on_status_changed(self, text: str) -> None:
        self.status_value = text
        self._save_session_state()

    def _save_remark(self) -> None:
        recording = self.current_recording
        if recording is None:
            self._set_status("Load a recording before saving remarks.")
            return
        remark_text = self.remark_input.toPlainText().strip()
        if not remark_text:
            self._set_status("Enter a remark before saving.")
            return

        data = {}
        if recording.session_path.exists():
            try:
                with recording.session_path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
            except Exception:
                data = {}
        remarks = self._extract_remarks(data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        remarks.append(
            f"{timestamp} - {self.time_series_viewer.current_cursor_time():.3f}s - {remark_text}"
        )
        data.update(
            {
                "stop_position": self.time_series_viewer.current_cursor_time(),
                "status": self.status_value,
                "remark": remark_text,
                "remarks": remarks,
            }
        )
        try:
            with recording.session_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle)
        except Exception as exc:
            self._set_status(f"Failed to save remark: {exc}")
            return
        self._refresh_remark_history(remarks)
        self._refresh_dataset_summary()
        self._set_status("Remark saved.")

    def _extract_remarks(self, data: dict) -> list[str]:
        remarks = data.get("remarks")
        if isinstance(remarks, list):
            return [str(item).strip() for item in remarks if str(item).strip()]
        remark = str(data.get("remark", "")).strip()
        return [remark] if remark else []

    def _refresh_remark_history(self, remarks: list[str]) -> None:
        self.remark_history_list.clear()
        for remark in remarks:
            self.remark_history_list.addItem(remark)

    def _toggle_annotation_play(self, checked: bool) -> None:
        if checked:
            self._start_annotation_play()
        else:
            self._stop_annotation_play()

    def _start_annotation_play(self) -> None:
        interval_ms = int(self.play_speed_spinbox.value() * 1000)
        self._annotation_play_timer.start(interval_ms)
        self.play_button.setChecked(True)
        self.play_button.setText("Stop Annotations")

    def _stop_annotation_play(self) -> None:
        self._annotation_play_timer.stop()
        self.play_button.setChecked(False)
        self.play_button.setText("Play Annotations")

    def _annotation_play_tick(self) -> None:
        interval_ms = int(self.play_speed_spinbox.value() * 1000)
        if self._annotation_play_timer.interval() != interval_ms:
            self._annotation_play_timer.setInterval(interval_ms)
        self.time_series_viewer.jump_to_next_annotation()

    def _toggle_forward_play(self) -> None:
        if self._forward_play_timer.isActive():
            self._stop_forward_play()
        else:
            self._start_forward_play()

    def _toggle_forward_play_button(self, checked: bool) -> None:
        if checked:
            self._start_forward_play()
        else:
            self._stop_forward_play()

    def _start_forward_play(self) -> None:
        interval_ms = max(16, int(self.forward_play_speed_spinbox.value() * 1000))
        self._forward_play_timer.start(interval_ms)
        self.forward_play_button.setChecked(True)
        self.forward_play_button.setText("Stop Forward")
        self._set_status("Auto-forward started. Press F or ESC to stop.")

    def _stop_forward_play(self) -> None:
        self._forward_play_timer.stop()
        self.forward_play_button.setChecked(False)
        self.forward_play_button.setText("Forward Play")
        self._set_status("Auto-forward stopped.")

    def _forward_play_tick(self) -> None:
        interval_ms = max(16, int(self.forward_play_speed_spinbox.value() * 1000))
        if self._forward_play_timer.interval() != interval_ms:
            self._forward_play_timer.setInterval(interval_ms)
        current_time = self.time_series_viewer.current_cursor_time()
        step = self.step_seconds_input.value()
        self._goto_time(current_time + step)
        duration = self.time_series_viewer.signal_duration_seconds()
        if duration is not None and self.time_series_viewer.current_cursor_time() >= duration:
            self._stop_forward_play()
            self._set_status("Auto-forward reached end of recording.")

    def _stop_all_play(self) -> None:
        self._stop_annotation_play()
        self._stop_forward_play()

    def _setup_shortcuts(self) -> None:
        self.save_shortcut = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_S), self)
        self.save_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.save_shortcut.activated.connect(self.time_series_viewer.save_annotations)

        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.left_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.left_shortcut.activated.connect(lambda: self._step_if_allowed(-1))

        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.right_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.right_shortcut.activated.connect(lambda: self._step_if_allowed(1))

        self.next_annotation_shortcut = QShortcut(QKeySequence(Qt.Key_BracketRight), self)
        self.next_annotation_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.next_annotation_shortcut.activated.connect(
            self.time_series_viewer.jump_to_next_annotation
        )

        self.previous_annotation_shortcut = QShortcut(QKeySequence(Qt.Key_BracketLeft), self)
        self.previous_annotation_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.previous_annotation_shortcut.activated.connect(
            self.time_series_viewer.jump_to_previous_annotation
        )

        self.next_annotation_letter = QShortcut(QKeySequence(Qt.Key_N), self)
        self.next_annotation_letter.setContext(Qt.WidgetWithChildrenShortcut)
        self.next_annotation_letter.activated.connect(self.time_series_viewer.jump_to_next_annotation)

        self.previous_annotation_letter = QShortcut(QKeySequence(Qt.Key_B), self)
        self.previous_annotation_letter.setContext(Qt.WidgetWithChildrenShortcut)
        self.previous_annotation_letter.activated.connect(
            self.time_series_viewer.jump_to_previous_annotation
        )

        self.delete_annotation_shortcut = QShortcut(QKeySequence(Qt.Key_D), self)
        self.delete_annotation_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.delete_annotation_shortcut.activated.connect(self._delete_annotation_if_allowed)

        self.play_shortcut = QShortcut(QKeySequence(Qt.Key_P), self)
        self.play_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.play_shortcut.activated.connect(self._start_annotation_play)

        self.forward_play_shortcut = QShortcut(QKeySequence(Qt.Key_F), self)
        self.forward_play_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.forward_play_shortcut.activated.connect(self._toggle_forward_play)

        self.stop_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.stop_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.stop_shortcut.activated.connect(self._stop_all_play)

        self.time_series_viewer.annotation_jump_requested.connect(self._on_annotation_jump)

    def _step_if_allowed(self, direction: int) -> None:
        if self._shortcut_allowed():
            self._step_time(direction)

    def _delete_annotation_if_allowed(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.delete_selected_annotation()

    def _shortcut_allowed(self) -> bool:
        focus_widget = QApplication.focusWidget()
        return not isinstance(focus_widget, (QLineEdit, QDoubleSpinBox, QTextEdit))

    def _on_annotation_jump(self, seconds: float) -> None:
        self._update_current_time_label()
        self._set_status(f"Jumped to annotation at {seconds:.3f}s.")

    def _set_status(self, message: str) -> None:
        self.status_bar.showMessage(message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_all_play()
        self._save_session_state()
        super().closeEvent(event)
