"""Cao 2018 sustained-attention driving dataset annotation review interface."""
from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPainter, QPen
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
    QScrollArea,
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

DEFAULT_CAO_ROOT = Path(r"D:\dataset\sustained_attention_driving")
CAO_DEFAULT_CHANNELS = {"fp1", "fp2"}
_STATUSES = ["Pending", "Ongoing", "Complete", "Issue"]
_STATUS_COLORS = {
    "Pending": "#9e9e9e",
    "Ongoing": "#1976d2",
    "Complete": "#388e3c",
    "Issue": "#d32f2f",
}


class CaoTimeSeriesViewer(TimeSeriesViewer):
    """TimeSeriesViewer that shows only FP1 and FP2 by default."""

    @classmethod
    def _is_always_visible_channel(cls, channel_name: str) -> bool:
        return channel_name.casefold() in CAO_DEFAULT_CHANNELS

    def _required_plot_channel_indices(self) -> list:
        if self.raw is None:
            return []
        return [
            i for i, name in enumerate(self.raw.ch_names)
            if name.casefold() in CAO_DEFAULT_CHANNELS
        ] or list(range(len(self.raw.ch_names)))

    def inject_annotations(self, rows: list) -> None:
        """Inject pre-built annotation dicts into the viewer without a CSV file."""
        if self._times is None or self._times.size == 0:
            return
        from time_series import Annotation
        self._annotations = [
            Annotation(onset=float(r["onset"]), duration=float(r["duration"]), description=r["description"])
            for r in rows
        ]
        self._update_annotation_filter_options()
        self._set_annotations_dirty(False)
        self._refresh_visible_annotation_regions()


SESSION_FILENAME = "Cao2018Viewer.yaml"
EPOCH_HEALTH_FILENAME = "epoch_health.csv"
EPOCH_WINDOW_SECONDS = 30.0
BLINKER_PICKLE = "blinker_results.pkl"
DEFAULT_SESSION_STATE = {
    "stop_position": 0.0,
    "status": "Pending",
    "remark": "",
    "remarks": [],
}


@dataclass(frozen=True)
class CaoRecording:
    """Files associated with one Cao 2018 recording session."""

    subject_id: str   # e.g. "S01"
    session_id: str   # e.g. "051017m"
    folder: Path
    ts_path: Path     # .fif file

    @property
    def csv_path(self) -> Path:
        # Always use the original FIF stem for the CSV so annotations survive
        # a switch between full-resolution and downsampled files.
        stem = self.ts_path.stem.removesuffix("_ds20hz")
        return self.ts_path.with_name(stem + ".csv")

    @property
    def blinker_path(self) -> Path:
        return self.folder / BLINKER_PICKLE

    @property
    def epoch_health_path(self) -> Path:
        return self.folder / EPOCH_HEALTH_FILENAME

    @property
    def session_path(self) -> Path:
        return self.folder / SESSION_FILENAME

    @property
    def display_name(self) -> str:
        return f"{self.subject_id}/{self.session_id}"


def ensure_cao_session_file(folder: Path) -> Path:
    """Create the Cao review session YAML in a recording folder if missing."""
    session_path = folder / SESSION_FILENAME
    if not session_path.exists():
        with session_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(DEFAULT_SESSION_STATE, handle, sort_keys=False)
    return session_path


class EpochTimelineWidget(QWidget):
    """Horizontal mini-timeline showing 30-second epoch health states.

    Each epoch is drawn as a coloured block: green (Good), red (Bad), or grey (unlabelled).
    The current epoch receives a dark border.  Click any block to jump to that epoch.
    """

    epoch_clicked = pyqtSignal(int)

    _FILL = {"Good": "#4caf50", "Bad": "#f44336", "": "#bdbdbd"}
    _BORDER = "#212121"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._total = 0
        self._states: list[str] = []
        self._current = 0
        self.setFixedHeight(36)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_data(self, total: int, states: list[str], current: int) -> None:
        self._total = total
        self._states = list(states)
        self._current = current
        self.update()

    def _block_rect(self, i: int) -> tuple[int, int, int, int]:
        w = self.width()
        x = int(i * w / self._total)
        x2 = int((i + 1) * w / self._total)
        return x, 2, max(1, x2 - x - 1), self.height() - 4

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        if self._total == 0:
            painter.end()
            return
        for i in range(self._total):
            state = self._states[i] if i < len(self._states) else ""
            x, y, rw, rh = self._block_rect(i)
            painter.fillRect(x, y, rw, rh, QColor(self._FILL.get(state, self._FILL[""])))
            if rw >= 18:
                painter.setPen(QColor("#ffffff") if state else QColor("#616161"))
                f = painter.font()
                f.setPixelSize(9)
                painter.setFont(f)
                painter.drawText(x, y, rw, rh, Qt.AlignCenter, str(i))
            if i == self._current:
                painter.setPen(QPen(QColor(self._BORDER), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(x + 1, y + 1, max(0, rw - 2), max(0, rh - 2))
        painter.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._total == 0 or event.button() != Qt.LeftButton:
            return
        i = max(0, min(self._total - 1, int(event.x() / max(1, self.width()) * self._total)))
        self.epoch_clicked.emit(i)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._total == 0:
            self.setToolTip("")
            return
        i = max(0, min(self._total - 1, int(event.x() / max(1, self.width()) * self._total)))
        state = self._states[i] if i < len(self._states) else ""
        s0, s1 = i * EPOCH_WINDOW_SECONDS, (i + 1) * EPOCH_WINDOW_SECONDS
        self.setToolTip(f"Epoch {i}  ({s0:.0f}s – {s1:.0f}s)  [{state or 'unlabelled'}]")


class Cao2018Viewer(QMainWindow):
    """Standalone FIF review UI for the Cao 2018 sustained-attention driving dataset."""

    DEFAULT_STEP_SECONDS = 20.0
    RECORDINGS_PANEL_WIDTH = 300
    RECORDINGS_PANEL_MAX_WIDTH = 340
    DETACHED_PANEL_MAX_WIDTH = 16777215

    def __init__(self, dataset_root: Path = DEFAULT_CAO_ROOT) -> None:
        super().__init__()
        self.setWindowTitle("Cao 2018 Annotation Reviewer")
        self.dataset_root = dataset_root
        self.recordings: list[CaoRecording] = []
        self.current_recording: Optional[CaoRecording] = None
        self.status_value = "Pending"
        self._epoch_health_mode = False

        self.time_series_viewer = CaoTimeSeriesViewer()

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

        self.epoch_health_label = QLabel("")
        self.epoch_health_label.setAlignment(Qt.AlignCenter)
        self.epoch_health_label.setFixedHeight(24)

        self.epoch_timeline_widget = EpochTimelineWidget()
        self.epoch_timeline_widget.epoch_clicked.connect(self._on_epoch_timeline_clicked)
        self.epoch_timeline_widget.setVisible(False)

        ts_container = QWidget()
        ts_layout = QVBoxLayout()
        ts_layout.setContentsMargins(0, 0, 0, 0)
        ts_layout.setSpacing(0)
        ts_layout.addWidget(self.epoch_health_label)
        ts_layout.addWidget(self.epoch_timeline_widget)
        ts_layout.addWidget(self.time_series_viewer)
        ts_layout.setStretch(0, 0)
        ts_layout.setStretch(1, 0)
        ts_layout.setStretch(2, 1)
        ts_container.setLayout(ts_layout)

        self.main_splitter.addWidget(self.side_tabs)
        self.main_splitter.addWidget(ts_container)
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
        directory_group = QGroupBox("Cao 2018 Dataset Directory")
        directory_layout = QHBoxLayout()
        directory_group.setLayout(directory_layout)

        self.directory_input = QLineEdit()
        self.directory_input.setPlaceholderText(r"D:\dataset\sustained_attention_driving")
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
        statistics_tab = self._build_statistics_tab()

        tabs.addTab(recordings_tab, "Recordings")
        tabs.addTab(channels_tab, "Channels")
        tabs.addTab(summary_tab, "Summary")
        tabs.addTab(statistics_tab, "Statistics")
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

        layout.addWidget(QLabel("Discovered FIF Recordings"))
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
        self.step_seconds_input.setSingleStep(0.5)
        self.step_seconds_input.setValue(self.DEFAULT_STEP_SECONDS)

        self.step_dec_button = QPushButton("-")
        self.step_dec_button.setFixedWidth(28)
        self.step_dec_button.setToolTip("Decrease step size")
        self.step_dec_button.clicked.connect(self._decrease_step)
        self.step_inc_button = QPushButton("+")
        self.step_inc_button.setFixedWidth(28)
        self.step_inc_button.setToolTip("Increase step size")
        self.step_inc_button.clicked.connect(self._increase_step)

        step_layout = QHBoxLayout()
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_layout.addWidget(self.step_dec_button)
        step_layout.addWidget(self.step_seconds_input)
        step_layout.addWidget(self.step_inc_button)

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

        self.window_dropdown = QComboBox()
        self.window_dropdown.addItems(["5 s", "10 s", "15 s", "30 s", "40 s", "50 s", "60 s"])
        self.window_dropdown.currentIndexChanged.connect(self._on_window_changed)

        layout.addWidget(QLabel("Time (s):"), 0, 0)
        layout.addWidget(self.time_input, 0, 1)
        layout.addWidget(self.time_search_button, 0, 2)
        layout.addWidget(QLabel("Step (s):"), 1, 0)
        layout.addLayout(step_layout, 1, 1, 1, 2)
        layout.addWidget(QLabel("Window:"), 2, 0)
        layout.addWidget(self.window_dropdown, 2, 1, 1, 2)
        layout.addWidget(self.left_button, 3, 0)
        layout.addWidget(self.right_button, 3, 1)
        layout.addWidget(self.save_annotations_button, 3, 2)
        layout.addWidget(QLabel("Status:"), 4, 0)
        layout.addWidget(self.status_dropdown, 4, 1, 1, 2)
        layout.addWidget(self.current_time_label, 5, 0, 1, 3)

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
        layout.addLayout(play_layout, 6, 0, 1, 3)

        forward_play_layout = QHBoxLayout()
        self.forward_play_button = QPushButton("Forward Play")
        self.forward_play_button.setCheckable(True)
        self.forward_play_button.clicked.connect(self._toggle_forward_play_button)
        self.forward_play_speed_spinbox = QDoubleSpinBox()
        self.forward_play_speed_spinbox.setRange(0.016, 30.0)
        self.forward_play_speed_spinbox.setValue(0.1)
        self.forward_play_speed_spinbox.setSingleStep(0.1)
        self.forward_play_speed_spinbox.setSuffix(" s")
        self.forward_play_speed_spinbox.setToolTip("Seconds between each forward step (F or Q to toggle)")
        forward_play_layout.addWidget(self.forward_play_button)
        forward_play_layout.addWidget(QLabel("Speed:"))
        forward_play_layout.addWidget(self.forward_play_speed_spinbox)
        layout.addLayout(forward_play_layout, 7, 0, 1, 3)

        self.epoch_health_mode_button = QPushButton("Epoch Health Mode")
        self.epoch_health_mode_button.setCheckable(True)
        self.epoch_health_mode_button.setToolTip(
            "Lock step and window to 30 s and show the epoch health timeline"
        )
        self.epoch_health_mode_button.toggled.connect(self._toggle_epoch_health_mode)
        layout.addWidget(self.epoch_health_mode_button, 8, 0, 1, 3)

        return control_group

    def _build_summary_tab(self) -> QWidget:
        summary_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.summary_fif_label = QLabel("FIF: (none selected)")
        self.summary_csv_label = QLabel("CSV: (not loaded)")
        self.summary_blinker_label = QLabel("Blinker: (not loaded)")
        self.summary_epoch_health_label = QLabel("Epoch Health: (none selected)")
        for label in (
            self.summary_fif_label,
            self.summary_csv_label,
            self.summary_blinker_label,
            self.summary_epoch_health_label,
        ):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setWordWrap(True)

        self.summary_overall_label = QLabel("Dataset summary: (scan dataset to populate)")
        self.summary_overall_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_overall_label.setWordWrap(True)

        self.summary_table = QTableWidget(0, 5)
        self.summary_table.setHorizontalHeaderLabels(
            ["Recording", "Status", "CSV", "Blinker", "FIF"]
        )
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setSelectionMode(QTableWidget.NoSelection)

        remark_group = QGroupBox("Remarks")
        remark_layout = QVBoxLayout()
        remark_group.setLayout(remark_layout)
        self.remark_input = QTextEdit()
        self.remark_input.setPlaceholderText("Enter remarks for this recording review...")
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

        layout.addWidget(self.summary_fif_label)
        layout.addWidget(self.summary_csv_label)
        layout.addWidget(self.summary_blinker_label)
        layout.addWidget(self.summary_epoch_health_label)
        layout.addWidget(self.summary_overall_label)
        layout.addWidget(self.summary_table)
        layout.addWidget(remark_group)
        layout.addStretch()
        summary_tab.setLayout(layout)
        return summary_tab

    def _build_statistics_tab(self) -> QWidget:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        tab = QWidget()
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(6)

        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.clicked.connect(self._refresh_statistics_tab)
        outer_layout.addWidget(refresh_btn)

        self._stats_overall_fig = Figure(figsize=(6, 3.5), tight_layout=True)
        self._stats_overall_canvas = FigureCanvasQTAgg(self._stats_overall_fig)
        self._stats_overall_canvas.setMinimumHeight(260)

        self._stats_subject_fig = Figure(figsize=(6, 4), tight_layout=True)
        self._stats_subject_canvas = FigureCanvasQTAgg(self._stats_subject_fig)
        self._stats_subject_canvas.setMinimumHeight(320)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(4)
        inner_layout.addWidget(QLabel("Overall Status Distribution"))
        inner_layout.addWidget(self._stats_overall_canvas)
        inner_layout.addWidget(QLabel("Status by Subject"))
        inner_layout.addWidget(self._stats_subject_canvas)
        inner_layout.addStretch()
        inner.setLayout(inner_layout)
        scroll.setWidget(inner)

        outer_layout.addWidget(scroll)
        tab.setLayout(outer_layout)
        return tab

    def _refresh_statistics_tab(self) -> None:
        if not hasattr(self, "_stats_overall_fig"):
            return
        status_counts: dict[str, int] = {s: 0 for s in _STATUSES}
        subject_counts: dict[str, dict[str, int]] = {}
        for recording in self.recordings:
            status = self._status_for_recording(recording)
            status_counts[status] = status_counts.get(status, 0) + 1
            subj = recording.subject_id
            if subj not in subject_counts:
                subject_counts[subj] = {s: 0 for s in _STATUSES}
            subject_counts[subj][status] += 1
        self._draw_overall_pie(status_counts)
        self._draw_subject_bars(subject_counts)

    def _draw_overall_pie(self, status_counts: dict[str, int]) -> None:
        fig = self._stats_overall_fig
        fig.clear()
        ax = fig.add_subplot(111)
        total = sum(status_counts.values())
        labels = [s for s in _STATUSES if status_counts.get(s, 0) > 0]
        sizes = [status_counts[s] for s in labels]
        colors = [_STATUS_COLORS[s] for s in labels]
        if total == 0:
            ax.text(0.5, 0.5, "No recordings", ha="center", va="center", transform=ax.transAxes)
        else:
            def _fmt(pct: float) -> str:
                n = int(round(pct * total / 100))
                return f"{pct:.0f}%\n({n})"
            ax.pie(sizes, labels=labels, colors=colors, autopct=_fmt, startangle=90)
        ax.set_title(f"Overall Status  (n={total})")
        fig.tight_layout()
        self._stats_overall_canvas.draw()

    def _draw_subject_bars(self, subject_counts: dict[str, dict[str, int]]) -> None:
        fig = self._stats_subject_fig
        fig.clear()
        if not subject_counts:
            self._stats_subject_canvas.draw()
            return
        subjects = sorted(subject_counts.keys(), key=lambda x: self._subject_sort_key(x))
        n = len(subjects)
        canvas_height = max(320, n * 28 + 80)
        self._stats_subject_canvas.setMinimumHeight(canvas_height)
        ax = fig.add_subplot(111)
        bottoms = [0.0] * n
        for status in _STATUSES:
            vals = [float(subject_counts[s].get(status, 0)) for s in subjects]
            if any(v > 0 for v in vals):
                ax.barh(subjects, vals, left=bottoms, label=status, color=_STATUS_COLORS[status])
                bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xlabel("Sessions")
        ax.set_title("Status by Subject")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        self._stats_subject_canvas.draw()

    def _browse_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select Cao 2018 dataset root", str(self.dataset_root)
        )
        if selected:
            self.directory_input.setText(selected)
            self._scan_directory()

    def _scan_directory(self) -> None:
        self._save_session_state()
        root_text = self.directory_input.text().strip()
        dataset_root = Path(root_text).expanduser().resolve() if root_text else DEFAULT_CAO_ROOT
        self.dataset_root = dataset_root
        self.directory_input.setText(str(dataset_root))

        self.recording_list.clear()
        self.recordings = []
        self.current_recording = None
        self.time_series_viewer.load_time_series_file(None)
        self._update_review_controls(False)
        self._update_selected_summary(None)
        self._update_epoch_health_label()

        if not dataset_root.exists():
            self._set_status(f"Cao 2018 dataset root not found at {dataset_root}.")
            self._refresh_dataset_summary()
            return

        self.recordings = self._discover_recordings(dataset_root)
        for recording in self.recordings:
            csv_state = "CSV" if recording.csv_path.exists() else "No CSV"
            blinker_state = "Blinker" if recording.blinker_path.exists() else "No Blinker"
            item = QListWidgetItem(
                f"{recording.display_name} | {recording.ts_path.name} | {csv_state} | {blinker_state}"
            )
            item.setData(Qt.UserRole, recording)
            self.recording_list.addItem(item)

        self._refresh_dataset_summary()
        if self.recordings:
            self._set_status(f"Found {len(self.recordings)} Cao 2018 FIF recording(s).")
        else:
            self._set_status(f"No FIF recordings found under {dataset_root}.")

    def _discover_recordings(self, dataset_root: Path) -> list[CaoRecording]:
        recordings: list[CaoRecording] = []
        subject_dirs = sorted(
            (p for p in dataset_root.iterdir() if p.is_dir()),
            key=lambda p: self._subject_sort_key(p.name),
        )
        for subject_dir in subject_dirs:
            session_dirs = sorted(
                (p for p in subject_dir.iterdir() if p.is_dir()),
                key=lambda p: p.name.lower(),
            )
            for session_dir in session_dirs:
                ds_files = sorted(session_dir.glob("*_ds20hz.fif"), key=lambda f: f.name.lower())
                raw_files = sorted(
                    (f for f in session_dir.glob("*.fif")
                     if not f.name.endswith("-epo.fif") and not f.name.endswith("_ds20hz.fif")),
                    key=lambda f: f.name.lower(),
                )
                if not ds_files and not raw_files:
                    continue
                ts_file = ds_files[0] if ds_files else raw_files[0]
                recording = CaoRecording(
                    subject_id=subject_dir.name,
                    session_id=session_dir.name,
                    folder=session_dir,
                    ts_path=ts_file,
                )
                try:
                    ensure_cao_session_file(session_dir)
                except Exception:
                    pass
                recordings.append(recording)
        return recordings

    def _subject_sort_key(self, value: str) -> tuple[int, str]:
        stripped = value.lstrip("Ss")
        try:
            return (int(stripped), value)
        except ValueError:
            return (1_000_000, value.lower())

    def _load_selected_recording(self) -> None:
        selected_items = self.recording_list.selectedItems()
        if not selected_items:
            return

        self._save_session_state()
        recording = selected_items[0].data(Qt.UserRole)
        if not isinstance(recording, CaoRecording):
            return

        self.current_recording = recording
        self._set_status(f"Loading {recording.display_name} — {recording.ts_path.name}…")
        QApplication.processEvents()
        self.time_series_viewer.load_time_series_file(recording.ts_path, recording.csv_path)
        loaded_ts, loaded_csv = self.time_series_viewer.last_loaded_paths()
        loaded = loaded_ts is not None

        if loaded and not self._csv_has_annotations(recording.csv_path):
            self._inject_pkl_annotations(recording)

        self._update_review_controls(loaded)
        self._update_selected_summary(recording, loaded_ts, loaded_csv)
        self._refresh_current_list_item()
        self._load_session_state()
        self._refresh_dataset_summary()
        self._update_epoch_health_label()

        if loaded:
            self._set_status(f"Loaded Cao 2018 recording {recording.display_name}.")
        else:
            self._set_status(f"Failed to load Cao 2018 recording {recording.display_name}.")

    def _inject_pkl_annotations(self, recording: CaoRecording) -> None:
        if not recording.blinker_path.exists():
            return
        try:
            rows = self._annotations_from_blinker(recording.blinker_path)
        except Exception:
            return
        self.time_series_viewer.inject_annotations(rows)

    @staticmethod
    def _csv_has_annotations(csv_path: Path) -> bool:
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                return next(reader, None) is not None
        except OSError:
            return False

    def _annotations_from_blinker(self, pickle_path: Path) -> list[dict[str, str]]:
        with pickle_path.open("rb") as handle:
            payload = pickle.load(handle)

        return self._annotations_from_blinker_payload(payload)

    def _annotations_from_blinker_payload(self, payload) -> list[dict[str, str]]:
        rows = self._annotations_from_frame_payload(payload)
        if rows:
            return rows

        if not isinstance(payload, dict):
            return []

        rows = self._annotations_from_frame_payload(payload.get("frames"))
        if rows:
            return rows

        channels = payload.get("channels")
        if not isinstance(channels, dict) or not channels:
            return []

        preferred_names = ["FP1", "FP2"]
        ordered_names = preferred_names + [
            name for name in channels.keys() if name not in preferred_names
        ]
        for name in ordered_names:
            channel_payload = channels.get(name)
            rows = self._annotations_from_channel_payload(channel_payload)
            if rows:
                return rows
        return []

    def _annotations_from_channel_payload(self, channel_payload) -> list[dict[str, str]]:
        rows = self._annotations_from_frame_payload(channel_payload)
        if rows:
            return rows
        if not isinstance(channel_payload, dict):
            return []
        return self._annotations_from_frame_payload(channel_payload.get("frames"))

    def _annotations_from_frame_payload(self, payload) -> list[dict[str, str]]:
        if payload is None:
            return []

        direct_rows = self._rows_from_onset_duration_table(payload)
        if direct_rows:
            return direct_rows

        if not isinstance(payload, dict):
            return []

        events = payload.get("events")
        event_rows = self._rows_from_pyblinker_events(events, payload)
        if event_rows:
            return event_rows

        blink_fits = payload.get("blinkFits")
        return self._rows_from_blink_fits(blink_fits, payload)

    def _rows_from_onset_duration_table(self, table) -> list[dict[str, str]]:
        if not hasattr(table, "iterrows") or not hasattr(table, "columns"):
            return []
        if "onset" not in table.columns or "duration" not in table.columns:
            return []

        rows: list[dict[str, str]] = []
        for _, row in table.iterrows():
            onset = self._numeric_value(row.get("onset"))
            duration = self._numeric_value(row.get("duration"))
            if onset is None or duration is None or duration <= 0:
                continue
            rows.append(self._annotation_row(onset, duration))
        rows.sort(key=lambda item: float(item["onset"]))
        return rows

    def _rows_from_pyblinker_events(self, events, frames: dict) -> list[dict[str, str]]:
        if events is None or not hasattr(events, "itertuples"):
            return []

        srate = self._pyblinker_sample_rate(frames)
        rows: list[dict[str, str]] = []
        for event in events.itertuples(index=False):
            event_data = event._asdict()
            bounds = self._pyblinker_bounds(event_data, srate)
            if bounds is None:
                continue
            left_frame, right_frame = bounds
            onset = max(0.0, left_frame / srate)
            duration = max(0.1, (right_frame - left_frame) / srate)
            rows.append(self._annotation_row(onset, duration))
        rows.sort(key=lambda item: float(item["onset"]))
        return rows

    def _rows_from_blink_fits(self, blink_fits, frames: dict) -> list[dict[str, str]]:
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
            rows.append(self._annotation_row(onset, duration))
        rows.sort(key=lambda item: float(item["onset"]))
        return rows

    def _annotation_row(self, onset: float, duration: float) -> dict[str, str]:
        return {
            "onset": f"{onset:.6f}",
            "duration": f"{duration:.6f}",
            "description": "eye_blink",
        }

    def _pyblinker_sample_rate(self, frames: dict) -> float:
        metrics = frames.get("metrics")
        if isinstance(metrics, dict):
            value = self._numeric_value(metrics.get("sampling_rate_hz"))
            if value is not None and value > 0:
                return value
        return self._blinker_sample_rate(frames)

    def _pyblinker_bounds(
        self, event: dict[str, object], srate: float
    ) -> Optional[tuple[float, float]]:
        for left_name, right_name in (
            ("start_blink", "end_blink"),
            ("left_base", "right_base"),
            ("left_zero", "right_zero"),
            ("outer_start", "outer_end"),
        ):
            left = self._numeric_value(event.get(left_name))
            right = self._numeric_value(event.get(right_name))
            if left is not None and right is not None and right > left:
                return left, right

        peak = self._numeric_value(event.get("max_blink"))
        if peak is None:
            peak = self._numeric_value(event.get("peak_time_blink"))
        if peak is None:
            return None
        half_width = 0.15 * srate
        return max(0.0, peak - half_width), peak + half_width

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
        if self._epoch_health_mode:
            current_epoch = int(
                self.time_series_viewer.current_cursor_time() // EPOCH_WINDOW_SECONDS
            )
            target_epoch = max(0, current_epoch + (1 if direction > 0 else -1))
            self._goto_epoch(target_epoch)
        else:
            step = self.step_seconds_input.value() * (1 if direction > 0 else -1)
            self._goto_time(self.time_series_viewer.current_cursor_time() + step)

    def _goto_epoch(self, epoch_index: int) -> None:
        epoch_start = epoch_index * EPOCH_WINDOW_SECONDS
        # Always lock the viewer span to exactly one epoch window, even if the
        # user previously zoomed (Ctrl+scroll) while in epoch health mode.
        viewer = self.time_series_viewer
        viewer.view_span_seconds = EPOCH_WINDOW_SECONDS
        viewer.default_view_span_seconds = EPOCH_WINDOW_SECONDS
        self._goto_time(epoch_start + EPOCH_WINDOW_SECONDS / 2)
        # Windows 11 defers the pyqtgraph paint event until the next resize;
        # flush it now so the view updates immediately after every step.
        QApplication.processEvents()

    def _goto_time(self, seconds: float) -> None:
        if self.current_recording is None:
            self._set_status("Load a recording before navigating.")
            return
        self.time_series_viewer.seek_time(max(0.0, seconds))
        self._update_current_time_label()
        self._update_epoch_health_label()
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
            self.step_dec_button,
            self.step_inc_button,
            self.window_dropdown,
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
        elif self._epoch_health_mode:
            self.step_seconds_input.setEnabled(False)
            self.step_dec_button.setEnabled(False)
            self.step_inc_button.setEnabled(False)
            self.window_dropdown.setEnabled(False)

    def _update_selected_summary(
        self,
        recording: Optional[CaoRecording],
        loaded_ts: Optional[Path] = None,
        loaded_csv: Optional[Path] = None,
    ) -> None:
        if recording is None:
            self.summary_fif_label.setText("FIF: (none selected)")
            self.summary_csv_label.setText("CSV: (not loaded)")
            self.summary_blinker_label.setText("Blinker: (not loaded)")
            self.summary_epoch_health_label.setText("Epoch Health: (none selected)")
            return

        self.summary_fif_label.setText(
            self._format_path_line("FIF", recording.ts_path, loaded_ts is not None)
        )
        self.summary_csv_label.setText(
            self._format_path_line("CSV", recording.csv_path, loaded_csv is not None)
        )
        self.summary_blinker_label.setText(
            self._format_path_line("Blinker", recording.blinker_path, recording.blinker_path.exists())
        )
        self.summary_epoch_health_label.setText(
            self._format_epoch_health_summary(recording)
        )

    def _format_epoch_health_summary(self, recording: "CaoRecording") -> str:
        csv_path = recording.epoch_health_path
        exists = csv_path.exists()
        if not exists:
            return f"Epoch Health: {csv_path} [missing]"
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            good = sum(1 for r in rows if r.get("health") == "Good")
            bad = sum(1 for r in rows if r.get("health") == "Bad")
            total = len(rows)
            return (
                f"Epoch Health: {csv_path} [found, {total} epoch(s): "
                f"{good} Good, {bad} Bad]"
            )
        except Exception:
            return f"Epoch Health: {csv_path} [found, unreadable]"

    def _format_path_line(self, label: str, path: Path, loaded: bool, extra: str = "") -> str:
        exists = "found" if path.exists() else "missing"
        state = "loaded" if loaded else "expected"
        base = f"{label}: {path} [{state}, {exists}]"
        return f"{base} [{extra}]" if extra else base

    def _refresh_current_list_item(self) -> None:
        selected_items = self.recording_list.selectedItems()
        if not selected_items:
            return
        recording = selected_items[0].data(Qt.UserRole)
        if not isinstance(recording, CaoRecording):
            return
        csv_state = "CSV" if recording.csv_path.exists() else "No CSV"
        blinker_state = "Blinker" if recording.blinker_path.exists() else "No Blinker"
        selected_items[0].setText(
            f"{recording.display_name} | {recording.ts_path.name} | {csv_state} | {blinker_state}"
        )

    def _refresh_dataset_summary(self) -> None:
        totals = {
            "recordings": len(self.recordings),
            "csv": sum(1 for item in self.recordings if item.csv_path.exists()),
            "blinker": sum(1 for item in self.recordings if item.blinker_path.exists()),
            "fif": sum(1 for item in self.recordings if item.ts_path.exists()),
        }
        self.summary_overall_label.setText(
            "Dataset summary: "
            f"{totals['recordings']} recording(s) | "
            f"CSV: {totals['csv']}, "
            f"Blinker: {totals['blinker']}, "
            f"FIF: {totals['fif']}"
        )
        self.summary_table.setRowCount(len(self.recordings))
        for row, recording in enumerate(self.recordings):
            values = [
                recording.display_name,
                self._status_for_recording(recording),
                "yes" if recording.csv_path.exists() else "missing",
                "yes" if recording.blinker_path.exists() else "missing",
                "yes" if recording.ts_path.exists() else "missing",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.summary_table.setItem(row, col, item)
        self._refresh_statistics_tab()

    def _status_for_recording(self, recording: CaoRecording) -> str:
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

    def _increase_step(self) -> None:
        self.step_seconds_input.setValue(
            self.step_seconds_input.value() + self.step_seconds_input.singleStep()
        )

    def _decrease_step(self) -> None:
        self.step_seconds_input.setValue(
            max(self.step_seconds_input.minimum(),
                self.step_seconds_input.value() - self.step_seconds_input.singleStep())
        )

    def _on_window_changed(self, index: int) -> None:
        spans = [5.0, 10.0, 15.0, 30.0, 40.0, 50.0, 60.0]
        span = spans[index]
        viewer = self.time_series_viewer
        viewer.view_span_seconds = span
        viewer.default_view_span_seconds = span
        viewer.zoom_label.setText(viewer._zoom_label_text())
        viewer._ensure_view_range(viewer._last_cursor_time)

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

        self.delete_annotation_shortcut_slash = QShortcut(QKeySequence(Qt.Key_Slash), self)
        self.delete_annotation_shortcut_slash.setContext(Qt.WidgetWithChildrenShortcut)
        self.delete_annotation_shortcut_slash.activated.connect(self._delete_annotation_if_allowed)

        self.play_shortcut = QShortcut(QKeySequence(Qt.Key_P), self)
        self.play_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.play_shortcut.activated.connect(self._start_annotation_play)

        self.forward_play_shortcut = QShortcut(QKeySequence(Qt.Key_F), self)
        self.forward_play_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.forward_play_shortcut.activated.connect(self._toggle_forward_play)

        self.forward_play_shortcut_q = QShortcut(QKeySequence(Qt.Key_Q), self)
        self.forward_play_shortcut_q.setContext(Qt.WidgetWithChildrenShortcut)
        self.forward_play_shortcut_q.activated.connect(self._toggle_forward_play)

        self.stop_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.stop_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.stop_shortcut.activated.connect(self._stop_all_play)

        self.epoch_good_shortcut = QShortcut(QKeySequence(Qt.Key_J), self)
        self.epoch_good_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.epoch_good_shortcut.activated.connect(lambda: self._log_epoch_health_if_allowed("Good"))

        self.epoch_bad_shortcut = QShortcut(QKeySequence(Qt.Key_K), self)
        self.epoch_bad_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.epoch_bad_shortcut.activated.connect(lambda: self._log_epoch_health_if_allowed("Bad"))

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

    def _update_epoch_health_label(self) -> None:
        if self.current_recording is None:
            self.epoch_health_label.setText("")
            self.epoch_health_label.setStyleSheet("")
            self._update_epoch_timeline()
            return

        current_time = self.time_series_viewer.current_cursor_time()
        epoch_index = int(current_time // EPOCH_WINDOW_SECONDS)
        epoch_start = epoch_index * EPOCH_WINDOW_SECONDS
        epoch_end = epoch_start + EPOCH_WINDOW_SECONDS

        health = self._read_epoch_health(self.current_recording, epoch_index)
        if health == "Good":
            self.epoch_health_label.setStyleSheet(
                "background-color: #c8e6c9; color: #1b5e20; font-weight: bold; font-size: 13px;"
            )
            self.epoch_health_label.setText(
                f"Epoch {epoch_index}  ({epoch_start:.0f}s – {epoch_end:.0f}s)  ✓ Good"
            )
        elif health == "Bad":
            self.epoch_health_label.setStyleSheet(
                "background-color: #ffcdd2; color: #b71c1c; font-weight: bold; font-size: 13px;"
            )
            self.epoch_health_label.setText(
                f"Epoch {epoch_index}  ({epoch_start:.0f}s – {epoch_end:.0f}s)  ✗ Bad"
            )
        else:
            self.epoch_health_label.setStyleSheet(
                "background-color: #f5f5f5; color: #616161; font-size: 13px;"
            )
            self.epoch_health_label.setText(
                f"Epoch {epoch_index}  ({epoch_start:.0f}s – {epoch_end:.0f}s)  — unlabelled  [J = Good, K = Bad]"
            )
        self._update_epoch_timeline()

    def _read_epoch_health(self, recording: "CaoRecording", epoch_index: int) -> str:
        csv_path = recording.epoch_health_path
        if not csv_path.exists():
            return ""
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if (row.get("subject_id") == recording.subject_id
                            and row.get("session_id") == recording.session_id
                            and row.get("epoch_index") == str(epoch_index)):
                        return row.get("health", "")
        except Exception:
            pass
        return ""

    def _log_epoch_health_if_allowed(self, health: str) -> None:
        if self._shortcut_allowed():
            self._log_epoch_health(health)

    def _log_epoch_health(self, health: str) -> None:
        if self.current_recording is None:
            self._set_status("Load a recording before logging epoch health.")
            return

        current_time = self.time_series_viewer.current_cursor_time()
        epoch_index = int(current_time // EPOCH_WINDOW_SECONDS)
        epoch_start = epoch_index * EPOCH_WINDOW_SECONDS
        epoch_end = epoch_start + EPOCH_WINDOW_SECONDS

        recording = self.current_recording
        csv_path = recording.epoch_health_path
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows: list[dict] = []
        if csv_path.exists():
            try:
                with csv_path.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                rows = []

        updated = False
        for row in rows:
            if (row.get("subject_id") == recording.subject_id
                    and row.get("session_id") == recording.session_id
                    and row.get("epoch_index") == str(epoch_index)):
                row["health"] = health
                row["timestamp"] = timestamp
                updated = True
                break

        if not updated:
            rows.append({
                "subject_id": recording.subject_id,
                "session_id": recording.session_id,
                "epoch_index": str(epoch_index),
                "epoch_start_s": f"{epoch_start:.3f}",
                "epoch_end_s": f"{epoch_end:.3f}",
                "health": health,
                "timestamp": timestamp,
            })

        rows.sort(key=lambda r: (r.get("subject_id", ""), r.get("session_id", ""), int(r.get("epoch_index", 0))))

        fieldnames = ["subject_id", "session_id", "epoch_index", "epoch_start_s", "epoch_end_s", "health", "timestamp"]
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            self._set_status(f"Failed to save epoch health: {exc}")
            return

        self._update_epoch_health_label()
        if self.current_recording is not None:
            self.summary_epoch_health_label.setText(
                self._format_epoch_health_summary(self.current_recording)
            )
        self._set_status(
            f"Epoch {epoch_index} ({epoch_start:.0f}s–{epoch_end:.0f}s) marked as {health}."
        )

    def _toggle_epoch_health_mode(self, checked: bool) -> None:
        self._epoch_health_mode = checked
        if checked:
            self._pre_mode_step = self.step_seconds_input.value()
            self._pre_mode_window_index = self.window_dropdown.currentIndex()
            self.step_seconds_input.setValue(EPOCH_WINDOW_SECONDS)
            self.window_dropdown.setCurrentIndex(3)  # 30 s
            # Explicitly reset span in case the dropdown was already at index 3
            # (currentIndexChanged won't fire when the index doesn't change).
            self.time_series_viewer.view_span_seconds = EPOCH_WINDOW_SECONDS
            self.time_series_viewer.default_view_span_seconds = EPOCH_WINDOW_SECONDS
            self.step_seconds_input.setEnabled(False)
            self.step_dec_button.setEnabled(False)
            self.step_inc_button.setEnabled(False)
            self.window_dropdown.setEnabled(False)
            self.epoch_health_mode_button.setText("Exit Epoch Mode")
            current_epoch = int(
                self.time_series_viewer.current_cursor_time() // EPOCH_WINDOW_SECONDS
            )
            self._goto_epoch(current_epoch)
            self._set_status("Epoch Health Mode active — step and window locked to 30 s.")
        else:
            self.step_seconds_input.setValue(
                getattr(self, "_pre_mode_step", self.DEFAULT_STEP_SECONDS)
            )
            self.window_dropdown.setCurrentIndex(
                getattr(self, "_pre_mode_window_index", 0)
            )
            self.step_seconds_input.setEnabled(True)
            self.step_dec_button.setEnabled(True)
            self.step_inc_button.setEnabled(True)
            self.window_dropdown.setEnabled(True)
            self.epoch_health_mode_button.setText("Epoch Health Mode")
            self._set_status("Epoch Health Mode disabled.")
        self._update_epoch_timeline()

    def _update_epoch_timeline(self) -> None:
        if not self._epoch_health_mode or self.current_recording is None:
            self.epoch_timeline_widget.setVisible(False)
            return
        duration = self.time_series_viewer.signal_duration_seconds()
        if duration is None or duration <= 0:
            self.epoch_timeline_widget.setVisible(False)
            return
        total = math.ceil(duration / EPOCH_WINDOW_SECONDS)
        current_time = self.time_series_viewer.current_cursor_time()
        current_epoch = int(current_time // EPOCH_WINDOW_SECONDS)
        states: list[str] = [""] * total
        csv_path = self.current_recording.epoch_health_path
        if csv_path.exists():
            try:
                with csv_path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if (
                            row.get("subject_id") == self.current_recording.subject_id
                            and row.get("session_id") == self.current_recording.session_id
                        ):
                            idx = self._numeric_value(row.get("epoch_index"))
                            if idx is not None:
                                i = int(idx)
                                if 0 <= i < total:
                                    states[i] = row.get("health", "")
            except Exception:
                pass
        self.epoch_timeline_widget.set_data(total, states, current_epoch)
        self.epoch_timeline_widget.setVisible(True)

    def _on_epoch_timeline_clicked(self, epoch_index: int) -> None:
        self._goto_epoch(epoch_index)

    def _set_status(self, message: str) -> None:
        self.status_bar.showMessage(message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_all_play()
        self._save_session_state()
        super().closeEvent(event)
