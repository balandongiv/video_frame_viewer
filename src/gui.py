"""PyQt5 GUI for the video frame viewer application."""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml
from PyQt5.QtCore import QEvent, QPoint, QSize, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig, save_config
from paths import derive_annotation_path, derive_time_series_path
from time_series import TimeSeriesViewer
from utils import (
    extract_subject_label,
    find_md_mff_videos,
    find_mov_videos,
    frame_to_pixmap,
    seconds_to_frame_index,
    subject_label_sort_key,
    subject_sort_key,
)
from video_handler import VideoHandler

SESSION_FILENAME = "VideoFrameViewers.yaml"


class PannableLabel(QLabel):
    """QLabel that supports click-and-drag panning when larger than its viewport."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scroll_area: Optional[QScrollArea] = None
        self._dragging = False
        self._drag_start_pos = None
        self._h_start = 0
        self._v_start = 0

    def set_scroll_area(self, scroll_area: QScrollArea) -> None:
        self._scroll_area = scroll_area

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._scroll_area is not None:
            self._dragging = True
            self._drag_start_pos = event.pos()
            self._h_start = self._scroll_area.horizontalScrollBar().value()
            self._v_start = self._scroll_area.verticalScrollBar().value()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._dragging and self._scroll_area is not None and self._drag_start_pos is not None:
            delta = event.pos() - self._drag_start_pos
            h_bar = self._scroll_area.horizontalScrollBar()
            v_bar = self._scroll_area.verticalScrollBar()
            h_bar.setValue(self._h_start - delta.x())
            v_bar.setValue(self._v_start - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self._drag_start_pos = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


class VideoFrameViewer(QMainWindow):
    """Main window for browsing and navigating video frames."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEST_DATASET_ROOT = PROJECT_ROOT / "test_data" / "drowsy_driving_raja"
    TEST_PROCESSED_ROOT = PROJECT_ROOT / "test_data" / "drowsy_driving_raja_processed"
    SINGLE_STEP = 1
    JUMP_STEP = 10
    LARGE_STEP = 30
    TIME_BASE_FPS = 30
    MIN_ZOOM = 0.25
    MAX_ZOOM = 10.0

    def __init__(
        self,
        config: AppConfig,
        config_path: Optional[Path] = None,
        config_source: str = "",
    ) -> None:
        super().__init__()
        self.setWindowTitle("Video Frame Viewer")

        self.config = config
        self.config_path = config_path
        self.config_source = config_source
        self.video_handler = VideoHandler()
        self.video_paths: List[Path] = []
        self.current_frame_index: int = 0
        self.shift_value: int = 0
        self.left_jump_size: int = self.JUMP_STEP
        self.right_jump_size: int = self.JUMP_STEP
        self.sync_offset_seconds: float = 0.0
        self.zoom_factor: float = 1.0
        self.last_frame = None
        self.status_value: str = "Pending"
        self.summary_subject_counts: dict[str, dict[str, int]] = {}
        self.time_series_viewer = TimeSeriesViewer(
            time_series_root=self.config.time_series_root,
            annotation_root=self.config.annotation_root,
            ui_settings=self.config.ui,
        )
        self.time_series_viewer.ui_setting_changed.connect(self._save_ui_setting)
        self.use_test_data = False

        self._annotation_play_timer = QTimer(self)
        self._annotation_play_timer.timeout.connect(self._annotation_play_tick)

        self._forward_play_timer = QTimer(self)
        self._forward_play_timer.timeout.connect(self._forward_play_tick)

        self._setup_ui()
        self._setup_shortcuts()
        self.frame_scroll.viewport().installEventFilter(self)
        self.time_series_viewer.annotation_jump_requested.connect(
            self._jump_to_annotation_time
        )
        self._update_navigation_state(False)
        self._initialize_dataset_root()

    def _save_ui_setting(self, key: str, value: object) -> None:
        self.config.ui[key] = value
        try:
            saved_path = save_config(self.config, self.config_path)
            self.config_path = saved_path
        except OSError as exc:
            self.statusBar().showMessage(
                f"Could not save UI setting '{key}': {exc}",
                5000,
            )

    # UI setup
    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self._build_directory_controls(main_layout)

        self._main_splitter = QSplitter(Qt.Vertical)
        self._main_splitter.setChildrenCollapsible(False)

        top_container = QWidget()
        top_layout = QVBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_container.setLayout(top_layout)

        self.upper_splitter = QSplitter(Qt.Horizontal)
        self.upper_splitter.setChildrenCollapsible(False)

        self.side_tabs = self._build_side_tabs()
        self.frame_panel = self._build_frame_panel()

        self.side_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.side_tabs.setMinimumWidth(200)
        self.frame_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.upper_splitter.addWidget(self.side_tabs)
        self.upper_splitter.addWidget(self.frame_panel)
        self.upper_splitter.setStretchFactor(0, 1)
        self.upper_splitter.setStretchFactor(1, 4)

        top_layout.addWidget(self.upper_splitter)

        self.time_series_group = self._build_time_series_panel()

        self._main_splitter.addWidget(top_container)
        self._main_splitter.addWidget(self.time_series_group)
        self._main_splitter.setStretchFactor(0, 5)
        self._main_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self._main_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.side_tabs.currentChanged.connect(self._on_side_tab_changed)
        self._on_side_tab_changed(self.side_tabs.currentIndex())
        QTimer.singleShot(0, self._apply_upper_splitter_ratio)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_upper_splitter_ratio()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_upper_splitter_ratio)

    def _apply_upper_splitter_ratio(self) -> None:
        total_width = self.upper_splitter.width()
        if total_width <= 0:
            total_width = self.width()
        if total_width <= 0:
            total_width = 1
        side_width = max(1, int(total_width * 0.2))
        frame_width = max(1, total_width - side_width)
        self.upper_splitter.setSizes([side_width, frame_width])

    def _build_directory_controls(self, parent_layout: QVBoxLayout) -> None:
        directory_group = QGroupBox("Dataset Directory")
        directory_layout = QHBoxLayout()
        directory_group.setLayout(directory_layout)

        self.directory_input = QLineEdit()
        self.directory_input.setPlaceholderText("Select or enter dataset root")
        self.directory_input.returnPressed.connect(self._scan_directory)
        browse_button = QPushButton("Browse")
        scan_button = QPushButton("Rescan")
        self.debug_toggle = QCheckBox("Use test data")

        browse_button.clicked.connect(self._browse_directory)
        scan_button.clicked.connect(self._scan_directory)
        self.debug_toggle.stateChanged.connect(self._toggle_debug_data)

        directory_layout.addWidget(QLabel("Root:"))
        directory_layout.addWidget(self.directory_input)
        directory_layout.addWidget(browse_button)
        directory_layout.addWidget(scan_button)
        directory_layout.addWidget(self.debug_toggle)

        parent_layout.addWidget(directory_group)

    def _build_video_list_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(QLabel("Discovered Videos"))

        self.video_list = QListWidget()
        self.video_list.itemSelectionChanged.connect(self._load_selected_video)
        layout.addWidget(self.video_list)

        return container

    def _build_frame_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(QLabel("Frame Display"))

        frame_area = QWidget()
        frame_area_layout = QVBoxLayout()
        frame_area_layout.setContentsMargins(0, 0, 0, 0)
        frame_area_layout.setSpacing(4)
        frame_area.setLayout(frame_area_layout)

        self.frame_label = PannableLabel("Scan and select a video to view frames.")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_label.setMinimumSize(960, 540)

        self.frame_scroll = QScrollArea()
        self.frame_scroll.setWidgetResizable(False)
        self.frame_scroll.setWidget(self.frame_label)
        self.frame_scroll.setAlignment(Qt.AlignCenter)
        self.frame_label.set_scroll_area(self.frame_scroll)

        self.frame_info_label = QLabel("Frame: -")
        self.frame_info_label.setAlignment(Qt.AlignCenter)

        frame_area_layout.addWidget(self.frame_scroll)
        frame_area_layout.addWidget(self.frame_info_label)

        layout.addWidget(frame_area)

        return container

    def _build_time_series_panel(self) -> QWidget:
        time_series_group = QGroupBox("Time Series")
        time_series_layout = QVBoxLayout()
        self.time_series_viewer.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding
        )
        self.time_series_viewer.setMinimumHeight(140)
        time_series_layout.addWidget(self.time_series_viewer)
        time_series_group.setLayout(time_series_layout)
        time_series_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        return time_series_group

    def _build_side_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        videos_tab = QWidget()
        videos_layout = QVBoxLayout()
        videos_layout.setContentsMargins(0, 0, 0, 0)
        videos_layout.setSpacing(6)
        videos_layout.addWidget(self._build_video_list_panel())
        videos_layout.addWidget(self._build_control_panel())
        videos_layout.setStretch(0, 1)
        videos_layout.setStretch(1, 0)
        videos_tab.setLayout(videos_layout)

        channels_tab = QWidget()
        channels_layout = QVBoxLayout()
        channels_layout.setContentsMargins(0, 0, 0, 0)
        channels_layout.setSpacing(6)
        channels_layout.addWidget(self.time_series_viewer.channel_controls())
        channels_layout.addStretch()
        channels_tab.setLayout(channels_layout)

        navigation_tab = self._build_navigation_tab()
        summary_tab = self._build_summary_tab()
        help_tab = self._build_help_tab()

        tabs.addTab(videos_tab, "Videos")
        tabs.addTab(channels_tab, "Channels")
        tabs.addTab(navigation_tab, "Navigation")
        tabs.addTab(summary_tab, "Summary")
        tabs.addTab(help_tab, "Help")

        return tabs

    def _build_navigation_tab(self) -> QWidget:
        navigation_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        jump_group = QGroupBox("Jump Settings")
        jump_layout = QGridLayout()
        jump_group.setLayout(jump_layout)

        self.left_jump_input = QSpinBox()
        self.left_jump_input.setMinimum(1)
        self.left_jump_input.setMaximum(1_000_000)
        self.left_jump_input.setValue(self.left_jump_size)
        self.left_jump_input.valueChanged.connect(self._update_left_jump_size)

        self.right_jump_input = QSpinBox()
        self.right_jump_input.setMinimum(1)
        self.right_jump_input.setMaximum(1_000_000)
        self.right_jump_input.setValue(self.right_jump_size)
        self.right_jump_input.valueChanged.connect(self._update_right_jump_size)

        jump_layout.addWidget(QLabel("Setting"), 0, 0)
        jump_layout.addWidget(QLabel("Frames"), 0, 1)
        jump_layout.addWidget(QLabel("Shortcut"), 0, 2)

        jump_layout.addWidget(QLabel("Left Jump (frames):"), 1, 0)
        jump_layout.addWidget(self.left_jump_input, 1, 1)
        jump_layout.addWidget(QLabel("←"), 1, 2)

        jump_layout.addWidget(QLabel("Right Jump (frames):"), 2, 0)
        jump_layout.addWidget(self.right_jump_input, 2, 1)
        jump_layout.addWidget(QLabel("→"), 2, 2)

        jump_layout.addWidget(QLabel("Quick Step (frames):"), 3, 0)
        jump_layout.addWidget(QLabel(str(self.LARGE_STEP)), 3, 1)
        jump_layout.addWidget(QLabel("Ctrl+Shift+←/→"), 3, 2)

        layout.addWidget(jump_group)
        layout.addStretch()
        navigation_tab.setLayout(layout)
        return navigation_tab

    def _build_summary_tab(self) -> QWidget:
        summary_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.summary_video_label = QLabel("Video: (none selected)")
        self.summary_video_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_fif_label = QLabel("FIF: (not loaded)")
        self.summary_fif_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_csv_label = QLabel("CSV: (not loaded)")
        self.summary_csv_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_overall_label = QLabel("Dataset summary: (scan dataset to populate)")
        self.summary_overall_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_overall_label.setWordWrap(True)

        self.summary_table = QTableWidget(0, 9)
        self.summary_table.setHorizontalHeaderLabels(
            [
                "Subject",
                "Pending",
                "daphne_check",
                "Ongoing",
                "Complete",
                "complete_eeg",
                "Issue",
                "Missing CSV",
                "Missing FIF",
            ]
        )
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setSelectionMode(QTableWidget.NoSelection)

        remark_group = QGroupBox("Remarks")
        remark_layout = QVBoxLayout()
        remark_group.setLayout(remark_layout)
        self.remark_input = QTextEdit()
        self.remark_input.setPlaceholderText("Enter remarks for this video session...")
        self.remark_input.setMinimumHeight(80)
        remark_buttons = QHBoxLayout()
        self.remark_save_button = QPushButton("Save Remark")
        self.remark_save_button.clicked.connect(self._save_remark)
        remark_buttons.addStretch()
        remark_buttons.addWidget(self.remark_save_button)
        self.remark_history_label = QLabel("Saved remarks:")
        self.remark_history_list = QListWidget()
        self.remark_history_list.setSelectionMode(QListWidget.NoSelection)
        self.remark_history_list.setMinimumHeight(120)
        remark_layout.addWidget(self.remark_input)
        remark_layout.addLayout(remark_buttons)
        remark_layout.addWidget(self.remark_history_label)
        remark_layout.addWidget(self.remark_history_list)

        layout.addWidget(self.summary_video_label)
        layout.addWidget(self.summary_fif_label)
        layout.addWidget(self.summary_csv_label)
        layout.addWidget(self.summary_overall_label)
        layout.addWidget(self.summary_table)
        layout.addWidget(remark_group)
        layout.addStretch()
        summary_tab.setLayout(layout)
        return summary_tab

    def _build_help_tab(self) -> QWidget:
        help_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        help_text = QTextBrowser()
        help_text.setOpenExternalLinks(False)
        help_text.setHtml(
            """
            <h2>Video Frame Viewer Help</h2>

            <h3>Basic Workflow</h3>
            <ol>
              <li>Select or enter the dataset root, then click <b>Rescan</b>.</li>
              <li>Select a video from <b>Discovered Videos</b>.</li>
              <li>Use frame/time controls or shortcuts to navigate.</li>
              <li>Use the time-series panel to inspect EEG/EOG/EAR signals and annotations.</li>
              <li>After annotation edits, click <b>Save annotations</b>.</li>
            </ol>

            <h3>Shortcuts</h3>
            <table border="1" cellspacing="0" cellpadding="4">
              <tr><th>Shortcut</th><th>Action</th></tr>
              <tr><td colspan="2"><i>Navigation</i></td></tr>
              <tr><td>Left / Right</td><td>Jump left or right by the configured jump size.</td></tr>
              <tr><td>Ctrl + Left / Ctrl + Right</td><td>Move one frame left or right.</td></tr>
              <tr><td>Ctrl + Shift + Left / Right</td><td>Move by the large step size.</td></tr>
              <tr><td colspan="2"><i>Annotation Navigation</i></td></tr>
              <tr><td>] or N</td><td>Go to next annotation.</td></tr>
              <tr><td>Ctrl + N</td><td>Go to next annotation and center on EAR minimum within it.</td></tr>
              <tr><td>[ or B</td><td>Go to previous annotation.</td></tr>
              <tr><td colspan="2"><i>Playback</i></td></tr>
              <tr><td>P</td><td>Start auto-play through annotations (jumps to each annotation in sequence).</td></tr>
              <tr><td>F</td><td>Toggle auto-forward through the time series (advances by the right jump size at the set speed).</td></tr>
              <tr><td>ESC</td><td>Stop all active playback (annotation play and forward play).</td></tr>
              <tr><td colspan="2"><i>Annotation Editing</i></td></tr>
              <tr><td>Space</td><td>Auto-repair the selected annotation.</td></tr>
              <tr><td>R</td><td>Auto-repair EOG for the selected annotation.</td></tr>
              <tr><td>E</td><td>Auto-repair EAR for the selected annotation (EAR-avg_ear: finds minimum, walks to local peaks).</td></tr>
              <tr><td>D</td><td>Delete the selected annotation.</td></tr>
              <tr><td>Ctrl + Z</td><td>Undo the last annotation creation (or merge).</td></tr>
              <tr><td colspan="2"><i>File</i></td></tr>
              <tr><td>Ctrl + S</td><td>Save annotations.</td></tr>
            </table>

            <h3>Creating and Editing Annotations</h3>
            <ul>
              <li>Ctrl + left-drag on the main time-series plot to create an annotation.</li>
              <li>Drag annotation boundaries to manually adjust onset/duration.</li>
              <li>Press <b>D</b> to delete the selected annotation.</li>
              <li>Right-click an annotation for <b>Edit label</b>, <b>auto_repair</b>,
                  <b>auto_repair_eog</b>, <b>auto_repair_ear</b>, <b>Revert auto_repair</b>, or <b>Delete annotation</b>.</li>
              <li>The <b>Unsaved changes</b> label means annotations changed in memory but are not saved yet.</li>
            </ul>

            <h3>Auto Repair</h3>
            <p>
              <b>auto_repair</b> uses <b>EEG-E8</b> with zero crossings.
              <b>auto_repair_eog</b> uses <b>EOG-EEG-eog_vert_left</b>: finds the maximum
              amplitude within the annotation window, then walks left and right to the first
              local minimum in each direction, expanding beyond the annotation if needed.
              <b>auto_repair_ear</b> uses <b>EAR-avg_ear</b>: finds the minimum amplitude
              within the annotation window, then walks left and right to the first local
              maximum in each direction, expanding beyond the annotation if needed.
            </p>
            <ul>
              <li><b>Space</b>: repair the currently selected annotation.</li>
              <li><b>R</b>: run auto_repair_eog on the currently selected annotation.</li>
              <li><b>E</b>: run auto_repair_ear on the currently selected annotation.</li>
              <li><b>Right-click &gt; auto_repair</b>: repair the clicked annotation.</li>
              <li><b>Right-click &gt; auto_repair_eog</b>: repair the clicked annotation using EOG-EEG-eog_vert_left.</li>
              <li><b>Right-click &gt; auto_repair_ear</b>: repair the clicked annotation using EAR-avg_ear.</li>
              <li><b>Bulk auto_repair</b>: repair all annotations using the EOG peak-trough algorithm (same as R). This does <b>not</b> save automatically.</li>
              <li>Bulk auto_repair never merges annotations. Each annotation remains separate.</li>
              <li><b>Right-click &gt; Revert auto_repair</b>: restore that annotation's original onset/duration.</li>
              <li><b>Revert all auto_repair</b>: restore every auto-repaired annotation to its original bounds.</li>
            </ul>

            <h3>Bulk Auto Repair Review</h3>
            <ul>
              <li><b>Blue annotation</b>: auto-repaired; revert is available.</li>
              <li><b>SPLIT boundary used</b> in the tooltip/status means one side had no safe
                  zero crossing before the neighboring blink, so the boundary was kept at the
                  split point instead of borrowing the neighbor's crossing.</li>
              <li><b>Yellow annotation with red border</b>: auto-repaired and overlaps another annotation;
                  review before saving.</li>
              <li>The status line reports changed, already aligned, failed, deleted, merged, and overlap-review counts.</li>
              <li>Current auto repair does not automatically delete or merge annotations. Overlaps are marked only for review.</li>
              <li>If an expanded annotation covers a neighbor and is hard to inspect, click
                  <b>Revert all auto_repair</b> before saving.</li>
            </ul>

            <h3>Status Dropdown</h3>
            <p>
              Use the status dropdown to mark the current video session:
              <b>Pending</b>, <b>daphne_check</b>, <b>Ongoing</b>, <b>Complete</b>, <b>complete_eeg</b>, or <b>Issue</b>.
            </p>

            <h3>Saving</h3>
            <p>
              Auto repair, bulk repair, manual boundary edits, label edits, and deletes stay in memory until
              <b>Save annotations</b> is clicked. Saving writes the annotation CSV and clears auto-repair
              visual review markers.
            </p>
            """
        )
        layout.addWidget(help_text)
        help_tab.setLayout(layout)
        return help_tab

    def _build_control_panel(self) -> QGroupBox:
        control_group = QGroupBox("Frame Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("Enter frame number (0-based)")
        self.frame_input.returnPressed.connect(self._search_frame)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._search_frame)

        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("Enter time in seconds")
        self.time_input.returnPressed.connect(self._search_time)
        self.time_search_button = QPushButton("Search Time")
        self.time_search_button.clicked.connect(self._search_time)

        self.shift_input = QLineEdit()
        self.shift_input.setPlaceholderText("Shift frame (can be negative)")
        self.shift_input.returnPressed.connect(self._apply_shift)
        apply_shift_button = QPushButton("Apply Shift")
        apply_shift_button.clicked.connect(self._apply_shift)

        self.sync_offset_input = QLineEdit()
        self.sync_offset_input.setPlaceholderText(
            "Sync offset in seconds (can be negative)"
        )
        self.sync_offset_input.returnPressed.connect(self._apply_sync_offset)
        apply_sync_offset_button = QPushButton("Apply Sync Offset")
        apply_sync_offset_button.clicked.connect(self._apply_sync_offset)

        self.csv_picker_button = QPushButton("Pick CSV")
        self.csv_picker_button.clicked.connect(self._browse_annotation_csv)

        control_layout.addWidget(QLabel("Frame Number:"), 0, 0)
        control_layout.addWidget(self.frame_input, 0, 1)
        control_layout.addWidget(self.search_button, 0, 2)

        control_layout.addWidget(QLabel("Time (seconds):"), 1, 0)
        control_layout.addWidget(self.time_input, 1, 1)
        control_layout.addWidget(self.time_search_button, 1, 2)

        control_layout.addWidget(QLabel("Shift Frame:"), 2, 0)
        control_layout.addWidget(self.shift_input, 2, 1)
        control_layout.addWidget(apply_shift_button, 2, 2)

        control_layout.addWidget(QLabel("Sync Offset (s):"), 3, 0)
        control_layout.addWidget(self.sync_offset_input, 3, 1)
        control_layout.addWidget(apply_sync_offset_button, 3, 2)

        self.status_dropdown = QComboBox()
        self.status_dropdown.addItems(
            ["Pending", "daphne_check", "Ongoing", "Complete", "complete_eeg", "Issue"]
        )
        self.status_dropdown.currentTextChanged.connect(self._on_status_changed)
        control_layout.addWidget(QLabel("Status:"), 4, 0)
        control_layout.addWidget(self.status_dropdown, 4, 1, 1, 2)

        control_layout.addWidget(QLabel("Segment CSV:"), 5, 0)
        control_layout.addWidget(self.csv_picker_button, 5, 1, 1, 2)

        play_layout = QHBoxLayout()
        self.play_button = QPushButton("Play Annotations")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self._toggle_annotation_play)
        self.play_speed_spinbox = QDoubleSpinBox()
        self.play_speed_spinbox.setRange(0.5, 30.0)
        self.play_speed_spinbox.setValue(2.0)
        self.play_speed_spinbox.setSingleStep(0.5)
        self.play_speed_spinbox.setSuffix(" s")
        self.play_speed_spinbox.setToolTip("Seconds between each annotation jump")
        play_layout.addWidget(self.play_button)
        play_layout.addWidget(QLabel("Speed:"))
        play_layout.addWidget(self.play_speed_spinbox)
        control_layout.addLayout(play_layout, 6, 0, 1, 3)

        forward_play_layout = QHBoxLayout()
        self.forward_play_button = QPushButton("Forward Play")
        self.forward_play_button.setCheckable(True)
        self.forward_play_button.clicked.connect(self._toggle_forward_play_button)
        self.forward_play_speed_spinbox = QDoubleSpinBox()
        self.forward_play_speed_spinbox.setRange(0.1, 30.0)
        self.forward_play_speed_spinbox.setValue(0.5)
        self.forward_play_speed_spinbox.setSingleStep(0.1)
        self.forward_play_speed_spinbox.setSuffix(" s")
        self.forward_play_speed_spinbox.setToolTip("Seconds between each forward step (F to toggle)")
        forward_play_layout.addWidget(self.forward_play_button)
        forward_play_layout.addWidget(QLabel("Speed:"))
        forward_play_layout.addWidget(self.forward_play_speed_spinbox)
        control_layout.addLayout(forward_play_layout, 7, 0, 1, 3)

        navigation_layout = QHBoxLayout()
        self.left_button = QPushButton("Left")
        self.right_button = QPushButton("Right")
        self.left_jump_button = QPushButton("Left_Jump")
        self.right_jump_button = QPushButton("Right_Jump")

        self.left_button.clicked.connect(self._trigger_left_step)
        self.right_button.clicked.connect(self._trigger_right_step)
        self.left_jump_button.clicked.connect(self._trigger_left_jump)
        self.right_jump_button.clicked.connect(self._trigger_right_jump)

        control_layout.addLayout(navigation_layout, 8, 0, 1, 3)

        zoom_layout = QHBoxLayout()
        self.zoom_out_button = QPushButton("Zoom -")
        self.zoom_out_button.clicked.connect(lambda: self._adjust_zoom(-0.25))
        self.zoom_in_button = QPushButton("Zoom +")
        self.zoom_in_button.clicked.connect(lambda: self._adjust_zoom(0.25))
        self.zoom_reset_button = QPushButton("Reset Zoom")
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        self.zoom_label = QLabel(self._zoom_label_text())

        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_reset_button)
        zoom_layout.addWidget(self.zoom_label)

        control_layout.addLayout(zoom_layout, 8, 0, 1, 3)

        self.current_frame_label = QLabel("Current frame: -")
        control_layout.addWidget(self.current_frame_label, 9, 0, 1, 3)

        return control_group

    # Directory logic
    def _browse_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select dataset root", str(self.config.dataset_root)
        )
        if selected:
            self.directory_input.setText(selected)
            self._scan_directory()

    def _video_root_for(self, dataset_root: Path) -> Path:
        if self.config.mov_dir:
            return (dataset_root / self.config.mov_dir).expanduser()
        return dataset_root

    def _time_series_root_for(self, dataset_root: Path) -> Path:
        if self.config.fif_dir:
            return (dataset_root / self.config.fif_dir).expanduser()
        return dataset_root.parent / f"{dataset_root.name}_processed"

    def _annotation_root_for(self, dataset_root: Path) -> Path:
        if self.config.csv_dir:
            return (dataset_root / self.config.csv_dir).expanduser()
        return self._time_series_root_for(dataset_root)

    def _initialize_dataset_root(self) -> None:
        self.directory_input.setText(str(self.config.dataset_root))
        self.time_series_viewer.set_processed_root(self.config.time_series_root)
        self.time_series_viewer.set_annotation_root(self.config.annotation_root)
        if self.config_path:
            self._set_status(
                f"Loaded configuration from {self.config_path} (source: {self.config_source})."
            )
        else:
            self._set_status(f"Using configuration source: {self.config_source or 'default'}.")
        self._scan_directory()

    def _toggle_debug_data(self, state: int) -> None:
        self.use_test_data = state == Qt.Checked
        if self.use_test_data:
            self.directory_input.setText(str(self.TEST_DATASET_ROOT))
            self.time_series_viewer.set_processed_root(self.TEST_PROCESSED_ROOT)
            self.time_series_viewer.set_annotation_root(self.TEST_PROCESSED_ROOT)
            self._set_status("Debug mode enabled: using bundled test data.")
        else:
            self.directory_input.setText(str(self.config.dataset_root))
            self.time_series_viewer.set_processed_root(self.config.time_series_root)
            self.time_series_viewer.set_annotation_root(self.config.annotation_root)
            self._set_status("Debug mode disabled: using default dataset paths.")
        self._scan_directory()

    def _scan_directory(self) -> None:
        root_text = self.directory_input.text().strip()
        default_root = self.TEST_DATASET_ROOT if self.use_test_data else self.config.dataset_root
        dataset_root = (
            Path(root_text).expanduser().resolve()
            if root_text
            else default_root.expanduser().resolve()
        )
        self.directory_input.setText(str(dataset_root))

        if not dataset_root.exists():
            self.video_list.clear()
            self._set_status(
                f"Dataset root not found at {dataset_root}. Please choose another folder."
            )
            return

        if self.use_test_data:
            video_root = self.TEST_DATASET_ROOT
            time_series_root = self.TEST_PROCESSED_ROOT
            annotation_root = self.TEST_PROCESSED_ROOT
        else:
            video_root = self._video_root_for(dataset_root)
            time_series_root = self._time_series_root_for(dataset_root)
            annotation_root = self._annotation_root_for(dataset_root)

        self.time_series_viewer.set_processed_root(time_series_root)
        self.time_series_viewer.set_annotation_root(annotation_root)

        if not video_root.exists():
            self.video_list.clear()
            self._set_status(
                "Video directory not found at "
                f"{video_root}. Check dataset_root and mov_dir in your config."
            )
            return

        if self.use_test_data:
            self.video_paths = find_mov_videos(video_root)
        else:
            self.video_paths = find_md_mff_videos(video_root)

        self.video_list.clear()
        for video_path in sorted(self.video_paths, key=subject_sort_key):
            subject = extract_subject_label(video_path) or "Unknown"
            item = QListWidgetItem(f"{subject} | {video_path.name}")
            item.setData(Qt.UserRole, str(video_path))
            self.video_list.addItem(item)

        if self.video_paths:
            descriptor = "test .mov" if self.use_test_data else "MD.mff .mov"
            self._set_status(
                f"Found {len(self.video_paths)} {descriptor} file(s) in the dataset root."
            )
        else:
            descriptor = "test .mov" if self.use_test_data else "MD.mff .mov"
            self._set_status(f"No {descriptor} files found in the dataset root.")
        self._refresh_dataset_summary()
        self._update_summary(None, None, None, None, None)

    def _load_selected_video(self) -> None:
        selected_items = self.video_list.selectedItems()
        if not selected_items:
            return

        self._save_session_state()

        stored_path = selected_items[0].data(Qt.UserRole)
        video_path = Path(stored_path or selected_items[0].text())
        if not self.video_handler.load(video_path):
            self._set_status("Unable to open the selected video.")
            self._update_navigation_state(False)
            self.time_series_viewer.load_for_video(None)
            self._update_summary(video_path, None, None, None, None)
            return

        has_frames = self.video_handler.frame_count > 0
        self.current_frame_index = 0
        self._update_navigation_state(has_frames)
        self.time_series_viewer.load_for_video(video_path)
        expected_fif, expected_csv = self.time_series_viewer.expected_paths()
        loaded_fif, loaded_csv = self.time_series_viewer.last_loaded_paths()
        self._update_summary(video_path, expected_fif, expected_csv, loaded_fif, loaded_csv)

        if has_frames:
            self._goto_frame(0, show_status=False)
            self._load_session_state()
            self._set_status(
                f"Loaded video: {video_path.name}. Total frames: {self.video_handler.frame_count}"
            )
        else:
            self._set_status("The selected video contains no frames.")

    def _browse_annotation_csv(self) -> None:
        if not self.video_handler.capture or self.video_handler.video_path is None:
            self._set_status("Load a video before choosing a CSV file.")
            return

        if self.time_series_viewer.has_unsaved_annotation_changes():
            result = QMessageBox.question(
                self,
                "Replace CSV?",
                "Current annotation edits are not saved. Replace the loaded CSV anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                self._set_status("CSV replacement cancelled.")
                return

        _, loaded_csv = self.time_series_viewer.last_loaded_paths()
        start_dir = loaded_csv.parent if loaded_csv is not None else self.directory_input.text()
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select annotation CSV for this segment",
            str(start_dir),
            "CSV files (*.csv);;All files (*)",
        )
        if not selected:
            return

        csv_path = Path(selected).expanduser().resolve()
        if not self.time_series_viewer.load_annotation_file(csv_path):
            self._set_status(f"Failed to load CSV: {csv_path}")
            return

        expected_fif, expected_csv = self.time_series_viewer.expected_paths()
        loaded_fif, loaded_csv = self.time_series_viewer.last_loaded_paths()
        self._update_summary(
            self.video_handler.video_path, expected_fif, expected_csv, loaded_fif, loaded_csv
        )
        self._load_session_state()
        self._set_status(f"Loaded replacement CSV for this segment: {csv_path}")

    def _save_session_state(self) -> None:
        _, loaded_csv = self.time_series_viewer.last_loaded_paths()
        if loaded_csv is None or not loaded_csv.parent.exists():
            return

        session_path = loaded_csv.parent / SESSION_FILENAME
        existing_data = {}
        if session_path.exists():
            try:
                with session_path.open("r", encoding="utf-8") as f:
                    existing_data = yaml.safe_load(f) or {}
            except Exception:
                existing_data = {}
        remarks = existing_data.get("remarks")
        if not isinstance(remarks, list):
            remarks = []
        data = {
            "shift_frame": self.shift_value,
            "stop_position": self.current_frame_index,
            "status": self.status_value,
            "remark": self.remark_input.toPlainText().strip(),
            "remarks": remarks,
        }
        try:
            with session_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)
        except Exception as e:
            self._set_status(f"Failed to save session state: {e}")
        self._refresh_dataset_summary()

    def _load_session_state(self) -> None:
        _, loaded_csv = self.time_series_viewer.last_loaded_paths()
        if loaded_csv is None:
            return

        session_path = loaded_csv.parent / SESSION_FILENAME
        if not session_path.exists():
            # Reset shift to 0 and status to Pending for new sessions without history
            self.shift_input.setText("0")
            self._apply_shift()
            self.status_dropdown.setCurrentText("Pending")
            self.remark_input.setPlainText("")
            self._refresh_remark_history([])
            return

        try:
            with session_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            shift = data.get("shift_frame", 0)
            stop_pos = data.get("stop_position", 0)
            status = data.get("status", "Pending")
            remark = data.get("remark", "")

            self.shift_input.setText(str(shift))
            self._apply_shift()

            self.status_dropdown.setCurrentText(status)
            self.status_value = status
            self.remark_input.setPlainText(str(remark))
            remarks = self._extract_remarks(data)
            self._refresh_remark_history(remarks)

            if stop_pos > 0:
                self._goto_frame(stop_pos, show_status=False)

            self._set_status(f"Resumed session: shift {shift}, frame {stop_pos}, status {status}")

        except Exception as e:
            self._set_status(f"Failed to load session state: {e}")
            self._refresh_remark_history([])

    def _on_status_changed(self, text: str) -> None:
        self.status_value = text
        self._save_session_state()

    def _save_remark(self) -> None:
        if not self.video_handler.capture:
            self._set_status("Load a video before saving remarks.")
            return
        remark_text = self.remark_input.toPlainText().strip()
        if not remark_text:
            self._set_status("Enter a remark before saving.")
            return
        _, loaded_csv = self.time_series_viewer.last_loaded_paths()
        if loaded_csv is None:
            self._set_status("Remarks storage is unavailable for this video.")
            return
        session_path = loaded_csv.parent / SESSION_FILENAME
        data = {}
        if session_path.exists():
            try:
                with session_path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                data = {}
        remarks = self._extract_remarks(data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_seconds = self._current_time_seconds()
        remark_entry = (
            f"{timestamp} — Frame {self.current_frame_index} "
            f"({session_seconds:.2f}s) — {remark_text}"
        )
        remarks.append(remark_entry)
        data.update(
            {
                "shift_frame": self.shift_value,
                "stop_position": self.current_frame_index,
                "status": self.status_value,
                "remark": remark_text,
                "remarks": remarks,
            }
        )
        try:
            with session_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)
        except Exception as e:
            self._set_status(f"Failed to save remark: {e}")
            return
        self._save_session_state()
        self._refresh_remark_history(remarks)
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

    # Shift and navigation logic
    def _apply_shift(self) -> None:
        shift_text = self.shift_input.text().strip()
        if not shift_text:
            self.shift_value = 0
            self.sync_offset_seconds = 0.0
            self.sync_offset_input.setText("")
            self._set_status("Shift cleared (0).")
            self._update_time_series_cursor()
            self._update_frame_label()
            return

        try:
            self.shift_value = int(shift_text)
        except ValueError:
            self._set_status("Invalid shift value. Please enter an integer.")
            return

        self.sync_offset_seconds = self.shift_value / self.TIME_BASE_FPS
        self.sync_offset_input.setText(f"{self.sync_offset_seconds:.3f}")
        self._set_status(
            f"Shift set to {self.shift_value} frame(s) ({self.sync_offset_seconds:+.3f}s)."
        )
        self._update_time_series_cursor()
        self._update_frame_label()
        self._save_session_state()

    def _apply_sync_offset(self) -> None:
        offset_text = self.sync_offset_input.text().strip()
        if not offset_text:
            self.sync_offset_seconds = 0.0
            self.shift_value = 0
            self.shift_input.setText("")
            self._set_status("Sync offset cleared (0s).")
            self._update_time_series_cursor()
            self._update_frame_label()
            return

        try:
            self.sync_offset_seconds = float(offset_text)
        except ValueError:
            self._set_status("Invalid sync offset. Please enter a number.")
            return

        self.shift_value = int(round(self.sync_offset_seconds * self.TIME_BASE_FPS))
        self.shift_input.setText(str(self.shift_value))
        self._set_status(
            f"Sync offset set to {self.sync_offset_seconds:+.3f} second(s) "
            f"({self.shift_value:+d} frame(s))."
        )
        self._update_time_series_cursor()
        self._update_frame_label()
        self._save_session_state()

    def _search_frame(self) -> None:
        if not self.video_handler.capture:
            self._set_status("Load a video before searching for frames.")
            return

        frame_text = self.frame_input.text().strip()
        if not frame_text:
            self._set_status("Enter a frame number to search.")
            return

        try:
            requested_frame = int(frame_text)
        except ValueError:
            self._set_status("Invalid frame number. Please enter an integer.")
            return

        self._goto_frame(requested_frame)

    def _search_time(self) -> None:
        if not self.video_handler.capture:
            self._set_status("Load a video before searching for frames.")
            return

        time_text = self.time_input.text().strip()
        if not time_text:
            self._set_status("Enter a time value in seconds to search.")
            return

        try:
            seconds = float(time_text)
        except ValueError:
            self._set_status("Invalid time value. Please enter a number of seconds.")
            return

        requested_frame = seconds_to_frame_index(seconds, self.TIME_BASE_FPS)
        self._goto_frame(requested_frame)

    def _step_frames(self, step: int) -> None:
        if not self.video_handler.capture:
            self._set_status("Load a video to navigate frames.")
            return

        target_frame = self.current_frame_index + step
        self._goto_frame(target_frame)

    def _trigger_left_step(self) -> None:
        self._step_frames(-self.SINGLE_STEP)

    def _trigger_right_step(self) -> None:
        self._step_frames(self.SINGLE_STEP)

    def _trigger_left_large_step(self) -> None:
        self._step_frames(-self.LARGE_STEP)

    def _trigger_right_large_step(self) -> None:
        self._step_frames(self.LARGE_STEP)

    def _trigger_left_jump(self) -> None:
        self._step_frames(-self.left_jump_size)

    def _trigger_right_jump(self) -> None:
        self._step_frames(self.right_jump_size)

    def _update_left_jump_size(self, value: int) -> None:
        self.left_jump_size = max(1, value)

    def _update_right_jump_size(self, value: int) -> None:
        self.right_jump_size = max(1, value)

    def _goto_frame(self, frame_index: int, show_status: bool = True) -> None:
        if self.video_handler.frame_count <= 0:
            self._set_status("No frames available in this video.")
            return

        clamped_index = self.video_handler.clamp_index(frame_index)
        if clamped_index != frame_index and show_status:
            boundary = "first" if clamped_index == 0 else "last"
            self._set_status(f"Reached {boundary} frame. Showing frame {clamped_index}.")

        frame = self.video_handler.read_frame(clamped_index)
        if frame is None:
            self._set_status("Failed to read the requested frame.")
            return

        self.current_frame_index = clamped_index
        self._display_frame(frame)
        self._update_frame_label()
        self._update_time_series_cursor()
        if show_status:
            self._set_status(
                f"Displaying frame {clamped_index} of "
                f"{self.video_handler.frame_count - 1} (0-based)."
            )
        expected_fif, expected_csv = self.time_series_viewer.expected_paths()
        loaded_fif, loaded_csv = self.time_series_viewer.last_loaded_paths()
        self._update_summary(
            self.video_handler.video_path, expected_fif, expected_csv, loaded_fif, loaded_csv
        )

    # Display helpers
    def _display_frame(self, frame) -> None:
        target_size = self._target_display_size()
        pixmap = frame_to_pixmap(frame, target_size)
        if pixmap is None:
            self.frame_label.setText("Unable to display frame.")
            return
        self.last_frame = frame
        self.frame_label.setPixmap(pixmap)
        self.frame_label.resize(pixmap.size())

    def _update_frame_label(self) -> None:
        if self.video_handler.frame_count:
            info = f"{self.current_frame_index} / {self.video_handler.frame_count - 1} (0-based)"
            seconds_info = self._seconds_info_text()
            shift_info = f"Shift: {self.shift_value:+d} frame(s)"
            sync_info = f"Sync offset: {self.sync_offset_seconds:+.3f}s"
            total_frames = max(1, self.video_handler.frame_count - 1)
            percent = (self.current_frame_index / total_frames) * 100
            percent_info = f"[ {percent:.2f}% ]"
            self.frame_info_label.setText(
                f"Frame {info} {percent_info}\n{seconds_info}\n{shift_info} | {sync_info}"
            )
            self.current_frame_label.setText(
                "Current frame: "
                f"{info} {percent_info} | {seconds_info} | {shift_info} | {sync_info}"
            )
        else:
            self.frame_info_label.setText("Frame: -")
            self.current_frame_label.setText("Current frame: -")

    def _seconds_info_text(self) -> str:
        fps = self.video_handler.fps
        if fps <= 0 or self.video_handler.frame_count <= 0:
            return "Seconds: unavailable"

        current_seconds = self.current_frame_index / fps
        total_seconds = self.video_handler.frame_count / fps
        return f"Seconds: {current_seconds:.2f} / {total_seconds:.2f}"

    def _current_time_seconds(self) -> float:
        fps = self.video_handler.fps
        if fps <= 0:
            return 0.0
        return self.current_frame_index / fps

    def _synced_time_seconds(self) -> float:
        return self._current_time_seconds() + self.sync_offset_seconds

    def _update_time_series_cursor(self) -> None:
        self.time_series_viewer.update_cursor_time(self._synced_time_seconds())

    def _jump_to_annotation_time(self, annotation_time: float) -> None:
        if not self.video_handler.capture:
            self._set_status("Load a video to sync frame navigation.")
            return

        target_seconds = max(0.0, annotation_time - self.sync_offset_seconds)
        target_frame = seconds_to_frame_index(target_seconds, self.video_handler.fps)
        self._goto_frame(target_frame, show_status=False)
        self.time_series_viewer.update_cursor_time(annotation_time)

    def _update_navigation_state(self, enabled: bool) -> None:
        for button in [
            self.left_button,
            self.right_button,
            self.left_jump_button,
            self.right_jump_button,
            self.search_button,
            self.time_search_button,
            self.csv_picker_button,
        ]:
            button.setEnabled(enabled)
        self.remark_input.setEnabled(enabled)
        self.remark_save_button.setEnabled(enabled)
        self.remark_history_label.setEnabled(enabled)
        self.remark_history_list.setEnabled(enabled)

    def _on_side_tab_changed(self, index: int) -> None:
        show_frame = index == 0
        self.frame_panel.setVisible(show_frame)
        self.time_series_group.setVisible(show_frame)

    def _adjust_zoom(self, delta: float, anchor: Optional[QPoint] = None) -> None:
        self._set_zoom(self.zoom_factor + delta, anchor)

    def _reset_zoom(self) -> None:
        self._set_zoom(1.0)

    def _set_zoom(self, zoom: float, anchor: Optional[QPoint] = None) -> None:
        pixmap = self.frame_label.pixmap()
        h_bar = self.frame_scroll.horizontalScrollBar()
        v_bar = self.frame_scroll.verticalScrollBar()

        if pixmap:
            current_width = max(1, pixmap.width())
            current_height = max(1, pixmap.height())
            if anchor is not None:
                anchor_x = anchor.x()
                anchor_y = anchor.y()
            else:
                anchor_x = 0
                anchor_y = 0
            center_x_ratio = (h_bar.value() + anchor_x) / current_width
            center_y_ratio = (v_bar.value() + anchor_y) / current_height
            center_x_ratio = max(0.0, min(1.0, center_x_ratio))
            center_y_ratio = max(0.0, min(1.0, center_y_ratio))
        else:
            anchor_x = 0
            anchor_y = 0
            center_x_ratio = 0.0
            center_y_ratio = 0.0

        clamped_zoom = max(self.MIN_ZOOM, min(zoom, self.MAX_ZOOM))
        self.zoom_factor = clamped_zoom
        self.zoom_label.setText(self._zoom_label_text())
        self._refresh_displayed_frame()

        pixmap = self.frame_label.pixmap()
        if not pixmap:
            return

        new_width = max(1, pixmap.width())
        new_height = max(1, pixmap.height())

        target_x = (center_x_ratio * new_width) - anchor_x
        target_y = (center_y_ratio * new_height) - anchor_y

        h_bar.setValue(int(max(0, min(target_x, h_bar.maximum()))))
        v_bar.setValue(int(max(0, min(target_y, v_bar.maximum()))))

    def _zoom_label_text(self) -> str:
        percent = int(self.zoom_factor * 100)
        return f"Zoom: {percent}%"

    def _refresh_displayed_frame(self) -> None:
        if self.last_frame is not None:
            self._display_frame(self.last_frame)

    def _target_display_size(self) -> QSize:
        viewport_size = self.frame_scroll.viewport().size()
        width = max(1, int(viewport_size.width() * self.zoom_factor))
        height = max(1, int(viewport_size.height() * self.zoom_factor))
        return QSize(width, height)

    def _set_status(self, message: str) -> None:
        self.status_bar.showMessage(message)

    def _update_summary(
        self,
        video_path: Optional[Path],
        expected_fif: Optional[Path],
        expected_csv: Optional[Path],
        loaded_fif: Optional[Path],
        loaded_csv: Optional[Path],
    ) -> None:
        self.summary_video_label.setText(
            self._format_summary_line("Video", video_path, loaded=video_path is not None)
        )
        self.summary_fif_label.setText(
            self._format_summary_line("FIF", expected_fif, loaded_fif is not None)
        )
        self.summary_csv_label.setText(
            self._format_summary_line("CSV", expected_csv, loaded_csv is not None)
        )
        self._update_summary_table()

    def _format_summary_line(self, label: str, path: Optional[Path], loaded: bool) -> str:
        if path is None:
            status = "not selected"
            return f"{label}: ({status})"

        exists = "found" if path.exists() else "missing"
        state = "loaded" if loaded else "expected"
        return f"{label}: {path} [{state}, {exists}]"

    def _refresh_dataset_summary(self) -> None:
        self.summary_subject_counts = {}
        if not self.video_paths:
            self.summary_overall_label.setText("Dataset summary: (no videos found)")
            self._update_summary_table()
            return

        for video_path in self.video_paths:
            subject = extract_subject_label(video_path) or "Unknown"
            subject_counts = self.summary_subject_counts.setdefault(
                subject,
                {
                    "Pending": 0,
                    "daphne_check": 0,
                    "Ongoing": 0,
                    "Complete": 0,
                    "complete_eeg": 0,
                    "Issue": 0,
                    "Missing CSV": 0,
                    "Missing FIF": 0,
                },
            )
            status = self._status_for_video(video_path)
            subject_counts[status] += 1
            subject_counts["Missing CSV"] += int(self._missing_csv(video_path))
            subject_counts["Missing FIF"] += int(self._missing_fif(video_path))

        totals = {
            "Pending": 0,
            "daphne_check": 0,
            "Ongoing": 0,
            "Complete": 0,
            "complete_eeg": 0,
            "Issue": 0,
            "Missing CSV": 0,
            "Missing FIF": 0,
        }
        for counts in self.summary_subject_counts.values():
            for key in totals:
                totals[key] += counts.get(key, 0)

        total_videos = sum(totals.values())
        self.summary_overall_label.setText(
            "Dataset summary: "
            f"{total_videos} video(s) | "
            f"Pending: {totals['Pending']}, "
            f"daphne_check: {totals['daphne_check']}, "
            f"Ongoing: {totals['Ongoing']}, "
            f"Complete: {totals['Complete']}, "
            f"complete_eeg: {totals['complete_eeg']}, "
            f"Issue: {totals['Issue']} | "
            f"Missing CSV: {totals['Missing CSV']}, "
            f"Missing FIF: {totals['Missing FIF']}"
        )
        self._update_summary_table()

    def _status_for_video(self, video_path: Path) -> str:
        try:
            csv_path = derive_annotation_path(
                video_path,
                processed_root=self.time_series_viewer.time_series_root,
                csv_root=self.time_series_viewer.annotation_root,
            )
        except ValueError:
            return "Pending"

        session_path = csv_path.parent / SESSION_FILENAME
        if not session_path.exists():
            return "Pending"

        try:
            with session_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return "Pending"

        status = data.get("status", "Pending")
        if status not in {"Pending", "daphne_check", "Ongoing", "Complete", "complete_eeg", "Issue"}:
            return "Pending"
        return status

    def _update_summary_table(self) -> None:
        subjects = sorted(
            self.summary_subject_counts.keys(),
            key=lambda item: (item == "Unknown", subject_label_sort_key(item)),
        )
        self.summary_table.setRowCount(len(subjects))

        for row, subject in enumerate(subjects):
            counts = self.summary_subject_counts.get(subject, {})
            values = [
                subject,
                str(counts.get("Pending", 0)),
                str(counts.get("daphne_check", 0)),
                str(counts.get("Ongoing", 0)),
                str(counts.get("Complete", 0)),
                str(counts.get("complete_eeg", 0)),
                str(counts.get("Issue", 0)),
                str(counts.get("Missing CSV", 0)),
                str(counts.get("Missing FIF", 0)),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.summary_table.setItem(row, col, item)

    def _missing_csv(self, video_path: Path) -> bool:
        try:
            csv_path = derive_annotation_path(
                video_path,
                processed_root=self.time_series_viewer.time_series_root,
                csv_root=self.time_series_viewer.annotation_root,
            )
        except ValueError:
            return True
        return not csv_path.exists()

    def _missing_fif(self, video_path: Path) -> bool:
        try:
            ts_path = derive_time_series_path(
                video_path,
                processed_root=self.time_series_viewer.time_series_root,
            )
        except ValueError:
            return True
        return not ts_path.exists()

    def _setup_shortcuts(self) -> None:
        left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        left_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        left_shortcut.activated.connect(self._handle_left_shortcut)

        right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        right_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        right_shortcut.activated.connect(self._handle_right_shortcut)

        left_step_shortcut = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Left), self)
        left_step_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        left_step_shortcut.activated.connect(self._trigger_left_step)

        right_step_shortcut = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Right), self)
        right_step_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        right_step_shortcut.activated.connect(self._trigger_right_step)

        left_large_step_shortcut = QShortcut(
            QKeySequence(Qt.CTRL | Qt.SHIFT | Qt.Key_Left), self
        )
        left_large_step_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        left_large_step_shortcut.activated.connect(self._trigger_left_large_step)

        right_large_step_shortcut = QShortcut(
            QKeySequence(Qt.CTRL | Qt.SHIFT | Qt.Key_Right), self
        )
        right_large_step_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        right_large_step_shortcut.activated.connect(self._trigger_right_large_step)

        next_annotation_shortcut = QShortcut(QKeySequence(Qt.Key_BracketRight), self)
        next_annotation_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        next_annotation_shortcut.activated.connect(self._handle_next_annotation_shortcut)

        previous_annotation_shortcut = QShortcut(QKeySequence(Qt.Key_BracketLeft), self)
        previous_annotation_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        previous_annotation_shortcut.activated.connect(self._handle_previous_annotation_shortcut)

        next_annotation_letter = QShortcut(QKeySequence(Qt.Key_N), self)
        next_annotation_letter.setContext(Qt.WidgetWithChildrenShortcut)
        next_annotation_letter.activated.connect(self._handle_next_annotation_shortcut)

        next_annotation_min_shortcut = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_N), self)
        next_annotation_min_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        next_annotation_min_shortcut.activated.connect(
            self._handle_next_annotation_min_shortcut
        )

        previous_annotation_letter = QShortcut(QKeySequence(Qt.Key_B), self)
        previous_annotation_letter.setContext(Qt.WidgetWithChildrenShortcut)
        previous_annotation_letter.activated.connect(self._handle_previous_annotation_shortcut)

        save_annotations_shortcut = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_S), self)
        save_annotations_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        save_annotations_shortcut.activated.connect(self._handle_save_annotations_shortcut)

        auto_repair_annotation_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        auto_repair_annotation_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        auto_repair_annotation_shortcut.activated.connect(
            self._handle_auto_repair_annotation_shortcut
        )

        auto_repair_eog_shortcut = QShortcut(QKeySequence(Qt.Key_R), self)
        auto_repair_eog_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        auto_repair_eog_shortcut.activated.connect(
            self._handle_auto_repair_eog_shortcut
        )

        auto_repair_ear_shortcut = QShortcut(QKeySequence(Qt.Key_E), self)
        auto_repair_ear_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        auto_repair_ear_shortcut.activated.connect(
            self._handle_auto_repair_ear_shortcut
        )

        delete_annotation_shortcut = QShortcut(QKeySequence(Qt.Key_D), self)
        delete_annotation_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        delete_annotation_shortcut.activated.connect(self._handle_delete_annotation_shortcut)

        play_shortcut = QShortcut(QKeySequence(Qt.Key_P), self)
        play_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        play_shortcut.activated.connect(self._start_annotation_play)

        forward_play_shortcut = QShortcut(QKeySequence(Qt.Key_F), self)
        forward_play_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        forward_play_shortcut.activated.connect(self._toggle_forward_play)

        stop_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        stop_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        stop_shortcut.activated.connect(self._stop_all_play)

        undo_shortcut = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Z), self)
        undo_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        undo_shortcut.activated.connect(self._handle_undo_shortcut)

        self.left_shortcut = left_shortcut
        self.right_shortcut = right_shortcut
        self.left_step_shortcut = left_step_shortcut
        self.right_step_shortcut = right_step_shortcut
        self.left_large_step_shortcut = left_large_step_shortcut
        self.right_large_step_shortcut = right_large_step_shortcut
        self.next_annotation_shortcut = next_annotation_shortcut
        self.previous_annotation_shortcut = previous_annotation_shortcut
        self.next_annotation_letter = next_annotation_letter
        self.next_annotation_min_shortcut = next_annotation_min_shortcut
        self.previous_annotation_letter = previous_annotation_letter
        self.save_annotations_shortcut = save_annotations_shortcut
        self.auto_repair_annotation_shortcut = auto_repair_annotation_shortcut
        self.auto_repair_eog_shortcut = auto_repair_eog_shortcut
        self.auto_repair_ear_shortcut = auto_repair_ear_shortcut
        self.delete_annotation_shortcut = delete_annotation_shortcut
        self.forward_play_shortcut = forward_play_shortcut
        self.undo_shortcut = undo_shortcut

    def _handle_left_shortcut(self) -> None:
        if self._shortcut_allowed():
            self._trigger_left_jump()

    def _handle_right_shortcut(self) -> None:
        if self._shortcut_allowed():
            self._trigger_right_jump()

    def _handle_next_annotation_shortcut(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.jump_to_next_annotation()

    def _handle_next_annotation_min_shortcut(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.jump_to_next_annotation_minimum()

    def _handle_previous_annotation_shortcut(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.jump_to_previous_annotation()

    def _handle_save_annotations_shortcut(self) -> None:
        self.time_series_viewer.save_annotations()

    def _handle_auto_repair_annotation_shortcut(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.auto_repair_selected_annotation()

    def _handle_auto_repair_eog_shortcut(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.auto_repair_selected_annotation_eog()

    def _handle_auto_repair_ear_shortcut(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.auto_repair_selected_annotation_ear()

    def _handle_delete_annotation_shortcut(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.delete_selected_annotation()

    def _handle_undo_shortcut(self) -> None:
        if self._shortcut_allowed():
            self.time_series_viewer.undo_last_annotation()

    def _shortcut_allowed(self) -> bool:
        focus_widget = QApplication.focusWidget()
        return not isinstance(focus_widget, (QLineEdit, QSpinBox, QTextBrowser, QTextEdit))

    def _toggle_annotation_play(self, checked: bool) -> None:
        if checked:
            self._start_annotation_play()
        else:
            self._stop_annotation_play()

    def _start_annotation_play(self) -> None:
        if not self._shortcut_allowed():
            return
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
        if not self._shortcut_allowed():
            return
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
        interval_ms = int(self.forward_play_speed_spinbox.value() * 1000)
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
        interval_ms = int(self.forward_play_speed_spinbox.value() * 1000)
        if self._forward_play_timer.interval() != interval_ms:
            self._forward_play_timer.setInterval(interval_ms)
        prev_frame = self.current_frame_index
        self._trigger_right_jump()
        if self.current_frame_index == prev_frame:
            self._stop_forward_play()
            self._set_status("Auto-forward reached end of video.")

    def _stop_all_play(self) -> None:
        self._stop_annotation_play()
        self._stop_forward_play()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_session_state()
        self.video_handler.release()
        super().closeEvent(event)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if (
            obj is self.frame_scroll.viewport()
            and event.type() == QEvent.Wheel
            and event.modifiers() & Qt.ControlModifier
        ):
            delta = event.angleDelta().y()
            step = 0.25 if delta > 0 else -0.25
            self._adjust_zoom(step, anchor=event.pos())
            return True

        return super().eventFilter(obj, event)


def run_app() -> None:
    app = QApplication([])
    window = VideoFrameViewer()
    window.show()
    app.exec_()
