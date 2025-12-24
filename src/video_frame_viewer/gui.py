"""PyQt5 GUI for the video frame viewer application."""
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QEvent, QPoint, QSize, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from video_frame_viewer.config import AppConfig
from video_frame_viewer.time_series import TimeSeriesViewer
from video_frame_viewer.utils import (
    find_md_mff_videos,
    find_mov_videos,
    frame_to_pixmap,
    seconds_to_frame_index,
)
from video_frame_viewer.video_handler import VideoHandler


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
        self.time_series_viewer = TimeSeriesViewer(
            time_series_root=self.config.time_series_root,
            annotation_root=self.config.annotation_root,
            ui_settings=self.config.ui,
        )
        self.use_test_data = False

        self._setup_ui()
        self._setup_shortcuts()
        self.frame_scroll.viewport().installEventFilter(self)
        self.time_series_viewer.annotation_jump_requested.connect(
            self._jump_to_annotation_time
        )
        self._update_navigation_state(False)
        self._initialize_dataset_root()

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

        upper_splitter = QSplitter(Qt.Horizontal)
        upper_splitter.setChildrenCollapsible(False)

        side_tabs = self._build_side_tabs()
        frame_panel = self._build_frame_panel()

        upper_splitter.addWidget(side_tabs)
        upper_splitter.addWidget(frame_panel)
        upper_splitter.setStretchFactor(0, 1)
        upper_splitter.setStretchFactor(1, 5)

        top_layout.addWidget(upper_splitter)

        time_series_group = self._build_time_series_panel()

        self._main_splitter.addWidget(top_container)
        self._main_splitter.addWidget(time_series_group)
        self._main_splitter.setStretchFactor(0, 5)
        self._main_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self._main_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

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
        self.frame_scroll.setWidgetResizable(True)
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

        tabs.addTab(videos_tab, "Videos")
        tabs.addTab(channels_tab, "Channels")
        tabs.addTab(navigation_tab, "Navigation")
        tabs.addTab(summary_tab, "Summary")

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

        jump_layout.addWidget(QLabel("Left Jump (frames):"), 0, 0)
        jump_layout.addWidget(self.left_jump_input, 0, 1)
        jump_layout.addWidget(QLabel("Right Jump (frames):"), 1, 0)
        jump_layout.addWidget(self.right_jump_input, 1, 1)

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

        layout.addWidget(self.summary_video_label)
        layout.addWidget(self.summary_fif_label)
        layout.addWidget(self.summary_csv_label)
        layout.addStretch()
        summary_tab.setLayout(layout)
        return summary_tab

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

        navigation_layout = QHBoxLayout()
        self.left_button = QPushButton("Left")
        self.right_button = QPushButton("Right")
        self.left_jump_button = QPushButton("Left_Jump")
        self.right_jump_button = QPushButton("Right_Jump")

        self.left_button.clicked.connect(self._trigger_left_step)
        self.right_button.clicked.connect(self._trigger_right_step)
        self.left_jump_button.clicked.connect(self._trigger_left_jump)
        self.right_jump_button.clicked.connect(self._trigger_right_jump)

        navigation_layout.addWidget(self.left_button)
        navigation_layout.addWidget(self.right_button)
        navigation_layout.addWidget(self.left_jump_button)
        navigation_layout.addWidget(self.right_jump_button)

        control_layout.addLayout(navigation_layout, 4, 0, 1, 3)

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

        control_layout.addLayout(zoom_layout, 5, 0, 1, 3)

        self.current_frame_label = QLabel("Current frame: -")
        control_layout.addWidget(self.current_frame_label, 6, 0, 1, 3)

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
        for video_path in sorted(self.video_paths):
            item = QListWidgetItem(str(video_path))
            self.video_list.addItem(item)

        if self.video_paths:
            descriptor = "test .mov" if self.use_test_data else "MD.mff .mov"
            self._set_status(
                f"Found {len(self.video_paths)} {descriptor} file(s) in the dataset root."
            )
        else:
            descriptor = "test .mov" if self.use_test_data else "MD.mff .mov"
            self._set_status(f"No {descriptor} files found in the dataset root.")
        self._update_summary(None, None, None, None, None)

    def _load_selected_video(self) -> None:
        selected_items = self.video_list.selectedItems()
        if not selected_items:
            return

        video_path = Path(selected_items[0].text())
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
            self._set_status(
                f"Loaded video: {video_path.name}. Total frames: {self.video_handler.frame_count}"
            )
        else:
            self._set_status("The selected video contains no frames.")

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
            self.frame_info_label.setText(
                f"Frame {info}\n{seconds_info}\n{shift_info} | {sync_info}"
            )
            self.current_frame_label.setText(
                f"Current frame: {info} | {seconds_info} | {shift_info} | {sync_info}"
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
        ]:
            button.setEnabled(enabled)

    def _adjust_zoom(self, delta: float, anchor: Optional[QPoint] = None) -> None:
        self._set_zoom(self.zoom_factor + delta, anchor)

    def _reset_zoom(self) -> None:
        self._set_zoom(1.0)

    def _set_zoom(self, zoom: float, anchor: Optional[QPoint] = None) -> None:
        pixmap = self.frame_label.pixmap()
        viewport_size = self.frame_scroll.viewport().size()
        h_bar = self.frame_scroll.horizontalScrollBar()
        v_bar = self.frame_scroll.verticalScrollBar()

        if pixmap:
            current_width = max(1, pixmap.width())
            current_height = max(1, pixmap.height())
            if anchor is not None:
                center_x_ratio = (h_bar.value() + anchor.x()) / current_width
                center_y_ratio = (v_bar.value() + anchor.y()) / current_height
            else:
                center_x_ratio = (
                    h_bar.value() + viewport_size.width() / 2
                ) / current_width
                center_y_ratio = (
                    v_bar.value() + viewport_size.height() / 2
                ) / current_height
            center_x_ratio = max(0.0, min(1.0, center_x_ratio))
            center_y_ratio = max(0.0, min(1.0, center_y_ratio))
        else:
            center_x_ratio = 0.5
            center_y_ratio = 0.5

        clamped_zoom = max(self.MIN_ZOOM, min(zoom, self.MAX_ZOOM))
        self.zoom_factor = clamped_zoom
        self.zoom_label.setText(self._zoom_label_text())
        self._refresh_displayed_frame()

        pixmap = self.frame_label.pixmap()
        if not pixmap:
            return

        new_width = max(1, pixmap.width())
        new_height = max(1, pixmap.height())

        target_x = (center_x_ratio * new_width) - (viewport_size.width() / 2)
        target_y = (center_y_ratio * new_height) - (viewport_size.height() / 2)

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

    def _format_summary_line(self, label: str, path: Optional[Path], loaded: bool) -> str:
        if path is None:
            status = "not selected"
            return f"{label}: ({status})"

        exists = "found" if path.exists() else "missing"
        state = "loaded" if loaded else "expected"
        return f"{label}: {path} [{state}, {exists}]"

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

        previous_annotation_letter = QShortcut(QKeySequence(Qt.Key_P), self)
        previous_annotation_letter.setContext(Qt.WidgetWithChildrenShortcut)
        previous_annotation_letter.activated.connect(self._handle_previous_annotation_shortcut)

        save_annotations_shortcut = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_S), self)
        save_annotations_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        save_annotations_shortcut.activated.connect(self._handle_save_annotations_shortcut)

        self.left_shortcut = left_shortcut
        self.right_shortcut = right_shortcut
        self.left_step_shortcut = left_step_shortcut
        self.right_step_shortcut = right_step_shortcut
        self.next_annotation_shortcut = next_annotation_shortcut
        self.previous_annotation_shortcut = previous_annotation_shortcut
        self.next_annotation_letter = next_annotation_letter
        self.next_annotation_min_shortcut = next_annotation_min_shortcut
        self.previous_annotation_letter = previous_annotation_letter
        self.save_annotations_shortcut = save_annotations_shortcut

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

    def _shortcut_allowed(self) -> bool:
        focus_widget = QApplication.focusWidget()
        return not isinstance(focus_widget, (QLineEdit, QSpinBox))

    def closeEvent(self, event) -> None:  # type: ignore[override]
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
