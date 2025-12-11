"""PyQt5 GUI for the video frame viewer application."""
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QApplication,
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
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from viewer.utils import (
    frame_to_pixmap,
    find_md_mff_videos,
    placeholder_pixmap,
    seconds_to_frame_index,
)
from viewer.video_handler import VideoHandler


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


class PreviewWidget(QWidget):
    """Container for a single preview frame thumbnail and its caption."""

    def __init__(self, preview_size: QSize, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.preview_size = preview_size

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.image_label = QLabel()
        self.image_label.setFixedSize(self.preview_size)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.caption_label = QLabel("-")
        self.caption_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.image_label)
        layout.addWidget(self.caption_label)
        self.setLayout(layout)

    def set_pixmap(self, pixmap: Optional):
        """Set the preview image pixmap, defaulting to a placeholder."""
        if pixmap is None:
            pixmap = placeholder_pixmap(self.preview_size)
        self.image_label.setPixmap(pixmap)

    def set_caption(self, caption: str) -> None:
        self.caption_label.setText(caption)


class VideoFrameViewer(QMainWindow):
    """Main window for browsing and navigating video frames."""

    DATASET_ROOT = Path(r"D:\dataset\drowsy_driving_raja")
    SINGLE_STEP = 1
    JUMP_STEP = 10
    TIME_BASE_FPS = 30
    PREVIEW_RANGE = 5
    PREVIEW_SIZE = QSize(120, 68)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Frame Viewer")

        self.video_handler = VideoHandler()
        self.video_paths: List[Path] = []
        self.current_frame_index: int = 0
        self.shift_value: int = 0
        self.zoom_factor: float = 1.0
        self.last_frame = None

        self._setup_ui()
        self._update_navigation_state(False)
        self._initialize_dataset_root()

    # UI setup
    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self._build_directory_controls(main_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        video_list_panel = self._build_video_list_panel()
        frame_panel = self._build_frame_panel()

        splitter.addWidget(video_list_panel)
        splitter.addWidget(frame_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

        control_group = self._build_control_panel()
        main_layout.addWidget(control_group)

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

        browse_button.clicked.connect(self._browse_directory)
        scan_button.clicked.connect(self._scan_directory)

        directory_layout.addWidget(QLabel("Root:"))
        directory_layout.addWidget(self.directory_input)
        directory_layout.addWidget(browse_button)
        directory_layout.addWidget(scan_button)

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

        self.preview_group = QGroupBox("Surrounding Frames")
        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.setSpacing(4)

        preview_strip = QWidget()
        preview_strip_layout = QHBoxLayout()
        preview_strip_layout.setContentsMargins(0, 0, 0, 0)
        preview_strip_layout.setSpacing(6)
        preview_strip.setLayout(preview_strip_layout)

        self.preview_widgets: List[PreviewWidget] = []
        for _ in range(self._preview_count):
            widget = PreviewWidget(self.PREVIEW_SIZE)
            widget.set_pixmap(None)
            self.preview_widgets.append(widget)
            preview_strip_layout.addWidget(widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(preview_strip)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        preview_layout.addWidget(scroll_area)
        self.preview_group.setLayout(preview_layout)
        layout.addWidget(self.preview_group)

        self.frame_label = PannableLabel("Scan and select a video to view frames.")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.frame_label.setMinimumSize(640, 360)

        self.frame_scroll = QScrollArea()
        self.frame_scroll.setWidgetResizable(False)
        self.frame_scroll.setWidget(self.frame_label)
        self.frame_scroll.setAlignment(Qt.AlignCenter)
        self.frame_label.set_scroll_area(self.frame_scroll)

        self.frame_info_label = QLabel("Frame: -")
        self.frame_info_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.frame_scroll)
        layout.addWidget(self.frame_info_label)

        return container

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
        apply_shift_button = QPushButton("Apply Shift")
        apply_shift_button.clicked.connect(self._apply_shift)

        control_layout.addWidget(QLabel("Frame Number:"), 0, 0)
        control_layout.addWidget(self.frame_input, 0, 1)
        control_layout.addWidget(self.search_button, 0, 2)

        control_layout.addWidget(QLabel("Time (seconds):"), 1, 0)
        control_layout.addWidget(self.time_input, 1, 1)
        control_layout.addWidget(self.time_search_button, 1, 2)

        control_layout.addWidget(QLabel("Shift Frame:"), 2, 0)
        control_layout.addWidget(self.shift_input, 2, 1)
        control_layout.addWidget(apply_shift_button, 2, 2)

        navigation_layout = QHBoxLayout()
        self.left_button = QPushButton("Left")
        self.right_button = QPushButton("Right")
        self.left_jump_button = QPushButton("Left_Jump")
        self.right_jump_button = QPushButton("Right_Jump")

        self.left_button.clicked.connect(lambda: self._step_frames(-self.SINGLE_STEP))
        self.right_button.clicked.connect(lambda: self._step_frames(self.SINGLE_STEP))
        self.left_jump_button.clicked.connect(lambda: self._step_frames(-self.JUMP_STEP))
        self.right_jump_button.clicked.connect(lambda: self._step_frames(self.JUMP_STEP))

        navigation_layout.addWidget(self.left_button)
        navigation_layout.addWidget(self.right_button)
        navigation_layout.addWidget(self.left_jump_button)
        navigation_layout.addWidget(self.right_jump_button)

        control_layout.addLayout(navigation_layout, 3, 0, 1, 3)

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

        control_layout.addLayout(zoom_layout, 4, 0, 1, 3)

        self.current_frame_label = QLabel("Current frame: -")
        control_layout.addWidget(self.current_frame_label, 5, 0, 1, 3)

        return control_group

    @property
    def _preview_count(self) -> int:
        return (self.PREVIEW_RANGE * 2) + 1

    # Directory logic
    def _browse_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select dataset root", str(self.DATASET_ROOT)
        )
        if selected:
            self.directory_input.setText(selected)
            self._scan_directory()

    def _initialize_dataset_root(self) -> None:
        self.directory_input.setText(str(self.DATASET_ROOT))
        self._scan_directory()

    def _scan_directory(self) -> None:
        root_text = self.directory_input.text().strip()
        root_path = Path(root_text).expanduser() if root_text else self.DATASET_ROOT
        self.directory_input.setText(str(root_path))

        if not root_path.exists():
            self.video_list.clear()
            self._set_status(
                f"Dataset root not found at {root_path}. Please choose another folder."
            )
            return

        self.video_paths = find_md_mff_videos(root_path)

        self.video_list.clear()
        for video_path in sorted(self.video_paths):
            item = QListWidgetItem(str(video_path))
            self.video_list.addItem(item)

        if self.video_paths:
            self._set_status(
                f"Found {len(self.video_paths)} MD.mff .mov file(s) in the dataset root."
            )
        else:
            self._set_status("No MD.mff .mov files found in the dataset root.")

    def _load_selected_video(self) -> None:
        selected_items = self.video_list.selectedItems()
        if not selected_items:
            return

        video_path = Path(selected_items[0].text())
        if not self.video_handler.load(video_path):
            self._set_status("Unable to open the selected video.")
            self._update_navigation_state(False)
            return

        has_frames = self.video_handler.frame_count > 0
        self.current_frame_index = 0
        self._update_navigation_state(has_frames)

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
            self._set_status("Shift cleared (0).")
            return

        try:
            self.shift_value = int(shift_text)
            self._set_status(f"Shift set to {self.shift_value}.")
        except ValueError:
            self._set_status("Invalid shift value. Please enter an integer.")

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

        effective_frame = requested_frame + self.shift_value
        self._goto_frame(effective_frame)

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
        effective_frame = requested_frame + self.shift_value
        self._goto_frame(effective_frame)

    def _step_frames(self, step: int) -> None:
        if not self.video_handler.capture:
            self._set_status("Load a video to navigate frames.")
            return

        target_frame = self.current_frame_index + step
        self._goto_frame(target_frame)

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
        self._update_previews()
        if show_status:
            self._set_status(
                f"Displaying frame {clamped_index} of {self.video_handler.frame_count - 1} (0-based)."
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
            self.frame_info_label.setText(f"Frame {info}")
            self.current_frame_label.setText(f"Current frame: {info}")
        else:
            self.frame_info_label.setText("Frame: -")
            self.current_frame_label.setText("Current frame: -")

    def _update_previews(self) -> None:
        if self.video_handler.frame_count <= 0:
            for widget in self.preview_widgets:
                widget.set_pixmap(None)
                widget.set_caption("-")
            return

        start_index = self.current_frame_index - self.PREVIEW_RANGE
        indices = [start_index + offset for offset in range(self._preview_count)]

        for widget, index in zip(self.preview_widgets, indices):
            if index < 0 or index >= self.video_handler.frame_count:
                widget.set_pixmap(None)
                widget.set_caption(f"Frame {index} (out)")
                continue

            frame = self.video_handler.read_frame(index)
            pixmap = frame_to_pixmap(frame, self.PREVIEW_SIZE)
            widget.set_pixmap(pixmap)
            widget.set_caption(f"Frame {index}")

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

    def _adjust_zoom(self, delta: float) -> None:
        self._set_zoom(self.zoom_factor + delta)

    def _reset_zoom(self) -> None:
        self._set_zoom(1.0)

    def _set_zoom(self, zoom: float) -> None:
        clamped_zoom = max(0.25, min(zoom, 3.0))
        self.zoom_factor = clamped_zoom
        self.zoom_label.setText(self._zoom_label_text())
        self._refresh_displayed_frame()

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

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.video_handler.release()
        super().closeEvent(event)


def run_app() -> None:
    app = QApplication([])
    window = VideoFrameViewer()
    window.show()
    app.exec_()
