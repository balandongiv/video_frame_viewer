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

from viewer.utils import frame_to_pixmap, placeholder_pixmap
from viewer.video_handler import VideoHandler


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

    SINGLE_STEP = 1
    JUMP_STEP = 10
    PREVIEW_RANGE = 5
    PREVIEW_SIZE = QSize(120, 68)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Frame Viewer")

        self.video_handler = VideoHandler()
        self.video_paths: List[Path] = []
        self.current_frame_index: int = 0
        self.shift_value: int = 0

        self._setup_ui()
        self._update_navigation_state(False)

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
        browse_button = QPushButton("Browse")
        scan_button = QPushButton("Scan for .mov files")

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

        self.frame_label = QLabel("Scan and select a video to view frames.")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_label.setMinimumSize(640, 360)

        self.frame_info_label = QLabel("Frame: -")
        self.frame_info_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.frame_label)
        layout.addWidget(self.frame_info_label)

        return container

    def _build_control_panel(self) -> QGroupBox:
        control_group = QGroupBox("Frame Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("Enter frame number (0-based)")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._search_frame)

        self.shift_input = QLineEdit()
        self.shift_input.setPlaceholderText("Shift frame (can be negative)")
        apply_shift_button = QPushButton("Apply Shift")
        apply_shift_button.clicked.connect(self._apply_shift)

        control_layout.addWidget(QLabel("Frame Number:"), 0, 0)
        control_layout.addWidget(self.frame_input, 0, 1)
        control_layout.addWidget(self.search_button, 0, 2)

        control_layout.addWidget(QLabel("Shift Frame:"), 1, 0)
        control_layout.addWidget(self.shift_input, 1, 1)
        control_layout.addWidget(apply_shift_button, 1, 2)

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

        control_layout.addLayout(navigation_layout, 2, 0, 1, 3)

        self.current_frame_label = QLabel("Current frame: -")
        control_layout.addWidget(self.current_frame_label, 3, 0, 1, 3)

        return control_group

    @property
    def _preview_count(self) -> int:
        return (self.PREVIEW_RANGE * 2) + 1

    # Directory logic
    def _browse_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            self.directory_input.setText(directory)

    def _scan_directory(self) -> None:
        root_text = self.directory_input.text().strip()
        if not root_text:
            self._set_status("Please enter or choose a root directory.")
            return

        root_path = Path(root_text)
        if not root_path.exists():
            self._set_status("The specified directory does not exist.")
            return

        self.video_paths = [
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mov"
        ]

        self.video_list.clear()
        for video_path in sorted(self.video_paths):
            item = QListWidgetItem(str(video_path))
            self.video_list.addItem(item)

        if self.video_paths:
            self._set_status(f"Found {len(self.video_paths)} .mov file(s). Select one to load.")
        else:
            self._set_status("No .mov files found in the selected directory.")

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
        pixmap = frame_to_pixmap(frame, self.frame_label.size())
        if pixmap is None:
            self.frame_label.setText("Unable to display frame.")
            return
        self.frame_label.setPixmap(pixmap)

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
        ]:
            button.setEnabled(enabled)

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
