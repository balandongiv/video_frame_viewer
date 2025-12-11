"""
Video Frame Viewer Application.

Provides a PyQt5-based UI for scanning directories for .mov files, selecting a video,
searching and navigating frames with optional shift offset, and displaying the current frame.
"""
from pathlib import Path
from typing import List, Optional

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
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
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class VideoFrameViewer(QMainWindow):
    """Main window for browsing and navigating video frames."""

    SINGLE_STEP = 1
    JUMP_STEP = 10

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Frame Viewer")

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_paths: List[Path] = []
        self.frame_count: int = 0
        self.current_frame_index: int = 0
        self.shift_value: int = 0

        self._setup_ui()
        self._update_navigation_state(False)

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Directory controls
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

        main_layout.addWidget(directory_group)

        # Video list and frame display area
        content_layout = QGridLayout()

        self.video_list = QListWidget()
        self.video_list.itemSelectionChanged.connect(self._load_selected_video)

        self.frame_label = QLabel("Scan and select a video to view frames.")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_label.setMinimumSize(640, 360)

        content_layout.addWidget(QLabel("Discovered Videos"), 0, 0)
        content_layout.addWidget(QLabel("Frame Display"), 0, 1)
        content_layout.addWidget(self.video_list, 1, 0)
        content_layout.addWidget(self.frame_label, 1, 1)

        main_layout.addLayout(content_layout)

        # Frame search and shift controls
        control_group = QGroupBox("Frame Controls")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("Enter frame number (0-based)")
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._search_frame)

        self.shift_input = QLineEdit()
        self.shift_input.setPlaceholderText("Shift frame (can be negative)")
        apply_shift_button = QPushButton("Apply Shift")
        apply_shift_button.clicked.connect(self._apply_shift)

        control_layout.addWidget(QLabel("Frame Number:"), 0, 0)
        control_layout.addWidget(self.frame_input, 0, 1)
        control_layout.addWidget(search_button, 0, 2)

        control_layout.addWidget(QLabel("Shift Frame:"), 1, 0)
        control_layout.addWidget(self.shift_input, 1, 1)
        control_layout.addWidget(apply_shift_button, 1, 2)

        # Navigation buttons
        navigation_layout = QHBoxLayout()
        left_button = QPushButton("Left")
        right_button = QPushButton("Right")
        left_jump_button = QPushButton("Left_Jump")
        right_jump_button = QPushButton("Right_Jump")

        left_button.clicked.connect(lambda: self._step_frames(-self.SINGLE_STEP))
        right_button.clicked.connect(lambda: self._step_frames(self.SINGLE_STEP))
        left_jump_button.clicked.connect(lambda: self._step_frames(-self.JUMP_STEP))
        right_jump_button.clicked.connect(lambda: self._step_frames(self.JUMP_STEP))

        navigation_layout.addWidget(left_button)
        navigation_layout.addWidget(right_button)
        navigation_layout.addWidget(left_jump_button)
        navigation_layout.addWidget(right_jump_button)

        control_layout.addLayout(navigation_layout, 2, 0, 1, 3)

        self.current_frame_label = QLabel("Current frame: -")
        control_layout.addWidget(self.current_frame_label, 3, 0, 1, 3)

        main_layout.addWidget(control_group)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

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
        if self.video_capture:
            self.video_capture.release()

        self.video_capture = cv2.VideoCapture(str(video_path))
        if not self.video_capture.isOpened():
            self._set_status("Unable to open the selected video.")
            self._update_navigation_state(False)
            return

        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = max(frame_count, 0)
        self.current_frame_index = 0
        has_frames = self.frame_count > 0
        self._update_navigation_state(has_frames)

        if has_frames:
            self._goto_frame(0, show_status=False)
            self._set_status(
                f"Loaded video: {video_path.name}. Total frames: {self.frame_count}"
            )
        else:
            self._set_status("The selected video contains no frames.")

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
        if not self.video_capture:
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
        if not self.video_capture:
            self._set_status("Load a video to navigate frames.")
            return

        target_frame = self.current_frame_index + step
        self._goto_frame(target_frame)

    def _goto_frame(self, frame_index: int, show_status: bool = True) -> None:
        if self.frame_count <= 0:
            self._set_status("No frames available in this video.")
            return

        clamped_index = max(0, min(frame_index, self.frame_count - 1))
        if clamped_index != frame_index and show_status:
            boundary = "first" if clamped_index == 0 else "last"
            self._set_status(f"Reached {boundary} frame. Showing frame {clamped_index}.")

        if not self.video_capture:
            self._set_status("Video capture is not initialized.")
            return

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, clamped_index)
        success, frame = self.video_capture.read()
        if not success:
            self._set_status("Failed to read the requested frame.")
            return

        self.current_frame_index = clamped_index
        self._display_frame(frame)
        self._update_frame_label()
        if show_status:
            self._set_status(f"Displaying frame {clamped_index} of {self.frame_count - 1} (0-based).")

    def _display_frame(self, frame) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.frame_label.setPixmap(scaled_pixmap)

    def _update_navigation_state(self, enabled: bool) -> None:
        for button in self.findChildren(QPushButton):
            if button.text() in {"Left", "Right", "Left_Jump", "Right_Jump", "Search"}:
                button.setEnabled(enabled)

    def _update_frame_label(self) -> None:
        if self.frame_count:
            self.current_frame_label.setText(
                f"Current frame: {self.current_frame_index} / {self.frame_count - 1} (0-based)"
            )
        else:
            self.current_frame_label.setText("Current frame: -")

    def _set_status(self, message: str) -> None:
        self.status_bar.showMessage(message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.video_capture:
            self.video_capture.release()
        super().closeEvent(event)


def main() -> None:
    app = QApplication([])
    window = VideoFrameViewer()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
