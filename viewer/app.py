"""Application entry point helpers."""
import sys

from PyQt5.QtWidgets import QApplication

from viewer.gui import VideoFrameViewer


def main() -> None:
    app = QApplication(sys.argv)
    window = VideoFrameViewer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
