"""Application entry point helpers."""
import sys

from PyQt5.QtWidgets import QApplication

from viewer.annotation_editor import AnnotationEditorWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = AnnotationEditorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
