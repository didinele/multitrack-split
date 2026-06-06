import shutil
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .window import MainWindow

def launch():
    app = QApplication.instance() or QApplication(sys.argv)

    if shutil.which("ffmpeg") is None:
        QMessageBox.critical(
            None,
            "ffmpeg not found",
            "ffmpeg was not found on PATH.\n\n"
            "Please install ffmpeg and ensure it is available on your PATH, then restart the app.\n\n"
            "Windows: download from https://ffmpeg.org/download.html and add the bin/ folder to your PATH.",
        )
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
