import sys

from PySide6.QtWidgets import QApplication

from .window import MainWindow


def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
