import sys
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from gui import TrayApplication, resource_path
from logging_setup import configure_logging

def main():
    configure_logging()
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("app_icon.ico")))
    app.setQuitOnLastWindowClosed(False)
    
    tray = TrayApplication(app)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
