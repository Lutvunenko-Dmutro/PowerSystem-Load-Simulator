import sys
from PySide6.QtWidgets import QApplication

import pyqtgraph as pg
try:
    pg.setConfigOption('useOpenGL', False)
    pg.setConfigOption('enableExperimental', False)
except Exception:
    pass 

from app_ui import PowerSystemDashboard

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = PowerSystemDashboard()
    dashboard.show()
    sys.exit(app.exec())