import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTabWidget,
    QLabel,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QComboBox,
    QSlider,
    QGroupBox,
    QListWidget,
    QSplitter,
    QMessageBox,
    QCheckBox,
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
import requests
import json
from datetime import datetime

from ui.desktop.client import RestorAIClient
from api.config import get_settings


class WorkerThread(QThread):
    """Background worker thread for API calls"""

    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, task_type, file_path, params):
        super().__init__()
        self.task_type = task_type
        self.file_path = file_path
        self.params = params
        self.client = RestorAIClient()
        self.task_id = None

    def run(self):
        try:
            # Submit task
            self.task_id = self.client.submit_task(
                self.task_type, self.file_path, self.params
            )

            if not self.task_id:
                self.error_signal.emit("Failed to submit task")
                return

            # Poll for progress
            while True:
                status = self.client.get_task_status(self.task_id)

                if status["status"] in ["completed", "failed", "cancelled"]:
                    if status["status"] == "completed":
                        self.result_signal.emit(status["result"])
                    else:
                        self.error_signal.emit(status.get("error", "Task failed"))
                    break
                else:
                    progress = int(status.get("progress", 0) * 100)
                    self.progress_signal.emit(progress)

                self.msleep(1000)  # Poll every second

        except Exception as e:
            self.error_signal.emit(str(e))


class RestorAIMainWindow(QMainWindow):
    """Main application window for RestorAI Desktop"""

    def __init__(self):
        super().__init__()
        self.client = RestorAIClient()
        self.current_task = None
        self.output_dir = Path.home() / "RestorAI" / "Outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.init_ui()
        self.load_settings()

        # Status polling timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_system_status)
        self.status_timer.start(5000)
