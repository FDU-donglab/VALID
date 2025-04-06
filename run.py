import sys
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QDesktopWidget, QMessageBox)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from test_pipeline import TestingWindow
from train_pipeline import TrainingWindow

class MainWindow(QMainWindow):
    """Main window for selecting between training and testing."""

    def __init__(self):
        """
        Initializes the main window.

        Input:
        - None

        Output:
        - None
        """
        super().__init__()

        self.setWindowTitle("Training or Testing")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        # Add logo
        self.logo = QLabel(self)
        pixmap = QPixmap("./resource/logo_lr.png")
        self.logo.setPixmap(pixmap)
        self.logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo)

        # Add buttons for training and testing
        self.train_button = QPushButton("Training")
        self.train_button.clicked.connect(self.show_training_page)
        layout.addWidget(self.train_button)

        self.test_button = QPushButton("Testing")
        self.test_button.clicked.connect(self.show_testing_page)
        layout.addWidget(self.test_button)

        self.train_button.setEnabled(True)
        self.test_button.setEnabled(True)

        # Add bottom text and link
        bottom_layout = QHBoxLayout()
        self.bottom_text = QLabel("For more information, visit ")
        self.bottom_text.setStyleSheet("color: white;")
        self.link = QLabel('<a href="https://guyuanjie.com">this paper</a>')
        self.link.setOpenExternalLinks(True)
        self.link.setStyleSheet("color: #1E90FF;")
        bottom_layout.addWidget(self.bottom_text)
        bottom_layout.addWidget(self.link)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.apply_styles()
        self.setup_exception_handling()

        self.training_window = None
        self.testing_window = None

    def setup_exception_handling(self):
        """
        Sets up exception handling for the application.

        Input:
        - None

        Output:
        - None
        """
        def excepthook(exc_type, exc_value, exc_tb):
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.show_critical_error(f"Unhandled exception:\n{tb}")
            QApplication.quit()
        sys.excepthook = excepthook

    def show_critical_error(self, message):
        """
        Displays a critical error message.

        Input:
        - message (str): The error message to display.

        Output:
        - None
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2E2E2E;
                color: white;
            }
            QPushButton {
                background-color: #4A4A4A;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 15px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
        """)
        msg_box.exec_()

    def on_destroy(self):
        """
        Closes the training and testing windows when the main window is destroyed.

        Input:
        - None

        Output:
        - None
        """
        if hasattr(self, 'training_window'):
            self.training_window.close()
            self.train_button.setEnabled(True)
        if hasattr(self, 'testing_window'):
            self.testing_window.close()
            self.test_button.setEnabled(True)

    def center(self):
        """
        Centers the main window on the screen.

        Input:
        - None

        Output:
        - None
        """
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.width()) / 2
        self.move(int(newLeft), int(newTop))

    def apply_styles(self):
        """
        Applies styles to the UI components.

        Input:
        - None

        Output:
        - None
        """
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E2E2E;
            }
            QPushButton {
                font-family: Arial;
                font-size: 14px;
                color: white;
                background-color: #4A4A4A;
                border: none;
                padding: 10px;
                border-radius: 15px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
            QPushButton:disabled {
                background-color: #3A3A3A;
                color: #7F7F7F;
            }
            QLabel {
                font-family: Arial;
                font-size: 14px;
                color: white;
            }
        """)

    def show_training_page(self):
        """
        Opens the training window.

        Input:
        - None

        Output:
        - None
        """
        try:
            if self.training_window and self.training_window.isVisible():
                self.training_window.activateWindow()
                return
            self.train_button.setEnabled(False)
            if not self.training_window or not self.training_window.isVisible():
                self.training_window = TrainingWindow()
                self.training_window.window_closed.connect(lambda: self.enable_train_button(True))
                self.training_window.show()
            else:
                self.training_window.activateWindow()
                self.test_button.setEnabled(True)
        except Exception as e:
            self.show_critical_error(f"Failed to open training window: {str(e)}")
            self.train_button.setEnabled(True)

    def show_testing_page(self):
        """
        Opens the testing window.

        Input:
        - None

        Output:
        - None
        """
        try:
            if self.testing_window and self.testing_window.isVisible():
                self.testing_window.activateWindow()
                return
            self.test_button.setEnabled(False)
            if not self.testing_window or not self.testing_window.isVisible():
                self.testing_window = TestingWindow()
                self.testing_window.window_closed.connect(lambda: self.enable_test_button(True))
                self.testing_window.show()
            else:
                self.testing_window.activateWindow()
        except Exception as e:
            self.show_critical_error(f"Failed to open testing window: {str(e)}")
            self.test_button.setEnabled(True)

    def enable_train_button(self, enable):
        """
        Enables or disables the training button.

        Input:
        - enable (bool): Whether to enable the training button.

        Output:
        - None
        """
        self.train_button.setEnabled(enable)
        if not enable:
            self.training_window = None

    def enable_test_button(self, enable):
        """
        Enables or disables the testing button.

        Input:
        - enable (bool): Whether to enable the testing button.

        Output:
        - None
        """
        self.test_button.setEnabled(enable)
        if not enable:
            self.testing_window = None

    def closeEvent(self, event):
        """
        Handles the window close event.

        Input:
        - event (QCloseEvent): The close event.

        Output:
        - None
        """
        if self.training_window:
            self.training_window.close()
        if self.testing_window:
            self.testing_window.close()
        super().closeEvent(event)

if __name__ == "__main__":
    """
    Entry point for the application.

    Input:
    - None

    Output:
    - None
    """
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setWindowTitle('VALID Denoising')
    main_window.setWindowIcon(QIcon("./resource/icon.ico"))
    main_window.center()
    main_window.show()
    sys.exit(app.exec_())