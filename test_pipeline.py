# -*- coding: utf-8 -*-
import os
import sys
import json
import traceback
import datetime
import threading
import logging

from torch.utils.data import DataLoader
import numpy as np
from skimage import io
from tqdm import tqdm

from datasets.dataset_fs import ReadDatasets, custom_collate_fn
from datasets.sampling import *
from models.network import Network_CNR
from utils import *
from utils.config import json2args

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSplitter, QTextEdit, QLineEdit, QGridLayout,
                              QMessageBox,QFileDialog)
from PyQt5.QtGui import  QTextCursor,  QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

import warnings
warnings.filterwarnings("ignore")


class EmittingStream(QObject):
    """Redirects stdout and stderr to PyQt signals."""
    textWritten = pyqtSignal(str)

    def write(self, text):
        """
        Redirects text to the PyQt signal.

        Input:
        - text (str): The text to be redirected.

        Output:
        - None
        """
        self.textWritten.emit(str(text))

    def flush(self):
        """
        Flushes the stream (no-op for this implementation).

        Input:
        - None

        Output:
        - None
        """
        pass

class TestingWindow(QMainWindow):
    """Main window for the testing GUI."""
    window_closed = pyqtSignal()

    def __init__(self):
        """
        Initializes the testing window.

        Input:
        - None

        Output:
        - None
        """
        super().__init__()
        self.setWindowTitle("Testing VALID")
        self.setWindowIcon(QIcon("./resource/icon.ico"))
        self.setGeometry(100, 100, 500, 500)
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Upper part of the left side
        upper_left_widget = QWidget()
        upper_left_layout = QGridLayout()

        self.param_inputs = {}
        params = {
            "params_path": r"./jsons/params.json",
            "test_folder": r"./data/train",
        }

        row = 0

        for key, value in params.items():
            self.param_inputs[key] = QLineEdit(str(value))
            upper_left_layout.addWidget(self.param_inputs[key], row,1)
            browse_button = QPushButton(f"Load {key}")
            browse_button.clicked.connect(lambda _, k=key: self.browse_file(k))
            upper_left_layout.addWidget(browse_button, row,2)
            row += 1
        
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        upper_left_layout.addLayout(button_layout, row, 1)
        upper_left_widget.setLayout(upper_left_layout)

        # Lower part of the left side
        lower_left_widget = QWidget()
        lower_left_layout = QVBoxLayout()
        self.info_output = QTextEdit()
        self.info_output.setReadOnly(True)
        lower_left_layout.addWidget(QLabel("Logging:"))
        lower_left_layout.addWidget(self.info_output)
        lower_left_widget.setLayout(lower_left_layout)

        left_layout.addWidget(upper_left_widget)
        left_layout.addWidget(lower_left_widget)
        left_widget.setLayout(left_layout)

        main_splitter.addWidget(left_widget)

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.addWidget(main_splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        self.apply_styles()
        self.setup_exception_handling()
        self.setup_stdout_redirect()
        self.start_button.clicked.connect(self.start_testing)
        self.stop_button.clicked.connect(self.stop_testing)

        self.is_testing = False
        self.thread = None
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.thread_lock = threading.Lock()

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
            QLabel {
                font-family: Arial;
                font-size: 14px;
                color: white;
            }
            QLineEdit, QTextEdit {
                font-family: Arial;
                font-size: 14px;
                color: white;
                background-color: #4A4A4A;
                border: none;
                padding: 10px;
            }
            QPushButton:disabled {
                background-color: #3A3A3A;
                color: #7F7F7F;
            }
        """)

    def browse_file(self, key):
        """
        Opens a file or folder dialog for the given key.

        Input:
        - key (str): The key indicating which file or folder to browse.

        Output:
        - None
        """
        if key == "test_folder":
            folder = QFileDialog.getExistingDirectory(self, "Select Test Folder")
            if folder:
                self.param_inputs[key].setText(folder)
        elif key == "params_path":
            file, _ = QFileDialog.getOpenFileName(self, "Select Json File", "", "Model Files (*.json)")
            if file:
                self.param_inputs[key].setText(file)

    def validate_inputs(self):
        """
        Validates the user inputs.

        Input:
        - None

        Output:
        - errors (list): List of error messages, if any.
        """
        errors = []
        params_path = self.param_inputs["params_path"].text()
        if not params_path:
            errors.append("The path of the parameter file cannot be empty")
        elif not os.path.exists(params_path):
            errors.append("he parameter file does not exist")
        elif not params_path.endswith('.json'):
            errors.append("The parameter file must be in JSON format")
        test_folder = self.param_inputs["test_folder"].text()
        if not test_folder:
            errors.append("The test folder cannot be empty")
        elif not os.path.exists(test_folder):
            errors.append("The test folder does not exist")
        if not errors and params_path:
            try:
                with open(params_path, 'r') as f:
                    json.load(f)
            except Exception as e:
                errors.append(f"Error in the parameter file format: {str(e)}")

        return errors

    def start_testing(self):
        """
        Starts the testing process.

        Input:
        - None

        Output:
        - None
        """
        if self.is_testing:
            QMessageBox.warning(self, "Warning", "The test is currently underway！")
            return
        errors = self.validate_inputs()
        if errors:
            QMessageBox.critical(self, "Input error", "\n".join(errors))
            return
        params = {}
        for key in self.param_inputs:
            value = self.param_inputs[key].text()
            params[key] = value
        try:
            self.is_testing = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.test_json_params_path = params["params_path"]
            self.test_folder_path = params["test_folder"]
            with self.thread_lock:
                self.thread = TestingThread(params_path=self.test_json_params_path, test_folder_path=self.test_folder_path, GUI_HANDLER=self)
                self.thread.output_written.connect(self.handle_stdout)
                self.thread.error_written.connect(self.handle_stderr)
                self.thread.finished.connect(self.on_testing_finished)
                self.thread.error_occurred.connect(self.handle_thread_error)
                self.thread.start()
            self.info_output.append("--- The test has been successfully initiated ! ---")
        except Exception as e:
            self.handle_thread_error(f"The test startup failed: {str(e)}")

    def stop_testing(self):
        """
        Stops the testing process.

        Input:
        - None

        Output:
        - None
        """
        with self.thread_lock:
            if self.is_testing and self.thread:
                reply = QMessageBox.question(
                    self, 'Confirm to stop',
                    'Are you sure you want to terminate the current test?',
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.thread.stop()
                    self.info_output.append("The test is being stopped...")
                    self.thread.wait(3000)
                    self.is_testing = False
                    self.stop_button.setEnabled(False)
                    self.start_button.setEnabled(True)
                    self.thread = None  # Reset the thread

    def on_testing_finished(self):
        """
        Handles the completion of the testing process.

        Input:
        - None

        Output:
        - None
        """
        self.info_output.setTextColor(Qt.red)
        self.info_output.append("--- Test completed ! ---")
        self.info_output.setTextColor(Qt.white)
        self.is_testing = False
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        with self.thread_lock:
            self.thread = None  # Reset the thread

    def handle_thread_error(self, error_msg):
        """
        Handles errors in the testing thread.

        Input:
        - error_msg (str): The error message to display.

        Output:
        - None
        """
        self.show_critical_error(error_msg)
        self.info_output.append(f"! Error: {error_msg}")

    def handle_stdout(self, text):
        """
        Handles stdout redirection.

        Input:
        - text (str): The text to display in the logging area.

        Output:
        - None
        """
        try:
            cursor = self.info_output.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)
            self.info_output.setTextCursor(cursor)
            self.info_output.ensureCursorVisible()
        except RuntimeError:
            pass

    def handle_stderr(self, text):
        """
        Handles stderr redirection.

        Input:
        - text (str): The error text to display in the logging area.

        Output:
        - None
        """
        try:
            self.info_output.setTextColor(Qt.red)
            self.handle_stdout(text)
            self.info_output.setTextColor(Qt.white)
        except RuntimeError:
            pass

    def closeEvent(self, event):
        """
        Handles the window close event.

        Input:
        - event (QCloseEvent): The close event.

        Output:
        - None
        """
        if self.is_testing:
            self.stop_testing()
            if self.thread and self.thread.isRunning():
                self.thread.wait(2000)
        self.window_closed.emit()
        super().closeEvent(event)

    def setup_stdout_redirect(self):
        """
        Redirects stdout and stderr to the logging area.

        Input:
        - None

        Output:
        - None
        """
        sys.stdout = EmittingStream(textWritten=self.write_stdout)
        sys.stderr = EmittingStream(textWritten=self.write_stderr)

    def write_stdout(self, text):
        """
        Writes stdout text to the logging area.

        Input:
        - text (str): The text to write.

        Output:
        - None
        """
        self.info_output.append(text)

    def write_stderr(self, text):
        """
        Writes stderr text to the logging area.

        Input:
        - text (str): The error text to write.

        Output:
        - None
        """
        self.info_output.setTextColor(Qt.red)
        self.info_output.append(text)
        self.info_output.setTextColor(Qt.white)

class TestingThread(QThread):
    """Thread for running the testing process."""
    output_written = pyqtSignal(str)
    error_written = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, params_path, test_folder_path, GUI_HANDLER=None):
        """
        Initializes the testing thread.

        Input:
        - params_path (str): Path to the testing parameters JSON file.
        - test_folder_path (str): Path to the folder containing test data.
        - GUI_HANDLER (TestingWindow): Reference to the GUI handler.

        Output:
        - None
        """
        super().__init__()
        self.params_path = params_path
        self.test_folder_path = test_folder_path
        self._lock = threading.Lock()
        self._is_running = False
        self.GUI_HANDLER = GUI_HANDLER

    def run(self):
        """
        Executes the testing process.

        Input:
        - None

        Output:
        - None
        """
        try:
            with self._lock:
                self._is_running = True
            sys.stdout = EmittingStream(textWritten=self.output_written.emit)
            sys.stderr = EmittingStream(textWritten=self.error_written.emit)
            goTestingVALID(self.params_path, 
                          test_folder_path=self.test_folder_path,
                          GUI_HANDLER=self.GUI_HANDLER, 
                          STOP_HANDLER=self)
        except Exception as e:
            self.error_occurred.emit(f"Testing thread error: {str(e)}")
            traceback.print_exc()
        finally:
            self._is_running = False
            self.finished.emit()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def stop(self):
        """
        Stops the testing process.

        Input:
        - None

        Output:
        - None
        """
        with self._lock:
            self._is_running = False

def goTestingVALID(param_inputs_path, test_folder_path=None, GUI_HANDLER=None,STOP_HANDLER=None):
    """
    Main function for testing.

    Input:
    - param_inputs_path (str): Path to the testing parameters JSON file.
    - test_folder_path (str): Path to the folder containing test data.
    - GUI_HANDLER (TestingWindow): Reference to the GUI handler.
    - STOP_HANDLER (TestingThread): Reference to the thread handling stop requests.

    Output:
    - None
    """
    args = json2args(param_inputs_path)
    args.test_path = test_folder_path
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("Testing Start!!!")
    test_set = ReadDatasets(
        dataPath=args.test_path,
        dataType=args.data_type,
        dataExtension=args.data_extension,
        dataNum=args.train_frame_num,
        mode="test",
        z_patch=args.z_patch,
        w_patch=args.w_patch,
        h_patch=args.h_patch,
        z_overlap=args.z_overlap,
        w_overlap=args.w_overlap,
        h_overlap=args.h_overlap,
    )

    filename = "models_" + os.path.basename(args.train_folder)
    testSave_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "result", filename
    )
    print("The result saved at : {}".format(testSave_dir))
    if not os.path.exists(testSave_dir):
        os.makedirs(testSave_dir)
    test_loader = DataLoader(
        dataset=test_set,
        sampler=None,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    print("Samples for testing = {}".format(len(test_set)))

    model = Network_CNR(
        in_channels=1, out_channels=1, f_maps=args.base_features, n_groups=args.n_groups
    ).to(device)

    model_path = args.checkpoint_path
    if not os.path.exists(model_path):
        print(f"Model checkpoint '{args.checkpoint}' not found.")
        return

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    model.eval()
    pbar = tqdm(test_loader, total=len(test_loader), dynamic_ncols=False, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    patches = []
    indices = []
    with torch.no_grad():
        for i, data in enumerate(pbar):
            if STOP_HANDLER and not STOP_HANDLER._is_running:
                print("Testing stopped by user.")
                break
            input, image_indices, t_indices, w_indices, h_indices = data
            input = input.to(device)
            output = model(input)
            patches.extend(torch.squeeze(output, dim=1).detach().cpu())
            indices.extend(
                [
                    (image_indices, t_indices, w_indices, h_indices)
                    for image_indices, t_indices, w_indices, h_indices in zip(
                        image_indices, t_indices, w_indices, h_indices
                    )
                ]
            )
        patches = torch.stack(patches)
        restructed_images = test_set.reconstruct_dataset(patches, indices)
        for i in range(len(test_set.inputFileNames)):
            result_name = os.path.join(
                testSave_dir,
                datetime.datetime.now().strftime("%Y%m%d%H%M")
                + "_"
                + test_set.inputFileNames[i],
            )
            root, ext = os.path.splitext(result_name)
            tif_name = root + ".tif"
            restructed_images[i] = torch.clamp(restructed_images[i], min=0, max=65535)
            io.imsave(
                tif_name,
                restructed_images[i].numpy().astype(np.uint16),
            )
    print("-----Testing Finished !-----")
    pbar.close()
