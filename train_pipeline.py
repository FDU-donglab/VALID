# -*- coding: utf-8 -*-
import time
import os
import sys
import json
import traceback
import argparse
import datetime
import threading

from torch.utils.data import DataLoader
import numpy as np
from skimage import io
from tqdm import tqdm
import setproctitle

from datasets.dataset_fs import ReadDatasets, custom_collate_fn
from datasets.sampling import *
from models.network import Network_CNR, HessianConstraintLoss3D
from utils import *
from utils.config import json2args, args2json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSplitter, QTextEdit, QLineEdit, QGridLayout,
                             QSlider, QMessageBox)
from PyQt5.QtGui import QPixmap, QTextCursor, QImage, QIcon
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


class TrainingThread(QThread):
    """Thread for running the training process."""
    finished = pyqtSignal()
    output_written = pyqtSignal(str)
    error_written = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, params_path=None, GUI_HANDLER=None):
        """
        Initializes the training thread.

        Input:
        - params_path (str): Path to the training parameters JSON file.
        - GUI_HANDLER (TrainingWindow): Reference to the GUI handler.

        Output:
        - None
        """
        super().__init__()
        self.GUI_HANDLER = GUI_HANDLER
        self._is_running = False
        self.params_path = params_path
        self._lock = threading.Lock()

    def run(self):
        """
        Executes the training process.

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
            goTrainingVALID(self.params_path, GUI_HANDLER=self.GUI_HANDLER, STOP_HANDLER=self)
        except Exception as e:
            self.error_occurred.emit(f"Training thread error: {str(e)}")
            traceback.print_exc()
        finally:
            self._is_running = False
            self.finished.emit()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def stop(self):
        """
        Stops the training process.

        Input:
        - None

        Output:
        - None
        """
        with self._lock:
            self._is_running = False


class TrainingWindow(QMainWindow):
    """Main window for the training GUI."""
    window_closed = pyqtSignal()

    def __init__(self):
        """
        Initializes the training window.

        Input:
        - None

        Output:
        - None
        """
        super().__init__()
        self.setWindowTitle("Training VALID")
        self.setWindowIcon(QIcon("./resource/icon.ico"))
        self.setGeometry(100, 100, 500, 500)

        self.thread = None
        self.is_training = False
        self.thread_lock = threading.Lock()

        self.init_ui()
        self.apply_styles()
        self.setup_exception_handling()

    def init_ui(self):
        """
        Initializes the UI components.

        Input:
        - None

        Output:
        - None
        """
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Upper part of the left side
        upper_left_widget = QWidget()
        upper_left_layout = QGridLayout()
        self.param_inputs = {}
        params = {
            "epochs": 100,
            "train_frame_num": 99999,
            "w_patch": 256,
            "h_patch": 256,
            "z_patch": 64,
            "w_overlap": 0.1,
            "h_overlap": 0.1,
            "z_overlap": 0.1,
            "patch_num": -1,
            "gpu_ids": [0],
            "save_freq": 10,
            "train_folder": r"./data/train",
            "num_workers": 0,
            "weight_reg": 0.0001,
        }
        self.default_params = {
            "data_extension": "tif",
            "withGT": False,
            "lr": 0.0001,
            "amsgrad": True,
            "base_features": 16,
            "n_groups": 4,
            "train": True,
            "test": False,
            "data_type": "3D",
            "seed": 3407,
            "clip_gradients": 20.0,
            "mode": "train",
            "batch_size": 1
        }
        row = 0
        col = 0
        for key, value in params.items():
            self.param_inputs[key] = QLineEdit(str(value))
            upper_left_layout.addWidget(QLabel(key), row, col * 2)
            upper_left_layout.addWidget(self.param_inputs[key], row, col * 2 + 1)
            if col == 1:
                row += 1
            col = (col + 1) % 2

        button_layout_start = QHBoxLayout()
        button_layout_stop = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        button_layout_start.addWidget(self.start_button)
        button_layout_stop.addWidget(self.stop_button)
        upper_left_layout.addLayout(button_layout_start, row, col * 2 + 2)
        upper_left_layout.addLayout(button_layout_stop, row, col * 2 + 3)
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

        # Right side
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        self.image_display_1 = QLabel(self)
        self.image_display_2 = QLabel(self)
        self.show_default_images()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, params["z_patch"])
        self.slider.valueChanged.connect(self.update_slices)

        right_layout.addWidget(self.image_display_1)
        right_layout.addWidget(self.slider)
        right_layout.addWidget(self.image_display_2)
        right_widget.setLayout(right_layout)

        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.addWidget(main_splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
    def setup_exception_handling(self):
        def excepthook(exc_type, exc_value, exc_tb):
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.show_critical_error(f"Unhandled exception:\n{tb}")
            QApplication.quit()
        sys.excepthook = excepthook

    def show_critical_error(self, message):
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
    def update_volume_data(self, raw_data, denoised_data):
        self.volume_data1 = raw_data
        self.volume_data2 = denoised_data
        self.update_slices()
    def handle_load_failure(self):
        self.show_default_images()
    def update_slices(self):
        if self.volume_data1 is not None and self.volume_data2 is not None:
            index = self.slider.value()
            pixmap = self.array_to_pixmap(self.volume_data1[index,...])
            width = pixmap.width()
            height = pixmap.height()
            if width / self.image_display_1.width() >= height / self.image_display_1.height():
                ratio = width / self.image_display_1.width()
            else:
                ratio = height / self.label.height()
            new_width = width / ratio
            new_height = height / ratio
            pixmap = pixmap.scaled(new_width, new_height)
            self.image_display_1.setPixmap(pixmap)

            pixmap = self.array_to_pixmap(self.volume_data2[index, ...])
            width = pixmap.width()
            height = pixmap.height()
            if width / self.image_display_1.width() >= height / self.image_display_1.height():
                ratio = width / self.image_display_1.width()
            else:
                ratio = height / self.label.height()
            new_width = width / ratio
            new_height = height / ratio
            pixmap = pixmap.scaled(new_width, new_height)
            self.image_display_2.setPixmap(pixmap)
    def show_default_images(self):
        pixmap1 = QPixmap("./resource/f2.png")
        self.image_display_1.setPixmap(pixmap1)
        self.image_display_1.setFixedSize(pixmap1.size())
        pixmap2 = QPixmap("./resource/f1.png")
        self.image_display_2.setPixmap(pixmap2)
        self.image_display_2.setFixedSize(pixmap2.size())

    def array_to_pixmap(self, array):
        array = (array - array.min()) / (array.max() - array.min() + 1e-8) * 255
        array = array.astype(np.uint8)
        height, width = array.shape
        bytes_per_line = width
        q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        return QPixmap.fromImage(q_image)
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
            QLineEdit, QTextEdit {
                font-family: Arial;
                font-size: 14px;
                color: white;
                background-color: #4A4A4A;
                border: none;
                padding: 10px;
            }
            QSlider {
                background-color: #4A4A4A;
            }
        """)

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

    def start_training(self):
        """
        Starts the training process.

        Input:
        - None

        Output:
        - None
        """
        if self.is_training:
            QMessageBox.warning(self, "Warning", "Training is already running!")
            return
        try:
            self.is_training = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            params = {}
            for key in self.param_inputs:
                value = self.param_inputs[key].text()
                if key in ["epochs", "train_frame_num", "w_patch", "h_patch", "z_patch", "patch_num", "save_freq", "num_workers"]:
                    params[key] = int(value)
                elif key in ["w_overlap", "h_overlap", "z_overlap","weight_reg"]:
                    params[key] = float(value)
                elif key == "gpu_ids":
                    params[key] = [int(i) for i in value.strip('[]').split(',')]
                else:
                    params[key] = value

            args = argparse.Namespace(**(dict(params, **(self.default_params))))
            self.train_json_params_path = "./jsons/train_params.json"
            with open(self.train_json_params_path, mode="w") as f:
                json.dump(args.__dict__, f, indent=4)
            with self.thread_lock:
                self.thread = TrainingThread(params_path=self.train_json_params_path, GUI_HANDLER=self)
                self.thread.output_written.connect(self.handle_stdout)
                self.thread.error_written.connect(self.handle_stderr)
                self.thread.finished.connect(self.on_training_finished)
                self.thread.error_occurred.connect(self.handle_thread_error)
                self.thread.start()
        except Exception as e:
            self.handle_thread_error(str(e))

    def stop_training(self):
        """
        Stops the training process.

        Input:
        - None

        Output:
        - None
        """
        with self.thread_lock:
            if self.thread and self.thread.isRunning():
                self.thread.stop()
                self.info_output.append("The training is being stopped...")
                self.thread.quit()
                if not self.thread.wait(3000):
                    self.thread.terminate()
                    self.thread.wait()
                self.is_training = False
                self.stop_button.setEnabled(False)
                self.start_button.setEnabled(True)
                self.thread = None

    def on_training_finished(self):
        """
        Handles the completion of the training process.

        Input:
        - None

        Output:
        - None
        """
        self.info_output.append("-----Training finished!-----")
        self.is_training = False
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        with self.thread_lock:
            self.thread = None

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

    def handle_thread_error(self, error_msg):
        """
        Handles errors in the training thread.

        Input:
        - error_msg (str): The error message to display.

        Output:
        - None
        """
        self.show_critical_error(error_msg)

    def closeEvent(self, event):
        """
        Handles the window close event.

        Input:
        - event (QCloseEvent): The close event.

        Output:
        - None
        """
        if self.is_training:
            self.stop_training()
            if self.thread.isRunning():
                self.thread.wait(3000)
        self.window_closed.emit()
        super().closeEvent(event)

    def show_default_images(self):
        """
        Displays default images in the UI.

        Input:
        - None

        Output:
        - None
        """
        pixmap1 = QPixmap("./resource/f2.png")
        self.image_display_1.setPixmap(pixmap1)
        self.image_display_1.setFixedSize(pixmap1.size())
        pixmap2 = QPixmap("./resource/f1.png")
        self.image_display_2.setPixmap(pixmap2)
        self.image_display_2.setFixedSize(pixmap2.size())

    def update_slices(self):
        """
        Updates the displayed slices based on the slider value.

        Input:
        - None

        Output:
        - None
        """
        if self.volume_data1 is not None and self.volume_data2 is not None:
            index = self.slider.value()
            pixmap = self.array_to_pixmap(self.volume_data1[index, ...])
            self.image_display_1.setPixmap(pixmap)
            pixmap = self.array_to_pixmap(self.volume_data2[index, ...])
            self.image_display_2.setPixmap(pixmap)

    def array_to_pixmap(self, array):
        """
        Converts a NumPy array to a QPixmap.

        Input:
        - array (np.ndarray): The array to convert.

        Output:
        - QPixmap: The converted pixmap.
        """
        array = (array - array.min()) / (array.max() - array.min() + 1e-8) * 255
        array = array.astype(np.uint8)
        height, width = array.shape
        bytes_per_line = width
        q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_image)


def goTrainingVALID(param_inputs_path, GUI_HANDLER=None, STOP_HANDLER=None):
    """
    Main function for training.

    Input:
    - param_inputs_path (str): Path to the training parameters JSON file.
    - GUI_HANDLER (TrainingWindow): Reference to the GUI handler.
    - STOP_HANDLER (TrainingThread): Reference to the thread handling stop requests.

    Output:
    - None
    """
    args = json2args(param_inputs_path)
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Print GPU information
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Current Device: {torch.cuda.current_device()}\n")
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "log")
    log_file = log_dir + ".txt"
    log_args(log_file)
    print("-----This is all configurations-----")
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    print("-----This is a halving line-----")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Define the model and loss functions
    model = Network_CNR(
        in_channels=1, out_channels=1, f_maps=args.base_features, n_groups=args.n_groups
    )
    HessianLoss3D = HessianConstraintLoss3D()
    L2_pixelwise = torch.nn.MSELoss()

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lr_func = lambda epoch: min((epoch + 1) / (args.epochs * 0.1 + 1e-8), 0.5 * (math.cos(epoch / args.epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    checkpoint_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "checkpoint",
        os.path.basename(args.train_folder)
        + datetime.datetime.now().strftime("%Y%m%d%H%M"),
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logging.info("Training Start!!!")
    train_set = ReadDatasets(
        dataPath=args.train_folder,
        dataType=args.data_type,
        dataExtension=args.data_extension,
        dataNum=args.train_frame_num,
        mode="train",
        z_patch=args.z_patch,
        w_patch=args.w_patch,
        h_patch=args.h_patch,
        z_overlap=args.z_overlap,
        w_overlap=args.w_overlap,
        h_overlap=args.h_overlap,
        patch_num=args.patch_num
    )

    if not args.withGT:
        args.val_folder = args.train_folder

    train_loader = DataLoader(
        dataset=train_set,
        sampler=None,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    DWT3D = DWT_3D(wavename="haar")
    print("Samples for train = {}".format(len(train_set)))
    start_time = time.time()
    torch.set_grad_enabled(True)
    print(("\n" + "%15s" * 3) % ("Epoch", "GPU_mem", "total_loss"))
    pbar = range(args.epochs)
    pbar = tqdm(pbar, total=args.epochs, dynamic_ncols=False, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

    for epoch in pbar:
        setproctitle.setproctitle("{}/{}".format(epoch + 1, args.epochs))
        for batch_idx, data in enumerate(train_loader):
            if not STOP_HANDLER._is_running:
                print("Training stopped by user.")
                break  # Stop training
            adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
            input, _, _, _, _ = data
            input = input.to(device)
            mask1, mask2, mask3, mask4 = generate_mask_pair(input.squeeze(1))
            noisy_sub1 = generate_subimages(input.squeeze(1), mask1).unsqueeze(1)
            noisy_sub2 = generate_subimages(input.squeeze(1), mask2).unsqueeze(1)
            noisy_sub3 = generate_subimages(input.squeeze(1), mask3).unsqueeze(1)
            noisy_sub4 = generate_subimages(input.squeeze(1), mask4).unsqueeze(1)

            noisy_output_1 = model(noisy_sub1)
            noisy_output_2 = model(noisy_sub2)

            LLL_n1, LLH_n1, LHL_n1, _, HLL_n1, _, _, _ = DWT3D(noisy_output_1)
            LLL_n2, LLH_n2, LHL_n2, _, HLL_n2, _, _, _ = DWT3D(noisy_output_1)
            loss2neighbor = 0.5 * L2_pixelwise(noisy_output_1, noisy_sub3) + 0.5 * L2_pixelwise(noisy_output_2, noisy_sub4)
            loss_idt = L2_pixelwise(noisy_output_1, noisy_output_2)
            loss_reg = HessianLoss3D(torch.cat([LLL_n1, LLH_n1, LHL_n1, HLL_n1, LLL_n2, LLH_n2, LHL_n2, HLL_n2], dim=1))

            optimizer.zero_grad()
            # Total loss
            Total_loss = loss2neighbor + loss_idt + args.weight_reg * loss_reg
            Total_loss.backward()
            optimizer.step()

            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
            pbar.set_description(
                f"Epoch:{epoch + 1}/{args.epochs}, Batch:{batch_idx + 1}/{len(train_loader)}\n"
                + "Mem:{},loss:{:.4f}".format(mem, Total_loss))

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.clip_gradients
            )
        if not STOP_HANDLER._is_running:
            break  # Stop training

        if (epoch + 1) % int(args.save_freq) == 0:
            model.eval()
            print(("Validating...\n"))
            patches = []
            input_patches = []
            indices = []
            with torch.no_grad():
                for data in train_loader:
                    input, image_indices, t_indices, w_indices, h_indices = data
                    input = input.to(device)
                    output = model(input)
                    input_patches.extend(torch.squeeze(input, dim=1).detach().cpu())
                    patches.extend(torch.squeeze(output, dim=1).detach().cpu())
                    indices.extend(
                        [
                            (image_indices, t_indices, w_indices, h_indices)
                            for image_indices, t_indices, w_indices, h_indices in zip(
                                image_indices, t_indices, w_indices, h_indices
                            )
                        ]
                    )
                input_patches = torch.stack(input_patches)
                patches = torch.stack(patches)
                input_images = train_set.reconstruct_dataset(input_patches, indices)

                restructed_images = train_set.reconstruct_dataset(patches, indices)
                display_raw_images = torch.clamp(input_images[0], min=0, max=65535).numpy().astype(np.uint16)
                display_valid_images = torch.clamp(restructed_images[0], min=0, max=65535).numpy().astype(np.uint16)
                GUI_HANDLER.update_volume_data(display_raw_images, display_valid_images)
            filename = (
                os.path.basename(args.train_folder)
                + "_Epoch_"
                + str(epoch + 1)
                + ".pth"
            )
            final_name = os.path.join(checkpoint_dir, filename)
            # Check if the model is wrapped in DataParallel
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(
                {
                    "state_dict": state_dict,
                },
                final_name,
            )

            print("model is saved in " + filename)
            args.checkpoint_path = final_name
            args.mode = "test"
            args2json(args, checkpoint_dir + "//config.json")
            model.train()
        lr_scheduler.step()
        print(f"Epoch {epoch}, Learning Rate: {lr_scheduler.get_last_lr()[0]}")
        torch.cuda.empty_cache()
    end_time = time.time()
    total_time = (end_time - start_time) / 3600

    print("The total training time is {:.2f} hours".format(total_time))
    print("-----The training process finished!-----")
