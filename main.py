import sys
import os
import cv2
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QFileDialog, QMessageBox, QGroupBox, QDialog, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# --- Import Custom Modules ---
from GUI.guiCanvas import PaintBoard  # Import the drawing board
from scripts.utils import preprocess_image  # Import preprocessing function

# Must add 'scripts' to path to import CustomCNN
sys.path.append('scripts')
from scripts.modelDefinition import CustomCNN


class DigitRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handwritten Digit Recognition System")
        self.setGeometry(100, 100, 900, 600)

        # 1. Load Models
        self.load_models()

        # 2. Initialize UI
        self.init_ui()

    def load_models(self):
        """Loads SVM and CNN models"""
        self.svm_model = None
        self.cnn_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load SVM
        try:
            with open('models/traditional/svm_model.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)
            print("SVM Model loaded successfully")
        except Exception as e:
            print(f"SVM loading failed: {e}")

        # Load CNN
        try:
            self.cnn_model = CustomCNN().to(self.device)
            # Load weights
            self.cnn_model.load_state_dict(
                torch.load('models/deep_learning/cnn_model.pth', map_location=self.device)
            )
            self.cnn_model.eval()  # Set to evaluation mode
            print("CNN Model loaded successfully")
        except Exception as e:
            print(f"CNN loading failed: {e}")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main Layout: Horizontal split (Input | Preprocessing | Result)
        layout = QHBoxLayout()

        # --- Left: Input Area (PaintBoard + Buttons) ---
        input_group = QGroupBox("1. Input")
        input_layout = QVBoxLayout()

        self.paint_board = PaintBoard()

        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Clear Board")
        self.btn_clear.clicked.connect(self.paint_board.clear)
        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.clicked.connect(self.upload_image)

        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_upload)

        input_layout.addWidget(self.paint_board)
        input_layout.addLayout(btn_layout)
        input_group.setLayout(input_layout)

        # --- Center: Preprocessing & Model Selection ---
        process_group = QGroupBox("2. Preprocessing & Model")
        process_layout = QVBoxLayout()

        self.lbl_processed_img = QLabel("Processed Image Display")
        self.lbl_processed_img.setAlignment(Qt.AlignCenter)
        self.lbl_processed_img.setFixedSize(200, 200)
        self.lbl_processed_img.setStyleSheet("background-color: #eee; border: 1px dashed gray;")

        self.combo_model = QComboBox()
        self.combo_model.addItems(["CNN (Deep Learning)", "SVM (Traditional ML)"])

        self.btn_predict = QPushButton("Recognize Digit")
        self.btn_predict.setMinimumHeight(40)
        self.btn_predict.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.btn_predict.clicked.connect(self.run_prediction)

        self.btn_eval = QPushButton("View Evaluation")
        self.btn_eval.clicked.connect(self.show_evaluation_window)

        process_layout.addWidget(QLabel("Grayscale / Binarized Result:"))
        process_layout.addWidget(self.lbl_processed_img)
        process_layout.addStretch()
        process_layout.addWidget(QLabel("Select Model:"))
        process_layout.addWidget(self.combo_model)
        process_layout.addWidget(self.btn_predict)
        process_group.setLayout(process_layout)

        # --- Right: Result Display ---
        result_group = QGroupBox("3. Result")
        result_layout = QVBoxLayout()

        # Move evaluation button to result group for better flow (optional re-arrangement)
        result_layout.addWidget(self.btn_eval)

        self.lbl_result = QLabel("?")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setStyleSheet("font-size: 80px; color: blue; font-weight: bold;")

        self.lbl_confidence = QLabel("Confidence: -")
        self.lbl_confidence.setStyleSheet("font-size: 16px;")

        result_layout.addWidget(QLabel("Recognition Output:"))
        result_layout.addWidget(self.lbl_result)
        result_layout.addWidget(self.lbl_confidence)
        result_layout.addStretch()
        result_group.setLayout(result_layout)

        # Add to main layout
        layout.addWidget(input_group)
        layout.addWidget(process_group)
        layout.addWidget(result_group)

        main_widget.setLayout(layout)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', '.', 'Image files (*.jpg *.png *.jpeg)')
        if fname:
            # Load and display the image on the board area
            img = QImage(fname)
            if not img.isNull():
                self.paint_board.image = img.scaled(self.paint_board.size())
                self.paint_board.update()

    def run_prediction(self):
        # 1. Get image from board and save to a temporary file
        temp_path = "temp_input.jpg"
        self.paint_board.save_image(temp_path)

        # --- Invert Colors if necessary (White background written with black ink) ---
        img_cv = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)

        if img_cv is None:
            QMessageBox.warning(self, "Error", "Could not load image file.")
            return

        # If the image is mostly white (average pixel value > 127), invert it to black background
        if np.mean(img_cv) > 127:
            img_cv = cv2.bitwise_not(img_cv)
            cv2.imwrite(temp_path, img_cv)  # Overwrite with inverted image

        # 3. Preprocess using utils
        # binary=True for cleaner visualization and model input
        model_input, display_img = preprocess_image(temp_path, target_size=(28, 28), binary=True)

        if model_input is None:
            QMessageBox.warning(self, "Error", "Image processing failed.")
            return

        # 4. Display the processed image in the GUI
        h, w = display_img.shape
        q_img = QImage(display_img.data, w, h, w, QImage.Format_Grayscale8)
        self.lbl_processed_img.setPixmap(QPixmap.fromImage(q_img).scaled(200, 200, Qt.KeepAspectRatio))

        # 5. Predict
        model_type = self.combo_model.currentText()

        if "SVM" in model_type:
            if self.svm_model:
                # SVM requires flattened data (1, 784)
                flat_data = model_input.flatten().reshape(1, -1)
                prediction = self.svm_model.predict(flat_data)[0]
                self.lbl_result.setText(str(prediction))
                self.lbl_confidence.setText("Confidence: N/A (SVM)")
            else:
                QMessageBox.critical(self, "Error", "SVM Model not loaded!")

        elif "CNN" in model_type:
            if self.cnn_model:
                # CNN requires (1, 1, 28, 28)
                tensor_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.cnn_model(tensor_input)
                    probabilities = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                self.lbl_result.setText(str(predicted.item()))
                self.lbl_confidence.setText(f"Confidence: {confidence.item() * 100:.2f}%")
            else:
                QMessageBox.critical(self, "Error", "CNN Model not loaded!")

    def read_report_content(self, path):
        """尝试读取报告文件内容，如果失败则返回提示信息"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                return f"⚠️ Error reading file: {path}"
        else:
            return f"⚠️ File not found: {path} (Please run training scripts)"

    def show_evaluation_window(self):
        """Pops up a new window to display evaluation charts AND reports"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Model Performance Evaluation")
        dialog.resize(1000, 750)  # 增大窗口，以容纳文本报告

        v_layout = QVBoxLayout()
        scroll = QScrollArea()
        content_widget = QWidget()
        content_layout = QVBoxLayout()

        # --- 1. 图片展示 (Visualization) ---
        images_to_show = [
            ("CNN Learning Curve", "results/cnn_learning_curve.png"),
            ("CNN Confusion Matrix", "results/chart_and_report/cnn_conf_matrix.png"),
            ("SVM Confusion Matrix", "results/chart_and_report/svm_conf_matrix.png")
        ]

        for title, path in images_to_show:
            if os.path.exists(path):
                content_layout.addWidget(QLabel(f"<h2>{title}</h2>"))
                lbl_img = QLabel()
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    lbl_img.setPixmap(pixmap.scaledToWidth(950, Qt.SmoothTransformation))
                    content_layout.addWidget(lbl_img)
            else:
                content_layout.addWidget(QLabel(f"⚠️ File not found: {path} (Please run training scripts)"))

        # --- 2. 报告文本展示 (Metrics: P/R/F1) ---
        content_layout.addWidget(QLabel("<h2>Classification Reports (P/R/F1 Metrics)</h2>"))

        # CNN Report
        cnn_report_path = "results/chart_and_report/cnn_report.txt"
        cnn_report_content = self.read_report_content(cnn_report_path)
        content_layout.addWidget(QLabel("<h3>CNN Report:</h3>"))
        lbl_cnn_report = QLabel(cnn_report_content)
        lbl_cnn_report.setStyleSheet("background-color: #f0f0f0; padding: 10px; font-family: monospace;")
        content_layout.addWidget(lbl_cnn_report)

        # SVM Report
        svm_report_path = "results/chart_and_report/svm_report.txt"
        svm_report_content = self.read_report_content(svm_report_path)
        content_layout.addWidget(QLabel("<h3>SVM Report:</h3>"))
        lbl_svm_report = QLabel(svm_report_content)
        lbl_svm_report.setStyleSheet("background-color: #f0f0f0; padding: 10px; font-family: monospace;")
        content_layout.addWidget(lbl_svm_report)

        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)

        v_layout.addWidget(scroll)
        dialog.setLayout(v_layout)
        dialog.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DigitRecognizerApp()
    window.show()
    sys.exit(app.exec_())