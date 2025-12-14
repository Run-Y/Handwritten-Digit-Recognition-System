import sys
import os
import cv2
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QScrollArea # 记得在顶部导入

# --- 导入自定义模块 ---
from GUI.guiCanvas import PaintBoard # 导入刚才写的画板
from scripts.utils import preprocess_image  # 导入之前写的预处理

# 必须把 scripts 加入路径才能导入 CustomCNN
sys.path.append('scripts')
from scripts.modelDefinition import CustomCNN


class DigitRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手写数字识别系统 (Handwritten Digit Recognition)")
        self.setGeometry(100, 100, 900, 600)

        # 1. 加载模型
        self.load_models()

        # 2. 初始化界面
        self.init_ui()

    def load_models(self):
        """加载 SVM 和 CNN 模型"""
        self.svm_model = None
        self.cnn_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 SVM
        try:
            with open('models/traditional/svm_model.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)
            print("SVM 模型加载成功")
        except Exception as e:
            print(f"SVM 加载失败: {e}")

        # 加载 CNN
        try:
            self.cnn_model = CustomCNN().to(self.device)
            # 加载权重
            self.cnn_model.load_state_dict(
                torch.load('models/deep_learning/cnn_model.pth', map_location=self.device)
            )
            self.cnn_model.eval()  # 设为评估模式
            print("CNN 模型加载成功")
        except Exception as e:
            print(f"CNN 加载失败: {e}")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局：水平分为 左(输入) 中(处理) 右(结果)
        layout = QHBoxLayout()

        # --- 左侧：输入区 (画板 + 按钮) ---
        input_group = QGroupBox("1. 输入 (Input)")
        input_layout = QVBoxLayout()

        self.paint_board = PaintBoard()

        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("清空画板")
        self.btn_clear.clicked.connect(self.paint_board.clear)
        self.btn_upload = QPushButton("上传图片")
        self.btn_upload.clicked.connect(self.upload_image)

        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_upload)

        input_layout.addWidget(self.paint_board)
        input_layout.addLayout(btn_layout)
        input_group.setLayout(input_layout)

        # --- 中间：预处理展示 (Preprocessing) ---
        process_group = QGroupBox("2. 预处理 (Preprocessing)")
        process_layout = QVBoxLayout()

        self.lbl_processed_img = QLabel("预处理图像将显示在此")
        self.lbl_processed_img.setAlignment(Qt.AlignCenter)
        self.lbl_processed_img.setFixedSize(200, 200)
        self.lbl_processed_img.setStyleSheet("background-color: #eee; border: 1px dashed gray;")

        self.combo_model = QComboBox()
        self.combo_model.addItems(["CNN (深度学习)", "SVM (传统机器学习)"])

        self.btn_predict = QPushButton("开始识别 (Recognize)")
        self.btn_predict.setMinimumHeight(40)
        self.btn_predict.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.btn_predict.clicked.connect(self.run_prediction)
        # ... 在 process_group 或者 result_group 里添加一个按钮
        self.btn_eval = QPushButton("查看模型评估 (Evaluation)")
        self.btn_eval.clicked.connect(self.show_evaluation_window)



        process_layout.addWidget(QLabel("二值化/灰度化结果:"))
        process_layout.addWidget(self.lbl_processed_img)
        process_layout.addStretch()
        process_layout.addWidget(QLabel("选择模型:"))
        process_layout.addWidget(self.combo_model)
        process_layout.addWidget(self.btn_predict)
        process_group.setLayout(process_layout)

        # --- 右侧：结果展示 (Result) ---
        result_group = QGroupBox("3. 结果 (Result)")
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.btn_eval)  # 放在结果区下面

        self.lbl_result = QLabel("?")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setStyleSheet("font-size: 80px; color: blue; font-weight: bold;")

        self.lbl_confidence = QLabel("置信度: -")
        self.lbl_confidence.setStyleSheet("font-size: 16px;")

        result_layout.addWidget(QLabel("识别结果:"))
        result_layout.addWidget(self.lbl_result)
        result_layout.addWidget(self.lbl_confidence)
        result_layout.addStretch()
        result_group.setLayout(result_layout)

        # 添加到主布局
        layout.addWidget(input_group)
        layout.addWidget(process_group)
        layout.addWidget(result_group)

        main_widget.setLayout(layout)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image files (*.jpg *.png *.jpeg)')
        if fname:
            # 加载并显示在画板位置（或者单独处理），这里为了简单，我们只是保存路径供后续使用
            # 更好的做法是把图片画到画板上
            img = QImage(fname)
            if not img.isNull():
                self.paint_board.image = img.scaled(self.paint_board.size())
                self.paint_board.update()

    def run_prediction(self):
        # 1. 从画板获取图像并保存为临时文件
        # 这是为了复用 utils.py 中的 preprocess_image 接口
        temp_path = "temp_input.jpg"
        self.paint_board.save_image(temp_path)

        # 2. 调用 utils.py 进行预处理
        # 注意：这里有个坑！画板是白底黑字，MNIST是黑底白字。
        # 我们需要在 utils 里或者这里做颜色反转。
        # 既然 utils 是通用的，我们可以在这里先反转颜色再保存，或者假设 utils 处理了。
        # 通常手写板需要自己反转：

        # --- 手动读取并反转颜色 (Invert Colors) ---
        img_cv = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        # 如果图片主要是白色的（平均像素 > 127），则反转
        if np.mean(img_cv) > 127:
            img_cv = cv2.bitwise_not(img_cv)
            cv2.imwrite(temp_path, img_cv)  # 覆盖保存反转后的黑底白字图

        # 3. 使用 utils 预处理
        # binary=True 以获得更清晰的二值化效果用于显示
        model_input, display_img = preprocess_image(temp_path, target_size=(28, 28), binary=True)

        if model_input is None:
            QMessageBox.warning(self, "错误", "无法处理图像")
            return

        # 4. 在界面显示预处理后的图
        h, w = display_img.shape
        q_img = QImage(display_img.data, w, h, w, QImage.Format_Grayscale8)
        self.lbl_processed_img.setPixmap(QPixmap.fromImage(q_img).scaled(200, 200, Qt.KeepAspectRatio))

        # 5. 预测
        model_type = self.combo_model.currentText()

        if "SVM" in model_type:
            if self.svm_model:
                # SVM 需要展平 (1, 784)
                flat_data = model_input.flatten().reshape(1, -1)
                prediction = self.svm_model.predict(flat_data)[0]
                self.lbl_result.setText(str(prediction))
                self.lbl_confidence.setText("置信度: N/A (SVM)")
            else:
                QMessageBox.critical(self, "错误", "SVM 模型未加载！")

        elif "CNN" in model_type:
            if self.cnn_model:
                # CNN 需要 (1, 1, 28, 28)
                tensor_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.cnn_model(tensor_input)
                    probabilities = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                self.lbl_result.setText(str(predicted.item()))
                self.lbl_confidence.setText(f"置信度: {confidence.item() * 100:.2f}%")
            else:
                QMessageBox.critical(self, "错误", "CNN 模型未加载！")

    def show_evaluation_window(self):
        """弹出一个新窗口展示图表"""
        dialog = QDialog(self)
        dialog.setWindowTitle("模型性能评估 (Model Evaluation)")
        dialog.resize(800, 600)

        v_layout = QVBoxLayout()

        # 创建一个滚动区域，防止图太大显示不全
        scroll = QScrollArea()
        content_widget = QWidget()
        content_layout = QVBoxLayout()

        # 定义要展示的图片列表
        images_to_show = [
            ("CNN 学习曲线 (Learning Curve)", "results/cnn_learning_curve.png"),
            ("CNN 混淆矩阵 (Confusion Matrix)", "results/cnn_conf_matrix.png"),
            ("SVM 混淆矩阵 (Confusion Matrix)", "results/svm_conf_matrix.png")
        ]

        for title, path in images_to_show:
            if os.path.exists(path):
                # 标题
                content_layout.addWidget(QLabel(f"<b>{title}</b>"))
                # 图片
                lbl_img = QLabel()
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    # 缩放一下适应窗口宽度
                    lbl_img.setPixmap(pixmap.scaledToWidth(700, Qt.SmoothTransformation))
                    content_layout.addWidget(lbl_img)
            else:
                content_layout.addWidget(QLabel(f"⚠️ 未找到文件: {path} (请运行 generate_charts.py)"))

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