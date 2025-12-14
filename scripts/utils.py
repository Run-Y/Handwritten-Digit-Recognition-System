# utils.py
import cv2
import numpy as np

"""
为什么需要这个文件？ 
这是为了解决我之前提到的一致性问题。
我们在训练前处理数据用的逻辑（比如转灰度、调整大小）
在 GUI 识别用户上传的图片时必须一模一样。
"""
def preprocess_image(image_path, target_size=(28, 28), binary=False):
    """
    预处理图像，返回两个结果：
    1. model_input: 用于模型训练/推理 (Float32, 0.0-1.0)
    2. display_img: 用于 GUI 展示 (Uint8, 0-255)
    """
    # 1. 读取图像
    # cv2.IMREAD_GRAYSCALE 直接读进来就是灰度，省去转换步骤
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None, None  # 返回两个 None

    # 2. 二值化 (Binarization) - 如果 GUI 勾选了二值化或者是训练需要
    if binary:
        # 使用 OTSU 自动阈值
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 调整尺寸 (Resize) - 比如调整到 28x28
    # 这就是 GUI 即将展示的样子 (0-255)
    display_img = cv2.resize(img, target_size)

    # 4. 准备模型输入 (Normalization)
    # 转换类型并归一化到 0-1
    model_input = display_img.astype('float32') / 255.0

    # 返回 (模型用的数据, GUI展示用的图片)
    return model_input, display_img