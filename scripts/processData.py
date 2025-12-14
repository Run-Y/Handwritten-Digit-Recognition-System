# step1_process_data.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import preprocess_image  # 导入刚才写的通用函数

# 配置路径
RAW_DATA_DIR = '../rawDataset'  # 你的原始文件夹路径
OUTPUT_DIR = '../processedData'  # 处理后数据的保存路径
TARGET_SIZE = (28, 28)  # 统一尺寸，你可以改为 (64, 64)


def create_dataset():
    all_images = []
    all_labels = []

    print("开始读取并处理原始图片...")

    # 遍历 0 到 9 的文件夹
    for label in range(10):
        folder_path = os.path.join(RAW_DATA_DIR, str(label))
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder_path} 不存在，跳过。")
            continue

        files = os.listdir(folder_path)
        print(f"正在处理数字 '{label}' 的文件夹，共 {len(files)} 张图片...")

        for file_name in files:
            # 检查文件扩展名
            if not file_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue

            file_path = os.path.join(folder_path, file_name)

            # 注意这里：用逗号接收两个返回值，但我只需要 model_input 用于保存
            # binary=True/False 取决于你是否决定强制二值化训练数据
            model_input, _ = preprocess_image(file_path, TARGET_SIZE, binary=True)

            if model_input is not None:
                all_images.append(model_input)  # 只存模型数据
                all_labels.append(label)

    # 转换为 NumPy 数组
    X = np.array(all_images)
    y = np.array(all_labels)

    # 增加一个维度以适配 CNN (N, H, W) -> (N, 1, H, W) 或 (N, H, W, 1)
    # PyTorch 通常需要 (N, 1, H, W)
    X = X[:, np.newaxis, :, :]

    print(f"\n读取完成！总样本数: {len(X)}")
    print(f"数据形状: {X.shape}")

    # --- 数据集划分 ---
    # 1. 先划分出 训练集 (Training) 和 剩余数据 (Temp)
    # 假设比例：训练 70%, 验证 15%, 测试 15%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2. 再将 剩余数据 划分为 验证集 (Validation) 和 测试集 (Test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"训练集: {len(X_train)}")
    print(f"验证集: {len(X_val)}")
    print(f"测试集: {len(X_test)}")

    # --- 保存数据 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_path = os.path.join(OUTPUT_DIR, 'mnist_custom_data.npz')
    np.savez_compressed(
        save_path,
        x_train=X_train, y_train=y_train,
        x_val=X_val, y_val=y_val,
        x_test=X_test, y_test=y_test
    )
    print(f"\n所有数据已保存至: {save_path}")
    print("你可以直接在训练脚本中加载这个文件，无需再次读取原始图片。")


if __name__ == '__main__':
    create_dataset()