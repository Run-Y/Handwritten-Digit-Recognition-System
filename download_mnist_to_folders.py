import os
from torchvision import datasets
import cv2
import numpy as np

# 这一步会自动下载那4个文件，并解压到 ./temp_data 文件夹下
# download=True 表示如果没这4个文件，它就自动去下；如果有，就直接用
print("正在检查/下载 MNIST 数据 (包含那4个文件)...")
train_data = datasets.MNIST(root='./temp_data', train=True, download=True)
test_data = datasets.MNIST(root='./temp_data', train=False, download=True)

# 定义我们要把它们保存到哪里 (作为你后续项目的原始数据)
save_dir = 'rawDataset'


def save_images(dataset, prefix):
    print(f"正在转换 {prefix} 数据...")
    for i, (img, label) in enumerate(dataset):
        # 创建对应的数字文件夹，例如 raw_dataset/0/
        folder = os.path.join(save_dir, str(label))
        os.makedirs(folder, exist_ok=True)

        # 保存为jpg
        # 文件名示例: raw_dataset/5/train_0001.jpg
        filename = f"{prefix}_{i}.jpg"
        img_np = np.array(img)
        cv2.imwrite(os.path.join(folder, filename), img_np)


# 执行转换
save_images(train_data, "train")
save_images(test_data, "test")

print(f"\n搞定！那4个复杂的文件已经被转换成了 {save_dir} 文件夹里的 .jpg 图片。")
print("你现在可以去文件夹里直接查看这些图片了。")