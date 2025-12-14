import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time  # 【新增】用于计时
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report  # 【新增】用于生成报告

# 导入刚才定义的模型
# 确保你的 scripts 文件夹下有 modelDefinition.py
from modelDefinition import CustomCNN

# 配置
DATA_PATH = '../processedData/mnist_custom_data.npz'
MODEL_SAVE_DIR = '../models/deep_learning'
RESULT_SAVE_DIR = '../results'  # 【新增】结果保存路径
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保文件夹存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)


def load_pytorch_data():
    """读取 npz 并封装为 PyTorch DataLoader"""
    print("正在加载数据...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到文件: {DATA_PATH}，请先运行 step1_process_data.py")

    data = np.load(DATA_PATH)

    # 1. 训练集
    x_train = torch.tensor(data['x_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.long)

    # 2. 验证集
    x_val = torch.tensor(data['x_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.long)

    # 3. 【新增】测试集 (用于最终生成报告)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # 创建 Dataset
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    test_ds = TensorDataset(x_test, y_test)  # 【新增】

    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)  # 【新增】

    return train_loader, val_loader, test_loader


def train_cnn():
    print(f"使用设备: {DEVICE}")
    # 【修改】接收三个 loader
    train_loader, val_loader, test_loader = load_pytorch_data()

    # 1. 初始化模型
    model = CustomCNN().to(DEVICE)

    # 2. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}

    print(f"\n--- 开始训练 CNN (共 {EPOCHS} 轮) ---")
    start_time = time.time()  # 开始计时

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    end_time = time.time()
    duration = end_time - start_time
    print(f"训练完成！耗时: {duration:.2f} 秒")

    # 3. 保存模型
    save_path = os.path.join(MODEL_SAVE_DIR, 'cnn_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

    # 4. 保存学习曲线图
    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('CNN Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(RESULT_SAVE_DIR, 'cnn_learning_curve.png')
    plt.savefig(curve_path)
    print(f"学习曲线已保存至: {curve_path}")

    # --- 5. 【新增】生成并保存测试集评估报告 ---
    print("\n--- 正在生成测试集评估报告 ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 将 Tensor 转回 CPU 并转为 list，用于 sklearn 计算
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 生成 Scikit-learn 分类报告
    report_str = classification_report(all_labels, all_preds, digits=4)
    print(report_str)

    # 保存到 txt 文件
    report_path = os.path.join(RESULT_SAVE_DIR, 'cnn_report.txt')
    with open(report_path, 'w') as f:
        f.write("CNN Model Evaluation Report\n")
        f.write("===========================\n")
        f.write(f"Training Epochs: {EPOCHS}\n")
        f.write(f"Training Duration: {duration:.2f} seconds\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%\n\n")
        f.write("Test Set Classification Report:\n")
        f.write(report_str)

    print(f"详细评估报告已保存至: {report_path}")


if __name__ == '__main__':
    train_cnn()