import numpy as np
import os
import pickle
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 配置路径
DATA_PATH = '../processedData/mnist_custom_data.npz'
MODEL_SAVE_DIR = '../models/traditional'
RESULT_SAVE_DIR = '../results'  # 【新增】结果保存路径

# 确保文件夹存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)  # 【新增】


def train_svm():
    # ... (前面的加载数据代码保持不变) ...
    print("--- 正在加载数据 ---")
    if not os.path.exists(DATA_PATH):
        print(f"错误：找不到数据文件 {DATA_PATH}，请先运行 step1_process_data.py")
        return

    data = np.load(DATA_PATH)
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']

    # 数据展平
    print(f"原始训练数据形状: {X_train.shape}")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # ... (中间的模型定义保持不变) ...
    print("\n--- 开始训练 SVM (这可能需要几分钟) ---")
    svm = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=42)

    start_time = time.time()
    svm.fit(X_train_flat, y_train)
    end_time = time.time()
    print(f"训练完成！耗时: {end_time - start_time:.2f} 秒")

    # --- 3. 评估模型并保存结果 (修改部分) ---
    print("\n--- 正在评估模型 ---")
    y_pred = svm.predict(X_test_flat)

    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {acc * 100:.2f}%")

    # 生成分类报告字符串
    report_str = classification_report(y_test, y_pred)

    # 打印到控制台
    print("\n分类报告:")
    print(report_str)

    # 【新增】保存分类报告到 txt 文件
    report_path = os.path.join(RESULT_SAVE_DIR, 'svm_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"SVM Model Evaluation\n")
        f.write(f"====================\n")
        f.write(f"Training Duration: {end_time - start_time:.2f} seconds\n")
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"评估报告已保存至: {report_path}")

    # 4. 保存模型
    save_path = os.path.join(MODEL_SAVE_DIR, 'svm_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(svm, f)
    print(f"模型已保存至: {save_path}")


if __name__ == '__main__':
    train_svm()