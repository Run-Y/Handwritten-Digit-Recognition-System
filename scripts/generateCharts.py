import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.modelDefinition import CustomCNN

DATA_PATH = '../processedData/mnist_custom_data.npz'
RESULTS_DIR = '../results/chart_and_report'
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved:{filename}")


def save_text_report(y_true, y_pred, title, filename):
    report = classification_report(y_true, y_pred)
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        f.write(f"--- {title} ---\n\n")
        f.write(report)
    print(f"Saved: {filename}")


def main():
    data = np.load(DATA_PATH)
    x_test = data['x_test']
    y_test = data['y_test']

    print(f"Test set size:{len(y_test)}")

    print("\nUnder evaluation SVM...")
    try:
        with open('../models/traditional/svm_model.pkl', 'rb') as f:
            svm = pickle.load(f)

        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        svm_pred = svm.predict(x_test_flat)

        plot_confusion_matrix(y_test, svm_pred, "SVM Confusion Matrix", "svm_conf_matrix.png")
        save_text_report(y_test, svm_pred, "SVM Report", "svm_report.txt")

    except Exception as e:
        print(f" {e}")

    print("\nCNN is being evaluated...")
    try:
        model = CustomCNN().to(DEVICE)
        model.load_state_dict(
            torch.load('../models/deep_learning/cnn_model.pth', map_location=DEVICE, weights_only=True))
        model.eval()

        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

        cnn_preds = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(x_tensor), batch_size):
                batch_x = x_tensor[i:i + batch_size]
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                cnn_preds.extend(predicted.cpu().numpy())

        cnn_preds = np.array(cnn_preds)

        plot_confusion_matrix(y_test, cnn_preds, "CNN Confusion Matrix", "cnn_conf_matrix.png")
        save_text_report(y_test, cnn_preds, "CNN Report", "cnn_report.txt")

    except Exception as e:
        print(f"{e}")


if __name__ == '__main__':
    main()