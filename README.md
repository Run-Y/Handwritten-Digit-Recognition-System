# Handwritten Digit Recognition System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![PyTorch](https://img.shields.io/badge/Model-PyTorch%20%26%20Sklearn-orange)

## 📖 Introduction
This is a comprehensive Handwritten Digit Recognition System developed in Python. The project features a Graphical User Interface (GUI) built with PyQt5, allowing users to draw digits on a canvas in real-time or upload images for recognition.

To compare the performance of different algorithms, the system integrates two primary machine learning models:
1. CNN (Convolutional Neural Network): A deep learning model implemented via PyTorch for high-precision recognition.
2. SVM (Support Vector Machine): A traditional machine learning model implemented via Scikit-learn.

The project also includes a Performance Evaluation Module capable of displaying Confusion Matrices, Learning Curves, and detailed Classification Reports (Precision, Recall, and F1-Score).

## ✨ Key Features
* Interactive Canvas: Draw directly with a mouse for real-time recognition.
* Dual Model Support: Toggle between CNN (High Accuracy) and SVM (Traditional) modes.
* Preprocessing Visualization: Real-time display of binarized and scaled input images for debugging.
* Evaluation & Visualization: A dedicated window for:
    - Training Learning Curves (Loss/Accuracy).
    - Confusion Matrix comparisons.
    - Detailed Metrics (Precision, Recall, F1-Score).

## 🛠️ Technology Stack
* GUI: PyQt5
* Deep Learning: PyTorch
* Machine Learning: Scikit-learn
* Data Processing: NumPy, OpenCV (cv2)
* Visualization: Matplotlib, Seaborn

## 🚀 Installation & Setup

1. Clone the Repository:
```bash
git clone https://github.com/Run-Y/Handwritten-Digit-Recognition-System.git
cd Handwritten-Digit-Recognition-System
```
2. Install Dependencies:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ⚡ Usage Workflow

Step 1: Data Preparation
Download and process the MNIST dataset into NumPy format.
```bash
python download_mnist_to_folders.py
cd scripts
python processData.py
```

Step 2: Model Training
Train models and save them to the models/ directory.
```bash
python trainSVM.py
python trainCNN.py
```

Step 3: Generate Evaluation Charts
Execute this to enable the "View Evaluation" feature in the GUI.
```bash
python generateCharts.py
```

Step 4: Run Application
Launch the GUI from the root directory.
```bash
python main.py
```

## 📂 Project Structure

```Plaintext
Handwritten-Digit-Recognition-System/
├── GUI/
│   ├── guiCanvas.py        # Canvas logic and UI components
├── models/                 # Trained .pth and .pkl files
├── processedData/          # .npz dataset files
├── rawDataset/             # Raw MNIST images
├── results/                # Evaluation charts and reports
├── scripts/
│   ├── generateCharts.py   # Core evaluation script
│   ├── modelDefinition.py  # Definition of CNN model
│   ├── processData.py      # Data processing script
│   ├── trainCNN.py         # CNN training
│   ├── trainSVM.py         # SVM training
│   └── utils.py            # Utility functions
├── main.py                 # Main entry point
├── requirements.txt        # Dependency list
└── README.md               # 项目说明文档
```