# 手写数字识别系统 (Handwritten Digit Recognition System)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![PyTorch](https://img.shields.io/badge/Model-PyTorch%20%26%20Sklearn-orange)

## 📖 项目简介 (Introduction)

这是一个基于 Python 开发的完整**手写数字识别系统**。项目包含一个使用 **PyQt5** 构建的图形用户界面 (GUI)，允许用户在画板上实时手写数字或上传图片进行识别。

为了对比不同算法的性能，系统集成了两种主流的机器学习模型：
1.  **CNN (卷积神经网络)**：基于 **PyTorch** 实现的深度学习模型，精度更高。
2.  **SVM (支持向量机)**：基于 **Scikit-learn** 实现的传统机器学习模型。

此外，项目还包含一个完整的**性能评估模块**，可展示混淆矩阵 (Confusion Matrix)、学习曲线 (Learning Curve) 以及详细的分类报告 (P/R/F1-Score)。

## ✨ 主要功能 (Key Features)

* **交互式画板**：支持鼠标在 GUI 上直接书写数字，体验实时识别。
* **双模型支持**：可随时在 **CNN** (高精度) 和 **SVM** (传统方法) 之间切换。
* **预处理可视化**：实时显示输入图像经过二值化、缩放后的效果，方便调试。
* **评估与可视化**：提供独立的评估窗口，展示：
    * 训练过程的学习曲线 (Loss/Accuracy)。
    * 两个模型的混淆矩阵对比。
    * 详细的分类指标报告 (精确率、召回率、F1分数)。

## 🛠️ 技术栈 (Technology Stack)

* **图形界面**: PyQt5
* **深度学习**: PyTorch (构建 CNN)
* **机器学习**: Scikit-learn (构建 SVM)
* **数据处理**: NumPy, OpenCV (cv2)
* **图表绘制**: Matplotlib, Seaborn

## 🚀 安装步骤 (Installation)

### 1. 克隆仓库
```bash
git clone [https://github.com/Run-Y/Handwritten-Digit-Recognition-System.git](https://github.com/Run-Y/Handwritten-Digit-Recognition-System.git)
cd Handwritten-Digit-Recognition-System
```
### 2. 创建虚拟环境（推荐）
为了避免依赖冲突，建议创建 Python 虚拟环境。
```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```
### 3. 安装依赖库
```bash
pip install -r requirements.txt
```
## ⚡ 使用流程 (Usage Workflow)
>为了确保系统正常运行，请务必按照以下顺序执行脚本：

**第一步：数据准备 (Data Preparation)**

下载 MNIST 数据集并将其处理为 NumPy 格式。
```bash
# 1. 下载原始图片数据
python scripts/download_mnist_to_folders.py

# 2. 处理并保存为 .npz 文件
python scripts/processData.py
```
**第二步：模型训练 (Model Training)**

分别训练 CNN 和 SVM 模型。训练好的模型将保存在 models/ 目录下。

```bash
# 训练 SVM 模型
python scripts/trainSVM.py

# 训练 CNN 模型 (默认 10 轮)
python scripts/trainCNN.py
```

**第三步：生成评估图表 (⚠️ 重要)**

运行此脚本以生成混淆矩阵图片和评估报告。 
>注意： 如果跳过此步，GUI 中的“查看评估”功能将无法显示图片。
```bash
python scripts/generateCharts.py
```

**第四步：启动系统 (Run Application)**

启动图形化界面。

```bash
python main.py
```

## 📂 项目结构 (Project Structure)

```Plaintext
Handwritten-Digit-Recognition-System/
├── GUI/
│   ├── guiCanvas.py        # 画板组件逻辑
├── models/                 # 训练好的模型文件 (.pth / .pkl)
├── processedData/          # 处理后的数据集 (.npz)
├── rawDataset/             # 原始 MNIST 图片数据
├── results/                # 生成的评估图表和报告
├── scripts/
│   ├── generateCharts.py   # [核心] 生成评估图表与报告脚本
│   ├── modelDefinition.py  # CNN 模型网络结构定义
│   ├── processData.py      # 数据预处理脚本
│   ├── trainCNN.py         # CNN 训练脚本
│   ├── trainSVM.py         # SVM 训练脚本
│   └── utils.py            # 通用工具函数
├── main.py                 # 程序入口文件
├── requirements.txt        # 项目依赖列表
└── README.md               # 项目说明文档
```