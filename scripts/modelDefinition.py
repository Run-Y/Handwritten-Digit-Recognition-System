import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # --- 这里的代码展示了你自己设计的架构 ---

        # 第一层卷积块
        # 输入: (1, 28, 28)
        # Conv2d: 输入通道1, 输出32, 核大小3, padding=1 (保持尺寸) -> (32, 28, 28)
        # MaxPool2d: 2x2 -> (32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积块
        # 输入: (32, 14, 14)
        # Conv2d: 32 -> 64, 核大小3, padding=1 -> (64, 14, 14)
        # MaxPool2d: 2x2 -> (64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层 (Fully Connected Layers)
        # 展平后维度计算: 64个通道 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 隐藏层
        self.dropout = nn.Dropout(p=0.5)  # Dropout 防止过拟合
        self.fc2 = nn.Linear(128, 10)  # 输出层 (10个数字分类)

    def forward(self, x):
        # 定义前向传播路径 (Forward Process)

        # 1. 卷积 -> 激活(ReLU) -> 池化
        x = self.pool1(F.relu(self.conv1(x)))

        # 2. 卷积 -> 激活(ReLU) -> 池化
        x = self.pool2(F.relu(self.conv2(x)))

        # 3. 展平 (Flatten)
        # x.size(0) 是 batch_size
        x = x.view(x.size(0), -1)

        # 4. 全连接 -> 激活 -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 5. 输出层 (不加 softmax，因为 CrossEntropyLoss 会自动加)
        x = self.fc2(x)

        return x