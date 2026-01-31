import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleCNN(nn.Module):
    """
    一个基础的 1D-CNN，用于作为对比实验的 Baseline。
    结构：4层卷积 + 全局平均池化 + 分类器
    """

    def __init__(self, num_classes=3):
        super(SampleCNN, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            # 第一层使用大卷积核，模仿 WDCNN 提取低频特征
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),  # 1024 -> 128
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),  # 128 -> 64

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),  # 64 -> 32

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),  # 32 -> 16

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 16 -> 1 (全局池化)
        )

        # 分类层
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # 输入检查 [B, 1024] -> [B, 1, 1024]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # 提取特征
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)  # [B, 128]

        # 分类
        logits = self.classifier(feat)
        return logits, feat