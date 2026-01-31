import torch
import torch.nn as nn
import random
import numpy as np
from baseline_sample_cnn_model import SampleCNN  # 仅用于参考，这里我们重新定义一个


class MixStyle(nn.Module):
    """
    MixStyle Layer for 1D signals (Batch, Channel, Length)
    Reference: "Domain Generalization with MixStyle" (ICLR 2021)
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p  # 激活概率 (通常 0.5)
        self.alpha = alpha  # Beta 分布参数 (通常 0.1)
        self.eps = eps  # 数值稳定性 epsilon

    def forward(self, x):
        # 仅在训练模式且满足概率 p 时启用
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)

        # 1. 计算当前 Batch 的实例统计量 (Mean & Std)
        # x shape: [B, C, L] -> 沿 L 维度计算
        mu = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        var = x.var(dim=2, keepdim=True)
        sig = (var + self.eps).sqrt()  # [B, C, 1]

        # 2. 归一化输入 x
        x_normed = (x - mu) / sig

        # 3. 生成随机打乱的索引
        perm = torch.randperm(B).to(x.device)

        # 4. 获取打乱后的统计量 (即 "Reference Style")
        mu_perm = mu[perm]
        sig_perm = sig[perm]

        # 5. 生成混合系数 lambda (从 Beta 分布采样)
        # 注意：lam 是标量或 [B, 1, 1] 均可，这里为了简单对整个 Batch 用同一个分布采样，但通常逐样本采样效果更好
        # 为了与论文一致，我们从 Beta(alpha, alpha) 采样一个标量，用于混合整个 Batch
        lmda = np.random.beta(self.alpha, self.alpha)
        # 或者逐样本混合 (更强的随机性):
        # lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)
        # 这里使用标量混合即可:
        lmda = torch.tensor(lmda).to(x.device)

        # 6. 混合统计量
        mu_mix = lmda * mu + (1 - lmda) * mu_perm
        sig_mix = lmda * sig + (1 - lmda) * sig_perm

        # 7. 将混合后的风格应用回归一化的特征
        return x_normed * sig_mix + mu_mix


class MixStyleCNN(nn.Module):
    """
    带有 MixStyle 的 CNN 模型。
    通常 MixStyle 插入在浅层网络（如第 1、2 个 Block 后）。
    """

    def __init__(self, num_classes=4, mixstyle_p=0.5, mixstyle_alpha=0.1):
        super(MixStyleCNN, self).__init__()

        # 定义 MixStyle 层
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        # Block 3 (后续层通常不加 MixStyle，以免破坏语义信息)
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Block 1 -> MixStyle
        x = self.block1(x)
        x = self.mixstyle(x)

        # Block 2 -> MixStyle
        x = self.block2(x)
        x = self.mixstyle(x)

        # Remaining Blocks
        x = self.block3(x)
        x = self.block4(x)

        # Classification
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits, x