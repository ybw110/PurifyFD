import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================
# 1. 特征提取器 (Feature Extractor F)
# 对应论文 Table 1，适配 CWRU/JNU (1, 1024)
# ==========================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Input: [Batch, 1, 1024]
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)  # -> [64, 512]
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)  # -> [128, 256]
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)  # -> [128, 128]
        )
        # Flatten size = 128 * 128 = 16384 (根据 pooling 次数调整)
        # 你的 Config sequence_length=1024.
        # Layer1: 1024/2=512; Layer2: 512/2=256; Layer3: 256/2=128.
        # Output shape: [Batch, 128, 128]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x  # Return feature map for AdaIN [Batch, Channel, Length]


# ==========================================
# 2. 特征变换器 (Feature Transformer H - AdaIN)
# 论文核心：注入噪声模拟未知域
# ==========================================
class AdaIN(nn.Module):
    def __init__(self, num_channels=128):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm1d(num_channels, affine=False)

        # 论文 Eq: h1(n1) 和 h2(n2)
        # 用于生成 Gamma (缩放) 和 Beta (平移)
        # 输入噪声维度设为 16 (可调整)，输出必须等于特征通道数
        self.fc_gamma = nn.Sequential(
            nn.Linear(1, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(1, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, x):
        # x shape: [Batch, Channel, Length]
        batch_size = x.size(0)

        # 生成噪声 (论文 Section 3.1.3)
        # n1 ~ U(0.05, 1.95) for Gamma (scale)
        n1 = (0.05 + (1.95 - 0.05) * torch.rand(batch_size, 1)).to(x.device)

        # n2 ~ N(0, 1) for Beta (bias)
        n2 = torch.randn(batch_size, 1).to(x.device)

        # 计算 Gamma 和 Beta
        gamma = self.fc_gamma(n1).unsqueeze(2)  # [Batch, Channel, 1]
        beta = self.fc_beta(n2).unsqueeze(2)  # [Batch, Channel, 1]

        # AdaIN 变换
        out = self.norm(x) * gamma + beta
        return out


# ==========================================
# 3. 域不变特征提取器 (Domain Invariant G)
# ==========================================
class DomainInvariantExtractor(nn.Module):
    def __init__(self, input_dim=128 * 128, hidden_dim=256):
        super(DomainInvariantExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, hidden_dim),  # Domain Invariant Features (g)
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 4. 分类器 (Classifier C)
# ==========================================
class Classifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=4):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 5. 判别器 (Discriminator D)
# 接收 (Feature x Prediction) 的外积 (CDAN 思想)
# ==========================================
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 6. DACN 整体模型封装
# ==========================================
class DACN(nn.Module):
    def __init__(self, num_classes=4):
        super(DACN, self).__init__()

        self.F = FeatureExtractor()
        self.H = AdaIN(num_channels=128)

        # 计算 Flatten 后的维度: 128 (Channel) * 128 (Length after pooling)
        flat_dim = 128 * 128
        self.G = DomainInvariantExtractor(input_dim=flat_dim, hidden_dim=256)
        self.C = Classifier(input_dim=256, num_classes=num_classes)

        # Discriminator 输入维度 = FeatureDim(256) * ClassNum(4) (CDAN外积)
        self.D = Discriminator(input_dim=256 * num_classes)

    def forward_pretrain(self, x):
        """预训练阶段仅使用 F -> G -> C"""
        f = self.F(x)
        g = self.G(f)
        logits = self.C(g)
        return logits, g

    def forward_train(self, x):
        """正式训练阶段：生成伪样本并返回所有结果"""
        # 1. 真实样本特征
        f_real = self.F(x)  # [B, 128, 128]

        # 2. 生成伪样本特征 (Feature Transformer)
        f_fake = self.H(f_real)  # [B, 128, 128]

        # 3. 拼接 (Batch维度翻倍)
        f_concat = torch.cat([f_real, f_fake], dim=0)  # [2B, ...]

        # 4. 提取域不变特征
        g_concat = self.G(f_concat)  # [2B, 256]

        # 5. 分类预测
        logits_concat = self.C(g_concat)  # [2B, Class]

        return g_concat, logits_concat  # 返回拼接后的特征和预测，供Loss计算


# ==========================================
# 7. Supervised Contrastive Loss (SupCon)
# 直接复用你提供的 SupConLoss，稍微精简
# ==========================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [Batch, Dim] -> Normalize
        features = F.normalize(features, dim=1)

        # Similarity Matrix
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # Mask for self-contrast
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)

        # Mask for same class (positive pairs)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(features.device)
        mask_pos = mask_pos - torch.eye(batch_size).to(features.device)  # remove self

        # Numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits)

        # Denominator: sum of all other exp_logits (excluding self)
        # 这里简化处理，严谨的SupCon需要排除自身
        exp_logits = exp_logits * (~mask).float()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Mean log-likelihood for positive pairs
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-6)

        loss = - mean_log_prob_pos.mean()
        return loss