import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. 域生成模块 (Domain Generation Module D)
# 论文中 D 输出生成的样本 x+。为了保持 x+ 与 x_s 形状一致 (1, 1024),
# 这里设计为 U-Net 风格的 1D 卷积自编码器。
# ==========================================
class DomainGenerator(nn.Module):
    def __init__(self, input_channel=1):
        super(DomainGenerator, self).__init__()

        # Encoder: 下采样
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)  # -> 512
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)  # -> 256
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)  # -> 128
        )

        # Decoder: 上采样恢复形状
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()  # -> 256
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()  # -> 512
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(16, input_channel, kernel_size=3, padding=1),
            nn.Tanh()  # 归一化到 [-1, 1] 之间，假设输入已标准化
        )  # -> 1024

    def forward(self, x):
        # x: [Batch, 1, 1024]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        z = self.enc3(e2)

        d1 = self.dec1(z)
        d2 = self.dec2(d1 + e2)  # Skip connection
        out = self.dec3(d2 + e1)  # Skip connection

        # 残差连接：生成的是扰动，使得生成的样本保留原始内容
        return x + out


# ==========================================
# 2. 任务诊断模块 (Task Diagnosis Module T)
# 包含特征提取器 (G) 和 分类器 (C)
# ==========================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Input: [Batch, 1, 1024]
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=64, stride=1, padding=32),  # 大卷积核提取整体特征
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 512

            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 256

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 128

            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> 64
        )
        # Output Flatten Dim = 64 * 64 = 4096
        self.fc_proj = nn.Linear(64 * 64, 256)  # 投影到 latent space z

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.fc_proj(x)
        return z  # [Batch, 256]


class Classifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=4):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, z):
        return self.fc(z)


# ==========================================
# 3. 互信息估计器 (CLUB Estimator q_theta)
# 用于估计 MI 的上界 p(z+|zs)
# ==========================================
class CLUB(nn.Module):
    def __init__(self, feature_dim=256, hidden_size=128):
        super(CLUB, self).__init__()
        # q(z+|zs) 建模为高斯分布 N(mu(zs), var(zs))
        self.p_mu = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim),
            nn.Tanh()  # 限制 logvar 范围防止数值不稳定
        )

    def get_mu_logvar(self, z_s):
        mu = self.p_mu(z_s)
        logvar = self.p_logvar(z_s)
        return mu, logvar

    def loglikeli(self, z_s, z_plus):
        # 计算 log q(z+|zs)
        mu, logvar = self.get_mu_logvar(z_s)
        # log_pdf = -0.5 * (log(2pi) + logvar + (z+ - mu)^2 / exp(logvar))
        return (-(mu - z_plus) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, z_s, z_plus):
        # 训练 q 网络本身的 Loss: 最大化似然 -> 最小化负对数似然
        return - self.loglikeli(z_s, z_plus)

    def mi_est(self, z_s, z_plus):
        # 估计互信息 I(zs; z+)
        mu, logvar = self.get_mu_logvar(z_s)

        sample_size = z_s.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - z_plus) ** 2 / logvar.exp()
        negative = - (mu - z_plus[random_index]) ** 2 / logvar.exp()

        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # Divide by 2 is technically correct for Gaussian CLUB but paper formula uses direct log diff
        return upper_bound / 2.0

    # ==========================================


# 4. 辅助 Loss: CMMD & SupCon
# ==========================================
class CMMDLoss(nn.Module):
    """Conditional Maximum Mean Discrepancy (语义一致性)"""

    def __init__(self):
        super(CMMDLoss, self).__init__()

    def forward(self, z_s, z_plus, labels):
        loss = 0.0
        classes = torch.unique(labels)
        for c in classes:
            idx = (labels == c)
            if idx.sum() > 0:
                # 计算同一类别下的中心距离
                mean_s = z_s[idx].mean(0)
                mean_p = z_plus[idx].mean(0)
                loss += ((mean_s - mean_p) ** 2).sum()
        return loss / len(classes)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (InfoNCE based)"""

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [Batch, Dim] (normalize first)
        features = F.normalize(features, dim=1)

        # Similarity matrix
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # Mask calculation
        batch_size = features.shape[0]
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        # Exp logits
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Mean log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = - mean_log_prob_pos.mean()
        return loss