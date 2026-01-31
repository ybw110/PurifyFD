import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 配置区域 =================
# 适配 CWRU 轴承数据集 (长度 1024)
var_size = {
    'in_size': 1,  # 单通道振动信号
    'sequence_length': 1024,  # 总长度
    'hidden_size': 64,  # 特征维度
    'num_timesteps': 32,  # 时间步数
    'segment_size': 32  # 分段大小
}


class CNN(nn.Module):
    """局部特征提取器"""

    def __init__(self, in_channel=1, out_channel=64, dropout_rate=0.2):
        super(CNN, self).__init__()

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SelfAttention(nn.Module):
    """缩放点积注意力机制"""

    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, Q, K, V):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class CWRU_Baseline_Network(nn.Module):
    def __init__(self, num_classes=4):
        super(CWRU_Baseline_Network, self).__init__()

        self.input_size = var_size['in_size']
        self.hidden_size = var_size['hidden_size']
        self.num_timesteps = var_size['num_timesteps']
        self.segment_size = var_size['segment_size']

        # === 特征提取 Backbone (与 Diversify 保持一致) ===
        self.fcn = CNN(in_channel=self.input_size, out_channel=self.hidden_size)

        self.lstm = nn.LSTM(
            input_size=self.segment_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.conv1x3 = nn.Conv1d(self.input_size, self.hidden_size, kernel_size=3, padding=1)
        self.attention = SelfAttention()

        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # === 分类器 (Diversify把这部分拆出去了，这里Baseline需要加回来) ===
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # 1. 维度调整
        batch_size = x.size(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        expected_length = self.num_timesteps * self.segment_size
        if x.shape[2] != expected_length:
            x = F.interpolate(x, size=expected_length, mode='linear', align_corners=False)

        # 2. CNN 分支
        local_features = self.fcn(x)  # [B, 64, 32]
        local_features = local_features.transpose(1, 2)  # [B, 32, 64]

        if local_features.size(1) != self.num_timesteps:
            local_features = local_features.transpose(1, 2)
            local_features = F.interpolate(local_features, size=self.num_timesteps, mode='linear')
            local_features = local_features.transpose(1, 2)

        # 3. LSTM 分支
        x_reshaped = x.view(batch_size, self.num_timesteps, self.segment_size)
        lstm_out, _ = self.lstm(x_reshaped)  # [B, 32, 64]

        # 4. Shortcut
        conv_out = self.conv1x3(x)
        conv_out = F.interpolate(conv_out, size=self.num_timesteps, mode='linear')
        conv_out = conv_out.transpose(1, 2)

        # 5. Attention
        attn_output, _ = self.attention(lstm_out, conv_out, conv_out)
        global_features = attn_output + lstm_out

        # 6. 融合
        combined = torch.cat([local_features, global_features], dim=-1)
        output = self.fusion(combined)

        # 7. 平均池化得到特征向量
        features = torch.mean(output, dim=1)  # [B, 64]

        # 8. 分类 (这是Baseline独有的步骤，直接输出logits)
        logits = self.classifier(features)

        return logits, features  # 返回 logits 用于Loss，features 用于t-SNE