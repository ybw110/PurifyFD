import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 配置区域 =================
# 适配 CWRU 轴承数据集 (长度 1024)
var_size = {
    'in_size': 1,  # 单通道振动信号
    'sequence_length': 1024,  # 总长度
    'hidden_size': 256,  # 特征维度
    'num_timesteps': 32,  # 时间步数 (LSTM序列长度)
    'segment_size': 32  # 每个时间步的输入长度 (32*32=1024)
}


# ===========================================

class CNN(nn.Module):

    def __init__(self, in_channel=1, out_channel=64, dropout_rate=0.2):
        super(CNN, self).__init__()

        # Layer 1: 大卷积核提取原始信号特征 (类似 WDCNN)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=2, padding=32),  # 下采样 /2
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))                                      # 下采样 /2 -> 总 /4

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))                                      # 下采样 /2 -> 总 /8

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))                                      # 下采样 /2 -> 总 /16

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))                                      # 下采样 /2 -> 总 /32



    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
                                                                                        # Output shape: [Batch, 64, 32] (特征维度, 时间长度)
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


class CWRUNetwork(nn.Module):
    def __init__(self):
        super(CWRUNetwork, self).__init__()

        self.input_size = var_size['in_size']
        self.hidden_size = var_size['hidden_size']
        self.num_timesteps = var_size['num_timesteps']
        self.segment_size = var_size['segment_size']

        # === 1. CNN 分支 (局部特征) ===
        self.fcn = CNN(in_channel=self.input_size, out_channel=self.hidden_size)

        # === 2. LSTM 分支 (全局特征) ===
        self.lstm = nn.LSTM(
            input_size=self.segment_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # === 3. Shortcut 卷积分支 ===
        self.conv1x3 = nn.Conv1d(self.input_size, self.hidden_size, kernel_size=3, padding=1)

        # === 4. 注意力与融合 ===
        self.attention = SelfAttention()

        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Diversify 算法必须属性：告诉外部这个网络的输出特征维度是多少
        self.in_features = self.hidden_size

    def forward(self, x):
        """
        Input: [Batch, 1, 1024] or [Batch, 1024]
        Output: [Batch, 64]
        """
        batch_size = x.size(0)

        # 1. 维度检查与调整
        if len(x.shape) == 2:
            x = x.unsqueeze(1)                                               # [Batch, 1024] -> [Batch, 1, 1024]

        # 2. 强制插值，确保输入严格为 1024 (防止数据处理误差)
        expected_length = self.num_timesteps * self.segment_size  # 1024
        if x.shape[2] != expected_length:
            x = F.interpolate(x, size=expected_length, mode='linear', align_corners=False)

        # === 分支 A: CNN 局部特征 ===
        # CNN Output: [Batch, 64, 32]
        local_features = self.fcn(x)

        # 转置以匹配 LSTM 维度: [Batch, 32, 64] (Batch, Time, Feat)
        local_features = local_features.transpose(1, 2)

        # 双重保险：强制对齐时间步长 (防止CNN下采样有些许偏差)
        if local_features.size(1) != self.num_timesteps:
            local_features = local_features.transpose(1, 2)                         # [B, 64, T]
            local_features = F.interpolate(local_features, size=self.num_timesteps, mode='linear')
            local_features = local_features.transpose(1, 2)                         # [B, 32, 64]

        # === 分支 B: LSTM 全局特征 ===
        # View: [Batch, 32, 32]
        x_reshaped = x.view(batch_size, self.num_timesteps, self.segment_size)
        lstm_out, _ = self.lstm(x_reshaped)                                         # [Batch, 32, 64]

        # === 分支 C: Shortcut 卷积 ===
        conv_out = self.conv1x3(x)                                                  # [Batch, 64, 1024]
        conv_out = F.interpolate(conv_out, size=self.num_timesteps, mode='linear')
        conv_out = conv_out.transpose(1, 2)                                         # [Batch, 32, 64]

        # === 注意力交互 ===
        # Q=LSTM(全局), K=Conv(浅层), V=Conv(浅层)
        attn_output, _ = self.attention(lstm_out, conv_out, conv_out)

        # 残差连接
        global_features = attn_output + lstm_out                                    # [Batch, 32, 64]

        # === 最终融合 ===
        # 拼接 CNN特征 和 LSTM+Attention特征
        combined = torch.cat([local_features, global_features], dim=-1)       # [Batch, 32, 128]
        output = self.fusion(combined)                                              # [Batch, 32, 64]

        # 平均池化得到最终特征向量
        output = torch.mean(output, dim=1)                                          # [Batch, 64]

        return output