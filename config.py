import torch
import os


class Config:
    # ================= 1. 路径配置 =================
    RAW_DATA_ROOT_A = "/data2/ybw25/轴承项目/1楼数据集"
    RAW_DATA_ROOT_B = "/data2/ybw25/轴承项目/6楼数据集"

    # ================= 2. 预处理数据路径 (Processed Data) =================
    # [新增] 这里存放转换后的 .npz 文件，读取速度极快
    PROCESSED_DATA_ROOT = "/data2/ybw25/轴承DG/Processed_Data_Numpy"

    # ================= 3. 实验结果保存路径 =================
    # 这里只定义根目录，具体子目录由训练脚本动态生成
    RESULT_ROOT = "./results"

    # ================= 4. 数据参数 =================
    SEQUENCE_LENGTH = 1024  # 输入信号长度
    INPUT_CHANNELS = 1  # 输入通道数

    # ============== OOD / Open-set 配置 ==============
    all_num_classes = 4  # 物理总类数：0/1/2/3

    # 例：把“中等(2)”当 unknown（你之前 CWRU/JNU 默认也类似）
    known_classes = [0, 1, 3]
    unknown_classes = [2]

    # 训练输出维度 = known 类数
    num_classes = len(known_classes)


    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ================= 5. 训练超参数 =================
    latent_domain_num = 3           # 潜在域数量 (Diversify算法聚类数)
    batch_size = 64

    # --- 训练循环结构 ---
    max_rounds = 10  # 总轮次 (类似 Global Epochs)
    local_epochs = 10  # 每个阶段内部的微调轮次 (Local Epochs)
    epochs = 20  # 这是普通cnn使用的

    # 学习率配置
    lr = 1e-4               # 基础学习率
    lr_decay1 = 0.1          # 特征提取器的学习率衰减
    lr_decay2 = 1.0          # 分类器的学习率衰减
    weight_decay = 1e-4     # L2正则化系数


    # 模型参数
    bottleneck = 256         # bottleneck层维度
    dis_hidden = 256         # 判别器隐藏层维度

    beta1 = 0.9
    # alpha改成alpha2，方便写论文
    alpha1 = 0.3  # 辅助域适应的梯度反转层系数
    alpha = 0.1  # 梯度反转层系数
    lam = 0.5                # 熵损失权重

    layer = "bn"             # bottleneck层类型
    classifier = "linear"    # 分类器类型
    algorithm = 'diversify'

    # ================= 6. 系统配置 =================
    seed = 42
    # seed = 28

    factor = 0.1  # 新的学习率是旧的学习率乘以该因子。
    patience = 3  # 在监测量多少个周期没有改善后调整学习率。
    num_workers = 4
    dropout = 0.3
    device = "cuda:0"  # 默认使用cuda:3



