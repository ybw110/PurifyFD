# config.py

class Config:
    # 正常数据文件夹 (Normal Baseline Data)
    data_root_normal = '/data2/ybw25/轴承DG/CWRU/参考文献/Normal Baseline Data'
    # 故障数据文件夹 (12k DriveEndBearingFault Data)
    data_root_fault = '/data2/ybw25/轴承DG/CWRU/参考文献/12k DriveEndBearingFault Data'

    # ================= 2. 信号处理参数 =================
    sequence_length = 1024          # 输入信号长度 (适配 CWRU)
    stride = 512                    # 滑动窗口步长 (512代表50%重叠)

    # ================= 3. 训练与模型参数 =================
    num_classes = 4                 # 4分类: Normal, Inner, Ball, Outer
    latent_domain_num = 3           # 潜在域数量 (Diversify算法聚类数)
    batch_size = 64

    # --- 训练循环结构 ---
    max_rounds = 10                           # 总轮次 (类似 Global Epochs)
    local_epochs = 10                           # 每个阶段内部的微调轮次 (Local Epochs)
    # 注: 总迭代次数 approx = max_rounds * local_epochs * 3 (三个阶段)
    epochs = 20    #这是普通cnn使用的


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



    # 其他配置
    seed = 42    #初始
    # seed = 28      #第二次
    factor = 0.1  # 新的学习率是旧的学习率乘以该因子。
    patience = 3  # 在监测量多少个周期没有改善后调整学习率。
    num_workers = 4
    dropout = 0.3
    split_ratio = [0.7, 0.15, 0.15]  # Train / Val / Test 比例
    device = "cuda:0"  # 默认使用cuda:3