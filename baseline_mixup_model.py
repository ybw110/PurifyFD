import torch
import torch.nn as nn
import numpy as np
from sample_cnn_model import SampleCNN as BaseCNN # 复用你现有的 Baseline

# 继承现有的 CNN，保持结构一致
class MixupCNN(BaseCNN):
    def __init__(self, num_classes=4):
        super(MixupCNN, self).__init__(num_classes)

# ================= Mixup 核心工具函数 =================

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''
    返回混合后的输入 (mixed_x) 以及两组标签 (y_a, y_b) 和混合比例 (lam)
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    Mixup 的损失函数：按照 lambda 比例混合两个标签的损失
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)