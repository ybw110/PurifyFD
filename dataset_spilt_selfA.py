import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config

# 实例化配置
cfg = Config()

# ================= 1. SelfData 配置区域 =================

ALL_SPEEDS = ['1200', '1800']
DOMAIN_MAP = {'1200': 0, '1800': 1}
CLASS_MAP = {'正常': 0, '轻微': 1, '中等': 2, '严重': 3}

# ================= 2. Dataset 类 (标准格式) =================

class SelfADataset(Dataset):
    def __init__(self, segments, labels, true_domains):
        self.segments = segments
        self.labels = labels
        self.true_domains = true_domains
        self.indices = np.arange(len(segments))
        self.pdlabels = np.zeros(len(segments), dtype=np.int64)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # 1. 获取数据并调整维度
        # 原始: (1024,) -> 目标: (1, 1024) 适配 1D-CNN
        segment = self.segments[idx]
        segment = torch.FloatTensor(segment).unsqueeze(0)

        # 2. 标签封装
        label = torch.LongTensor([self.labels[idx]])[0]  # Class (0-3)
        true_domain = torch.LongTensor([self.true_domains[idx]])[0]  # Domain (0-1)
        pdlabel = torch.LongTensor([self.pdlabels[idx]])[0]  # Latent Domain
        index = torch.LongTensor([self.indices[idx]])[0]  # Sample Index

        # 3. 返回 6 元组 (适配 Diversify 算法接口)
        # 格式: (Data, Class, TrueDomain, Dummy, PdLabel, Index)
        return segment, label, true_domain, torch.zeros(1), pdlabel, index

    def set_labels_by_index(self, pred_labels, indices, label_type='pdlabel'):
        """供算法反向更新潜在域标签"""
        pred_labels = np.asarray(pred_labels)
        indices = np.asarray(indices)
        if label_type == 'pdlabel':
            for i, idx in enumerate(indices):
                if 0 <= idx < len(self.pdlabels):
                    self.pdlabels[int(idx)] = int(pred_labels[i])


def _standardize_by_train_stats(X_train, X_val, X_test):
    """
    用训练集统计量做标准化（避免泄漏）
    标准化方式：全局 mean/std（对整个矩阵一起算），更适合时序信号。
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    mean = float(X_train.mean())
    std = float(X_train.std())
    if std < 1e-6:
        std = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)


# ================= 4. 外部调用接口 =================

def get_dataloaders(X, y, d_true):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    d_true = np.asarray(d_true, dtype=np.int64)

    # Train / Temp
    X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
        X, y, d_true, train_size=cfg.TRAIN_RATIO, stratify=y, random_state=cfg.seed
    )

    # Val / Test
    test_rel_size = cfg.TEST_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
        X_temp, y_temp, d_temp, test_size=test_rel_size, stratify=y_temp, random_state=cfg.seed
    )

    # ✅ split 后做标准化：train fit，val/test transform
    X_train, X_val, X_test = _standardize_by_train_stats(X_train, X_val, X_test)

    loaders = {
        'train': DataLoader(SelfADataset(X_train, y_train, d_train),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=True),
        'val': DataLoader(SelfADataset(X_val, y_val, d_val),
                          batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers),
        'test': DataLoader(SelfADataset(X_test, y_test, d_test),
                           batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers)
    }
    return loaders


def prepare_mixed_self_domains(domain_list):
    """
    读取预处理好的 .npz 文件 (SelfA)
    """
    print("=" * 80)
    print(f"Loading Processed SelfA Data: {domain_list}")
    print("=" * 80)

    data_root = os.path.join(cfg.PROCESSED_DATA_ROOT, 'SelfA')

    all_X, all_y, all_d = [], [], []

    for speed in domain_list:
        file_path = os.path.join(data_root, f"domain_{speed}.npz")

        if not os.path.exists(file_path):
            print(f"Error: File not found {file_path}. Please run preprocess_to_numpy.py first!")
            continue

        print(f">>> Loading {speed} RPM from npz...")
        data = np.load(file_path)
        X = data['X']
        y = data['y']

        # 生成 domain 标签
        d = np.full(len(y), DOMAIN_MAP[speed], dtype=np.int64)

        all_X.append(X)
        all_y.append(y)
        all_d.append(d)

    if not all_X:
        raise ValueError("No data loaded. Check preprocess script.")

    X_mixed = np.concatenate(all_X, axis=0)
    y_mixed = np.concatenate(all_y, axis=0)
    d_mixed = np.concatenate(all_d, axis=0)

    print(f"  Total Samples: {len(y_mixed)}")

    return get_dataloaders(X_mixed, y_mixed, d_mixed)


def prepare_self_detailed():
    """单源域加载 (用于测试)"""
    full_data = {}
    data_root = os.path.join(cfg.PROCESSED_DATA_ROOT, 'SelfA')

    for speed in ALL_SPEEDS:
        file_path = os.path.join(data_root, f"domain_{speed}.npz")
        if os.path.exists(file_path):
            print(f">>> Loading {speed} RPM...")
            data = np.load(file_path)
            d = np.full(len(data['y']), DOMAIN_MAP[speed])
            full_data[speed] = get_dataloaders(data['X'], data['y'], d)

    return full_data


if __name__ == '__main__':
    print("\n" + "=" * 40)
    print("🔍 [SelfA Dataset Integrity Check]")
    print("=" * 40)

    try:
        # 1. 测试全量加载
        loaders = prepare_mixed_self_domains(['1200', '1800'])
        train_set = loaders['train'].dataset

        print(f"\n✅ Load Success!")
        print(f"Total Train Samples: {len(train_set)}")

        # 2. 检查一个 Batch
        data, label, domain, _, pdlabel, idx = next(iter(loaders['train']))

        print("\n📊 Batch Inspection:")
        print(f"  - Data Shape:   {data.shape}  [Expected: (64, 1, 1024)]")
        print(f"  - Label Shape:  {label.shape} [Expected: (64)]")
        print(f"  - Unique Labels in batch: {torch.unique(label).numpy()}")
        print(f"  - Unique Domains in batch: {torch.unique(domain).numpy()}")

        # 3. 数据范围检查 (验证归一化)
        print("\n📈 Data Statistics (Single Batch):")
        print(f"  - Mean: {data.mean().item():.4f} (Should be close to 0)")
        print(f"  - Std:  {data.std().item():.4f}  (Should be close to 1)")
        print(f"  - Max:  {data.max().item():.4f}")
        print(f"  - Min:  {data.min().item():.4f}")

        # 4. 形状断言
        assert data.shape[1:] == (1, 1024), "❌ Input dimension wrong!"
        assert label.max() < 4, "❌ Label index out of range!"

        print("\n✨ All Checks Passed for SelfA!")

    except Exception as e:
        print(f"\n❌ CHECK FAILED: {e}")