import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config

# 实例化配置
cfg = Config()

# ================= 1. JNU 文件映射表 =================
# [新增] 常量：所有转速
ALL_SPEEDS = ['600', '800', '1000']

# 工况映射 (转速作为 Domain ID)
DOMAIN_MAP_JNU = {'600': 0, '800': 1, '1000': 2}

# 类别映射 (n: Normal, ib: Inner, tb: Ball, ob: Outer)
# 注意：JNU数据通常包含两列，我们取振动信号列
CLASS_MAP_JNU = {'n': 0, 'ib': 1, 'tb': 2, 'ob': 3}

# 文件前缀映射
FILE_PREFIXES = ['n', 'ib', 'tb', 'ob']
SPEEDS = ['600', '800', '1000']


# ================= 2. Dataset 类 (保持与 CWRU 一致) =================

class JNUDataset(Dataset):
    def __init__(self, segments, labels, true_domains):
        self.segments = segments
        self.labels = labels
        self.true_domains = true_domains
        self.indices = np.arange(len(segments))
        # 潜在域标签，默认全0
        self.pdlabels = np.zeros(len(segments), dtype=np.int64)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        if len(segment.shape) == 1:
            segment = torch.FloatTensor(segment).unsqueeze(0)
        else:
            segment = torch.FloatTensor(segment)

        label = torch.LongTensor([self.labels[idx]])[0]
        true_domain = torch.LongTensor([self.true_domains[idx]])[0]
        pdlabel = torch.LongTensor([self.pdlabels[idx]])[0]
        index = torch.LongTensor([self.indices[idx]])[0]

        # 返回 6 元组以适配 Diversify
        return segment, label, true_domain, torch.zeros(1), pdlabel, index

    def set_labels_by_index(self, pred_labels, indices, label_type='pdlabel'):
        pred_labels, indices = np.asarray(pred_labels), np.asarray(indices)
        if label_type == 'pdlabel':
            for i, idx in enumerate(indices):
                if 0 <= idx < len(self.pdlabels):
                    self.pdlabels[int(idx)] = int(pred_labels[i])


# ================= 3. 数据加载与处理工具 =================

def load_jnu_csv(file_path):
    """读取JNU CSV文件并提取信号"""
    try:
        # JNU 数据通常第二列是振动信号
        df = pd.read_csv(file_path, header=None)
        # 自动探测信号列（通常取最后一列或第二列）
        signal = df.iloc[:, -1].values.flatten()
        return signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def process_jnu_domain(speed, root_path):
    """处理特定转速下的所有故障类型"""
    all_samples, all_labels, all_true_domains = [], [], []
    true_domain_id = DOMAIN_MAP_JNU[speed]

    for prefix in FILE_PREFIXES:
        label = CLASS_MAP_JNU[prefix]
        # 匹配文件名，如 n600_3_2.csv 或 ib600_2.csv
        # 这里使用简单的包含逻辑，你可以根据实际文件名精确匹配
        target_files = [f for f in os.listdir(root_path) if f.startswith(prefix) and speed in f and f.endswith('.csv')]

        for fname in target_files:
            path = os.path.join(root_path, fname)
            signal = load_jnu_csv(path)
            if signal is None: continue

            # 标准化
            scaler = StandardScaler()
            signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

            # 切片
            n_samples = (len(signal_norm) - cfg.sequence_length) // cfg.stride + 1
            if n_samples <= 0: continue

            for i in range(n_samples):
                start = i * cfg.stride
                segment = signal_norm[start: start + cfg.sequence_length]
                all_samples.append(segment)
                all_labels.append(label)
                all_true_domains.append(true_domain_id)

    return np.array(all_samples), np.array(all_labels), np.array(all_true_domains)


def get_jnu_dataloaders(X, y, d_true):
    """划分数据集"""
    X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
        X, y, d_true, train_size=cfg.split_ratio[0], stratify=y, random_state=cfg.seed
    )
    rel_test_size = cfg.split_ratio[2] / (cfg.split_ratio[1] + cfg.split_ratio[2])
    X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
        X_temp, y_temp, d_temp, test_size=rel_test_size, stratify=y_temp, random_state=cfg.seed
    )

    loaders = {
        'train': DataLoader(JNUDataset(X_train, y_train, d_train), batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=True),
        'val': DataLoader(JNUDataset(X_val, y_val, d_val), batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers),
        'test': DataLoader(JNUDataset(X_test, y_test, d_test), batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers)
    }
    return loaders


def prepare_jnu_detailed(root_path):
    """主接口：生成JNU所有单独工况的Loader"""
    full_data = {}
    print("=" * 80)
    print(f"Loading JNU Data (Single Domain Mode)")
    print("=" * 80)

    for speed in ALL_SPEEDS:
        print(f"\n>>> Processing JNU Speed: {speed} RPM...")
        X, y, d_true = process_jnu_domain(speed, root_path)
        loaders = get_jnu_dataloaders(X, y, d_true)
        full_data[speed] = loaders

    return full_data


def prepare_mixed_jnu_domains(domain_list, root_path):
    """
    [新增] 多源域 JNU 数据加载
    输入: domain_list (list), 例如 ['600', '800']
    """
    print("=" * 80)
    print(f"Preparing Mixed JNU Domains: {domain_list}")
    print("=" * 80)

    all_X, all_y, all_d = [], [], []

    for speed in domain_list:
        print(f">>> Collecting {speed}...")
        X, y, d_true = process_jnu_domain(speed, root_path)
        all_X.append(X)
        all_y.append(y)
        all_d.append(d_true)

    X_mixed = np.concatenate(all_X, axis=0)
    y_mixed = np.concatenate(all_y, axis=0)
    d_mixed = np.concatenate(all_d, axis=0)

    print(f"  Mixed Total: {len(y_mixed)}")

    loaders = get_jnu_dataloaders(X_mixed, y_mixed, d_mixed)
    return loaders

if __name__ == '__main__':
    # 测试路径
    jnu_path = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'
    all_jnu_data = prepare_jnu_detailed(jnu_path)

    # 验证一个Batch
    loader = all_jnu_data['600']['train']
    segments, labels, domains, _, pd_labels, indices = next(iter(loader))
    print(f"\nBatch Shape: {segments.shape}")  # 应为 [64, 1, 1024]