import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config




# 实例化配置
cfg = Config()

# ================= 1. CWRU 基础配置 =================

# 常量：定义所有可用工况，方便全量训练调用
ALL_DOMAINS = ['0HP', '1HP', '2HP', '3HP']

# 真实工况映射 (True Domain ID)
DOMAIN_MAP = {'0HP': 0, '1HP': 1, '2HP': 2, '3HP': 3}

# 文件映射 (Class ID: 0-Normal, 1-Inner, 2-Ball, 3-Outer)
FILE_MAP = {
    '0HP': {0: ['97.mat'], 1: ['105.mat', '169.mat', '209.mat'], 2: ['118.mat', '185.mat', '222.mat'],
            3: ['130.mat', '197.mat', '234.mat']},
    '1HP': {0: ['98.mat'], 1: ['106.mat', '170.mat', '210.mat'], 2: ['119.mat', '186.mat', '223.mat'],
            3: ['131.mat', '198.mat', '235.mat']},
    '2HP': {0: ['99.mat'], 1: ['107.mat', '171.mat', '211.mat'], 2: ['120.mat', '187.mat', '224.mat'],
            3: ['132.mat', '199.mat', '236.mat']},
    '3HP': {0: ['100.mat'], 1: ['108.mat', '172.mat', '212.mat'], 2: ['121.mat', '188.mat', '225.mat'],
            3: ['133.mat', '200.mat', '237.mat']}
}


# ================= 2. Dataset 类 =================

class CWRUDataset(Dataset):
    def __init__(self, segments, labels, true_domains):
        self.segments = segments
        self.labels = labels
        self.true_domains = true_domains
        self.indices = np.arange(len(segments))

        # 初始化潜在域标签 (pdlabel)，默认全0，等待算法聚类更新
        self.pdlabels = np.zeros(len(segments), dtype=np.int64)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # 1. 获取数据并调整维度 (1, Length): (1024,) -> (1, 1024)
        segment = self.segments[idx]
        if len(segment.shape) == 1:
            segment = torch.FloatTensor(segment).unsqueeze(0)
        else:
            segment = torch.FloatTensor(segment)

        # 2. 标签封装
        label = torch.LongTensor([self.labels[idx]])[0]                 # Class (0-3)
        true_domain = torch.LongTensor([self.true_domains[idx]])[0]     # True Domain (0-3)
        pdlabel = torch.LongTensor([self.pdlabels[idx]])[0]             # Potential Domain
        index = torch.LongTensor([self.indices[idx]])[0]                # Sample Index

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


# ================= 3. 数据处理与统计工具 =================

def load_signal(file_path):
    try:
        data = scipy.io.loadmat(file_path)
        for key in data.keys():
            if key.endswith('DE_time'): return data[key].flatten()
    except:
        pass
    return None


def process_domain_data(domain_name):
    all_samples, all_labels, all_true_domains = [], [], []

    file_dict = FILE_MAP[domain_name]
    true_domain_id = DOMAIN_MAP[domain_name]

    for label, files in file_dict.items():
        for fname in files:
            root = cfg.data_root_normal if label == 0 else cfg.data_root_fault
            path = os.path.join(root, fname)

            signal = load_signal(path)
            if signal is None: continue

            # 标准化
            scaler = StandardScaler()
            signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

            # 切片
            n_samples = (len(signal_norm) - cfg.sequence_length) // cfg.stride + 1
            if n_samples <= 0: continue

            for i in range(n_samples):
                start = i * cfg.stride
                # 切出 Config 指定长度的信号
                segment = signal_norm[start: start + cfg.sequence_length]

                all_samples.append(segment)
                all_labels.append(label)
                all_true_domains.append(true_domain_id)

    return np.array(all_samples), np.array(all_labels), np.array(all_true_domains)


def get_dataloaders(X, y, d_true):
    # 1. 划分 Train / Temp
    X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
        X, y, d_true, train_size=cfg.split_ratio[0], stratify=y, random_state=cfg.seed
    )
    # 2. 划分 Val / Test
    rel_test_size = cfg.split_ratio[2] / (cfg.split_ratio[1] + cfg.split_ratio[2])
    X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
        X_temp, y_temp, d_temp, test_size=rel_test_size, stratify=y_temp, random_state=cfg.seed
    )

    loaders = {
        'train': DataLoader(CWRUDataset(X_train, y_train, d_train),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=True),

        'val': DataLoader(CWRUDataset(X_val, y_val, d_val),
                          batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers),

        'test': DataLoader(CWRUDataset(X_test, y_test, d_test),
                           batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers)
    }
    return loaders


def print_rich_info(labels, name="Dataset"):
    """辅助打印函数"""
    if isinstance(labels, torch.Tensor): labels = labels.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    cls_map = {0: 'Normal', 1: 'Inner', 2: 'Ball', 3: 'Outer'}
    details = " | ".join([f"{cls_map.get(k,k)}: {v}" for k,v in zip(unique, counts)])
    print(f"  [{name:<6}] Total: {len(labels):<5} || {details}")

# ================= 4. 主程序：生成与展示 =================

def prepare_all_domains_detailed():
    """获取所有单独域的Loader (用于 单源域训练 或 测试)"""
    full_data = {}
    print("=" * 80)
    print(f"Loading CWRU Data (Single Domain Mode)")
    print("=" * 80)

    for domain in ALL_DOMAINS:
        print(f"\n>>> Processing {domain}...")
        X, y, d_true = process_domain_data(domain)
        loaders = get_dataloaders(X, y, d_true)
        full_data[domain] = loaders

    return full_data


def prepare_mixed_domains(domain_list):
    """
    [多源域 / 全量数据加载]
    输入: domain_list (list), 例如 ['0HP', '1HP'] 或 ALL_DOMAINS
    输出: loaders (dict), 包含混合后的 train, val, test loader
    """
    print("=" * 80)
    print(f"Preparing Mixed CWRU Domains: {domain_list}")
    print("=" * 80)

    all_X, all_y, all_d = [], [], []

    # 1. 遍历列表，收集所有数据
    for domain in domain_list:
        print(f">>> Collecting {domain}...")
        X, y, d_true = process_domain_data(domain)
        all_X.append(X)
        all_y.append(y)
        all_d.append(d_true)

    # 2. 合并数据
    X_mixed = np.concatenate(all_X, axis=0)
    y_mixed = np.concatenate(all_y, axis=0)
    d_mixed = np.concatenate(all_d, axis=0)

    print_rich_info(y_mixed, "Mixed-CWRU")

    # 3. 统一划分 Train/Val
    loaders = get_dataloaders(X_mixed, y_mixed, d_mixed)

    return loaders

