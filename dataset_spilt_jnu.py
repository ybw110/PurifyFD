import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config

cfg = Config()

# ================= 1. JNU 文件映射表 =================
ALL_SPEEDS = ['600', '800', '1000']
DOMAIN_MAP_JNU = {'600': 0, '800': 1, '1000': 2}
CLASS_MAP_JNU = {'n': 0, 'ib': 1, 'tb': 2, 'ob': 3}
FILE_PREFIXES = ['n', 'ib', 'tb', 'ob']

# [新增] 标签重映射
LABEL_MAP = {original: mapped for mapped, original in enumerate(cfg.known_classes)}

# ================= 2. Dataset 类 =================

class JNUDataset(Dataset):
    def __init__(self, segments, labels, true_domains):
        self.segments = segments
        self.labels = labels
        self.true_domains = true_domains
        self.indices = np.arange(len(segments))
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

        return segment, label, true_domain, torch.zeros(1), pdlabel, index

    def set_labels_by_index(self, pred_labels, indices, label_type='pdlabel'):
        pred_labels, indices = np.asarray(pred_labels), np.asarray(indices)
        if label_type == 'pdlabel':
            for i, idx in enumerate(indices):
                if 0 <= idx < len(self.pdlabels):
                    self.pdlabels[int(idx)] = int(pred_labels[i])

# ================= 3. 数据加载与处理工具 =================

def load_jnu_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        signal = df.iloc[:, -1].values.flatten()
        return signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def process_jnu_domain(speed, root_path):
    all_samples, all_labels, all_true_domains = [], [], []
    true_domain_id = DOMAIN_MAP_JNU[speed]

    for prefix in FILE_PREFIXES:
        label = CLASS_MAP_JNU[prefix]
        target_files = [f for f in os.listdir(root_path) if f.startswith(prefix) and speed in f and f.endswith('.csv')]

        for fname in target_files:
            path = os.path.join(root_path, fname)
            signal = load_jnu_csv(path)
            if signal is None: continue

            scaler = StandardScaler()
            signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

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
    X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
        X, y, d_true, train_size=cfg.split_ratio[0], stratify=y, random_state=cfg.seed
    )
    rel_test_size = cfg.split_ratio[2] / (cfg.split_ratio[1] + cfg.split_ratio[2])
    X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
        X_temp, y_temp, d_temp, test_size=rel_test_size, stratify=y_temp, random_state=cfg.seed
    )

    # === [核心修改] 处理函数 ===
    def process_subset(X_sub, y_sub, d_sub, is_training=False):
        if is_training:
            # 过滤已知类
            mask = np.isin(y_sub, cfg.known_classes)
            X_sub, y_sub, d_sub = X_sub[mask], y_sub[mask], d_sub[mask]
            # 重映射标签
            y_remapped = np.array([LABEL_MAP[label] for label in y_sub])
            return JNUDataset(X_sub, y_remapped, d_sub)
        else:
            # 测试集保留全量和原始标签
            return JNUDataset(X_sub, y_sub, d_sub)

    train_ds = process_subset(X_train, y_train, d_train, is_training=True)
    val_ds = process_subset(X_val, y_val, d_val, is_training=True)
    test_ds = process_subset(X_test, y_test, d_test, is_training=False)

    loaders = {
        'train': DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=True),
        'val': DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers),
        'test': DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers)
    }
    return loaders


def prepare_jnu_detailed(root_path):
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