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

ALL_DOMAINS = ['0HP', '1HP', '2HP', '3HP']
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

# [新增] 标签重映射字典：将物理标签映射为网络训练标签
# 例如 known_classes=[0, 1, 3] -> {0:0, 1:1, 3:2}
LABEL_MAP = {original: mapped for mapped, original in enumerate(cfg.known_classes)}


# ================= 2. Dataset 类 =================

class CWRUDataset(Dataset):
    def __init__(self, segments, labels, true_domains):
        self.segments = segments
        self.labels = labels
        self.true_domains = true_domains
        self.indices = np.arange(len(segments))

        # 初始化潜在域标签 (pdlabel)，默认全0
        self.pdlabels = np.zeros(len(segments), dtype=np.int64)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # 1. 获取数据并调整维度
        segment = self.segments[idx]
        if len(segment.shape) == 1:
            segment = torch.FloatTensor(segment).unsqueeze(0)
        else:
            segment = torch.FloatTensor(segment)

        # 2. 标签封装
        label = torch.LongTensor([self.labels[idx]])[0]
        true_domain = torch.LongTensor([self.true_domains[idx]])[0]
        pdlabel = torch.LongTensor([self.pdlabels[idx]])[0]
        index = torch.LongTensor([self.indices[idx]])[0]

        return segment, label, true_domain, torch.zeros(1), pdlabel, index

    def set_labels_by_index(self, pred_labels, indices, label_type='pdlabel'):
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
        # 如果是 OOD 设置，虽然这里我们读取了所有数据，但在 get_dataloaders 里会过滤
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
                segment = signal_norm[start: start + cfg.sequence_length]

                all_samples.append(segment)
                all_labels.append(label)
                all_true_domains.append(true_domain_id)

    return np.array(all_samples), np.array(all_labels), np.array(all_true_domains)


def get_dataloaders(X, y, d_true):
    """
    修改后的数据加载器：
    - Train/Val: 过滤未知类，并重映射标签 (0,1,3 -> 0,1,2)
    - Test: 保留所有类别，保留原始标签 (0,1,2,3)
    """

    # 1. 划分 Train / Temp
    X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
        X, y, d_true, train_size=cfg.split_ratio[0], stratify=y, random_state=cfg.seed
    )
    # 2. 划分 Val / Test
    rel_test_size = cfg.split_ratio[2] / (cfg.split_ratio[1] + cfg.split_ratio[2])
    X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
        X_temp, y_temp, d_temp, test_size=rel_test_size, stratify=y_temp, random_state=cfg.seed
    )

    # === [核心修改] 数据子集处理函数 ===
    def process_subset(X_sub, y_sub, d_sub, is_training=False):
        if is_training:
            # A. 过滤：只保留已知类
            mask = np.isin(y_sub, cfg.known_classes)
            X_sub, y_sub, d_sub = X_sub[mask], y_sub[mask], d_sub[mask]

            # B. 重映射：将物理标签转为网络标签 (例如 0,1,3 -> 0,1,2)
            # 注意：必须确保 y_sub 中的标签都在 LABEL_MAP 中
            y_remapped = np.array([LABEL_MAP[label] for label in y_sub])
            return CWRUDataset(X_sub, y_remapped, d_sub)
        else:
            # 测试集：不过滤，不重映射 (用于 OOD 检测评估)
            return CWRUDataset(X_sub, y_sub, d_sub)

    # 3. 创建数据集实例
    train_ds = process_subset(X_train, y_train, d_train, is_training=True)
    val_ds = process_subset(X_val, y_val, d_val, is_training=True)  # 验证集通常也只验证已知类效果
    test_ds = process_subset(X_test, y_test, d_test, is_training=False)  # 测试集包含未知类

    print(f"Dataset Split Info:")
    print(f"  Train (Known Only): {len(train_ds)} samples")
    print(f"  Val   (Known Only): {len(val_ds)} samples")
    print(f"  Test  (Open Set)  : {len(test_ds)} samples")

    loaders = {
        'train': DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=True),

        'val': DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers),

        'test': DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers)
    }
    return loaders


def print_rich_info(labels, name="Dataset"):
    if isinstance(labels, torch.Tensor): labels = labels.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    cls_map = {0: 'Normal', 1: 'Inner', 2: 'Ball', 3: 'Outer'}
    details = " | ".join([f"{cls_map.get(k, k)}: {v}" for k, v in zip(unique, counts)])
    print(f"  [{name:<6}] Total: {len(labels):<5} || {details}")


def prepare_all_domains_detailed():
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
    print("=" * 80)
    print(f"Preparing Mixed CWRU Domains: {domain_list}")
    print("=" * 80)

    all_X, all_y, all_d = [], [], []

    for domain in domain_list:
        print(f">>> Collecting {domain}...")
        X, y, d_true = process_domain_data(domain)
        all_X.append(X)
        all_y.append(y)
        all_d.append(d_true)

    X_mixed = np.concatenate(all_X, axis=0)
    y_mixed = np.concatenate(all_y, axis=0)
    d_mixed = np.concatenate(all_d, axis=0)

    print_rich_info(y_mixed, "Mixed-CWRU")
    loaders = get_dataloaders(X_mixed, y_mixed, d_mixed)

    return loaders