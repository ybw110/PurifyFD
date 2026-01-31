import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Config

cfg = Config()

# ================= 1. 映射配置 (保持和 CWRU 逻辑一致) =================
# [新增] 动态生成映射字典: {0:0, 1:1, 3:2}
LABEL_MAP = {original: mapped for mapped, original in enumerate(cfg.known_classes)}

ALL_SPEEDS = ['1200', '1800']
DOMAIN_MAP = {'1200': 0, '1800': 1}

# B数据集不需要 Class Map，因为已经在预处理阶段转成了数字标签

class SelfBDataset(Dataset):
    def __init__(self, segments, labels, true_domains):
        self.segments = segments
        self.labels = labels
        self.true_domains = true_domains
        self.indices = np.arange(len(segments))
        self.pdlabels = np.zeros(len(segments), dtype=np.int64)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        true_domain = torch.LongTensor([self.true_domains[idx]])[0]
        pdlabel = torch.LongTensor([self.pdlabels[idx]])[0]
        index = torch.LongTensor([self.indices[idx]])[0]
        return segment, label, true_domain, torch.zeros(1), pdlabel, index

    def set_labels_by_index(self, pred_labels, indices, label_type='pdlabel'):
        if label_type == 'pdlabel':
            for i, idx in enumerate(indices):
                self.pdlabels[int(idx)] = int(pred_labels[i])


def _filter_and_remap_known(X, y, d_true, cfg):
    X = np.asarray(X)
    y = np.asarray(y)
    d_true = np.asarray(d_true)

    known_set = set(cfg.known_classes)
    mask = np.array([yy in known_set for yy in y], dtype=bool)

    Xk = X[mask]
    yk_orig = y[mask]
    dk = d_true[mask]

    # remap: 物理标签 -> 训练标签 0..K-1
    yk = np.array([LABEL_MAP[int(yy)] for yy in yk_orig], dtype=np.int64)

    return Xk, yk, dk


def _standardize_by_train_stats(X_train, X_val, X_test):
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

def get_dataloaders(X, y, d_true):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    d_true = np.asarray(d_true, dtype=np.int64)

    # 1) 先按“全类”做切分（保证 unknown 只出现在 test 里也有分布）
    X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
        X, y, d_true, train_size=cfg.TRAIN_RATIO, stratify=y, random_state=cfg.seed
    )
    test_rel_size = cfg.TEST_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
        X_temp, y_temp, d_temp, test_size=test_rel_size, stratify=y_temp, random_state=cfg.seed
    )

    # 2) Train/Val 过滤 unknown + 重映射
    X_train_k, y_train_k, d_train_k = _filter_and_remap_known(X_train, y_train, d_train, cfg)
    X_val_k, y_val_k, d_val_k = _filter_and_remap_known(X_val, y_val, d_val, cfg)

    # 3) 用“过滤后的 train(known)”做标准化统计量，再作用于 val(known) 和 test(full)
    X_train_k, X_val_k, X_test = _standardize_by_train_stats(X_train_k, X_val_k, X_test)

    loaders = {
        'train': DataLoader(SelfBDataset(X_train_k, y_train_k, d_train_k),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=True),
        'val':   DataLoader(SelfBDataset(X_val_k, y_val_k, d_val_k),
                            batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers),
        # ✅ test：y_test 保持原始物理标签 0~3（包含 unknown）
        'test':  DataLoader(SelfBDataset(X_test, y_test, d_test),
                            batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
    }
    return loaders



def prepare_mixed_self_b_domains(domain_list):
    print("=" * 80)
    print(f"Loading Processed SelfB Data: {domain_list}")
    print("=" * 80)

    data_root = os.path.join(cfg.PROCESSED_DATA_ROOT, 'SelfB')
    all_X, all_y, all_d = [], [], []

    for speed in domain_list:
        file_path = os.path.join(data_root, f"domain_{speed}.npz")
        if not os.path.exists(file_path):
            print(f"Error: File not found {file_path}")
            continue

        print(f">>> Loading {speed} RPM from npz...")
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        d = np.full(len(y), DOMAIN_MAP[speed], dtype=np.int64)

        all_X.append(X)
        all_y.append(y)
        all_d.append(d)

    if not all_X: raise ValueError("No data loaded.")

    X_mixed = np.concatenate(all_X, axis=0)
    y_mixed = np.concatenate(all_y, axis=0)
    d_mixed = np.concatenate(all_d, axis=0)
    print(f"  Total Samples: {len(y_mixed)}")

    return get_dataloaders(X_mixed, y_mixed, d_mixed)


def prepare_self_b_detailed():
    full_data = {}
    data_root = os.path.join(cfg.PROCESSED_DATA_ROOT, 'SelfB')
    for speed in ALL_SPEEDS:
        file_path = os.path.join(data_root, f"domain_{speed}.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            d = np.full(len(data['y']), DOMAIN_MAP[speed])
            full_data[speed] = get_dataloaders(data['X'], data['y'], d)
    return full_data


if __name__ == '__main__':
    print("\n" + "=" * 40)
    print("🔍 [SelfB Dataset Integrity Check]")
    print("=" * 40)

    try:
        loaders = prepare_mixed_self_b_domains(['1200', '1800'])
        train_set = loaders['train'].dataset

        print(f"\n✅ Load Success!")
        print(f"Total Train Samples: {len(train_set)}")

        data, label, domain, _, pdlabel, idx = next(iter(loaders['train']))

        print("\n📊 Batch Inspection:")
        print(f"  - Data Shape:   {data.shape}  [Expected: (64, 1, 1024)]")
        print(f"  - Label Shape:  {label.shape} [Expected: (64)]")
        print(f"  - Unique Labels: {torch.unique(label).numpy()}")
        print(f"  - Unique Domains: {torch.unique(domain).numpy()}")

        print("\n📈 Data Statistics:")
        print(f"  - Mean: {data.mean().item():.4f}")
        print(f"  - Std:  {data.std().item():.4f}")

        assert data.shape[1:] == (1, 1024), "❌ Input dimension wrong!"

        print("\n✨ All Checks Passed for SelfB!")

    except Exception as e:
        print(f"\n❌ CHECK FAILED: {e}")