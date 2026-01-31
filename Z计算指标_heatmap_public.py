import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import scipy.io
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from config import Config

cfg = Config()

# ================= 配置 =================
SAVE_DIR = './Visualizations/Stats'
os.makedirs(SAVE_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


# ================= 1. 核心算法 =================
def compute_mmd(x, y):
    sigma = np.sqrt(x.shape[1]) if x.shape[1] > 0 else 1.0

    def gaussian_kernel(a, b):
        beta = 1.0 / (2.0 * sigma ** 2)
        dist = cdist(a, b, 'sqeuclidean')
        return np.exp(-beta * dist)

    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)
    return np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)


def extract_features(signal_segment, sample_len=1024, n_samples=100):
    total_len = len(signal_segment)
    if total_len < sample_len: return None
    indices = np.linspace(0, total_len - sample_len, n_samples, dtype=int)
    features = []
    for idx in indices:
        seg = signal_segment[idx: idx + sample_len]
        f_rms = np.sqrt(np.mean(seg ** 2))
        f_std = np.std(seg)
        f_peak = np.max(np.abs(seg))
        f_kurt = kurtosis(seg)
        f_crest = f_peak / (f_rms + 1e-9)
        features.append([f_rms, f_std, f_peak, f_kurt, f_crest])
    return np.array(features)


# ================= 2. 数据获取 (Public) =================
def get_cwru_pair():
    # 0HP Inner Race
    filename = '105.mat'
    path = os.path.join(cfg.data_root_fault, filename)
    try:
        data = scipy.io.loadmat(path)
        for k in data:
            if k.endswith('DE_time'):
                sig = data[k].flatten()
                n = len(sig)
                return sig[:int(0.2 * n)], sig[-int(0.2 * n):]
    except:
        pass
    return None, None


def get_jnu_pair():
    root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'
    try:
        for f in os.listdir(root):
            if f.startswith('ib600') and f.endswith('.csv'):
                df = pd.read_csv(os.path.join(root, f), header=None)
                sig = df.iloc[:, -1].values.flatten()
                n = len(sig)
                return sig[:int(0.2 * n)], sig[-int(0.2 * n):]
    except:
        pass
    return None, None


# ================= 3. 主程序 =================
if __name__ == "__main__":
    print(">>> Generating Heatmap for PUBLIC Datasets (CWRU & JNU)...")

    data_store = {}

    # 获取 CWRU
    s1, s2 = get_cwru_pair()
    if s1 is not None:
        data_store['CWRU (Early)'] = extract_features(s1)
        data_store['CWRU (Late)'] = extract_features(s2)

    # 获取 JNU
    s1, s2 = get_jnu_pair()
    if s1 is not None:
        data_store['JNU (Early)'] = extract_features(s1)
        data_store['JNU (Late)'] = extract_features(s2)

    labels = ['CWRU (Early)', 'CWRU (Late)', 'JNU (Early)', 'JNU (Late)']
    valid_labels = [l for l in labels if l in data_store]

    if len(valid_labels) < 2:
        print("❌ Not enough data found.")
        exit()

    # 标准化
    all_features = np.vstack([data_store[k] for k in valid_labels])
    scaler = StandardScaler()
    scaler.fit(all_features)
    data_norm = {k: scaler.transform(data_store[k]) for k in valid_labels}

    # 计算矩阵
    n = len(valid_labels)
    mmd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mmd_matrix[i, j] = 0.0
            else:
                mmd_matrix[i, j] = compute_mmd(data_norm[valid_labels[i]], data_norm[valid_labels[j]])

    # 绘图
    plt.figure(figsize=(7, 6))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(mmd_matrix, xticklabels=valid_labels, yticklabels=valid_labels,
                     annot=True, fmt=".2f", cmap="Blues", vmin=0,
                     linewidths=1, linecolor='white',
                     cbar_kws={'label': 'MMD Distance'})

    plt.title("Distribution Shift: Public Benchmarks", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, 'Heatmap_Public_CWRU_JNU.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")