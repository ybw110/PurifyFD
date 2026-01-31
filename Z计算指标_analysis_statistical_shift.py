import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import os
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, kurtosis
from sklearn.preprocessing import StandardScaler

# 引入配置
from config import Config

cfg = Config()

# ================= 配置 =================
# 字体设置
plt.rcParams['font.family'] = 'Arial'
SAVE_DIR = './Visualizations/Stats'
os.makedirs(SAVE_DIR, exist_ok=True)


# ================= 1. 核心算法: MMD 计算 =================
def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma ** 2)
    dist = cdist(x, y, 'sqeuclidean')
    return np.exp(-beta * dist)


def compute_mmd(x, y, sigma=None):
    """计算两个集合 X 和 Y 之间的 MMD 距离"""
    if sigma is None:
        sigma = np.sqrt(x.shape[1])  # 启发式设置 sigma

    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    return np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)


# ================= 2. 数据处理 =================
def extract_features(signal_long, sample_len=1024, n_samples=100):
    """从长信号中切片并提取特征(RMS, Kurtosis, Peak)"""
    total_len = len(signal_long)
    if total_len < sample_len: return None

    # 均匀切片
    indices = np.linspace(0, total_len - sample_len, n_samples, dtype=int)
    features = []

    for idx in indices:
        seg = signal_long[idx: idx + sample_len]
        # 提取时域特征 (比纯原始数据计算更稳定)
        f_rms = np.sqrt(np.mean(seg ** 2))
        f_kurt = kurtosis(seg)
        f_peak = np.max(np.abs(seg))
        f_std = np.std(seg)
        features.append([f_rms, f_kurt, f_peak, f_std])

    return np.array(features)


def analyze_temporal_shift(signal_long, fs, name):
    """分析单个长信号的时间偏移"""
    # 1. 切分时间段
    n_points = len(signal_long)
    # Early: 前 20%
    sig_early = signal_long[:int(0.2 * n_points)]
    # Late: 后 20%
    sig_late = signal_long[-int(0.2 * n_points):]

    # 2. 提取特征样本
    feat_early = extract_features(sig_early)
    feat_late = extract_features(sig_late)

    if feat_early is None or feat_late is None:
        return None, None

    # 3. 归一化 (非常重要，否则不同量纲无法计算距离)
    scaler = StandardScaler()
    scaler.fit(feat_early)  # 以 Early 为基准
    feat_early_norm = scaler.transform(feat_early)
    feat_late_norm = scaler.transform(feat_late)

    # 4. 计算指标
    # MMD: 衡量多维分布差异
    mmd_score = compute_mmd(feat_early_norm, feat_late_norm)

    # Baseline: 计算 Early 内部的差异 (前半 vs 后半) 作为对照组
    mid = len(feat_early_norm) // 2
    mmd_baseline = compute_mmd(feat_early_norm[:mid], feat_early_norm[mid:])

    # Wasserstein (以 RMS 为例)
    wd_score = wasserstein_distance(feat_early_norm[:, 0], feat_late_norm[:, 0])

    print(f"[{name}] Time Shift Analysis:")
    print(f"  > MMD (Early vs Late): {mmd_score:.5f}")
    print(f"  > MMD Baseline (Self): {mmd_baseline:.5f}")
    print(f"  > Drift Ratio: {mmd_score / (mmd_baseline + 1e-9):.2f}x")

    return mmd_score, mmd_baseline


# ================= 3. 数据读取 (复用之前的逻辑) =================
def get_raw_cwru():
    try:
        path = os.path.join(cfg.data_root_fault, '105.mat')
        data = scipy.io.loadmat(path)
        for k in data:
            if 'DE_time' in k: return data[k].flatten()
    except:
        return None


def get_raw_jnu():
    try:
        # 需替换为您实际的 JNU 路径和文件名
        root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'
        # 简单搜寻一个 ib600 文件
        for f in os.listdir(root):
            if f.startswith('ib600') and f.endswith('.csv'):
                df = pd.read_csv(os.path.join(root, f), header=None)
                return df.iloc[:, -1].values.flatten()
    except:
        return None


# ================= 4. 主程序与绘图 =================
if __name__ == "__main__":

    # 读取数据
    sig_cwru = get_raw_cwru()
    sig_jnu = get_raw_jnu()

    results = {'Dataset': [], 'Metric': [], 'Value': []}

    if sig_cwru is not None:
        mmd, base = analyze_temporal_shift(sig_cwru, 12000, "CWRU")
        results['Dataset'].extend(['CWRU', 'CWRU'])
        results['Metric'].extend(['Baseline (Self)', 'Temporal Shift (Early vs Late)'])
        results['Value'].extend([base, mmd])

    if sig_jnu is not None:
        mmd, base = analyze_temporal_shift(sig_jnu, 51200, "JNU")
        results['Dataset'].extend(['JNU', 'JNU'])
        results['Metric'].extend(['Baseline (Self)', 'Temporal Shift (Early vs Late)'])
        results['Value'].extend([base, mmd])

    # 绘图
    df_res = pd.DataFrame(results)

    plt.figure(figsize=(8, 6))

    # 颜色映射
    colors = {'Baseline (Self)': '#cccccc', 'Temporal Shift (Early vs Late)': '#d62728'}

    # 简单的柱状图逻辑
    datasets = df_res['Dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35

    for i, ds in enumerate(datasets):
        subset = df_res[df_res['Dataset'] == ds]
        val_base = subset[subset['Metric'] == 'Baseline (Self)']['Value'].values[0]
        val_shift = subset[subset['Metric'] == 'Temporal Shift (Early vs Late)']['Value'].values[0]

        plt.bar(i - width / 2, val_base, width, label='Baseline (Stable)' if i == 0 else "", color='#999999')
        plt.bar(i + width / 2, val_shift, width, label='Temporal Shift' if i == 0 else "", color='#d62728')

        # 标注倍数
        ratio = val_shift / val_base if val_base > 0 else 0
        plt.text(i + width / 2, val_shift + 0.01, f"{ratio:.1f}x", ha='center', fontweight='bold')

    plt.xticks(x, datasets, fontsize=12)
    plt.ylabel("Maximum Mean Discrepancy (MMD)", fontsize=12)
    plt.title("Quantification of Temporal Distribution Shift", fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    save_path = os.path.join(SAVE_DIR, 'Statistical_Shift_Quantification.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved to: {save_path}")