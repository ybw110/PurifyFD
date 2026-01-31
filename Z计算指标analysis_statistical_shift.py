import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from config import Config

# === 关键修改：导入新版 Viz 中的加载函数 ===
# 确保 Z画图viz_temporal_self.py 是上一轮提供的 V2 版本
from Z画图viz_temporal_self import load_data_for_viz

cfg = Config()

# ================= 全局配置 =================
SAVE_DIR = './Visualizations_Self/Stats'
os.makedirs(SAVE_DIR, exist_ok=True)
FS = 100000

plt.rcParams['font.family'] = 'Arial'


# ================= 1. 核心算法 (MMD) =================
def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma ** 2)
    dist = cdist(x, y, 'sqeuclidean')
    return np.exp(-beta * dist)


def compute_mmd(x, y):
    sigma = np.sqrt(x.shape[1])
    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    return np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)


def extract_features(signal_segment, sample_len=1024, n_samples=50):
    """
    提取特征
    注意：这里的 signal_segment 现在是一个单独的文件 (比如0.3s长)
    我们要从中随机采样若干个 1024 长度的片段来计算分布
    """
    total_len = len(signal_segment)
    if total_len < sample_len:
        return None

    # 从该文件中随机提取 n_samples 个片段，形成特征分布
    # 如果文件较短，就允许重叠
    indices = np.linspace(0, total_len - sample_len, n_samples, dtype=int)
    features = []

    for idx in indices:
        seg = signal_segment[idx: idx + sample_len]
        # 计算时域特征
        f_rms = np.sqrt(np.mean(seg ** 2))
        f_std = np.std(seg)
        f_peak = np.max(np.abs(seg))
        f_kurt = kurtosis(seg)
        f_crest = f_peak / (f_rms + 1e-9)
        features.append([f_rms, f_std, f_peak, f_kurt, f_crest])

    return np.array(features)


def analyze_drift_explicit(sig_early, sig_late, name):
    """
    计算 Early (File 0) vs Late (File 50) 的 MMD 漂移
    不再需要手动切分 signal_long，因为传入的已经是分开的两个信号
    """
    print(f"[{name}] Analyzing Drift...")
    print(f"  > Early shape: {sig_early.shape}")
    print(f"  > Late  shape: {sig_late.shape}")

    # 1. 提取特征分布
    feat_early = extract_features(sig_early)
    feat_late = extract_features(sig_late)

    if feat_early is None or feat_late is None:
        print(f"[{name}] ❌ Signal too short for feature extraction.")
        return None, None

    # 2. 归一化 (以 Early 为基准 fit)
    scaler = StandardScaler()
    scaler.fit(feat_early)
    feat_early_norm = scaler.transform(feat_early)
    feat_late_norm = scaler.transform(feat_late)

    # 3. 计算指标
    # (A) Temporal Shift: Early 分布 vs Late 分布的距离
    mmd_drift = compute_mmd(feat_early_norm, feat_late_norm)

    # (B) Baseline: Early 内部的前半部分 vs 后半部分 (作为噪音基准)
    # 将 Early 的特征再拆分两半计算自差异
    mid = len(feat_early_norm) // 2
    mmd_baseline = compute_mmd(feat_early_norm[:mid], feat_early_norm[mid:])

    print(f"  > Drift MMD: {mmd_drift:.5f}")
    print(f"  > Base  MMD: {mmd_baseline:.5f}")

    ratio = mmd_drift / (mmd_baseline + 1e-9)
    print(f"  > Ratio    : {ratio:.2f}x")

    return mmd_drift, mmd_baseline


# ================= 2. 主程序 =================
if __name__ == "__main__":
    print(">>> Calculating Statistical Drift (File 0 vs File 50)...")

    results = {'Dataset': [], 'Metric': [], 'Value': []}

    # --- SelfA ---
    # 直接利用 Viz 脚本加载好的 Early/Late 数据
    _, sA_early, sA_late, infoA = load_data_for_viz('SelfA')

    if sA_early is not None and sA_late is not None:
        drift, base = analyze_drift_explicit(sA_early, sA_late, "SelfA")
        if drift is not None:
            results['Dataset'].extend(['SelfA', 'SelfA'])
            results['Metric'].extend(['Baseline', 'Temporal Shift'])
            results['Value'].extend([base, drift])

    # --- SelfB ---
    _, sB_early, sB_late, infoB = load_data_for_viz('SelfB')

    if sB_early is not None and sB_late is not None:
        drift, base = analyze_drift_explicit(sB_early, sB_late, "SelfB")
        if drift is not None:
            results['Dataset'].extend(['SelfB', 'SelfB'])
            results['Metric'].extend(['Baseline', 'Temporal Shift'])
            results['Value'].extend([base, drift])

    # ================= 3. 绘图逻辑 =================
    if len(results['Value']) > 0:
        df = pd.DataFrame(results)
        plt.figure(figsize=(8, 6))
        colors = ['#cccccc', '#ff7f0e']

        datasets = df['Dataset'].unique()
        x = np.arange(len(datasets))
        width = 0.35

        for i, ds in enumerate(datasets):
            subset = df[df['Dataset'] == ds]
            if subset.empty: continue

            v_base = subset[subset['Metric'] == 'Baseline']['Value'].values[0]
            v_shift = subset[subset['Metric'] == 'Temporal Shift']['Value'].values[0]

            plt.bar(i - width / 2, v_base, width, color=colors[0], label='Baseline (Within-File)' if i == 0 else "")
            plt.bar(i + width / 2, v_shift, width, color=colors[1],
                    label='Time Shift (Early vs Late)' if i == 0 else "")

            # 标注倍数
            if v_base > 1e-9:
                ratio = v_shift / v_base
                plt.text(i + width / 2, v_shift, f"{ratio:.1f}x", ha='center', va='bottom', fontweight='bold')

        plt.xticks(x, datasets, fontsize=12)
        plt.ylabel("MMD Distance (Distribution Shift)", fontsize=12)
        plt.title("Statistical Quantification of Temporal Drift", fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        save_path = os.path.join(SAVE_DIR, 'SelfAB_Statistical_Shift.png')
        plt.savefig(save_path, dpi=300)
        print(f"\n✅ Stats plot saved: {save_path}")
    else:
        print("❌ No valid data calculated.")