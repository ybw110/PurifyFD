import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from config import Config
import matplotlib.colors as mcolors # <--- 【新增1】 需要导入这个库来创建自定义色系
# 必须依赖 Viz 脚本来加载独立文件
try:
    from Z画图viz_temporal_self import load_data_for_viz
except ImportError:
    print("❌ Error: 'Z画图viz_temporal_self.py' not found!")
    exit()

cfg = Config()

# ================= 配置 =================
SAVE_DIR = './Visualizations/Stats'
os.makedirs(SAVE_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14


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


# ================= 2. 主程序 =================
if __name__ == "__main__":
    print(">>> Generating Heatmap for PRIVATE Datasets (SelfA & SelfB)...")

    data_store = {}

    # 获取 SelfA
    _, sA_early, sA_late, _ = load_data_for_viz('SelfA')
    if sA_early is not None:
        data_store['SelfA (Early)'] = extract_features(sA_early)
        data_store['SelfA (Late)'] = extract_features(sA_late)

    # 获取 SelfB
    _, sB_early, sB_late, _ = load_data_for_viz('SelfB')
    if sB_early is not None:
        data_store['SelfB (Early)'] = extract_features(sB_early)
        data_store['SelfB (Late)'] = extract_features(sB_late)

    labels = ['SelfA (Early)', 'SelfA (Late)', 'SelfB (Early)', 'SelfB (Late)']
    valid_labels = [l for l in labels if l in data_store]

    if len(valid_labels) < 2:
        print("❌ Not enough data found. Check your paths.")
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

    # ================= 绘图部分修改开始 =================

    # 【新增2】 定义自定义颜色列表：从蓝到红
    # #BBD5E7 (低MMD，蓝色) -> #E59693 (高MMD，红色)
    custom_colors = ["#BBD5E7", "#E59693"]

    # 【新增3】 创建自定义 colormap 对象
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue_red", custom_colors)

    plt.figure(figsize=(8, 6))
    # sns.set(font_scale=1.1)

    # 使用自定义的 cmap
    ax = sns.heatmap(mmd_matrix,
                     xticklabels=valid_labels,
                     yticklabels=valid_labels,
                     annot=True,
                     fmt=".2f",
                     cmap=custom_cmap,  # <--- 【修改点】这里替换了原来的 "OrRd"
                     vmin=0,
                     linewidths=1.5, linecolor='white',
                     annot_kws={"size": 20,},
                     cbar_kws={'label': 'MMD Distance'})

    # plt.title("Distribution Shift: Private Datasets", fontsize=14, pad=15)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)  # 色条刻度大小
    cbar.set_label('MMD Distance', fontsize=18)  # 色条标题大小

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, 'Heatmap_Private_SelfAB_CustomColor.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")