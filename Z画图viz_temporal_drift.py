import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle

# 引入配置以获取路径
from config import Config

cfg = Config()

# ================= 全局配置 =================
SAVE_DIR = './Visualizations/Temporal_Shift_Fixed'
os.makedirs(SAVE_DIR, exist_ok=True)

# 字体配置 (保持统一)
FONT_CONFIG = {
    'font_family': 'Arial',
    'title_size': 14,
    'label_size': 12,
    'tick_size': 10
}
plt.rcParams['font.family'] = FONT_CONFIG['font_family']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']


# ================= 1. 数据读取函数 =================

def read_raw_cwru(fault_code=1):
    """
    读取 CWRU 0HP 工况下的原始长信号
    fault_code: 1 (Inner Race, 105.mat)
    """
    filename = '105.mat'
    file_path = os.path.join(cfg.data_root_fault, filename)

    print(f"Reading CWRU File: {file_path}")
    try:
        data = scipy.io.loadmat(file_path)
        for key in data.keys():
            if key.endswith('DE_time'):
                return data[key].flatten()
    except Exception as e:
        print(f"❌ Error reading CWRU: {e}")
    return None


def read_raw_jnu(root_path, speed='600'):
    """
    读取 JNU 指定转速下的原始长信号
    Target: Inner Race Fault (ib)
    """
    target_prefix = f"ib{speed}"
    try:
        files = os.listdir(root_path)
        for f in files:
            if f.startswith(target_prefix) and f.endswith('.csv'):
                found_file = os.path.join(root_path, f)
                print(f"Reading JNU File: {found_file}")
                df = pd.read_csv(found_file, header=None)
                return df.iloc[:, -1].values.flatten()
    except Exception as e:
        print(f"❌ Error reading JNU: {e}")
    return None


# ================= 2. 核心绘图逻辑 =================

# --- 图类型 A: 原版带连线的宏观+微观图 ---
def plot_macro_with_connections(signal_long, fs, dataset_name, save_name):
    """
    [原版逻辑] 绘制 1秒长信号 + 前后0.1秒放大区域 + 连线
    """
    # 参数设置
    duration_total = 1.0  # 总展示时长 1s
    duration_zoom = 0.1  # 放大窗口时长 0.1s

    n_total = int(duration_total * fs)
    n_zoom = int(duration_zoom * fs)

    # 截取中间稳定的 1s
    if len(signal_long) < n_total:
        sig_display = signal_long
        n_total = len(signal_long)
    else:
        start_offset = 1000
        sig_display = signal_long[start_offset: start_offset + n_total]

    t_full = np.linspace(0, len(sig_display) / fs, len(sig_display))

    # 切片
    sig_head = sig_display[:n_zoom]
    t_head = t_full[:n_zoom]
    sig_tail = sig_display[-n_zoom:]
    t_tail = t_full[-n_zoom:]

    # 绘图
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax_main = fig.add_subplot(gs[0, :])
    ax_zoom1 = fig.add_subplot(gs[1, 0])
    ax_zoom2 = fig.add_subplot(gs[1, 1])

    # 1. Macro View
    ax_main.plot(t_full, sig_display, color='#333333', lw=0.5, alpha=0.8)
    ax_main.set_title(f"{dataset_name} - 1s Continuous Waveform (Macro View)", fontsize=FONT_CONFIG['title_size'])
    ax_main.set_ylabel("Amplitude", fontsize=FONT_CONFIG['label_size'])
    ax_main.set_xlim(0, t_full[-1])
    ax_main.grid(True, alpha=0.2)

    # Rectangles
    y_min, y_max = min(sig_display), max(sig_display)
    h = y_max - y_min
    rect1 = Rectangle((0, y_min), duration_zoom, h, edgecolor='#1f77b4', facecolor='none', lw=2, zorder=10)
    rect2 = Rectangle((t_tail[0], y_min), duration_zoom, h, edgecolor='#d62728', facecolor='none', lw=2, zorder=10)
    ax_main.add_patch(rect1)
    ax_main.add_patch(rect2)

    # 2. Micro View 1
    ax_zoom1.plot(t_head, sig_head, color='#1f77b4', lw=1)
    ax_zoom1.set_title(f"Start Segment (0.0 - {duration_zoom}s)", fontsize=FONT_CONFIG['label_size'], color='#1f77b4')
    ax_zoom1.set_xlabel("Time (s)", fontsize=FONT_CONFIG['label_size'])
    ax_zoom1.set_ylabel("Amplitude", fontsize=FONT_CONFIG['label_size'])
    ax_zoom1.grid(True, alpha=0.3)

    # 3. Micro View 2
    ax_zoom2.plot(t_tail, sig_tail, color='#d62728', lw=1)
    ax_zoom2.set_title(f"End Segment ({t_tail[0]:.2f} - {t_tail[-1]:.2f}s)", fontsize=FONT_CONFIG['label_size'],
                       color='#d62728')
    ax_zoom2.set_xlabel("Time (s)", fontsize=FONT_CONFIG['label_size'])
    ax_zoom2.grid(True, alpha=0.3)

    # 4. Connections
    con1 = ConnectionPatch(xyA=(duration_zoom / 2, y_min), coordsA=ax_main.transData,
                           xyB=(duration_zoom / 2, max(sig_head)), coordsB=ax_zoom1.transData,
                           color='#1f77b4', alpha=0.2, lw=2)
    con2 = ConnectionPatch(xyA=(t_tail[0] + duration_zoom / 2, y_min), coordsA=ax_main.transData,
                           xyB=(t_tail[0] + duration_zoom / 2, max(sig_tail)), coordsB=ax_zoom2.transData,
                           color='#d62728', alpha=0.2, lw=2)
    fig.add_artist(con1)
    fig.add_artist(con2)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Macro+Connected: {save_path}")


# --- 图类型 B: 典型波形 (Typical) ---
def plot_typical_waveform(signal, fs, dataset_name, save_name):
    """
    [新增] 绘制典型波形 (前 0.5s)，用于展示信号细节
    """
    plt.figure(figsize=(10, 4))

    # 限制展示长度 (0.5s)
    display_len = min(len(signal), int(0.5 * fs))
    sig_disp = signal[:display_len]

    duration = len(sig_disp) / fs
    t = np.linspace(0, duration, len(sig_disp))

    plt.plot(t, sig_disp, color='#333333', lw=0.8)

    plt.title(f"{dataset_name} - Typical Waveform (First {duration:.2f}s)", fontsize=FONT_CONFIG['title_size'])
    plt.xlabel("Time (s)", fontsize=FONT_CONFIG['label_size'])
    plt.ylabel("Amplitude", fontsize=FONT_CONFIG['label_size'])
    plt.grid(True, alpha=0.3)
    plt.xlim(0, duration)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Typical: {save_path}")


# --- 图类型 C: 前后漂移对比 (Early vs Late) ---
def plot_drift_comparison(signal, fs, dataset_name, save_name):
    """
    [新增] 漂移对比图：对比文件头 (Early) vs 文件尾 (Late)
    严格统一 Y 轴，分为上下两个子图
    """
    n = len(signal)
    crop_len = int(0.1 * fs)  # 0.1秒片段

    # Early: 文件开始
    s1 = signal[:crop_len]
    # Late: 文件末尾
    s2 = signal[-crop_len:]

    t1 = np.linspace(0, len(s1) / fs, len(s1))
    t2 = np.linspace(0, len(s2) / fs, len(s2))

    # 统一 Y 轴刻度，确保视觉对比公正
    y_min = min(s1.min(), s2.min())
    y_max = max(s1.max(), s2.max())
    margin = (y_max - y_min) * 0.1
    y_lim = (y_min - margin, y_max + margin)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Subplot 1: Early
    ax1.plot(t1, s1, color='#1f77b4', lw=1)
    ax1.set_title(f"{dataset_name} - Early Stage (Start of Recording)", fontsize=13, fontweight='bold', color='#1f77b4')
    ax1.set_ylabel("Amplitude", fontsize=FONT_CONFIG['label_size'])
    ax1.set_ylim(y_lim)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Late
    ax2.plot(t2, s2, color='#d62728', lw=1)
    ax2.set_title(f"{dataset_name} - Late Stage (End of Recording)", fontsize=13, fontweight='bold', color='#d62728')
    ax2.set_xlabel("Time (s)", fontsize=FONT_CONFIG['label_size'])
    ax2.set_ylabel("Amplitude", fontsize=FONT_CONFIG['label_size'])
    ax2.set_ylim(y_lim)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Drift Comparison: {save_path}")


# ================= 3. 主程序 =================

if __name__ == "__main__":

    # --- 1. 处理 CWRU (12k Hz) ---
    print("\n--- Processing CWRU ---")
    sig_cwru = read_raw_cwru(fault_code=1)
    if sig_cwru is not None:
        # A. 原版带连线宏观图
        plot_macro_with_connections(sig_cwru, 12000, "CWRU", "CWRU_Macro_Connected.png")
        # B. 典型波形图
        plot_typical_waveform(sig_cwru, 12000, "CWRU", "CWRU_Typical.png")
        # C. 前后漂移对比图
        plot_drift_comparison(sig_cwru, 12000, "CWRU", "CWRU_Drift_Compare.png")

    # --- 2. 处理 JNU (51.2k Hz) ---
    print("\n--- Processing JNU ---")
    JNU_ROOT = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'
    sig_jnu = read_raw_jnu(JNU_ROOT, speed='600')
    if sig_jnu is not None:
        # A. 原版带连线宏观图
        plot_macro_with_connections(sig_jnu, 51200, "JNU", "JNU_Macro_Connected.png")
        # B. 典型波形图
        plot_typical_waveform(sig_jnu, 51200, "JNU", "JNU_Typical.png")
        # C. 前后漂移对比图
        plot_drift_comparison(sig_jnu, 51200, "JNU", "JNU_Drift_Compare.png")

    print(f"\n🎉 All tasks finished. Check folder: {os.path.abspath(SAVE_DIR)}")