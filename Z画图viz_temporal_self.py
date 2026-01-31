import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config

cfg = Config()

# ================= 全局配置 =================
SAVE_DIR = './Visualizations_Self/Temporal_Shift_Separate'  # 修改保存路径以示区别
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 100000  # 100kHz
TARGET_SPEED = '1200'
TARGET_LABEL = '中等'

# SelfA -> SelfB 标签映射
LABEL_MAPPING_A_TO_B = {
    '正常': '正常',
    '轻微': '轻度',
    '中等': '中度',
    '严重': '重度'
}

# 绘图字体设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 底层读取工具 (保持 CSV 差异性) =================

def read_self_a_csv(file_path):
    """SelfA: 带表头，取第2列"""
    try:
        df = pd.read_csv(file_path, header=0)
        col_idx = 1 if df.shape[1] >= 2 else 0
        return df.iloc[:, col_idx].values.astype(np.float32)
    except:
        return None


def read_self_b_csv(file_path):
    """SelfB: 无标准表头(或需跳过)，取第2列"""
    try:
        df = pd.read_csv(file_path, skiprows=1, header=None, usecols=[1])
        return df.iloc[:, 0].values.astype(np.float32)
    except:
        return None


# ================= 2. 文件查找与加载策略 =================

def get_files_self_a():
    """找到 SelfA 的所有 CSV 文件路径"""
    folder_name = f"{TARGET_LABEL}-{TARGET_SPEED}"
    # 优先找 raw_data
    target_dir = os.path.join(cfg.RAW_DATA_ROOT_A, folder_name, 'raw_data')
    if not os.path.exists(target_dir):
        target_dir = os.path.join(cfg.RAW_DATA_ROOT_A, folder_name)

    if not os.path.exists(target_dir):
        print(f"❌ SelfA dir not found: {target_dir}")
        return []

    return sorted(glob.glob(os.path.join(target_dir, "*.csv")))


def get_files_self_b():
    """找到 SelfB 的所有 CSV 文件路径 (搜索 train/val/test)"""
    label_b = LABEL_MAPPING_A_TO_B.get(TARGET_LABEL, TARGET_LABEL)
    folder_name = f"{label_b}-{TARGET_SPEED}"

    all_files = []
    # 只需要找到一个存在的文件夹即可，通常 train 里的文件最多
    for sub in ['train', 'val', 'test']:
        target_dir = os.path.join(cfg.RAW_DATA_ROOT_B, sub, folder_name, 'raw_data')
        if os.path.exists(target_dir):
            files = sorted(glob.glob(os.path.join(target_dir, "*.csv")))
            if len(files) > 0:
                print(f"✅ Found SelfB files in: {target_dir}")
                return files  # 找到一组就返回，避免混淆

    print(f"❌ SelfB dir not found for {folder_name}")
    return []


def load_data_for_viz(dataset_name):
    """
    加载用于可视化的数据
    返回: (sig_typical, sig_early, sig_late, file_info)
    """
    if dataset_name == 'SelfA':
        files = get_files_self_a()
        reader = read_self_a_csv
    else:
        files = get_files_self_b()
        reader = read_self_b_csv

    if not files:
        return None, None, None, "No Files"

    # 1. Typical Sample (取第1个文件，保证绝对连续)
    sig_typical = reader(files[0])

    # 2. Drift Samples (对比 Early vs Late)
    # Early: 第1个文件
    sig_early = sig_typical

    # Late: 第50个文件 (如果不够50，就取最后一个)
    # 注意：这里取 50 是为了看时间跨度上的变化
    late_idx = 600 if len(files) > 50 else len(files) - 1
    sig_late = reader(files[late_idx])

    info_str = f"Total Files: {len(files)}. Early=File 0, Late=File {late_idx}."
    print(f"[{dataset_name}] {info_str}")

    return sig_typical, sig_early, sig_late, info_str


# ================= 3. 绘图逻辑 (修改版) =================

def plot_typical_waveform(signal, fs, dataset_name, save_name):
    """
    画单个文件的典型波形 (不做任何拼接)
    长度自适应：文件多长画多长
    """
    plt.rcParams['font.family'] = 'Arial'

    plt.figure(figsize=(12, 4))

    duration = len(signal) / fs
    t = np.linspace(0, duration, len(signal))

    plt.plot(t, signal, color='#CBCBCB',
             linewidth=1,
             linestyle='-',
             alpha=0.9)

    # plt.title(f"{dataset_name} - Typical Single-File Waveform ({duration:.2f}s)", fontsize=14)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14, colors='black')

    plt.grid(True, alpha=0.3)
    plt.xlim(0, duration)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Typical: {save_path}")


def plot_short_segment(signal, fs, dataset_name, stage_name, save_name, duration_sec=0.01):
    """
    画单个信号指定时长的片段，并根据 Stage 定制坐标轴和时间偏移
    """
    crop_len = int(duration_sec * fs)
    sig_segment = signal[:min(len(signal), crop_len)]
    actual_duration = len(sig_segment) / fs

    # --- 核心逻辑修改：根据 Stage 设置起始时间和刻度 ---
    if stage_name == 'Late':
        start_time = 0.59  # Late 强制从 0.99s 开始
        custom_ticks = [0.59, 0.595, 0.60]
        line_style = '-'  # Late 为虚线
    else:
        start_time = 0.00  # Early 从 0.00s 开始
        custom_ticks = [0.00, 0.005, 0.01]
        line_style = '-'  # Early 为实线

    # 生成偏移后的时间轴，确保数据能落在 0.99-1.00 区间内显示
    t = np.linspace(start_time, start_time + actual_duration, len(sig_segment))

    plt.figure(figsize=(6,3))

    color_map = {'Early': '#E59693', 'Late': '#BBD5E7'}
    plot_color = color_map.get(stage_name, 'black')

    plt.plot(t, sig_segment,
             color=plot_color,
             linewidth=0.5,  # 稍微加粗使虚线更清晰
             linestyle=line_style,
             alpha=0.9)

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)

    # 应用自定义刻度
    plt.xticks(custom_ticks)
    plt.tick_params(axis='both', which='major', labelsize=14, colors='black')

    plt.grid(True, alpha=0.3)
    plt.xlim(start_time, start_time + actual_duration)  # 锁定范围

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved {stage_name} segment (Range: {start_time}-{start_time + actual_duration:.2f}s): {save_path}")


# ================= 4. 主执行块 =================

if __name__ == "__main__":
    print(f"Target Condition: {TARGET_SPEED} RPM - {TARGET_LABEL}")
    print("-" * 50)

    # --- Process SelfA ---
    sA_typ, sA_early, sA_late, infoA = load_data_for_viz('SelfA')
    if sA_typ is not None:
        # 1. 画单个典型波形 (全长)
        plot_typical_waveform(sA_typ, FS, "SelfA (Rig 1)", "SelfA_Typical_Full.png")
        # 2. 画 Early 前 0.1s 独立图
        plot_short_segment(sA_early, FS, "SelfA (Rig 1)", "Early", "SelfA_Early_0.1s.png")
        # 3. 画 Late 前 0.1s 独立图
        plot_short_segment(sA_late, FS, "SelfA (Rig 1)", "Late", "SelfA_Late_0.1s.png")

    # --- Process SelfB ---
    print("-" * 50)
    sB_typ, sB_early, sB_late, infoB = load_data_for_viz('SelfB')
    if sB_typ is not None:
        # 1. 画单个典型波形 (全长，约0.3s)
        plot_typical_waveform(sB_typ, FS, "SelfB (Rig 6)", "SelfB_Typical_Full.png")
        # 2. 画 Early 前 0.1s 独立图
        plot_short_segment(sB_early, FS, "SelfB (Rig 6)", "Early", "SelfB_Early_0.1s.png")
        # 3. 画 Late 前 0.1s 独立图
        plot_short_segment(sB_late, FS, "SelfB (Rig 6)", "Late", "SelfB_Late_0.1s.png")

    print("\n🎉 Visualization Completed!")
    print(f"Images saved to: {os.path.abspath(SAVE_DIR)}")