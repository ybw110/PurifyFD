import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import signal
from scipy.fft import fft, fftfreq
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 用于局部放大图

# ================= 导入自定义数据加载器 =================
# 请确保 dataset_spilt_selfA.py, dataset_spilt_selfB.py 和 config.py 在同一目录下
import dataset_spilt_selfA as selfa_loader
import dataset_spilt_selfB as selfb_loader
from config import Config

# ================= 1. 全局配置 =================
cfg = Config()

# --- 采样率 (根据您的实验环境) ---
FS_SELFA = 100000  # 100 kHz
FS_SELFB = 100000  # 100 kHz

# --- 绘图保存路径 ---
SAVE_ROOT = './Visualizations_Self'
sub_dirs = ['SelfA_Individual', 'SelfB_Individual', 'Comparisons_Self']
for d in sub_dirs:
    os.makedirs(os.path.join(SAVE_ROOT, d), exist_ok=True)

# --- 目标故障类型 ---
# 0:正常, 1:轻微, 2:中等, 3:严重
TARGET_FAULT_IDX = 3
FAULT_NAME = "Severe Fault"

# --- 转速列表 ---
SPEEDS = ['1200', '1800']

# --- 图表全局样式 ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


# ================= 2. 核心计算函数 =================

def get_one_sample(loader, label_idx):
    """从 DataLoader 中提取指定标签的单个样本"""
    for segments, labels, _, _, _, _ in loader:
        mask = (labels == label_idx)
        if mask.sum() > 0:
            # 提取第一个符合条件的样本，并转为 numpy
            return segments[mask][0].squeeze().numpy()
    return None


def compute_fft(sig, fs):
    """计算 FFT 频谱"""
    N = len(sig)
    yf = fft(sig)
    xf = fftfreq(N, 1 / fs)[:N // 2]
    amp = 2.0 / N * np.abs(yf[0:N // 2])
    return xf, amp


# ================= 3. 绘图功能模块 =================

def plot_time_and_spectrogram(sig, fs, title, save_path):
    """
    绘制单样本的 [时域图 + 时频图] 概览
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 时域图 ---
    duration = len(sig) / fs
    t = np.linspace(0, duration, len(sig))
    ax1.plot(t, sig, color='#ff7f0e', linewidth=0.8)
    ax1.set_title(f"{title} - Time Domain", fontsize=12)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, duration)
    ax1.grid(True, alpha=0.3)

    # --- 时频图 ---
    f, t_spec, Sxx = signal.spectrogram(sig, fs=fs, nperseg=512, noverlap=256)
    pcm = ax2.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='plasma')
    ax2.set_title(f"{title} - Spectrogram", fontsize=12)
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    fig.colorbar(pcm, ax=ax2, label='Power (dB)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_aesthetic_comparison(sig_a, fs_a, sig_b, fs_b, title, save_path=None):
    """
    【美化版对比图】
    包含：截断频率范围、半透明填充、局部放大子图
    """
    # 1. 计算 FFT
    xf_a, amp_a = compute_fft(sig_a, fs_a)
    xf_b, amp_b = compute_fft(sig_b, fs_b)

    # 2. 数据截断：只展示 0 - 6000 Hz (有效机械故障频率通常在此范围内)
    #    去掉 50kHz 处的空白，使图像更紧凑
    MAX_FREQ_DISPLAY = 5000

    mask_a = xf_a <= MAX_FREQ_DISPLAY
    mask_b = xf_b <= MAX_FREQ_DISPLAY

    xf_a_disp, amp_a_disp = xf_a[mask_a], amp_a[mask_a]
    xf_b_disp, amp_b_disp = xf_b[mask_b], amp_b[mask_b]

    # --- 开始绘图 ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # 3. 绘制 SelfB (红色，低频主导)
    ax.plot(xf_b_disp, amp_b_disp,
            color='#E59693',
            linewidth=2,
            linestyle='-',
            alpha=0.9,
            label='SelfB')
    ax.fill_between(xf_b_disp, 0, amp_b_disp, color='#d62728', alpha=0.1)  # 红色填充

    # 4. 绘制 SelfA (蓝色，高频主导)
    ax.plot(xf_a_disp, amp_a_disp,
            color='#BBD5E7',
            linewidth=2,
            linestyle='--',
            alpha=0.8,
            label='SelfA')
    ax.fill_between(xf_a_disp, 0, amp_a_disp, color='#1f77b4', alpha=0.05)  # 蓝色极淡填充

    ax.tick_params(axis='both', which='major', labelsize=20)  # <--- 修改这里：数字变大
    # 5. 自动标注最大峰值
    # B 的最大值 (通常在低频)
    idx_b = np.argmax(amp_b_disp)
    ax.annotate(f'{xf_b_disp[idx_b]:.0f}Hz (Max B)',
                xy=(xf_b_disp[idx_b], amp_b_disp[idx_b]),
                xytext=(xf_b_disp[idx_b] + 300, amp_b_disp[idx_b]),
                arrowprops=dict(facecolor='#d62728', arrowstyle='->'),
                fontsize=18, color='#d62728', fontweight='bold')

    # A 的最大值 (通常在高频)
    idx_a = np.argmax(amp_a_disp)
    ax.annotate(f'{xf_a_disp[idx_a]:.0f}Hz (Max A)',
                xy=(xf_a_disp[idx_a], amp_a_disp[idx_a]),
                xytext=(xf_a_disp[idx_a], amp_a_disp[idx_a] + 0.2),  # 稍微往上提一点
                arrowprops=dict(facecolor='#1f77b4', arrowstyle='->'),
                fontsize=18, color='#1f77b4', fontweight='bold')

    # --- 6. 嵌入子图 (Inset Zoom) ---
    # 专门放大 2000Hz - 4000Hz 区域，展示 SelfA 的细节
    axins = inset_axes(ax, width="45%", height="35%", loc='upper center', borderpad=6)

    # 定义放大区域
    ZOOM_START, ZOOM_END = 2000, 4000
    zoom_mask = (xf_a > ZOOM_START) & (xf_a < ZOOM_END)
    zoom_mask_b = (xf_b > ZOOM_START) & (xf_b < ZOOM_END)

    # 子图也要保持线型一致，看起来才专业
    axins.plot(xf_a[zoom_mask], amp_a[zoom_mask], color='#1f77b4', linestyle='--', linewidth=2)  # 保持虚线
    axins.plot(xf_b[zoom_mask_b], amp_b[zoom_mask_b], color='#d62728', linestyle='-', alpha=0.3, linewidth=2)  # 保持实线

    axins.set_title(f"Zoom Area: {ZOOM_START}-{ZOOM_END} Hz", fontsize=18)
    axins.tick_params(axis='both', labelsize=16)
    axins.grid(True, alpha=0.2, linestyle='--')
    # axins.set_facecolor('#f9f9f9')  # 子图背景微灰，区分层级

    # --- 样式调整 ---
    # ax.set_title(f"Comparison: Domain Shift in {FAULT_NAME} - 1200 RPM", fontsize=15, fontweight='bold', pad=15)
    # --- 标签设置 ---
    ax.set_xlabel("Frequency (Hz)", fontsize=22)  # <--- 轴标题字体变大
    ax.set_ylabel("Amplitude", fontsize=22)  # <--- 轴标题字体变大
    ax.legend(fontsize=20, loc='upper right')

    # 去掉不必要的边框 (Tufte Style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


# ================= 4. 批量处理流程 =================

def process_individual_dataset(data_dict, dataset_name, fs, save_subdir):
    """处理单个数据集的时域/时频图"""
    print(f"\n>>> Visualizing {dataset_name} ...")
    for speed in SPEEDS:
        loader = data_dict[speed]['test']
        sig = get_one_sample(loader, TARGET_FAULT_IDX)

        if sig is None:
            print(f"⚠️ Warning: No data for {dataset_name} {speed}")
            continue

        save_name = os.path.join(SAVE_ROOT, save_subdir, f'{dataset_name}_{speed}RPM.png')
        plot_time_and_spectrogram(sig, fs, f"{dataset_name} {speed}RPM", save_name)
    print(f"✅ {dataset_name} Done.")


def run_comparison_analysis(data_a, data_b):
    """运行对比分析"""
    print("\n>>> Generating Comparison (SelfA vs SelfB)...")

    # 选取 1200 RPM 进行对比
    target_speed = '1200'

    sig_a = get_one_sample(data_a[target_speed]['test'], TARGET_FAULT_IDX)
    sig_b = get_one_sample(data_b[target_speed]['test'], TARGET_FAULT_IDX)

    if sig_a is None or sig_b is None:
        print("Error: Missing data for comparison.")
        return

    # 保存路径
    save_path = os.path.join(SAVE_ROOT, 'Comparisons_Self', 'Final_Comparison_Aesthetic.png')

    # 调用美化版绘图
    plot_aesthetic_comparison(
        sig_a, FS_SELFA,
        sig_b, FS_SELFB,
        FAULT_NAME,
        save_path
    )
    print(f"✅ Aesthetic Comparison Saved to: {save_path}")


# ================= 5. 主程序入口 =================

if __name__ == "__main__":
    print("=== Starting Visualization Process ===")

    # 1. 加载数据 (这可能需要一点时间)
    print("loading SelfA data...")
    selfa_full = selfa_loader.prepare_self_detailed()

    print("loading SelfB data...")
    selfb_full = selfb_loader.prepare_self_b_detailed()

    # 2. 生成单个数据集的概览图
    # process_individual_dataset(selfa_full, "SelfA", FS_SELFA, 'SelfA_Individual')
    # process_individual_dataset(selfb_full, "SelfB", FS_SELFB, 'SelfB_Individual')

    # 3. 生成美化后的对比图 (解决不美观、数据差异大的问题)
    run_comparison_analysis(selfa_full, selfb_full)

    print("\n🎉 All visualizations completed.")