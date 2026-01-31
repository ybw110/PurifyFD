import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# 导入你的数据加载器
import dataset_spilt_CWRU as cwru_loader
import dataset_spilt_jnu as jnu_loader
from config import Config

# ================= 1. 全局配置与参数 =================
cfg = Config()

# --- 采样率配置 ---
FS_CWRU = 12000  # CWRU 驱动端
FS_JNU = 51200  # JNU 数据集

# --- 绘图保存路径 ---
SAVE_ROOT = './Visualizations'
sub_dirs = ['CWRU_Individual', 'JNU_Individual', 'Comparisons']
for d in sub_dirs:
    os.makedirs(os.path.join(SAVE_ROOT, d), exist_ok=True)

# --- 故障类型索引 ---
TARGET_FAULT_IDX = 1
FAULT_NAME = "Inner Race Fault"

# ================= [新增] 字体与尺寸统一配置 =================
# 在这里修改数值，会应用到所有图片
FONT_CONFIG = {
    'font_family': 'Arial',  # 字体名称
    'title_size': 14,  # 标题字体大小
    'label_size': 12,  # 坐标轴名称(xlabel/ylabel)字体大小
    'tick_size': 10,  # 坐标刻度数值字体大小
    'legend_size': 10,  # 图注字体大小
    'annotate_size': 9  # 峰值标注字体大小
}

# 应用全局字体设置
plt.rcParams['font.family'] = FONT_CONFIG['font_family']
# 如果系统中没有Arial，作为回退防止报错
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ================= 2. 核心工具函数 =================

def get_one_sample(loader, label_idx):
    """从 Loader 获取单个样本 (1024 points)"""
    for segments, labels, _, _, _, _ in loader:
        mask = (labels == label_idx)
        if mask.sum() > 0:
            return segments[mask][0].squeeze().numpy()
    return None


def compute_fft(sig, fs):
    """计算 FFT 频谱"""
    N = len(sig)
    yf = fft(sig)
    xf = fftfreq(N, 1 / fs)[:N // 2]
    amp = 2.0 / N * np.abs(yf[0:N // 2])
    return xf, amp


def find_top_peaks(xf, amp, top_k=3, height_ratio=0.1):
    """寻找频谱中的显著峰值"""
    max_val = np.max(amp)
    peaks, properties = find_peaks(amp, height=max_val * height_ratio, distance=10)

    if len(peaks) == 0:
        return [], []

    peak_heights = properties['peak_heights']
    sorted_indices = np.argsort(peak_heights)[::-1][:top_k]
    real_peak_indices = peaks[sorted_indices]
    return xf[real_peak_indices], amp[real_peak_indices]


# ================= 3. 绘图功能模块 (已应用字体配置) =================

def apply_plot_style(ax, title, xlabel, ylabel):
    """统一应用样式设置"""
    ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='medium')
    ax.set_xlabel(xlabel, fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['label_size'])
    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick_size'])
    ax.grid(True, alpha=0.3)


def plot_single_time_domain(sig, fs, title, save_path=None, ax=None):
    """绘制单张时域图"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        is_standalone = True
    else:
        is_standalone = False

    duration = len(sig) / fs
    t = np.linspace(0, duration, len(sig))

    ax.plot(t, sig, color='#1f77b4', linewidth=1.0)
    ax.set_xlim(0, duration)

    apply_plot_style(ax, f"{title} - Time Domain", "Time (s)", "Amplitude (Norm)")

    if is_standalone and save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_single_spectrogram(sig, fs, title, save_path=None, ax=None):
    """绘制单张时频图"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        is_standalone = True
    else:
        is_standalone = False

    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=256, noverlap=128)

    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')

    ax.set_title(f"{title} - Spectrogram", fontsize=FONT_CONFIG['title_size'])
    ax.set_xlabel("Time (s)", fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel("Frequency (Hz)", fontsize=FONT_CONFIG['label_size'])
    ax.tick_params(axis='both', labelsize=FONT_CONFIG['tick_size'])

    if is_standalone:
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Power (dB)', fontsize=FONT_CONFIG['label_size'])
        cbar.ax.tick_params(labelsize=FONT_CONFIG['tick_size'])

    if is_standalone and save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_spectrum_with_peaks(sig, fs, title, color='tab:blue', save_path=None, ax=None):
    """绘制带标注的频谱图"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        is_standalone = True
    else:
        is_standalone = False

    xf, amp = compute_fft(sig, fs)

    ax.plot(xf, amp, color=color, linewidth=1.2)

    # 峰值标注
    peak_x, peak_y = find_top_peaks(xf, amp, top_k=2, height_ratio=0.15)
    print(f"[{title}] Peaks: ", end="")
    for px, py in zip(peak_x, peak_y):
        print(f"{px:.1f}Hz ", end="")
        ax.annotate(f'{px:.0f}Hz', xy=(px, py), xytext=(px, py * 1.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    horizontalalignment='center', fontsize=FONT_CONFIG['annotate_size'])
    print("")  # Newline

    ax.set_ylim(0, max(amp) * 1.4)
    apply_plot_style(ax, f"{title} - FFT Spectrum", "Frequency (Hz)", "Amplitude")

    if is_standalone and save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


# ================= 4. 批量处理逻辑 =================

def process_cwru_visualization(cwru_data):
    print("\n>>> Visualizing CWRU Data...")
    domains = ['0HP', '1HP', '2HP', '3HP']

    fig_comb, axes_comb = plt.subplots(len(domains), 2, figsize=(12, 3 * len(domains)))
    fig_comb.suptitle(f"CWRU {FAULT_NAME} - Overview", fontsize=FONT_CONFIG['title_size'] + 2)

    for i, domain in enumerate(domains):
        loader = cwru_data[domain]['test']
        sig = get_one_sample(loader, TARGET_FAULT_IDX)
        if sig is None: continue

        # 1. 保存单图
        plot_single_time_domain(sig, FS_CWRU, f"CWRU {domain}",
                                save_path=os.path.join(SAVE_ROOT, 'CWRU_Individual', f'{domain}_Time.png'))
        plot_single_spectrogram(sig, FS_CWRU, f"CWRU {domain}",
                                save_path=os.path.join(SAVE_ROOT, 'CWRU_Individual', f'{domain}_Spectrogram.png'))

        # 2. 绘制总图
        plot_single_time_domain(sig, FS_CWRU, f"{domain}", ax=axes_comb[i, 0])
        plot_single_spectrogram(sig, FS_CWRU, f"{domain}", ax=axes_comb[i, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, 'CWRU_Combined_Overview.png'), dpi=300)
    plt.close()
    print("✅ CWRU Done.")


def process_jnu_visualization(jnu_data):
    """[新增] JNU 数据集的批量可视化"""
    print("\n>>> Visualizing JNU Data...")
    # JNU 的工况通常是转速
    speeds = ['600', '800', '1000']

    fig_comb, axes_comb = plt.subplots(len(speeds), 2, figsize=(12, 3 * len(speeds)))
    fig_comb.suptitle(f"JNU {FAULT_NAME} - Overview", fontsize=FONT_CONFIG['title_size'] + 2)

    for i, speed in enumerate(speeds):
        loader = jnu_data[speed]['test']
        sig = get_one_sample(loader, TARGET_FAULT_IDX)

        if sig is None:
            print(f"⚠️ Warning: No sample found for JNU {speed}")
            continue

        label_name = f"{speed} RPM"

        # 1. 保存单图
        plot_single_time_domain(sig, FS_JNU, f"JNU {label_name}",
                                save_path=os.path.join(SAVE_ROOT, 'JNU_Individual', f'{speed}_Time.png'))
        plot_single_spectrogram(sig, FS_JNU, f"JNU {label_name}",
                                save_path=os.path.join(SAVE_ROOT, 'JNU_Individual', f'{speed}_Spectrogram.png'))

        # 2. 绘制总图
        plot_single_time_domain(sig, FS_JNU, f"{label_name}", ax=axes_comb[i, 0])
        plot_single_spectrogram(sig, FS_JNU, f"{label_name}", ax=axes_comb[i, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, 'JNU_Combined_Overview.png'), dpi=300)
    plt.close()
    print("✅ JNU Done.")


def process_comparison_visualization(cwru_data, jnu_data):
    print("\n>>> Generating Comparison Plots...")
    sig_cwru = get_one_sample(cwru_data['0HP']['test'], TARGET_FAULT_IDX)
    sig_jnu = get_one_sample(jnu_data['600']['test'], TARGET_FAULT_IDX)

    if sig_cwru is None or sig_jnu is None: return

    # 单图保存
    save_c = os.path.join(SAVE_ROOT, 'Comparisons', 'Spectrum_CWRU_0HP.png')
    plot_spectrum_with_peaks(sig_cwru, FS_CWRU, "Rig A (CWRU 0HP)", color='#1f77b4', save_path=save_c)

    save_j = os.path.join(SAVE_ROOT, 'Comparisons', 'Spectrum_JNU_600RPM.png')
    plot_spectrum_with_peaks(sig_jnu, FS_JNU, "Rig B (JNU 600RPM)", color='#d62728', save_path=save_j)

    # 合并图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_spectrum_with_peaks(sig_cwru, FS_CWRU, "Rig A (CWRU - 0HP)", color='#1f77b4', ax=ax1)
    plot_spectrum_with_peaks(sig_jnu, FS_JNU, "Rig B (JNU - 600RPM)", color='#d62728', ax=ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, 'Comparisons', 'Comparison_CWRU_vs_JNU.png'), dpi=300)
    plt.close()
    print("✅ Comparison Done.")


# ================= 5. 主程序 =================

if __name__ == "__main__":
    # 加载数据
    cwru_full = cwru_loader.prepare_all_domains_detailed()

    # 路径确认
    jnu_path = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'
    jnu_full = jnu_loader.prepare_jnu_detailed(jnu_path)

    # 执行可视化
    process_cwru_visualization(cwru_full)
    process_jnu_visualization(jnu_full)  # 新增的函数调用
    process_comparison_visualization(cwru_full, jnu_full)

    print(f"\n🎉 All images saved to: {os.path.abspath(SAVE_ROOT)}")