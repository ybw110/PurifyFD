import matplotlib.pyplot as plt
import numpy as np
import os  # <--- 1. 引入 os 模块用于管理文件夹


plt.rcParams['font.family'] = 'Arial'

# 定义保存图片的文件夹名称
output_folder = 'pareto_charts'

# 如果文件夹不存在，则创建它 (exist_ok=True 表示如果已存在也不报错)
os.makedirs(output_folder, exist_ok=True)
print(f"图片将保存至文件夹: ./{output_folder}/")


# === 1. 最新数据 ===
methods = ['CNN', 'Mixup', 'MixStyle', 'AMINet', 'DACN', 'PurifyFD']

# Y轴: 性能 (Type C Avg AUROC)
auroc_scores = [48.97, 48.37, 47.09, 40.13, 45.80, 63.60]

# X轴-1: 时间 (Latency ms) - 已更新
# 对应: [CNN, Mixup, MixStyle, AMINet, DACN, PurifyFD]
latency = [0.1850, 0.2324, 0.2073, 1.8882, 8.3874, 1.7462]

# X轴-2: 能耗 (FLOPs M) - 保持原数据
flops = [0.8545, 0.8545, 0.8545, 27.7594, 60.0973, 19.6309]

# 气泡大小依据: 存储 (Params M) - 保持原数据
params = [0.0344, 0.0344, 0.0344, 1.1504, 34.4531, 0.5553]

# 统一配色
colors = ['#5D8CA8', '#F5E1CC', '#CDD192', '#E3D8DE', '#98C0DE', '#D99488']

DOMAIN_COLORS = [
    '#5D8CA8',  # SS-A
    '#E0AC69',  # SS-B
    '#D99488',  # SS-C
    '#AE93A3',  # ST-D
    '#98C0DE',  # ST-1
    '#CDD192',  # ST-2
    '#C3D3E0',  # SS-A Light
    '#F5E1CC',  # SS-B Light
    '#F0D3CE',  # SS-C Light
    '#E3D8DE',  # ST-D Light
    '#DCEAF2',  # ST-1 Light
    '#E9EBCF'  # ST-2 Light
]

# === 2. 气泡大小计算函数 ===
def get_bubble_size(param_val):
    if param_val > 10: return 1800  # DACN
    elif param_val > 1: return 1200  # AMINet
    elif param_val > 0.1: return 600  # PurifyFD
    else: return 300  # CNNs

sizes = [get_bubble_size(p) for p in params]

# ==========================================
# 图 1: Latency vs AUROC (时间效率)
# ==========================================
plt.figure(figsize=(5.5, 5))  # 稍微加宽以容纳图例
plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

for i, method in enumerate(methods):
    plt.scatter(latency[i], auroc_scores[i], s=sizes[i], c=colors[i], label=method,
                alpha=0.85, edgecolors='black', marker='o', zorder=10)

# --- 修改核心：使用图例代替文本标签 ---
# frameon=True: 图例有边框
# shadow=True: 图例有阴影
leg = plt.legend(loc='lower right', title="Methods", fontsize=16,
                 title_fontsize=16,
                 # frameon=True, fancybox=True, shadow=True
                 )

# 强制统一图例中圆点的大小 (否则图例里的圆圈会忽大忽小)
for handle in leg.legend_handles:
    handle.set_sizes([100.0])

# 标注 PurifyFD (Pareto) - 保留分析性标注，但去掉模型名
plt.annotate('Pareto Optimal\n(High Acc)',
             xy=(1.7462, 63.6),  # 修正：箭头尖端指向新的 Latency 数据点
             xytext=(2.5, 55),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5),
             fontsize=16, fontweight='normal', color='#b57979', zorder=20)

# --- 添加箭头标注 (Annotation) ---
# text: 显示的文字
# xy: 箭头尖端指向的坐标 (Latency=11.02, AUROC=63.6)
# xytext: 文字放置的坐标
# arrowprops: 箭头的样式 (黑色，收缩0.05，半透明)

plt.xlabel('Inference Latency (ms) ', fontsize=20)
plt.ylabel('OOD Robustness (AUROC %)', fontsize=20)
# plt.title('Time Efficiency: Latency vs. Robustness', fontsize=16)
plt.xlim(-1, 14) # 稍微扩大范围以防图例遮挡
plt.ylim(35, 70)
plt.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
# --- 2. 修改保存路径 ---
# 使用 os.path.join 拼接文件夹和文件名
save_path_pdf = os.path.join(output_folder, 'pareto_latency_legend.pdf')
save_path_png = os.path.join(output_folder, 'pareto_latency_legend.png')

plt.savefig(save_path_pdf, dpi=300)
plt.savefig(save_path_png, dpi=300)
print(f"图1已保存: {save_path_png}")

plt.show()

# ==========================================
# 图 2: FLOPs vs AUROC (能耗效率)
# ==========================================
plt.figure(figsize=(5.5, 5))
plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

for i, method in enumerate(methods):
    plt.scatter(flops[i], auroc_scores[i], s=sizes[i], c=colors[i], label=method,
                alpha=0.85, edgecolors='black', marker='o', zorder=10)

# --- 修改核心：使用图例 ---
leg = plt.legend(loc='upper right', title="Methods", fontsize=16,
                 title_fontsize=16
                 # , frameon=True, fancybox=True, shadow=True
                 )

for handle in leg.legend_handles:
    handle.set_sizes([100.0])

# 标注 DACN (High Energy)
plt.annotate('High Energy Cost\n(60M FLOPs)',
             xy=(60.1, 45.8), xytext=(40, 38),
             arrowprops=dict(facecolor='#9f9f9f', shrink=0.05, alpha=0.5),
             fontsize=16, color='gray', zorder=20)

# 标注 PurifyFD (Energy Efficient)
plt.annotate('Energy Efficient\n(Moderate FLOPs)',
             xy=(19.6, 63.6), xytext=(10, 55),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5),
             fontsize=16, fontweight='normal', color='#b57979', zorder=20)

plt.xlabel('Computational Cost (M FLOPs)', fontsize=20)
plt.ylabel('OOD Robustness (AUROC %)', fontsize=20)
# plt.title('Energy Efficiency: FLOPs vs. Robustness', fontsize=16)
plt.xlim(-5, 75)
plt.ylim(35, 70)
plt.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
# --- 2. 修改保存路径 ---
save_path_pdf_2 = os.path.join(output_folder, 'pareto_flops_legend.pdf')
save_path_png_2 = os.path.join(output_folder, 'pareto_flops_legend.png')

plt.savefig(save_path_pdf_2, dpi=300)
plt.savefig(save_path_png_2, dpi=300)
print(f"图2已保存: {save_path_png_2}")

plt.show()