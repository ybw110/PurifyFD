import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# === t-SNE 库 ===
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# === 导入自定义模块 ===
# 确保 aminet_model.py 存在
from baseline_AMINet_model import FeatureExtractor, Classifier
from config import Config
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS

config = Config()

# 标签逆映射: 0,1,2 -> 0,1,3 (物理含义)
LABEL_MAP = {original: mapped for mapped, original in enumerate(config.known_classes)}
REVERSE_MAP = {mapped: original for original, mapped in LABEL_MAP.items()}


def calculate_fpr95(known_scores, unknown_scores):
    """
    手动计算 FPR @ TPR95
    known_scores: 已知类的异常分数 (值应该较低)
    unknown_scores: 未知类的异常分数 (值应该较高)
    """
    k = np.array(known_scores)
    u = np.array(unknown_scores)

    # 排序
    k.sort()

    # 设定阈值：使得 95% 的 Known 样本被正确保留 (分数 < 阈值)
    num_k = len(k)
    threshold_idx = int(num_k * 0.95)
    if threshold_idx >= num_k:
        threshold_idx = num_k - 1
    threshold = k[threshold_idx]

    # 计算 FPR: 有多少 Unknown 样本的分数 < 阈值 (被误判为 Known)
    fp_count = np.sum(u < threshold)
    fpr95 = fp_count / len(u)
    return fpr95


def calculate_ood_metrics(y_true, y_conf, unknown_classes, score_type='energy'):
    """
    计算 OOD 核心三指标
    """
    # 1. 构建二值标签：0=Known, 1=Unknown
    is_unknown = np.isin(y_true, unknown_classes).astype(int)

    if np.sum(is_unknown) == 0 or np.sum(1 - is_unknown) == 0:
        return None

    # 2. 计算异常分数 (Anomaly Score)
    # 目标: Known 样本分数低, Unknown 样本分数高

    if score_type == 'softmax':
        # Softmax: y_conf 是概率 (0~1)，越大越确信是 Known
        scores = 1.0 - y_conf

    elif score_type == 'energy':
        # Energy: y_conf 是 LogSumExp，越大越确信是 Known
        # 因此异常分数取负号即可
        scores = -y_conf

    else:
        scores = -y_conf  # 默认处理

    # 分离分数 (后续逻辑保持不变)
    known_scores = scores[is_unknown == 0]
    unknown_scores = scores[is_unknown == 1]

    # --- 指标计算 (完全不用改) ---
    auroc = roc_auc_score(is_unknown, scores)
    aupr_out = average_precision_score(is_unknown, scores, pos_label=1)
    fpr95 = calculate_fpr95(known_scores, unknown_scores)

    return {
        'AUROC': auroc,
        'AUPR_Out': aupr_out,
        'FPR95': fpr95
    }


import seaborn as sns  # 如果没有安装 seaborn，可以用 pip install seaborn，或者只用 matplotlib

def plot_confidence_distribution(y_true, y_conf, unknown_classes, task_name, save_dir):
    """
    绘制 ID 和 OOD 的置信度分数分布直方图
    """
    # 1. 区分 ID 和 OOD 数据
    is_unknown = np.isin(y_true, unknown_classes)

    id_scores = y_conf[~is_unknown]  # 已知类分数 (通常较高)
    ood_scores = y_conf[is_unknown]  # 未知类分数 (通常较低)

    # 如果缺乏某一类数据，跳过绘图
    if len(id_scores) == 0 or len(ood_scores) == 0:
        print(f"  [Plot] Skipping plot for {task_name}: missing ID or OOD samples.")
        return

    # 2. 设置绘图风格
    plt.figure(figsize=(6,4))
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.color': '0.9'})

    # 3. 绘制直方图 (使用 seaborn 的 histplot 或 kdeplot)

    sns.kdeplot(id_scores, color='#C4E0E8', label='Known (ID)',fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(ood_scores, color='#E09D94', label='Unknown (OOD)',fill=True, alpha=0.3, linewidth=2)

    # sns.histplot(id_scores, color='#6D95C3', label='ID', kde=True, stat="density", bins=30, alpha=0.5, element="step")
    # sns.histplot(ood_scores, color='#E09D94', label='OOD', kde=True, stat="density", bins=30, alpha=0.5, element="step")

    # 4. 装饰图表
    # plt.title(f"Confidence Score Distribution: {task_name}", fontsize=14)
    plt.xlabel("Confidence Score (Energy)", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.legend(loc='upper left', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # 5. 保存图片
    filename = f"dist_{task_name}.png"
    save_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭画布，防止内存泄漏
    print(f"  [Plot] Saved distribution plot to: {filename}")


def test_pipeline(G, C, test_loader, device='cuda', score_type='energy', T=1.0):
    """AMINet 开集测试核心函数"""
    G.eval()
    C.eval()
    y_true = []
    y_pred_mapped = []
    y_conf = []

    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data[0].to(device).float()
            labels = batch_data[1].to(device).long()

            # AMINet 推理: G -> C
            features = G(inputs)
            logits = C(features)

            # 获取预测类别
            preds_idx = torch.argmax(logits, dim=1)

            # 计算 Energy Score
            if score_type == 'energy':
                conf_scores = T * torch.logsumexp(logits / T, dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
                conf_scores, _ = torch.max(probs, dim=1)

            # 还原为物理标签
            current_preds_mapped = [REVERSE_MAP[idx.item()] for idx in preds_idx]

            y_true.extend(labels.cpu().numpy())
            y_pred_mapped.extend(current_preds_mapped)
            y_conf.extend(conf_scores.cpu().numpy())

    return np.array(y_true), np.array(y_pred_mapped), np.array(y_conf)

# ==========================================
#               主测试逻辑
# ==========================================
def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 打印测试配置
    print(f"Known Classes: {config.known_classes}")
    print(f"Unknown Classes: {config.unknown_classes}")

    # ================= 1. 准备测试数据 =================
    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

    # A. 单工况加载 (用于表 S1, B1, B2, B4)
    cwru_single = prepare_all_domains_detailed()
    jnu_single = prepare_jnu_detailed(jnu_root)

    # B. 混合工况加载 (用于 B3, B5, C1, C2)
    cwru_mixed = prepare_mixed_domains(ALL_DOMAINS)
    jnu_mixed = prepare_mixed_jnu_domains(ALL_SPEEDS, jnu_root)

    # ================= 2. 定义测试任务 =================
    TASKS = []

    # --- 表 S1: 单工况基准 ---
    TASKS.append(('S1_CWRU_0HP', cwru_single['0HP']['test']))
    TASKS.append(('S1_CWRU_3HP', cwru_single['3HP']['test']))
    TASKS.append(('S1_JNU_600', jnu_single['600']['test']))

    # --- 表 1: 跨工况 (重点关注你的 B-3 模型表现) ---
    TASKS.append(('B1_B3_Target_CWRU_3HP', cwru_single['3HP']['test']))  # 目标域是 3HP
    TASKS.append(('B2_Target_CWRU_0HP', cwru_single['0HP']['test']))  # 目标域是 0HP
    TASKS.append(('B4_B5_Target_JNU_1000', jnu_single['1000']['test']))  # 目标域是 1000

    # --- 表 2: 跨设备 (高光实验) ---
    TASKS.append(('C1_Target_JNU_All', jnu_mixed['test']))
    TASKS.append(('C2_Target_CWRU_All', cwru_mixed['test']))

    # OOD1
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD1/AMINet_Multi_0HP_1HP_2HP_20260125_231856/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD1/AMINet_Multi_0HP_1HP_2HP_3HP_20260126_114055/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD1/AMINet_Multi_600_800_1000_20260126_114456/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD1/AMINet_Multi_600_800_20260126_113512/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD1/AMINet_Single_0HP_20260125_231357/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD1/AMINet_Single_3HP_20260125_231259/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD1/AMINet_Single_600_20260125_231433/best_model.pth'

    # OOD2
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD2/AMINet_Multi_0HP_1HP_2HP_20260126_120815/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD2/AMINet_Multi_0HP_1HP_2HP_3HP_20260126_120023/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD2/AMINet_Multi_600_800_1000_20260126_115140/best_model.pth'
    MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD2/AMINet_Multi_600_800_20260126_121113/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD2/AMINet_Single_0HP_20260126_120342/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD2/AMINet_Single_3HP_20260126_120439/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/aminet_OOD2/AMINet_Single_600_20260126_120605/best_model.pth'




    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model path not found: {MODEL_PATH}")
        print("请在代码中修改 MODEL_PATH 变量为实际的 .pth 文件路径")
        return

    print(f"loading model from: {MODEL_PATH}")

    # 1. 初始化模型结构 (测试只需要 G 和 C)
    G = FeatureExtractor().to(device)
    C = Classifier(num_classes=config.num_classes).to(device)

    # 2. 加载权重 (注意：AMINet 保存的是字典 {'G': ..., 'C': ...})
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'G' in checkpoint and 'C' in checkpoint:
        G.load_state_dict(checkpoint['G'])
        C.load_state_dict(checkpoint['C'])
        print("✅ 成功加载 AMINet 分离权重 (G & C)")
    else:
        # 兼容性处理：如果通过其他方式保存的
        try:
            # 尝试直接加载 (假设只保存了整个 state_dict，虽然不推荐)
            G.load_state_dict(checkpoint)
            print("⚠️ 警告：检测到非标准权重格式，尝试直接加载到 G...")
        except:
            print("❌ 权重加载失败：pth 文件格式不匹配。请确保使用 aminet_train.py 生成的权重。")
            return

    G.eval()
    C.eval()

    # 准备结果保存目录
    results_dir = os.path.dirname(MODEL_PATH)

    # ================= 5. 开始测试循环 =================

    # === 设置使用的分数类型 ===
    SCORE_TYPE = 'energy'  # 修改为 energy
    TEMP = 1.0  # Energy Temperature (通常设为1.0即可，也可尝试其他值如 5.0)

    results = []
    txt_report_content = ""

    for task_name, loader in TASKS:
        print(f"\n--- Task: {task_name} ---")
        task_block_str = f"\n--- Task: {task_name} ---\n"

        # 推理
        y_true, y_pred, y_conf = test_pipeline(G, C, loader, device, score_type=SCORE_TYPE, T=TEMP)

        # ==========================================
        # === [新增] 调用绘图函数 ===
        # ==========================================
        plot_dir = os.path.join(results_dir, 'plots_energy')  # 区分于原来的 plots
        os.makedirs(plot_dir, exist_ok=True)

        plot_confidence_distribution(
            y_true, y_conf, config.unknown_classes,
            task_name, plot_dir
        )
        # ==========================================

        # OOD Metrics
        ood_metrics = calculate_ood_metrics(
            y_true, y_conf, config.unknown_classes,
            score_type=SCORE_TYPE
        )

        # 分类指标 (只计算已知类)
        is_unknown = np.isin(y_true, config.unknown_classes).astype(int)
        known_mask = (is_unknown == 0)

        if np.sum(known_mask) > 0:
            acc_known = accuracy_score(y_true[known_mask], y_pred[known_mask])
            f1_known = f1_score(y_true[known_mask], y_pred[known_mask], average='macro')
        else:
            acc_known = 0.0;
            f1_known = 0.0

        print(f"  [CLS] Acc (Known): {acc_known:.4f}")
        task_block_str += f"  [CLS] Acc (Known): {acc_known:.4f}\n"

        if ood_metrics:
            print(f"  [OOD] AUROC:       {ood_metrics['AUROC']:.4f}")
            print(f"  [OOD] FPR95:       {ood_metrics['FPR95']:.4f}")
            task_block_str += f"  [OOD] AUROC:       {ood_metrics['AUROC']:.4f}\n"
            task_block_str += f"  [OOD] FPR95:       {ood_metrics['FPR95']:.4f}\n"
        else:
            print("  [OOD] Metrics N/A")
            task_block_str += "  [OOD] Metrics N/A\n"

        txt_report_content += task_block_str
        res_dict = {'Task': task_name, 'Acc_Known': acc_known, 'F1_Known': f1_known}
        if ood_metrics:
            res_dict.update(ood_metrics)
        else:
            res_dict.update({'AUROC': 0, 'AUPR_Out': 0, 'FPR95': 1.0})
        results.append(res_dict)

        # 保存
    df = pd.DataFrame(results)
    save_path = os.path.join(results_dir, 'aminet_ood_benchmark.csv')
    df.to_csv(save_path, index=False)
    print(f"\n✅ Results saved to: {save_path}")

    txt_save_path = os.path.join(results_dir, 'aminet_ood_benchmark.txt')
    with open(txt_save_path, 'w') as f:
        f.write(txt_report_content.strip())
    print(f"✅ Text Report saved to: {txt_save_path}")


if __name__ == '__main__':
    main()