import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, average_precision_score, f1_score
# === 导入自定义模块 ===
from 真实物理标签_diversify_algorithm import Diversify
from config import Config
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS

config = Config()

# 标签逆映射：网络预测(0,1,2) -> 物理含义(0,1,3)
# 根据 config.known_classes 自动生成
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
    plt.figure(figsize=(6,5))
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.color': '0.9'})

    # 3. 绘制直方图 (使用 seaborn 的 histplot 或 kdeplot)
    # color='blue' 代表 ID, color='red' 代表 OOD
    # sns.histplot(id_scores, color='#6D95C3', label='In-Distribution (Known)',
    #              kde=True, stat="density", element="step", alpha=0.6)
    # sns.histplot(ood_scores, color='#E09D94', label='Out-of-Distribution (Unknown)',
    #              kde=True, stat="density", element="step", alpha=0.6)

    sns.kdeplot(id_scores, color='#C4E0E8', label='Known (ID)',fill=True, alpha=0.4, linewidth=2)
    sns.kdeplot(ood_scores, color='#E09D94', label='Unknown (OOD)',fill=True, alpha=0.4, linewidth=2)

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



def test_pipeline(model, test_loader, device='cuda', score_type='energy', T=1.0):
    """
    开集测试核心函数
    Args:
        score_type: 'softmax' (原版) 或 'energy' (新版)
        T: 温度参数 (Temperature), Energy Score 中常用 1.0
    """
    model.eval()
    y_true = []
    y_pred_mapped = []
    y_conf = []

    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data[0].to(device).float()
            labels = batch_data[1].to(device).long()

            # 1. 模型预测 logits: [batch, num_classes]
            logits = model.predict(inputs)

            # 2. 获取预测类别 (不管是 Softmax 还是 Energy，预测类都是 logits 最大值)
            preds_idx = torch.argmax(logits, dim=1)

            # 3. 计算置信度/分数
            if score_type == 'softmax':
                # 原有逻辑: MSP (Maximum Softmax Probability)
                probs = F.softmax(logits, dim=1)
                conf_scores, _ = torch.max(probs, dim=1)

            elif score_type == 'energy':
                # === 修改点: Energy Score 计算 ===
                # E(x) = -T * logsumexp(logits / T)
                # 我们通常使用负能量 (-Energy) 作为置信度，即 LogSumExp
                # 值越大 -> 越可能是已知类 (ID)
                # 值越小 -> 越可能是未知类 (OOD)
                conf_scores = T * torch.logsumexp(logits / T, dim=1)

            else:
                raise ValueError("Unknown score_type")

            # 4. 还原为物理标签
            current_preds_mapped = [REVERSE_MAP[idx.item()] for idx in preds_idx]

            y_true.extend(labels.cpu().numpy())
            y_pred_mapped.extend(current_preds_mapped)
            y_conf.extend(conf_scores.cpu().numpy())

    return np.array(y_true), np.array(y_pred_mapped), np.array(y_conf)


def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

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
    # TASKS.append(('S1_CWRU_0HP', cwru_single['0HP']['test']))
    # TASKS.append(('S1_CWRU_3HP', cwru_single['3HP']['test']))
    # TASKS.append(('S1_JNU_600', jnu_single['600']['test']))
    #
    # # --- 表 1: 跨工况 (重点关注你的 B-3 模型表现) ---
    # TASKS.append(('B1_B3_Target_CWRU_3HP', cwru_single['3HP']['test']))  # 目标域是 3HP
    # TASKS.append(('B2_Target_CWRU_0HP', cwru_single['0HP']['test']))  # 目标域是 0HP
    # TASKS.append(('B4_B5_Target_JNU_1000', jnu_single['1000']['test']))  # 目标域是 1000

    # --- 表 2: 跨设备 (高光实验) ---
    TASKS.append(('C1_Target_JNU_All', jnu_mixed['test']))
    TASKS.append(('C2_Target_CWRU_All', cwru_mixed['test']))


    # ================= 2. 加载模型 =================
    model = Diversify(config).to(device)
    # OOD1
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD1/SingleSource_0HP_20260103_204606/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD1/SingleSource_3HP_20260103_204608/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD1/MultiSource_0HP_1HP_2HP_20260103_211607/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD1/MultiSource_0HP_1HP_2HP_3HP_20260103_201047/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD1/SingleSource_600_20260103_204738/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD1/MultiSource_600_800_20260103_204751/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD1/MultiSource_600_800_1000_20260103_202932/best_model.pth'

    #OOD2
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2/SingleSource_0HP_20260102_215748/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2/SingleSource_3HP_20260102_215134/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2/MultiSource_0HP_1HP_2HP_20260102_205208/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2/MultiSource_0HP_1HP_2HP_3HP_20260102_224746/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2/SingleSource_600_20260102_220751/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2/MultiSource_600_800_20260102_230543/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2/MultiSource_600_800_1000_20260102_221951/best_model.pth'


    # OOD2  K=2
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_K2/MultiSource_0HP_1HP_2HP_3HP_20260117_114339/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_K2/MultiSource_600_800_1000_20260117_124248/best_model.pth'

    # OOD2  K=3
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_K3/MultiSource_0HP_1HP_2HP_3HP_20260117_135312/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_K3/MultiSource_600_800_1000_20260117_135331/best_model.pth'
    #
    # # OOD2  K=5
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_K5/MultiSource_0HP_1HP_2HP_3HP_20260117_114355/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_K5/MultiSource_600_800_1000_20260117_124240/best_model.pth'
    #
    # # OOD2  K=10
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_K10/MultiSource_0HP_1HP_2HP_3HP_20260117_122417/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_K10/MultiSource_600_800_1000_20260117_124235/best_model.pth'

    # # OOD2  without_updata_a
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_without_update_a/MultiSource_0HP_1HP_2HP_3HP_20260127_210831/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_without_update_a/MultiSource_600_800_1000_20260127_213045/best_model.pth'

    # # OOD2  without_updata_d
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_without_update_d/MultiSource_0HP_1HP_2HP_3HP_20260127_210836/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_without_update_d/MultiSource_600_800_1000_20260127_213038/best_model.pth'

    # # OOD2  True label
    #要改K=4
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_True_label/MultiSource_0HP_1HP_2HP_3HP_20260131_104641/best_model.pth'

    #要改K=3
    model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/train_output_diversify_OOD2_True_label/MultiSource_600_800_1000_20260131_104649/best_model.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"⚠️ Warning: Model path not found: {model_path}. Using random weights.")

    results_dir = os.path.dirname(model_path) if os.path.exists(model_path) else 'results/test_output'
    os.makedirs(results_dir, exist_ok=True)

    # ================= 4. 执行测试 =================

    # === 设置使用的分数类型 ===
    SCORE_TYPE = 'energy'  # 可选: 'softmax' 或 'energy'
    TEMP = 1  # Energy Temperature (通常设为1.0即可)


    results = []
    txt_report_content = ""

    for task_name, loader in TASKS:
        print(f"\n--- Testing Task: {task_name} ---")
        task_block_str = f"\n--- Task: {task_name} ---\n"

        # 1. 调用 test_pipeline 时传入 score_type
        y_true, y_pred, y_conf = test_pipeline(
            model, loader, device,
            score_type=SCORE_TYPE, T=TEMP
        )

        # ==========================================
        # === [新增] 调用绘图函数 ===
        # ==========================================
        # 创建一个专门存放图片的子文件夹
        plot_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        plot_confidence_distribution(
            y_true,
            y_conf,
            config.unknown_classes,
            task_name,
            plot_dir
        )
        # ==========================================

        # --- A. 计算 OOD 指标 (AUROC, AUPR-Out, FPR95) ---
        ood_metrics = calculate_ood_metrics(
            y_true, y_conf, config.unknown_classes,
            score_type=SCORE_TYPE
        )

        # --- B. 计算已知类指标 (Acc, Macro-F1) ---
        is_unknown = np.isin(y_true, config.unknown_classes).astype(int)
        known_mask = (is_unknown == 0)

        if np.sum(known_mask) > 0:
            acc_known = accuracy_score(y_true[known_mask], y_pred[known_mask])
            f1_known = f1_score(y_true[known_mask], y_pred[known_mask], average='macro')
        else:
            acc_known = 0.0
            f1_known = 0.0

        # --- 填充文本块内容 ---
        task_block_str += f"  [CLS] Acc (Known): {acc_known:.4f}\n"
        task_block_str += f"  [CLS] F1  (Known): {f1_known:.4f}\n"
        # --- 打印结果 ---
        print(f"  [CLS] Acc (Known): {acc_known:.4f}")
        print(f"  [CLS] F1  (Known): {f1_known:.4f}")

        if ood_metrics:
            task_block_str += f"  [OOD] AUROC:       {ood_metrics['AUROC']:.4f}\n"
            task_block_str += f"  [OOD] AUPR-Out:    {ood_metrics['AUPR_Out']:.4f}\n"
            task_block_str += f"  [OOD] FPR95:       {ood_metrics['FPR95']:.4f}\n"
        else:
            task_block_str += "  [OOD] Metrics N/A (Data missing OOD or ID samples)\n"

            # 打印到控制台
        print(task_block_str.strip())

        # 追加到总报告字符串
        txt_report_content += task_block_str

        # --- 收集数据 ---
        res_dict = {
            'Task': task_name,
            'Acc_Known': acc_known,
            'F1_Known': f1_known
        }
        if ood_metrics:
            res_dict.update(ood_metrics)
        else:
            # 填充空值以保持表格对齐
            res_dict.update({'AUROC': 0, 'AUPR_Out': 0, 'FPR95': 1.0})

        results.append(res_dict)

    # ================= 5. 保存汇总 =================
    # 1. 保存 CSV
    df = pd.DataFrame(results)
    # 格式化一下列顺序
    cols = ['Task', 'Acc_Known', 'F1_Known', 'AUROC', 'AUPR_Out', 'FPR95']
    df = df[cols]

    save_path = os.path.join(results_dir, 'final_ood_benchmark.csv')
    df.to_csv(save_path, index=False)
    print(f"\n✅ Benchmark CSV saved to:\n{save_path}")

    # 2. 保存 TXT
    txt_save_path = os.path.join(results_dir, 'final_ood_benchmark.txt')
    with open(txt_save_path, 'w') as f:
        f.write(txt_report_content.strip())
    print(f"✅ Benchmark Text Report saved to:\n{txt_save_path}")

    print("\nPreview:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()