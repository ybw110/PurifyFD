import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, average_precision_score, f1_score

# === 导入自定义模块 ===
from backbone_model import CWRU_Baseline_Network
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

            # === 修正点：使用 model(inputs) 并解包 ===
            # CWRU_Baseline_Network forward 返回 (logits, features)
            logits, _ = model(inputs)

            # 2. 获取预测类别
            preds_idx = torch.argmax(logits, dim=1)

            # 3. 计算置信度/分数
            if score_type == 'softmax':
                # MSP (Maximum Softmax Probability)
                probs = F.softmax(logits, dim=1)
                conf_scores, _ = torch.max(probs, dim=1)

            elif score_type == 'energy':
                # Energy Score
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

    # 打印测试配置
    print(f"Known Classes: {config.known_classes}")
    print(f"Unknown Classes: {config.unknown_classes}")

    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

    # ================= 1. 数据准备 =================

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

    # OOD2
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/backbone_OOD2/MultiSource_0HP_1HP_2HP_3HP_20260117_152748/best_model.pth'
    MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU+OOD/results/backbone_OOD2/MultiSource_600_800_1000_20260117_152855/best_model.pth'


    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    results_dir = os.path.dirname(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'results/test_output'
    os.makedirs(results_dir, exist_ok=True)

    # 2. 初始化模型
    model = CWRU_Baseline_Network(num_classes=config.num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()


    # ================= 4. 执行测试 =================

    # === 设置使用的分数类型 ===
    SCORE_TYPE = 'energy'  # 可选: 'softmax' 或 'energy'
    TEMP = 1.0  # Energy Temperature (通常设为1.0即可)


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