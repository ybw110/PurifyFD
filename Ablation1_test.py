import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === 导入自定义模块 ===
from Ablation1_model import CWRU_Baseline_Network
from config import Config
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS


config = Config()


def plot_confusion_matrix(cm, classes, domain_name, save_path):
    """保存归一化的混淆矩阵图片"""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("custom", ["#F8F4F2", '#1B3B70'])
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm_norm[i, j], fmt), ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black")

    ax.set_title(f'Confusion Matrix - {domain_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # === 配置：此处填写训练好的模型路径 ===
    # 加载权重,CWRU多域
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_0HP_1HP_2HP_20251231_132127/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_0HP_1HP_3HP_20251231_132353/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_0HP_2HP_3HP_20251231_132657/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_1HP_2HP_3HP_20251231_133148/best_model.pth'
    #
    # # # 加载权重,JNU 10LOCAL多域
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_600_800_20251231_133902/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_600_1000_20251231_134901/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_800_1000_20251231_134113/best_model.pth'
    #
    # # # 加载权重,完整域
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_0HP_1HP_2HP_3HP_20260102_134330/best_model.pth'
    MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/baseline_cnn/MultiSource_600_800_1000_20260102_134017/best_model.pth'




    class_names = ['Normal', 'Inner', 'Ball', 'Outer']

    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

    # ================= 1. 数据准备 =================

    # A. 加载 CWRU 单独工况 (用于细粒度分析)
    all_cwru_individual = prepare_all_domains_detailed()

    # B. 加载 JNU 单独工况
    all_jnu_individual = prepare_jnu_detailed(jnu_root)

    # C. [新增] 加载 CWRU 完整测试集 (混合所有工况)
    print("Preparing Full CWRU Test Set...")
    cwru_mixed_loaders = prepare_mixed_domains(ALL_DOMAINS)
    full_cwru_test_loader = cwru_mixed_loaders['test']

    # D. [新增] 加载 JNU 完整测试集 (混合所有工况)
    print("Preparing Full JNU Test Set...")
    jnu_mixed_loaders = prepare_mixed_jnu_domains(ALL_SPEEDS, jnu_root)
    full_jnu_test_loader = jnu_mixed_loaders['test']

    # ================= 2. 定义测试任务 =================

    TEST_TASKS = [
        # --- 单独工况 ---
        ('CWRU_0HP', all_cwru_individual['0HP']['test']),
        ('CWRU_1HP', all_cwru_individual['1HP']['test']),
        ('CWRU_2HP', all_cwru_individual['2HP']['test']),
        ('CWRU_3HP', all_cwru_individual['3HP']['test']),
        ('JNU_600', all_jnu_individual['600']['test']),
        ('JNU_800', all_jnu_individual['800']['test']),
        ('JNU_1000', all_jnu_individual['1000']['test']),

        # --- [新增] 完整数据集整体测试 ---
        ('FULL_CWRU', full_cwru_test_loader),  # CWRU 全家桶
        ('FULL_JNU', full_jnu_test_loader)  # JNU 全家桶
    ]

    # 2. 初始化模型
    model = CWRU_Baseline_Network(num_classes=config.num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    results_dir = os.path.dirname(MODEL_PATH)
    summary_results = []

    for task_name, test_loader in TEST_TASKS:
        print(f"\n>>> Testing Task: {task_name}")

        preds_list, labels_list = [], []
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device).float()
                labels = data[1].to(device).long()
                logits, _ = model(inputs)
                _, preds = torch.max(logits, 1)
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        preds_all, labels_all = np.array(preds_list), np.array(labels_list)

        # 指标计算
        acc = np.mean(preds_all == labels_all)
        cm = confusion_matrix(labels_all, preds_all)

        # 为该工况创建保存子目录
        domain_save_dir = os.path.join(results_dir, f'test_results_{task_name}')
        os.makedirs(domain_save_dir, exist_ok=True)

        # 保存报告
        report = classification_report(labels_all, preds_all, target_names=class_names, digits=4)
        with open(os.path.join(domain_save_dir, 'report.txt'), 'w') as f:
            f.write(report)

        # 绘图
        plot_confusion_matrix(cm, class_names, task_name, os.path.join(domain_save_dir, 'cm.png'))

        summary_results.append({"Domain": task_name, "Accuracy": acc})
        print(f"Accuracy on {task_name}: {acc:.4%}")

    # 最终汇总对照表
    print("\n" + "=" * 35)
    print(f"{'Domain':<10} | {'Test Accuracy':<15}")
    print("-" * 35)
    for res in summary_results:
        print(f"{res['Domain']:<10} | {res['Accuracy']:.2%}")
    print("=" * 35)


if __name__ == '__main__':
    main()