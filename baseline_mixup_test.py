import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === 导入自定义模块 ===
from baseline_mixup_model import MixupCNN
from config import Config
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS

# === 新增 t-SNE 库 ===
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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


# ==========================================
#              t-SNE 可视化工具 (单域版)
# ==========================================
def extract_features(model, loader, device):
    """提取特征和标签"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device).float()
            target = data[1].to(device).long()

            # SampleCNN forward 返回 (logits, feat)
            _, feat = model(inputs)

            features.append(feat.cpu().numpy())
            labels.append(target.cpu().numpy())

    return np.concatenate(features), np.concatenate(labels)


def plot_tsne_single_domain(model, loader, device, save_path, title="t-SNE Visualization", max_samples=2000):
    """
    绘制单个 Task 的 t-SNE 分布图
    :param max_samples: 最大显示样本数 (默认 2000)，超过则随机采样
    """
    print(f"⏳ [t-SNE] Processing: {title} ...")

    # 1. 提取所有特征
    feats, labels = extract_features(model, loader, device)
    total_samples = len(feats)

    # === [新增] 控制点数量逻辑 ===
    if total_samples > max_samples:
        print(f"⚠️ 样本数 ({total_samples}) > {max_samples}，正在随机采样...")
        # 生成随机索引
        indices = np.random.choice(total_samples, max_samples, replace=False)
        # 筛选数据
        feats = feats[indices]
        labels = labels[indices]

    # 2. PCA 预处理
    if feats.shape[1] > 50:
        pca = PCA(n_components=50)
        feats = pca.fit_transform(feats)

    # 3. t-SNE 计算
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(feats)

    # 4. 绘图配置 (同前)
    plt.figure(figsize=(5,4))
    plt.rcParams['font.family'] = 'Arial'

    class_names = ['Normal', 'Inner Race', 'Ball Fault', 'Outer Race']
    colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C']
    colors = ['#6D95C3', '#EBB577', '#CDD797', '#E09D94']

    markers = ['o', '^', 's', 'D']

    # 循环绘制
    for c in range(4):
        idx = (labels == c)
        if np.sum(idx) > 0:
            plt.scatter(
                embeddings[idx, 0], embeddings[idx, 1],
                c=colors[c],
                marker=markers[c],
                label=class_names[c],
                s=50,
                alpha=0.8,
                edgecolors='k',
                linewidth=0.5
            )

    # plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc='best', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE saved: {save_path}")

def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # === 配置：此处填写训练好的模型路径 ===
    # 加载权重,CWRU多域
    # seed42
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_0HP_1HP_2HP_20260104_165034/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_0HP_1HP_2HP_3HP_20260104_165955/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_0HP_1HP_3HP_20260104_165132/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_0HP_2HP_3HP_20260104_165237/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_1HP_2HP_3HP_20260104_164835/best_model.pth'
    #
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_600_1000_20260104_165550/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_600_800_1000_20260104_170127/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_600_800_20260104_165403/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_800_1000_20260104_164658/best_model.pth'
    #
    # # seed28
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_0HP_1HP_2HP_20260104_180916/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_0HP_1HP_2HP_3HP_20260104_181346/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_0HP_1HP_3HP_20260104_181034/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_0HP_2HP_3HP_20260104_175840/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_1HP_2HP_3HP_20260104_181222/best_model.pth'
    #
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_600_1000_20260104_175434/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_600_800_1000_20260104_175015/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_600_800_20260104_175550/best_model.pth'
    MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/mixup_cnn/Mixup_MultiSource_800_1000_20260104_175658/best_model.pth'




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
    # 初始化 Mixup 模型
    model = MixupCNN(num_classes=config.num_classes).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded Mixup model from {MODEL_PATH}")
    except FileNotFoundError:
        print("Model file not found. Please set MODEL_PATH correctly.")
        return
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
        # plot_confusion_matrix(cm, class_names, task_name, os.path.join(domain_save_dir, 'cm.png'))
        #
        # # 3. [新增] 绘制当前 Task 的 t-SNE
        # tsne_path = os.path.join(domain_save_dir, f'tsne_{task_name}.png')
        # plot_tsne_single_domain(model, test_loader, device, tsne_path, title=f"SampleCNN Features: {task_name}",max_samples=200)

        summary_results.append({"Domain": task_name, "Accuracy": acc})
        print(f"Accuracy on {task_name}: {acc:.4%}")

    # 最终汇总对照表
    print("\n" + "=" * 35)
    print(f"{'Domain':<10} | {'Test Accuracy':<15}")
    print("-" * 35)
    for res in summary_results:
        print(f"{res['Domain']:<10} | {res['Accuracy']:.2%}")
    print("=" * 35)

    # === 保存总汇总到一个 txt ===
    summary_txt = os.path.join(results_dir, "test_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"MODEL_PATH: {MODEL_PATH}\n")
        f.write("Domain\tAccuracy\n")
        for res in summary_results:
            f.write(f"{res['Domain']}\t{res['Accuracy']:.6f}\n")

    print(f"\n[Saved] Summary txt -> {summary_txt}")


if __name__ == '__main__':
    main()