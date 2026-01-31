import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === 新增 t-SNE 库 ===
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# === 导入自定义模块 ===
from baseline_mixup_model import MixupCNN
from config import Config
from dataset_spilt_selfA import prepare_self_detailed, prepare_mixed_self_domains
from dataset_spilt_selfB import prepare_self_b_detailed, prepare_mixed_self_b_domains


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


def plot_tsne_comparison(model, source_loader, target_loader, device, save_path, title="t-SNE"):
    """
    绘制 Source vs Target 的 t-SNE 对比图
    """
    print(f"⏳ 正在计算 {title} 的 t-SNE...")

    # 1. 提取特征
    src_feat, src_labels = extract_features(model, source_loader, device)
    tgt_feat, tgt_labels = extract_features(model, target_loader, device)

    # 2. 合并数据
    # 域标签: 0=Source, 1=Target
    src_domain_labels = np.zeros(len(src_labels))
    tgt_domain_labels = np.ones(len(tgt_labels))

    combined_feat = np.concatenate([src_feat, tgt_feat])
    combined_labels = np.concatenate([src_labels, tgt_labels])
    combined_domains = np.concatenate([src_domain_labels, tgt_domain_labels])

    # 3. PCA 预处理 (加速 t-SNE)
    if combined_feat.shape[1] > 50:
        pca = PCA(n_components=50)
        combined_feat = pca.fit_transform(combined_feat)

    # 4. 运行 t-SNE
    # perplexity 设为 30 左右通常效果较好
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(combined_feat)

    # 5. 绘图
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Arial'  # 防止中文乱码，或替换为 'Arial'

    # 颜色映射 (4类)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
    class_names = ["Normal", "Slight", "Medium", "Severe"]

    # 绘制 Source (空心圆圈 o)
    for c in range(4):
        idx = (combined_labels == c) & (combined_domains == 0)
        if np.sum(idx) > 0:
            plt.scatter(embeddings[idx, 0], embeddings[idx, 1],
                        c=colors[c], marker='o', label=f'Src-{class_names[c]}',
                        alpha=0.3, s=30, edgecolors='k', linewidth=0.5)

    # 绘制 Target (实心三角 ^)
    for c in range(4):
        idx = (combined_labels == c) & (combined_domains == 1)
        if np.sum(idx) > 0:
            plt.scatter(embeddings[idx, 0], embeddings[idx, 1],
                        c=colors[c], marker='^', label=f'Tgt-{class_names[c]}',
                        alpha=0.8, s=40, edgecolors='k', linewidth=0.5)

    plt.title(title, fontsize=14)
    # 图例放在外侧，防止遮挡
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, borderaxespad=0.)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE 保存至: {save_path}")


def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # -------- 方式 A：手动指定（只测一个模型）--------
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfA_Single_1200_1/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfA_Single_1800_1/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfA_Multi_1200_1800_1/best_model.pth'

    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfB_Single_1200_1/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfB_Single_1800_1/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfB_Multi_1200_1800_1/best_model.pth'

    #28
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfA_Single_1200_20260124_145235/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfA_Single_1800_20260124_145206/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfA_Multi_1200_1800_20260124_144932/best_model.pth'
    #
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfB_Single_1200_20260124_145122/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfB_Single_1800_20260124_145048/best_model.pth'
    MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/mixup_cnn/Mixup_SelfB_Multi_1200_1800_20260124_144809/best_model.pth'

    # 类别名
    class_names = ["Normal", "Slight", "Medium", "Severe"]

    # ==========================================================
    # 2) 准备数据：A单域、B单域、A完整、B完整
    # ==========================================================
    # 单独工况
    all_a_individual = prepare_self_detailed()  # {'1200': loaders, '1800': loaders}  (SelfA)
    all_b_individual = prepare_self_b_detailed()  # {'1200': loaders, '1800': loaders}  (SelfB)

    # 完整混合测试集
    print("Preparing FULL_A (SelfA mixed: 1200+1800) ...")
    a_mixed_loaders = prepare_mixed_self_domains(["1200", "1800"])
    full_a_test_loader = a_mixed_loaders["test"]

    print("Preparing FULL_B (SelfB mixed: 1200+1800) ...")
    b_mixed_loaders = prepare_mixed_self_b_domains(["1200", "1800"])
    full_b_test_loader = b_mixed_loaders["test"]

    # ==========================================================
    # 3) 你的 TEST_TASKS（按你要求）
    # ==========================================================
    TEST_TASKS = [
        ("A1200", all_a_individual["1200"]["test"]),
        ("A1800", all_a_individual["1800"]["test"]),
        ("B1200", all_b_individual["1200"]["test"]),
        ("B1800", all_b_individual["1800"]["test"]),
        ("FULL_A", full_a_test_loader),
        ("FULL_B", full_b_test_loader),
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

        summary_results.append({"Domain": task_name, "Accuracy": acc})
        print(f"Accuracy on {task_name}: {acc:.4%}")

    # 最终汇总对照表
    print("\n" + "=" * 35)
    print(f"{'Domain':<10} | {'Test Accuracy':<15}")
    print("-" * 35)
    for res in summary_results:
        print(f"{res['Domain']:<10} | {res['Accuracy']:.2%}")
    print("=" * 35)

    # 保存总汇总
    summary_txt = os.path.join(results_dir, "test_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"MODEL_PATH: {MODEL_PATH}\n")
        f.write("Domain\tAccuracy\n")
        for res in summary_results:
            f.write(f"{res['Domain']}\t{res['Accuracy']:.6f}\n")
    print(f"\n[Saved] Summary txt -> {summary_txt}")



    # # ==========================================================
    # # 4) [新增] t-SNE 可视化
    # # ==========================================================
    # print("\n" + "=" * 50)
    # print(">>> Generating t-SNE Visualizations")
    # print("=" * 50)
    #
    # # 场景 1: Cross-Condition (比如 A1200 -> A1800)
    # if "1200" in all_a_individual and "1800" in all_a_individual:
    #     src_loader = all_a_individual["1200"]["test"]
    #     tgt_loader = all_a_individual["1800"]["test"]
    #     save_path = os.path.join(results_dir, 'tsne_A1200_vs_A1800.png')
    #
    #     plot_tsne_comparison(
    #         model, src_loader, tgt_loader, device, save_path,
    #         title="Mixup: A1200 (Source) vs A1800 (Target)"
    #     )
    #
    # # 场景 2: Cross-Machine (Full A -> Full B)
    # save_path_ab = os.path.join(results_dir, 'tsne_FullA_vs_FullB.png')
    # plot_tsne_comparison(
    #     model, full_a_test_loader, full_b_test_loader, device, save_path_ab,
    #     title="Mixup: SelfA (Source) vs SelfB (Target)"
    # )


if __name__ == '__main__':
    main()