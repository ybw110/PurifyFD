import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob

# === 新增 t-SNE 库 ===
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# === 导入自定义模块 ===
from baseline_sample_cnn_model import SampleCNN
from config import Config

# === 自建数据集加载 ===
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

    class_names = ["Normal", "Slight", "Medium", "Severe"]
    class_names = ['Normal', 'Inner Race', 'Ball Fault', 'Outer Race']
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

    # seed 42-------- 方式 A：手动指定（只测一个模型）--------
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfA_Single_1200_20260124_142600/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfA_Single_1800_20260124_142625/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfA_Multi_1200_1800_20260124_142752/best_model.pth'

    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfB_Single_1200_20260124_142654/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfB_Single_1800_20260124_142723/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfB_Multi_1200_1800_20260124_142817/best_model.pth'

    #28
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfA_Single_1200_20260124_145243/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfA_Single_1800_20260124_145204/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfA_Multi_1200_1800_20260124_144939/best_model.pth'

    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfB_Multi_1200_1800_20260124_144803/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfB_Single_1200_20260124_145129/best_model.pth'
    MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/sample_cnn/SelfB_Single_1800_20260124_145040/best_model.pth'

    # 类别名
    class_names = ["Normal", "Slight", "Medium", "Severe"]

    # ==========================================================
    # 2) 准备数据：A单域、B单域、A完整、B完整
    # ==========================================================
    # 单独工况
    all_a_individual = prepare_self_detailed()      # {'1200': loaders, '1800': loaders}  (SelfA)
    all_b_individual = prepare_self_b_detailed()    # {'1200': loaders, '1800': loaders}  (SelfB)

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
    model = SampleCNN(num_classes=config.num_classes).to(device)
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