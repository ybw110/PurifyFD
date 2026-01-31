import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# === t-SNE 库 ===
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# === 导入自定义模块 ===
# 注意：这里导入的是 dacn_model，确保文件名一致
from baseline_dacn_model import DACN
from config import Config
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS

config = Config()


def plot_confusion_matrix(cm, classes, domain_name, save_path):
    """保存归一化的混淆矩阵图片 (保持风格一致)"""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    # 使用与之前一致的蓝白配色
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
#        DACN 专用的 t-SNE 提取与绘图
# ==========================================
def extract_features_dacn(model, loader, device, max_samples=2000):
    """提取 DACN 的域不变特征 (Feature G)"""
    model.eval()
    features = []
    labels = []

    # 计数器，防止数据量过大爆内存或绘图太慢
    total_count = 0

    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device).float()
            target = data[1].to(device).long()

            # DACN 推理逻辑：使用 forward_pretrain (F -> G -> C)
            # 返回 (logits, g_features)
            _, feat = model.forward_pretrain(inputs)

            features.append(feat.cpu().numpy())
            labels.append(target.cpu().numpy())

            total_count += len(target)
            # if total_count > max_samples * 1.5: # 稍微多采一点用于随机筛选
            #     break

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 随机采样到 max_samples
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    return features, labels


def plot_tsne_dacn(model, loader, device, save_path, title="DACN Features"):
    print(f"⏳ [t-SNE] Processing: {title} ...")

    # 1. 提取特征
    feats, labels = extract_features_dacn(model, loader, device, max_samples=200)

    # 2. PCA 降维 (加速 t-SNE)
    if feats.shape[1] > 50:
        pca = PCA(n_components=50)
        feats = pca.fit_transform(feats)

    # 3. t-SNE 计算
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(feats)

    # 4. 绘图
    plt.figure(figsize=(5, 4))
    plt.rcParams['font.family'] = 'Arial'

    class_names = ['Normal', 'Inner Race', 'Ball Fault', 'Outer Race']
    # 保持一致的配色方案
    colors = ['#6D95C3', '#EBB577', '#CDD797', '#E09D94']
    markers = ['o', '^', 's', 'D']

    for c in range(4):
        idx = (labels == c)
        if np.sum(idx) > 0:
            plt.scatter(
                embeddings[idx, 0], embeddings[idx, 1],
                c=colors[c], marker=markers[c], label=class_names[c],
                s=50, alpha=0.8, edgecolors='k', linewidth=0.5
            )

    plt.legend(loc='best', fontsize=14)
    plt.xticks([]);
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE saved: {save_path}")


# ==========================================
#               主测试逻辑
# ==========================================
def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # [重要] 请在这里修改你要测试的模型路径 (.pth 文件)
    # seed42
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_0HP_1HP_2HP_20260123_104058/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_0HP_1HP_3HP_20260123_104340/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_0HP_2HP_3HP_20260123_104633/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_1HP_2HP_3HP_20260123_102844/best_model.pth'
    MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_0HP_1HP_2HP_3HP_20260123_105110/best_model.pth'

    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_600_800_20260123_110704/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_600_1000_20260123_110301/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_800_1000_20260123_111222/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_600_800_1000_20260123_105547/best_model.pth'
    #
    # # seed24
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_0HP_1HP_2HP_20260123_112153/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_0HP_1HP_3HP_20260123_112434/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_0HP_2HP_3HP_20260123_111849/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_1HP_2HP_3HP_20260123_112935/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_0HP_1HP_2HP_3HP_20260123_113240/best_model.pth'
    #
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_600_800_20260123_120039/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_600_1000_20260123_114543/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_800_1000_20260123_114234/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/dacn/DACN_Multi_600_800_1000_20260123_113550/best_model.pth'

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model path not found: {MODEL_PATH}")
        print("请在代码中修改 MODEL_PATH 变量为实际的 .pth 文件路径")
        return

    print(f"loading model from: {MODEL_PATH}")

    # 初始化模型结构
    model = DACN(num_classes=config.num_classes).to(device)
    # 加载权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 准备结果保存目录
    results_dir = os.path.dirname(MODEL_PATH)
    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

    # ================= 1. 数据准备 =================
    # 根据你的需求，可以选择加载哪些数据进行测试

    # A. CWRU 单独工况
    all_cwru_individual = prepare_all_domains_detailed()

    # B. JNU 单独工况
    all_jnu_individual = prepare_jnu_detailed(jnu_root)

    # C. CWRU 完整测试集 (混合所有工况)
    print("Preparing Full CWRU Test Set...")
    cwru_mixed_loaders = prepare_mixed_domains(ALL_DOMAINS)
    full_cwru_test_loader = cwru_mixed_loaders['test']

    # D. JNU 完整测试集
    print("Preparing Full JNU Test Set...")
    jnu_mixed_loaders = prepare_mixed_jnu_domains(ALL_SPEEDS, jnu_root)
    full_jnu_test_loader = jnu_mixed_loaders['test']

    # ================= 2. 定义测试任务列表 =================
    # 你可以在这里注释掉不需要测试的任务
    TEST_TASKS = [
        # --- CWRU 单独工况 ---
        # ('CWRU_0HP', all_cwru_individual['0HP']['test']),
        # ('CWRU_1HP', all_cwru_individual['1HP']['test']),
        # ('CWRU_2HP', all_cwru_individual['2HP']['test']),
        # ('CWRU_3HP', all_cwru_individual['3HP']['test']),
        #
        # # --- JNU 单独工况 ---
        # ('JNU_600', all_jnu_individual['600']['test']),
        # ('JNU_800', all_jnu_individual['800']['test']),
        # ('JNU_1000', all_jnu_individual['1000']['test']),
        #
        # # --- 综合测试 ---
        # ('FULL_CWRU', full_cwru_test_loader),
        ('FULL_JNU', full_jnu_test_loader),
    ]

    summary_results = []
    class_names = ['Normal', 'Inner', 'Ball', 'Outer']

    # ================= 3. 开始测试循环 =================
    for task_name, test_loader in TEST_TASKS:
        print(f"\n>>> Testing Task: {task_name}")

        preds_list, labels_list = [], []

        # 推理循环
        with torch.no_grad():
            for data in tqdm(test_loader, ncols=80, leave=False):
                inputs = data[0].to(device).float()
                labels = data[1].to(device).long()

                # DACN 推理：只走 F->G->C 路径
                logits, _ = model.forward_pretrain(inputs)

                _, preds = torch.max(logits, 1)
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        preds_all = np.array(preds_list)
        labels_all = np.array(labels_list)

        # 1. 计算准确率
        acc = np.mean(preds_all == labels_all)

        # 2. 计算混淆矩阵
        cm = confusion_matrix(labels_all, preds_all)

        # 3. 创建保存子目录
        domain_save_dir = os.path.join(results_dir, f'test_results_{task_name}')
        os.makedirs(domain_save_dir, exist_ok=True)

        # 4. 保存 Report (Precision, Recall, F1)
        report = classification_report(labels_all, preds_all, target_names=class_names, digits=4)
        with open(os.path.join(domain_save_dir, 'report.txt'), 'w') as f:
            f.write(f"Task: {task_name}\n")
            f.write(f"Accuracy: {acc:.4%}\n\n")
            f.write(report)

        # 5. 绘制混淆矩阵
        plot_confusion_matrix(cm, class_names, task_name, os.path.join(domain_save_dir, 'cm.png'))

        # 6. 绘制 t-SNE
        tsne_path = os.path.join(domain_save_dir, f'tsne_{task_name}.png')
        plot_tsne_dacn(model, test_loader, device, tsne_path, title=f"DACN Features: {task_name}")

        # 记录汇总
        summary_results.append({"Domain": task_name, "Accuracy": acc})
        print(f"   Accuracy: {acc:.2%}")

    # ================= 4. 输出最终汇总 =================
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