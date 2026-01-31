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
# 1. SelfA (1楼数据集) - 直接导入原函数名
from dataset_spilt_selfA import prepare_self_detailed, prepare_mixed_self_domains
from dataset_spilt_selfA import ALL_SPEEDS as SELFA_SPEEDS

# 2. SelfB (6楼数据集) - 直接导入原函数名
from dataset_spilt_selfB import prepare_self_b_detailed, prepare_mixed_self_b_domains
from dataset_spilt_selfB import ALL_SPEEDS as SELFB_SPEEDS


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
    feats, labels = extract_features_dacn(model, loader, device, max_samples=500)

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

    class_names = ['Normal', 'Inner', 'Ball', 'Outer']
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE saved: {save_path}")


# ==========================================
#               主测试逻辑
# ==========================================
def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # [重要] 请在这里修改你要测试的模型路径 (.pth 文件)
    # seed48
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfA_Multi_1200_1800_20260124_144010/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfA_Single_1200_20260124_143422/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfA_Single_1800_20260124_143513/best_model.pth'

    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfB_Multi_1200_1800_20260124_144218/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfB_Single_1200_20260124_143609/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfB_Single_1800_20260124_143736/best_model.pth'
    #
    # #28
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfA_Multi_1200_1800_20260124_150103/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfA_Single_1200_20260124_150241/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfA_Single_1800_20260124_150344/best_model.pth'

    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfB_Multi_1200_1800_20260124_145752/best_model.pth'
    # MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfB_Single_1200_20260124_150442/best_model.pth'
    MODEL_PATH = '/data2/ybw25/轴承DG/实验self1+self2/results/dacn/DACN_SelfB_Single_1800_20260124_150620/best_model.pth'



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



    results_dir = os.path.dirname(MODEL_PATH)
    summary_results = []

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

        # # 5. 绘制混淆矩阵
        # plot_confusion_matrix(cm, class_names, task_name, os.path.join(domain_save_dir, 'cm.png'))
        #
        # # 6. 绘制 t-SNE
        # tsne_path = os.path.join(domain_save_dir, f'tsne_{task_name}.png')
        # plot_tsne_dacn(model, test_loader, device, tsne_path, title=f"DACN Features: {task_name}")

        # 记录汇总
        summary_results.append({"Domain": task_name, "Accuracy": acc})
        print(f"   Accuracy: {acc:.2%}")

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