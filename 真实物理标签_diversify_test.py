import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
# === 导入自定义模块 ===
from 真实物理标签_diversify_algorithm import Diversify  # 核心算法类
from config import Config        # 配置参数
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS
# === 新增 t-SNE 库 ===
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

config = Config()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=None, save_path=None):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 如果 cmap 为 None，则使用自定义蓝色调
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F8F4F2", '#1B3B70'])

    # 创建图像
    fig, ax = plt.subplots(figsize=(5, 4))  # 明确创建 fig, ax
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(cax, ax=ax)  # 绑定 colorbar 到 ax

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=14)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=14)


    # 在矩阵中添加文本
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=16)

    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.tight_layout()

    # 保存混淆矩阵图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # `bbox_inches='tight'` 解决超出问题
        print(f"Confusion matrix saved to {save_path}")
    # plt.show()
    plt.close()


def test_model(model, test_loader, device='cuda'):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data[0].to(device).float()
            labels = batch_data[1].to(device).long()

            outputs = model.predict(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def calculate_specificity(cm):

    specificity = {}
    num_classes = cm.shape[0]
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity[i] = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return specificity


def calculate_per_class_accuracy(cm):

    per_class_accuracy = {}
    num_classes = cm.shape[0]
    for i in range(num_classes):
        TP = cm[i, i]
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - TP)
        per_class_accuracy[i] = (TP + TN) / cm.sum() if cm.sum() > 0 else 0.0
    return per_class_accuracy


def calculate_weighted_average(metrics, labels, class_names):
    from collections import defaultdict

    support = defaultdict(int)
    for label in labels:
        support[label] += 1

    weighted_sum = 0.0
    total = len(labels)
    for i, class_name in enumerate(class_names):
        weighted_sum += metrics[i] * support[i]

    weighted_avg = weighted_sum / total if total > 0 else 0.0
    return weighted_avg

# ==========================================
#              t-SNE 可视化工具 (适配 Diversify)
# ==========================================
def extract_features(model, loader, device):
    """
    提取特征和标签
    注意：Diversify 的 predict 只返回 logits，
    这里我们需要手动调用 featurizer 和 bottleneck 来获取特征。
    """
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device).float()
            target = data[1].to(device).long()

            # === 核心修改：手动提取特征 ===
            # Diversify 结构: inputs -> featurizer -> bottleneck -> classifier
            # 我们需要 bottleneck 的输出作为 t-SNE 的特征
            feat = model.bottleneck(model.featurizer(inputs))

            features.append(feat.cpu().numpy())
            labels.append(target.cpu().numpy())

    return np.concatenate(features), np.concatenate(labels)


def plot_tsne_single_domain(model, loader, device, save_path, title="t-SNE Visualization", max_samples=2000):
    """
    绘制单个 Task 的 t-SNE 分布图 (保持与 Baseline 一致的风格)
    """
    print(f"⏳ [t-SNE] Processing: {title} ...")

    # 1. 提取所有特征
    feats, labels = extract_features(model, loader, device)
    total_samples = len(feats)

    # === 控制点数量逻辑 ===
    if total_samples > max_samples:
        print(f"⚠️ 样本数 ({total_samples}) > {max_samples}，正在随机采样...")
        indices = np.random.choice(total_samples, max_samples, replace=False)
        feats = feats[indices]
        labels = labels[indices]

    # 2. PCA 预处理
    if feats.shape[1] > 50:
        pca = PCA(n_components=50)
        feats = pca.fit_transform(feats)

    # 3. t-SNE 计算
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(feats)

    # 4. 绘图配置 (与 Baseline 保持一致)
    plt.figure(figsize=(5, 4))
    plt.rcParams['font.family'] = 'Arial'

    # === 这里的配置与您修改后的 baseline_sample_cnn_test.py 保持完全一致 ===
    class_names = ['Normal', 'Inner Race', 'Ball Fault', 'Outer Race']
    # 您的自定义配色
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

    plt.legend(loc='best', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE saved: {save_path}")


# 定义支持的算法
ALGORITHMS = ['diversify']


def get_algorithm_class(algorithm_name):
    """返回算法类"""
    if algorithm_name.lower() not in ALGORITHMS:
        raise NotImplementedError(f"Algorithm not found: {algorithm_name}")
    return Diversify  # 直接返回 Diversify 类


def main():
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

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

    # 初始化算法
    algorithm_class = get_algorithm_class(config.algorithm)
    model = algorithm_class(config)

    # 42
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_0HP_1HP_2HP_20251231_132127/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_0HP_1HP_3HP_20251231_133157/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_0HP_2HP_3HP_20251231_134149/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_1HP_2HP_3HP_20251231_135202/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_0HP_1HP_2HP_3HP_20260102_132517/best_model.pth'

    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_600_800_1000_20260102_134025/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_600_1000_20260102_122306/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_600_800_20260102_120032/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_800_1000_20251231_143649/best_model.pth'


    # 28
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_0HP_1HP_2HP_20260104_191913/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_0HP_1HP_3HP_20260104_191915/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_0HP_2HP_3HP_20260104_195747/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_1HP_2HP_3HP_20260104_195750/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_0HP_1HP_2HP_3HP_20260104_191910/best_model.pth'
    #
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_600_800_1000_20260104_203113/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_600_1000_20260104_203129/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_600_800_20260104_211327/best_model.pth'
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify/MultiSource_800_1000_20260104_211259/best_model.pth'

# 真实物理标签，seed28
    # K=4
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify_domain_true/MultiSource_0HP_1HP_2HP_3HP_20260131_115023/best_model.pth'
    # K=3
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify_domain_true/MultiSource_600_800_1000_20260131_115002/best_model.pth'

    # 真实物理标签，seed42
    # K=4
    # model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify_domain_true/MultiSource_0HP_1HP_2HP_3HP_20260131_135942/best_model.pth'
    # K=3
    model_path = '/data2/ybw25/轴承DG/实验CWRU+JNU/results/train_output_diversify_domain_true/MultiSource_600_800_1000_20260131_135941/best_model.pth'

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # 获取模型目录路径
    results_dir = os.path.dirname(model_path)

    # 类别标签
    class_names = [f'Class {i}' for i in range(config.num_classes)]

    summary_data = []

    for task_name, test_loader in TEST_TASKS:
        print(f"\n" + "=" * 50)
        print(f">>> Testing Task: {task_name}")

        # 执行测试 (内部逻辑不变)
        test_preds, test_labels = test_model(model, test_loader, device)

        # 生成分类报告
        report = classification_report(test_labels, test_preds, target_names=class_names, digits=5)
        print("Classification Report:")
        print(report)

        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, test_preds)
        # 计算特异度和每类准确率
        specificity = calculate_specificity(cm)
        per_class_accuracy = calculate_per_class_accuracy(cm)

        # 计算加权的整体指标：精确率、召回率、F1 分数
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
        weighted_metrics = {
            'Weighted Precision': precision,
            'Weighted Recall': recall,
            'Weighted F1 Score': f1
        }

        # 计算每类的指标：准确率（Accuracy）、阳性预测值（PPV/Precision）、敏感度（Sensitivity/Recall）、特异度（Specificity）
        per_class_metrics = {}
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            test_labels, test_preds, average=None)

        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'Accuracy': per_class_accuracy[i],
                'PPV (Precision)': precision_per_class[i],
                'Sensitivity (Recall)': recall_per_class[i],
                'Specificity': specificity[i]
            }

        # 计算特异度和准确率的加权平均
        weighted_specificity = calculate_weighted_average(specificity, test_labels, class_names)
        weighted_accuracy = calculate_weighted_average(per_class_accuracy, test_labels, class_names)

        # 打印指标
        print("\nWeighted Metrics:")
        for metric, value in weighted_metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"Weighted Specificity: {weighted_specificity:.4f}")
        print(f"Weighted Accuracy: {weighted_accuracy:.4f}")

        print("\nPer-Class Metrics:")
        for class_name, metrics in per_class_metrics.items():
            print(f"{class_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        summary_data.append({
            'Task': task_name,
            'Accuracy': weighted_accuracy,  # 使用加权准确率
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Specificity': weighted_specificity
        })

        # 为每个域创建独立的保存路径
        domain_results_dir = os.path.join(results_dir, f'test_results_{task_name}')
        os.makedirs(domain_results_dir, exist_ok=True)

        # 保存分类报告和额外指标
        report_path = os.path.join(domain_results_dir, 'test_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Domain: {task_name}\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nWeighted Metrics:\n")
            for metric, value in weighted_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write(f"Weighted Specificity: {weighted_specificity:.4f}\n")
            f.write(f"Weighted Accuracy: {weighted_accuracy:.4f}\n")
            f.write("\nPer-Class Metrics:\n")
            for class_name, metrics in per_class_metrics.items():
                f.write(f"{class_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        print(f"Classification report for {task_name} saved to {report_path}")

        # cm_path = os.path.join(domain_results_dir, f'test_confusion_matrix_{task_name}.png')
        # plot_confusion_matrix(cm, classes=class_names, normalize=True, title=f'Test Confusion Matrix - {task_name}',
        #                       cmap=None, save_path=cm_path)
        #
        # # 2. [新增] 绘制当前 Task 的 t-SNE
        # tsne_path = os.path.join(domain_results_dir, f'tsne_{task_name}.png')
        # plot_tsne_single_domain(model, test_loader, device, tsne_path,
        #                         title=f"PurifyFD Features: {task_name}",
        #                         max_samples=200)  # 采样数可按需调整

    # === [新增] 循环结束后，生成汇总表格并保存 ===
    print("\n" + "=" * 80)
    print(">>> Final Summary Table")
    print("=" * 80)

    # 创建 DataFrame
    df_summary = pd.DataFrame(summary_data)

    # 格式化显示（保留4位小数）
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(df_summary)

    # 保存汇总 CSV
    summary_path = os.path.join(results_dir, 'final_summary_metrics.csv')
    df_summary.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"\n✅ Final summary table saved to: {summary_path}")

if __name__ == '__main__':
    main()