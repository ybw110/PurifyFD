import time
import torch
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
# === 导入自定义模块 ===
from 真实物理标签_diversify_algorithm import Diversify  # 核心算法
from config import Config  # 配置参数
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS

from collections import Counter  # 必须导入，用于统计配对频数
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns            # 用于绘制热力图
from sklearn.metrics import confusion_matrix # 用于计算对齐矩阵

# 实例化配置
config = Config()

# 定义支持的算法
ALGORITHMS = ['diversify']

# ================= 辅助函数 =================
def get_algorithm_class(algorithm_name):
    """返回算法类"""
    if algorithm_name.lower() not in ALGORITHMS:
        raise NotImplementedError(f"Algorithm not found: {algorithm_name}")
    return Diversify  # 直接返回 Diversify 类

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_row(values, colwidth=15):
    """打印对齐的表格行"""
    print("".join([str(x).ljust(colwidth) for x in values]))


def accuracy(algorithm, loader, weights=None):
    """计算准确率"""
    algorithm.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = algorithm.predict(x)

            if weights is not None:
                w = weights[y].cuda()
            else:
                w = torch.ones(len(y)).cuda()

            correct += (p.argmax(1) == y).float().mul(w).sum().item()
            total += w.sum().item()

    algorithm.train()
    return correct / total if total > 0 else 0.0


def save_metrics(metrics, save_dir, filename):
    """保存训练指标到文件"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, filename)
    df = pd.DataFrame(metrics)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def plot_features(features, labels, class_labels, title="Feature Visualization", save_path=None):
    # 数据校验
    if len(features) == 0 or len(labels) == 0:
        print(f"❌ [ERROR] {title} 没有数据，跳过绘制！")
        return

    # 数据预处理优化
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    class_labels = np.concatenate(class_labels, axis=0)

    # 可选：减少样本数量，提高可读性
    max_samples = 300  # 限制最多显示的样本数
    if len(features) > max_samples:
        print(f"⚠️ 样本数量过多({len(features)})，随机选择{max_samples}个进行可视化")
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
        class_labels = class_labels[indices]

    # 动态计算perplexity
    num_samples = len(features)
    perplexity_value = min(30, max(5, num_samples // 2))

    # 优化步骤1：PCA预处理
    if features.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95, random_state=42)
        features = pca.fit_transform(features)
        print(f"✅ PCA降维到{pca.n_components_}维，保留95%方差")

    # 优化步骤2：配置t-SNE参数
    tsne_params = {
        'n_components': 2,
        'perplexity': perplexity_value,
        'random_state': 42,
        'max_iter': 500,
        'learning_rate': 'auto'
    }

    try:
        from sklearn.manifold import TSNE
        tsne_params.update({
            'method': 'barnes_hut',
            'angle': 0.5,
            'n_jobs': -1,
            'init': 'pca'
        }) if hasattr(TSNE, 'method') else None
    except ImportError:
        pass

    # 执行t-SNE
    print(f"⏳ 开始t-SNE计算，样本量{num_samples}，perplexity={perplexity_value}...")
    tsne = TSNE(**tsne_params)
    start_time = time.time()
    features_2d = tsne.fit_transform(features)
    print(f"✅ t-SNE完成，耗时{time.time() - start_time:.1f}秒")

    # 优化步骤3：高效绘图
    plt.figure(figsize=(5, 4))

    # ================= 修改点：扩充配色方案以支持 K=5 =================
    DOMAIN_COLORS = [
        '#5D8CA8',  # SS-A
        '#E0AC69',  # SS-B
        '#D99488',  # SS-C
        '#AE93A3',  # ST-D
        '#98C0DE',  # ST-1
        '#CDD192'  # ST-2
        '#C3D3E0',  # SS-A Light
        '#F5E1CC',  # SS-B Light
        '#F0D3CE',  # SS-C Light
        '#E3D8DE',  # ST-D Light
        '#DCEAF2',  # ST-1 Light
        '#E9EBCF'  # ST-2 Light
    ]
    # =============================================================

    # 形状代表 4 个故障类别
    CLASS_MARKERS = ['o', 'v', 's', 'd']
    CLASS_NAMES = {0: "Normal", 1: "Inner", 2: "Ball", 3: "Outer"}

    unique_domains = np.unique(labels).astype(int)

    # ========== 字体设置 ==========
    plt.rcParams['font.family'] = 'Arial'

    # 4. 显式嵌套循环绘制
    # 注意：这里假设传入的 class_labels 是物理标签 (0,1,2,3)
    for c_id in range(4):
        for d_id in unique_domains:
            mask = (class_labels == c_id) & (labels == d_id)
            if not np.any(mask):
                continue

            # 确保 d_id 不会越界 (d_id % len)
            color_idx = d_id % len(DOMAIN_COLORS)

            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=DOMAIN_COLORS[color_idx],  # Domain 决定颜色
                marker=CLASS_MARKERS[c_id % len(CLASS_MARKERS)],  # Class 决定形状
                s=70,
                alpha=0.8,
                edgecolor='k',  # 你的白色描边风格
                linewidth=0.6,
                label=f"{CLASS_NAMES.get(c_id, c_id)} (Dom {d_id})"
            )

    # 移除刻度（保留你的风格）
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title="Class & Latent Domain")

    # 保存输出
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ 可视化保存至 {save_path}")
    plt.close()
# ================= 主程序 =================


def plot_domain_alignment_heatmap(true_domains, latent_domains, physical_names, save_path):
    """
    方案 B: 混淆矩阵热力图
    展示物理工况 (True) 与 潜在域 (Latent) 的对齐程度
    """
    plt.figure(figsize=(8, 6))

    # 计算频数矩阵
    cm = confusion_matrix(true_domains, latent_domains)
    # 归一化：每一行（物理工况）求和为1，展示该物理工况的样本流向了哪些潜在域
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    latent_names = [f'Latent {i}' for i in range(cm.shape[1])]

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=latent_names, yticklabels=physical_names)

    plt.title("Alignment: Physical Domains vs. Mined Latent Domains")
    plt.xlabel("Mined Latent Domains")
    plt.ylabel("Ground Truth Physical Domains")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_domain_sankey(true_domains, latent_domains, physical_names, save_path):
    """
    方案 A: 简易流向图 (桑基图逻辑)
    使用 Matplotlib 模拟连线粗细，展示对应关系
    """
    from matplotlib.path import Path
    import matplotlib.patches as patches

    plt.figure(figsize=(10, 7))

    # 统计配对数量
    pairs = list(zip(true_domains, latent_domains))
    counts = Counter(pairs)
    total_samples = len(true_domains)

    unique_true = np.unique(true_domains)
    unique_latent = np.unique(latent_domains)

    # 坐标设置
    left_x, right_x = 0.2, 0.8
    y_true = {val: 1 - (i + 1) / (len(unique_true) + 1) for i, val in enumerate(unique_true)}
    y_latent = {val: 1 - (i + 1) / (len(unique_latent) + 1) for i, val in enumerate(unique_latent)}

    # 绘制节点
    for val, y in y_true.items():
        name = physical_names[val] if val < len(physical_names) else f"P-{val}"
        plt.text(left_x - 0.05, y, name, ha='right', va='center', fontsize=12, fontweight='bold')
        plt.plot(left_x, y, 'ro', markersize=10)

    for val, y in y_latent.items():
        plt.text(right_x + 0.05, y, f"Latent {val}", ha='left', va='center', fontsize=12, fontweight='bold')
        plt.plot(right_x, y, 'bo', markersize=10)

    # 绘制连线 (粗细代表样本比例)
    for (t, l), count in counts.items():
        width = (count / total_samples) * 50  # 缩放因子
        color = plt.cm.get_cmap('tab10')(t % 10)

        # 绘制贝塞尔曲线
        verts = [(left_x, y_true[t]), (0.5, y_true[t]), (0.5, y_latent[l]), (right_x, y_latent[l])]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=width, alpha=0.3)
        plt.gca().add_patch(patch)

    plt.axis('off')
    plt.title("Sankey Diagram: Physical to Latent Mapping", fontsize=15)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()



def main():
    # 1. 设置实验环境
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

    # [新增] 打印开集配置
    print("\n" + "=" * 40)
    print(" OPEN SET CONFIGURATION")
    print(f" Known Classes: {config.known_classes}")
    print(f" Unknown Classes: {config.unknown_classes}")
    print(f" Model Output Dim: {config.num_classes}")
    print("=" * 40 + "\n")

    # ================= 配置数据源 =================

    # --- 模式 A: 单源域设置 (Single Source) ---
    # MODE = 'SINGLE'
    # DATASET_TYPE = 'CWRU'       # 'CWRU' or 'JNU'
    # SOURCE_DOMAINS = '3HP'         # '0HP', '1HP'... or '600', '800'...

    # 选项 B: 使用 JNU (若想切换，取消下面两行注释并注释掉上面两行)
    # MODE = 'SINGLE'
    # DATASET_TYPE = 'JNU'
    # SOURCE_DOMAINS = '600'  # 可选: '600', '800', '1000'


    # ================= 配置数据源 (多源域设置) =================

    # === 选项 A: CWRU ===
    # MODE = 'MULTI'
    # DATASET_TYPE = 'CWRU'
    # SOURCE_DOMAINS = ['0HP', '1HP', '2HP']  # 3HP作为测试(含OOD

    # === 选项 B: JNU (若想切换，取消注释下方) ===
    # MODE = 'MULTI'
    # DATASET_TYPE = 'JNU'
    # SOURCE_DOMAINS = ['600', '800']
    # ==========================================================

    # --- 模式 C: 完整数据设置 (Full Dataset) ---
    # 只要在 SOURCE_DOMAINS 里填入所有工况，即为 Full Dataset 训练
    MODE = 'MULTI'
    DATASET_TYPE = 'CWRU'
    SOURCE_DOMAINS = ALL_DOMAINS        # 或者: SOURCE_DOMAINS = ['0HP', '1HP', '2HP', '3HP']

    # MODE = 'MULTI'
    # DATASET_TYPE = 'JNU'
    # SOURCE_DOMAINS = ALL_SPEEDS      # ['600', '800', '1000']

    # 1. 数据加载逻辑
    print(f"Current Mode: {MODE} | Dataset: {DATASET_TYPE}")

    if MODE == 'SINGLE':
        print(f"Source Domain: {SOURCE_DOMAINS}")
        task_name = f"SingleSource_{SOURCE_DOMAINS}"

        if DATASET_TYPE == 'CWRU':
            all_data = prepare_all_domains_detailed()
            loaders = all_data[SOURCE_DOMAINS]
        else:
            all_data = prepare_jnu_detailed(jnu_root)
            loaders = all_data[SOURCE_DOMAINS]

    elif MODE == 'MULTI':
        print(f"Source Domains: {SOURCE_DOMAINS}")
        task_name = f"MultiSource_{'_'.join(SOURCE_DOMAINS)}"

        if DATASET_TYPE == 'CWRU':
            loaders = prepare_mixed_domains(SOURCE_DOMAINS)
        else:
            loaders = prepare_mixed_jnu_domains(SOURCE_DOMAINS, jnu_root)

    else:
        raise ValueError("Invalid Mode")

    train_loader = loaders['train']
    val_loader = loaders['val']


    # 3. 初始化记录器与保存路径
    metrics = {
        'round': [], 'epoch': [], 'phase': [],
        'class_loss': [], 'dis_loss': [], 'ent_loss': [], 'total_loss': [],
        'train_acc': [], 'val_acc': [], 'time_elapsed': []
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', 'train_output_diversify_OOD2_True_label', f'{task_name}_{timestamp}')
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)


    # 4. 初始化模型与优化器
    algorithm_class = get_algorithm_class(config.algorithm)
    algorithm = algorithm_class(config).to(device)
    algorithm.train()

    # 优化器配置 (保持原逻辑)
    params_adv = [
        {'params': algorithm.dbottleneck.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.dclassifier.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.ddiscriminator.parameters(), 'lr': config.lr_decay2 * config.lr}
    ]
    optimizer_adv = torch.optim.Adam(params_adv, lr=config.lr, weight_decay=config.weight_decay,betas=(config.beta1, 0.99))

    params_cls = [
        {'params': algorithm.bottleneck.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.classifier.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.discriminator.parameters(), 'lr': config.lr_decay2 * config.lr}
    ]
    optimizer_cls = torch.optim.Adam(params_cls, lr=config.lr, weight_decay=config.weight_decay,betas=(config.beta1, 0.99))

    params_all = [
        {'params': algorithm.featurizer.parameters(), 'lr': config.lr_decay1 * config.lr},
        {'params': algorithm.abottleneck.parameters(), 'lr': config.lr_decay2 * config.lr},
        {'params': algorithm.aclassifier.parameters(), 'lr': config.lr_decay2 * config.lr}
    ]
    optimizer_all = torch.optim.Adam(params_all, lr=config.lr, weight_decay=config.weight_decay,betas=(config.beta1, 0.99))

    best_valid_acc = 0

    # ================= 训练循环 =================
    for round_idx in range(config.max_rounds):
        print(f"\n====== ROUND {round_idx + 1}/{config.max_rounds} ======")

        if round_idx == config.max_rounds - 1:  # 仅在最后一个 epoch 记录 t-SNE 数据
            algorithm.is_last_epoch = True
            # algorithm.tsne_features_before_d.clear()
            # algorithm.tsne_domain_labels_before_d.clear()
            algorithm.tsne_features_d.clear()
            algorithm.tsne_domain_labels_d.clear()
            algorithm.tsne_features.clear()
            algorithm.tsne_domain_labels.clear()
        else:
            algorithm.is_last_epoch = False

        if round_idx == 0:
            algorithm.record_initial_features(train_loader)  # 记录 update_d 之前的初始特征

        # # 1. 潜在域特征更新
        # print("Stage 1: Feature Update")
        # for epoch in range(config.local_epochs):
        #     start_time = time.time()
        #     epoch_losses = []
        #
        #     for data in tqdm(train_loader, desc=f"Feature Update (Epoch {epoch + 1}/{config.local_epochs})"):
        #         loss_dict = algorithm.update_a(data, optimizer_all)
        #         epoch_losses.append(loss_dict['class'])
        #
        #     avg_class_loss = np.mean(epoch_losses)
        #     time_elapsed = time.time() - start_time
        #
        #     print(f"Epoch {epoch + 1}/{config.local_epochs}, Class Loss: {avg_class_loss:.4f}, "
        #           f"Time: {time_elapsed:.2f}s")
        #
        #     # 记录指标
        #     metrics['round'].append(round_idx + 1)
        #     metrics['epoch'].append(epoch + 1)
        #     metrics['phase'].append('feature_update')
        #
        #     metrics['class_loss'].append(avg_class_loss)
        #     metrics['dis_loss'].append(None)
        #     metrics['ent_loss'].append(None)
        #     metrics['total_loss'].append(None)
        #
        #     metrics['train_acc'].append(None)
        #     metrics['val_acc'].append(None)
        #     metrics['time_elapsed'].append(time_elapsed)
        #
        # # 2. 域标签更新
        # print("Stage 2: Latent Domain Characterization")
        # for epoch in range(config.local_epochs):
        #     start_time = time.time()
        #     epoch_total_losses = []
        #     epoch_dis_losses = []
        #     epoch_ent_losses = []
        #
        #     for data in tqdm(train_loader,
        #                      desc=f"Domain Characterization (Epoch {epoch + 1}/{config.local_epochs})"):
        #         loss_dict = algorithm.update_d(data, optimizer_adv)
        #         epoch_total_losses.append(loss_dict['total'])
        #         epoch_dis_losses.append(loss_dict['dis'])
        #         epoch_ent_losses.append(loss_dict['ent'])
        #
        #     avg_total_loss = np.mean(epoch_total_losses)
        #     avg_dis_loss = np.mean(epoch_dis_losses)
        #     avg_ent_loss = np.mean(epoch_ent_losses)
        #     time_elapsed = time.time() - start_time
        #
        #     print(f"Epoch {epoch + 1}/{config.local_epochs}, " +
        #           f"Total Loss: {avg_total_loss:.4f}, " +
        #           f"Disc Loss: {avg_dis_loss:.4f}, " +
        #           f"Ent Loss: {avg_ent_loss:.4f}, " +
        #           f"Time: {time_elapsed:.2f}s")
        #
        #     # 保存指标
        #     metrics['round'].append(round_idx + 1)
        #     metrics['epoch'].append(epoch + 1)
        #     metrics['phase'].append('domain_characterization')
        #
        #     metrics['class_loss'].append(None)
        #     metrics['dis_loss'].append(avg_dis_loss)
        #     metrics['ent_loss'].append(avg_ent_loss)
        #     metrics['total_loss'].append(avg_total_loss)
        #
        #     metrics['train_acc'].append(None)
        #     metrics['val_acc'].append(None)
        #     metrics['time_elapsed'].append(time_elapsed)
        #
        #     algorithm.set_dlabel(train_loader)                            #这行代码调用 algorithm 实例的 set_dlabel 方法，并将 train_loader 作为参数传入。该方法可能用于设置训练数据的域标签。


        # 3. 域不变特征学习
        print("Stage 3: Domain-invariant Feature Learning")  # 进入域不变特征学习阶段：

        for epoch in range(config.local_epochs):
            start_time = time.time()
            epoch_total_losses = []
            epoch_class_losses = []
            epoch_dis_losses = []

            for data in tqdm(train_loader,desc=f"Domain-invariant Learning (Epoch {epoch + 1}/{config.local_epochs})"):
                loss_dict = algorithm.update(data, optimizer_cls)
                epoch_total_losses.append(loss_dict['total'])
                epoch_class_losses.append(loss_dict['class'])
                epoch_dis_losses.append(loss_dict['dis'])

            # 计算训练指标
            avg_total_loss = np.mean(epoch_total_losses)
            avg_class_loss = np.mean(epoch_class_losses)
            avg_dis_loss = np.mean(epoch_dis_losses)

            train_acc = accuracy(algorithm, train_loader, None)
            val_acc = accuracy(algorithm, val_loader, None)

            time_elapsed = time.time() - start_time

            print(f"Epoch {epoch + 1}/{config.local_epochs}, " +
                  f"Total Loss: {avg_total_loss:.4f}, " +
                  f"Class Loss: {avg_class_loss:.4f}, " +
                  f"Disc Loss: {avg_dis_loss:.4f}, " +
                  f"Train Acc: {train_acc:.4f}, " +
                  f"Val Acc: {val_acc:.4f}, " +
                  f"Time: {time_elapsed:.2f}s")

            # 记录指标
            metrics['round'].append(round_idx + 1)
            metrics['epoch'].append(epoch + 1)
            metrics['phase'].append('domain_invariant')

            metrics['class_loss'].append(avg_class_loss)
            metrics['dis_loss'].append(avg_dis_loss)
            metrics['ent_loss'].append(None)
            metrics['total_loss'].append(avg_total_loss)

            metrics['train_acc'].append(train_acc)
            metrics['val_acc'].append(val_acc)
            metrics['time_elapsed'].append(time_elapsed)


            # 保存最佳模型逻辑（基于 target_acc）
            if val_acc > best_valid_acc:
                best_valid_acc = val_acc  # 更新最佳 val_acc
                torch.save(algorithm.state_dict(), os.path.join(save_dir, 'best_model.pth'))  # 保存模型

                # 保存检查点包含更多信息
                checkpoint = {
                    'model_state_dict': algorithm.state_dict(),
                    'optimizer_all_state_dict': optimizer_all.state_dict(),
                    'optimizer_adv_state_dict': optimizer_adv.state_dict(),
                    'optimizer_cls_state_dict': optimizer_cls.state_dict(),
                    'round': round_idx + 1,
                    'epoch': epoch + 1,
                    'config': config,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_checkpoint.pth'))

            # 每个round结束保存一次指标
        save_metrics(metrics, log_dir, f'metrics_round_{round_idx + 1}_{timestamp}.csv')
        if round_idx == 0:
            plot_features(algorithm.tsne_features_before_d,
                          algorithm.tsne_domain_labels_before_d,
                          algorithm.tsne_class_labels_before_d,

                          "Initial Features (All Domain 0)",
                          f"{save_dir}/tsne_init.png")

        if round_idx == config.max_rounds - 1:
            plot_features(algorithm.tsne_features_d,
                          algorithm.tsne_domain_labels_d,
                          algorithm.tsne_class_labels_d,

                          "After Domain Update",
                          f"{save_dir}/tsne_after_d.png")

            plot_features(algorithm.tsne_features,
                          algorithm.tsne_domain_labels,
                          algorithm.tsne_class_labels,

                          "Final Features",
                          f"{save_dir}/tsne_final.png")

        # 1. 获取物理工况名称列表
        if DATASET_TYPE == 'CWRU':
            physical_names = SOURCE_DOMAINS  # 如 ['0HP', '1HP', '2HP', '3HP']
        else:
            physical_names = SOURCE_DOMAINS  # 如 ['600', '800', '1000']

        # 2. 从 train_loader 的 dataset 中提取已更新的标签
        # 注意：在 Stage 2 结束后，dataset.pdlabels 已经被 set_dlabel 更新了
        all_true_physical = train_loader.dataset.true_domains
        all_mined_latent = train_loader.dataset.pdlabels

        print("⏳ 正在生成域对齐分析图...")

        # 定义物理工况的名称列表（对应你的 SOURCE_DOMAINS）
        physical_names = SOURCE_DOMAINS

        # 绘图
        # 3. 调用绘图函数
        plot_domain_alignment_heatmap(all_true_physical, all_mined_latent,
                                      physical_names, f"{save_dir}/domain_alignment_heatmap.png")

        plot_domain_sankey(all_true_physical, all_mined_latent,
                           physical_names, f"{save_dir}/domain_alignment_sankey.png")
        print(f"✅ 域对齐图已保存至 {save_dir}")

    # # 加载最佳模型
    # algorithm.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    #
    # # 保存最终模型(包含所有必要组件)
    # model_dict = {
    #     'algorithm': algorithm,
    #     'config': config,
    #     'val_acc': val_acc,
    #     'timestamp': timestamp
    # }
    #
    # torch.save(model_dict, os.path.join(save_dir, 'diversify_model.pt'))
    # 保存完整指标记录
    save_metrics(metrics, log_dir, f'complete_metrics_{timestamp}.csv')


if __name__ == '__main__':
    main()