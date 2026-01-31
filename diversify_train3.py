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
from diversify_algorithm import Diversify  # 核心算法
from config import Config  # 配置参数

# === SelfA / SelfB 数据集 ===
from dataset_spilt_selfA import prepare_self_detailed, prepare_mixed_self_domains
from dataset_spilt_selfB import prepare_self_b_detailed, prepare_mixed_self_b_domains

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

    # 动态计算perplexity（保持原有逻辑）
    num_samples = len(features)
    perplexity_value = min(30, max(5, num_samples // 2))

    # 优化步骤1：PCA预处理
    if features.shape[1] > 50:  # 仅在特征维度高时启用
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

    # 版本兼容性处理
    try:
        from sklearn.manifold import TSNE
        # 新版sklearn添加加速参数
        tsne_params.update({
            'method': 'barnes_hut',
            'angle': 0.5,
            'n_jobs': -1,
            'init': 'pca'
        }) if hasattr(TSNE, 'method') else None
    except ImportError:
        pass

    # 执行t-SNE（耗时部分）
    print(f"⏳ 开始t-SNE计算，样本量{num_samples}，perplexity={perplexity_value}...")
    tsne = TSNE(**tsne_params)
    start_time = time.time()
    features_2d = tsne.fit_transform(features)
    print(f"✅ t-SNE完成，耗时{time.time() - start_time:.1f}秒")

    # 优化步骤3：高效绘图
    plt.figure(figsize=(4.5, 4))

    # 颜色代表 3 个潜在域 (Domain 0, 1, 2)
    DOMAIN_COLORS = ['#d36a87', '#ea9979', '#619cf5']
    # 形状代表 4 个故障类别 (Normal, Inner, Ball, Outer)
    CLASS_MARKERS = ['o', 'v', 's', 'd']
    CLASS_NAMES = {0: "Normal", 1: "Inner", 2: "Ball", 3: "Outer"}

    unique_domains = np.unique(labels).astype(int)  # 动态获取现有的域标签 [0] 或 [1] 或 [3]
    # ========== 新增字体设置 ==========
    plt.rcParams['font.family'] = 'Arial'  # 设置全局字体
    plt.rcParams['xtick.labelsize'] = 12  # X轴刻度字体
    plt.rcParams['ytick.labelsize'] = 12  # Y轴刻度字体

    # 4. 显式嵌套循环绘制，确保样式与标签一一对应
    for c_id in range(4):  # 遍历 4 个故障类
        for d_id in unique_domains: # 修改这里：不再死循环 range(3)
            mask = (class_labels == c_id) & (labels == d_id)
            if not np.any(mask):
                continue

            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=DOMAIN_COLORS[d_id % len(DOMAIN_COLORS)],  # Domain 决定颜色
                marker=CLASS_MARKERS[c_id % len(CLASS_MARKERS)],  # Class 决定形状
                s=50,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.6,
                label=f"{CLASS_NAMES.get(c_id, c_id)} (Dom {d_id})"
            )

    # 显式设置刻度（兼容性保障）
    # 坐标轴设置
    plt.xticks(fontsize=14)  # 更大的坐标轴字体
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)  # 添加网格线

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title="Class & Latent Domain")

    # 保存输出
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ 可视化保存至 {save_path}")
    plt.close()
# ================= 主程序 =================

def main():
    # 1. 设置实验环境
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

    # =========================================================================
    #                                 实验配置区域
    # =========================================================================

    # MODE = 'SINGLE'
    # DATASET_TYPE = 'SELFA'
    # SOURCE_DOMAINS = ['1200']

    # MODE = 'SINGLE'
    # DATASET_TYPE = 'SELFA'
    # SOURCE_DOMAINS = ['1800']
    #
    # MODE = 'SINGLE'
    # DATASET_TYPE = 'SELFB'
    # SOURCE_DOMAINS = ['1200']

    # MODE = 'SINGLE'
    # DATASET_TYPE = 'SELFB'
    # SOURCE_DOMAINS = ['1800']
    #
    # MODE = 'MULTI'
    # DATASET_TYPE = 'SELFA'
    # SOURCE_DOMAINS = ['1200', '1800']
    #
    MODE = 'MULTI'
    DATASET_TYPE = 'SELFB'
    SOURCE_DOMAINS = ['1200', '1800']

    # =========================================================================

    print("\n" + "=" * 70)
    print("🚀 Start Diversify Training")
    print(f"   Dataset: {DATASET_TYPE}")
    print(f"   Mode:    {MODE}")
    print(f"   Sources: {SOURCE_DOMAINS}")
    print("=" * 70)

    # 1) 数据加载
    if DATASET_TYPE == 'SELFA':
        if MODE == 'SINGLE':
            domain = SOURCE_DOMAINS[0]
            loaders = prepare_self_detailed()[domain]
            task_name = f"SelfA_Single_{domain}"
        else:
            loaders = prepare_mixed_self_domains(SOURCE_DOMAINS)
            task_name = f"SelfA_Multi_{'_'.join(SOURCE_DOMAINS)}"

    elif DATASET_TYPE == 'SELFB':
        if MODE == 'SINGLE':
            domain = SOURCE_DOMAINS[0]
            loaders = prepare_self_b_detailed()[domain]
            task_name = f"SelfB_Single_{domain}"
        else:
            loaders = prepare_mixed_self_b_domains(SOURCE_DOMAINS)
            task_name = f"SelfB_Multi_{'_'.join(SOURCE_DOMAINS)}"
    else:
        raise ValueError(f"Invalid DATASET_TYPE: {DATASET_TYPE}")

    train_loader = loaders['train']
    val_loader = loaders['val']


    # 3. 初始化记录器与保存路径
    metrics = {
        'round': [], 'epoch': [], 'phase': [],
        'class_loss': [], 'dis_loss': [], 'ent_loss': [], 'total_loss': [],
        'train_acc': [], 'val_acc': [], 'time_elapsed': []
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', 'train_output_diversify_OOD2', f'{task_name}_{timestamp}')
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

        # 1. 潜在域特征更新
        print("Stage 1: Feature Update")
        for epoch in range(config.local_epochs):
            start_time = time.time()
            epoch_losses = []

            for data in tqdm(train_loader, desc=f"Feature Update (Epoch {epoch + 1}/{config.local_epochs})"):
                loss_dict = algorithm.update_a(data, optimizer_all)
                epoch_losses.append(loss_dict['class'])

            avg_class_loss = np.mean(epoch_losses)
            time_elapsed = time.time() - start_time

            print(f"Epoch {epoch + 1}/{config.local_epochs}, Class Loss: {avg_class_loss:.4f}, "
                  f"Time: {time_elapsed:.2f}s")

            # 记录指标
            metrics['round'].append(round_idx + 1)
            metrics['epoch'].append(epoch + 1)
            metrics['phase'].append('feature_update')

            metrics['class_loss'].append(avg_class_loss)
            metrics['dis_loss'].append(None)
            metrics['ent_loss'].append(None)
            metrics['total_loss'].append(None)

            metrics['train_acc'].append(None)
            metrics['val_acc'].append(None)
            metrics['time_elapsed'].append(time_elapsed)

        # 2. 域标签更新
        print("Stage 2: Latent Domain Characterization")
        for epoch in range(config.local_epochs):
            start_time = time.time()
            epoch_total_losses = []
            epoch_dis_losses = []
            epoch_ent_losses = []

            for data in tqdm(train_loader,
                             desc=f"Domain Characterization (Epoch {epoch + 1}/{config.local_epochs})"):
                loss_dict = algorithm.update_d(data, optimizer_adv)
                epoch_total_losses.append(loss_dict['total'])
                epoch_dis_losses.append(loss_dict['dis'])
                epoch_ent_losses.append(loss_dict['ent'])

            avg_total_loss = np.mean(epoch_total_losses)
            avg_dis_loss = np.mean(epoch_dis_losses)
            avg_ent_loss = np.mean(epoch_ent_losses)
            time_elapsed = time.time() - start_time

            print(f"Epoch {epoch + 1}/{config.local_epochs}, " +
                  f"Total Loss: {avg_total_loss:.4f}, " +
                  f"Disc Loss: {avg_dis_loss:.4f}, " +
                  f"Ent Loss: {avg_ent_loss:.4f}, " +
                  f"Time: {time_elapsed:.2f}s")

            # 保存指标
            metrics['round'].append(round_idx + 1)
            metrics['epoch'].append(epoch + 1)
            metrics['phase'].append('domain_characterization')

            metrics['class_loss'].append(None)
            metrics['dis_loss'].append(avg_dis_loss)
            metrics['ent_loss'].append(avg_ent_loss)
            metrics['total_loss'].append(avg_total_loss)

            metrics['train_acc'].append(None)
            metrics['val_acc'].append(None)
            metrics['time_elapsed'].append(time_elapsed)

            algorithm.set_dlabel(train_loader)                            #这行代码调用 algorithm 实例的 set_dlabel 方法，并将 train_loader 作为参数传入。该方法可能用于设置训练数据的域标签。

            # # 在调用set_dlabel之前，打印部分dlabel值
            # print("Before set_dlabel:",
            #       train_loader.dataset.subset.dataset.dlabels[:10])  # 注意如果是Subset包装的，需要通过这个路径访问底层dataset
            #
            # # 调用set_dlabel
            # algorithm.set_dlabel(train_loader)
            #
            # # 调用之后，打印同样位置的dlabel值，观察是否更新
            # print("After set_dlabel:", train_loader.dataset.subset.dataset.dlabels[:10])

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