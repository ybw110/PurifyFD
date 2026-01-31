import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 导入自定义模块 ===
# 确保你已经保存了之前的 aminet_model.py
from baseline_AMINet_model import DomainGenerator, FeatureExtractor, Classifier, CLUB, CMMDLoss, SupConLoss
from config import Config
# 1. SelfA (1楼数据集) - 直接导入原函数名
from dataset_spilt_selfA import prepare_self_detailed, prepare_mixed_self_domains
from dataset_spilt_selfA import ALL_SPEEDS as SELFA_SPEEDS

# 2. SelfB (6楼数据集) - 直接导入原函数名
from dataset_spilt_selfB import prepare_self_b_detailed, prepare_mixed_self_b_domains
from dataset_spilt_selfB import ALL_SPEEDS as SELFB_SPEEDS


# === t-SNE 相关 ===
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

config = Config()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_training_history(metrics, save_path):
    """绘制训练损失和准确率曲线 (风格与 SampleCNN 一致)"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制 Loss (这里绘制 Task Loss 作为主要参考)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss (Task)', color='tab:red', fontsize=14)
    line1 = ax1.plot(metrics['epoch'], metrics['train_loss'], color='tab:red', linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=12)
    ax1.grid(True, alpha=0.3)

    # 绘制 Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue', fontsize=14)
    line3 = ax2.plot(metrics['epoch'], metrics['train_acc'], '--', color='tab:cyan', linewidth=2, label='Train Acc')
    line4 = ax2.plot(metrics['epoch'], metrics['val_acc'], color='tab:blue', linewidth=2, label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)

    # 合并图例
    lines = line1 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=11, framealpha=0.9)

    plt.title('Training History (AMINet)', fontsize=16, pad=15)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
#           适配 AMINet 的 t-SNE 提取函数
# ==========================================
def extract_features_aminet(G, loader, device, max_samples=2000):
    """提取 AMINet 特征提取器 (G) 的特征"""
    G.eval()
    features = []
    labels = []
    total_count = 0

    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device).float()
            targets = data[1].to(device).long()

            # AMINet 特征提取
            feat = G(inputs)

            features.append(feat.cpu().numpy())
            labels.append(targets.cpu().numpy())

            total_count += len(targets)
            if total_count >= max_samples:
                break

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    if len(features) > max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]

    return features, labels


def plot_tsne_for_training(G, loader, device, save_path, title="AMINet Features"):
    print(f"\n🎨 正在绘制 AMINet 训练集 t-SNE: {title} ...")

    # 提取特征
    feats, labels = extract_features_aminet(G, loader, device, max_samples=300)

    # PCA 降维加速
    if feats.shape[1] > 50:
        pca = PCA(n_components=50)
        feats = pca.fit_transform(feats)

    # t-SNE 计算
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(feats)

    # 绘图
    plt.figure(figsize=(5, 4))
    plt.rcParams['font.family'] = 'Arial'

    # 统一配色 (Normal, Inner, Ball, Outer)
    colors = ['#6D95C3', '#EBB577', '#CDD797', '#E09D94']
    class_names = ['Normal', 'Inner', 'Ball', 'Outer']
    markers = ['o', '^', 's', 'D']

    for c in range(4):
        idx = (labels == c)
        if np.sum(idx) > 0:
            plt.scatter(
                embeddings[idx, 0], embeddings[idx, 1],
                c=colors[c], marker=markers[c], label=class_names[c],
                s=50, alpha=0.8, edgecolors='k', linewidth=0.6
            )

    plt.legend(loc='best', fontsize=14)
    plt.xticks([]);
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE 图已保存至: {save_path}")


# ==========================================
#               主训练逻辑
# ==========================================
def main():
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

    # =========================================================================
    #                                 实验配置区域
    # =========================================================================

    MODE = 'SINGLE'
    DATASET_TYPE = 'SELFA'
    SOURCE_DOMAINS = ['1200']

    # MODE = 'SINGLE'
    # DATASET_TYPE = 'SELFA'
    # SOURCE_DOMAINS = ['1800']
    # #
    # MODE = 'SINGLE'
    # DATASET_TYPE = 'SELFB'
    # SOURCE_DOMAINS = ['1200']
    #
    # MODE = 'SINGLE'
    # DATASET_TYPE = 'SELFB'
    # SOURCE_DOMAINS = ['1800']
    #
    # MODE = 'MULTI'
    # DATASET_TYPE = 'SELFA'
    # SOURCE_DOMAINS = ['1200', '1800']
    #
    # MODE = 'MULTI'
    # DATASET_TYPE = 'SELFB'
    # SOURCE_DOMAINS = ['1200', '1800']

    # =========================================================================

    print(f"\n" + "=" * 60)
    print(f"🚀 Start Training")
    print(f"   Dataset: {DATASET_TYPE}")
    print(f"   Mode:    {MODE}")
    print(f"   Sources: {SOURCE_DOMAINS}")
    print("=" * 60)

    # 1. 数据加载逻辑 (仅支持 SelfA 和 SelfB)
    if DATASET_TYPE == 'SELFA':
        if MODE == 'SINGLE':
            domain = SOURCE_DOMAINS[0]
            loaders = prepare_self_detailed()[domain]
            task_name = f"AMINet_SelfA_Single_{domain}"
        else:
            loaders = prepare_mixed_self_domains(SOURCE_DOMAINS)
            task_name = f"AMINet_SelfA_Multi_{'_'.join(SOURCE_DOMAINS)}"

    elif DATASET_TYPE == 'SELFB':
        if MODE == 'SINGLE':
            domain = SOURCE_DOMAINS[0]
            loaders = prepare_self_b_detailed()[domain]
            task_name = f"AMINet_SelfB_Single_{domain}"
        else:
            loaders = prepare_mixed_self_b_domains(SOURCE_DOMAINS)
            task_name = f"AMINet_SelfB_Multi_{'_'.join(SOURCE_DOMAINS)}"
    else:
        raise ValueError(f"Unsupported DATASET_TYPE: {DATASET_TYPE}")


    train_loader = loaders['train']
    val_loader = loaders['val']

    # ================= 2. 初始化与路径 =================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', 'aminet', f'{task_name}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)

    # === 初始化 AMINet 组件 ===
    # 注意：AMINet 由四个独立模块组成
    D = DomainGenerator(input_channel=1).to(device)
    G = FeatureExtractor().to(device)
    C = Classifier(num_classes=config.num_classes).to(device)
    q_net = CLUB(feature_dim=256).to(device)  # MI 估计网络

    # === 优化器 (分开定义) ===
    # 1. 域生成器 D
    opt_D = optim.Adam(D.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # 2. 任务模块 T (G + C)
    opt_T = optim.Adam(list(G.parameters()) + list(C.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    # 3. MI 估计网络 q
    opt_q = optim.Adam(q_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # === 损失函数 ===
    criterion_cls = nn.CrossEntropyLoss()
    criterion_cmmd = CMMDLoss()
    criterion_supcon = SupConLoss(temperature=0.07)

    # === 超参数 (参考论文) ===
    beta = 1.0  # MI Minimization weight
    alpha = 1.0  # SupCon weight

    # 记录
    metrics = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0

    # ================= 3. 预热阶段 (Pre-training T) =================
    # 简单训练几轮 G+C，防止初始特征太差导致 MI 计算不稳定
    PRETRAIN_EPOCHS = 5
    print(f"\n>>> [Phase 1] Pre-training Task Module for {PRETRAIN_EPOCHS} epochs...")
    # for epoch in range(PRETRAIN_EPOCHS):

    epoch_bar1 = tqdm(range(PRETRAIN_EPOCHS), desc="Pre-training", colour='cyan', ncols=100)
    for epoch in epoch_bar1:
        G.train();
        C.train()
        total_loss = 0

        for data in train_loader:
            x = data[0].to(device).float()
            y = data[1].to(device).long()

            opt_T.zero_grad()
            z = G(x)
            pred = C(z)
            loss = criterion_cls(pred, y)
            loss.backward()
            opt_T.step()

            total_loss += loss.item()

        # 在进度条末尾实时显示平均 Loss
        avg_loss = total_loss / len(train_loader)
        epoch_bar1.set_postfix({'Loss': f'{avg_loss:.4f}'})
    print(">>> Pre-training Finished.\n")

    # ================= 4. 正式训练 (AMINet Algorithm) =================
    print(f">>> [Phase 2] AMINet Adversarial Training for {config.epochs} epochs...")

    epoch_bar = tqdm(range(config.epochs), desc="Training Progress", colour='green', ncols=100)

    for epoch in epoch_bar:
        D.train();
        G.train();
        C.train();
        q_net.train()

        total_loss_t = 0  # 记录 Task 相关的 Loss
        correct = 0
        total = 0

        for data in train_loader:
            x_s = data[0].to(device).float()
            y_s = data[1].to(device).long()

            # --- Step 1: Update Domain Generator (D) ---
            # 目标: 最小化 MI(zs, z+) -> 生成差异大的样本

            x_plus = D(x_s)

            # 此时 G 固定
            with torch.no_grad():
                z_s = G(x_s)
                z_plus = G(x_plus)

            # MI 上界 + 语义约束
            mi_upper = q_net.mi_est(z_s, z_plus)
            loss_semantic = criterion_cmmd(z_s, z_plus, y_s)

            loss_D = beta * mi_upper + loss_semantic

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Step 2: Update Task Module (T = G+C) & MI Estimator (q) ---
            # 目标: T 最大化准确率 & 互信息下界(拉近同类)

            # 重新生成 x_plus (断开 D 的梯度)
            with torch.no_grad():
                x_plus = D(x_s)

            z_s = G(x_s)
            z_plus = G(x_plus)  # 这里 G 需要梯度

            pred_s = C(z_s)
            pred_plus = C(z_plus)

            # 2.1 更新 MI 估计器 q (最大化似然)
            loss_likeli = q_net.learning_loss(z_s.detach(), z_plus.detach())
            opt_q.zero_grad()
            loss_likeli.backward()
            opt_q.step()

            # 2.2 更新 G 和 C
            # 分类 Loss (源域 + 伪域)
            loss_task_cls = criterion_cls(pred_s, y_s) + criterion_cls(pred_plus, y_s)

            # 对比 Loss (InfoNCE, 拉近 z_s 和 z_plus)
            features_cat = torch.cat([z_s, z_plus], dim=0)
            labels_cat = torch.cat([y_s, y_s], dim=0)
            loss_supcon = criterion_supcon(features_cat, labels_cat)

            loss_T = loss_task_cls + alpha * loss_supcon

            opt_T.zero_grad()
            loss_T.backward()
            opt_T.step()

            # 记录数据
            total_loss_t += loss_T.item()
            _, predicted = torch.max(pred_s, 1)
            total += y_s.size(0)
            correct += (predicted == y_s).sum().item()

        # 计算 Epoch 指标
        train_loss_epoch = total_loss_t / len(train_loader)
        train_acc_epoch = 100 * correct / total

        # === Validation ===
        G.eval();
        C.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for data in val_loader:
                # --- 修改点 3：Validation 也需要解包 ---
                x = data[0].to(device).float()
                y = data[1].to(device).long()

                z = G(x)
                pred = C(z)
                _, predicted = torch.max(pred, 1)
                v_total += y.size(0)
                v_correct += (predicted == y).sum().item()

        val_acc_epoch = 100 * v_correct / v_total

        # 更新进度条
        epoch_bar.set_postfix({
            'Loss': f'{train_loss_epoch:.4f}',
            'Train': f'{train_acc_epoch:.1f}%',
            'Val': f'{val_acc_epoch:.1f}%'
        })

        # 记录 History
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss_epoch)
        metrics['train_acc'].append(train_acc_epoch)
        metrics['val_acc'].append(val_acc_epoch)

        # 保存最佳模型
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            # 为了方便测试，我们把 G 和 C 保存到一个文件里
            # 如果你有 D 的需求也可以加进去，但测试通常只需要 G 和 C
            state_dict = {
                'G': G.state_dict(),
                'C': C.state_dict(),
                'best_acc': best_val_acc
            }
            torch.save(state_dict, os.path.join(save_dir, 'best_model.pth'))

    # ================= 5. 结束工作 (CSV & t-SNE) =================

    # 保存 CSV
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    # 绘制训练曲线
    plot_training_history(metrics, os.path.join(save_dir, 'training_curves.png'))

    # print("\n" + "=" * 50)
    # print(f"Best Val Acc: {best_val_acc:.2f}%")
    # print(">>> 训练结束，开始绘制 AMINet 训练集 t-SNE...")
    #
    # # 加载最佳模型
    # best_path = os.path.join(save_dir, 'best_model.pth')
    # if os.path.exists(best_path):
    #     checkpoint = torch.load(best_path)
    #     G.load_state_dict(checkpoint['G'])
    #     print("✅ 已加载最佳模型权重")
    #
    # # 绘制 t-SNE
    # tsne_save_path = os.path.join(save_dir, 'train_final_tsne.png')
    # plot_tsne_for_training(
    #     G=G,
    #     loader=train_loader,
    #     device=device,
    #     save_path=tsne_save_path,
    #     title=f"AMINet Features: {task_name}"
    # )
    # print("=" * 50 + "\n")


if __name__ == '__main__':
    main()