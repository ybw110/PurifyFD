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
# 确保你已经保存了我之前提供的 baseline_dacn_model.py
from baseline_dacn_model import DACN, SupConLoss
from config import Config
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS

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
    """绘制训练损失和准确率曲线"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制 Loss
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss (Total)', color='tab:red', fontsize=14)
    # 这里的 train_loss 我们记录的是 FGC 步骤的总 Loss
    line1 = ax1.plot(metrics['epoch'], metrics['train_loss'], color='tab:red', linewidth=2, label='Train Loss')
    # DACN 的 Validation 只有 Accuracy，Loss 参考意义不大（因为只有分类Loss），这里仅画 Train Loss
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=12)
    ax1.grid(True, alpha=0.3)

    # 绘制 Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue', fontsize=14)
    line3 = ax2.plot(metrics['epoch'], metrics['train_acc'], '--', color='tab:cyan', linewidth=2, label='Train Acc')
    line4 = ax2.plot(metrics['epoch'], metrics['val_acc'], color='tab:blue', linewidth=2, label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)

    lines = line1 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=11, framealpha=0.9)

    plt.title('Training History (DACN)', fontsize=16, pad=15)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
#           适配 DACN 的 t-SNE 提取函数
# ==========================================
def extract_features_dacn(model, loader, device, max_samples=2000):
    """提取 DACN 的域不变特征 (Feature G)"""
    model.eval()
    features = []
    labels = []
    total_count = 0

    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device).float()
            targets = data[1].to(device).long()

            # DACN 推理时使用 forward_pretrain 路径 (F -> G -> C)
            # forward_pretrain 返回 (logits, g)
            _, feat = model.forward_pretrain(inputs)

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


def plot_tsne_for_training(model, loader, device, save_path, title="DACN Features"):
    print(f"\n🎨 正在绘制 DACN 训练集 t-SNE: {title} ...")

    # 使用适配 DACN 的提取函数
    feats, labels = extract_features_dacn(model, loader, device, max_samples=300)

    if feats.shape[1] > 50:
        pca = PCA(n_components=50)
        feats = pca.fit_transform(feats)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(feats)

    plt.figure(figsize=(5, 4))
    plt.rcParams['font.family'] = 'Arial'
    colors = ['#6D95C3', '#EBB577', '#CDD797', '#E09D94']  # 你的配色
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

    plt.legend(loc='best', fontsize=12)
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

    # ================= 1. 数据配置 (保持和你的一致) =================

    # --- 模式 A: 单源域设置 (Single Source) ---
    # MODE = 'SINGLE'
    # DATASET_TYPE = 'CWRU'       # 'CWRU' or 'JNU'
    # SOURCE_DOMAINS = '0HP'         # '0HP', '1HP'... or '600', '800'...

    # 选项 B: 使用 JNU (若想切换，取消下面两行注释并注释掉上面两行)
    # MODE = 'SINGLE'
    # DATASET_TYPE = 'JNU'
    # SOURCE_DOMAINS = '1000'  # 可选: '600', '800', '1000'

    # ================= 配置数据源 (多源域设置) =================

    # === 选项 A: CWRU ===
    # MODE = 'MULTI'
    # DATASET_TYPE = 'CWRU'
    # SOURCE_DOMAINS = ['1HP', '2HP', '3HP']  # 3HP作为测试

    # === 选项 B: JNU (若想切换，取消注释下方) ===
    MODE = 'MULTI'
    DATASET_TYPE = 'JNU'
    SOURCE_DOMAINS = ['600', '800']
    # # ==========================================================

    # --- 模式 C: 完整数据设置 (Full Dataset) ---
    # 只要在 SOURCE_DOMAINS 里填入所有工况，即为 Full Dataset 训练
    # MODE = 'MULTI'
    # DATASET_TYPE = 'CWRU'
    # SOURCE_DOMAINS = ALL_DOMAINS        # 或者: SOURCE_DOMAINS = ['0HP', '1HP', '2HP', '3HP']

    # MODE = 'MULTI'
    # DATASET_TYPE = 'JNU'
    # SOURCE_DOMAINS = ALL_SPEEDS             # ['600', '800', '1000']


    print(f"Current Mode: {MODE} | Dataset: {DATASET_TYPE} | Source: {SOURCE_DOMAINS}")

    # 数据加载
    if MODE == 'SINGLE':
        task_name = f"DACN_Single_{SOURCE_DOMAINS}"
        if DATASET_TYPE == 'CWRU':
            all_data = prepare_all_domains_detailed()
            loaders = all_data[SOURCE_DOMAINS]
        else:
            all_data = prepare_jnu_detailed(jnu_root)
            loaders = all_data[SOURCE_DOMAINS]
    elif MODE == 'MULTI':
        task_name = f"DACN_Multi_{'_'.join(SOURCE_DOMAINS)}"
        if DATASET_TYPE == 'CWRU':
            loaders = prepare_mixed_domains(SOURCE_DOMAINS)
        else:
            loaders = prepare_mixed_jnu_domains(SOURCE_DOMAINS, jnu_root)
    else:
        raise ValueError("Invalid Mode")

    train_loader = loaders['train']
    val_loader = loaders['val']

    # ================= 2. 初始化与路径 =================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', 'dacn', f'{task_name}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)

    # 模型
    model = DACN(num_classes=config.num_classes).to(device)

    # Loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=0.07)
    criterion_bce = nn.BCELoss()

    # 优化器 (DACN 需要分开优化)
    # F, G, C 是一组
    optimizer_F_G_C = optim.Adam(
        list(model.F.parameters()) + list(model.G.parameters()) + list(model.C.parameters()),
        lr=config.lr, weight_decay=config.weight_decay
    )
    # H (Generator) 是一组
    optimizer_H = optim.Adam(model.H.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # D (Discriminator) 是一组
    optimizer_D = optim.Adam(model.D.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 记录
    metrics = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0

    # ================= 3. 预训练阶段 (Pre-training) =================
    PRETRAIN_EPOCHS = 5  # 可以根据需要调整，通常几轮就够
    print(f"\n>>> [Phase 1] Pre-training for {PRETRAIN_EPOCHS} epochs...")

    epoch_bar1 = tqdm(range(PRETRAIN_EPOCHS), desc="Pre-training", colour='green', ncols=100)
    for epoch in epoch_bar1:
    # for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        for data in train_loader:
            inputs, labels = data[0].to(device).float(), data[1].to(device).long()

            optimizer_F_G_C.zero_grad()
            logits, _ = model.forward_pretrain(inputs)
            loss = criterion_cls(logits, labels)
            loss.backward()
            optimizer_F_G_C.step()
    print(">>> Pre-training Finished.\n")

    # ================= 4. 正式对抗训练阶段 (Adversarial Training) =================
    print(f">>> [Phase 2] Adversarial Training for {config.epochs} epochs...")

    epoch_bar = tqdm(range(config.epochs), desc="DACN Training", colour='green', ncols=100)

    for epoch in epoch_bar:
        model.train()

        # 统计变量
        total_loss_fgc = 0  # 主任务 Loss
        correct = 0
        total = 0

        for data in train_loader:
            inputs, labels = data[0].to(device).float(), data[1].to(device).long()
            batch_size = inputs.size(0)

            # === Step A: 前向传播 (获取真实特征与伪造特征) ===
            g_concat, logits_concat = model.forward_train(inputs)

            # 拆分 Real / Fake
            g_real, g_fake = g_concat.chunk(2)
            c_real, c_fake = logits_concat.chunk(2)

            # Softmax 用于 CDAN
            softmax_real = torch.softmax(c_real, dim=1).detach()
            softmax_fake = torch.softmax(c_fake, dim=1).detach()

            # === Step B: 训练判别器 D (Discriminator) ===
            optimizer_D.zero_grad()

            # 构造 CDAN 输入 (Feature x Prob)
            op_real = torch.bmm(softmax_real.unsqueeze(2), g_real.detach().unsqueeze(1)).view(batch_size, -1)
            op_fake = torch.bmm(softmax_fake.unsqueeze(2), g_fake.detach().unsqueeze(1)).view(batch_size, -1)

            d_out_real = model.D(op_real)
            d_out_fake = model.D(op_fake)

            # D 的目标：Real->1, Fake->0
            loss_d = criterion_bce(d_out_real, torch.ones_like(d_out_real)) + \
                     criterion_bce(d_out_fake, torch.zeros_like(d_out_fake))

            loss_d.backward()
            optimizer_D.step()

            # === Step C: 训练生成器 H (Feature Transformer) ===
            optimizer_H.zero_grad()

            # 为了计算图完整，这里通常需要重新 forward 或利用 retain_graph (这里简化，重新 forward 计算 H 的梯度流)
            g_concat_2, logits_concat_2 = model.forward_train(inputs)
            _, g_fake_2 = g_concat_2.chunk(2)
            _, c_fake_2 = logits_concat_2.chunk(2)
            softmax_fake_2 = torch.softmax(c_fake_2, dim=1)

            op_fake_2 = torch.bmm(softmax_fake_2.unsqueeze(2), g_fake_2.unsqueeze(1)).view(batch_size, -1)
            d_out_fake_2 = model.D(op_fake_2)

            # H 的对抗目标：让 D 能够轻易区分 (Minimize Discriminative Loss, 即使得 fake->0)
            # 注意：论文思路独特，H 增加多样性，让 D 容易区分；而 G 才是负责欺骗 D。
            loss_h_adv = criterion_bce(d_out_fake_2, torch.zeros_like(d_out_fake_2))

            # 语义一致性 (L_c2) 和 对比损失 (SupCon on Fake)
            loss_h_cls = criterion_cls(c_fake_2, labels)
            loss_h_sup = criterion_supcon(g_fake_2, labels)

            loss_h_total = loss_h_adv + loss_h_cls + loss_h_sup

            # 需要保留图给 F_G_C 用
            loss_h_total.backward(retain_graph=True)
            optimizer_H.step()

            # === Step D: 训练特征提取器 (F, G, C) ===
            optimizer_F_G_C.zero_grad()

            # 源域分类 Loss (L_c1)
            # 注意：c_real 来自第一次 forward，但我们要反传梯度到 F/G，最好复用第二次的 g_concat_2 拆分结果
            g_real_2, _ = g_concat_2.chunk(2)
            c_real_2, _ = logits_concat_2.chunk(2)

            loss_c1 = criterion_cls(c_real_2, labels)

            # 对比学习 Loss (Real + Fake)
            labels_concat = torch.cat([labels, labels], dim=0)
            loss_sup_fgc = criterion_supcon(g_concat_2, labels_concat)

            # G 的对抗目标：欺骗 D (让 Fake 看起来像 Real -> 1)
            # 重新计算 D 对 fake 的判断 (因为 H 变了，理论上需要再算，但为效率复用 op_fake_2 即可)
            # 或者为了严谨，针对 G 的更新，应该让 D 觉得 fake 是 1
            d_out_fake_3 = model.D(op_fake_2)
            loss_g_adv = criterion_bce(d_out_fake_3, torch.ones_like(d_out_fake_3))

            loss_fgc_total = loss_c1 + loss_sup_fgc + loss_g_adv
            loss_fgc_total.backward()
            optimizer_F_G_C.step()

            # === 记录数据 ===
            total_loss_fgc += loss_fgc_total.item()
            _, predicted = torch.max(c_real_2, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算 Epoch 指标
        train_loss_epoch = total_loss_fgc / len(train_loader)
        train_acc_epoch = 100 * correct / total

        # === Validation ===
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device).float(), data[1].to(device).long()
                # 验证时只用前向推理路径
                logits, _ = model.forward_pretrain(inputs)
                _, predicted = torch.max(logits, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()

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
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    # ================= 5. 结束工作 (CSV & t-SNE) =================

    # 保存 CSV
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    # 绘制训练曲线
    plot_training_history(metrics, os.path.join(save_dir, 'training_curves.png'))

    print("\n" + "=" * 50)
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(">>> 训练结束，开始绘制 DACN 训练集 t-SNE...")

    # 加载最佳模型
    best_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print("✅ 已加载最佳模型权重")

    # 绘制 t-SNE
    tsne_save_path = os.path.join(save_dir, 'train_final_tsne.png')
    plot_tsne_for_training(
        model=model,
        loader=train_loader,
        device=device,
        save_path=tsne_save_path,
        title=f"DACN Features: {task_name}"
    )
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()