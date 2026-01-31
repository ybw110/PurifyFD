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
#               主训练逻辑
# ==========================================
def main():
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    #                                 实验配置区域
    # =========================================================================

    MODE = 'SINGLE'
    DATASET_TYPE = 'SELFA'
    SOURCE_DOMAINS = ['1200']

    MODE = 'SINGLE'
    DATASET_TYPE = 'SELFA'
    SOURCE_DOMAINS = ['1800']
    # # #
    MODE = 'SINGLE'
    DATASET_TYPE = 'SELFB'
    SOURCE_DOMAINS = ['1200']
    # #
    MODE = 'SINGLE'
    DATASET_TYPE = 'SELFB'
    SOURCE_DOMAINS = ['1800']
    # #
    MODE = 'MULTI'
    DATASET_TYPE = 'SELFA'
    SOURCE_DOMAINS = ['1200', '1800']
    # #
    MODE = 'MULTI'
    DATASET_TYPE = 'SELFB'
    SOURCE_DOMAINS = ['1200', '1800']

    # =========================================================================

    # [新增] 打印开集配置
    print("\n" + "=" * 40)
    print(" OPEN SET CONFIGURATION")
    print(f" Known Classes: {config.known_classes}")
    print(f" Unknown Classes: {config.unknown_classes}")
    print(f" Model Output Dim: {config.num_classes}")
    print("=" * 40 + "\n")



    # 1. 数据加载逻辑 (仅支持 SelfA 和 SelfB)
    if DATASET_TYPE == 'SELFA':
        # SelfA (1楼)
        if MODE == 'SINGLE':
            domain = SOURCE_DOMAINS[0]
            # 直接使用原函数名
            loaders = prepare_self_detailed()[domain]
            task_name = f"DACN_SelfA_Single_{domain}"
        else:
            # 直接使用原函数名
            loaders = prepare_mixed_self_domains(SOURCE_DOMAINS)
            task_name = f"DACN_SelfA_Multi_{'_'.join(SOURCE_DOMAINS)}"

    elif DATASET_TYPE == 'SELFB':
        # SelfB (6楼)
        if MODE == 'SINGLE':
            domain = SOURCE_DOMAINS[0]
            # 直接使用原函数名 (SelfB 的函数名本身就带 _b_)
            loaders = prepare_self_b_detailed()[domain]
            task_name = f"DACN_SelfB_Single_{domain}"
        else:
            # 直接使用原函数名
            loaders = prepare_mixed_self_b_domains(SOURCE_DOMAINS)
            task_name = f"DACN_SelfB_Multi_{'_'.join(SOURCE_DOMAINS)}"

    else:
        raise ValueError(f"❌ 不支持的数据集类型: {DATASET_TYPE}。请检查配置区域。")


    train_loader = loaders['train']
    val_loader = loaders['val']

    # ================= 2. 初始化与路径 =================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', 'dacn_OOD2', f'{task_name}_{timestamp}')
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

    # print("\n" + "=" * 50)
    # print(f"Best Val Acc: {best_val_acc:.2f}%")
    # print(">>> 训练结束，开始绘制 DACN 训练集 t-SNE...")
    #
    # # 加载最佳模型
    # best_path = os.path.join(save_dir, 'best_model.pth')
    # if os.path.exists(best_path):
    #     model.load_state_dict(torch.load(best_path))
    #     print("✅ 已加载最佳模型权重")

    # 绘制 t-SNE
    # tsne_save_path = os.path.join(save_dir, 'train_final_tsne.png')
    # plot_tsne_for_training(
    #     model=model,
    #     loader=train_loader,
    #     device=device,
    #     save_path=tsne_save_path,
    #     title=f"DACN Features: {task_name}"
    # )
    # print("=" * 50 + "\n")


if __name__ == '__main__':
    main()