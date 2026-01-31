import time
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 导入 ===
from baseline_mixup_model import MixupCNN, mixup_data, mixup_criterion # 导入 Mixup 模型和工具
from config import Config
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains,ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains,ALL_SPEEDS


config = Config()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_training_history(metrics, save_path):
    """绘制训练损失和准确率曲线"""
    # 设置字体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12  # 默认字体大小，你可以调整

    fig, ax1 = plt.subplots(figsize=(6, 5))

    # 绘制 Loss
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', color='tab:red', fontsize=14)
    line1 = ax1.plot(metrics['epoch'], metrics['train_loss'], color='tab:red', linewidth=2, label='Train Loss')
    line2 = ax1.plot(metrics['epoch'], metrics['val_loss'], '--', color='tab:red', linewidth=2, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=12)
    ax1.grid(True, alpha=0.3)

    # 绘制 Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue', fontsize=14)
    line3 = ax2.plot(metrics['epoch'], metrics['train_acc'], '--', color='tab:cyan', linewidth=2, label='Train Acc')
    line4 = ax2.plot(metrics['epoch'], metrics['val_acc'], color='tab:blue', linewidth=2, label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)

    # 合并图例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=11, framealpha=0.9)

    plt.title('Training History (CNN Baseline)', fontsize=16, pad=15)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

    # [新增] 打印开集配置
    print("\n" + "=" * 40)
    print(" OPEN SET CONFIGURATION")
    print(f" Known Classes: {config.known_classes}")
    print(f" Unknown Classes: {config.unknown_classes}")
    print(f" Model Output Dim: {config.num_classes}")
    print("=" * 40 + "\n")


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
    # SOURCE_DOMAINS = ['0HP', '1HP', '2HP']  # 3HP作为测试

    # === 选项 B: JNU (若想切换，取消注释下方) ===
    # MODE = 'MULTI'
    # DATASET_TYPE = 'JNU'
    # SOURCE_DOMAINS = ['600', '800']
    # ==========================================================

    # --- 模式 C: 完整数据设置 (Full Dataset) ---
    # 只要在 SOURCE_DOMAINS 里填入所有工况，即为 Full Dataset 训练
    # MODE = 'MULTI'
    # DATASET_TYPE = 'CWRU'
    # SOURCE_DOMAINS = ALL_DOMAINS        # 或者: SOURCE_DOMAINS = ['0HP', '1HP', '2HP', '3HP']

    MODE = 'MULTI'
    DATASET_TYPE = 'JNU'
    SOURCE_DOMAINS = ALL_SPEEDS  # ['600', '800', '1000']

    # Mixup 超参数 (建议 alpha 取值 0.2 ~ 1.0)
    MIXUP_ALPHA = 0.3

    # 1. 数据加载逻辑
    print(f"Current Mode: {MODE} | Dataset: {DATASET_TYPE} | Mixup Alpha: {MIXUP_ALPHA}")

    if MODE == 'SINGLE':
        task_name = f"Mixup_SingleSource_{SOURCE_DOMAINS}"
        if DATASET_TYPE == 'CWRU':
            loaders = prepare_all_domains_detailed()[SOURCE_DOMAINS]
        else:
            loaders = prepare_jnu_detailed(jnu_root)[SOURCE_DOMAINS]
    elif MODE == 'MULTI':
        task_name = f"Mixup_MultiSource_{'_'.join(SOURCE_DOMAINS)}"
        if DATASET_TYPE == 'CWRU':
            loaders = prepare_mixed_domains(SOURCE_DOMAINS)
        else:
            loaders = prepare_mixed_jnu_domains(SOURCE_DOMAINS, jnu_root)
    else:
        raise ValueError("Invalid Mode")

    train_loader = loaders['train']
    val_loader = loaders['val']

    # 3. 初始化记录器与保存路径

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', 'mixup_cnn_OOD3', f'{task_name}_{timestamp}')
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 3. 初始化模型 (标准 CNN)
    model = MixupCNN(num_classes=config.num_classes).to(device)

    # 4. 优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0


    # # 5. 训练循环

    # 用一个总的 epoch 进度条
    epoch_bar = tqdm(range(config.epochs), desc="Mixup Training", colour='green',ncols=100)

    for epoch in epoch_bar:
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # --- Training ---
        for data in train_loader:
            # 这里的 dataset 返回 6 元组，我们只需要 input(0) and label(1)
            inputs = data[0].to(device).float()
            labels = data[1].to(device).long()  # 原始标签

            optimizer.zero_grad()

            # >>> Mixup 核心步骤 Start <<<
            # 1. 生成混合数据和混合标签
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA, use_cuda=True)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

            # 2. 前向传播
            logits, _ = model(inputs)

            # 3. 计算 Mixup 损失
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            # >>> Mixup 核心步骤 End <<<

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)

            total += labels.size(0)
            if lam > 0.5:
                correct += (predicted == targets_a).sum().item()
            else:
                correct += (predicted == targets_b).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 训练准确率计算 (稍微复杂一点，通常比较 argmax 和原始 label，或者加权)
        # 这里为了简单展示，我们仍然统计是否预测对了“主要成分”的那个类 (lam > 0.5)
        # 但更标准的 Mixup 训练通常不看 Training Acc，主要看 Validation Acc
        # 统计准确率逻辑：如果 lam > 0.5，应该接近 target_a；否则接近 target_b
        # 这里简单起见，仅计算 loss，Acc 仅供参考

        # === Validation Phase (不使用 Mixup，正常测试) ===
        model.eval()
        v_total_loss, v_correct, v_total = 0, 0, 0

        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device).float()
                labels = data[1].to(device).long()

                logits, _ = model(inputs)
                v_loss = criterion(logits, labels)

                v_total_loss += v_loss.item()
                _, predicted = torch.max(logits, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()

            val_loss = v_total_loss / len(val_loader)
            val_acc = 100 * v_correct / v_total

        # 更新进度条显示当前指标（可选）
        epoch_bar.set_postfix({'Loss': f'{train_loss:.4f}', 'Val Acc': f'{val_acc:.2f}%'})

        # 记录与打印
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)  # 记录 val_loss
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        # 保存 CSV
        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
        plot_training_history(metrics, os.path.join(save_dir, 'training_curves.png'))
        print(f"Training finished. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()