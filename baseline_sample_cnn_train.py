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
from baseline_sample_cnn_model import SampleCNN
from config import Config
from dataset_spilt_CWRU import prepare_all_domains_detailed, prepare_mixed_domains, ALL_DOMAINS
from dataset_spilt_jnu import prepare_jnu_detailed, prepare_mixed_jnu_domains, ALL_SPEEDS

# === 新增 t-SNE 库 ===
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


config = Config()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# def print_row(values, colwidth=15):
#     print("".join([str(x).ljust(colwidth) for x in values]))


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


# ==========================================
#           新增：训练集 t-SNE 绘图函数
# ==========================================
def extract_features(model, loader, device, max_samples=2000):
    """提取训练集特征 (带采样，防止过大)"""
    model.eval()
    features = []
    labels = []

    total_count = 0

    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device).float()
            targets = data[1].to(device).long()

            # SampleCNN forward 返回 (logits, feat)
            _, feat = model(inputs)

            features.append(feat.cpu().numpy())
            labels.append(targets.cpu().numpy())

            total_count += len(targets)
            if total_count >= max_samples:
                break

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 如果超出稍微截断一下，确保精确匹配 max_samples (可选)
    if len(features) > max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]

    return features, labels


def plot_tsne_for_training(model, loader, device, save_path, title="Training Features"):
    print(f"\n🎨 正在绘制 Baseline 训练集 t-SNE: {title} ...")

    # 1. 提取数据
    feats, labels = extract_features(model, loader, device, max_samples=300)

    # 2. PCA 预处理
    if feats.shape[1] > 50:
        pca = PCA(n_components=50)
        feats = pca.fit_transform(feats)

    # 3. t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(feats)

    # 4. 绘图配置 (保持统一风格)
    plt.figure(figsize=(5,4))  # 稍微宽一点适配图例
    plt.rcParams['font.family'] = 'Arial'

    # 自定义配色 (与您的 Diversify 保持一致)
    # 对应: Normal, Inner, Ball, Outer
    colors = ['#6D95C3', '#EBB577', '#CDD797', '#E09D94']
    class_names = ['Normal', 'Inner', 'Ball', 'Outer']
    markers = ['o', '^', 's', 'D']

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
                linewidth=0.6
            )

    # plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc='best', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE 图已保存至: {save_path}")


def main():
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    jnu_root = '/data2/ybw25/轴承DG/JNU/JNU-Bearing-Dataset-main'

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
    # MODE = 'MULTI'
    # DATASET_TYPE = 'JNU'
    # SOURCE_DOMAINS = ['600', '1000']
    # # ==========================================================

    # --- 模式 C: 完整数据设置 (Full Dataset) ---
    # 只要在 SOURCE_DOMAINS 里填入所有工况，即为 Full Dataset 训练
    # MODE = 'MULTI'
    # DATASET_TYPE = 'CWRU'
    # SOURCE_DOMAINS = ALL_DOMAINS        # 或者: SOURCE_DOMAINS = ['0HP', '1HP', '2HP', '3HP']

    MODE = 'MULTI'
    DATASET_TYPE = 'JNU'
    SOURCE_DOMAINS = ALL_SPEEDS  # ['600', '800', '1000']

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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', 'sample_cnn', f'{task_name}_{timestamp}')
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 3. 初始化模型 (标准 CNN)
    model = SampleCNN(num_classes=config.num_classes).to(device)

    # 4. 优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0

    # # 5. 训练循环
    # print_row(['Epoch', 'Loss', 'Train Acc', 'Val Acc'], colwidth=15)

    # 用一个总的 epoch 进度条
    epoch_bar = tqdm(range(config.epochs),
                     desc="Training Progress",
                     colour='red',
                     ncols=100)

    for epoch in epoch_bar:
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # --- Training ---
        for data in train_loader:
            # 这里的 dataset 返回 6 元组，我们只需要 input(0) and label(1)
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # --- Validation ---
        model.eval()
        v_total_loss, v_correct, v_total = 0, 0, 0

        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                labels = data[1].to(device)
                logits, _ = model(inputs)

                v_loss = criterion(logits, labels)
                v_total_loss += v_loss.item()

                _, predicted = torch.max(logits, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()

            val_loss = v_total_loss / len(val_loader)
            val_acc = 100 * v_correct / v_total

        # 更新进度条显示当前指标（可选）
        epoch_bar.set_postfix({
            'Loss': f'{train_loss:.4f}',
            'Train': f'{train_acc:.1f}%',
            'Val': f'{val_acc:.1f}%'
        })

        # 记录与打印
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)  # 记录 val_loss
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)

        # print_row([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}%", f"{val_acc:.2f}%"], colwidth=15)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    # 保存 CSV
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
    plot_training_history(metrics, os.path.join(save_dir, 'training_curves.png'))

    # ==========================================
    #      新增：训练结束后绘制训练集 t-SNE
    # ==========================================
    print("\n" + "=" * 50)
    print(">>> 训练结束，开始绘制训练集 t-SNE...")

    # 1. 重新加载最佳模型权重
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("✅ 已加载最佳模型权重")
    else:
        print("⚠️ 未找到最佳模型权重，使用当前最后轮次权重")

    # 2. 绘制 t-SNE
    tsne_save_path = os.path.join(save_dir, 'train_final_tsne.png')
    plot_tsne_for_training(
        model=model,
        loader=train_loader,  # 使用训练集 DataLoader
        device=device,
        save_path=tsne_save_path,
        title="Baseline CNN: Final Training Features"
    )
    print("=" * 50 + "\n")

if __name__ == '__main__':
    main()