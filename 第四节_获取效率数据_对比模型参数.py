import torch
import torch.nn as nn
import time
import numpy as np
from thop import profile  # 关键库：用于计算 FLOPs

# === 1. 导入您的模型定义 ===
# 确保所有 .py 文件都在同一目录下
from baseline_sample_cnn_model import SampleCNN
from baseline_mixstyle_model import MixStyleCNN
from baseline_AMINet_model import FeatureExtractor as AMINet_G, Classifier as AMINet_C
from baseline_dacn_model import DACN
from model_diversify_network import CWRUNetwork  # PurifyFD Backbone
from model_common_network import feat_bottleneck, feat_classifier
import platform
import subprocess
# === 2. 封装模型以模拟推理过程 ===

class DACN_Wrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = DACN(num_classes=num_classes)

    def forward(self, x):
        # 手动构建推理路径: F (特征) -> G (域不变) -> C (分类)
        # 避开 H (AdaIN) 和 D (判别器)，因为推理不需要它们
        f = self.model.F(x)
        g = self.model.G(f)
        logits = self.model.C(g)
        return logits


class PurifyFD_Inference(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.featurizer = CWRUNetwork()
        # 模拟 diversify_algorithm.py 中的推理结构
        # 丢弃 Discriminator，只保留 Backbone + Bottleneck + Classifier
        self.bottleneck = feat_bottleneck(self.featurizer.in_features, config.bottleneck, config.layer)
        self.classifier = feat_classifier(config.num_classes, config.bottleneck, config.classifier)

    def forward(self, x):
        feat = self.featurizer(x)
        feat = self.bottleneck(feat)
        logits = self.classifier(feat)
        return logits


class AMINet_Inference(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.G = AMINet_G()
        self.C = AMINet_C(input_dim=256, num_classes=num_classes)

    def forward(self, x):
        return self.C(self.G(x))


# === 3. 配置与辅助函数 ===
class Config:
    in_features = 256
    bottleneck = 256
    layer = 'bn'
    classifier = 'linear'
    num_classes = 3  # Known classes (0,1,3)


def count_parameters(model):
    """计算模型参数量 (Million)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_latency(model, input_tensor, repeats=200):
    """测量推理延迟 (ms)"""
    model.eval()
    device = input_tensor.device

    # 确保模型在正确的设备上
    model.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(50):
            _ = model(input_tensor)

    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end = time.time()

    return (end - start) / repeats * 1000  # ms


def measure_flops(model, input_tensor):
    """计算 FLOPs (Million)"""
    model.eval()
    # thop.profile 返回 (MACs, Params)
    # 这里的 input_tensor 只需要形状正确即可，不需要真实数据
    macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    return macs / 1e6  # 转换为 M (Million)

def get_cpu_name():
    """获取 CPU 具体型号"""
    try:
        if platform.system() == "Windows":
            return subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split('\n')[1]
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo | grep 'model name' | uniq"
            return subprocess.check_output(command, shell=True).decode().strip().split(':')[1].strip()
        else:
            return platform.processor()
    except Exception:
        return platform.processor()


# === 4. 主程序 ===
if __name__ == '__main__':
    # 获取并打印 CPU 型号
    cpu_model = get_cpu_name()
    print("=" * 80)
    print(f"Hardware Environment: {cpu_model}")
    # 优先使用 CPU 进行测试，更能模拟边缘设备环境
    # 如果您想看 GPU 速度，改为 'cuda'
    device_name = 'cpu'
    device = torch.device(device_name)
    print(f"Testing on device: {device} (Simulating Edge Node)")
    print("=" * 80)

    config = Config()
    # 输入维度: [Batch=1, Channel=1, Length=1024]
    dummy_input = torch.randn(1, 1, 1024).to(device)

    # 初始化模型字典
    # Mixup 和 CNN 推理结构完全一致
    models = {
        "CNN": SampleCNN(num_classes=config.num_classes),
        "Mixup": SampleCNN(num_classes=config.num_classes),
        "MixStyle": MixStyleCNN(num_classes=config.num_classes),
        "AMINet": AMINet_Inference(num_classes=config.num_classes),
        "DACN": DACN_Wrapper(num_classes=config.num_classes),
        "PurifyFD": PurifyFD_Inference(config)
    }

    # 打印表头
    header = f"{'Method':<15} | {'Params (M)':<15} | {'FLOPs (M)':<15} | {'Latency (ms)':<15}"
    print(header)
    print("-" * len(header))

    # 循环测试
    for name, model in models.items():
        try:
            # 移动模型到指定设备
            model.to(device)

            # 1. 计算参数量
            params = count_parameters(model)

            # 2. 计算 FLOPs (注意：thop 需要模型在 CPU 或 GPU 均可，这里复用 device)
            flops = measure_flops(model, dummy_input)

            # 3. 计算延迟
            latency = measure_latency(model, dummy_input)

            print(f"{name:<15} | {params:<15.4f} | {flops:<15.4f} | {latency:<15.4f}")

        except Exception as e:
            print(f"{name:<15} | Error: {e}")
            # print(e) # 打开此行可查看详细错误堆栈