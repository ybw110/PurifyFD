import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config

cfg = Config()

# 定义映射
CLASS_MAP = {'正常': 0, '轻微': 1, '轻度': 1, '中等': 2, '中度': 2, '严重': 3, '重度': 3}
SPEEDS = ['1200', '1800']


# def read_csv_self_a(file_path):
#     """
#     SelfA 读取逻辑:
#     文件无表头(或需跳过)，内容为 25行 x 64列。
#     逻辑: 读取索引 1 到 25 (共25行) -> 展平
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#
#         if len(lines) < 26:
#             return None
#
#         data_list = []
#         for line in lines[1:26]:
#             values = [float(x.strip()) for x in line.strip().split(',')]
#             if len(values) == 64:
#                 data_list.extend(values)
#
#         if len(data_list) == 0:
#             return None
#
#         return np.array(data_list, dtype=np.float32)
#
#     except Exception as e:
#         print(f"[SelfA Read Error] {file_path}: {e}")
#         return None


def read_csv_self_a(file_path):
    """
    SelfA 修正读取逻辑:
    读取 raw_data 下的时域信号 CSV。
    格式: time (s), vibration... (带表头)
    """
    try:
        # 1. 读取 CSV，第一行是表头 (header=0)
        df = pd.read_csv(file_path, header=0)

        # 2. 提取数据列
        # 根据截图，第1列是时间，第2列(索引1)是振动值
        if df.shape[1] >= 2:
            signal = df.iloc[:, 1].values
        else:
            # 防御性：如果只有1列，则取第1列
            signal = df.iloc[:, 0].values

        return signal.astype(np.float32)

    except Exception as e:
        # 仅打印严重错误，忽略空文件
        # print(f"[SelfA Read Error] {file_path}: {e}")
        return None


def read_csv_self_b(file_path):
    """
    SelfB 读取逻辑:
    标准CSV，有表头，数据在第2列(Index 1)。
    """
    try:
        df = pd.read_csv(file_path, skiprows=1, header=None, usecols=[1])
        signal = df.iloc[:, 0].values.astype(np.float32)
        return signal
    except Exception as e:
        print(f"[SelfB Read Error] {file_path}: {e}")
        return None


def process_signal_length(signal, target_len=1024, mode='tail'):
    """统一信号长度"""
    if len(signal) >= target_len:
        return signal[:target_len] if mode == 'head' else signal[-target_len:]
    else:
        return np.pad(signal, (0, target_len - len(signal)), 'constant')

#
# def process_dataset_A():
#     print("\n" + "=" * 50)
#     print("Processing SelfData A (1楼数据集)...")
#     print("=" * 50)
#
#     save_dir = os.path.join(cfg.PROCESSED_DATA_ROOT, 'SelfA')
#     os.makedirs(save_dir, exist_ok=True)
#
#     if not os.path.exists(cfg.RAW_DATA_ROOT_A):
#         print(f"❌ Error: Raw path not found: {cfg.RAW_DATA_ROOT_A}")
#         return
#
#     for speed in SPEEDS:
#         print(f"Processing Domain: {speed} RPM")
#         all_X, all_y = [], []
#
#         subfolders = [
#             f for f in os.listdir(cfg.RAW_DATA_ROOT_A)
#             if os.path.isdir(os.path.join(cfg.RAW_DATA_ROOT_A, f))
#         ]
#
#         for folder_name in tqdm(subfolders, desc=f"Scanning folders"):
#             if '-' not in folder_name:
#                 continue
#
#             try:
#                 label_str, speed_str = folder_name.split('-', 1)
#             except:
#                 continue
#
#             if speed_str != speed:
#                 continue
#             if label_str not in CLASS_MAP:
#                 continue
#
#             label_id = CLASS_MAP[label_str]
#
#             vib_path = os.path.join(cfg.RAW_DATA_ROOT_A, folder_name, 'vibration')
#             if not os.path.exists(vib_path):
#                 vib_path = os.path.join(cfg.RAW_DATA_ROOT_A, folder_name)
#
#             csv_files = glob.glob(os.path.join(vib_path, "*.csv"))
#
#             for csv_file in csv_files:
#                 raw_sig = read_csv_self_a(csv_file)
#                 if raw_sig is None:
#                     continue
#
#                 # 只做长度统一，不做标准化（标准化放到 dataset_split 阶段）
#                 sig_fixed = process_signal_length(raw_sig, cfg.SEQUENCE_LENGTH, mode='head')
#                 all_X.append(sig_fixed.astype(np.float32))
#                 all_y.append(label_id)
#
#         if len(all_X) > 0:
#             save_path = os.path.join(save_dir, f"domain_{speed}.npz")
#             np.savez(
#                 save_path,
#                 X=np.array(all_X, dtype=np.float32),
#                 y=np.array(all_y, dtype=np.int64)
#             )
#             print(f"✅ Saved {speed} RPM: {len(all_X)} samples -> {save_path}")
#         else:
#             print(f"⚠️ Warning: No data found for {speed} RPM in SelfA")


def process_dataset_A():
    print("\n" + "=" * 50)
    print("Processing SelfData A (1楼数据集) - [Corrected Mode]")
    print("=" * 50)

    save_dir = os.path.join(cfg.PROCESSED_DATA_ROOT, 'SelfA')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(cfg.RAW_DATA_ROOT_A):
        print(f"❌ Error: Raw path not found: {cfg.RAW_DATA_ROOT_A}")
        return

    for speed in SPEEDS:
        print(f"Processing Domain: {speed} RPM")
        all_X, all_y = [], []

        # 扫描主目录下的子文件夹
        subfolders = [
            f for f in os.listdir(cfg.RAW_DATA_ROOT_A)
            if os.path.isdir(os.path.join(cfg.RAW_DATA_ROOT_A, f))
        ]

        for folder_name in tqdm(subfolders, desc=f"Scanning folders"):
            # 过滤非标准命名的文件夹
            if '-' not in folder_name:
                continue

            try:
                label_str, speed_str = folder_name.split('-', 1)
            except:
                continue

            # 匹配转速和标签
            if speed_str != speed:
                continue
            if label_str not in CLASS_MAP:
                continue

            label_id = CLASS_MAP[label_str]

            # [关键修改] 指向 raw_data 文件夹，而不是 vibration
            vib_path = os.path.join(cfg.RAW_DATA_ROOT_A, folder_name, 'raw_data')

            # 容错：如果 raw_data 不存在，尝试直接读文件夹
            if not os.path.exists(vib_path):
                vib_path = os.path.join(cfg.RAW_DATA_ROOT_A, folder_name)

            csv_files = glob.glob(os.path.join(vib_path, "*.csv"))

            for csv_file in csv_files:
                raw_sig = read_csv_self_a(csv_file)

                # 过滤读取失败或长度过短的文件
                if raw_sig is None or len(raw_sig) < 100:
                    continue

                # 截取固定长度 (默认取头部 1024 点)
                sig_fixed = process_signal_length(raw_sig, cfg.SEQUENCE_LENGTH, mode='head')

                all_X.append(sig_fixed.astype(np.float32))
                all_y.append(label_id)

        # 保存为 .npz
        if len(all_X) > 0:
            save_path = os.path.join(save_dir, f"domain_{speed}.npz")
            np.savez(
                save_path,
                X=np.array(all_X, dtype=np.float32),
                y=np.array(all_y, dtype=np.int64)
            )
            print(f"✅ Saved {speed} RPM: {len(all_X)} samples -> {save_path}")
        else:
            print(f"⚠️ Warning: No data found for {speed} RPM in SelfA (Check folder structure!)")


def process_dataset_B():
    print("\n" + "=" * 50)
    print("Processing SelfData B (6楼数据集)...")
    print("=" * 50)

    save_dir = os.path.join(cfg.PROCESSED_DATA_ROOT, 'SelfB')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(cfg.RAW_DATA_ROOT_B):
        print(f"❌ Error: Raw path not found: {cfg.RAW_DATA_ROOT_B}")
        return

    SUB_DIRS = ['train', 'val', 'test']

    for speed in SPEEDS:
        print(f"Processing Domain: {speed} RPM")
        all_X, all_y = [], []

        for sub in SUB_DIRS:
            sub_root = os.path.join(cfg.RAW_DATA_ROOT_B, sub)
            if not os.path.exists(sub_root):
                continue

            subfolders = [
                f for f in os.listdir(sub_root)
                if os.path.isdir(os.path.join(sub_root, f))
            ]

            for folder_name in tqdm(subfolders, desc=f"Scanning {sub}"):
                if '-' not in folder_name:
                    continue

                try:
                    label_str, speed_str = folder_name.split('-', 1)
                except:
                    continue

                if speed_str != speed:
                    continue
                if label_str not in CLASS_MAP:
                    continue

                label_id = CLASS_MAP[label_str]

                vib_path = os.path.join(sub_root, folder_name, 'raw_data')
                if not os.path.exists(vib_path):
                    continue

                csv_files = glob.glob(os.path.join(vib_path, "*.csv"))

                for csv_file in csv_files:
                    raw_sig = read_csv_self_b(csv_file)
                    if raw_sig is None:
                        continue

                    # 只做长度统一，不做标准化（标准化放到 dataset_split 阶段）
                    sig_fixed = process_signal_length(raw_sig, cfg.SEQUENCE_LENGTH, mode='tail')
                    all_X.append(sig_fixed.astype(np.float32))
                    all_y.append(label_id)

        if len(all_X) > 0:
            save_path = os.path.join(save_dir, f"domain_{speed}.npz")
            np.savez(
                save_path,
                X=np.array(all_X, dtype=np.float32),
                y=np.array(all_y, dtype=np.int64)
            )
            print(f"✅ Saved {speed} RPM: {len(all_X)} samples -> {save_path}")
        else:
            print(f"⚠️ Warning: No data found for {speed} RPM in SelfB")


if __name__ == '__main__':
    process_dataset_A()
    process_dataset_B()
    print("\n🎉 All Preprocessing Finished!")
