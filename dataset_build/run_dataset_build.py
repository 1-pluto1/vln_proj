import tensorflow_datasets as tfds

# 这一行会导入你的 builder 脚本 (uav_dataset_builder.py)
# 并使其在 TFDS 中 "注册" UavDataset 类
import uav_dataset_builder 
import os
import argparse

# 你希望构建好的 TFDS 数据集存放的 *父* 目录
DATA_DIR = "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/test_data"

# 你要构建的数据集名字
DATASET_NAME = "uav_dataset"

print(f"--- 正在构建 {DATASET_NAME} ---")
print(f"--- 输出到: {DATA_DIR} ---")

# --- 关键改动在这里 ---
# 我们在调用 tfds.builder() 时，
parser = argparse.ArgumentParser("Build TFDS dataset for UAV-Flow RLDS")
parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="TFDS 数据集的父目录（不要包含数据集名）")
parser.add_argument("--dataset_name", type=str, default=DATASET_NAME, help="数据集名称（与 Builder 类名 snake_case 一致）")
args = parser.parse_args()

# 防止把 data_dir 误设为包含子目录 'uav_dataset'
basename = os.path.basename(args.data_dir.rstrip("/"))
if basename == args.dataset_name:
    corrected = os.path.dirname(args.data_dir.rstrip("/"))
    print(f"[警告] data_dir 不应包含数据集子目录。改用父目录: {corrected}")
    args.data_dir = corrected

# 让 Builder 的源数据路径和 TFDS 的父目录一致（Builder 内部读取 RLDS_SOURCE_DIR）
os.environ.setdefault("RLDS_SOURCE_DIR", args.data_dir)
# 让后续训练用 tfds.load 时无需传 data_dir
os.environ.setdefault("TFDS_DATA_DIR", args.data_dir)

# *直接* 把 data_dir 作为参数传进去
builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)

# 运行完整的下载和构建流程
builder.download_and_prepare()

built_dir = os.path.join(args.data_dir, args.dataset_name, "1.0.0")
print(f"--- 构建完成！路径: {built_dir} ---")