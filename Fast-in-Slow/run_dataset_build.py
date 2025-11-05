import tensorflow_datasets as tfds

# 这一行会导入你的 builder 脚本 (uav_dataset_builder.py)
# 并使其在 TFDS 中 "注册" UavDataset 类
import vla.datasets.uav_dataset_builder 

# 你希望构建好的 TFDS 数据集存放的 *父* 目录
DATA_DIR = "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/rlds_data"

# 你要构建的数据集名字
DATASET_NAME = "uav_dataset"

print(f"--- 正在构建 {DATASET_NAME} ---")
print(f"--- 输出到: {DATA_DIR} ---")

# --- 关键改动在这里 ---
# 我们在调用 tfds.builder() 时，
# *直接* 把 data_dir 作为参数传进去
builder = tfds.builder(DATASET_NAME, data_dir=DATA_DIR)

# (之前那行 builder.data_dir = DATA_DIR 已经被删除了)

# 3. 运行完整的下载和构建流程
builder.download_and_prepare()

print("--- 构建完成！ ---")