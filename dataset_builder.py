import tensorflow_datasets as tfds
import tensorflow as tf
import glob
import json
import numpy as np
# 假设你还有处理图像或点云的库
# import cv2 
# import open3d as o3d

# 1. 定义你的数据集类
class UavDataset(tfds.core.GeneratorBasedBuilder):
    """你的无人机数据集的构建器。"""

    # 2. 定义版本号
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': '初始版本。',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """在这里定义数据集的元数据和“特征”。"""
        
        # 3. 这是最关键的一步：定义你的数据结构
        features = tfds.features.FeaturesDict({
            'episode_id': tfds.features.Text(),
            'instruction': tfds.features.Text(),
            'steps': tfds.features.Sequence({
                'image': tfds.features.Image(shape=(224, 224, 3), encoding_format='jpeg'),
                'point_cloud': tfds.features.Tensor(shape=(1024, 3), dtype=tf.float32), # 假设点云是 1024 个点
                'action': tfds.features.Tensor(shape=(4,), dtype=tf.float32), # 假设动作是一个4维向量
                # ... 你需要的任何其他数据 ...
            })
        })

        return tfds.core.DatasetInfo(
            builder=self,
            description="用于 UAV-VLN 任务的数据集。",
            features=features,
            homepage="https://your-dataset-homepage.com", # 选填
            citation=r"""@article{...}""", # 选填
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """
        指定 "train", "validation", "test" 分割。
        它应该返回一个 tfds.core.SplitGenerator 列表。
        
        你需要把 /path/to/your/RAW_DATA 替换成你存放
        原始数据（比如 .json, .mp4 文件）的真实路径。
        """
        # 假设你所有原始数据都在这个路径
        raw_data_path = "/path/to/your/RAW_DATA" 

        # 告诉 TFDS 你的训练集和验证集在哪里
        return {
            'train': self._generate_examples(path=f"{raw_data_path}/train"),
            'val': self._generate_examples(path=f"{raw_data_path}/val"),
        }

    def _generate_examples(self, path: str):
        """
        这是“配方”的核心。
        你需要在这里编写代码，读取你的原始文件 (JSON, 图像, ...)，
        并 'yield' (产出) 符合 _info() 中定义的数据。
        """
        
        # 示例：假设你的原始数据是每个 episode 一个 json 文件
        # e.g., /path/to/your/RAW_DATA/train/episode_001.json
        
        episode_files = glob.glob(f"{path}/*.json")
        
        for filepath in episode_files:
            # 1. 读取你的原始数据
            with tf.io.gfile.GFile(filepath, "r") as f:
                data = json.load(f)

            episode_id = data['episode_id']
            instruction = data['instruction']
            
            steps_data = []
            for step in data['steps']:
                # 2. 加载图像和点云
                # (你必须自己实现这部分逻辑)
                image = self._load_image(step['image_path'])
                point_cloud = self._load_point_cloud(step['pcd_path'])
                action = np.array(step['action'], dtype=np.float32)

                steps_data.append({
                    'image': image,
                    'point_cloud': point_cloud,
                    'action': action,
                })

            # 3. 'yield' 一个唯一的 ID 和一个数据字典
            # 这个字典的结构必须和 _info() 中定义的 features 完全一致
            yield episode_id, {
                'episode_id': episode_id,
                'instruction': instruction,
                'steps': steps_data
            }

    # 你需要自己实现这些辅助函数
    def _load_image(self, path):
        # TODO: 编写加载和预处理图像的代码
        # 示例：
        # img = cv2.imread(path)
        # img = cv2.resize(img, (224, 224))
        # return img
        pass

    def _load_point_cloud(self, path):
        # TODO: 编写加载和处理点云的代码
        # 示例：
        # pcd = o3d.io.read_point_cloud(path)
        # points = np.asarray(pcd.points, dtype=np.float32)
        # ... (采样或补齐到 1024 个点) ...
        # return points
        pass