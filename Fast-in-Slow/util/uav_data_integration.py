"""
TravelUAV数据集与FiS模型集成的数据处理模块

处理TravelUAV数据集格式转换，支持：
1. 静态12k数据集加载和预处理
2. DAgger收集数据的格式化
3. 异步采样所需的多模态数据对齐
4. 6-DoF动作空间的标准化
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)


@dataclass
class UAVTrajectoryPoint:
    """UAV轨迹点数据结构"""
    timestamp: float
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [roll, pitch, yaw]
    velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    
    # 传感器数据
    images: Dict[str, np.ndarray]  # {'head': img, 'left': img, 'right': img}
    point_cloud: Optional[np.ndarray]  # [N, 3]
    
    # 语义信息
    instruction: str
    waypoint_progress: float  # 0.0 to 1.0
    
    def to_6dof_action(self) -> np.ndarray:
        """转换为6-DoF动作"""
        return np.concatenate([self.velocity, self.angular_velocity])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp,
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(), 
            'velocity': self.velocity.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'instruction': self.instruction,
            'waypoint_progress': self.waypoint_progress
        }


class TravelUAVDataProcessor:
    """TravelUAV数据处理器"""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 normalize_actions: bool = True,
                 action_bounds: Dict[str, List[float]] = None):
        self.image_size = image_size
        self.normalize_actions = normalize_actions
        
        # 默认动作边界
        self.action_bounds = action_bounds or {
            'velocity': [-5.0, 5.0],
            'angular_velocity': [-2.0, 2.0]
        }
        
        # 动作标准化参数
        if normalize_actions:
            self._setup_action_normalization()
    
    def _setup_action_normalization(self):
        """设置动作标准化参数"""
        vel_min, vel_max = self.action_bounds['velocity']
        ang_vel_min, ang_vel_max = self.action_bounds['angular_velocity']
        
        self.action_min = np.array([vel_min] * 3 + [ang_vel_min] * 3)
        self.action_max = np.array([vel_max] * 3 + [ang_vel_max] * 3)
        self.action_scale = self.action_max - self.action_min
        
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """标准化动作到[-1, 1]"""
        if not self.normalize_actions:
            return action
        return 2.0 * (action - self.action_min) / self.action_scale - 1.0
    
    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """反标准化动作"""
        if not self.normalize_actions:
            return normalized_action
        return (normalized_action + 1.0) * self.action_scale / 2.0 + self.action_min
    
    def process_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """处理图像数据"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 调整大小
        image = image.resize(self.image_size, Image.LANCZOS)
        
        # 转换为tensor并标准化
        image_tensor = torch.from_numpy(np.array(image)).float()
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # 标准化到[0, 1]
        image_tensor = image_tensor / 255.0
        
        return image_tensor
    
    def process_point_cloud(self, point_cloud: np.ndarray, max_points: int = 1024) -> torch.Tensor:
        """处理点云数据"""
        if point_cloud is None:
            return torch.zeros(max_points, 3)
        
        # 随机采样或填充到固定点数
        if len(point_cloud) > max_points:
            indices = np.random.choice(len(point_cloud), max_points, replace=False)
            point_cloud = point_cloud[indices]
        elif len(point_cloud) < max_points:
            # 填充零点
            padding = np.zeros((max_points - len(point_cloud), 3))
            point_cloud = np.concatenate([point_cloud, padding], axis=0)
        
        return torch.from_numpy(point_cloud).float()
    
    def load_traveluav_episode(self, episode_path: Path) -> List[UAVTrajectoryPoint]:
        """加载TravelUAV episode数据"""
        trajectory_points = []
        
        # 假设episode数据存储格式
        episode_data_file = episode_path / "trajectory.json"
        images_dir = episode_path / "images"
        pointclouds_dir = episode_path / "pointclouds"
        
        if not episode_data_file.exists():
            logger.warning(f"Episode data file not found: {episode_data_file}")
            return trajectory_points
        
        with open(episode_data_file, 'r') as f:
            episode_data = json.load(f)
        
        instruction = episode_data.get('instruction', '')
        trajectory = episode_data.get('trajectory', [])
        
        for i, point_data in enumerate(trajectory):
            # 加载图像
            images = {}
            for camera in ['head', 'left', 'right']:
                img_path = images_dir / f"step_{i:06d}_{camera}.jpg"
                if img_path.exists():
                    images[camera] = np.array(Image.open(img_path))
            
            # 加载点云
            pc_path = pointclouds_dir / f"step_{i:06d}.npy"
            point_cloud = None
            if pc_path.exists():
                point_cloud = np.load(pc_path)
            
            # 创建轨迹点
            trajectory_point = UAVTrajectoryPoint(
                timestamp=point_data.get('timestamp', i * 0.1),
                position=np.array(point_data['position']),
                orientation=np.array(point_data['orientation']),
                velocity=np.array(point_data['velocity']),
                angular_velocity=np.array(point_data['angular_velocity']),
                images=images,
                point_cloud=point_cloud,
                instruction=instruction,
                waypoint_progress=point_data.get('waypoint_progress', i / len(trajectory))
            )
            
            trajectory_points.append(trajectory_point)
        
        return trajectory_points


class FiSUAVDataset(Dataset):
    """FiS-UAV训练数据集"""
    
    def __init__(self,
                 data_paths: List[Path],
                 processor: TravelUAVDataProcessor,
                 sequence_length: int = 10,
                 async_sampling_config: Optional[Dict] = None):
        self.data_paths = data_paths
        self.processor = processor
        self.sequence_length = sequence_length
        self.async_config = async_sampling_config or {}
        
        # 加载所有数据
        self.episodes = []
        self._load_all_episodes()
        
        # 创建序列索引
        self.sequence_indices = []
        self._create_sequence_indices()
        
    def _load_all_episodes(self):
        """加载所有episode数据"""
        logger.info(f"Loading {len(self.data_paths)} episodes...")
        
        for episode_path in self.data_paths:
            try:
                trajectory_points = self.processor.load_traveluav_episode(episode_path)
                if len(trajectory_points) > 0:
                    self.episodes.append(trajectory_points)
            except Exception as e:
                logger.warning(f"Failed to load episode {episode_path}: {e}")
        
        logger.info(f"Loaded {len(self.episodes)} episodes")
    
    def _create_sequence_indices(self):
        """创建序列索引"""
        for episode_idx, episode in enumerate(self.episodes):
            for start_idx in range(len(episode) - self.sequence_length + 1):
                self.sequence_indices.append((episode_idx, start_idx))
        
        logger.info(f"Created {len(self.sequence_indices)} sequences")
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, start_idx = self.sequence_indices[idx]
        episode = self.episodes[episode_idx]
        
        # 提取序列
        sequence = episode[start_idx:start_idx + self.sequence_length]
        
        # 处理数据
        return self._process_sequence(sequence)
    
    def _process_sequence(self, sequence: List[UAVTrajectoryPoint]) -> Dict[str, torch.Tensor]:
        """处理序列数据"""
        batch_data = {
            'images_head': [],
            'images_left': [],
            'images_right': [],
            'point_clouds': [],
            'actions': [],
            'positions': [],
            'orientations': [],
            'instructions': [],
            'waypoint_progress': []
        }
        
        for point in sequence:
            # 处理图像
            for camera in ['head', 'left', 'right']:
                if camera in point.images:
                    img_tensor = self.processor.process_image(point.images[camera])
                    batch_data[f'images_{camera}'].append(img_tensor)
                else:
                    # 创建空图像
                    batch_data[f'images_{camera}'].append(torch.zeros(3, *self.processor.image_size))
            
            # 处理点云
            pc_tensor = self.processor.process_point_cloud(point.point_cloud)
            batch_data['point_clouds'].append(pc_tensor)
            
            # 处理动作
            action = point.to_6dof_action()
            if self.processor.normalize_actions:
                action = self.processor.normalize_action(action)
            batch_data['actions'].append(torch.from_numpy(action).float())
            
            # 其他数据
            batch_data['positions'].append(torch.from_numpy(point.position).float())
            batch_data['orientations'].append(torch.from_numpy(point.orientation).float())
            batch_data['instructions'].append(point.instruction)
            batch_data['waypoint_progress'].append(point.waypoint_progress)
        
        # 堆叠tensor数据
        for key in ['images_head', 'images_left', 'images_right', 'point_clouds', 
                   'actions', 'positions', 'orientations']:
            if batch_data[key]:
                batch_data[key] = torch.stack(batch_data[key])
        
        batch_data['waypoint_progress'] = torch.tensor(batch_data['waypoint_progress'])
        
        return batch_data


class DAggerDataAggregator:
    """DAgger数据聚合器"""
    
    def __init__(self, 
                 max_dataset_size: int = 100000,
                 expert_data_ratio: float = 0.3):
        self.max_dataset_size = max_dataset_size
        self.expert_data_ratio = expert_data_ratio
        
        self.model_data = []
        self.expert_data = []
        
    def add_model_data(self, data: List[Dict[str, Any]]):
        """添加模型生成的数据"""
        self.model_data.extend(data)
        self._maintain_dataset_size()
    
    def add_expert_data(self, data: List[Dict[str, Any]]):
        """添加专家纠正的数据"""
        self.expert_data.extend(data)
        self._maintain_dataset_size()
    
    def _maintain_dataset_size(self):
        """维护数据集大小"""
        total_size = len(self.model_data) + len(self.expert_data)
        
        if total_size > self.max_dataset_size:
            # 计算目标大小
            target_expert_size = int(self.max_dataset_size * self.expert_data_ratio)
            target_model_size = self.max_dataset_size - target_expert_size
            
            # 随机采样保持比例
            if len(self.expert_data) > target_expert_size:
                indices = np.random.choice(len(self.expert_data), target_expert_size, replace=False)
                self.expert_data = [self.expert_data[i] for i in indices]
            
            if len(self.model_data) > target_model_size:
                indices = np.random.choice(len(self.model_data), target_model_size, replace=False)
                self.model_data = [self.model_data[i] for i in indices]
    
    def get_aggregated_dataset(self) -> List[Dict[str, Any]]:
        """获取聚合后的数据集"""
        return self.model_data + self.expert_data
    
    def save_dataset(self, save_path: Path):
        """保存数据集"""
        dataset = {
            'model_data': self.model_data,
            'expert_data': self.expert_data,
            'metadata': {
                'total_size': len(self.model_data) + len(self.expert_data),
                'model_data_size': len(self.model_data),
                'expert_data_size': len(self.expert_data),
                'expert_ratio': len(self.expert_data) / (len(self.model_data) + len(self.expert_data))
            }
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"Dataset saved to {save_path}")
        logger.info(f"Total size: {dataset['metadata']['total_size']}")
        logger.info(f"Expert ratio: {dataset['metadata']['expert_ratio']:.2%}")
    
    def load_dataset(self, load_path: Path):
        """加载数据集"""
        with open(load_path, 'rb') as f:
            dataset = pickle.load(f)
        
        self.model_data = dataset['model_data']
        self.expert_data = dataset['expert_data']
        
        logger.info(f"Dataset loaded from {load_path}")
        logger.info(f"Total size: {dataset['metadata']['total_size']}")


def create_dataloader(dataset: FiSUAVDataset,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def load_traveluav_12k_dataset(dataset_root: Path,
                              split: str = 'train',
                              processor: Optional[TravelUAVDataProcessor] = None) -> FiSUAVDataset:
    """加载TravelUAV 12k数据集"""
    if processor is None:
        processor = TravelUAVDataProcessor()
    
    # 查找episode路径
    split_dir = dataset_root / split
    episode_paths = []
    
    if split_dir.exists():
        for episode_dir in split_dir.iterdir():
            if episode_dir.is_dir():
                episode_paths.append(episode_dir)
    
    logger.info(f"Found {len(episode_paths)} episodes in {split} split")
    
    return FiSUAVDataset(
        data_paths=episode_paths,
        processor=processor
    )


# 示例使用
if __name__ == "__main__":
    # 创建数据处理器
    processor = TravelUAVDataProcessor(
        image_size=(224, 224),
        normalize_actions=True
    )
    
    # 加载数据集
    dataset_root = Path("/path/to/traveluav/12k/dataset")
    if dataset_root.exists():
        dataset = load_traveluav_12k_dataset(dataset_root, 'train', processor)
        dataloader = create_dataloader(dataset, batch_size=4)
        
        # 测试数据加载
        for batch in dataloader:
            print("Batch keys:", batch.keys())
            print("Images head shape:", batch['images_head'].shape)
            print("Actions shape:", batch['actions'].shape)
            break
    else:
        print(f"Dataset not found at {dataset_root}")
        print("This is a demonstration of the data processing pipeline")