import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from util.depth_to_pointcloud import DepthToPointCloud

class UAVDataProcessor:
    """Process UAV data for FiS integration"""
    
    def __init__(self):
        self.depth_to_pc = DepthToPointCloud()
    
    def process_uav_state(self, uav_state: Dict) -> np.ndarray:
        """
        Convert UAV state to 6-DoF representation
        
        Args:
            uav_state: Dictionary containing position and orientation
            
        Returns:
            state_6dof: [x, y, z, roll, pitch, yaw]
        """
        position = uav_state['position']  # [x, y, z]
        orientation = uav_state['orientation']  # [x, y, z, w] quaternion
        
        # Convert quaternion to Euler angles
        quat = [orientation[3], orientation[0], orientation[1], orientation[2]]  # [w, x, y, z]
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
        
        return np.concatenate([position, euler])
    
    def process_uav_observation(self, observation: Dict) -> Dict:
        """
        Process UAV observation for FiS model
        
        Args:
            observation: UAV observation dictionary
            
        Returns:
            processed_obs: Processed observation for FiS
        """
        processed = {}
        
        # Process RGB images
        if 'rgb' in observation:
            processed['pixel_values'] = self._process_rgb_images(observation['rgb'])
        
        # Process depth images to point cloud
        if 'depth' in observation:
            processed['pointcloud'] = self._process_depth_to_pointcloud(
                observation['depth'], 
                observation.get('rgb', None)
            )
        
        # Process UAV state
        if 'sensor' in observation and 'state' in observation['sensor']:
            processed['proprio'] = self.process_uav_state(observation['sensor']['state'])
        
        # Process instruction
        if 'instruction' in observation:
            processed['instruction'] = observation['instruction']
        
        return processed
    
    def _process_rgb_images(self, rgb_images: List[np.ndarray]) -> torch.Tensor:
        """Process RGB images for vision backbone"""
        # Take front camera as primary view
        if isinstance(rgb_images, list) and len(rgb_images) > 0:
            primary_image = rgb_images[0]  # Front camera
        else:
            primary_image = rgb_images
        
        # Normalize and convert to tensor
        if isinstance(primary_image, np.ndarray):
            image = torch.from_numpy(primary_image).float() / 255.0
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)  # HWC -> CHW
            return image.unsqueeze(0)  # Add batch dimension
        
        return primary_image
    
    def _process_depth_to_pointcloud(self, 
                                   depth_images: List[np.ndarray],
                                   rgb_images: Optional[List[np.ndarray]] = None) -> torch.Tensor:
        """Convert depth images to point cloud"""
        if not isinstance(depth_images, list):
            depth_images = [depth_images]
        
        if rgb_images is not None and not isinstance(rgb_images, list):
            rgb_images = [rgb_images]
        
        # Use front camera depth
        depth = depth_images[0] if len(depth_images) > 0 else np.zeros((512, 512))
        rgb = rgb_images[0] if rgb_images and len(rgb_images) > 0 else None
        
        # Convert to point cloud
        pointcloud = self.depth_to_pc.depth_to_pointcloud(depth, rgb)
        
        # Subsample if too many points
        if len(pointcloud) > 8192:
            indices = np.random.choice(len(pointcloud), 8192, replace=False)
            pointcloud = pointcloud[indices]
        elif len(pointcloud) < 1024:
            # Pad if too few points
            padding = np.zeros((1024 - len(pointcloud), pointcloud.shape[1]))
            pointcloud = np.concatenate([pointcloud, padding], axis=0)
        
        return torch.from_numpy(pointcloud).float()
    
    def create_training_batch(self, uav_observations: List[Dict]) -> Dict:
        """
        Create training batch from UAV observations
        
        Args:
            uav_observations: List of UAV observations
            
        Returns:
            batch: Training batch for FiS model
        """
        batch = {
            'pixel_values': [],
            'pointcloud': [],
            'proprio': [],
            'actions': [],
            'input_ids': [],
            'labels': []
        }
        
        for obs in uav_observations:
            processed = self.process_uav_observation(obs)
            
            batch['pixel_values'].append(processed.get('pixel_values'))
            batch['pointcloud'].append(processed.get('pointcloud'))
            batch['proprio'].append(processed.get('proprio'))
            
            # Extract teacher action if available
            if 'teacher_action' in obs:
                teacher_action = obs['teacher_action']
                if len(teacher_action) > 0:
                    # Convert to 6-DoF if needed
                    action_6dof = self._convert_to_6dof(teacher_action[0])
                    batch['actions'].append(action_6dof)
        
        # Stack tensors
        for key in batch:
            if batch[key] and batch[key][0] is not None:
                batch[key] = torch.stack(batch[key])
        
        return batch
    
    def _convert_to_6dof(self, action: np.ndarray) -> torch.Tensor:
        """Convert action to 6-DoF format"""
        if len(action) >= 6:
            return torch.from_numpy(action[:6]).float()
        else:
            # Pad if less than 6 dimensions
            padded = np.zeros(6)
            padded[:len(action)] = action
            return torch.from_numpy(padded).float()
    
    def process_teacher_action(self, waypoints: List[List[float]], current_pose: Dict[str, np.ndarray]) -> np.ndarray:
        """将7个航点列表和当前姿态转换为6-DoF动作。

        Args:
            waypoints: 7个航点 [[x, y, z], ...]
            current_pose: 包含 'position' 和 'orientation' 的字典

        Returns:
            6-DoF动作 [dx, dy, dz, 0, 0, 0]
        """
        if not waypoints or len(waypoints) == 0:
            return np.zeros(6)

        # 计算到第一个航点的位移
        first_waypoint = np.array(waypoints[0])
        current_position = current_pose["position"]
        displacement = first_waypoint - current_position

        # 假设没有旋转，只进行位移
        return np.array([displacement[0], displacement[1], displacement[2], 0, 0, 0])