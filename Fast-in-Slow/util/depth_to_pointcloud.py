import numpy as np
import torch
from typing import Tuple, Optional

class DepthToPointCloud:
    """Convert depth images to point clouds for FiS integration"""
    
    def __init__(self, 
                 camera_intrinsics: Optional[dict] = None,
                 max_depth: float = 100.0,
                 min_depth: float = 0.1):
        self.max_depth = max_depth
        self.min_depth = min_depth
        
        # Default camera intrinsics for AirSim
        if camera_intrinsics is None:
            self.fx = 512.0  # focal length x
            self.fy = 512.0  # focal length y
            self.cx = 512.0  # principal point x
            self.cy = 512.0  # principal point y
        else:
            self.fx = camera_intrinsics['fx']
            self.fy = camera_intrinsics['fy']
            self.cx = camera_intrinsics['cx']
            self.cy = camera_intrinsics['cy']
    
    def depth_to_pointcloud(self, 
                           depth_image: np.ndarray,
                           rgb_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert depth image to point cloud
        
        Args:
            depth_image: (H, W) depth image in meters
            rgb_image: (H, W, 3) RGB image (optional)
            
        Returns:
            point_cloud: (N, 3) or (N, 6) point cloud [x, y, z] or [x, y, z, r, g, b]
        """
        h, w = depth_image.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Filter valid depth values
        valid_mask = (depth_image > self.min_depth) & (depth_image < self.max_depth)
        
        # Convert to 3D coordinates
        z = depth_image[valid_mask]
        x = (u[valid_mask] - self.cx) * z / self.fx
        y = (v[valid_mask] - self.cy) * z / self.fy
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=1)
        
        # Add color if RGB image provided
        if rgb_image is not None:
            colors = rgb_image[valid_mask] / 255.0
            points = np.concatenate([points, colors], axis=1)
        
        return points
    
    def multi_view_to_pointcloud(self, 
                                depth_images: list,
                                rgb_images: list,
                                camera_poses: list) -> np.ndarray:
        """
        Convert multiple depth images to unified point cloud
        
        Args:
            depth_images: List of depth images
            rgb_images: List of RGB images
            camera_poses: List of camera poses (4x4 transformation matrices)
            
        Returns:
            unified_pointcloud: Combined point cloud from all views
        """
        all_points = []
        
        for depth, rgb, pose in zip(depth_images, rgb_images, camera_poses):
            # Convert to point cloud
            points = self.depth_to_pointcloud(depth, rgb)
            
            if len(points) == 0:
                continue
                
            # Transform to world coordinates
            if points.shape[1] >= 3:
                xyz = points[:, :3]
                xyz_homo = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
                xyz_world = (pose @ xyz_homo.T).T[:, :3]
                
                if points.shape[1] > 3:
                    points_world = np.concatenate([xyz_world, points[:, 3:]], axis=1)
                else:
                    points_world = xyz_world
                    
                all_points.append(points_world)
        
        if all_points:
            return np.concatenate(all_points, axis=0)
        else:
            return np.empty((0, 6))  # Empty point cloud with color channels