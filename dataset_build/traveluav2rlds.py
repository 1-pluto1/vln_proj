import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional

import os
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from typing import Dict, Any, List


# 在文件顶部导入
import numpy as np

# 添加四元数到欧拉角的转换函数
def quaternion_to_euler(q):
    x, y, z, w = q
    
    # roll (x轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # pitch (y轴旋转)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # yaw (z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return [roll, pitch, yaw]



def convert_traveluav_to_rlds(traveluav_data_dir: str, output_dir: str):
    """
    Convert TravelUAV data to RLDS format
    
    Args:
        traveluav_data_dir: TravelUAV data directory
        output_dir: Output RLDS data directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Traverse TravelUAV data directory for trajectories
    trajectories = []
    traj_dirs = [d for d in Path(traveluav_data_dir).iterdir() if d.is_dir()]
    
    for traj_dir in traj_dirs:
        # Read object_description.json to get target position and object description

        mark_file = traj_dir / "mark.json"
        target_position = [0, 0, 0]
        object_description = ""
        
        # Get target position and object name from mark.json
        if mark_file.exists():
            try:
                with open(mark_file, "r") as f:
                    mark_data = json.load(f)
                    if "target" in mark_data and "position" in mark_data["target"]:
                        target_position = mark_data["target"]["position"]

            except json.JSONDecodeError:
                print(f"Error parsing mark.json in {traj_dir}")
        
        # Get object description from object_description.json
        object_description_file = traj_dir / "object_description.json"
        if object_description_file.exists():
            try:
                with open(object_description_file, "r") as f:
                    description_data = json.load(f)
                    if isinstance(description_data, list) and len(description_data) > 0:
                        object_description = description_data[0]
            except json.JSONDecodeError:
                print(f"Error parsing object_description.json in {traj_dir}")
                object_description = "unknown"
        
        # Read trajectory log data
        log_dir = traj_dir / "log"
        if not log_dir.exists():
            continue
        
        log_files = sorted(log_dir.glob("*.json"))
        if not log_files:
            continue
        
        # Read all states
        states = []
        for log_file in log_files:
            with open(log_file, "r") as f:
                states.append(json.load(f))
        
        # Build trajectory
        traj_data = {
            "observation": {
                "base_pose_xyz": [],
                "base_pose_rpy": [],
                "image_head_slow": [],  # 前摄像头作为head_slow
                "image_head_fast": [],  # 前摄像头作为head_fast
                "image_left_slow": [],  # 左摄像头
                "image_left_fast": [],  # 左摄像头
                "image_right_slow": [], # 右摄像头
                "image_right_fast": []  # 右摄像头
            },
            "action": [],
            "task": {
                "language_instruction": f"Navigate to the {object_description} located at coordinates [{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}]."
                if object_description == "unknown" else f"Navigate to {object_description}"
            },
            "dataset_name": "traveluav_dataset"
        }
        
        # Process each timestep
        for i in range(len(states)):
            state = states[i]
            
            # Extract position and orientation from the nested structure
            if "sensors" in state and "state" in state["sensors"]:
                state_data = state["sensors"]["state"]
                position = state_data.get("position", [0, 0, 0])
                orientation = state_data.get("orientation", [0, 0, 0, 0])
                # 如果方向是四元数格式(x,y,z,w)，转换为欧拉角(roll,pitch,yaw)
                if len(orientation) == 4:
                    orientation = quaternion_to_euler(orientation)
            else:
                position = state.get("position", [0, 0, 0])
                orientation = state.get("orientation", [0, 0, 0])
            
            # Add to trajectory data
            traj_data["observation"]["base_pose_xyz"].append(position)
            traj_data["observation"]["base_pose_rpy"].append(orientation)
            
            # Process images - based on TravelUAV dataset format
            frame_id = state.get("frame", i * 5)  # 使用frame字段或默认计算
            front_img_path = str(traj_dir / "frontcamera" / f"{frame_id:06d}.png")
            left_img_path = str(traj_dir / "leftcamera" / f"{frame_id:06d}.png")
            right_img_path = str(traj_dir / "rightcamera" / f"{frame_id:06d}.png")
            
            # 添加各个摄像头的图像路径
            traj_data["observation"]["image_head_slow"].append(front_img_path)
            traj_data["observation"]["image_head_fast"].append(front_img_path)
            traj_data["observation"]["image_left_slow"].append(left_img_path)
            traj_data["observation"]["image_left_fast"].append(left_img_path)
            traj_data["observation"]["image_right_slow"].append(right_img_path)
            traj_data["observation"]["image_right_fast"].append(right_img_path)
            
            # Calculate action (if not the last state)
            if i < len(states) - 1:
                next_state = states[i + 1]
                
                # 从下一个状态提取位置和方向
                if "sensors" in next_state and "state" in next_state["sensors"]:
                    next_state_data = next_state["sensors"]["state"]
                    next_position = next_state_data.get("position", [0, 0, 0])
                    next_orientation = next_state_data.get("orientation", [0, 0, 0, 0])
                    if len(next_orientation) == 4:
                        next_orientation = quaternion_to_euler(next_orientation)
                else:
                    next_position = next_state.get("position", [0, 0, 0])
                    next_orientation = next_state.get("orientation", [0, 0, 0])
                
                # Calculate position difference
                delta_pos = [next_position[j] - position[j] for j in range(3)]
                # Calculate orientation difference
                delta_rot = [next_orientation[j] - orientation[j] for j in range(len(orientation)) if j < 3]
                # 确保delta_rot有3个元素
                while len(delta_rot) < 3:
                    delta_rot.append(0)
                
                action = delta_pos + delta_rot
            else:
                # Set action to zero for the last state
                action = [0, 0, 0, 0, 0, 0]
            
            traj_data["action"].append(action)
        
        # Convert lists to numpy arrays
        traj_data["observation"]["base_pose_xyz"] = np.array(traj_data["observation"]["base_pose_xyz"], dtype=np.float32)
        traj_data["observation"]["base_pose_rpy"] = np.array(traj_data["observation"]["base_pose_rpy"], dtype=np.float32)
        traj_data["action"] = np.array(traj_data["action"], dtype=np.float32)
        
        trajectories.append(traj_data)
    
    # 保存为RLDS格式
    for i, traj in enumerate(trajectories):
        traj_dir = os.path.join(output_dir, f"trajectory_{i:05d}")
        os.makedirs(traj_dir, exist_ok=True)
        
        # 保存轨迹数据
        with open(os.path.join(traj_dir, "data.json"), "w") as f:
            # 将numpy数组转换为列表以便JSON序列化
            serializable_traj = {
                "observation": {
                    "base_pose_xyz": traj["observation"]["base_pose_xyz"].tolist(),
                    "base_pose_rpy": traj["observation"]["base_pose_rpy"].tolist(),
                    "image_head_slow": traj["observation"]["image_head_slow"],
                    "image_head_fast": traj["observation"]["image_head_fast"],
                    "image_left_slow": traj["observation"]["image_left_slow"],
                    "image_left_fast": traj["observation"]["image_left_fast"],
                    "image_right_slow": traj["observation"]["image_right_slow"],
                    "image_right_fast": traj["observation"]["image_right_fast"]
                },
                "action": traj["action"].tolist(),
                "task": traj["task"],
                "dataset_name": traj["dataset_name"]
            }
            json.dump(serializable_traj, f, indent=2)
    
    # 创建数据集信息文件
    dataset_info = {
        "name": "traveluav_dataset",
        "description": "TravelUAV dataset converted to RLDS format",
        "version": "1.0.0",
        "splits": {
            "train": {"num_trajectories": len(trajectories)}
        }
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"成功转换 {len(trajectories)} 条轨迹到RLDS格式")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将TravelUAV数据转换为RLDS格式")
    parser.add_argument("--input_dir", type=str, required=True, help="TravelUAV数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出RLDS数据目录")
    
    args = parser.parse_args()
    convert_traveluav_to_rlds(args.input_dir, args.output_dir)