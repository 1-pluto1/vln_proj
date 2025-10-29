"""
uav_dataset.py

UAV-specific dataset implementation for TravelUAV integration with Fast-in-Slow VLA framework.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Optional
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from models.backbones.llm.prompting import PromptBuilder
from models.backbones.vision import ImageTransform
from vla.action_tokenizer import ActionTokenizer
from util.data_utils import tree_map
from vla.action_tokenizer import ActionTokenizer
from vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100




# @dataclass
# class UAVBatchTransform:
#     action_tokenizer: ActionTokenizer
#     base_tokenizer: PreTrainedTokenizerBase
#     image_transform: ImageTransform
#     prompt_builder_fn: Type[PromptBuilder]
#     predict_stop_token: bool = True
#     load_pointcloud: bool = False
#     action_chunk: int = 1
#     lang_subgoals_exist: bool = False

#     def __call__(self, uav_batch: Dict[str, Any]) -> Dict[str, Any]:
#         """Converts a UAV batch to the format expected by the OpenVLA collator/models."""
        
#         observation = uav_batch["observation"] if "observation" in uav_batch else {}
#         action = uav_batch.get("action", np.zeros((1, 6)))  # 默认6-DoF动作
#         dataset_name = uav_batch.get("dataset_name", "traveluav_dataset")
        
#         # 处理本体感受数据，确保与RLDS格式一致
#         proprio = None
#         if "proprio" in observation:
#             proprio = observation["proprio"]
#         elif "sensors" in observation and "state" in observation["sensors"]:
#             proprio = self.uav_processor.process_uav_state(observation["sensors"]["state"])
#             proprio = np.array(proprio) if not isinstance(proprio, np.ndarray) else proprio
#         elif "position" in uav_batch:
#             # 从TravelUAV数据中提取位置和方向信息
#             position = uav_batch.get("position", [0, 0, 0])
#             orientation = uav_batch.get("orientation", [0, 0, 0])
#             proprio = np.concatenate([position, orientation])
#         else:
#             proprio = np.zeros(6)  # 默认6维本体感受向量
        
#         # 处理TravelUAV特定的数据格式
#         # 从mark.json和merged_data.json中提取数据
#         if "mark" in uav_batch:
#             mark_data = uav_batch["mark"]
#             # 提取起点、终点和目标位置
#             start_pos = mark_data.get("start", [0, 0, 0])
#             end_pos = mark_data.get("end", [0, 0, 0])
#             target = mark_data.get("target", {"position": [0, 0, 0], "rotation": [0, 0]})
        
#         # 获取指令(语言)，确保与RLDS格式一致
#         lang = ""
#         if "task" in uav_batch and "language_instruction" in uav_batch["task"]:
#             lang_data = uav_batch["task"]["language_instruction"]
#             if isinstance(lang_data, bytes):
#                 lang = lang_data.decode().lower()
#             else:
#                 lang = str(lang_data).lower()
#         elif "instruction" in uav_batch:
#             lang = str(uav_batch["instruction"]).lower()
#         else:
#             # 为TravelUAV数据集生成导航指令
#             object_name = uav_batch.get("mark", {}).get("object_name", "destination")
#             lang = f"navigate to the {object_name}"
        
#         # 处理语言子目标，与RLDS格式保持一致
#         lang_subgoals = ""
#         if self.lang_subgoals_exist is True:
#             if "task" in uav_batch and "language_subgoals" in uav_batch["task"]:
#                 subgoal_data = uav_batch["task"]["language_subgoals"]
#                 if isinstance(subgoal_data, bytes):
#                     lang_subgoals = subgoal_data.decode().lower()
#                 else:
#                     lang_subgoals = str(subgoal_data).lower()
#             elif "subgoals" in uav_batch:
#                 lang_subgoals = str(uav_batch["subgoals"]).lower()
            
#             if lang_subgoals:
#                 lang_subgoals = f"Now, you need to do the following operation: {lang_subgoals}."
        
#         # 确保proprio是正确的形状
#         if proprio is None:
#             proprio = np.zeros(6)  # 默认6维本体感受向量
        
#         if isinstance(proprio, list):
#             proprio = np.array(proprio)
        
#         # 确保proprio是二维数组
#         if len(proprio.shape) == 1:
#             proprio = proprio.reshape(1, -1)
        
#         # 根据条件创建对话格式(匹配RLDSBatchTransform逻辑)
#         if self.action_tokenizer is None and not self.lang_subgoals_exist:
#             conversation = [
#                 {"from": "human", "value": f"What action should the UAV take to {lang}?"},
#                 {"from": "gpt", "value": "<BOD><EOD>"},
#             ]
#         elif self.action_tokenizer is not None and not self.lang_subgoals_exist:
#             gpt_values = ""
#             # 根据动作类型处理
#             if isinstance(action, (list, np.ndarray)) and len(action) > 0 and hasattr(action, '__iter__'):
#                 for act in action:
#                     if isinstance(act, np.ndarray):
#                         act_tensor = torch.from_numpy(act[:6]).float()  # 确保6-DoF
#                     else:
#                         act_tensor = torch.tensor(act[:6]).float()
#                     gpt_values += self.action_tokenizer(act_tensor)
#             else:
#                 if isinstance(action, np.ndarray):
#                     act_tensor = torch.from_numpy(action[:6]).float() if len(action.shape) == 1 else torch.from_numpy(action[0][:6]).float()
#                 else:
#                     act_tensor = torch.tensor(action[:6]).float() if len(action) == 6 else torch.tensor(action[0][:6]).float()
#                 gpt_values += self.action_tokenizer(act_tensor)
            
#             conversation = [
#                 {"from": "human", "value": f"What action should the UAV take to {lang}?"},
#                 {"from": "gpt", "value": f"<BOD><EOD>{gpt_values}"},
#             ]
#         elif self.action_tokenizer is None and self.lang_subgoals_exist:
#             conversation = [
#                 {"from": "human", "value": f"What action should the UAV take to {lang}?"},
#                 {"from": "gpt", "value": f"<BOD><EOD>{lang_subgoals}"},
#             ]
#         else:
#             gpt_values = ""
#             # 根据动作类型处理
#             if isinstance(action, (list, np.ndarray)) and len(action) > 0 and hasattr(action, '__iter__'):
#                 for act in action:
#                     if isinstance(act, np.ndarray):
#                         act_tensor = torch.from_numpy(act[:6]).float()  # 确保6-DoF
#                     else:
#                         act_tensor = torch.tensor(act[:6]).float()
#                     gpt_values += self.action_tokenizer(act_tensor)
#             else:
#                 if isinstance(action, np.ndarray):
#                     act_tensor = torch.from_numpy(action[:6]).float() if len(action.shape) == 1 else torch.from_numpy(action[0][:6]).float()
#                 else:
#                     act_tensor = torch.tensor(action[:6]).float() if len(action) == 6 else torch.tensor(action[0][:6]).float()
#                 gpt_values += self.action_tokenizer(act_tensor)
            
#             conversation = [
#                 {"from": "human", "value": f"What action should the UAV take to {lang}?"},
#                 {"from": "gpt", "value": f"<BOD><EOD>{gpt_values}{lang_subgoals}"},
#             ]

#         # 构建基于聊天的提示
#         prompt_builder = self.prompt_builder_fn("openvla")
#         for turn in conversation:
#             prompt_builder.add_turn(turn["from"], turn["value"])
        
#         # 标记化
#         input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
#         labels = list(input_ids)
        
#         # 张量化
#         input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

#         # 处理图像 - 将TravelUAV相机格式映射到预期格式
#         images = {
#             "head_slow": None,
#             "head_fast": None,
#             "right_slow": None,
#             "right_fast": None,
#             "left_slow": None,
#             "left_fast": None,
#         }
        
#         # 处理TravelUAV特定的相机格式
#         # 从目录结构中可以看到frontcamera, leftcamera, rightcamera, rearcamera, downcamera
#         camera_dirs = {
#             "frontcamera": "head_slow",
#             "leftcamera": "left_slow",
#             "rightcamera": "right_slow",
#             "rearcamera": "rear_slow",
#             "downcamera": "down_slow",
#         }
        
#         # 处理来自observation的图像
#         if "rgb" in observation:
#             rgb_images = observation["rgb"]
#             if isinstance(rgb_images, list) and len(rgb_images) > 0:
#                 images["head_slow"] = Image.fromarray(rgb_images[0])  # 前置相机
#                 if len(rgb_images) > 1:
#                     images["right_slow"] = Image.fromarray(rgb_images[1])  # 右侧相机(如果可用)
#                 if len(rgb_images) > 2:
#                     images["left_slow"] = Image.fromarray(rgb_images[2])  # 左侧相机(如果可用)
#             elif isinstance(rgb_images, np.ndarray):
#                 images["head_slow"] = Image.fromarray(rgb_images)
        
#         # 处理TravelUAV特定的相机图像
#         for camera_key, img_key in camera_dirs.items():
#             if camera_key in uav_batch:
#                 img_data = uav_batch[camera_key]
#                 if isinstance(img_data, list) and len(img_data) > 0:
#                     images[img_key] = Image.fromarray(img_data[0])
#                 elif isinstance(img_data, np.ndarray):
#                     images[img_key] = Image.fromarray(img_data)
#                 elif isinstance(img_data, str) and img_data.endswith(".png"):
#                     # 处理图像路径
#                     try:
#                         img_path = Path(img_data)
#                         if img_path.exists():
#                             images[img_key] = Image.open(img_path).convert('RGB')
#                     except Exception:
#                         pass
        
#         # 处理特定相机视图(如果存在)
#         camera_mapping = {
#             "rgb_front": "head_slow",
#             "rgb_right": "right_slow",
#             "rgb_left": "left_slow",
#             "rgb_front_hd": "head_fast",
#             "rgb_right_hd": "right_fast",
#             "rgb_left_hd": "left_fast",
#         }
        
#         for obs_key, img_key in camera_mapping.items():
#             if obs_key in observation and observation[obs_key] is not None:
#                 img_data = observation[obs_key]
#                 if isinstance(img_data, list) and len(img_data) > 0:
#                     images[img_key] = Image.fromarray(img_data[0])
#                 elif isinstance(img_data, np.ndarray):
#                     images[img_key] = Image.fromarray(img_data)
        
#         # 复用slow图像作为fast图像
#         # 创建slow到fast的映射关系，与datasets.py中的key_mapping保持一致
#         slow_to_fast_mapping = {
#             "head_slow": "head_fast",
#             "right_slow": "right_fast",
#             "left_slow": "left_fast",
#         }
        
#         # 复制slow图像到对应的fast位置
#         for slow_key, fast_key in slow_to_fast_mapping.items():
#             if slow_key in images and fast_key not in images:
#                 images[fast_key] = images[slow_key]
        
#         # 转换图像
#         pixel_values = {}
#         for prefix, img in images.items():
#             if img is not None:
#                 transformed = self.image_transform(img)
#                 for key, value in transformed.items():
#                     pixel_values[f"{prefix}_{key}"] = value

#         # 处理点云
#         point_cloud = None
#         if self.load_pointcloud:
#             if "pointcloud" in observation:
#                 point_cloud = observation["pointcloud"]
#                 if isinstance(point_cloud, list) and len(point_cloud) > 0:
#                     point_cloud = point_cloud[0]
#                 point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
#             elif "depth" in observation:
#                 depth_images = observation["depth"]
#                 rgb_images = observation.get("rgb", None)
#                 point_cloud = _process_depth_to_pointcloud(depth_images, rgb_images)
#                 if point_cloud is not None:
#                     point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
#             # 处理TravelUAV特定的深度相机数据
#             for depth_key in ["frontcamera_depth", "leftcamera_depth", "rightcamera_depth"]:
#                 if depth_key in uav_batch and point_cloud is None:
#                     depth_data = uav_batch[depth_key]
#                     if isinstance(depth_data, np.ndarray):
#                         # 使用第一个可用的深度图像
#                         point_cloud = _process_depth_to_pointcloud(depth_data, None)
#                         if point_cloud is not None:
#                             point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
#                             break

#         # 处理动作
#         action_mask = None
#         if "action_mask" in uav_batch:
#             action_mask = torch.tensor(uav_batch["action_mask"], dtype=torch.bool)
        
#         # 将动作转换为具有适当形状的张量
#         if isinstance(action, np.ndarray):
#             action = torch.from_numpy(action).float()
#         elif isinstance(action, list):
#             action = torch.tensor(action).float()
        
#         # 重塑动作以匹配预期格式
#         if len(action.shape) == 1:
#             action = action.unsqueeze(0)
#         action = action.reshape(self.action_chunk, -1)
        
#         # 将本体感受转换为张量
#         proprio = torch.tensor(proprio, dtype=torch.float32)
#         if len(proprio.shape) == 1:
#             proprio = proprio.unsqueeze(0)

#         # 设置训练标签
#         if self.action_tokenizer is None and not self.lang_subgoals_exist:
#             labels[:] = IGNORE_INDEX
#         else:
#             matches = (labels == 32002).nonzero().view(-1)
#             last_position = matches[-1].item() if len(matches) > 0 else None
#             if last_position is not None:
#                 labels[:last_position+2] = IGNORE_INDEX

#         if not self.predict_stop_token:
#             labels[-1] = IGNORE_INDEX

#         # 以与RLDSBatchTransform相同的格式返回
#         return dict(
#             pixel_values=pixel_values, 
#             pointcloud=point_cloud, 
#             input_ids=input_ids, 
#             labels=labels, 
#             dataset_name=dataset_name, 
#             actions=action, 
#             action_masks=action_mask, 
#             proprio=proprio
#         )


# class UAVDataset(IterableDataset):
#     """UAV Dataset for TravelUAV integration with VLA framework"""

#     def __init__(
#         self,
#         data_root_dir: Path,
#         batch_transform: UAVBatchTransform,
#         resize_resolution: Tuple[int, int],
#         train: bool = True,
#         load_pointcloud: bool = False,
#         action_chunk: int = 1,
#         # Added for compatibility with RLDSDataset
#         data_mix: str = "uav_dataset",
#         shuffle_buffer_size: int = 1000,
#         future_action_window_size: int = 0,
#         past_action_window_size: int = 0,
#         image_aug: bool = False,
#         load_all_data_for_training: bool = True,
#         camera_view: Optional[str] = "primary",
#     ) -> None:
#         """UAV Dataset for loading TravelUAV data"""
#         self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

#         self.resize_resolution = resize_resolution
#         self.train = train
#         self.load_pointcloud = load_pointcloud
#         self.action_chunk = action_chunk

#         camera_view = tuple(camera_view.split(","))
#         # Discover all trajectory paths
#         self.trajectory_paths = []
#         for scene_dir in self.data_root_dir.iterdir():
#             if scene_dir.is_dir():
#                 for traj_dir in scene_dir.iterdir():
#                     if traj_dir.is_dir() and (traj_dir / "log").exists():
#                         self.trajectory_paths.append(traj_dir)
        
#         self.dataset_length = len(self.trajectory_paths)

#     def __iter__(self) -> Dict[str, Any]:
#         """Iterate over UAV dataset"""
#         for traj_path in self.trajectory_paths:
#             try:
#                 with open(traj_path / "object_description.json", "r") as f:
#                     instruction = json.load(f)[0]
#             except (FileNotFoundError, IndexError):
#                 instruction = "navigate to the destination"

#             log_dir = traj_path / "log"
#             log_files = sorted(log_dir.glob("*.json"))

#             if not log_files:
#                 continue

#             # Load all states first to compute teacher actions
#             states = []
#             for log_file in log_files:
#                 with open(log_file, "r") as f:
#                     states.append(json.load(f))

#             for i in range(len(states) - 1):
#                 current_state_data = states[i]
#                 next_state_data = states[i+1]

#                 # Construct observation
#                 frame_id = current_state_data["frame"]
#                 image_path = traj_path / "rgb" / f"{frame_id:06d}.png"
#                 if not image_path.exists():
#                     continue
                
#                 rgb_image = Image.open(image_path).convert("RGB")

#                 observation = {
#                     "rgb": [np.array(rgb_image)],
#                     "sensors": {
#                         "state": current_state_data["sensors"]["state"]
#                     },
#                     "instruction": {"text": instruction}
#                 }

#                 # Calculate teacher action
#                 current_pos = np.array(current_state_data["sensors"]["state"]["position"])
#                 next_pos = np.array(next_state_data["sensors"]["state"]["position"])
#                 # Calculate displacement to the first waypoint
#                 first_waypoint = np.array(next_pos["position"])
#                 current_position = np.array(current_pos["position"])
#                 displacement = first_waypoint - current_position

#                 # Assume no rotation, only displacement
#                 teacher_action = np.array([displacement[0], displacement[1], displacement[2], 0, 0, 0])

#                 uav_batch = {
#                     "observation": observation,
#                     "action": teacher_action,
#                     "instruction": instruction,
#                 }
#                 yield self.batch_transform(uav_batch)

#     def __len__(self) -> int:
#         return self.dataset_length

#     def __getitem__(self, idx: int) -> None:
#         raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

@dataclass
class UAVBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    load_pointcloud: bool = False
    action_chunk: int = 1
    lang_subgoals_exist: bool = False

    def __call__(self, uav_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""

        dataset_name, action, proprio = uav_batch["dataset_name"], uav_batch["action"], uav_batch["observation"]["proprio"]
        lang = uav_batch["task"]["language_instruction"].decode().lower()

        lang_subgoals = ""
        if self.lang_subgoals_exist is True and 'language_subgoals' in uav_batch["task"]:
            lang_subgoals = uav_batch["task"]["language_subgoals"].decode().lower()
            lang_subgoals = f"Now, you need to do the following operation: {lang_subgoals}."

        # If action tokenizer is not used, we don't add the action to the chat answer
        if self.action_tokenizer is None and self.lang_subgoals_exist is None:
            conversation = [
                {"from": "human", "value": f"What action should the uav take to {lang}?"},
                {"from": "gpt", "value": "<BOD><EOD>"},
            ]
        elif self.action_tokenizer is not None and self.lang_subgoals_exist is None:
            gpt_values = ""
            for act in action:
                gpt_values += self.action_tokenizer(act[:len(proprio[0])])
            # cur_robot_state = ""
            # for pro in proprio:
            #     cur_robot_state += self.action_tokenizer(pro)
            ##  The current robot state is {cur_robot_state}. 
            conversation = [
                {"from": "human", "value": f"What action should the uav take to {lang}?"},
                {"from": "gpt", "value": f"<BOD><EOD>{gpt_values}"},
            ]
        elif self.action_tokenizer is None and self.lang_subgoals_exist is not None:
            conversation = [
                {"from": "human", "value": f"What action should the uav take to {lang}?"},
                {"from": "gpt", "value": f"<BOD><EOD>{lang_subgoals}"},
            ]
        else:
            gpt_values = ""
            for act in action:
                gpt_values += self.action_tokenizer(act[:len(proprio[0])])
            # cur_robot_state = ""
            # for pro in proprio:
            #     cur_robot_state += self.action_tokenizer(pro)
            ##  The current robot state is {cur_robot_state}. 
            conversation = [
                {"from": "human", "value": f"What action should the uav take to {lang}?"},
                {"from": "gpt", "value": f"<BOD><EOD>{gpt_values}{lang_subgoals}."},
            ]

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        # if self.action_tokenizer is not None:
        #     last_index = len(input_ids) - 1 - input_ids[::-1].index(29871) if 29871 in input_ids else None
        #     if last_index is not None:
        #         input_ids.pop(last_index)

        labels = list(input_ids)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels  = torch.tensor(input_ids), torch.tensor(labels)

        images = {
            "head_slow": None,
            "head_fast": None,
            "right_slow": None,
            "right_fast": None,
            "left_slow": None,
            "left_fast": None,
        }
        key_mapping = {
            "image_head_slow": "head_slow",
            "image_head_fast": "head_fast",
            "image_right_slow": "right_slow",
            "image_right_fast": "right_fast",
            "image_left_slow": "left_slow",
            "image_left_fast": "left_fast",
        }

        for obs_key, prefix in key_mapping.items():
            if obs_key in uav_batch["observation"]:
                images[prefix] = Image.fromarray(uav_batch["observation"][obs_key][0])
        pixel_values = {}
        for prefix, img in images.items():
            if img is not None:
                transformed = self.image_transform(img)
                for key, value in transformed.items():
                    pixel_values[f"{prefix}_{key}"] = value

        point_cloud = None
        if self.load_pointcloud and "pointcloud" in uav_batch["observation"]:
            point_cloud = uav_batch["observation"]["pointcloud"][0]  # (1024, 3)
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        action_mask = None
        action = torch.tensor(action, dtype=torch.float32).reshape(self.action_chunk, -1)
        proprio = torch.tensor(proprio, dtype=torch.float32)
        if "action_mask" in uav_batch:
            action_mask = torch.tensor(uav_batch["action_mask"], dtype=torch.bool)

        if self.action_tokenizer is None and self.lang_subgoals_exist is None:
            labels[:] = IGNORE_INDEX
        else:
            matches = (labels == 32002).nonzero().view(-1)
            last_position = matches[-1].item() if len(matches) > 0 else None
            labels[: last_position+2] = IGNORE_INDEX   ##  last_position is the position of 32002, then +1 is because 29871

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, pointcloud=point_cloud, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=action, action_masks=action_mask, proprio = proprio)


class UAVDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: UAVBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        future_action_window_size: int = 0,
        past_action_window_size: int = 0,
        train: bool = True,
        image_aug: bool = False,
        load_all_data_for_training: bool = True,
        camera_view: Optional[str] = "primary",
        load_pointcloud: bool = False,
        action_chunk : int = 1,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        camera_view = tuple(camera_view.split(","))

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=camera_view,
            load_depth=False,
            load_proprio=False,
            load_language=True,
            load_pointcloud=load_pointcloud,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
            action_chunk = action_chunk,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=past_action_window_size + 1,                                    # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                                        # Skip trajectories without language labels
                #goal_relabeling_strategy="uniform",                                        # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            load_all_data_for_training=load_all_data_for_training,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")



class EpisodicUAVDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, uav_config):
        per_dataset_kwargs = uav_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=uav_config["train"],
            traj_transform_kwargs=uav_config["traj_transform_kwargs"],
            frame_transform_kwargs=uav_config["frame_transform_kwargs"],
            load_all_data_for_training=uav_config["load_all_data_for_training"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for uav_batchatch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], uav_batch))  # noqa: B023
                for i in range(uav_batch["action"].shape[0])
            ]
            yield out


class DummyUAVDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)




def _process_depth_to_pointcloud(self, 
                                   depth_images: List[np.ndarray],
                                   rgb_images: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """将深度图像转换为点云"""
        if not isinstance(depth_images, list):
            depth_images = [depth_images]
        
        if rgb_images is not None and not isinstance(rgb_images, list):
            rgb_images = [rgb_images]
        
        # 使用前置相机深度
        depth = depth_images[0] if len(depth_images) > 0 else np.zeros((512, 512))
        rgb = rgb_images[0] if rgb_images and len(rgb_images) > 0 else None
        
        # 默认相机内参
        fx = 512.0  # 焦距x
        fy = 512.0  # 焦距y
        cx = 512.0  # 主点x
        cy = 512.0  # 主点y
        
        h, w = depth.shape
        
        # 创建坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 过滤有效深度值
        min_depth = 0.1
        max_depth = 100.0
        valid_mask = (depth > min_depth) & (depth < max_depth)
        
        # 转换为3D坐标
        z = depth[valid_mask]
        x = (u[valid_mask] - cx) * z / fx
        y = (v[valid_mask] - cy) * z / fy
        
        # 堆叠坐标
        points = np.stack([x, y, z], axis=1)
        
        # 如果提供了RGB图像，添加颜色
        if rgb is not None:
            colors = rgb[valid_mask] / 255.0
            points = np.concatenate([points, colors], axis=1)
        
        # 对点云进行采样或填充
        if len(points) > 8192:
            indices = np.random.choice(len(points), 8192, replace=False)
            points = points[indices]
        elif len(points) < 1024 and len(points) > 0:
            # 如果点太少，进行填充
            padding = np.zeros((1024 - len(points), points.shape[1]))
            points = np.concatenate([points, padding], axis=0)
        
        return points


    def _process_uav_state(self, uav_state: Dict) -> np.ndarray:
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
        # quat = [orientation[3], orientation[0], orientation[1], orientation[2]]  # [w, x, y, z]
        quat = [orientation[0], orientation[1], orientation[2], orientation[3]] # [x, y, z, w]
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
        
        return np.concatenate([position, euler])