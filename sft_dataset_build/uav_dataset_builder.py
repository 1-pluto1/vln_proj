#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow_datasets as tfds
import tensorflow as tf
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Iterator, Tuple
import os

# 1. 将类名修改为 UavDataset，以匹配你的训练脚本正在寻找的 "uav_dataset"
class UavDataset(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for UAV-Flow dataset, formatted for RLDS."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release, formatted as RLDS (steps).',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Define dataset meta information and features in RLDS format."""
        
        # 2. 将特征定义改为 RLDS 兼容的 "steps" 结构
        # 这是一个 "Array of Structures" (AoS)
        features = tfds.features.FeaturesDict({
            'steps': tfds.features.Sequence(
                tfds.features.FeaturesDict({
                    'observation': tfds.features.FeaturesDict({
                        'base_pose_xyz': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                        'base_pose_rpy': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                        # 与源数据的 .png 文件一致，避免误重编码
                        'image_head_slow': tfds.features.Image(encoding_format='png'),
                        'image_head_fast': tfds.features.Image(encoding_format='png'),
                        'image_left_slow': tfds.features.Image(encoding_format='png'),
                        'image_left_fast': tfds.features.Image(encoding_format='png'),
                        'image_right_slow': tfds.features.Image(encoding_format='png'),
                        'image_right_fast': tfds.features.Image(encoding_format='png'),
                    }),
                    'action': tfds.features.Tensor(shape=(6,), dtype=tf.float32),
                    'language_instruction': tfds.features.Text(),
                    'is_first': tf.bool,
                    'is_last': tf.bool,
                    'is_terminal': tf.bool,
                })
            ),
            'dataset_name': tfds.features.Text(),
        })

        return tfds.core.DatasetInfo(
            builder=self,
            description="UAV-Flow dataset formatted as RLDS steps.",
            features=features,
        )


    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Specify dataset splits in the old (list) format."""
        
        rlds_data_path = os.environ.get(
            "RLDS_SOURCE_DIR",
            "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/test_data"
        )
        if not os.path.exists(rlds_data_path):
            raise ValueError(f"RLDS path not found: {rlds_data_path}. Run uavflow2rlds.py first.")
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"path": rlds_data_path},
            ),
        ]


    def _generate_examples(self, path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Generate examples from RLDS-like JSON produced by uavflow2rlds.py.
        """
        
        rlds_root = Path(path)
        trajectory_dirs = [d for d in rlds_root.iterdir() if d.is_dir() and d.name.startswith("trajectory_")]
        
        for traj_dir in sorted(trajectory_dirs, key=lambda p: p.name):
            data_path = traj_dir / "data.json"
            if not data_path.exists():
                continue

            with open(data_path, "r") as f:
                traj_data = json.load(f)

            raw_obs = traj_data["observation"]
            raw_actions = traj_data["action"]
            lang_instruction = traj_data["task"]["language_instruction"]
            
            steps = []
            num_steps = len(raw_actions)
            
            if num_steps == 0:
                continue

            img_keys = ["image_head_slow", "image_head_fast", "image_left_slow", 
                        "image_left_fast", "image_right_slow", "image_right_fast"]
            
            # 检查图像文件是否存在
            image_paths = {}
            valid_episode = True
            for key in img_keys:
                image_paths[key] = []
                for img_path in raw_obs[key]:
                    if not os.path.exists(img_path):
                        print(f"Warning: Skipping trajectory {traj_dir.name}. Missing image: {img_path}")
                        valid_episode = False
                        break
                    image_paths[key].append(img_path)
                if not valid_episode:
                    break
            
            if not valid_episode:
                continue # 跳过这个损坏的 trajectory
                
            # 压缩所有数据流
            zipped_data = zip(
                raw_obs["base_pose_xyz"],
                raw_obs["base_pose_rpy"],
                image_paths["image_head_slow"],
                image_paths["image_head_fast"],
                image_paths["image_left_slow"],
                image_paths["image_left_fast"],
                image_paths["image_right_slow"],
                image_paths["image_right_fast"],
                raw_actions
            )

            for i, (xyz, rpy, im_h_s, im_h_f, im_l_s, im_l_f, im_r_s, im_r_f, action) in enumerate(zipped_data):
                step_data = {
                    'observation': {
                        'base_pose_xyz': np.array(xyz, dtype=np.float32),
                        'base_pose_rpy': np.array(rpy, dtype=np.float32),
                        'image_head_slow': im_h_s,
                        'image_head_fast': im_h_f,
                        'image_left_slow': im_l_s,
                        'image_left_fast': im_l_f,
                        'image_right_slow': im_r_s,
                        'image_right_fast': im_r_f,
                    },
                    'action': np.array(action, dtype=np.float32),
                    'language_instruction': lang_instruction, # 语言指令在每一步都相同
                    'is_first': (i == 0),
                    'is_last': (i == num_steps - 1),
                    'is_terminal': (i == num_steps - 1), # 假设最后一步就是终止步
                }
                steps.append(step_data)
            
            if not steps:
                continue

            # 最终的 example 是一个包含 'steps' 列表的字典
            example_data = {
                'steps': steps,
                'dataset_name': traj_data["dataset_name"]
            }
            
            trajectory_id = traj_dir.name
            yield trajectory_id, example_data
