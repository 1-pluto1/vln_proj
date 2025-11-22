#!/usr/bin/env python3
"""
Drone Navigation GRPO Dataset Builder
从SFT数据构建用于GRPO训练的prompt数据集
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from scipy.spatial.transform import Rotation as R

def collect_images(traj_dir: Path) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in traj_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = sorted(files, key=lambda p: p.name)
    return [str(p.resolve()) for p in files]

def to_euler_degrees(ori: Any) -> List[float]:
    if isinstance(ori, (list, tuple)) and len(ori) == 4:
        e = R.from_quat([ori[0], ori[1], ori[2], ori[3]]).as_euler('xyz', degrees=True)
        return e.tolist()
    if isinstance(ori, (list, tuple)) and len(ori) >= 3:
        roll_deg = float(ori[0]) * 180.0 / np.pi
        pitch_deg = float(ori[1]) * 180.0 / np.pi
        yaw_deg = float(ori[2]) * 180.0 / np.pi
        return [roll_deg, pitch_deg, yaw_deg]
    return [0.0, 0.0, 0.0]

def load_uav_flow_folder(input_root: str, max_files: Optional[int] = None) -> List[Dict[str, Any]]:
    root = Path(input_root).resolve()
    entries: List[Dict[str, Any]] = []
    for idx, traj_dir in enumerate(sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name)):
        if max_files is not None and idx >= max_files:
            break
        log_path = traj_dir / "log.json"
        if not log_path.exists():
            continue
        with open(log_path, "r") as f:
            log = json.load(f)
        imgs = collect_images(traj_dir)
        steps = []
        if isinstance(log.get("preprocessed_logs"), list) and log["preprocessed_logs"]:
            steps = log["preprocessed_logs"]
        elif isinstance(log.get("raw_logs"), list) and log["raw_logs"]:
            steps = log["raw_logs"]
        positions: List[List[float]] = []
        orientations: List[List[float]] = []
        for s in steps:
            if isinstance(s, (list, tuple)) and len(s) >= 6:
                x, y, z = float(s[0]), float(s[1]), float(s[2])
                roll = float(s[3])
                yaw_rad = float(s[4]) * np.pi / 180.0
                pitch = float(s[5])
                positions.append([x, y, z])
                orientations.append([roll, pitch, yaw_rad])
            elif isinstance(s, dict):
                pos = s.get("position") or []
                ori = s.get("orientation") or s.get("rpy") or []
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
                else:
                    positions.append([0.0, 0.0, 0.0])
                if isinstance(ori, (list, tuple)) and len(ori) >= 3:
                    orientations.append([float(ori[0]), float(ori[1]), float(ori[2])])
                else:
                    orientations.append([0.0, 0.0, 0.0])
        instruction = log.get("instruction_unified") or log.get("instruction") or ""
        entry = {
            "instruction_unified": instruction,
            "flight_logs": {"positions": positions, "orientations": orientations},
            "images": imgs,
        }
        entries.append(entry)
    return entries


def quat_to_euler(quat: List[float], degrees: bool = True) -> List[float]:
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)
    Args:
        quat: [x, y, z, w] 四元数
        degrees: 是否返回角度制
    Returns:
        [roll, pitch, yaw] 欧拉角
    """
    rotation = R.from_quat(quat)
    euler = rotation.as_euler('xyz', degrees=degrees)
    return euler.tolist()


def extract_target_from_log(log_entry: Dict[str, Any]) -> List[float]:
    """
    从飞行日志提取目标坐标（取最后位置作为目标）
    Args:
        log_entry: SFT数据条目
    Returns:
        [x, y, z] 目标坐标
    """
    positions = log_entry.get('flight_logs', {}).get('positions', [])
    if not positions:
        raise ValueError("日志中没有位置数据")
    
    # 取最后位置作为目标
    target = positions[-1]
    return [float(x) for x in target]


def sample_states_from_log(log_entry: Dict[str, Any], interval: int = 5) -> List[Dict[str, Any]]:
    """
    Sample aligned states and images by interval. Ensures one-to-one matching
    between log frames and images by limiting to the shortest stream length.
    """
    logs = log_entry.get('flight_logs', {})
    positions = logs.get('positions', [])
    orientations = logs.get('orientations', [])
    images = log_entry.get('images', [])
    
    if not (positions and orientations):
        raise ValueError("日志中缺少位置或朝向数据")
    
    states = []
    img_len = len(images) if images else len(positions)
    T = min(len(positions), len(orientations), img_len)
    for i in range(0, T, interval):
        euler_deg = to_euler_degrees(orientations[i])
        state = {
            'position': [float(x) for x in positions[i]],
            'orientation': euler_deg,
            'image_path': (images[i] if (images and i < len(images)) else ""),
            'step': i
        }
        states.append(state)
    
    return states


def build_vlm_prompt(instruction: str, current_state: Dict[str, Any], 
                     target_coords: List[float]) -> str:
    """
    构建VLM输入的prompt
    Args:
        instruction: 自然语言指令
        current_state: 当前状态字典
        target_coords: 目标坐标 [x, y, z]
        battery_level: 电量百分比
    Returns:
        格式化prompt字符串
    """
    euler_angles = current_state['orientation']
    prompt = f"""You are a UAV navigation controller.

[Task Instruction]
{instruction}

[Goal Coordinates]
Goal (meters): {target_coords}

[Current State]
- Position (m): [x={current_state['position'][0]:.2f}, y={current_state['position'][1]:.2f}, z={current_state['position'][2]:.2f}]
- Orientation (deg): [roll={euler_angles[0]:.1f}, pitch={euler_angles[1]:.1f}, yaw={euler_angles[2]:.1f}]


[Output]
Please output the goal coordinates and action in JSON:

```json
{{
  "target": [x, y, z],
  "action": [thrust, pitch, roll, yaw_rate]
}}
```
"""
    
    return prompt


def process_sft_entry(sft_entry: Dict[str, Any], interval: int = 5) -> List[Dict[str, Any]]:
    """
    处理单条SFT数据，生成多个GRPO prompt
    Args:
        sft_entry: SFT数据条目
        interval: 采样间隔
    Returns:
        GRPO prompt列表
    """
    instruction = sft_entry.get('instruction_unified') or sft_entry.get('instruction', '')
    if not instruction:
        raise ValueError("缺少指令字段")
    
    # 提取目标坐标
    target_coords = extract_target_from_log(sft_entry)
    
    states = sample_states_from_log(sft_entry, interval=interval)
    
    grpo_items = []
    for state in states:
        # 构建prompt
        prompt_text = build_vlm_prompt(
            instruction=instruction,
            current_state=state,
            target_coords=target_coords,
        )
        
        grpo_item = {
            "prompt": prompt_text,
            "images": [state['image_path']] if state['image_path'] else [],
            "target_coords": target_coords,  # 用于reward验证
            "state": state
        }
        grpo_items.append(grpo_item)
    
    return grpo_items


def build_grpo_dataset(
    input_path: str,
    output_dir: str,
    interval: int = 5,
    train_ratio: float = 0.9,
    debug: bool = False,
    max_samples: int = None
):
    """
    主函数：构建完整的GRPO数据集
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sft_data: List[Dict[str, Any]] = []
    if input_path.is_dir():
        sft_data = load_uav_flow_folder(str(input_path), max_files=max_samples)
    else:
        with open(input_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    entry = json.loads(line.strip())
                    sft_data.append(entry)
                except json.JSONDecodeError:
                    pass
    
    # 处理每条数据
    all_grpo_items = []
    for i, entry in enumerate(sft_data):
        try:
            grpo_items = process_sft_entry(entry, interval=interval)
            all_grpo_items.extend(grpo_items)
        except Exception as e:
            print(f"处理第 {i+1} 条数据失败: {e}")
            if debug:
                raise
    
    print(f"生成 {len(all_grpo_items)} 条GRPO prompt")
    
    # 划分训练/验证集
    np.random.shuffle(all_grpo_items)
    split_idx = int(len(all_grpo_items) * train_ratio)
    
    train_items = all_grpo_items[:split_idx]
    val_items = all_grpo_items[split_idx:]
    
    train_df = pd.DataFrame([
        {
            "data_source": "uav_flow",
            "prompt": [{"role": "user", "content": item["prompt"]}],
            "ability": "uav_navigation",
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {"goal": item["target_coords"], "state": item["state"], "images": item["images"]},
        }
        for item in train_items
    ])
    train_file = output_dir / "train.parquet"
    train_df.to_parquet(train_file)
    
    val_df = pd.DataFrame([
        {
            "data_source": "uav_flow",
            "prompt": [{"role": "user", "content": item["prompt"]}],
            "ability": "uav_navigation",
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {"goal": item["target_coords"], "state": item["state"], "images": item["images"]},
        }
        for item in val_items
    ])
    val_file = output_dir / "test.parquet"
    val_df.to_parquet(val_file)
    
    # 保存元数据（用于调试）
    meta_file = output_dir / "metadata.jsonl"
    with open(meta_file, 'w') as f:
        for item in all_grpo_items:
            # 保存完整信息供reward函数验证
            meta_item = {
                "prompt": item["prompt"],
                "images": item["images"],
                "target_coords": item["target_coords"],
                "state": item["state"]
            }
            f.write(json.dumps(meta_item) + "\n")
    
    print("Dataset built:")
    print(f"  train: {train_file} ({len(train_items)} samples)")
    print(f"  test: {val_file} ({len(val_items)} samples)")
    print(f"  meta: {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description="从SFT数据构建GRPO训练数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build_grpo_dataset.py --input sft_data.jsonl --output data/grpo
  python build_grpo_dataset.py --input sft_data.jsonl --output data/grpo --interval 3 --max-samples 100
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/uav_flow_data",
        help="输入的SFT数据文件路径（jsonl格式）"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/grpo_data",
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="从飞行日志采样的间隔步数（默认: 5）"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="训练集比例（默认: 0.9）"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="最大处理的SFT样本数（用于快速测试）"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式：遇到错误时抛出异常"
    )
    
    args = parser.parse_args()
    
    build_grpo_dataset(
        input_path=args.input,
        output_dir=args.output,
        interval=args.interval,
        train_ratio=args.train_ratio,
        debug=args.debug,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()