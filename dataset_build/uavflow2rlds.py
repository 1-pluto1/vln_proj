#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import math


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def to_xyz_from_any(pos: Any) -> List[float]:
    if isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
        return [float(pos["x"]), float(pos["y"]), float(pos["z"])]
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    return [0.0, 0.0, 0.0]


def to_rpy_from_any(ori: Any) -> List[float]:
    # 支持 dict: roll/pitch/yaw 或 r/p/y 或 x/y/z；list 长度3；其它返回零
    if isinstance(ori, dict):
        for ks in (("roll", "pitch", "yaw"), ("r", "p", "y"), ("x", "y", "z")):
            if all(k in ori for k in ks):
                return [float(ori[ks[0]]), float(ori[ks[1]]), float(ori[ks[2]])]
        return [0.0, 0.0, 0.0]
    if isinstance(ori, (list, tuple)) and len(ori) >= 3:
        return [float(ori[0]), float(ori[1]), float(ori[2])]
    return [0.0, 0.0, 0.0]


def extract_pose_from_step(step: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    pos = (
        safe_get(step, ["position"]) or
        safe_get(step, ["state", "position"]) or
        safe_get(step, ["pose", "position"]) or
        safe_get(step, ["obs", "position"])
    )
    ori = (
        safe_get(step, ["orientation"]) or
        safe_get(step, ["state", "orientation"]) or
        safe_get(step, ["pose", "orientation"]) or
        safe_get(step, ["obs", "orientation"]) or
        safe_get(step, ["rpy"]) or
        safe_get(step, ["state", "rpy"]) or
        safe_get(step, ["pose", "rpy"])
    )
    return to_xyz_from_any(pos), to_rpy_from_any(ori)


def parse_pose_from_list(vals: List[float]) -> Tuple[List[float], List[float]]:
    # UAV-Flow 列表格式：
    # raw_logs: [x, y, z, roll(rad), yaw(deg), pitch(rad), timestamp]
    # preprocessed_logs: [x, y, z, roll(rad), yaw(deg), pitch(rad)]
    # 目标 rpy 顺序为 [roll(rad), pitch(rad), yaw(rad)]
    if not isinstance(vals, (list, tuple)):
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if len(vals) >= 7:
        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
        roll = float(vals[3])
        yaw_rad = math.radians(float(vals[4]))
        pitch = float(vals[5])
        return [x, y, z], [roll, pitch, yaw_rad]
    if len(vals) >= 6:
        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
        roll = float(vals[3])
        yaw_rad = math.radians(float(vals[4]))
        pitch = float(vals[5])
        return [x, y, z], [roll, pitch, yaw_rad]
    if len(vals) >= 3:
        return [float(vals[0]), float(vals[1]), float(vals[2])], [0.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


def compute_actions(xyz_list: List[List[float]], rpy_list: List[List[float]]) -> List[List[float]]:
    acts = []
    T = len(xyz_list)
    for i in range(T):
        if i + 1 < T:
            dx = [xyz_list[i+1][j] - xyz_list[i][j] for j in range(3)]
            dr = [rpy_list[i+1][j] - rpy_list[i][j] for j in range(3)]
            acts.append(dx + dr)
        else:
            acts.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return acts


def collect_images(traj_dir: Path) -> List[str]:
    # 允许 .jpg/.jpeg/.png，按文件名排序
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in traj_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = sorted(files, key=lambda p: p.name)
    return [str(p.resolve()) for p in files]


def convert_folder_to_rlds(input_root: str, output_root: str, dataset_name: str = "traveluav_dataset") -> None:
    in_root = Path(input_root).resolve()
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    traj_dirs = [d for d in in_root.iterdir() if d.is_dir()]
    n_written = 0

    for traj_dir in sorted(traj_dirs, key=lambda p: p.name):
        log_path = traj_dir / "log.json"
        if not log_path.exists():
            continue

        with open(log_path, "r") as f:
            log = json.load(f)

        img_paths = collect_images(traj_dir)
        if len(img_paths) == 0:
            continue

        # 优先 preprocessed_logs（相对坐标，更适合训练），其次 raw_logs
        steps = []
        if isinstance(log.get("preprocessed_logs"), list) and log["preprocessed_logs"]:
            steps = log["preprocessed_logs"]
        elif isinstance(log.get("raw_logs"), list) and log["raw_logs"]:
            steps = log["raw_logs"]

        base_pose_xyz, base_pose_rpy = [], []
        if steps:
            T = min(len(steps), len(img_paths))
            for i in range(T):
                step_i = steps[i]
                if isinstance(step_i, (list, tuple)):
                    xyz, rpy = parse_pose_from_list(step_i)
                else:
                    xyz, rpy = extract_pose_from_step(step_i)
                base_pose_xyz.append(xyz)
                base_pose_rpy.append(rpy)
            # 如图像更多，补零对齐长度
            for _ in range(len(img_paths) - T):
                base_pose_xyz.append([0.0, 0.0, 0.0])
                base_pose_rpy.append([0.0, 0.0, 0.0])
        else:
            base_pose_xyz = [[0.0, 0.0, 0.0] for _ in img_paths]
            base_pose_rpy = [[0.0, 0.0, 0.0] for _ in img_paths]

        actions = compute_actions(base_pose_xyz, base_pose_rpy)

        instr = log.get("instruction_unified") or log.get("instruction") or ""

        data = {
            "observation": {
                "base_pose_xyz": base_pose_xyz,
                "base_pose_rpy": base_pose_rpy,
                "image_head_slow": img_paths,
                "image_head_fast": img_paths,
                "image_left_slow": img_paths,
                "image_left_fast": img_paths,
                "image_right_slow": img_paths,
                "image_right_fast": img_paths
            },
            "action": actions,
            "task": {"language_instruction": instr},
            "dataset_name": dataset_name
        }

        out_traj_dir = out_root / f"trajectory_{traj_dir.name}"
        out_traj_dir.mkdir(parents=True, exist_ok=True)
        with open(out_traj_dir / "data.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        n_written += 1

    dataset_info = {
        "name": dataset_name,
        "description": "UAV-Flow dataset converted to RLDS-like JSON format",
        "version": "1.0.0",
        "splits": {"train": {"num_trajectories": n_written}}
    }
    with open(out_root / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {n_written} trajectories to {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert UAV-Flow folder dataset (images + log.json) to RLDS-like JSON")
    parser.add_argument("--input_dir", type=str, default="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/uav_flow_data", help="Path to folder dataset root (contains many trajectory dirs)")
    parser.add_argument("--output_dir", type=str, default="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/rlds_data", help="Path to output directory")
    parser.add_argument("--dataset_name", type=str, default="uavflow_dataset")
    args = parser.parse_args()

    convert_folder_to_rlds(args.input_dir, args.output_dir, dataset_name=args.dataset_name)