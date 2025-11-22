import re
import json
import math
from typing import Any, List, Tuple, Optional
import logging

import numpy as np
import torch

from grpo_env import GRPO_UAVEnv

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

_PAT_GOAL_EN = re.compile(r"Goal\s*\(meters\)\s*:\s*\[\s*([\-\d\.eE]+)\s*,\s*([\-\d\.eE]+)\s*,\s*([\-\d\.eE]+)\s*\]", re.I)
_PAT_POS_EN = re.compile(r"Position\s*\(m\)\s*:\s*\[\s*x=([\-\d\.eE]+)\s*,\s*y=([\-\d\.eE]+)\s*,\s*z=([\-\d\.eE]+)\s*\]", re.I)
_PAT_RPY_EN = re.compile(r"Orientation\s*\(deg\)\s*:\s*\[\s*roll=([\-\d\.eE]+)\s*,\s*pitch=([\-\d\.eE]+)\s*,\s*yaw=([\-\d\.eE]+)\s*\]", re.I)
_PAT_JSON_BLOCK = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.I)

def _to_vec3(vals: List[str]) -> np.ndarray:
    return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)


def _extract_prompt(prompt: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.debug("Extracting prompt")
    m = _PAT_GOAL_EN.search(prompt)
    if not m:
        logger.error("Goal not found in prompt")
        raise ValueError("missing True target")
    goal = _to_vec3([m.group(1), m.group(2), m.group(3)])
    logger.debug(f"Parsed goal: {goal}")
    m = _PAT_POS_EN.search(prompt)
    if not m:
        logger.error("Position not found in prompt")
        raise ValueError("missing position")
    pos = _to_vec3([m.group(1), m.group(2), m.group(3)])
    logger.debug(f"Parsed position: {pos}")
    m = _PAT_RPY_EN.search(prompt)
    if not m:
        logger.error("Orientation not found in prompt")
        raise ValueError("missing orientation")
    rpy = _to_vec3([m.group(1), m.group(2), m.group(3)])
    logger.debug(f"Parsed orientation: {rpy}")

    return goal, pos, rpy


def _extract_response(resp: str) -> Tuple[np.ndarray, np.ndarray, float]:
    logger.debug("Extracting response JSON block")
    m = _PAT_JSON_BLOCK.search(resp)
    if not m:
        logger.warning("JSON block not found in response")
        return np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32), 0.0
    try:
        obj = json.loads(m.group(1))
    except Exception as e:
        logger.error(f"Failed to parse response JSON: {e}")
        return np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32), 0.0
    tgt = obj.get("target") if isinstance(obj, dict) else None
    act = obj.get("action") if isinstance(obj, dict) else None
    tgt_vec = np.array(tgt[:3], dtype=np.float32) if isinstance(tgt, list) and len(tgt) >= 3 else np.zeros(3, dtype=np.float32)
    act_vec = np.array(act[:4], dtype=np.float32) if isinstance(act, list) and len(act) >= 4 else np.zeros(4, dtype=np.float32)
    overflow = float(np.sum(np.maximum(np.abs(act_vec) - 1.0, 0.0)))
    logger.debug(f"Parsed target: {tgt_vec}, action: {act_vec}, overflow: {overflow}")
    return tgt_vec, act_vec, overflow


def _env_step(env: Any, pos: np.ndarray, rpy: np.ndarray, act: np.ndarray) -> Tuple[float, bool, float, float]:
    base_reward = 0.0
    collided = False
    dist = 0.0
    energy = 0.0
    logger.info(f"[ENV_STEP] Starting with action: {act}")
    logger.debug(f"[ENV_STEP] Env type: {type(env).__name__}")
    try:
        if hasattr(env, "set_state"):
            env.set_state({"position": pos.tolist(), "orientation": rpy.tolist()})
            logger.debug(f"[ENV_STEP] State set: pos={pos}, rpy={rpy}")
        else:
            logger.debug("[ENV_STEP] Env has no set_state")
    except Exception as e:
        logger.warning(f"[ENV_STEP] set_state failed: {e}")
    try:
        if hasattr(env, "step"):
            action_to_apply = act[:3] if act.shape[0] >= 3 else act
            result = env.step(action_to_apply)
            info = {}
            if isinstance(result, tuple):
                if len(result) >= 5:
                    _, r, terminated, truncated, info = result
                    base_reward = float(r)
                    logger.debug(f"[ENV_STEP] terminated={terminated}, truncated={truncated}")
                elif len(result) == 4:
                    _, r, done, info = result
                    base_reward = float(r)
                    logger.debug(f"[ENV_STEP] done={done}")
                elif len(result) == 2:
                    r, info = result
                    base_reward = float(r)
                else:
                    logger.debug(f"[ENV_STEP] Unexpected step tuple len={len(result)}")
            elif isinstance(result, dict):
                base_reward = float(result.get("reward", 0.0))
                info = result
            else:
                logger.debug(f"[ENV_STEP] Unexpected step return type={type(result).__name__}")
            collided = bool(info.get("collision", False))
            dist = float(info.get("distance_to_target", 0.0))
            energy_val = info.get("energy", None)
            energy = float(energy_val) if isinstance(energy_val, (int, float)) else 0.0
            logger.info(f"[ENV_STEP] Result: reward={base_reward}, collided={collided}, dist={dist}, energy={energy}")
        else:
            logger.error("[ENV_STEP] Env has no step method")
    except Exception as e:
        logger.exception(f"[ENV_STEP] step failed: {e}")
    logger.debug(f"[ENV_STEP] Returning: ({base_reward}, {collided}, {dist}, {energy})")
    return base_reward, collided, dist, energy

def compute_score(data_source: Any, solution_str: str, ground_truth: Any, extra_info: Optional[Any] = None, env: Optional[Any] = None) -> float:
    logger.info("Starting compute_score")
    if env is None: 
        logger.info("Creating default GRPO_UAVEnv")
        env = GRPO_UAVEnv(
            dataset_path="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/data/TravelUAV_unzip/",
            save_path="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/traveluav_data/dagger_data",
            eval_json_path="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/data/TravelUAV_data_json/data/uav_dataset/trainset.json",
            seed=42,
            success_reward=10.0,
            collision_penalty=1.0,
            step_penalty=0.01,
            activate_maps=["NYCEnvironmentMegapa"]
        )

    try:
        src_text = None
        if isinstance(extra_info, dict):
            src_text = extra_info.get("prompt", None)
        if src_text is None and isinstance(data_source, str):
            src_text = data_source
        if src_text is None:
            src_text = solution_str
        goal, pos, rpy = _extract_prompt(str(src_text))
        logger.debug(f"Parsed prompt: goal={goal}, pos={pos}, rpy={rpy}")
    except Exception as e:
        logger.exception(f"Failed to extract prompt: {e}")
        return -10.0
    pred_goal, act, overflow = _extract_response(solution_str)
    logger.debug(f"Parsed response: pred_goal={pred_goal}, act={act}, overflow={overflow}")
    base_reward, collided, dist, energy = _env_step(env, pos, rpy, act)
    env_reward = -dist
    if collided:
        env_reward -= 10.0
    if dist < 0.5:
        env_reward += 10.0
    env_reward -= energy * 0.01
    goal_penalty = -float(np.linalg.norm(pred_goal - goal)) * 0.1
    action_penalty = -overflow if overflow > 0 else 0.0
    total = float(base_reward + env_reward + goal_penalty + action_penalty)
    logger.info(f"Score components: base_reward={base_reward}, env_reward={env_reward}, goal_penalty={goal_penalty}, action_penalty={action_penalty}, total={total}")
    if math.isnan(total):
        return -5.0
    return total

def _test_compute_score():
    """测试compute_score函数的单个用例"""
    prompt = (
        "You are a UAV navigation controller.\n\n[Goal Coordinates]\nGoal (meters): [1.0, 2.0, 3.0]\n\n"
        "[Current State]\n- Position (m): [x=1.0, y=2.0, z=3.0]\n- Orientation (deg): [roll=0.0, pitch=0.0, yaw=0.0]\n"
        "- Battery: 100%"
    )
    response = """```json
        {
            "target": [1.0, 2.0, 3.0],
            "action": [0.1, 0.0, -0.1, 0.0]
        }
        ```"""
    score = compute_score(prompt, response, None, {"prompt": prompt})
    logger.debug(f"Prompt: {prompt}")
    logger.debug(f"Response: {response}")
    logger.info(f"Computed score: {score}")
    assert isinstance(score, float), f"Score should be float, got {type(score)}"
    assert not math.isnan(score), "Score should not be NaN"
    assert not math.isinf(score), "Score should not be infinite"
    logger.info("✓ compute_score function test passed!")
    return score

if __name__ == "__main__":
    score = _test_compute_score()
    logger.info(f"Final score: {score}")