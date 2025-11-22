# eval.py (已添加日志记录)

import os
from pathlib import Path
import sys
import time
import json
import shutil
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from utils.logger import logger
from utils.utils import *

# from src.model_wrapper.fis_model import FiSModelWrapper
from src.model_wrapper.travel_llm import TravelModelWrapper
from src.model_wrapper.fis_ipc import FiSIPCModelWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized


# +++ START: 添加日志辅助函数 +++


def _log_any(name, obj, step):
    try:
        if obj is None:
            logger.info(f"  {name}: None")
            return
        t = type(obj)
        logger.info(f"  {name} Type: {t}")
        try:
            l = len(obj)
            logger.info(f"  {name} Length: {l}")
        except Exception:
            pass
        if isinstance(obj, np.ndarray):
            logger.info(f"  {name} Shape: {obj.shape} Dtype: {obj.dtype} Bytes: {obj.nbytes}")
            try:
                logger.info(f"  {name} Stats: min={float(np.min(obj)):.4f} max={float(np.max(obj)):.4f} mean={float(np.mean(obj)):.4f}")
            except Exception:
                pass
            try:
                s = np.array2string(obj, threshold=obj.size, max_line_width=1000000)
                logger.info(f"  {name} Full: {s}")
            except Exception:
                logger.info(f"  {name} Full: {obj.tolist()}")
            return
        if torch.is_tensor(obj):
            bs = int(obj.element_size()) * int(obj.numel())
            logger.info(f"  {name} Tensor Shape: {tuple(obj.shape)} Dtype: {obj.dtype} Device: {obj.device} Bytes: {bs}")
            with torch.no_grad():
                try:
                    logger.info(f"  {name} Stats: min={float(obj.min()):.4f} max={float(obj.max()):.4f} mean={float(obj.float().mean()):.4f}")
                except Exception:
                    pass
                try:
                    arr = obj.detach().cpu().numpy()
                    s = np.array2string(arr, threshold=arr.size, max_line_width=1000000)
                    logger.info(f"  {name} Full: {s}")
                except Exception:
                    logger.info(f"  {name} Full: {obj.tolist()}")
            return
        if hasattr(obj, 'keys') and callable(getattr(obj, 'keys', None)):
            keys = []
            try:
                keys = list(obj.keys())
            except Exception:
                pass
            logger.info(f"  {name} Keys: {keys}")
            try:
                j = json.dumps(obj, ensure_ascii=False, default=lambda o: str(o))
                logger.info(f"  {name} Full: {j}")
            except Exception:
                logger.info(f"  {name} Full: {str(obj)}")
            return
        if isinstance(obj, (list, tuple, set)):
            seq = list(obj)
            try:
                nums = [x for x in seq if isinstance(x, (int, float))]
                if nums:
                    logger.info(f"  {name} Numeric Stats: min={min(nums):.4f} max={max(nums):.4f} mean={float(np.mean(nums)):.4f}")
            except Exception:
                pass
            logger.info(f"  {name} Full: {str(seq)}")
            return
        if isinstance(obj, (str, bytes)):
            try:
                s = obj if isinstance(obj, str) else obj.decode(errors='ignore')
                logger.info(f"  {name} String Length: {len(s)}")
                logger.info(f"  {name} Full: {s}")
            except Exception:
                pass
            return
        if hasattr(obj, '__dict__'):
            try:
                attrs = list(obj.__dict__.keys())
                logger.info(f"  {name} Attrs: {attrs}")
            except Exception:
                pass
        try:
            logger.info(f"  {name} Full: {str(obj)}")
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[{step}] {name} summarize error: {e}")


def log_input_info(inputs, step, assist_notices=None, rot_to_targets=None):
    try:
        logger.info(f"--- [Step {step} MODEL INPUT] ---")
        _log_any("inputs", inputs, step)
        batch_ctx = None
        try:
            if hasattr(inputs, 'get') and callable(getattr(inputs, 'get', None)):
                batch_ctx = inputs.get('batch_ctx')
            elif hasattr(inputs, 'batch_ctx'):
                batch_ctx = inputs.batch_ctx
        except Exception:
            pass
        if batch_ctx is not None:
            _log_any("inputs.batch_ctx", batch_ctx, step)
            try:
                if isinstance(batch_ctx, (list, tuple)) and len(batch_ctx) > 0:
                    _log_any("inputs.batch_ctx[0]", batch_ctx[0], step)
                elif hasattr(batch_ctx, '__iter__'):
                    logger.info(f"  inputs.batch_ctx Iterator: head unknown")
            except Exception:
                pass
        if assist_notices is not None:
            _log_any("assist_notices", assist_notices, step)
        if rot_to_targets is not None:
            _log_any("rot_to_targets", rot_to_targets, step)
        logger.info(f"-------------------------------------------")
    except Exception as e:
        logger.warning(f"[Step {step}] model input log error: {e}")
        try:
            logger.info(f"[Step {step}] inputs raw: {str(inputs)}")
        except Exception:
            pass

def log_prepare_params(episodes, target_positions, assist_notices, step):
    try:
        logger.info(f"--- [Step {step} PREPARE_INPUTS params] ---")
        _log_any("episodes", episodes, step)
        try:
            if isinstance(episodes, (list, tuple)) and len(episodes) > 0:
                ep0 = episodes[0]
                _log_any("episodes[0]", ep0, step)
                for attr in ['episode_id', 'scene_id', 'instruction', 'start_position', 'start_rotation', 'goal_position']:
                    if hasattr(ep0, attr):
                        _log_any(f"episodes[0].{attr}", getattr(ep0, attr), step)
        except Exception:
            pass
        _log_any("target_positions", target_positions, step)
        try:
            if isinstance(target_positions, (list, tuple)) and len(target_positions) > 0:
                _log_any("target_positions[0]", target_positions[0], step)
            elif isinstance(target_positions, np.ndarray) and target_positions.size > 0:
                head = target_positions.reshape(-1)[:min(5, target_positions.size)].tolist()
                _log_any("target_positions.head", head, step)
            elif torch.is_tensor(target_positions) and int(target_positions.numel()) > 0:
                head = target_positions.view(-1)[:min(5, int(target_positions.numel()))].tolist()
                _log_any("target_positions.head", head, step)
        except Exception:
            pass
        if assist_notices is not None:
            _log_any("assist_notices", assist_notices, step)
            try:
                if isinstance(assist_notices, (list, tuple)) and len(assist_notices) > 0:
                    _log_any("assist_notices[0]", assist_notices[0], step)
            except Exception:
                pass
        logger.info(f"--------------------------------------------------")
    except Exception as e:
        logger.warning(f"[Step {step}] prepare params log error: {e}")


def log_waypoint_info(waypoints_batch, step):
    try:
        logger.info(f"--- [Step {step} MODEL OUTPUT (refined_waypoints)] ---")
        _log_any("refined_waypoints", waypoints_batch, step)
        try:
            if isinstance(waypoints_batch, (list, tuple)) and len(waypoints_batch) > 0:
                first_path = waypoints_batch[0]
                _log_any("refined_waypoints[0]", first_path, step)
                if isinstance(first_path, (list, tuple)) and len(first_path) > 0:
                    first_waypoint = first_path[0]
                    _log_any("refined_waypoints[0][0]", first_waypoint, step)
                    coords = []
                    for wp in first_path:
                        if isinstance(wp, (list, tuple)):
                            coords.extend([c for c in wp if isinstance(c, (int, float))])
                        elif isinstance(wp, np.ndarray):
                            coords.extend([float(x) for x in wp.reshape(-1)[:50]])
                        elif torch.is_tensor(wp):
                            with torch.no_grad():
                                coords.extend([float(x) for x in wp.view(-1)[:50]])
                    if coords:
                        logger.info(f"  Numeric Stats (Agent 0): min={min(coords):.4f} max={max(coords):.4f} mean={float(np.mean(coords)):.4f}")
        except Exception:
            pass
        logger.info(f"--------------------------------------------------")
    except Exception as e:
        logger.warning(f"[Step {step}] run output log error: {e}")
        try:
            logger.info(f"[Step {step}] refined_waypoints raw: {str(waypoints_batch)}")
        except Exception:
            pass
            
# +++ END: 添加日志辅助函数 +++


def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir):
    model_wrapper.eval()
    
    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)

            pbar.update(n=eval_env.batch_size)
            
            log_prepare_params(batch_state.episodes, batch_state.target_positions, assist_notices=None, step=0)
            inputs, rot_to_targets = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions)
            log_input_info(inputs, step=0, assist_notices=None, rot_to_targets=rot_to_targets)

            for t in range(int(args.maxWaypoints) + 1):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
                
                refined_waypoints = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)
                
                log_waypoint_info(refined_waypoints, step=t)

                eval_env.makeActions(refined_waypoints)
                outputs = eval_env.get_obs()
                batch_state.update_from_env_output(outputs)
                
                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                
                batch_state.update_metric()
                
                assist_notices = batch_state.get_assist_notices()
                
                log_prepare_params(batch_state.episodes, batch_state.target_positions, assist_notices, step=t+1)
                inputs, rot_to_targets2 = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions, assist_notices)
                log_input_info(inputs, step=t+1, assist_notices=assist_notices, rot_to_targets=rot_to_targets2)

        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path
    
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    
    setup()

    assert CheckPort(), 'error port'

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False
    
    if args.name.lower().startswith("fis"):
        model_wrapper = FiSIPCModelWrapper(model_args=model_args, data_args=data_args)
    else:
        model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)
    
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)
    
    eval(model_wrapper=model_wrapper,
         assist=assist,
         eval_env=eval_env,
         eval_save_dir=eval_save_path)
    
    eval_env.delete_VectorEnvUtil()