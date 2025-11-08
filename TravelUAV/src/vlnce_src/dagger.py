# 顶部模块加载（module top-level）
import os
import sys
from pathlib import Path
import traceback
import tqdm
import json
import numpy as np
import torch

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from utils.logger import logger

logger.info(f"Module {__name__} loaded; cwd={os.getcwd()}")
logger.info(f"Appended to sys.path: {str(Path(str(os.getcwd())).resolve())}")

from src.model_wrapper.fis_model import FiSModelWrapper
from src.common.param import args, model_args, data_args
from src.vlnce_src.env_uav import AirVLNENV
from src.vlnce_src.assist import Assist
from src.vlnce_src.closeloop_util import DaggerBatchState, setup, CheckPort, initialize_env, is_dist_avail_and_initialized

logger.info("Imports loaded: FiSModelWrapper, AirVLNENV, Assist, Dagger utilities.")
import logging


def collect_data(model_wrapper, assist: Assist, train_env: AirVLNENV, data_it=0):
    logger.info(f"collect_data(entry): data_it={data_it}, collect_type={args.collect_type}, dagger_p={args.dagger_p}")
    assert args.collect_type in ['dagger']
    beta = float(args.dagger_p)
    logger.debug(f"collect_data(config): beta={beta}, env_batch_size={train_env.batch_size}")

    model_wrapper.eval()
    logger.debug("collect_data: model set to eval()")

    logger.debug(f"collect_data(config): maxWaypoints={getattr(args, 'maxWaypoints', None)}, batchSize={getattr(args, 'batchSize', None)}, activate_maps={getattr(args, 'activate_maps', None)}")
    logger.debug(f"collect_data(actors): model_wrapper={type(model_wrapper).__name__}, assist={type(assist).__name__}, env={type(train_env).__name__}, env_batch_size={getattr(train_env, 'batch_size', None)}")
    assert args.collect_type in ['dagger']
    beta = float(args.dagger_p)
    logger.debug(f"collect_data(config): beta={beta}, env_batch_size={train_env.batch_size}")

    model_wrapper.eval()
    logger.debug("collect_data: model set to eval()")

    with torch.no_grad():
        logger.debug("collect_data: entered torch.no_grad() block")

        # Helpers to dump variables comprehensively but safely
        def _safe_len(x):
            try:
                return len(x)
            except Exception:
                return None
        def _safe_type(x):
            try:
                return type(x).__name__
            except Exception:
                return None
        def _head(x, n=3):
            try:
                return x[:n]
            except Exception:
                return None
        def _tensor_stats(x):
            try:
                import numpy as _np
                if hasattr(x, 'detach') and hasattr(x, 'cpu'):
                    _x = x.detach().cpu()
                else:
                    _x = x
                if hasattr(_x, 'numpy'):
                    _x = _x.numpy()
                if hasattr(_x, 'shape'):
                    stats = {'shape': tuple(_x.shape)}
                    try:
                        stats.update({'min': float(_np.min(_x)), 'max': float(_np.max(_x)), 'mean': float(_np.mean(_x))})
                    except Exception:
                        pass
                    return stats
            except Exception:
                return {'type': _safe_type(x)}
            return {'type': _safe_type(x)}
        def _dump_var(name, val, full=False, max_seq=20, max_str=1000):
            try:
                if val is None:
                    logger.debug(f"{name}=None")
                    return
                if full:
                    # 始终输出完整内容（含大数组/张量）
                    try:
                        v = val
                        if hasattr(v, 'detach') and hasattr(v, 'cpu'):
                            v = v.detach().cpu()
                        import numpy as _np
                        if hasattr(v, 'numpy'):
                            v_np = v.numpy()
                            s = _np.array2string(v_np, threshold=_np.inf)
                            logger.debug(f"{name}(full)={s}")
                        else:
                            logger.debug(f"{name}(full)={v}")
                    except Exception:
                        logger.debug(f"{name}(full)={val}")
                    return
                t = _safe_type(val)
                if hasattr(val, 'shape') or _safe_type(val) in ('Tensor', 'ndarray'):
                    logger.debug(f"{name}({_safe_type(val)}): { _tensor_stats(val) }")
                    # small tensors: print head
                    try:
                        if hasattr(val, 'numel') and val.numel() <= 50:
                            logger.debug(f"{name}(content)={val}")
                    except Exception:
                        pass
                elif isinstance(val, dict):
                    keys = list(val.keys())
                    sample = {k: _safe_type(val[k]) for k in keys[:min(len(keys), 10)]}
                    logger.debug(f"{name}(dict): len={len(val)}, sample_keys_types={sample}")
                    if len(val) <= 10:
                        logger.debug(f"{name}(content)={val}")
                elif isinstance(val, (list, tuple)):
                    h = _head(val, n=10)
                    logger.debug(f"{name}({_safe_type(val)}): len={len(val)}, head={h}")
                    if len(val) <= max_seq:
                        logger.debug(f"{name}(content)={val}")
                elif isinstance(val, (str, bytes)):
                    s = val if isinstance(val, str) else str(val)
                    s_out = s if len(s) <= max_str else (s[:max_str] + "...(truncated)")
                    logger.debug(f"{name}({_safe_type(val)}): {s_out}")
                else:
                    r = repr(val)
                    r_out = r if len(r) <= max_str else (r[:max_str] + "...(truncated)")
                    logger.debug(f"{name}({_safe_type(val)}): {r_out}")
            except Exception as _e:
                logger.debug(f"_dump_var({name}) failed: {repr(_e)}")

        start_iter = 0
        end_iter = len(train_env.data)
        logger.info(f"collect_data(dataset): total_items={end_iter}")
        pbar = tqdm.tqdm(total=end_iter)
        while start_iter < end_iter:
            env_batchs = train_env.next_minibatch(skip_scenes=[])
            if env_batchs is None:
                logger.warning('collect_data: train_env.batch is None, stop collecting')
                break
            _dump_var("env_batchs", env_batchs)
            _dump_var("train_env.batch_size", train_env.batch_size)
            _dump_var("train_env.index_data", getattr(train_env, 'index_data', None))

            start_iter += train_env.batch_size
            pbar.update(n=train_env.batch_size)
            
            dagger_batch_state_manager = DaggerBatchState(train_env.batch_size, env_batchs, train_env)
            logger.debug(f"collect_data(state): DaggerBatchState created with bs={train_env.batch_size}")
            _dump_var("state.episodes", getattr(dagger_batch_state_manager, 'episodes', None))
            _dump_var("state.target_positions(init)", getattr(dagger_batch_state_manager, 'target_positions', None))

            outputs = train_env.reset()
            _dump_var("env.reset.outputs", outputs)
            logger.debug(f"collect_data(env.reset): index_data={getattr(train_env, 'index_data', None)}")

            dagger_batch_state_manager.update_from_env_output(outputs)
            _dump_var("state.target_positions(after_reset)", getattr(dagger_batch_state_manager, 'target_positions', None), full=True)
            _dump_var("state.object_infos(after_reset)", getattr(dagger_batch_state_manager, 'object_infos', None), full=True)
            _dump_var("state.trajs(after_reset)", getattr(dagger_batch_state_manager, 'trajs', None), full=True)
            _dump_var("state.episodes(after_reset)", getattr(dagger_batch_state_manager, 'episodes', None), full=True)

            inputs, rot_to_targets = model_wrapper.prepare_inputs(dagger_batch_state_manager.episodes, dagger_batch_state_manager.target_positions)
            _dump_var("inputs", inputs)
            _dump_var("rot_to_targets", rot_to_targets)
        
            # closeloop steps
            for t in range(int(args.maxWaypoints) + 1):
                logger.info('dagger_it: {} \t {} - {} / {}'.format(data_it, int(train_env.index_data)-int(train_env.batch_size), t, end_iter))
                try:
                    is_terminate = dagger_batch_state_manager.check_dagger_batch_termination(dagger_it=data_it)
                    _dump_var("is_terminate", is_terminate)
                    _dump_var("state.need_teacher", getattr(dagger_batch_state_manager, 'need_teacher', None))
                    if is_terminate:
                        logger.info("collect_data: termination condition met, break loop")
                        break
                    
                    refined_waypoints = model_wrapper.run(inputs=inputs, episodes=dagger_batch_state_manager.episodes, rot_to_targets=rot_to_targets)
                    _dump_var("refined_waypoints", refined_waypoints)

                    choose_teacher = torch.rand(args.batchSize) < float(args.dagger_p)
                    choose_teacher_list = choose_teacher.tolist() if hasattr(choose_teacher, 'tolist') else choose_teacher
                    _dump_var("choose_teacher", choose_teacher_list)
                    _dump_var("dagger_p(beta)", float(args.dagger_p))
                    for i in range(len(choose_teacher)):
                        if choose_teacher[i] or dagger_batch_state_manager.need_teacher[i]:
                            _dump_var(f"teacher_action[{i}]", dagger_batch_state_manager.episodes[i][-1]['teacher_action'])
                            refined_waypoints[i] = dagger_batch_state_manager.episodes[i][-1]['teacher_action']

                    train_env.makeActions(refined_waypoints)
                    _dump_var("actions_applied.last", refined_waypoints[-1] if _safe_len(refined_waypoints) else None)

                    outputs = train_env.get_obs()
                    _dump_var("env.get_obs.outputs", outputs)

                    dagger_batch_state_manager.update_from_env_output(outputs, assist.check_collision_by_depth)
                    _dump_var("state.trajs(after_obs)", getattr(dagger_batch_state_manager, 'trajs', None))
                    _dump_var("state.target_positions(after_obs)", getattr(dagger_batch_state_manager, 'target_positions', None))
                    _dump_var("state.object_infos(after_obs)", getattr(dagger_batch_state_manager, 'object_infos', None))

                    dagger_batch_state_manager.dagger_step_back()
                    _dump_var("state.episodes.lengths(after_step_back)", [ _safe_len(ep) for ep in getattr(dagger_batch_state_manager, 'episodes', []) ])

                    assist_notices = assist.get_assist_notice(
                        episodes=dagger_batch_state_manager.episodes,
                        trajs=dagger_batch_state_manager.trajs,
                        object_infos=dagger_batch_state_manager.object_infos,
                        target_positions=dagger_batch_state_manager.target_positions
                    )
                    _dump_var("assist_notices", assist_notices)

                    inputs, _ = model_wrapper.prepare_inputs(
                        dagger_batch_state_manager.episodes,
                        dagger_batch_state_manager.target_positions,
                        assist_notices
                    )
                    _dump_var("inputs(after_assist)", inputs)
                except Exception as e:
                    exe_type, exe_value, exe_traceback = sys.exc_info()
                    exe_info_list = traceback.format_exception(exe_type, exe_value, exe_traceback)
                    tracebacks = ''.join(exe_info_list)
                    logger.error("collect_data(exception): data_it={}, t={}, index_data={}\n{}".format(data_it, t, getattr(train_env, 'index_data', None), tracebacks))
                    print('traceback:', tracebacks)
                    print(e)
                    break
        pbar.close()
        logger.info("collect_data: progress bar closed")
    
    logger.info('END data_it: {}'.format(data_it))


if __name__ == "__main__":
    setup()
    logger.setLevel(logging.DEBUG)
    logger.info("Setup completed.")

    assert CheckPort(), 'error port'
    logger.info("Port check passed.")

    activate_maps = args.activate_maps
    dataset_path = args.dataset_path
    train_json_path = args.train_json_path
    dagger_save_path = args.dagger_save_path

    logger.info(f"Args snapshot: batchSize={args.batchSize}, dagger_it={args.dagger_it}, maxWaypoints={args.maxWaypoints}")
    logger.info(f"Paths: dataset_path={dataset_path}, train_json_path={train_json_path}, dagger_save_path={dagger_save_path}")
    logger.info(f"activate_maps={activate_maps}")

    if not os.path.isdir(dagger_save_path):
        os.makedirs(dagger_save_path, exist_ok=True)
        logger.info(f"Created dagger_save_path: {dagger_save_path}")
    else:
        logger.info(f"Using existing dagger_save_path: {dagger_save_path}")

    real_bachsize = args.batchSize
    logger.info(f"Resolved real_bachsize: {real_bachsize}")

    train_env = initialize_env(dataset_path=dataset_path, save_path=dagger_save_path, train_json_path=train_json_path, activate_maps=activate_maps)
    logger.info(f"Initialized train_env: batch_size={train_env.batch_size}, scenes_count={len(train_env.scenes)}")

    for dagger_it in range(int(args.dagger_it)):
        logger.info(f"Starting dagger iteration: {dagger_it}")

        if is_dist_avail_and_initialized():
            logger.warning("Distributed initialized; destroying process group.")
            torch.distributed.destroy_process_group()
            logger.info("Destroyed process group.")
        else:
            logger.info("Distributed not initialized; skipping destroy.")

        args.DistributedDataParallel = False
        args.batchSize = real_bachsize
        logger.info(f"Set args.DistributedDataParallel={args.DistributedDataParallel}, args.batchSize={args.batchSize}")

        try:
            logger.info("Initializing FiSModelWrapper...")
            model_wrapper = FiSModelWrapper(model_args=model_args, data_args=data_args, real_bachsize=real_bachsize)
            logger.info("FiSModelWrapper initialized.")
        except Exception as e:
            logger.error(f"FiSModelWrapper init failed: {e}")
            exe_type, exe_value, exe_traceback = sys.exc_info()
            logger.error("Traceback:\n" + ''.join(traceback.format_exception(exe_type, exe_value, exe_traceback)))
            raise

        try:
            logger.info("Initializing Assist...")
            assist = Assist(always_help=True, use_gt=True)
            logger.info("Assist initialized.")
        except Exception as e:
            logger.error(f"Assist init failed: {e}")
            exe_type, exe_value, exe_traceback = sys.exc_info()
            logger.error("Traceback:\n" + ''.join(traceback.format_exception(exe_type, exe_value, exe_traceback)))
            raise

        try:
            logger.info("Collecting data...")
            collect_data(model_wrapper=model_wrapper,
                         assist = assist,
                         train_env=train_env,
                         data_it=dagger_it)
            logger.info("Data collection completed.")
        except Exception as e:
            logger.error(f"collect_data failed: {e}")
            exe_type, exe_value, exe_traceback = sys.exc_info()
            logger.error("Traceback:\n" + ''.join(traceback.format_exception(exe_type, exe_value, exe_traceback)))
            raise

    logger.info("Deleting VectorEnvUtil...")
    train_env.delete_VectorEnvUtil()
    logger.info("Deleted VectorEnvUtil.")