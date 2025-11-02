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

from src.model_wrapper.travel_llm import TravelModelWrapper
from src.model_wrapper.fis_model import FiSModelWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from src.vlnce_src.env_uav import AirVLNENV
from src.vlnce_src.assist import Assist
from src.vlnce_src.closeloop_util import DaggerBatchState, setup, CheckPort, initialize_env, is_dist_avail_and_initialized


def collect_data(model_wrapper: BaseModelWrapper, assist: Assist, train_env: AirVLNENV, data_it=0):
    assert args.collect_type in ['dagger']
    beta = float(args.dagger_p)
    
    model_wrapper.eval()
    
    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        pbar = tqdm.tqdm(total=end_iter)
        while start_iter < end_iter:
            env_batchs = train_env.next_minibatch(skip_scenes=[])
            if env_batchs is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break
            start_iter += train_env.batch_size
            pbar.update(n=train_env.batch_size)
            
            dagger_batch_state_manager = DaggerBatchState(train_env.batch_size, env_batchs, train_env)
            
            outputs = train_env.reset()
            
            dagger_batch_state_manager.update_from_env_output(outputs)
            inputs, rot_to_targets = model_wrapper.prepare_inputs(dagger_batch_state_manager.episodes, dagger_batch_state_manager.target_positions)
        
            # closeloop steps
            for t in range(int(args.maxWaypoints) + 1):
                logger.info('dagger_it: {} \t {} - {} / {}'.format(data_it, int(train_env.index_data)-int(train_env.batch_size), t, end_iter))
                try:
                    is_terminate = dagger_batch_state_manager.check_dagger_batch_termination(dagger_it=data_it)
                    if is_terminate:
                        break
                    
                    # model action / teacher action
                    refined_waypoints = model_wrapper.run(inputs=inputs, episodes=dagger_batch_state_manager.episodes, rot_to_targets=rot_to_targets)
                    choose_teacher = torch.rand(args.batchSize) < beta
                    for i in range(len(choose_teacher)):
                        if choose_teacher[i] or dagger_batch_state_manager.need_teacher[i]:
                            refined_waypoints[i] = dagger_batch_state_manager.episodes[i][-1]['teacher_action']

                    train_env.makeActions(refined_waypoints)
                    outputs = train_env.get_obs()
                    dagger_batch_state_manager.update_from_env_output(outputs, assist.check_collision_by_depth)

                    dagger_batch_state_manager.dagger_step_back()
                    
                    assist_notices = assist.get_assist_notice(episodes=dagger_batch_state_manager.episodes, trajs=dagger_batch_state_manager.trajs, object_infos=dagger_batch_state_manager.object_infos, target_positions=dagger_batch_state_manager.target_positions)
                    inputs, _ = model_wrapper.prepare_inputs(dagger_batch_state_manager.episodes, dagger_batch_state_manager.target_positions, assist_notices)
                except Exception as e:
                    exe_type, exe_value, exe_traceback = sys.exc_info()
                    exe_info_list = traceback.format_exception(
                        exe_type, exe_value, exe_traceback)
                    tracebacks = ''.join(exe_info_list)
                    print('traceback:', tracebacks)
                    print(e)
                    break
        pbar.close()
    
    logger.info('END data_it: {}'.format(data_it))


if __name__ == "__main__":
    setup()

    assert CheckPort(), 'error port'

    activate_maps = args.activate_maps
    dataset_path = args.dataset_path
    train_json_path = args.train_json_path
    dagger_save_path = args.dagger_save_path

    if not os.path.isdir(dagger_save_path):
        os.makedirs(dagger_save_path, exist_ok=True)
    real_bachsize = args.batchSize

    train_env = initialize_env(dataset_path=dataset_path, save_path=dagger_save_path, train_json_path=train_json_path, activate_maps=activate_maps)

    for dagger_it in range(int(args.dagger_it)):
        
        if is_dist_avail_and_initialized():
            torch.distributed.destroy_process_group()

        args.DistributedDataParallel = False
        args.batchSize = real_bachsize
        
        model_wrapper = FiSModelWrapper(model_args=model_args, data_args=data_args)
        
        assist = Assist(always_help=True, use_gt=True)
        
        collect_data(model_wrapper=model_wrapper,
                     assist = assist,
                     train_env=train_env,
                     data_it=dagger_it)

    train_env.delete_VectorEnvUtil()