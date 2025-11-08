import numpy as np
import torch
from PIL import Image
from src.model_wrapper.base_model import BaseModelWrapper
# from src.model_wrapper.utils.fis_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor
import sys
from pathlib import Path
sys.path.append('/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow')
from models import load_vla 

import logging
# 初始化带文件名与行号的日志格
_logger_name = "FiSModelWrapper"
_module_logger = logging.getLogger(_logger_name)
if not _module_logger.handlers:
    _module_logger.setLevel(logging.DEBUG)
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s (%(filename)s:%(lineno)d) %(message)s")
    _handler.setFormatter(_formatter)
    _module_logger.addHandler(_handler)


class FiSModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args, real_bachsize):
        self.logger = logging.getLogger(_logger_name)
        self.logger.debug("Initializing FiSModelWrapper")
        # 使用统一的 model_load 来加载并设置设备/精度/评估模式
        self.model = self.model_load(model_args)
        self.tokenizer = self.model.vlm.llm_backbone.tokenizer
        self.image_processor = FiSImageProcessor(self.model.vision_backbone)
        self.traj_model = None # FiSvla 是端到端模型，不需要单独的 traj_model
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

        self.predict_mode = 'diff'
        self.slow_fast_ratio = 4
        self.use_robot_state = True
        self.load_pointcloud = False
        
        self.batch_size = real_bachsize
        self.logger.info(f"配置: predict_mode={self.predict_mode}, slow_fast_ratio={self.slow_fast_ratio}, "
                         f"use_robot_state={self.use_robot_state}, load_pointcloud={self.load_pointcloud}, "
                         f"batch_size={self.batch_size}")

        # 为 'diff' 模式创建缓存（Fast-in-Slow风格异步采样）
        self.slow_input_ids_cache = [None] * self.batch_size
        self.slow_embed_cache = [None] * self.batch_size
        
        # 异步采样状态跟踪（类似sim.py的slow_cnt）
        self.step_counters = {}  # 按batch跟踪步数
        
        # 步数计数器（用于异步采样）
        self.step_counters = {}

    def _log_var(self, name, value):
        """Log intermediate variable type, shape and sample"""
        try:
            if isinstance(value, torch.Tensor):
                info = f"Tensor shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
                if value.numel() > 0:
                    info += f", min={value.min().item():.4f}, max={value.max().item():.4f}"
                self.logger.debug(f"{name}: {info}")
            elif isinstance(value, np.ndarray):
                info = f"ndarray shape={value.shape}, dtype={value.dtype}"
                if value.size > 0 and np.issubdtype(value.dtype, np.number):
                    info += f", min={value.min():.4f}, max={value.max():.4f}"
                self.logger.debug(f"{name}: {info}")
            elif isinstance(value, (list, tuple)):
                preview = value[:2] if len(value) > 0 else []
                self.logger.debug(f"{name}: {type(value).__name__} len={len(value)} sample={preview}")
            elif isinstance(value, dict):
                keys = list(value.keys())
                preview_types = {k: type(value[k]).__name__ for k in keys[:5]}
                self.logger.debug(f"{name}: dict keys={keys} preview_types={preview_types}")
            elif hasattr(value, "size") and hasattr(value, "mode"):
                # PIL.Image
                self.logger.debug(f"{name}: PIL.Image size={value.size} mode={value.mode}")
            else:
                text = str(value)
                self.logger.debug(f"{name}: type={type(value).__name__} value={text[:200]}")
        except Exception as e:
            self.logger.warning(f"_log_var failed: name={name}, err={e}")


    def _extract_robot_state_from_episode(self, episode, batch_idx):
        try:
            if not episode or len(episode) == 0:
                self.logger.warning(f"[batch {batch_idx}] Episode is empty, returning default state")
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            current_step = episode[-1] 
            sensors = current_step.get('sensors', {})
            
            state_sensor = sensors.get('state', {})
            position = state_sensor.get('position', [0.0, 0.0, 0.0])
            
            orientation = state_sensor.get('orientation', [0.0, 0.0, 0.0, 1.0])
            
            if len(position) != 3:
                self.logger.warning(f"[batch {batch_idx}] Position dimension mismatch: expected 3, got {len(position)}")
                position = position[:3] if len(position) > 3 else position + [0.0] * (3 - len(position))
            
            if len(orientation) != 4:
                self.logger.warning(f"[batch {batch_idx}] Orientation dimension mismatch: expected 4 (quaternion), got {len(orientation)}")
                orientation = [0.0, 0.0, 0.0, 1.0]
            
            # 四元数转RPY
            rpy = _quaternion_to_rpy(orientation)
            
            # 组合成6维状态向量 [x, y, z, roll, pitch, yaw]
            robot_state = np.array(position + rpy, dtype=np.float32)
            
            self.logger.debug(f"[batch {batch_idx}] Robot state extracted successfully: position={position}, orientation={orientation}, rpy={rpy}")
            
            return robot_state
            
        except Exception as e:
            self.logger.error(f"[batch {batch_idx}] Robot state extraction failed: {e}, episode type={type(episode)}")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    def model_load(self, model_args):
        self.logger = logging.getLogger(_logger_name) if not hasattr(self, "logger") else self.logger
        self.logger.info("Starting to load FiS model")
        self._log_var("model_path", model_args.model_path)
        self._log_var("cuda", model_args.cuda)
        self.logger.debug(
            "Key parameters: "
            f"use_diff={model_args.use_diff}, diffusion_steps={model_args.training_diffusion_steps}, "
            f"llm_middle_layer={model_args.llm_middle_layer}, training_mode={model_args.training_mode}, "
            f"load_pointcloud={model_args.load_pointcloud}, pointcloud_pos={model_args.pointcloud_pos}, "
            f"action_chunk={model_args.action_chunk}, use_robot_state={model_args.use_robot_state}, "
            f"lang_subgoals_exist={model_args.lang_subgoals_exist}, action_dim={model_args.action_dim}"
        )
        model = load_vla(
                model_args.model_path,
                load_for_training=False,
                future_action_window_size=0,
                hf_token=model_args.hf_token,
                use_diff = 1,
                diffusion_steps = 100,
                llm_middle_layer = 30,
                training_mode = 'async',
                load_pointcloud = 0,
                pointcloud_pos="fast",
                action_chunk=1,
                load_state=1,
                lang_subgoals_exist=1,
                action_dim=7,
                )
        device = f'cuda:{model_args.cuda}'
        precision = torch.bfloat16
        
        self.logger.info(f"Moving model to device: device={device}, dtype={precision}")
        model.to(device=device, dtype=precision).eval()
        self.logger.debug("Model set to eval mode")

        return model


    def model_predict_slow_latent_embedding(self, prompt, slow_image):
        self._log_var("slow_prompt", prompt)
        self._log_var("slow_image", slow_image)
        input_ids, slow_latent_embedding = self.model.slow_system_forward(
            image_head_slow = slow_image,
            instruction = prompt,
            unnorm_key = 'rtx_dataset', 
            )
        self._log_var("slow_input_ids", input_ids)
        self._log_var("slow_latent_embedding", slow_latent_embedding)
        return input_ids, slow_latent_embedding

    def model_predict(self, model_args, predict_mode, model, image, prompt, cur_robot_state=None, slow_image=None, point_cloud=None, input_ids = None, slow_latent_embedding=None):
        self.logger.debug(f"model_predict called: predict_mode={predict_mode}")
        self._log_var("prompt", prompt)
        self._log_var("image_fast", image)
        if slow_image is not None:
            self._log_var("image_slow", slow_image)
        if point_cloud is not None:
            self._log_var("point_cloud", point_cloud)
        if cur_robot_state is not None:
            self._log_var("cur_robot_state", cur_robot_state)
        self.logger.debug(f"Sampling parameters: cfg_scale={float(model_args.cfg_scale)}, use_ddim=True, "
                          f"num_ddim_steps={int(model_args.ddim_steps)}, action_dim={model_args.action_dim}")
        
        if predict_mode == 'ar' or predict_mode == 'diff+ar':
            output = model.predict_action(
                    image_head_slow = slow_image,
                    image_head_fast = image,
                    point_cloud = point_cloud,
                    instruction = prompt,
                    unnorm_key='rtx_dataset',
                    cfg_scale = 1.0, 
                    use_ddim = True,
                    num_ddim_steps = 4,
                    cur_robot_state = cur_robot_state,
                    action_dim = 7,
                    predict_mode = predict_mode,
                    )
        elif predict_mode == 'diff':
            output = model.fast_system_forward(
                    image_head_fast = image,
                    point_cloud=None,
                    slow_latent_embedding = slow_latent_embedding,
                    input_ids = input_ids,
                    unnorm_key = 'rtx_dataset',
                    cur_robot_state = cur_robot_state,
                    cfg_scale = 1.0, 
                    use_ddim = True,
                    num_ddim_steps = 4,
                    action_dim = 7,
                    predict_mode = predict_mode,
                    )
            self._log_var("diff_output", output)
        else:
            self.logger.error(f"Unknown predict_mode: {predict_mode}")
            raise ValueError(f"Unknown predict_mode: {predict_mode}")
        return output

    def eval(self):
        self.logger.debug("Switching model to eval mode")
        self.model.eval()

    def _extract_images_async_sampling(self, current_state, current_step_index, batch_idx):
        fast_image, slow_image = None, None
        R = self.slow_fast_ratio
        
        try:
            if 'rgb' in current_state and isinstance(current_state['rgb'], list) and len(current_state['rgb']) > 0:
                fast_image = current_state['rgb'][0]  
                
                if current_step_index % R == 0:
                    slow_image = current_state['rgb'][0]  # Sampling moment: extract new slow image
                    self.logger.debug(f"[batch {batch_idx}] Sampling moment step={current_step_index}: extracting slow image")
                else:
                    # Non-sampling moment: do not extract new slow image, return None to indicate using cache
                    slow_image = None
                    self.logger.debug(f"[batch {batch_idx}] Non-sampling moment step={current_step_index}: using cached slow image")
                
        except Exception as e:
            self.logger.warning(f"[batch {batch_idx}] Failed to read rgb: {e}")
            
        return fast_image, slow_image
    
    def _compute_rotation_to_target(self, first_state, target_positions, batch_idx):
        rotation_to_target = None
        
        try:
            rot = np.array(first_state['sensors']['imu']["rotation"])
            pos0 = np.array(first_state['sensors']['state']['position'])
            
            if (isinstance(target_positions, list) and len(target_positions) > batch_idx and len(target_positions[batch_idx]) > 0):
                
                tgt_world = np.array(target_positions[batch_idx][0])
                vec_body = rot.T @ (tgt_world - pos0)
                x, y = float(vec_body[0]), float(vec_body[1])
                rotation_to_target = rotation_matrix_from_vector(x, y)
                
        except Exception as e:
            self.logger.debug(f"[batch {batch_idx}] rotation_to_target calculation failed: {e}")
            
        return rotation_to_target
    
    def _prepare_slow_frame_cache(self, current_episode, current_step_index, base_instruction, slow_image, batch_idx, R):
        is_sampling_frame = (current_step_index % R == 0)
        self.logger.debug(f"[batch {batch_idx}] is_sampling_frame={is_sampling_frame}, step={current_step_index}, R={R}")
        
        if (is_sampling_frame or self.slow_input_ids_cache[batch_idx] is None or self.slow_embed_cache[batch_idx] is None):
            
            sampling_step_idx = (current_step_index // R) * R
            slow_instruction = (current_episode[sampling_step_idx].get('instruction', base_instruction) if len(current_episode) > sampling_step_idx else base_instruction)
            
            self._log_var(f"[batch {batch_idx}] slow_instruction", slow_instruction)
            self.logger.debug(f"[batch {batch_idx}] Sampling step={sampling_step_idx}: extracting slow frame features")
            
            if slow_image is not None:
                input_ids, slow_embed = self.model_predict_slow_latent_embedding(
                    prompt=slow_instruction,
                    slow_image=slow_image
                )
                self.slow_input_ids_cache[batch_idx] = input_ids
                self.slow_embed_cache[batch_idx] = slow_embed
            else:
                self.logger.warning(f"[batch {batch_idx}] Sampling step but slow image is None, cannot extract features")
        else:
            self.logger.debug(f"[batch {batch_idx}] Non-sampling step: using cached slow frame features")
        
        return (self.slow_input_ids_cache[batch_idx], self.slow_embed_cache[batch_idx])

    def _transform_action_coordinates(self, action, rotation_matrix, batch_idx):
        if action is None or rotation_matrix is None:
            return action
            
        try:
            if action.shape[-1] >= 2:
                dx_body = float(action[..., 0])
                dy_body = float(action[..., 1])
                
                dx_world = rotation_matrix[0, 0] * dx_body + rotation_matrix[0, 1] * dy_body
                dy_world = rotation_matrix[1, 0] * dx_body + rotation_matrix[1, 1] * dy_body

                action_copy = action.copy() if hasattr(action, 'copy') else action
                action_copy[..., 0] = dx_world
                action_copy[..., 1] = dy_world
                
                self.logger.debug(f"[batch {batch_idx}] Action coordinate transform: "
                                f"({dx_body:.3f},{dy_body:.3f}) -> "
                                f"({dx_world:.3f},{dy_world:.3f})")
                
                return action_copy
                
        except Exception as e:
            self.logger.warning(f"[batch {batch_idx}] Action coordinate transform failed: {e}")
            
        return action

    def _build_model_predict_params(self, current_state, predict_mode, full_instruction,
                                   fast_image_pil, slow_image_pil, current_episode,
                                   current_step_index, base_instruction, batch_idx, slow_fast_ratio):
        params = {
            'predict_mode': predict_mode,
            'prompt': full_instruction,
            'image': fast_image_pil,  
            'slow_image': slow_image_pil,
        }

        if self.use_robot_state:

            robot_state_xyz_rpy = self._extract_robot_state_from_episode(current_episode, batch_idx)
            
            # Use existing proprio_numpy if available (backward compatibility)
            if 'proprio_numpy' in current_state:
                existing_state = current_state['proprio_numpy']
                if len(existing_state) == 6:
                    robot_state_xyz_rpy = np.array(existing_state, dtype=np.float32)
                    self.logger.debug(f"[batch {batch_idx}] Using existing proprio_numpy state")
                else:
                    self.logger.warning(f"[batch {batch_idx}] Existing proprio_numpy has abnormal dimensions ({len(existing_state)}), using state extracted from episode")
            
            # 规范化 robot_state 为长度 7，最后一位固定为 0
            import numpy as np
            robot_state_xyz_rpy = np.asarray(robot_state_xyz_rpy, dtype=np.float32).reshape(-1)
            if robot_state_xyz_rpy.size >= 6:
                robot_state_xyz_rpy = robot_state_xyz_rpy[:6]
            else:
                robot_state_xyz_rpy = np.pad(
                    robot_state_xyz_rpy, (0, 6 - robot_state_xyz_rpy.size), mode="constant", constant_values=0.0
                )
            robot_state_xyz_rpy = np.concatenate([robot_state_xyz_rpy, np.array([0.0], dtype=np.float32)], axis=0)

            params['cur_robot_state'] = robot_state_xyz_rpy
            self._log_var(f"[batch {batch_idx}] robot_state_xyz_rpy", params['cur_robot_state'])

        if self.load_pointcloud and 'point_cloud_numpy' in current_state:
            params['point_cloud'] = current_state['point_cloud_numpy']
            self._log_var(f"[batch {batch_idx}] point_cloud_numpy", params['point_cloud'])

        if predict_mode == 'diff':
            input_ids, slow_embed = self._prepare_slow_frame_cache(
                current_episode, current_step_index, base_instruction,
                slow_image_pil, batch_idx, slow_fast_ratio
            )
            params['input_ids'] = input_ids
            params['slow_latent_embedding'] = slow_embed

            is_sampling = (current_step_index % slow_fast_ratio == 0)
            self.logger.debug(f"[batch {batch_idx}] diff async sampling: step={current_step_index}, "
                            f"R={slow_fast_ratio}, is_sampling={is_sampling}, "
                            f"slow_image={'newly_extracted' if is_sampling else 'use_cache'}")
            self._log_var(f"[batch {batch_idx}] cache_input_ids", input_ids)
            self._log_var(f"[batch {batch_idx}] cache_slow_embed", slow_embed)

        return params

    def _extract_action_from_output(self, predicted_output, batch_idx):
        """Extract action from model output"""
        try:
            if self.predict_mode == 'diff':
                return predicted_output
            elif self.predict_mode == 'ar':
                actions_ar, _ = predicted_output if isinstance(predicted_output, tuple) else (predicted_output, None)
                return actions_ar
            elif self.predict_mode == 'diff+ar':
                actions_diff, actions_ar, _ = predicted_output
                return actions_diff
            else:
                self.logger.error(f"[batch {batch_idx}] Unknown predict_mode: {self.predict_mode}")
                return None
        except Exception as e:
            self.logger.error(f"[batch {batch_idx}] Action extraction failed: {e}")
            return None

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):

        self.logger.info(f"prepare_inputs started: episodes_len={len(episodes)}")
        
        inputs_list = []
        rot_to_targets = []
        predict_mode = self.predict_mode
        R = self.slow_fast_ratio
        
        batch_size = len(episodes)
        while len(self.slow_input_ids_cache) < batch_size:
            self.slow_input_ids_cache.append(None)
        while len(self.slow_embed_cache) < batch_size:
            self.slow_embed_cache.append(None)
        
        for i in range(batch_size):
            if i not in self.step_counters:
                self.step_counters[i] = 0
        
        for i in range(batch_size):
            current_episode = episodes[i]
            
            if not current_episode or len(current_episode) == 0:
                self.logger.warning(f"[batch {i}] episode is empty, skipping")
                rot_to_targets.append(None)
                continue
            
            first_state = current_episode[0]
            current_state = current_episode[-1]
            current_step_index = len(current_episode) - 1
            
            step_counter = self.step_counters[i]
            
            base_instruction = current_state.get('instruction', 'go forward')
            full_instruction = base_instruction
            
            fast_image, slow_image = self._extract_images_async_sampling(
                current_state, current_step_index, i
            )
            if fast_image is None:
                self.logger.error(f"[batch {i}] Fast image extraction failed, skipping")
                rot_to_targets.append(None)
                continue
                
            use_cached_slow_image = (slow_image is None and current_step_index > 0)
            if use_cached_slow_image:
                slow_image = fast_image  
                self.logger.debug(f"[batch {i}] Non-sampling step, using current frame as fallback slow image")
            
            try:
                fast_image_pil = Image.fromarray(fast_image)
                slow_image_pil = Image.fromarray(slow_image)
            except Exception as e:
                self.logger.error(f"[batch {i}] Image conversion failed: {e}")
                rot_to_targets.append(None)
                continue

            item_params = self._build_model_predict_params(
                current_state=current_state,
                predict_mode=predict_mode,
                full_instruction=full_instruction,
                fast_image_pil=fast_image_pil,
                slow_image_pil=slow_image_pil,
                current_episode=current_episode,
                current_step_index=current_step_index,
                base_instruction=base_instruction,
                batch_idx=i,
                slow_fast_ratio=R
            )
            
            self.step_counters[i] += 1
            
            rotation_to_target = self._compute_rotation_to_target(first_state, target_positions, i)
            rot_to_targets.append(rotation_to_target)
            
            self.logger.debug(f"[batch {i}] step={current_step_index}")
            self._log_var(f"[batch {i}] prompt", full_instruction)
            
            inputs_list.append(item_params)
        
        self.logger.info(f"prepare_inputs finished: successfully processed {len(inputs_list)}/{batch_size} samples")
        return inputs_list, rot_to_targets

    def run(self, inputs, episodes, rot_to_targets):

        batch_actions = []
        self.logger.info(f"Inference started: len(inputs)={len(inputs)}")
        
        for i in range(len(inputs)):
            item_params = inputs[i] 
            self._log_var(f"[run] item_params_keys[{i}]", list(item_params.keys()))

            try:
                predicted_output = self.model_predict(
                    model_args=self.model_args,
                    predict_mode=item_params['predict_mode'],
                    model=self.model,
                    image=item_params['image'],
                    prompt=item_params['prompt'],
                    cur_robot_state=item_params.get('cur_robot_state'),
                    slow_image=item_params.get('slow_image'),
                    point_cloud=item_params.get('point_cloud'),
                    input_ids=item_params.get('input_ids'),
                    slow_latent_embedding=item_params.get('slow_latent_embedding')
                )
                self._log_var(f"[run] predicted_output[{i}]", predicted_output)
                
                actions_for_this_item = self._extract_action_from_output(predicted_output, i)
                if actions_for_this_item is None:
                    self.logger.error(f"[run] predict_mode={self.predict_mode} did not return valid actions")
                    batch_actions.append(None)
                    continue
    
                if rot_to_targets[i] is not None:
                    actions_for_this_item = self._transform_action_coordinates(actions_for_this_item, rot_to_targets[i], i)
                
                self._log_var(f"[run] actions_for_item[{i}]", actions_for_this_item)
                batch_actions.append(actions_for_this_item[0] if actions_for_this_item is not None else None)
                
            except Exception as e:
                self.logger.error(f"[run] batch {i} inference failed: {e}", exc_info=True)
                batch_actions.append(None)

        valid_actions = [a for a in batch_actions if a is not None]
        if len(valid_actions) == 0:
            self.logger.error("All samples inference failed")
            return np.array([])
        refined_waypoints = np.stack(valid_actions)
        if refined_waypoints.ndim == 2:
            refined_waypoints = refined_waypoints[:, np.newaxis, :]

        refined_waypoints_xyz = refined_waypoints[:, :, :3]
        self._log_var("refined_waypoints", refined_waypoints)
        
        return refined_waypoints_xyz

    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        self.logger.info(f"predict_done: episodes={len(episodes)}, object_infos={len(object_infos)}")
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            self._log_var(f"[predict_done] episode[{i}]_len", len(episodes[i]))
            self._log_var(f"[predict_done] object_info[{i}]", object_infos[i])
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            self._log_var(f"[predict_done] result[{i}]", prediction_done)
            prediction_dones.append(prediction_done)
        return prediction_dones
        

def rotation_matrix_from_vector(x, y):
    v_x = np.array([x, y, 0])
    v_x = v_x / np.linalg.norm(v_x)
    v_y = np.array([-v_x[1], v_x[0], 0])
    v_y = v_y / np.linalg.norm(v_y)
    v_z = np.array([0, 0, 1])
    rotation_matrix = np.column_stack((v_x, v_y, v_z))
    return rotation_matrix

def _quaternion_to_rpy(quaternion):
    try:
        if len(quaternion) != 4:
            raise ValueError(f"Quaternion length must be 4, got {len(quaternion)}")
            
        w, x, y, z = quaternion
            
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
            
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
            
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
            
        return [roll, pitch, yaw]
    except Exception as e:
        raise RuntimeError(f"Quaternion to RPY conversion failed: {e}") from e


class FiSImageProcessor:

    def __init__(self, vision_backbone=None):
        self.vision_backbone = vision_backbone
        self.logger = logging.getLogger(_logger_name + ".FiSImageProcessor")
        self.logger.debug("初始化 FiSImageProcessor")
        
    def preprocess(self, images, return_tensors='pt'):
        """
        使用Fast-in-Slow模型的图像转换器预处理图像
        
        Args:
            images: numpy数组，形状为[batch_size, height, width, channels]
            return_tensors: 返回张量的类型，'pt'表示PyTorch
            
        Returns:
            包含'pixel_values'键的字典，值为预处理后的图像张量
        """
        self._log_images_meta(images)
        
        # 处理每张图像
        processed_images = []
        for idx, img in enumerate(images):
            # 转换为PIL图像
            pil_img = Image.fromarray(img)
            self.logger.debug(f"[preprocess] idx={idx}, PIL size={pil_img.size}, mode={pil_img.mode}")
            
            # 使用Fast-in-Slow的图像转换器
            if self.vision_backbone and hasattr(self.vision_backbone, 'image_transform'):
                # 使用DinoSigLIPImageTransform处理图像
                processed_img = self.vision_backbone.image_transform(pil_img)
                self.logger.debug(f"[preprocess] idx={idx}, 使用 image_transform, 类型={type(processed_img).__name__}")
            else:
                # 回退到基本处理
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                processed_img = transform(pil_img)
                self.logger.debug(f"[preprocess] idx={idx}, 使用基本 transform, "
                                  f"类型={type(processed_img).__name__}, "
                                  f"形状={tuple(processed_img.shape) if hasattr(processed_img, 'shape') else 'n/a'}")
            
            processed_images.append(processed_img)
        
        # 堆叠成批次
        if return_tensors == 'pt':
            if isinstance(processed_images[0], dict):
                # 如果是字典形式（DinoSigLIP可能返回多个特征）
                result = {}
                for key in processed_images[0].keys():
                    result[key] = torch.stack([img[key] for img in processed_images])
                pixel_values = result
                try:
                    shapes = {k: tuple(v.shape) for k, v in pixel_values.items()}
                    self.logger.debug(f"[preprocess] 输出为 dict, shapes={shapes}")
                except Exception as e:
                    self.logger.warning(f"[preprocess] dict 输出记录失败: {e}")
            else:
                pixel_values = torch.stack(processed_images)
                self.logger.debug(f"[preprocess] 输出为 Tensor, shape={tuple(pixel_values.shape)}, dtype={pixel_values.dtype}")
        else:
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        
        return {'pixel_values': pixel_values}

    def _log_images_meta(self, images):
        try:
            if isinstance(images, np.ndarray):
                self.logger.debug(f"[preprocess] 输入 ndarray shape={images.shape}, dtype={images.dtype}")
            elif isinstance(images, (list, tuple)) and len(images) > 0:
                sample = images[0]
                self.logger.debug(f"[preprocess] 输入 {type(images).__name__} len={len(images)}, sample_type={type(sample).__name__}")
            else:
                self.logger.debug(f"[preprocess] 输入类型={type(images).__name__}")
        except Exception as e:
            self.logger.warning(f"[preprocess] 记录输入信息失败: {e}")