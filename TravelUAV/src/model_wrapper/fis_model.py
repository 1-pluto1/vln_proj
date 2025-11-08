import numpy as np
import torch
from PIL import Image
from src.model_wrapper.base_model import BaseModelWrapper
# from src.model_wrapper.utils.fis_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor
import sys
from pathlib import Path
sys.path.append('/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow')
from models import load_vla  # 从Fast-in-Slow导入模型加载函数

import logging
# 初始化带文件名与行号的日志格式
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
        self.logger.debug("初始化 FiSModelWrapper")
        # 使用统一的 model_load 来加载并设置设备/精度/评估模式
        self.model = self.model_load(model_args)
        self.tokenizer = self.model.vlm.llm_backbone.tokenizer
        self.image_processor = FiSImageProcessor(self.model.vision_backbone)
        self.traj_model = None # FiSvla 是端到端模型，不需要单独的 traj_model
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

        self.predict_mode = 'diff+ar'
        self.slow_fast_ratio = 4
        self.use_robot_state = True
        self.load_pointcloud = False
        
        self.batch_size = real_bachsize
        self.logger.info(f"配置: predict_mode={self.predict_mode}, slow_fast_ratio={self.slow_fast_ratio}, "
                         f"use_robot_state={self.use_robot_state}, load_pointcloud={self.load_pointcloud}, "
                         f"batch_size={self.batch_size}")

        # 为 'diff' 模式创建缓存
        self.slow_input_ids_cache = [None] * self.batch_size
        self.slow_embed_cache = [None] * self.batch_size

    def _log_var(self, name, value):
        """打印中间变量的类型、形状与样例"""
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
            self.logger.warning(f"_log_var 失败: name={name}, err={e}")

    def model_load(self, model_args):
        self.logger = logging.getLogger(_logger_name) if not hasattr(self, "logger") else self.logger
        self.logger.info("开始加载 FiS 模型")
        self._log_var("model_path", model_args.model_path)
        self._log_var("cuda", model_args.cuda)
        self.logger.debug(
            "重要参数: "
            f"use_diff={model_args.use_diff}, diffusion_steps={model_args.training_diffusion_steps}, "
            f"llm_middle_layer={model_args.llm_middle_layer}, training_mode={model_args.training_mode}, "
            f"load_pointcloud={model_args.load_pointcloud}, pointcloud_pos={model_args.pointcloud_pos}, "
            f"action_chunk={model_args.action_chunk}, use_robot_state={model_args.use_robot_state}, "
            f"lang_subgoals_exist={model_args.lang_subgoals_exist}, action_dim={model_args.action_dim}"
        )
        model = load_vla(
                model_args.model_path,
                load_for_training=False,
                future_action_window_size=int(model_args.model_action_steps),
                hf_token=model_args.hf_token,
                use_diff = 1 if model_args.use_diff else 0,
                diffusion_steps = model_args.training_diffusion_steps,
                llm_middle_layer = model_args.llm_middle_layer,
                training_mode = model_args.training_mode,
                load_pointcloud = model_args.load_pointcloud,
                pointcloud_pos=model_args.pointcloud_pos,
                action_chunk=model_args.action_chunk,
                load_state=model_args.use_robot_state,
                lang_subgoals_exist=model_args.lang_subgoals_exist,
                action_dim=model_args.action_dim,
                )
        device = f'cuda:{model_args.cuda}'
        precision = torch.bfloat16
        
        self.logger.info(f"模型迁移到设备: device={device}, dtype={precision}")
        model.to(device=device, dtype=precision).eval()
        self.logger.debug("模型已设置为 eval 模式")

        return model


    def model_predict_slow_latent_embedding(self, prompt, slow_image):
        self._log_var("slow_prompt", prompt)
        self._log_var("slow_image", slow_image)
        input_ids, slow_latent_embedding = self.model.slow_system_forward(
            image_head_slow = slow_image,
            instruction = prompt,
            unnorm_key = 'rlbench', 
            )
        self._log_var("slow_input_ids", input_ids)
        self._log_var("slow_latent_embedding", slow_latent_embedding)
        return input_ids, slow_latent_embedding

    def model_predict(self, model_args, predict_mode, model, image, prompt, cur_robot_state=None, slow_image=None, point_cloud=None, input_ids = None, slow_latent_embedding=None):
        self.logger.debug(f"model_predict 调用: predict_mode={predict_mode}")
        self._log_var("prompt", prompt)
        self._log_var("image_fast", image)
        if slow_image is not None:
            self._log_var("image_slow", slow_image)
        if point_cloud is not None:
            self._log_var("point_cloud", point_cloud)
        if cur_robot_state is not None:
            self._log_var("cur_robot_state", cur_robot_state)
        self.logger.debug(f"采样参数: cfg_scale={float(model_args.cfg_scale)}, use_ddim=True, "
                          f"num_ddim_steps={int(model_args.ddim_steps)}, action_dim={model_args.action_dim}")
        
        if predict_mode == 'ar' or predict_mode == 'diff+ar':
            output = model.predict_action(
                    image_head_slow = slow_image,
                    image_head_fast = image,
                    point_cloud = point_cloud,
                    instruction = prompt,
                    unnorm_key='rlbench',
                    cfg_scale = float(model_args.cfg_scale), 
                    use_ddim = True,
                    num_ddim_steps = int(model_args.ddim_steps),
                    cur_robot_state = cur_robot_state,
                    action_dim = model_args.action_dim,
                    predict_mode = predict_mode,
                    )
            if predict_mode == 'ar':
                try:
                    actions, _ = output if isinstance(output, tuple) else (output, None)
                    self._log_var("ar_actions", actions)
                except Exception as e:
                    self.logger.warning(f"AR 模式输出记录失败: {e}")
            else:
                try:
                    actions_diff, actions_ar, aux = output
                    self._log_var("actions_diff", actions_diff)
                    self._log_var("actions_ar", actions_ar)
                    self._log_var("predict_aux", aux)
                except Exception as e:
                    self.logger.warning(f"diff+ar 模式输出记录失败: {e}")
        elif predict_mode == 'diff':
            output = model.fast_system_forward(
                    image_head_fast = image,
                    point_cloud=point_cloud,
                    slow_latent_embedding = slow_latent_embedding,
                    input_ids = input_ids,
                    unnorm_key = 'rlbench',
                    cur_robot_state = cur_robot_state,
                    cfg_scale = float(model_args.cfg_scale), 
                    use_ddim = True,
                    num_ddim_steps = int(model_args.ddim_steps),
                    action_dim = model_args.action_dim,
                    predict_mode = predict_mode,
                    )
            self._log_var("diff_output", output)
        else:
            self.logger.error(f"未知的 predict_mode: {predict_mode}")
            raise ValueError(f"Unknown predict_mode: {predict_mode}")
        return output

    def eval(self):
        self.logger.debug("切换模型到 eval 模式")
        self.model.eval()

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        self.logger.info(f"prepare_inputs 开始, episodes_len={len(episodes)}")
        
        inputs_list = []
        rot_to_targets = []
        predict_mode = self.predict_mode
        R = int(self.slow_fast_ratio)
    
        # 缓存长度对齐
        if len(self.slow_input_ids_cache) < len(episodes):
            self.slow_input_ids_cache.extend([None] * (len(episodes) - len(self.slow_input_ids_cache)))
        if len(self.slow_embed_cache) < len(episodes):
            self.slow_embed_cache.extend([None] * (len(episodes) - len(self.slow_embed_cache)))
    
        for i in range(len(episodes)):
            current_episode = episodes[i]
            if not current_episode or len(current_episode) == 0:
                self.logger.warning(f"[batch {i}] episode 为空，跳过")
                rot_to_targets.append(None)
                continue
    
            # 当前帧与首帧（用于旋转计算）
            first_state = current_episode[0]
            current_state = current_episode[-1]
            current_step_index = len(current_episode) - 1
    
            # 指令与辅助信息
            base_instruction = current_state.get('instruction', 'go forward')
            assist = assist_notices[i] if assist_notices is not None else None
            full_instruction = f"{base_instruction}. Assistant notice: {assist}" if assist else base_instruction
    
            # 选择快帧/慢帧：来自该步骤字典的 rgb/rgb_record
            fast_image_np = None
            slow_image_np = None
    
            try:
                if 'rgb' in current_state and isinstance(current_state['rgb'], list) and len(current_state['rgb']) > 0:
                    fast_image_np = current_state['rgb'][-1]                 # 最新一帧作为快帧
                    slow_image_np = current_state['rgb'][0]                  # 最早一帧近似慢帧
                if ('rgb_record' in current_state 
                    and isinstance(current_state['rgb_record'], list) 
                    and len(current_state['rgb_record']) > 0):
                    # 若有 record，优先用 record[0] 作为慢帧
                    slow_image_np = current_state['rgb_record'][0] if slow_image_np is None else slow_image_np
                    # 可选：也可用 record[-1] 更新快帧
                    if fast_image_np is None:
                        fast_image_np = current_state['rgb_record'][-1]
            except Exception as e:
                self.logger.warning(f"[batch {i}] 读取 rgb/rgb_record 失败: {e}")
    
            if fast_image_np is None and slow_image_np is None:
                self.logger.error(f"[batch {i}] 无有效图像，跳过该样本")
                rot_to_targets.append(None)
                continue
    
            # 兜底：若缺其中之一，复用另一张
            if fast_image_np is None:
                fast_image_np = slow_image_np
            if slow_image_np is None:
                slow_image_np = fast_image_np
    
            # numpy -> PIL
            try:
                fast_image_pil = Image.fromarray(fast_image_np)
                slow_image_pil = Image.fromarray(slow_image_np)
            except Exception as e:
                self.logger.error(f"[batch {i}] numpy->PIL 转换失败: {e}")
                rot_to_targets.append(None)
                continue
    
            item_params = {
                'model': self.model,
                'predict_mode': predict_mode,
                'prompt': full_instruction,
                'image': fast_image_pil,
                'slow_image': slow_image_pil,
            }
    
            # 机器人状态（仅当存在且形状匹配时再传入）
            if self.use_robot_state and 'proprio_numpy' in current_state:
                item_params['cur_robot_state'] = current_state.get('proprio_numpy')
                self._log_var(f"[batch {i}] proprio_numpy", item_params['cur_robot_state'])
            # 点云（如果该场景提供）
            if self.load_pointcloud and 'point_cloud_numpy' in current_state:
                item_params['point_cloud'] = current_state.get('point_cloud_numpy')
                self._log_var(f"[batch {i}] point_cloud_numpy", item_params['point_cloud'])
    
            # 仅 diff 模式维护慢帧缓存
            if predict_mode == 'diff':
                is_new_slow_frame = (current_step_index % R == 0)
                self.logger.debug(f"[batch {i}] is_new_slow_frame={is_new_slow_frame}")
                if is_new_slow_frame or self.slow_input_ids_cache[i] is None or self.slow_embed_cache[i] is None:
                    slow_instruction = (current_episode[(current_step_index // R) * R].get('instruction', base_instruction)
                                        if len(current_episode) > 0 else base_instruction)
                    self._log_var(f"[batch {i}] slow_instruction", slow_instruction)
                    input_ids, slow_embed = self.model_predict_slow_latent_embedding(
                        prompt=slow_instruction,
                        slow_image=item_params['slow_image']
                    )
                    self.slow_input_ids_cache[i] = input_ids
                    self.slow_embed_cache[i] = slow_embed
    
                item_params['input_ids'] = self.slow_input_ids_cache[i]
                item_params['slow_latent_embedding'] = self.slow_embed_cache[i]
                self._log_var(f"[batch {i}] cache_input_ids", item_params['input_ids'])
                self._log_var(f"[batch {i}] cache_slow_embed", item_params['slow_latent_embedding'])
    
            # 计算 rotation_to_target（可选：对齐目标方向）
            rotation_to_target = None
            try:
                rot = np.array(first_state['sensors']['imu']["rotation"])
                pos0 = np.array(first_state['sensors']['state']['position'])
                if isinstance(target_positions, list) and len(target_positions) > i and len(target_positions[i]) > 0:
                    tgt_world = np.array(target_positions[i][0])
                    vec_body = rot.T @ (tgt_world - pos0)
                    x, y = float(vec_body[0]), float(vec_body[1])
                    rotation_to_target = rotation_matrix_from_vector(x, y)
            except Exception as e:
                self.logger.debug(f"[batch {i}] rotation_to_target 计算失败: {e}")
            rot_to_targets.append(rotation_to_target)
    
            # 日志打印
            self.logger.debug(f"[batch {i}] step={current_step_index}")
            self._log_var(f"[batch {i}] prompt", full_instruction)
            self._log_var(f"[batch {i}] fast_image_pil", item_params['image'])
            self._log_var(f"[batch {i}] slow_image_pil", item_params['slow_image'])
    
            inputs_list.append(item_params)
    
        self.logger.info(f"prepare_inputs 结束: inputs_list_len={len(inputs_list)}")
        return inputs_list, rot_to_targets


    def run(self, inputs, episodes, rot_to_targets):
        batch_actions = []
        self.logger.info(f"开始推理: len(inputs)={len(inputs)}")
        
        for i in range(len(inputs)):
            item_params = inputs[i] 
            self._log_var(f"[run] item_params_keys[{i}]", list(item_params.keys()))

            predicted_output = self.model_predict(
                self.model_args,
                **item_params
            )
            self._log_var(f"[run] predicted_output[{i}]", predicted_output)
            
            actions_for_this_item = None
            if self.predict_mode == 'diff':
                actions_for_this_item = predicted_output
            elif self.predict_mode == 'ar':
                actions_for_this_item, _ = predicted_output # (actions, language_subgoals)
            elif self.predict_mode == 'diff+ar':
                actions_diff, actions_ar, _ = predicted_output
                actions_for_this_item = actions_diff 
            
            if actions_for_this_item is None:
                self.logger.error(f"[run] predict_mode={self.predict_mode} 未返回有效动作")
                raise ValueError(f"predict_mode '{self.predict_mode}' 没有返回有效动作")
                
            self._log_var(f"[run] actions_for_item[{i}]", actions_for_this_item)
            batch_actions.append(actions_for_this_item[0]) 

        refined_waypoints = np.stack(batch_actions)
        self._log_var("refined_waypoints", refined_waypoints)
        
        return refined_waypoints

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