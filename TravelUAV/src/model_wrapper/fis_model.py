import numpy as np
import torch
from PIL import Image
from src.model_wrapper.base_model import BaseModelWrapper
# from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor
import sys
from pathlib import Path
sys.path.append('/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow')
from models import load_vla  # 从Fast-in-Slow导入模型加载函数


class FiSModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        # 使用统一的 model_load 来加载并设置设备/精度/评估模式
        self.model = self.model_load(model_args)
        self.tokenizer = self.model.vlm.llm_backbone.tokenizer
        self.image_processor = FiSImageProcessor(self.model.vision_backbone)
        self.traj_model = None # FiSvla 是端到端模型，不需要单独的 traj_model
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

        self.predict_mode = self.model_args.predict_mode
        self.slow_fast_ratio = self.model_args.slow_fast_ratio
        self.use_robot_state = bool(self.model_args.use_robot_state)
        self.load_pointcloud = bool(self.model_args.load_pointcloud)
        
        self.batch_size = data_args.batchSize 


        # 为 'diff' 模式创建缓存
        self.slow_input_ids_cache = [None] * self.batch_size
        self.slow_embed_cache = [None] * self.batch_size

    def model_load(self, model_args):
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
        
        model.to(device=device, dtype=precision).eval()

        return model


    def model_predict_slow_latent_embedding(self, prompt, slow_image):
        input_ids, slow_latent_embedding = self.model.slow_system_forward(
            image_head_slow = slow_image,
            instruction = prompt,
            unnorm_key = 'rlbench', 
            )
        return input_ids, slow_latent_embedding

    def model_predict(self, model_args, predict_mode, model, image, prompt, cur_robot_state=None, slow_image=None, point_cloud=None, input_ids = None, slow_latent_embedding=None):
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
        return output
    
    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        
        inputs_list = []
        batch_size = len(episodes)
        R = self.slow_fast_ratio
        predict_mode = self.model_args.predict_mode

        # 确保缓存大小正确
        if len(self.slow_embed_cache) != batch_size:
            self.slow_input_ids_cache = [None] * batch_size
            self.slow_embed_cache = [None] * batch_size

        for i in range(batch_size):

            current_episode_history = episodes[i]
            current_state = current_episode_history[-1]
            current_step_index = len(current_episode_history) - 1
            
            base_instruction = current_state.get('instruction', 'go forward')
            assist = assist_notices[i] if assist_notices is not None else None
            
            # 注入辅助信息
            if assist:
                full_instruction = f"{base_instruction}. Assistant notice: {assist}"
            else:
                full_instruction = base_instruction
                
            item_params = {
                'prompt': full_instruction,
                'predict_mode': predict_mode,
            }
            
            item_params['image'] = current_state.get('image_head_fast_pil')

            slow_image_index = (current_step_index // R) * R
            slow_image_state = current_episode_history[slow_image_index]
            
            item_params['slow_image'] = slow_image_state.get('image_head_slow_pil')

            if self.use_robot_state:
                item_params['cur_robot_state'] = current_state.get('proprio_numpy')
            if self.load_pointcloud:
                item_params['point_cloud'] = current_state.get('point_cloud_numpy')

            if predict_mode == 'diff':
                is_new_slow_frame = (current_step_index % R == 0)
                
                if is_new_slow_frame:
                    slow_instruction = slow_image_state.get('instruction', 'go forward')
                    
                    input_ids, slow_embed = self.model_predict_slow_latent_embedding(
                        prompt=slow_instruction,
                        slow_image=item_params['slow_image'] # 使用我们刚获取的慢图像
                    )
                    # 更新缓存
                    self.slow_input_ids_cache[i] = input_ids
                    self.slow_embed_cache[i] = slow_embed

                item_params['input_ids'] = self.slow_input_ids_cache[i]
                item_params['slow_latent_embedding'] = self.slow_embed_cache[i]

            inputs_list.append(item_params)
            
        rot_to_targets = [None] * batch_size
        
        return inputs_list, rot_to_targets
    
    def eval(self):
        self.model.eval()
        

    def run(self, inputs, episodes, rot_to_targets):
        
        batch_actions = []
        batch_size = len(inputs)
        
        for i in range(batch_size):
            item_params = inputs[i] 
            

            predicted_output = self.model_predict(
                self.model_args,
                **item_params
            )
            
            actions_for_this_item = None
            if self.predict_mode == 'diff':
                actions_for_this_item = predicted_output
            elif self.predict_mode == 'ar':
                actions_for_this_item, _ = predicted_output # (actions, language_subgoals)
            elif self.predict_mode == 'diff+ar':
                actions_diff, actions_ar, _ = predicted_output
                actions_for_this_item = actions_diff 
            
            if actions_for_this_item is None:
                raise ValueError(f"predict_mode '{self.predict_mode}' 没有返回有效动作")
                

            batch_actions.append(actions_for_this_item[0]) 

        refined_waypoints = np.stack(batch_actions)
        
        return refined_waypoints
    
    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones
        
    # def run(self, inputs, episodes, rot_to_targets):
    #     # 懒初始化/扩展每环境状态缓存
    #     if len(self._env_state) != len(episodes):
    #         self._env_state = self._env_state[:len(episodes)] + [
    #             {'slow_cnt': 0, 'last_t': -1, 'input_ids': None, 'slow_latent_embedding': None,
    #              'slow_image': None, 'point_cloud_slow': None}
    #             for _ in range(len(episodes) - len(self._env_state))
    #         ]

    #     refined_waypoints_local = []
    #     for i in range(len(episodes)):
    #         traj = episodes[i]
    #         latest_t = len(traj) - 1
    #         state_i = self._env_state[i]
    #         ran_fast = False

    #         # 遍历新增的时间步：从上次处理的 last_t+1 到当前最新帧 latest_t
    #         if latest_t >= 0:
    #             start_t = max(state_i['last_t'] + 1, 0)
    #             for t in range(start_t, latest_t + 1):
    #                 frame = traj[t]

    #                 # 慢系统：命中慢周期或首次，更新慢系统缓存（以及慢点云缓存）
    #                 if self.predict_mode == 'diff':
    #                     if state_i['slow_latent_embedding'] is None or (state_i['slow_cnt'] % self.slow_fast_ratio == 0):
    #                         slow_image = frame['rgb']
    #                         prompt = frame['instruction']
    #                         input_ids, slow_latent_embedding = model_predict_slow_latent_embedding(self.model, prompt, slow_image)
    #                         state_i['slow_image'] = slow_image
    #                         state_i['input_ids'] = input_ids
    #                         state_i['slow_latent_embedding'] = slow_latent_embedding
    #                         if self.load_pointcloud and self.pointcloud_pos == 'slow':
    #                             state_i['point_cloud_slow'] = frame.get('point_cloud', None)

    #                 # 最新时间步：执行快系统推理一次
    #                 if t == latest_t:
    #                     fast_image = frame['rgb']
    #                     prompt = frame['instruction']
    #                     cur_robot_state = frame['sensors'].get('state', None)

    #                     if self.load_pointcloud and self.pointcloud_pos == 'fast':
    #                         point_cloud = frame.get('point_cloud', None)
    #                     elif self.load_pointcloud and self.pointcloud_pos == 'slow':
    #                         point_cloud = state_i['point_cloud_slow']
    #                     else:
    #                         point_cloud = None

    #                     output = model_predict(
    #                         self.model_args, self.predict_mode, self.model,
    #                         image=fast_image, prompt=prompt,
    #                         cur_robot_state=cur_robot_state,
    #                         slow_image=state_i.get('slow_image', None),
    #                         point_cloud=point_cloud,
    #                         input_ids=state_i.get('input_ids', None),
    #                         slow_latent_embedding=state_i.get('slow_latent_embedding', None)
    #                     )
    #                     refined_waypoints_local.append(np.array(output, dtype=np.float32))
    #                     ran_fast = True

    #                 # 推进该环境的慢系统步计数
    #                 state_i['slow_cnt'] += 1

    #             # 更新该环境处理到的最新时间步
    #             state_i['last_t'] = latest_t

    #         # 兜底：极端情况下未跑快系统，仍用最新帧与缓存执行一次
    #         if not ran_fast and latest_t >= 0:
    #             frame = traj[latest_t]
    #             fast_image = frame['rgb']
    #             prompt = frame['instruction']
    #             cur_robot_state = frame['sensors'].get('state', None)

    #             if self.load_pointcloud and self.pointcloud_pos == 'fast':
    #                 point_cloud = frame.get('point_cloud', None)
    #             elif self.load_pointcloud and self.pointcloud_pos == 'slow':
    #                 point_cloud = state_i['point_cloud_slow']
    #             else:
    #                 point_cloud = None

    #             output = model_predict(
    #                 self.model_args, self.predict_mode, self.model,
    #                 image=fast_image, prompt=prompt,
    #                 cur_robot_state=cur_robot_state,
    #                 slow_image=state_i.get('slow_image', None),
    #                 point_cloud=point_cloud,
    #                 input_ids=state_i.get('input_ids', None),
    #                 slow_latent_embedding=state_i.get('slow_latent_embedding', None)
    #             )
    #             refined_waypoints_local.append(np.array(output, dtype=np.float32))

    #     refined_waypoints = transform_to_world(
    #         np.array(refined_waypoints_local, dtype=np.float32), episodes
    #     )
    #     return refined_waypoints





class FiSImageProcessor:

    def __init__(self, vision_backbone=None):
        self.vision_backbone = vision_backbone
        
    def preprocess(self, images, return_tensors='pt'):
        """
        使用Fast-in-Slow模型的图像转换器预处理图像
        
        Args:
            images: numpy数组，形状为[batch_size, height, width, channels]
            return_tensors: 返回张量的类型，'pt'表示PyTorch
            
        Returns:
            包含'pixel_values'键的字典，值为预处理后的图像张量
        """

        
        # 处理每张图像
        processed_images = []
        for img in images:
            # 转换为PIL图像
            pil_img = Image.fromarray(img)
            
            # 使用Fast-in-Slow的图像转换器
            if self.vision_backbone and hasattr(self.vision_backbone, 'image_transform'):
                # 使用DinoSigLIPImageTransform处理图像
                processed_img = self.vision_backbone.image_transform(pil_img)
            else:
                # 回退到基本处理
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                processed_img = transform(pil_img)
            
            processed_images.append(processed_img)
        
        # 堆叠成批次
        if return_tensors == 'pt':
            if isinstance(processed_images[0], dict):
                # 如果是字典形式（DinoSigLIP可能返回多个特征）
                result = {}
                for key in processed_images[0].keys():
                    result[key] = torch.stack([img[key] for img in processed_images])
                pixel_values = result
            else:
                pixel_values = torch.stack(processed_images)
        else:
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        
        return {'pixel_values': pixel_values}