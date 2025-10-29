import numpy as np
import torch
from PIL import Image
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor
import sys
sys.path.append('/home/liusq/TravelUAV/vln_proj/Fast-in-Slow')
from models import load_vla  # 从Fast-in-Slow导入模型加载函数

class TravelModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        self.tokenizer, self.model, self.image_processor = load_model(model_args)
        self.traj_model = load_traj_model(model_args)
        self.model.to(torch.bfloat16)
        self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        inputs = []
        rot_to_targets = []
        
        for i in range(len(episodes)):
            input_item, rot_to_target = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)
        batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)

        inputs_device = {k: v.to(self.model.device) for k, v in batch.items() 
            if 'prompts' not in k and 'images' not in k and 'historys' not in k}
        inputs_device['prompts'] = [item for item in batch['prompts']]
        inputs_device['images'] = [item.to(self.model.device) for item in batch['images']]
        inputs_device['historys'] = [item.to(device=self.model.device, dtype=self.model.dtype) for item in batch['historys']]
        inputs_device['orientations'] = inputs_device['orientations'].to(dtype=self.model.dtype)
        inputs_device['return_waypoints'] = True
        inputs_device['use_cache'] = False
        
        return inputs_device, rot_to_targets

    def run_llm_model(self, inputs):
        waypoints_llm = self.model(**inputs).cpu().to(dtype=torch.float32).numpy()
        waypoints_llm_new = []
        for waypoint in waypoints_llm:
            waypoint_new = waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
            waypoints_llm_new.append(waypoint_new)
        return np.array(waypoints_llm_new)

    def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets):
        inputs = prepare_data_to_traj_model(episodes, waypoints_llm_new, self.image_processor, rot_to_targets)
        waypoints_traj = self.traj_model(inputs, None)
        refined_waypoints = waypoints_traj.cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = transform_to_world(refined_waypoints, episodes)
        return refined_waypoints
    
    def eval(self):
        self.model.eval()
        self.traj_model.eval()
        
    def run(self, inputs, episodes, rot_to_targets):
        waypoints_llm_new = self.run_llm_model(inputs)
        refined_waypoints = self.run_traj_model(episodes, waypoints_llm_new, rot_to_targets)
        
        return refined_waypoints
    
    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones



class FiSModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        self.model = load_vla(
            model_args.model_path,
            load_for_training=False,
            future_action_window_size=1,  # 根据需要调整
            use_diff=1,
            diffusion_steps=10,  # 根据需要调整
            load_pointcloud=1 if hasattr(model_args, 'use_pointcloud') and model_args.use_pointcloud else 0,
            action_dim=6,  # UAV通常使用6DoF动作空间
        )
        self.tokenizer = self.model.vlm.llm_backbone.tokenizer
        self.image_processor = FiSImageProcessor(self.model.vision_backbone)
        self.traj_model = self.model
        self.model.to(torch.bfloat16)
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        inputs = []
        rot_to_targets = []
        
        for i in range(len(episodes)):
            input_item, rot_to_target = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)
        batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)

        inputs_device = {k: v.to(self.model.device) for k, v in batch.items() 
            if 'prompts' not in k and 'images' not in k and 'historys' not in k}
        inputs_device['prompts'] = [item for item in batch['prompts']]
        inputs_device['images'] = [item.to(self.model.device) for item in batch['images']]
        inputs_device['historys'] = [item.to(device=self.model.device, dtype=self.model.dtype) for item in batch['historys']]
        inputs_device['orientations'] = inputs_device['orientations'].to(dtype=self.model.dtype)
        inputs_device['return_waypoints'] = True
        inputs_device['use_cache'] = False
        
        return inputs_device, rot_to_targets

    # def run_llm_model(self, inputs):
    #     waypoints_llm = self.model(**inputs).cpu().to(dtype=torch.float32).numpy()
    #     waypoints_llm_new = []
    #     for waypoint in waypoints_llm:
    #         waypoint_new = waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
    #         waypoints_llm_new.append(waypoint_new)
    #     return np.array(waypoints_llm_new)

    # def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets):
    #     inputs = prepare_data_to_traj_model(episodes, waypoints_llm_new, self.image_processor, rot_to_targets)
    #     waypoints_traj = self.traj_model(inputs, None)
    #     refined_waypoints = waypoints_traj.cpu().to(dtype=torch.float32).numpy()
    #     refined_waypoints = transform_to_world(refined_waypoints, episodes)
    #     return refined_waypoints
    
    def eval(self):
        self.model.eval()
        self.traj_model.eval()
        
    def run(self, inputs, episodes, rot_to_targets):
        # 直接使用 Fast-in-Slow 模型的功能生成航点
        batch_size = len(episodes)
        refined_waypoints = []
        
        for i in range(batch_size):
            # 获取当前样本的图像和指令
            current_episode = episodes[i]
            current_images = inputs['images'][i] if isinstance(inputs['images'], list) else inputs['images'][i:i+1]
            current_prompt = inputs['prompts'][i] if isinstance(inputs['prompts'], list) else inputs['prompts'][i:i+1]
            
            # 获取当前位置和方向信息
            current_position = np.array(current_episode[-1]['sensors']['imu']['position'])
            
            # 第一步：使用慢系统生成潜在嵌入
            slow_latent_embedding = None
            input_ids = None
            with torch.inference_mode():
                if hasattr(self.model, 'slow_system_forward') and callable(self.model.slow_system_forward):
                    # 使用慢系统生成潜在嵌入
                    input_ids, slow_latent_embedding = self.model.slow_system_forward(
                        image_head_slow=current_images,
                        instruction=current_prompt,
                        unnorm_key=None
                    )
            
            # 第二步：使用快系统生成最终航点
            with torch.inference_mode():
                if hasattr(self.model, 'fast_system_forward') and callable(self.model.fast_system_forward):
                    # 使用快系统生成航点，并传入慢系统的潜在嵌入
                    actions = self.model.fast_system_forward(
                        image_head_fast=current_images,
                        slow_latent_embedding=slow_latent_embedding,
                        input_ids=input_ids,
                        instruction=current_prompt,
                        unnorm_key=None
                    )
                else:
                    # 备选方案：使用 predict_action 方法
                    actions, _ = self.model.predict_action(
                        image_head_slow=current_images,
                        image_head_fast=current_images,
                        instruction=current_prompt,
                        unnorm_key=None,
                        use_ddim=True,
                        num_ddim_steps=10,
                        cfg_scale=1.0,
                        predict_mode='diff'
                    )
                
                # 将动作转换为航点
                waypoint = actions[0]  # 取第一个动作作为航点
                
                # 应用旋转变换
                rot_matrix = rot_to_targets[i]
                if rot_matrix is not None:
                    waypoint_transformed = np.dot(rot_matrix, waypoint[:3])
                else:
                    waypoint_transformed = waypoint[:3]
                
                # 归一化并缩放航点
                norm = np.linalg.norm(waypoint_transformed)
                if norm > 1e-6:
                    waypoint_transformed = waypoint_transformed / norm * abs(waypoint[3])
                
                # 将航点添加到当前位置，得到目标位置
                target_waypoint = current_position + waypoint_transformed
                refined_waypoints.append(target_waypoint)
        
        return np.array(refined_waypoints)
    
    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones
        




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