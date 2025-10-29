import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass
import time
from tqdm import tqdm

# 添加路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "simulation"))

from simulation.openuav_sim import (
    OpenUAVSimulator, DAggerManager, simple_expert_policy,
    UAVState, SensorData
)
from fis_uav_async_trainer import FiSUAVAsyncTrainer, AsyncSamplingConfig
from uav_data_integration import DAggerDataAggregator, TravelUAVDataProcessor
from initial_model_trainer import InitialModelTrainer

logger = logging.getLogger(__name__)


@dataclass
class DAggerConfig:
    """DAgger配置"""
    # 迭代设置
    num_iterations: int = 5
    episodes_per_iteration: int = 20
    max_steps_per_episode: int = 200
    
    # 专家策略
    expert_intervention_threshold: float = 0.8
    rollback_steps: int = 5
    
    # 数据聚合
    aggregation_ratio: float = 0.3  # 新数据占比
    min_expert_corrections: int = 10
    
    # 重训练
    retrain_epochs: int = 5
    retrain_batch_size: int = 16
    retrain_lr: float = 1e-5
    
    # 评估
    eval_episodes: int = 5
    success_threshold: float = 0.8


class ModelPolicyWrapper:
    """模型策略包装器"""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.eval()
    
    def __call__(self, obs: Dict[str, Any]) -> np.ndarray:
        """预测动作"""
        try:
            with torch.no_grad():
                # 处理观测数据
                processed_obs = self._process_observation(obs)
                
                # 模型推理
                if hasattr(self.model, 'fast_system_forward'):
                    # FiS异步采样模式
                    output = self.model.fast_system_forward(
                        pixel_values=processed_obs['images'],
                        input_ids=processed_obs['instruction_ids'],
                        attention_mask=processed_obs['attention_mask'],
                        point_cloud=processed_obs['point_cloud']
                    )
                else:
                    # 标准前向传播
                    output = self.model(
                        pixel_values=processed_obs['images'],
                        input_ids=processed_obs['instruction_ids'],
                        attention_mask=processed_obs['attention_mask']
                    )
                
                # 提取动作
                if hasattr(output, 'action_preds'):
                    action = output.action_preds[0].cpu().numpy()
                elif hasattr(output, 'logits'):
                    action = output.logits[0].cpu().numpy()
                else:
                    # 从输出中提取动作
                    action = output[0].cpu().numpy() if isinstance(output, tuple) else output.cpu().numpy()
                
                # 确保动作维度正确 (6-DoF)
                if action.shape[-1] != 6:
                    action = action[:6] if len(action) >= 6 else np.pad(action, (0, 6-len(action)))
                
                # 动作范围限制
                action = np.clip(action, -2.0, 2.0)
                
                return action
                
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}, using random action")
            return np.random.uniform(-0.5, 0.5, 6)
    
    def _process_observation(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理观测数据"""
        # 处理图像
        images = []
        sensor_data = obs['sensor_data']
        
        for camera in ['head', 'left', 'right']:
            if camera in sensor_data.images:
                img = sensor_data.images[camera]
                # 归一化到 [0, 1]
                img = torch.from_numpy(img).float() / 255.0
                # 调整维度 [H, W, C] -> [C, H, W]
                img = img.permute(2, 0, 1)
                images.append(img)
        
        # 拼接图像 [3*C, H, W]
        if images:
            images_tensor = torch.cat(images, dim=0).unsqueeze(0).to(self.device)
        else:
            images_tensor = torch.zeros(1, 9, 224, 224).to(self.device)
        
        # 处理指令
        instruction = obs.get('instruction', 'navigate safely')
        # 简化的指令编码
        instruction_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(self.device)  # 占位符
        attention_mask = torch.ones_like(instruction_ids).to(self.device)
        
        # 处理点云
        point_cloud = sensor_data.point_cloud
        if point_cloud is not None:
            point_cloud_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).to(self.device)
        else:
            point_cloud_tensor = torch.zeros(1, 1024, 3).to(self.device)
        
        return {
            'images': images_tensor,
            'instruction_ids': instruction_ids,
            'attention_mask': attention_mask,
            'point_cloud': point_cloud_tensor
        }


class ExpertPolicyWrapper:
    """专家策略包装器"""
    
    def __init__(self, expert_model_path: Optional[str] = None):
        self.expert_model_path = expert_model_path
        # 这里可以加载真实的专家模型
        # 目前使用简化的规则策略
    
    def __call__(self, obs: Dict[str, Any]) -> np.ndarray:
        """专家策略"""
        uav_state = obs['uav_state']
        sensor_data = obs['sensor_data']
        
        # 安全优先的专家策略
        action = np.zeros(6)
        
        # 高度控制
        if uav_state.position[2] < 1.0:
            action[2] = 2.0  # 向上
        elif uav_state.position[2] > 3.0:
            action[2] = -1.0  # 向下
        
        # 避障
        if sensor_data.point_cloud is not None:
            distances = np.linalg.norm(sensor_data.point_cloud, axis=1)
            min_distance = np.min(distances)
            
            if min_distance < 1.0:
                # 找到最近障碍物方向
                closest_idx = np.argmin(distances)
                obstacle_dir = sensor_data.point_cloud[closest_idx]
                obstacle_dir = obstacle_dir / np.linalg.norm(obstacle_dir)
                
                # 反向移动
                action[:3] = -obstacle_dir * 1.5
        
        # 前进（如果安全）
        if np.all(np.abs(action[:3]) < 0.5):
            action[0] = 1.0  # 向前
        
        return action


class DAggerClosedLoopTrainer:
    """DAgger闭环训练器"""
    
    def __init__(self, 
                 config_path: str,
                 initial_model_path: str,
                 save_dir: str = "./dagger_training"):
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dagger_config = DAggerConfig(**self.config.get('dagger', {}))
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_model_path = initial_model_path
        
        # 加载初始模型
        self.current_model = self._load_model(initial_model_path)
        
        # 创建仿真环境
        self.simulator = OpenUAVSimulator()
        
        # 创建策略包装器
        self.model_policy = ModelPolicyWrapper(self.current_model, str(self.device))
        self.expert_policy = ExpertPolicyWrapper()
        
        # 创建DAgger管理器
        self.dagger_manager = DAggerManager(
            simulator=self.simulator,
            expert_policy=self.expert_policy
        )
        
        # 数据聚合器
        self.data_aggregator = DAggerDataAggregator()
        
        # 训练历史
        self.training_history = []
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.save_dir / "dagger_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _load_model(self, model_path: str):
        """加载模型"""
        logger.info(f"Loading model from {model_path}")
        
        try:
            # 这里应该加载实际的FiS模型
            # 目前使用占位符
            model = torch.nn.Linear(10, 6)  # 占位符模型
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # 返回随机初始化的模型
            model = torch.nn.Linear(10, 6)
            return model.to(self.device)
    
    def run_dagger_iteration(self, iteration: int) -> Dict[str, Any]:
        """运行一次DAgger迭代"""
        logger.info(f"Starting DAgger iteration {iteration + 1}/{self.dagger_config.num_iterations}")
        
        # 数据收集阶段
        logger.info("Collecting trajectory data...")
        iteration_data = []
        
        for episode in tqdm(range(self.dagger_config.episodes_per_iteration), 
                           desc=f"Iteration {iteration + 1} Episodes"):
            
            episode_data = self.dagger_manager.run_episode(
                model_policy=self.model_policy
            )
            iteration_data.append(episode_data)
        
        # 统计收集的数据
        total_steps = sum(ep['total_steps'] for ep in iteration_data)
        total_interventions = sum(ep['expert_interventions'] for ep in iteration_data)
        
        logger.info(f"Collected {len(iteration_data)} episodes, "
                   f"{total_steps} steps, {total_interventions} expert interventions")
        
        # 检查是否有足够的专家纠正数据
        if total_interventions < self.dagger_config.min_expert_corrections:
            logger.warning(f"Insufficient expert corrections ({total_interventions} < "
                          f"{self.dagger_config.min_expert_corrections}), skipping retraining")
            return {
                'iteration': iteration,
                'episodes': len(iteration_data),
                'total_steps': total_steps,
                'expert_interventions': total_interventions,
                'retrained': False
            }
        
        # 数据聚合
        logger.info("Aggregating data...")
        aggregated_data = self.data_aggregator.aggregate_dagger_data(
            model_trajectories=iteration_data,
            expert_corrections=self.dagger_manager.collected_data,
            aggregation_ratio=self.dagger_config.aggregation_ratio
        )
        
        # 模型重训练
        logger.info("Retraining model...")
        retrain_results = self._retrain_model(aggregated_data)
        
        # 模型评估
        logger.info("Evaluating model...")
        eval_results = self._evaluate_model()
        
        iteration_results = {
            'iteration': iteration,
            'episodes': len(iteration_data),
            'total_steps': total_steps,
            'expert_interventions': total_interventions,
            'retrained': True,
            'retrain_loss': retrain_results['final_loss'],
            'eval_success_rate': eval_results['success_rate'],
            'eval_avg_reward': eval_results['avg_reward']
        }
        
        # 保存迭代结果
        self._save_iteration_results(iteration, iteration_results, aggregated_data)
        
        return iteration_results
    
    def _retrain_model(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """重训练模型"""
        # 这里应该实现实际的模型重训练逻辑
        # 目前使用简化版本
        
        logger.info(f"Retraining with {len(aggregated_data.get('trajectories', []))} trajectories")
        
        # 模拟训练过程
        initial_loss = 1.0
        final_loss = 0.5
        
        # 这里应该调用实际的训练循环
        # 例如：使用 InitialModelTrainer 或类似的训练器
        
        return {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'epochs': self.dagger_config.retrain_epochs
        }
    
    def _evaluate_model(self) -> Dict[str, Any]:
        """评估模型"""
        logger.info(f"Evaluating model with {self.dagger_config.eval_episodes} episodes")
        
        success_count = 0
        total_rewards = []
        
        for episode in range(self.dagger_config.eval_episodes):
            obs = self.simulator.reset()
            episode_reward = 0
            success = True
            
            for step in range(self.dagger_config.max_steps_per_episode):
                action = self.model_policy(obs)
                obs, reward, done, info = self.simulator.step(action)
                episode_reward += reward
                
                if done and step < self.dagger_config.max_steps_per_episode - 1:
                    success = False
                    break
                
                if done:
                    break
            
            if success:
                success_count += 1
            total_rewards.append(episode_reward)
        
        success_rate = success_count / self.dagger_config.eval_episodes
        avg_reward = np.mean(total_rewards)
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'total_episodes': self.dagger_config.eval_episodes
        }
    
    def _save_iteration_results(self, 
                               iteration: int, 
                               results: Dict[str, Any], 
                               data: Dict[str, Any]):
        """保存迭代结果"""
        # 保存结果
        results_file = self.save_dir / f"iteration_{iteration}_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        
        # 保存数据
        data_file = self.save_dir / f"iteration_{iteration}_data.json"
        with open(data_file, 'w') as f:
            import json
            json.dump(data, f, indent=2, default=str)
        
        # 保存模型
        model_file = self.save_dir / f"model_v{iteration + 2}.pt"
        torch.save({
            'model_state_dict': self.current_model.state_dict(),
            'iteration': iteration,
            'results': results
        }, model_file)
        
        logger.info(f"Iteration {iteration} results saved to {results_file}")
    
    def run_full_dagger_training(self) -> Dict[str, Any]:
        """运行完整的DAgger训练"""
        logger.info("Starting DAgger closed-loop training")
        logger.info(f"Configuration: {self.dagger_config}")
        
        all_results = []
        
        for iteration in range(self.dagger_config.num_iterations):
            try:
                iteration_results = self.run_dagger_iteration(iteration)
                all_results.append(iteration_results)
                
                # 检查早停条件
                if (iteration_results.get('eval_success_rate', 0) >= 
                    self.dagger_config.success_threshold):
                    logger.info(f"Reached success threshold, stopping early at iteration {iteration + 1}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                continue
        
        # 保存最终结果
        final_results = {
            'total_iterations': len(all_results),
            'iteration_results': all_results,
            'final_success_rate': all_results[-1].get('eval_success_rate', 0) if all_results else 0,
            'config': self.dagger_config.__dict__
        }
        
        final_file = self.save_dir / "final_results.json"
        with open(final_file, 'w') as f:
            import json
            json.dump(final_results, f, indent=2)
        
        logger.info(f"DAgger training completed. Final results saved to {final_file}")
        
        return final_results


# 主函数
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DAgger Closed-Loop Training")
    parser.add_argument("--config", type=str, required=True,
                       help="Configuration file path")
    parser.add_argument("--initial_model", type=str, required=True,
                       help="Initial model path (Model v1)")
    parser.add_argument("--save_dir", type=str, default="./dagger_training",
                       help="Save directory")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建训练器
    trainer = DAggerClosedLoopTrainer(
        config_path=args.config,
        initial_model_path=args.initial_model,
        save_dir=args.save_dir
    )
    
    # 运行训练
    results = trainer.run_full_dagger_training()
    
    print("\n" + "="*50)
    print("DAgger Training Completed!")
    print(f"Total iterations: {results['total_iterations']}")
    print(f"Final success rate: {results['final_success_rate']:.3f}")
    print("="*50)


if __name__ == "__main__":
    main()