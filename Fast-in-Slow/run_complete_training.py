import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
import torch
import time
from typing import Dict, Any

# 添加路径
sys.path.append(str(Path(__file__).parent))

from initial_model_trainer import InitialModelTrainer
from dagger_closed_loop_trainer import DAggerClosedLoopTrainer
from fis_uav_async_trainer import AsyncSamplingConfig

logger = logging.getLogger(__name__)


class CompleteTrainingPipeline:
    """完整训练管线"""
    
    def __init__(self, config_path: str, save_dir: str = "./complete_training"):
        self.config_path = config_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置日志
        self._setup_logging()
        
        # 阶段保存路径
        self.stage1_dir = self.save_dir / "stage1_initial_training"
        self.stage2_dir = self.save_dir / "stage2_dagger_training"
        
        self.stage1_dir.mkdir(parents=True, exist_ok=True)
        self.stage2_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.save_dir / "complete_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_stage1_initial_training(self) -> str:
        """阶段1：初始训练"""
        logger.info("="*60)
        logger.info("STAGE 1: Initial Training with TravelUAV 12k Dataset")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 创建初始训练器
            trainer = InitialModelTrainer(
                config=self.config,
                save_dir=str(self.stage1_dir)
            )
            
            # 运行训练
            logger.info("Starting initial model training...")
            training_results = trainer.train()
            
            # 获取最佳模型路径
            model_v1_path = training_results.get('best_model_path')
            if not model_v1_path or not os.path.exists(model_v1_path):
                # 使用最后保存的模型
                model_v1_path = str(self.stage1_dir / "model_final.pt")
                if not os.path.exists(model_v1_path):
                    raise FileNotFoundError("No trained model found")
            
            stage1_time = time.time() - start_time
            
            logger.info(f"Stage 1 completed in {stage1_time:.2f} seconds")
            logger.info(f"Model v1 saved to: {model_v1_path}")
            
            # 保存阶段1结果
            stage1_results = {
                'status': 'completed',
                'model_path': model_v1_path,
                'training_time': stage1_time,
                'training_results': training_results,
                'config': self.config.get('initial_training', {})
            }
            
            results_file = self.stage1_dir / "stage1_results.yaml"
            with open(results_file, 'w') as f:
                yaml.dump(stage1_results, f, indent=2)
            
            return model_v1_path
            
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            raise
    
    def run_stage2_dagger_training(self, model_v1_path: str) -> Dict[str, Any]:
        """阶段2：DAgger闭环训练"""
        logger.info("="*60)
        logger.info("STAGE 2: DAgger Closed-Loop Training")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 创建DAgger训练器
            trainer = DAggerClosedLoopTrainer(
                config_path=self.config_path,
                initial_model_path=model_v1_path,
                save_dir=str(self.stage2_dir)
            )
            
            # 运行DAgger训练
            logger.info("Starting DAgger closed-loop training...")
            dagger_results = trainer.run_full_dagger_training()
            
            stage2_time = time.time() - start_time
            
            logger.info(f"Stage 2 completed in {stage2_time:.2f} seconds")
            logger.info(f"Final success rate: {dagger_results.get('final_success_rate', 0):.3f}")
            
            # 保存阶段2结果
            stage2_results = {
                'status': 'completed',
                'initial_model_path': model_v1_path,
                'training_time': stage2_time,
                'dagger_results': dagger_results,
                'config': self.config.get('dagger', {})
            }
            
            results_file = self.stage2_dir / "stage2_results.yaml"
            with open(results_file, 'w') as f:
                yaml.dump(stage2_results, f, indent=2)
            
            return stage2_results
            
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """运行完整训练管线"""
        logger.info("Starting Complete FiS-UAV Training Pipeline")
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Save directory: {self.save_dir}")
        
        pipeline_start_time = time.time()
        
        try:
            # 阶段1：初始训练
            model_v1_path = self.run_stage1_initial_training()
            
            # 阶段间检查
            logger.info("Validating Model v1...")
            if not self._validate_model(model_v1_path):
                logger.warning("Model v1 validation failed, but continuing...")
            
            # 阶段2：DAgger训练
            stage2_results = self.run_stage2_dagger_training(model_v1_path)
            
            total_time = time.time() - pipeline_start_time
            
            # 汇总结果
            final_results = {
                'pipeline_status': 'completed',
                'total_training_time': total_time,
                'stage1': {
                    'model_v1_path': model_v1_path,
                    'save_dir': str(self.stage1_dir)
                },
                'stage2': stage2_results,
                'config': self.config
            }
            
            # 保存最终结果
            final_file = self.save_dir / "complete_training_results.yaml"
            with open(final_file, 'w') as f:
                yaml.dump(final_results, f, indent=2)
            
            # 打印总结
            self._print_training_summary(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            
            # 保存失败信息
            error_results = {
                'pipeline_status': 'failed',
                'error': str(e),
                'config': self.config
            }
            
            error_file = self.save_dir / "training_error.yaml"
            with open(error_file, 'w') as f:
                yaml.dump(error_results, f, indent=2)
            
            raise
    
    def _validate_model(self, model_path: str) -> bool:
        """验证模型"""
        try:
            if not os.path.exists(model_path):
                return False
            
            # 尝试加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 检查必要的键
            required_keys = ['model_state_dict']
            for key in required_keys:
                if key not in checkpoint:
                    logger.warning(f"Missing key in checkpoint: {key}")
                    return False
            
            logger.info("Model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _print_training_summary(self, results: Dict[str, Any]):
        """打印训练总结"""
        print("\n" + "="*80)
        print("🚁 FiS-UAV COMPLETE TRAINING PIPELINE SUMMARY 🚁")
        print("="*80)
        
        print(f"📊 Total Training Time: {results['total_training_time']:.2f} seconds")
        print(f"📁 Save Directory: {self.save_dir}")
        
        print("\n📈 STAGE 1 - Initial Training:")
        print(f"   ✅ Model v1: {results['stage1']['model_v1_path']}")
        print(f"   📂 Results: {results['stage1']['save_dir']}")
        
        print("\n🔄 STAGE 2 - DAgger Training:")
        stage2 = results['stage2']
        dagger_results = stage2.get('dagger_results', {})
        print(f"   🎯 Iterations: {dagger_results.get('total_iterations', 0)}")
        print(f"   📊 Final Success Rate: {dagger_results.get('final_success_rate', 0):.3f}")
        print(f"   📂 Results: {self.stage2_dir}")
        
        print("\n🎉 Training Pipeline Completed Successfully!")
        print("="*80)


def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        'initial_training': {
            'data': {
                'dataset_path': '/path/to/traveluav_12k',
                'batch_size': 16,
                'num_workers': 4,
                'sequence_length': 10
            },
            'model': {
                'use_diff': True,
                'ar_diff_loss': True,
                'diffusion_steps': 1000
            },
            'training': {
                'epochs': 10,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'warmup_steps': 1000
            },
            'async_sampling': {
                'system1_freq': 10,
                'system2_freq': 1,
                'context_window': 5
            }
        },
        'dagger': {
            'num_iterations': 5,
            'episodes_per_iteration': 20,
            'max_steps_per_episode': 200,
            'expert_intervention_threshold': 0.8,
            'rollback_steps': 5,
            'aggregation_ratio': 0.3,
            'min_expert_corrections': 10,
            'retrain_epochs': 5,
            'retrain_batch_size': 16,
            'retrain_lr': 1e-5,
            'eval_episodes': 5,
            'success_threshold': 0.8
        },
        'environment': {
            'world_bounds': [[-10, 10], [-10, 10], [0, 5]],
            'obstacle_density': 0.1,
            'wind_disturbance': 0.2
        },
        'hardware': {
            'device': 'cuda',
            'mixed_precision': True,
            'num_gpus': 1
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Complete FiS-UAV Training Pipeline")
    parser.add_argument("--config", type=str, 
                       help="Configuration file path")
    parser.add_argument("--save_dir", type=str, default="./complete_training",
                       help="Save directory for all training outputs")
    parser.add_argument("--create_config", action="store_true",
                       help="Create default configuration file")
    parser.add_argument("--stage", type=str, choices=['1', '2', 'all'], default='all',
                       help="Which stage to run (1: initial, 2: dagger, all: both)")
    parser.add_argument("--model_v1", type=str,
                       help="Path to Model v1 (for stage 2 only)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建默认配置
    if args.create_config:
        config_path = "fis_uav_complete_config.yaml"
        default_config = create_default_config()
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, indent=2)
        
        print(f"Default configuration created: {config_path}")
        print("Please edit the configuration file and run again with --config")
        return
    
    # 检查配置文件
    if not args.config:
        print("Error: Configuration file required. Use --create_config to create default config.")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    try:
        # 创建训练管线
        pipeline = CompleteTrainingPipeline(
            config_path=args.config,
            save_dir=args.save_dir
        )
        
        if args.stage == '1':
            # 只运行阶段1
            model_v1_path = pipeline.run_stage1_initial_training()
            print(f"\nStage 1 completed. Model v1 saved to: {model_v1_path}")
            
        elif args.stage == '2':
            # 只运行阶段2
            if not args.model_v1:
                print("Error: --model_v1 required for stage 2")
                return
            
            if not os.path.exists(args.model_v1):
                print(f"Error: Model v1 not found: {args.model_v1}")
                return
            
            stage2_results = pipeline.run_stage2_dagger_training(args.model_v1)
            print(f"\nStage 2 completed. Success rate: {stage2_results['dagger_results']['final_success_rate']:.3f}")
            
        else:
            # 运行完整管线
            results = pipeline.run_complete_pipeline()
            print(f"\nComplete training pipeline finished successfully!")
            print(f"Results saved to: {pipeline.save_dir}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()