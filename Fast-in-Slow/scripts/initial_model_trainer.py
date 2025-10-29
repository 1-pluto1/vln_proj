import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import time

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "TravelUAV"))

from models.vlas.fisvla import FiSvla
from vla.materialize import get_vla_dataset_and_collator
from training.materialize import get_train_strategy
from training.metrics import VLAMetrics
from overwatch import initialize_overwatch

logger = logging.getLogger(__name__)
overwatch = initialize_overwatch(__name__)


def smart_tokenizer_and_embedding_resize(tokenizer, model):
    """Resize tokenizer and embeddings to include <BOD>/<EOD> tokens."""
    num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["<BOD>", "<EOD>"]})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class InitialTrainingConfig:
    """初始训练配置"""
    # 数据配置
    dataset_path: str
    batch_size: int = 8
    num_workers: int = 4
    sequence_length: int = 10
    
    # 训练配置
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_ratio: float = 0.1
    train_strategy: str = "fsdp-shard-grad-op"

    # 异步采样/扩散配置
    repeated_diffusion_steps: int = 8
    use_diff: bool = False
    ar_diff_loss: bool = False

    # 数据构造细节
    load_all_data_for_training: bool = True
    future_action_window_size: int = 15
    past_action_window_size: int = 0
    action_tokenizer_exist: bool = False
    need_to_sub: int = 0
    camera_view: str = ""
    load_pointcloud: bool = False
    action_chunk: int = 1
    lang_subgoals_exist: bool = False
    use_uav_dataset: bool = True

    # 保存/日志配置
    save_dir: str = "./runs/initial_model"
    save_frequency: int = 1000
    eval_frequency: int = 500
    run_id: Optional[str] = None
    trackers: Tuple[str, ...] = ("jsonl", "wandb")
    wandb_project: str = ""
    wandb_entity: str = ""


class InitialModelTrainer:
    """初始模型训练器"""
    
    def __init__(self, 
                 model: FiSvla,
                 config: InitialTrainingConfig,
                 device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # 创建目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.save_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    # 替换数据加载为 FiS UAVDataset + Collator
    def load_dataset(self) -> Tuple[torch.utils.data.Dataset, Any, Any]:
        """加载数据集（FiS UAVDataset + Collator）"""
        dataset_root = Path(self.config.dataset_path)

        # 从 FiSvla 获取必要的变换与 tokenizer
        image_transform = self.model.vision_backbone.get_image_transform()
        tokenizer = self.model.llm_backbone.get_tokenizer()
        prompt_builder_fn = self.model.llm_backbone.prompt_builder_fn
        default_image_resolution = self.model.vision_backbone.default_image_resolution

        dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
            data_root_dir=dataset_root,
            data_mix="train",  
            image_transform=image_transform,
            tokenizer=tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            default_image_resolution=default_image_resolution,
            padding_side="right",
            predict_stop_token=True,
            shuffle_buffer_size=100_000,
            train=True,
            episodic=False,
            image_aug=False,
            future_action_window_size=self.config.future_action_window_size,
            past_action_window_size=self.config.past_action_window_size,
            load_all_data_for_training=self.config.load_all_data_for_training,
            action_tokenizer_exist=self.config.action_tokenizer_exist,
            need_to_sub=self.config.need_to_sub,
            camera_view=self.config.camera_view,
            load_pointcloud=self.config.load_pointcloud,
            action_chunk=self.config.action_chunk,
            lang_subgoals_exist=self.config.lang_subgoals_exist,
            use_uav_dataset=self.config.use_uav_dataset,
        )
        return dataset, action_tokenizer, collator

    def train(self) -> Path:
        logger.info("Starting initial model training with FiS strategy...")

        # 设置设备与运行目录
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id := (self.device.index or 0))
            torch.cuda.empty_cache()
        else:
            device_id = 0

        run_id = self.config.run_id or f"initial+n{1}+b{self.config.batch_size}+x{int(time.time())}"
        run_dir = self.save_dir / run_id
        os.makedirs(run_dir / "checkpoints", exist_ok=True)

        # 扩展 tokenizer 以支持 <BOD>/<EOD>
        smart_tokenizer_and_embedding_resize(
            tokenizer=self.model.llm_backbone.get_tokenizer(),
            model=self.model.llm_backbone.llm,
        )

        # 加载数据
        vla_dataset, _, collator = self.load_dataset()

        # 初始化训练策略
        train_strategy = get_train_strategy(
            train_strategy=self.config.train_strategy,
            vlm=self.model,
            device_id=device_id,
            stage="vla-train",
            epochs=self.config.num_epochs,
            max_steps=None,
            global_batch_size=self.config.batch_size * max(overwatch.world_size(), 1),
            per_device_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.gradient_clip_norm,
            lr_scheduler_type="cosine",
            warmup_ratio=self.config.warmup_ratio,
            enable_gradient_checkpointing=True,
            enable_mixed_precision_training=True,
            reduce_in_full_precision=False,
            worker_init_fn=None,
        )
        train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

        # 初始化指标
        metrics = VLAMetrics(
            self.config.trackers,
            run_id,
            run_dir,
            asdict(self.config),
            wandb_project=self.config.wandb_project,
            wandb_entity=self.config.wandb_entity,
            resume_step=None,
            resume_epoch=None,
        )

        # 运行 VLA 训练循环
        train_strategy.run_vla_training(
            vla_dataset,
            collator,
            metrics,
            save_interval=self.config.save_frequency,
            use_diff=self.config.use_diff,
            repeated_diffusion_steps=self.config.repeated_diffusion_steps,
            ar_diff_loss=self.config.ar_diff_loss,
            model_save_num=1,
        )

        metrics.finalize()
        logger.info("Training completed via FiS strategy. Checkpoints and logs saved.")
        return run_dir / "checkpoints"


def create_initial_trainer(model: FiSvla,
                          config: InitialTrainingConfig,
                          device: torch.device) -> InitialModelTrainer:
    """创建初始训练器"""
    return InitialModelTrainer(model, config, device)


# 示例使用
if __name__ == "__main__":
    # 配置
    config = InitialTrainingConfig(
        dataset_path="/path/to/traveluav/12k/dataset",
        batch_size=4,
        num_epochs=20,
        learning_rate=1e-4,
        save_dir="./runs/initial_model_test",
        use_uav_dataset=True,
        use_diff=True,
        ar_diff_loss=False,
    )
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 这里应该加载实际的FiS模型
    # model = FiSvla(...)
    model = None
    
    if model is not None:
        # 创建训练器
        trainer = create_initial_trainer(model, config, device)
        
        # 开始训练
        final_model_path = trainer.train()
        print(f"Training completed! Checkpoints saved at: {final_model_path}")
    else:
        print("Model not loaded - this is a demonstration of the training pipeline")
        print("The initial training system now reuses FiS strategy:")
        print("1. get_vla_dataset_and_collator for TravelUAV dataset")
        print("2. FSDP-based training strategy via run_vla_training")
        print("3. VLAMetrics integrated for logging")
        print("4. Checkpoints saved under runs/<run_id>/checkpoints/")