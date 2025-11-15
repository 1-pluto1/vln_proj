#
# 在 training/strategies/single_device_strategy.py 中
#


from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Optional, Union, Callable

from overwatch import initialize_overwatch

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, IterableDataset, Dataset

from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import CausalLMOutputWithPast

from training.strategies.base_strategy import TrainingStrategy

# 导入你的模型、工具等
from models.vlms import PrismaticVLM
from models import FiSvla
from training.metrics import Metrics, VLAMetrics
from util import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from tqdm import tqdm
import math

# 这是一个全新的类，专门用于单卡、非分布式训练
class SingleDeviceStrategy(TrainingStrategy):
    
    def __init__(
        self,
        vlm: Union[PrismaticVLM, FiSvla],
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        repeated_diffusion_steps: int = 4,
        **_: str,
    ) -> None:
        # 首先，调用基类的 __init__，传入所有必需的参数
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            repeated_diffusion_steps=repeated_diffusion_steps,
        )

        # === 单卡策略的关键 ===
        
        # 1. 定义我们的设备
        #    我们信任 CUDA_VISIBLE_DEVICES 已经设置，所以 device_id 总是 0
        self.device = torch.device(f"cuda:{self.device_id}")

        # 2. 覆盖基类中错误的梯度累积计算
        #    基类会除以 overwatch.world_size()，我们必须修正它
        overwatch = initialize_overwatch(__name__) # 获取 overwatch
        assert overwatch.world_size() == 1, "SingleDeviceStrategy 必须在非分布式模式下运行 (world_size=1)"
        
        # 修正梯度累积步数计算（移除world_size除法）
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size
        
        # 初始化训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        overwatch.info(f"初始化 SingleDeviceStrategy，将在 {self.device} 上运行。")
        overwatch.info(f"模型参数: {sum(p.numel() for p in vlm.parameters()):,}")
        overwatch.info(f"可训练参数: {sum(p.numel() for p in vlm.parameters() if p.requires_grad):,}")
        overwatch.info(f"梯度累积步数: {self.grad_accumulation_steps}")


    # --- 实现抽象方法 (Implement Abstract Methods) ---

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        """
        在单卡上设置模型、优化器和调度器。
        """
        overwatch = initialize_overwatch(__name__)
        
        # 创建运行目录
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # === 关键：将模型移动到我们的单个设备 ===
        self.vlm.to(self.device)

        # 只优化可训练参数
        trainable_params = []
        self.trainable_module_keys = set()
        
        for name, param in self.vlm.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                self.trainable_module_keys.add(name)
        
            
        overwatch.info(f"可训练参数数量: {len(trainable_params)}")

        # 计算训练步数
        steps_per_epoch = n_train_examples // self.global_batch_size
        total_steps = self.epochs * steps_per_epoch if self.max_steps is None else self.max_steps
        warmup_steps = int(self.warmup_ratio * total_steps)
        
        overwatch.info(f"训练设置: 轮数={self.epochs}, 总步数={total_steps}, 预热步数={warmup_steps}")
        overwatch.info(f"学习率={self.learning_rate}, 权重衰减={self.weight_decay}, 调度器类型={self.lr_scheduler_type}")

        # 设置优化器
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # 设置学习率调度器
        if self.lr_scheduler_type == "cosine_with_warmup":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        else:
            raise ValueError(f"不支持的调度器类型: {self.lr_scheduler_type}")
            
        overwatch.info("模型、优化器和 LR 调度器已在单个 GPU 上设置完毕。")

    def clip_grad_norm(self) -> None:
        """ 单卡的梯度裁剪，只处理可训练参数 """
        # 只裁剪可训练参数的梯度
        trainable_params = [p for p in self.vlm.parameters() if p.requires_grad]
        
        if not trainable_params:
            return
            
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=self.max_grad_norm
        )
        
        # 检查梯度是否为NaN或inf
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            overwatch = initialize_overwatch(__name__)
            overwatch.warning(f"检测到异常梯度范数: {grad_norm}, 跳过优化器步骤")
            
        return grad_norm

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """
        单卡的模型保存，无需 FSDP 状态。
        """
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"step-{global_step}.pt"

        # (这里你需要定义你想保存的状态)
        # 示例：
        state_dict = {
            "model": self.vlm.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "train_loss": train_loss,
        }
        
        torch.save(state_dict, checkpoint_path)
        overwatch.info(f"单卡检查点已保存至 {checkpoint_path}")
        
        # === 关键：没有 dist.barrier() ===


    def load_optimizer_and_scheduler(self, checkpoint_path: str) -> None:
        """
        从检查点加载优化器和调度器状态。
        """
        overwatch = initialize_overwatch(__name__)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载优化器状态
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            overwatch.info(f"优化器状态已从 {checkpoint_path} 加载")
        else:
            overwatch.warning(f"检查点 {checkpoint_path} 中未找到优化器状态")
        
        # 加载学习率调度器状态
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            overwatch.info(f"学习率调度器状态已从 {checkpoint_path} 加载")
        else:
            overwatch.warning(f"检查点 {checkpoint_path} 中未找到学习率调度器状态")

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """
        从检查点恢复训练。
        """
        overwatch = initialize_overwatch(__name__)
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            if "model" in checkpoint:
                model_state = checkpoint["model"]
                # 如果检查点只包含可训练参数，我们需要合并当前模型状态
                if all(k in self.trainable_module_keys for k in model_state.keys()):
                    current_state = self.vlm.state_dict()
                    current_state.update(model_state)
                    model_state = current_state
                
                self.vlm.load_state_dict(model_state)
                overwatch.info(f"模型状态已从 {checkpoint_path} 加载")
            else:
                raise KeyError(f"检查点 {checkpoint_path} 中未找到模型状态")
            
            # 加载优化器和调度器状态
            self.load_optimizer_and_scheduler(checkpoint_path)
            
            # 获取恢复的步骤数
            global_step = checkpoint.get("global_step", 0)
            overwatch.info(f"成功恢复到步骤: {global_step}")
            return global_step
            
        except Exception as e:
            overwatch.error(f"从检查点恢复失败: {e}")
            raise


    # --- 重写基类方法 (Override Base Class Methods) ---

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """
        重写的训练循环，移除了所有分布式组件。
        """
        
        # === 关键修改 1：使用 RandomSampler ===
        # 不再需要 SplitModalitySampler 或 DistributedSampler
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # 单卡模式下简化处理，直接使用RandomSampler
            sampler = RandomSampler(dataset)
        else:
            sampler = RandomSampler(dataset)

        # === 关键修改 2：使用标准 DataLoader ===
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
        )

        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            self.epochs = 100

        # === 训练 ===
        status = metrics.get_status()
        overwatch = initialize_overwatch(__name__)
        overwatch.info(f"开始训练: stage={stage}, batch_strategy={batch_construction_strategy}, epochs={self.epochs}")
        
        # === 关键修改 3：移除 TQDM 的 disable=not overwatch.is_rank_zero() ===
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=False,
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                
                # === 关键修改 4：移除 sampler.set_epoch(epoch) ===
                # RandomSampler 不需要这个

                self.optimizer.zero_grad()

                for train_idx, batch in enumerate(dataloader):
                    
                    # === 关键修改 5：手动将数据移动到设备 ===
                    # FSDP 会自动做，我们必须手动做
                    try:
                        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    except AttributeError:
                        # 处理 batch 不是字典的情况
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
                
                # 检查必需键
                required_keys = ["input_ids", "attention_mask", "labels"]
                missing_keys = [key for key in required_keys if key not in batch]
                if missing_keys:
                    overwatch = initialize_overwatch(__name__)
                    overwatch.error(f"缺少必需的键: {missing_keys}")
                    overwatch.error(f"可用键: {list(batch.keys()) if isinstance(batch, dict) else 'batch 是列表'}")
                    raise KeyError(f"缺少必需的键: {missing_keys}")
                    
                    # Debug: Log available keys in the first few batches
                    if train_idx < 3:
                        overwatch = initialize_overwatch(__name__)
                        available_keys = list(batch.keys()) if isinstance(batch, dict) else "batch is list"
                        overwatch.info(f"Batch {train_idx} available keys: {available_keys}")


                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        # Build arguments dynamically based on available keys
                        vlm_args = {
                            "input_ids": batch["input_ids"],
                            "attention_mask": batch["attention_mask"],
                            "labels": batch["labels"],
                            "repeated_diffusion_steps": self.repeated_diffusion_steps
                        }
                        if "pixel_values" in batch:
                            vlm_args["pixel_values"] = batch["pixel_values"]
                        if "multimodal_indices" in batch:
                            vlm_args["multimodal_indices"] = batch["multimodal_indices"]
                        if "action_masks" in batch:
                            vlm_args["action_masks"] = batch["action_masks"]
                        if "proprio" in batch:
                            vlm_args["proprio"] = batch["proprio"]
                        
                        output = self.vlm(**vlm_args)
                        # 从输出中提取损失
                        loss = output.loss if hasattr(output, 'loss') else output[0] if isinstance(output, tuple) else output

                    metrics.commit(loss=loss)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)
                        self.clip_grad_norm() # 调用我们自己的简单实现
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        progress.update()
                        progress.set_description(status)
        
        # 训练完成后的最终检查点保存
        if self.max_steps is None:
            final_loss = loss.item() if 'loss' in locals() else None
            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, final_loss)
            overwatch = initialize_overwatch(__name__)
            overwatch.info(f"训练完成，保存最终检查点 (步骤: {metrics.global_step}, 轮数: {epoch})")

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        use_diff: bool = False,
        repeated_diffusion_steps: int = 4,
        ar_diff_loss: bool = False,
        model_save_num: int = 1,
    ) -> None:

        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"

        # 创建 DataLoader =>> 设置 num_workers 为 0; RLDS loader 处理并行！
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === 训练 ===
        status = metrics.get_status()
        overwatch = initialize_overwatch(__name__)
        overwatch.info(f"开始VLA训练: use_diff={use_diff}, save_interval={save_interval}, repeated_diffusion_steps={repeated_diffusion_steps}")
        
        with tqdm(
            total=(self.epochs * math.ceil(len(dataloader) / self.grad_accumulation_steps)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=False,
        ) as progress:
            self.vlm.train()
            self.optimizer.zero_grad()

            for train_idx, batch in enumerate(dataloader):

                new_batch = {}
                for k, v in batch.items():
                    if k == "pixel_values" and isinstance(v, dict):
                        new_batch[k] = {sub_k: sub_v.to(self.device) for sub_k, sub_v in v.items()}
                    elif isinstance(v, torch.Tensor):
                        new_batch[k] = v.to(self.device)
                    else:
                        new_batch[k] = v
                batch = new_batch

                required_keys = ["input_ids", "attention_mask", "labels", "pixel_values"]
                if use_diff:
                    required_keys.append("actions")
                
                missing_keys = [key for key in required_keys if key not in batch]
                if missing_keys:
                    overwatch = initialize_overwatch(__name__)
                    overwatch.error(f"Missing required keys in batch: {missing_keys}")
                    overwatch.error(f"Available keys: {list(batch.keys()) if isinstance(batch, dict) else 'batch is list'}")
                    raise KeyError(f"Missing required keys: {missing_keys}")
                
                # Debug: Log available keys in the first few batches
                if train_idx < 3:
                    overwatch = initialize_overwatch(__name__)
                    available_keys = list(batch.keys()) if isinstance(batch, dict) else "batch is list"
                    overwatch.info(f"Batch {train_idx} available keys: {available_keys}")
                    if isinstance(batch, dict):
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                overwatch.info(f"  - {key}: shape {value.shape}")
                            elif isinstance(value, dict):
                                overwatch.info(f"  - {key}: dict with keys {list(value.keys())}")
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, torch.Tensor):
                                        overwatch.info(f"    - {sub_key}: shape {sub_value.shape}")
                            else:
                                overwatch.info(f"  - {key}: type {type(value)}")

                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    if use_diff:
                        # Build arguments dynamically based on available keys
                        vlm_args = {
                            "input_ids": batch["input_ids"],
                            "attention_mask": batch["attention_mask"],
                            "actions": batch["actions"],
                            "labels": batch["labels"],
                            "output_hidden_states": True,
                            "repeated_diffusion_steps": repeated_diffusion_steps,
                            "use_diff": True
                        }
                        
                        # Add optional arguments if they exist in batch
                        if "proprio" in batch:
                            vlm_args["proprio"] = batch["proprio"]
                        if "pixel_values" in batch:
                            vlm_args["pixel_values"] = batch["pixel_values"]
                        if "point_cloud" in batch:
                            vlm_args["point_cloud"] = batch["point_cloud"]
                        if "action_masks" in batch:
                            vlm_args["action_masks"] = batch["action_masks"]
                        
                        output = self.vlm(**vlm_args)
                        loss = output.loss if hasattr(output, 'loss') else output[0] if isinstance(output, tuple) else output
                        if ar_diff_loss:
                            loss = loss + (output.loss if hasattr(output, 'loss') else output[0] if isinstance(output, tuple) else output)
                    else:
                        # Build arguments dynamically based on available keys
                        vlm_args = {
                            "input_ids": batch["input_ids"],
                            "attention_mask": batch["attention_mask"],
                            "labels": batch["labels"],
                            "repeated_diffusion_steps": self.repeated_diffusion_steps
                        }
                        if "pixel_values" in batch:
                            vlm_args["pixel_values"] = batch["pixel_values"]
                        if "multimodal_indices" in batch:
                            vlm_args["multimodal_indices"] = batch["multimodal_indices"]
                        if "proprio" in batch:
                            vlm_args["proprio"] = batch["proprio"]
                        if "action_masks" in batch:
                            vlm_args["action_masks"] = batch["action_masks"]
                        
                        output = self.vlm(**vlm_args)
                        # 从输出中提取损失
                        loss = output.loss if hasattr(output, 'loss') else output[0] if isinstance(output, tuple) else output
                        vlm_args = {
                            "input_ids": batch["input_ids"],
                            "attention_mask": batch["attention_mask"],
                            "labels": batch["labels"]
                        }
                        
                        # Add optional arguments if they exist in batch
                        if "pixel_values" in batch:
                            vlm_args["pixel_values"] = batch["pixel_values"]
                        
                        # [Contract] self.vlm.forward() 必须自动计算 `loss` 并返回！
                        output: CausalLMOutputWithPast = self.vlm(**vlm_args)
                        loss = output.loss if hasattr(output, 'loss') else output[0] if isinstance(output, tuple) else output

                # 提交 Loss =>> 反向传播！
                metrics.commit(loss=loss)
                
                normalized_loss = loss / self.grad_accumulation_steps
                normalized_loss.backward()

                # === 梯度步骤 ===
                # Step =>> 只有在完成梯度累积时才执行
                if (train_idx + 1) % self.grad_accumulation_steps == 0:
                    # 裁剪梯度
                    grad_norm = self.clip_grad_norm()

                    # 优化器和 LR 调度器步骤
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # 更新指标
                    metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                    status = metrics.push()
                    
                    # 检查是否达到最大步数
                    if self.max_steps is not None and metrics.global_step >= self.max_steps:
                        overwatch = initialize_overwatch(__name__)
                        overwatch.info(f"达到最大步数 {self.max_steps}，训练结束")
                        self.save_checkpoint(metrics.run_dir, metrics.global_step, 0, loss.item())
                        return
                        
                    # 检查检查点保存间隔
                    if metrics.global_step % save_interval == 0:
                        overwatch = initialize_overwatch(__name__)
                        overwatch.info(f"保存检查点 (步骤: {metrics.global_step})")
                        self.save_checkpoint(metrics.run_dir, metrics.global_step, 0, loss.item(), only_trainable=not save_full_model)
                    
                    progress.update()
                    progress.set_description(status)
                    
                    # 使用完成的梯度步骤数计算 epoch 值
                    div = (len(vla_dataset) // self.global_batch_size) if (len(vla_dataset) // self.global_batch_size) != 0 else 1
                    epoch = (metrics.global_step + 1) // div

                    # 推送指标
                    metrics.commit(update_step_time=True, global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                    status = metrics.push()

                    # 保存检查点逻辑
                    target_steps = self.epochs * math.ceil(len(dataloader) / self.grad_accumulation_steps)
                    start_last_tenth = target_steps * 7 // 10
                    for i in range(1, model_save_num + 1):
                        checkpoint_step = start_last_tenth + (target_steps - start_last_tenth) * i // model_save_num
                        if metrics.global_step == checkpoint_step:
                            self.save_checkpoint(
                                metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                            )

                    if metrics.global_step >= (self.epochs * math.ceil(len(dataloader) / self.grad_accumulation_steps)):
                        return
                
                progress.update()
                progress.set_description(status)