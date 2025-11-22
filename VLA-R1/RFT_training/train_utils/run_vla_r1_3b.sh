set -x
ENGINE=${1:-vllm}

train_data="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/grpo_data/train.parquet"
val_data="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/grpo_data/val.parquet"
reward_function="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/src/vlnce_src/grpo_reward.py"
ref_model="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/exp/exp_uav_dataset_test_multi_key_STATE_true_ACTION_CHUNK_1_SLOW_FAST_RATIO_1_4_ddim100_PCfalse_POSfast_async_withARlossfalse_slow_fast_[after]_[-1]_30_fisvla_pretrain_window0/checkpoints/step-006724-epoch-00-loss=2.4647.pt"
save_path="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/ckpt/grpo"

cd RFT_training/verl
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_data} \
    data.val_files=${val_data} \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    reward_model.enable=False \
    custom_reward_function.path=${reward_function} \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=${ref_model} \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.default_local_dir=${save_path} \
    trainer.project_name='vla_r1' \
    trainer.experiment_name='vla_r1' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=50 $@
