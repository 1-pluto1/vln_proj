# parameter
export PYTHONPATH=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/transformers:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/timm:$PYTHONPATH 
# export HF_HOME=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/.cache/huggingface  # 指定 HuggingFace 缓存目录
export NCCL_DEBUG=INFO  # NCCL 调试日志级别（INFO：输出基本通信信息）
# export NCCL_SOCKET_IFNAME=eth  # 指定网络接口名（如使用以太网时可设置为 eth）
export NCCL_P2P_LEVEL=NVL  # NCCL 点对点通信优先级（NVL：优先使用 NVLink）
export NCCL_TIMEOUT=1800  # NCCL 通信超时时间（秒）

export CUDA_VISIBLE_DEVICES=0,1,2,3


# UAV特定参数
USE_UAV_DATASET=true  # 是否启用 UAV 数据集

TRAINING_MODE='async'        # 训练/推理控制模式（async：异步调度，见 models/vlms/prismatic.py）
LLM_MIDDLE_LAYER=30          # LLM 中间层位置（用于特征融合的层索引）
FUTURE_ACTION_STEPS=0        # 未来动作窗口大小（预测未来动作步数）
LOAD_POINTCLOUD=false        # 是否加载点云数据
LOAD_STATE=true              # 是否加载机器人状态（如位姿、速度等）
POINTCLOUD_POS='fast'        # 点云接入位置（fast/slow）
DIFFUSION_STEPS=100          # Diffusion 推理步数
ACTION_TOKENIZER_EXIST=true  # 是否启用动作分词器（关闭则不使用 AR 损失）
USE_DIFF=true                # 是否使用 Diffusion 模型
AR_DIFF_LOSS=true            # 是否启用自回归（AR）+ Diffusion 联合损失
REPEATED_DIFFUSION_STEPS=4   # 每步重复的 Diffusion 次数
CLASS_DROPOUT_PROB=0.0       # 类别 dropout 概率
FREEZE_VISON=true           # 是否冻结视觉骨干网络
FREEZE_LLM=false            # 是否冻结语言模型骨干网络
UNFREEZE_LAST_LLM_LAYER=false
SLOW_FAST_RATIO=1_4          # 慢-快系统帧比（例如 1:4）
ACTION_CHUNK=7               # 动作 chunk 大小（序列打包长度）
LANG_SUBGOALS_EXIST=true     # 是否包含语言子目标（分阶段指令）

SETTING="test_multi_key_STATE_${LOAD_STATE}_ACTION_CHUNK_${ACTION_CHUNK}_SLOW_FAST_RATIO_${SLOW_FAST_RATIO}_ddim${DIFFUSION_STEPS}_PC${LOAD_POINTCLOUD}_POS${POINTCLOUD_POS}_${TRAINING_MODE}_withARloss${LOAD_POINTCLOUD}_slow_fast_[after]_[-1]_${LLM_MIDDLE_LAYER}_fisvla_pretrain_window${FUTURE_ACTION_STEPS}"  # 实验设置字符串（基于上述变量拼接）

DATA_MIX=uav_dataset  # 数据混合方案
TASK=uav_dataset  # 任务/数据集子目录名
BATCH_SIZE=3         # 每设备的批大小
EPOCHS=1           # 训练轮数
LEARNING_RATE=2e-5   # 学习率
ACTION_DIM=6         # 动作空间维度
CAMERA_VIEW="head_slow,head_fast"  # 使用的相机视角组合

DATA_ROOT="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets"  # 数据根目录
EXP_ROOT=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/exp       # 实验输出目录
MODEL_SAVE_NUM=3  # 每次保存的模型数（保留的 checkpoint 数量）

NUM_GPUS=4     # 每节点使用的 GPU 数
NODES=1        # 节点数
MASTER_ADDR="127.0.0.1"  # 主节点地址
NODE_RANK=0    # 当前节点序号

# 创建日志目录
LOG_DIR="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/logs"
mkdir -p ${LOG_DIR}

# 生成简洁的日志文件名
LOG_FILE="${LOG_DIR}/train_${TASK}_STATE${LOAD_STATE}_PC${LOAD_POINTCLOUD}_ddim${DIFFUSION_STEPS}_$(date '+%m%d_%H%M').log"

echo "训练日志将保存到: ${LOG_FILE}"
echo "开始训练..."

torchrun --standalone --nnodes=$NODES --nproc-per-node=$NUM_GPUS --node_rank=$NODE_RANK --master_addr=${MASTER_ADDR} --master_port=29500 scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${DATA_MIX} \
  --vla.base_vlm prism-dinosiglip-224px+7b \
  --need_to_sub 0 \
  --vla.expected_world_size $((${NUM_GPUS} * ${NODES})) \
  --vla.per_device_batch_size ${BATCH_SIZE} \
  --vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE})) \
  --vla.learning_rate ${LEARNING_RATE} \
  --vla.epochs ${EPOCHS} \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --vla.unfreeze_last_llm_layer ${UNFREEZE_LAST_LLM_LAYER} \
  --data_root_dir ${DATA_ROOT} \
  --run_root_dir ${EXP_ROOT} \
  --run_id exp_${TASK}_${SETTING} \
  --image_aug false \
  --wandb_project 'vln_proj' \
  --wandb_entity 'vln_soolab_zhaoyang' \
  --save_interval 100 \
  --action_dim ${ACTION_DIM} \
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --ar_diff_loss ${AR_DIFF_LOSS} \
  --is_resume False \
  --llm_middle_layer ${LLM_MIDDLE_LAYER} \
  --camera_view ${CAMERA_VIEW} \
  --training_mode ${TRAINING_MODE} \
  --load_pointcloud ${LOAD_POINTCLOUD} \
  --diffusion_steps ${DIFFUSION_STEPS} \
  --model_save_num ${MODEL_SAVE_NUM} \
  --pointcloud_pos ${POINTCLOUD_POS} \
  --action_chunk ${ACTION_CHUNK} \
  --load_state ${LOAD_STATE} \
  --lang_subgoals_exist ${LANG_SUBGOALS_EXIST} \
  --pretrained_checkpoint "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/ckpt/fis/models--haosad--fisvla/snapshots/d5f66f3accd4c043742cfe348431f58c5609e950/fisvla_pretrain.pt" 2>&1 | tee -a ${LOG_FILE}

echo "训练完成！日志文件: ${LOG_FILE}"