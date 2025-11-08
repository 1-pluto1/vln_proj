# parameter
export PYTHONPATH=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/transformers:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/timm:$PYTHONPATH  # 设置 Python 模块搜索路径，确保可导入 Fast-in-Slow、transformers、timm
export HF_HOME=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/.cache/huggingface  # 指定 HuggingFace 缓存目录
export NCCL_DEBUG=INFO  # NCCL 调试日志级别（INFO：输出基本通信信息）
# export NCCL_SOCKET_IFNAME=eth  # 指定网络接口名（如使用以太网时可设置为 eth）
export NCCL_P2P_LEVEL=NVL  # NCCL 点对点通信优先级（NVL：优先使用 NVLink）
export NCCL_TIMEOUT=1800  # NCCL 通信超时时间（秒）
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定可见的 GPU 列表

# UAV特定参数
USE_UAV_DATASET=true  # 是否启用 UAV 数据集

TRAINING_MODE='async'        # 训练/推理控制模式（async：异步调度，见 models/vlms/prismatic.py）
LLM_MIDDLE_LAYER=30          # LLM 中间层位置（用于特征融合的层索引）
FUTURE_ACTION_STEPS=0        # 未来动作窗口大小（预测未来动作步数）
LOAD_POINTCLOUD=true         # 是否加载点云数据
LOAD_STATE=true              # 是否加载机器人状态（如位姿、速度等）
POINTCLOUD_POS='fast'        # 点云接入位置（fast/slow）
DIFFUSION_STEPS=100          # Diffusion 推理步数
ACTION_TOKENIZER_EXIST=true  # 是否启用动作分词器（关闭则不使用 AR 损失）
USE_DIFF=true                # 是否使用 Diffusion 模型
AR_DIFF_LOSS=true            # 是否启用自回归（AR）+ Diffusion 联合损失
REPEATED_DIFFUSION_STEPS=4   # 每步重复的 Diffusion 次数
CLASS_DROPOUT_PROB=0.0       # 类别 dropout 概率
FREEZE_VISON=false           # 是否冻结视觉骨干网络
FREEZE_LLM=false             # 是否冻结语言模型骨干网络
SLOW_FAST_RATIO=1_4          # 慢-快系统帧比（例如 1:4）
ACTION_CHUNK=1               # 动作 chunk 大小（序列打包长度）
LANG_SUBGOALS_EXIST=true     # 是否包含语言子目标（分阶段指令）

SETTING="test_multi_key_STATE_${LOAD_STATE}_ACTION_CHUNK_${ACTION_CHUNK}_SLOW_FAST_RATIO_${SLOW_FAST_RATIO}_ddim${DIFFUSION_STEPS}_PC${LOAD_POINTCLOUD}_POS${POINTCLOUD_POS}_${TRAINING_MODE}_withARloss${LOAD_POINTCLOUD}_slow_fast_[after]_[-1]_${LLM_MIDDLE_LAYER}_fisvla_pretrain_window${FUTURE_ACTION_STEPS}"  # 实验设置字符串（基于上述变量拼接）

DATA_MIX=UAVDataset  # 数据混合方案
TASK=12tasks_selected_keyframe_s1f4_pc_fast_sparsechunk_1_0518  # 任务/数据集子目录名
BATCH_SIZE=6         # 每设备的批大小
EPOCHS=300           # 训练轮数
LEARNING_RATE=2e-5   # 学习率
ACTION_DIM=6         # 动作空间维度
CAMERA_VIEW="head_slow,head_fast"  # 使用的相机视角组合

DATA_ROOT="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/rlds_data"  # 数据根目录
EXP_ROOT=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/exp       # 实验输出目录
MODEL_SAVE_NUM=3  # 每次保存的模型数（保留的 checkpoint 数量）

NUM_GPUS=4     # 每节点使用的 GPU 数
NODES=1        # 节点数
MASTER_ADDR="127.0.0.1"  # 主节点地址
NODE_RANK=0    # 当前节点序号

# 启动分布式训练（torchrun 配置）
torchrun --nnodes=$NODES --nproc-per-node=$NUM_GPUS --node_rank=$NODE_RANK --master_addr=${MASTER_ADDR} --master_port=29500 scripts/train.py \
  # 模型类型/配置（Prismatic + DinoSigLIP + OXE + Diffusion）
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  # 数据混合方案
  --vla.data_mix ${DATA_MIX} \
  # 基础 VLM 模型
  --vla.base_vlm prism-dinosiglip-224px+7b \
  # 内部开关（保持为 0）
  --need_to_sub 0 \
  # 预期全局进程数
  --vla.expected_world_size $((${NUM_GPUS} * ${NODES})) \
  # 每设备批大小
  --vla.per_device_batch_size ${BATCH_SIZE} \
  # 全局批大小
  --vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE})) \
  # 学习率
  --vla.learning_rate ${LEARNING_RATE} \
  # 训练轮数
  --vla.epochs ${EPOCHS} \
  # 是否冻结视觉骨干
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  # 是否冻结 LLM 骨干
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  # 数据目录
  --data_root_dir ${DATA_ROOT}/${TASK} \
  # 输出根目录
  --run_root_dir ${EXP_ROOT} \
  # 运行 ID / 实验名
  --run_id exp_${TASK}_${SETTING} \
  # 是否进行图像增强
  --image_aug false \
  # WandB 项目名
  --wandb_project '<wandb_project>' \
  # WandB 团队/用户
  --wandb_entity '<wandb_entity>' \
  # 保存间隔（step）
  --save_interval 100 \
  # 动作维度
  --action_dim ${ACTION_DIM} \
  # 重复 Diffusion 步数
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  # 动作分词器是否存在
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  # 未来动作窗口大小
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  # 类别 dropout 概率
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  # 启用 Diffusion
  --use_diff ${USE_DIFF} \
  # 启用 AR+Diff 损失
  --ar_diff_loss ${AR_DIFF_LOSS} \
  # 是否从检查点恢复
  --is_resume False \
  # LLM 中间层位置
  --llm_middle_layer ${LLM_MIDDLE_LAYER} \
  # 视角设置
  --camera_view ${CAMERA_VIEW} \
  # 训练模式
  --training_mode ${TRAINING_MODE} \
  # 加载点云
  --load_pointcloud ${LOAD_POINTCLOUD} \
  # Diffusion 步数
  --diffusion_steps ${DIFFUSION_STEPS} \
  # 保存模型份数
  --model_save_num ${MODEL_SAVE_NUM} \
  # 点云位置（fast/slow）
  --pointcloud_pos ${POINTCLOUD_POS} \
  # 动作 chunk 大小
  --action_chunk ${ACTION_CHUNK} \
  # 加载机器人状态
  --load_state ${LOAD_STATE} \
  # 语言子目标是否存在
  --lang_subgoals_exist ${LANG_SUBGOALS_EXIST} \
  # 预训练检查点目录
  --pretrained_checkpoint "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/ckpts/models--haosad--fisvla"