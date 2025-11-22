#!/usr/bin/env bash

# Set project root directory
ROOT_DIR="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow"

# Set Python path
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/transformers:${ROOT_DIR}/timm:${PYTHONPATH}"

# Create logs directory
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server_$(date '+%Y%m%d_%H%M%S').log"

# Set GPU device
export CUDA_VISIBLE_DEVICES="${4:-0}"

echo "Starting FiS API server" | tee -a "${LOG_FILE}"

# Start Python server
python -u "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/scripts/fis_api_server.py" \
    --host "127.0.0.1" \
    --port "5000" \
    --model-path "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/exp/exp_uav_dataset_test_multi_key_STATE_true_ACTION_CHUNK_1_SLOW_FAST_RATIO_1_4_ddim100_PCfalse_POSfast_async_withARlossfalse_slow_fast_[after]_[-1]_30_fisvla_pretrain_window0/checkpoints/step-006724-epoch-00-loss=2.4647.pt" \
    --training_mode 'async' \
    --slow-fast-ratio 4 \
    --cuda 0 \
    --training-diffusion-steps 100 \
    --llm_middle_layer 30 \
    --use-diff 1 \
    --use-ar 0 \
    --use_robot_state 1 \
    --model-action-steps 0 \
    --max-steps 10 \
    --num-episodes 20 \
    --load-pointcloud 1 \
    --pointcloud-pos "fast" \
    --action-chunk 1 \
    --sparse 1 \
    --angle_delta 0 \
    --action-dim 6 \
    --lang_subgoals_exist 1 \
    --ddim-steps 4 \
  2>&1 | tee -a "${LOG_FILE}"