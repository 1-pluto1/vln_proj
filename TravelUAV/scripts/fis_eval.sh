#!/bin/bash
# change the dataset_path to your own path

root_dir=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV
log_dir=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/logs/eval

# 创建日志目录
mkdir -p "$log_dir"

# 生成时间戳用于日志文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$log_dir/eval_ipc_${timestamp}.log"

# 让 Python 能找到项目根目录下的 `utils/` 和 `src/`
export PYTHONPATH="$model_dir:$root_dir:$PYTHONPATH"
export FIS_SERVER_URL="http://127.0.0.1:5000/predict"
# export PYTHONPATH=$model_dir:$root_dir:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/transformers:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/timm:$PYTHONPATH 


CUDA_VISIBLE_DEVICES=1 python -u /home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/src/vlnce_src/eval.py \
    --run_type eval \
    --name FiSIPC \
    --gpu_id 1 \
    --simulator_tool_port 30000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/data/TravelUAV_unzip \
    --eval_save_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/eval_test/fisllm_eval \
    --model_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/ckpt/fis/models--haosad--fisvla/snapshots/d5f66f3accd4c043742cfe348431f58c5609e950/fisvla_pretrain.pt \
    --eval_json_path $root_dir/data/TravelUAV_data_json/data/uav_dataset/test/test.json \
    --map_spawn_area_json_path $root_dir/data/TravelUAV_data_json/data/meta/map_spawnarea_info.json \
    --object_name_json_path $root_dir/data/TravelUAV_data_json/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth 2>&1 | tee "$log_file"
