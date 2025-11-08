#!/bin/bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${script_dir}/../logs"
mkdir -p "${log_dir}"
ts="$(date '+%Y%m%d-%H%M%S')"
log_file="${log_dir}/dagger_NYC_${ts}.log"
exec > >(tee -a "${log_file}") 2>&1
echo "日志文件: ${log_file}"
# change the dataset_path to your own path
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
# export PYTHONPATH="/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow:$PYTHONPATH"
# export PYTHONPATH=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/transformers:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/timm:$PYTHONPATH
# export PYTHONPATH=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow/transformers:$PYTHONPATH
unset PYTHONPATH
export PYTHONPATH=/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/Fast-in-Slow:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV:/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/src:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0 python -u /home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/src/vlnce_src/dagger.py \
    --run_type collect \
    --collect_type dagger \
    --name FiSLLM \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80002 \
    --batchSize 1 \
    --dagger_it 1 \
    --dagger_p 0.4 \
    --maxWaypoints 200 \
    --activate_maps NYCEnvironmentMegapa \
    --dataset_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/data/TravelUAV_unzip/ \
    --dagger_save_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/datasets/traveluav_data/dagger_data \
    --model_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/ckpt/fis/models--haosad--fisvla/snapshots/d5f66f3accd4c043742cfe348431f58c5609e950/fisvla_pretrain.pt \
    --train_json_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/data/TravelUAV_data_json/data/uav_dataset/trainset.json \
    --map_spawn_area_json_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/data/TravelUAV_data_json/data/meta/map_spawnarea_info.json \
    --object_name_json_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/data/TravelUAV_data_json/data/meta/object_description.json \
    --groundingdino_config /home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path /home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth


