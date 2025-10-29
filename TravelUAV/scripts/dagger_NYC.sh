#!/bin/bash
# change the dataset_path to your own path

CUDA_VISIBLE_DEVICES=0 python -u /home/liusq/TravelUAV/vln_proj/TravelUAV/src/vlnce_src/dagger.py \
    --run_type collect \
    --collect_type dagger \
    --name FiSLLM \
    --gpu_id 4 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80002 \
    --batchSize 1 \
    --dagger_it 1 \
    --dagger_p 0.4 \
    --maxWaypoints 200 \
    --activate_maps NYCEnvironmentMegapa \
    --dataset_path /mnt/data5/airdrone/dataset/replay_data_log0.1_image0.5/ \
    --dagger_save_path /home/liusq/TravelUAV/vln_proj/TravelUAV/data/dagger_data \
    --model_path /home/liusq/TravelUAV/vln_proj/Fast-in-Slow/ckpt/models--haosad--fisvla \
    --model_base /model_zoo/vicuna-7b-v1.5 \
    --vision_tower /model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor /home/liusq/TravelUAV/vln_proj/Fast-in-Slow/ckpt/models--haosad--fisvla/llamavid/processor/clip-patch14-224 \ 
    --traj_model_path /home/liusq/TravelUAV/vln_proj/Fast-in-Slow/ckpt/models--haosad--fisvla/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4 \
    --train_json_path /home/liusq/TravelUAV/vln_proj/TravelUAV/data/uav_dataset/trainset.json \
    --map_spawn_area_json_path /home/liusq/TravelUAV/vln_proj/TravelUAV/data/meta/map_spawnarea_info.json \
    --object_name_json_path /home/liusq/TravelUAV/vln_proj/TravelUAV/data/meta/object_description.json \
    --groundingdino_config /home/liusq/TravelUAV/vln_proj/TravelUAV/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path /home/liusq/TravelUAV/vln_proj/TravelUAV/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth


