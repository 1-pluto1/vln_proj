#!/bin/bash
# change the root_dir and dataset_path to your own path

root_dir=/home/airport/airdrone/TravelUAV/TravelUAV # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV
CUDA_VISIBLE_DEVICES=0 python -m llamavid.train.train_traj_model \
    --model_name_or_path $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --vision_tower $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --output_dir $model_dir/work_dirs/train_traj_model_bs128_drop0.1_lr_5e-4 \
    --data_path $root_dir/data/traj_train/train_balance.json \
    --dataset_path /mnt/data5/airdrone/dataset/replay_data_log0.1_image0.5/ \
    --val_data_path $root_dir/data/traj_train/val_8s_8k.json \
    --learning_rate 5e-4 \
    --drop_rate 0.1 \
    --bs 128 \
    --epoch 6 \
    --bf16 True
