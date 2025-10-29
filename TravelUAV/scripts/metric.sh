#!/bin/bash

# ROOT_DIR='/path/to/your/root/eval/result/dir' # ROOT_DIR="./closeloop_eval/"
# ANALYSIS_LIST="eval dir list" # ANALYSIS_LIST="baseline baseline2"
# PATH_TYPE_LIST="full easy hard" # full easy hard

ROOT_DIR='/mnt/data5/airdrone/dataset/eval_closeloop'
ANALYSIS_LIST="rebuttal_history_info"
PATH_TYPE_LIST="full easy hard"

# CUDA_VISIBLE_DEVICES=0 python3 ./AirVLN/utils/metric.py \
CUDA_VISIBLE_DEVICES=0 python3 /home/airport/airdrone/TravelUAV/AirVLN/utils/metric.py \
    --root_dir $ROOT_DIR \
    --analysis_list $ANALYSIS_LIST \
    --path_type_list $PATH_TYPE_LIST
