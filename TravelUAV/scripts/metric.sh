#!/bin/bash

# ROOT_DIR='/path/to/your/root/eval/result/dir' # ROOT_DIR="./closeloop_eval/"
# ANALYSIS_LIST="eval dir list" # ANALYSIS_LIST="baseline baseline2"
# PATH_TYPE_LIST="full easy hard" # full easy hard
# OUTPUT_DIR='./metrics' # Directory to save metrics JSON file

#ROOT_DIR='/home/gentoo/asus/liusq/UAV_VLN/TravelUAV/data/eval_test_1'
ROOT_DIR='/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/eval_test/fisllm_eval/'
ANALYSIS_LIST="./"
PATH_TYPE_LIST="full easy hard"
OUTPUT_DIR="$ROOT_DIR"  # Output directory under root_dir

# CUDA_VISIBLE_DEVICES=0 python3 ./AirVLN/utils/metric.py \
CUDA_VISIBLE_DEVICES=6 python /home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV/utils/metric.py \
    --root_dir $ROOT_DIR \
    --analysis_list $ANALYSIS_LIST \
    --path_type_list $PATH_TYPE_LIST \
    --output_dir $OUTPUT_DIR


