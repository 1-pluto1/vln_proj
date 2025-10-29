# LLaMA-UAV

## Contents
- [Install](#install)
- [Model](#model)
- [Preparation](#preparation)
- [Train](#train)
- [Evaluation](#evaluation)

## Install
Please follow the instructions below to install the required packages.

1. Install Package
```bash
conda create -n llamauav python=3.10 -y
conda activate llamauav
cd LLaMA-UAV
pip install -e .
```

2. Install additional packages for training cases
```bash
pip install ninja
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

## Preparation
### Pretrained Weights
We recommend users to download the pretrained weights from the following link [Vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5), [EVA-ViT-G](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), [QFormer-7b](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth), and put them in `model_zoo` following structure below.
```
LLaMA-UAV
├── model_zoo
│   ├── vicuna-7b-v1.5
│   ├── LAVIS
│   │   ├── eva_vit_g.pth
│   │   ├── instruct_blip_vicuna7b_trimmed.pth
```
### Dataset
<!-- training data prepara -->
1. Download the dataset split info from [here](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_data_json)
2. Download the dataset from [here](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV)
3. Generate the processed trajector data
```bash
python tools/generate_merged_json.py --root_dir <path to your dataset>
# By default, all maps are processed.
```
4. Generate processed multi-view camera image tensor
``` bash
python tools/preprocess_image2tensor.py --root_dir <path to your dataset>
# By default, all maps are processed.
```
The dataset partitioning details are saved in the `TravelUAV/data/uav_dataset`. 
## Train
Please make sure you download and organize the data following [Preparation](#preparation) before training. 

1. UAV navigation LLM
Verify the `root_dir`, `data_path`, and `output_dir` paths in the training script.

```bash
bash scripts/llm/train_uav_llm.sh.
```

2. Trajectory completion model

```bash
bash scripts/traj/train_traj_completion.sh
```
## Evaluation
### Checkpoint
LLaMA-UAV MLLM Model: Download the LLaMA-UAV model used in the paper from [here](https://huggingface.co/wangxiangyu0814/llama-uav-7b). 
Save it in the directory: `work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32`.

Trajectory Completion Model: Download the Trajectory Completion model from [here](https://huggingface.co/wangxiangyu0814/traveluav-traj-model). 
Save it in the directory: `work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4`.
### Close loop Evaluation

Please use the scripts in `TravelUAV/script/eval.sh` and follow this [Readme](../../README.md).
