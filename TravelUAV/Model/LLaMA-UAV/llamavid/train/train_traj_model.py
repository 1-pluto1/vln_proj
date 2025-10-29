import copy
from dataclasses import dataclass, field
import glob
import json
import math
import random
from typing import Optional
import numpy as np
import torch
import tqdm
import transformers
from llamavid.model.vis_traj_arch import VisionTrajectoryGenerator
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from scipy.spatial.transform import Rotation as R
import wandb


    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    freeze_backbone: bool = field(default=True)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    use_custom_loss: Optional[bool] = field(default=False),
    use_scale_waypoint_loss: Optional[bool] = field(default=False),


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data json."})
    dataset_path: str = field(default=None,
                            metadata={"help": "Path to the dataset dir."})
    val_data_path: str = field(default=None,
                           metadata={"help": "Path to the eval data."}) 

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lr_multi: Optional[str] = field(default=None)
    learning_rate: Optional[float] = field(default=5e-4)
    drop_step: Optional[int] = field(default=5)
    drop_rate: Optional[float] = field(default=0.1)
    bs: Optional[int] = field(default=128)
    epoch: Optional[int] = field(default=6)
    
def to_eularian_angles(q):
    x,y,z,w = q
    ysqr = y * y
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)
    return (pitch, roll, yaw)

def euler_to_rotation_matrix(e):
    rotation = R.from_euler('xyz', e, degrees=False)
    return rotation.as_matrix()

def project_this_state2target_state_axis(this_state, target_state):
    start_pos = target_state[0:3]
    start_eular = to_eularian_angles(target_state[3:])  # (pitch, roll, yaw)
    this_pos = this_state[0:3]
    this_eular = to_eularian_angles(this_state[3:])
    delta_pos = np.asarray(this_pos) - np.asarray(start_pos)
    delta_eular = np.asarray(this_eular) - np.asarray(start_eular)
    rot = euler_to_rotation_matrix(start_eular) 
    delta_pos = rot.T @ delta_pos
    return delta_pos


def generate_vision_tower_config(vision_tower, image_processor):
    default_vision_config={
    "model_type": "clip",
    "hidden_act": "silu",
    "hidden_size": 4096,
    "image_aspect_ratio": "square",
    "image_grid_pinpoints": None,
    "image_processor": "./llamavid/processor/clip-patch14-224",
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "max_token": 2048,
    "mm_hidden_size": 1408,
    "mm_projector_type": "mlp2x_gelu",
    "mm_use_im_patch_token": False,
    "mm_use_im_start_end": False,
    "mm_vision_select_feature": "patch",
    "mm_vision_select_layer": -2,
    "mm_vision_tower": "./model_zoo/LAVIS/eva_vit_g.pth",
    "torch_dtype": "float16"
    }
    default_vision_config['image_processor'] = image_processor
    default_vision_config['mm_vision_tower'] = vision_tower
    cf_path = os.path.join(os.path.split(vision_tower)[0], 'config.json')
    with open(cf_path, 'w') as f:
        json.dump(default_vision_config, f, indent=2)
    return cf_path

class CustomDataset(Dataset):
    def __init__(self, root_file, dataset_path, image_processor):
        self.root_file = root_file
        self.dataset_path = dataset_path
        self.image_processor = image_processor
        with open(self.root_file, 'r') as f:
            self.data_list = json.load(f)
            random.shuffle(self.data_list)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        img_path = item['img']
        img_path = os.path.join(self.dataset_path, img_path)
        ori_points = item['waypoints']
        waypoints = copy.deepcopy(ori_points)
        img = Image.open(img_path)
        waypoints_list = []
        for waypoint in waypoints:
            waypoint = project_this_state2target_state_axis(waypoint, ori_points[0])
            waypoints_list.append(waypoint)
        img = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'].squeeze(0)
        waypoints_list = torch.tensor(np.array(waypoints_list))
        label = waypoints_list[1:]
        input = {'img': img, 'target': waypoints_list[5], 'ori': ori_points}  
        return input, label

eval_step = 0
train_step = 0
def evaluate(model, val_dataset):
    global eval_step
    
    dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=0)
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in tqdm.tqdm(enumerate(dataloader)):
            loss, outputs = model(inputs, labels)
            wandb.log({"val_loss": loss.item()})
    eval_step = eval_step + i
    model.train()
    
parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


bs = training_args.bs
drop_rate = training_args.drop_rate
base_lr = training_args.learning_rate
num_epochs = training_args.epoch

name = f'bs_{bs}_drop_{drop_rate}_lr_{base_lr}_with_elu_balance'

os.makedirs(training_args.output_dir, exist_ok=True)


wandb.init(
    project="traj_predictor",
    name=name,
    config={
    "learning_rate": base_lr,
    "architecture": "ViT",
    "epochs": num_epochs,
    }
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vision_config = generate_vision_tower_config(model_args.vision_tower, model_args.image_processor)
config = transformers.AutoConfig.from_pretrained(vision_config, trust_remote_code=True)
model = VisionTrajectoryGenerator(config)
vision_tower = model.get_vision_tower()
vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
image_processor = vision_tower.image_processor
dataset = CustomDataset(data_args.data_path, data_args.dataset_path, image_processor)
val_dataset = CustomDataset(data_args.val_data_path, data_args.dataset_path, image_processor)
optimizer = optim.AdamW(model.parameters(), lr=base_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_args.drop_step, gamma=training_args.drop_rate)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4, prefetch_factor=2)

model.to(device=device, dtype=torch.bfloat16)
model.train()
ema_loss = 0
for epoch in range(num_epochs):
    progress_bar = tqdm.tqdm(total=len(dataset)//bs, desc=f"Epoch {epoch}/{num_epochs}")
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        loss, outputs = model(inputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)
        ema_loss = ema_loss * 0.9 + loss.item() * 0.1
        if loss.item() > 20:
            print('Warning: loss is greater than 20, loss: ', loss.item())
        optimizer.step()
        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item(), ema_loss=ema_loss)
        wandb.log({"loss": loss.item(), 'lr': scheduler.get_lr()})
        train_step += 1
    torch.save(model.state_dict(), os.path.join(training_args.output_dir, f"model_{epoch}.pth"))
    # evaluate(model, val_dataset)
    progress_bar.close()
    scheduler.step()   

wandb.finish()
