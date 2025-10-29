import argparse
import multiprocessing
import torch
import os
import tqdm
import numpy as np
from transformers.models.clip import CLIPImageProcessor
from typing import Dict, Optional, Sequence, List

from PIL import Image

RGB_FOLDER = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']

def arg_parse():
  parser = argparse.ArgumentParser(description="split video clip")
  parser.add_argument("--root_dir",
                      default='/path/to/your/dataset',
                      help='path to your dataset root dir')
  parser.add_argument("--map_list",
                        default=['NewYorkCity', 'ModernCityMap', 'NYCEnvironmentMegapa', 'TropicalIsland', 'ModularPark', 'Carla_Town01', 'Carla_Town02', 'Carla_Town03', 'Carla_Town04','Carla_Town05', 'Carla_Town06', 'Carla_Town07', 'Carla_Town10HD', 'Carla_Town15',
                                 'BattlefieldKitDesert', 'BrushifyCountryRoads', 'BrushifyForestPack', 'BrushifyUrban', 'Japanese_Street', 'London_Street', 'NordicHarbour', 'WesterTown'],
                      nargs="+",
                      help='processed map name')
  parser.add_argument("--workers",
                      default=16,
                      help='multiprocessing workers num')
  opt = parser.parse_args()
  return opt

clip_config = {
  "crop_size": {
    "height": 224,
    "width": 224
  },
  "do_center_crop": True,
  "do_convert_rgb": True,
  "do_normalize": True,
  "do_rescale": True,
  "do_resize": True,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 224
  }
}
args = arg_parse()
processer = CLIPImageProcessor(**clip_config)

if __name__ == "__main__":
  def worker(traj_dir):
    traj_camera_list = []
    for idx,camera_name in enumerate(RGB_FOLDER):
      traj_camera_list.append(sorted([os.path.join(traj_dir, camera_name, filename) for filename in os.listdir(os.path.join(traj_dir,camera_name))]))
    traj_frames = []
    for idx in range(len(traj_camera_list[0])):
      batch = []
      for iid in range(len(RGB_FOLDER)):
          batch.append(traj_camera_list[iid][idx])
      traj_frames.append(batch)
    traj_imgs = []
    for frame_imgs in traj_frames:
      images = [Image.open(img_path).convert('RGB') for img_path in frame_imgs]
      images = np.stack(images, axis=0)
      traj_imgs.append(images)
    imgs = np.array(traj_imgs).reshape(-1, 256, 256, 3)
    imgs = processer.preprocess(imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16)
    torch.save(imgs, os.path.join(traj_dir, 'rgb_imgs.tensor'))
  
  for map_name in args.map_list:
    directory_path = os.path.join(args.root_dir, map_name)
    print(directory_path)
    traj_list = []
    for traj in tqdm.tqdm(os.listdir(directory_path)):
      traj_dir = os.path.join(directory_path, traj)
      traj_list.append(traj_dir)
    with multiprocessing.Pool(args.workers) as p:
      r = list(tqdm.tqdm(p.imap_unordered(worker, traj_list), total=len(traj_list)))
    print(directory_path, 'finished.')