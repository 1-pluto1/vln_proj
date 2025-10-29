import argparse
import cProfile
import copy
import json
import math
import os
import tqdm
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

random.seed = 1

RGB_FOLDER = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']
DEPTH_FOLDER = [name + '_depth' for name in RGB_FOLDER]
clip_merged_file_name = 'merged_data.json'

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
    opt = parser.parse_args()
    return opt

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

def get_orientation(base_path, start_frame=None, end_frame=None):
    def to_eularian_yaw_angle(q):
        x,y,z,w = q
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * ( y * y + z*z)
        return math.atan2(t3, t4)
        
    log_dir = os.path.join(base_path, 'log')
    frames_idx = sorted([int(file.replace('.json','')) for file in os.listdir(log_dir)])
    if len(frames_idx) < 3:
        return None
    if frames_idx[-1] - 1 != frames_idx[-2]:
        frames_idx = frames_idx[0: -1]
    if start_frame is None or start_frame == 0:
        start_frame = frames_idx[0]
    if end_frame is None or end_frame == -1:
        end_frame = frames_idx[-1]
    with open(os.path.join(log_dir, str(end_frame).zfill(6) + '.json'), 'r') as f:
        frame = json.load(f)
        end_pos = frame['sensors']['state']['position']
        # end_yaw = to_eularian_yaw_angle(frame['sensors']['state']['orientation'])
    with open(os.path.join(log_dir, str(start_frame).zfill(6) + '.json'), 'r') as f:
        frame = json.load(f)
        start_pos = frame['sensors']['state']['position']
        start_yaw = to_eularian_yaw_angle(frame['sensors']['state']['orientation'])
    delta = np.asarray(end_pos) - np.asarray(start_pos)
    delta_arraw_yaw = math.atan2(delta[1],delta[0])
    delta_yaw = math.degrees(delta_arraw_yaw - start_yaw)  # TODO: check this yaw angle calcution
    rot = euler_to_rotation_matrix([0, 0, start_yaw])
    delta = rot.T @ delta
    delta_norm = np.linalg.norm(delta)
    delta_factor = 0.4
    res = []
    for i in range(2):
        if delta[i] > delta_norm * delta_factor:
            res.append(1)
        elif delta[i] < -delta_norm * delta_factor:
            res.append(-1)
        else:
            res.append(0)
    res_1_dict = {1: "right", -1: "left", 0: ""}
    res_0_dict = {1: "front", -1: "back", 0: ""}
    orientation_desc = res_1_dict[res[1]] + (" "+ res_0_dict[res[0]] if res[0]!= 0 else "")
    return orientation_desc, delta_yaw

def project_this_state2target_state_axis(this_state, target_state):
    start_pos = target_state['position']
    start_eular = to_eularian_angles(target_state['orientation'])  # (pitch, roll, yaw)
    this_pos = this_state['position']
    this_eular = to_eularian_angles(this_state['orientation'])
    delta_pos = np.asarray(this_pos) - np.asarray(start_pos)
    delta_eular = np.asarray(this_eular) - np.asarray(start_eular)
    rot = euler_to_rotation_matrix(start_eular) 
    delta_pos = rot.T @ delta_pos
    return {'position': delta_pos.tolist(), 'orientation': delta_eular.tolist()}


InstructionTemplet = r"There is a target in the %orientation_description% of uav. Using your front as the x-axis and your right as the y-axis, The target is at a yaw angle of %orientation_value% degrees from you. %object_description% Please control the drone and find the target."

def merge_map_logs(map_dir):
    trajs = os.listdir(map_dir)
    traj_paths = []
    for traj_name in tqdm.tqdm(trajs):
        traj_merged_info = {}
        traj_path = os.path.join(map_dir, traj_name)
        if os.path.exists(os.path.join(traj_path, clip_merged_file_name)):
            continue
        logs_dir = os.path.join(traj_path, 'log')
        logs_paths = sorted([os.path.join(logs_dir, name) for name in os.listdir(logs_dir) if name.endswith('.json')])
        logs_filter_path =[path for path in logs_paths if os.path.isfile(os.path.abspath(os.path.join(path, '../', '../frontcamera', os.path.basename(path).replace('json', 'png'))))]
        if len(logs_filter_path) < 5:
            continue
        
        detailed_frames_state = []
        filtered_frames_raw_state = []
        indexs = []
        for log_path in logs_paths:
            file_idx = int(os.path.basename(log_path).split('.')[0])
            try:
                with open(log_path, 'r') as log_f:
                    state = json.load(log_f)['sensors']['state']
                    state = {'position': state['position'], 'orientation': state['orientation']}
            except Exception as e:
                print(e, log_path)
                raise e
            detailed_frames_state.append(state)
            if log_path in logs_filter_path:
                indexs.append(file_idx)
                filtered_frames_raw_state.append(state)
            
        start_state = filtered_frames_raw_state[0]
        projected_position = []
        for state in filtered_frames_raw_state:
            rela_state = project_this_state2target_state_axis(state, start_state)
            projected_position.append(rela_state['position'] + rela_state['orientation'])
        # prep instruction
        with open(os.path.join(traj_path, 'object_description.json'),'r') as obj_f:
            obj_descriptions = json.load(obj_f)
        obj_desc = random.choice(obj_descriptions)
        orientation_desc, orientation_value = get_orientation(traj_path, start_frame=0, end_frame=-1)
        instruction = InstructionTemplet.replace(r'%orientation_description%', orientation_desc).\
                replace(r'%object_description%', obj_desc).replace(r'%orientation_value%', str(round(orientation_value, 0)))
        traj_merged_info['trajectory'] = projected_position
        traj_merged_info['trajectory_raw'] = filtered_frames_raw_state
        traj_merged_info['trajectory_raw_detailed'] = detailed_frames_state
        traj_merged_info['image_feature_path'] = os.path.join(traj_path, 'feature.tensor')
        traj_merged_info['index'] = indexs
        traj_merged_info['length'] = len(indexs)
        traj_merged_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\n" + instruction
                },
                {"from": "gpt", "value": ""}
            ]
        with open(os.path.join(traj_path, clip_merged_file_name), 'w') as f:
            json.dump(traj_merged_info, f)
        traj_paths.append((os.path.join(traj_path, clip_merged_file_name), len(indexs)))
    return traj_paths
        
        
    
args = arg_parse()
if __name__ == '__main__':
    maps = os.listdir(args.root_dir)
    
    for map in tqdm.tqdm(args.map_list):
        map_dir = os.path.join(args.root_dir, map)
        if not os.path.isdir(map_dir):
            continue
        traj_paths = merge_map_logs(map_dir)
        