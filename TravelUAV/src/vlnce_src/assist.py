import numpy as np
import math
import torch
import cv2
import os
from PIL import Image
from src.vlnce_src.env_uav import RGB_FOLDER, DEPTH_FOLDER
from collections import deque
class Assist:
    def __init__(self, always_help = False, use_gt = False, device=0):
        self.always_help = always_help
        self.use_gt = use_gt
        self.dino_monitor = None
        self.dino_results = []
        self.depth_results = []
        self.recent_help_deque = deque(maxlen=9)
    
    def find_shortest_pos(self, cur_pos, traj):
        x, y, z = cur_pos[0], cur_pos[1], cur_pos[2]
        shortest_distance = 99999
        shortest_pos = None
        true_index = -1
        for i in range(len(traj)):
            pos = traj[i]['position']
            cur_distance = math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)
            if cur_distance < shortest_distance:
                shortest_distance = cur_distance
                shortest_pos = pos
                true_index = i

        shortest_distance = 99999
        shortest_pos = None
        for i in range(min(true_index, len(traj)-1), len(traj)):
            pos = traj[i]['position']
            cur_distance = math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)
            if (cur_distance < shortest_distance and cur_distance > 3) or (i == len(traj) - 1):
                shortest_distance = cur_distance
                shortest_pos = pos
                break

        return shortest_pos

    def check_collision_by_depth(self, episodes, current_observations, collisions, dones):
        for i, prev_episode in enumerate(episodes):
            collision_type = None
            if collisions[i]:
                collision_type = 'already'
                if not dones[i]:
                    dones[i] = True
                continue
            
            diffs = []
            close_collision = False
            current_episode = current_observations[i]
            for cid, camera_name in enumerate(DEPTH_FOLDER):
                diff = np.mean(np.abs(prev_episode[-1]['depth'][cid] - current_episode[-1]['depth'][cid]))
                zero_cnt  = (current_episode[-1]['depth'][cid] <= 1).sum()
                if zero_cnt > 0.1 * current_episode[-1]['depth'][cid].size:
                    close_collision = True
                diffs.append(diff)
            distance = np.array(prev_episode[-1]["sensors"]["state"]["position"]) - np.array(current_episode[-1]["sensors"]["state"]["position"])
            distance = np.linalg.norm(np.array(distance))
            diffs = np.array(diffs)
            if np.all(diffs < 3):
                collision_type = 'tiny diff'
            elif close_collision:
                collision_type = 'close'
            elif distance < 0.1:
                collision_type = 'distance'
            
            if collision_type is not None:
                print('collision type: ', collision_type)
            collisions[i] = np.all(diff < 3) or close_collision or distance < 0.1
            if collisions[i] and not dones[i]:
                dones[i] = True
        return collisions, dones
    
    def depth_detection(self, episodes):
        self.depth_results = []
        is_helps = [False for _ in range(len(episodes))]
        for i in range(len(episodes)):
            depth_result = []
            for cid, camera_name in enumerate(RGB_FOLDER):       
                img_src = episodes[i][-1]['depth'][cid]
                img_src = np.array(img_src[64:192, 64:192])
                depth = min(min(row) for row in img_src) / 2.55
                depth_result.append(depth)
                if 'down' in camera_name and depth < 7:
                    is_helps[i] = True
                elif ('left' in camera_name or 'right' in camera_name) and depth < 7:
                    is_helps[i] = True
                elif 'front' in camera_name and depth < 10:
                    is_helps[i] = True
            self.depth_results.append(depth_result)
        return is_helps
    
    def judge_helps(self, episodes, object_infos):
        is_helps = self.depth_detection(episodes)
        self.dino_target_detection(episodes, object_infos)
        for i in range(len(episodes)):
            if any(self.dino_results[i]):
                is_helps[i] = True
        print("judge_helps:", is_helps)
        print("dino_results:", self.dino_results)
        print("depth_reslts:", self.depth_results)
        return is_helps
    
    def get_assist_notice_with_gt(self, episodes, trajs, is_helps):
        try:
            assist_notices = [None for _ in range(len(episodes))]
            for i in range(len(episodes)):
                if not is_helps[i]:
                    continue
                ep = episodes[i]
                if len(ep) < 6:
                    assist_notices[i] = 'take off'
                    continue
                traj = trajs[i]

                pre_pos = ep[-6]["sensors"]["state"]["position"]
                cur_pos = ep[-1]["sensors"]["state"]["position"]
                shortest_pos = self.find_shortest_pos(cur_pos=cur_pos, traj=traj)
                pre_vec = np.array(cur_pos) - np.array(pre_pos)
                cur_vec = np.array(shortest_pos) - np.array(cur_pos)
                axis_ratio = np.abs(cur_vec) / np.linalg.norm(cur_vec)

                last_pos = traj[-1]['position']
                distance_to_end = np.linalg.norm(np.array(cur_pos[0:2]) - np.array(last_pos[0:2]))
                state = 'cruise'

                if np.argmax(axis_ratio) == 2 or cur_vec[2] < 0 or distance_to_end < 10: 
                    if cur_vec[2] < -3:
                        state = 'take off'
                    elif cur_vec[2] < 0:
                        if self.always_help:
                            self.depth_detection(episodes)
                        if self.depth_results[i][-1] < 7:
                            state = 'take off' 
                        else:
                            pre_vec = pre_vec[0:2] 
                            cur_vec = cur_vec[0:2]
                            delta_angle = np.arccos(np.dot(pre_vec, cur_vec) / (np.linalg.norm(pre_vec) + 1e-6) / (np.linalg.norm(cur_vec) + 1e-6)) * 180 / np.pi
                            if delta_angle > 20:
                                if int(np.cross(pre_vec, cur_vec)) > 0:
                                    state = 'right'
                                else:
                                    state = 'left'  
                    elif cur_vec[2] > 7 or distance_to_end < 10:
                        state = 'landing'
                    else:
                        pre_vec = pre_vec[0:2] 
                        cur_vec = cur_vec[0:2]
                        delta_angle = np.arccos(np.dot(pre_vec, cur_vec) / (np.linalg.norm(pre_vec) + 1e-6) / (np.linalg.norm(cur_vec) + 1e-6)) * 180 / np.pi
                        if delta_angle > 20:
                            if int(np.cross(pre_vec, cur_vec)) > 0:
                                state = 'right'
                            else:
                                state = 'left'  
                else:
                    pre_vec = pre_vec[0:2] 
                    cur_vec = cur_vec[0:2]
                    delta_angle = np.arccos(np.dot(pre_vec, cur_vec) / (np.linalg.norm(pre_vec) + 1e-6) / (np.linalg.norm(cur_vec) + 1e-6)) * 180 / np.pi
                    if delta_angle > 20:
                        if int(np.cross(pre_vec, cur_vec)) > 0:
                            state = 'right'
                        else:
                            state = 'left'
                assist_notices[i] = state
        except Exception as e:
            import pdb; pdb.set_trace()
            print(f'Debug: {e}')
        
        return assist_notices
    
    def get_assist_notice_with_rule(self, episodes, object_infos, target_positions, is_helps):
        assist_notices = [None for _ in range(len(episodes))]
        if self.always_help:
            self.depth_detection(episodes)
            self.dino_target_detection(episodes, object_infos)
        for i in range(len(episodes)):
            if not is_helps[i]:
                continue
            ep = episodes[i]
            depth_result = self.depth_results[i]
            dino_result = self.dino_results[i]
            target_position = target_positions[i]
            if dino_result[RGB_FOLDER.index('frontcamera')] or dino_result[RGB_FOLDER.index('downcamera')]:
                assist_notices[i] = 'landing'
            elif depth_result[RGB_FOLDER.index('leftcamera')] < 7 or depth_result[RGB_FOLDER.index('rightcamera')] < 7:
                if depth_result[RGB_FOLDER.index('leftcamera')] < depth_result[RGB_FOLDER.index('rightcamera')]:
                    assist_notices[i] = 'right'
                elif depth_result[RGB_FOLDER.index('leftcamera')] > depth_result[RGB_FOLDER.index('rightcamera')]:
                    assist_notices[i] = 'left'
            elif dino_result[RGB_FOLDER.index('leftcamera')] or dino_result[RGB_FOLDER.index('rightcamera')]:
                if dino_result[RGB_FOLDER.index('leftcamera')] and dino_result[RGB_FOLDER.index('rightcamera')]:
                    assist_notices[i] = 'cruise'
                elif dino_result[RGB_FOLDER.index('leftcamera')]:
                    assist_notices[i] = 'left'
                else:
                    assist_notices[i] = 'right'
            elif depth_result[RGB_FOLDER.index('downcamera')] < 7:
                assist_notices[i] = 'take off'
            elif depth_result[RGB_FOLDER.index('frontcamera')] < 10:
                cur_rot = ep[-1]['sensors']['imu']["rotation"]
                cur_pos = ep[-1]['sensors']['state']['position']
                target_position = np.array(cur_rot).T @ np.array(np.array(target_position) - np.array(cur_pos))
                if target_position[1] < 0:
                    assist_notices[i] = 'left'
                else:
                    assist_notices[i] = 'right'
        return assist_notices

    def get_assist_notice(self, episodes, trajs, object_infos, target_positions):
        assist_notices = [None for _ in range(len(episodes))]
        is_helps = [False for _ in range(len(episodes))]
        if not self.always_help:
            is_helps = self.judge_helps(episodes, object_infos)
        else:
            is_helps = [True for _ in range(len(episodes))]
        forced_help = [not all([helps[idx] for helps in self.recent_help_deque]) for idx, _ in enumerate(episodes)]
        is_helps = [forced_help[idx] or is_helps[idx] for idx, _ in enumerate(episodes)]
        self.recent_help_deque.append(is_helps)
        if self.use_gt:
            assist_notices = self.get_assist_notice_with_gt(episodes, trajs, is_helps)
        else:
            assist_notices = self.get_assist_notice_with_rule(episodes, object_infos, target_positions, is_helps)
        return assist_notices
    
    def dino_target_detection(self, episodes, object_infos) -> list[list]:
        target_detections = []
        if self.dino_monitor is None:
            from src.vlnce_src.dino_monitor_online import DinoMonitor
            self.dino_monitor = DinoMonitor.get_instance()
        for idx, (epi, obj_info) in enumerate(zip(episodes, object_infos)):
            cameras_detect = [False] * len(RGB_FOLDER)
            for cid, camera_name in enumerate(RGB_FOLDER):       
                img = Image.fromarray(epi[-1]['rgb'][cid])
                boxes, _ = self.dino_monitor.detect(img, obj_info)
                if len(boxes) > 0:
                    cameras_detect[cid] = True
                # dest = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
                # if len(boxes) > 0:
                #     for point in boxes:
                #         point = list(map(int, point))
                #         dest = cv2.rectangle(dest, point[0:2], point[2:], (0,0,255), thickness=None, lineType=None, shift=None)
                #     cv2.imwrite(os.path.join(f'test/dino_{idx}_{prompt}_{camera_name}.png'), dest)
            target_detections.append(cameras_detect)
        self.dino_results = target_detections
        return target_detections


if __name__ == '__main__':
    ass = Assist()
