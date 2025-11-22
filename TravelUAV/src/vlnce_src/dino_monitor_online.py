import copy
import os

import cv2
import numpy as np
import math
import torch
from PIL import Image
import json
from src.common.param import model_args, args

# RGB_FOLDER = ['frontcamerarecord', 'downcamerarecord']

class DinoMonitor:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = DinoMonitor()
            return cls._instance
        return cls._instance
        
    def __init__(self, device=0):
        self.dino_model = None
        self.init_dino_model(device)
        self.object_desc_dict = dict()
        self.init_object_dict()
        
    def init_object_dict(self):
        with open(args.object_name_json_path, 'r') as f:
            file = json.load(f)
            for item in file:
                self.object_desc_dict[item['object_name']] = item['object_desc']
    
    def init_dino_model(self, device):
        import src.model_wrapper.utils.GroundingDINO as GroundingDINO
        import sys
        from functools import partial
        sys.path.append(GroundingDINO.__path__[0])
        from src.model_wrapper.utils.GroundingDINO.groundingdino.util.inference import load_model, predict
        device = torch.device(device)
        model = load_model(model_args.groundingdino_config, model_args.groundingdino_model_path)
        model.to(device=device)
        self.dino_model = partial(predict, model=model)
    
    def get_dino_results(self, episode, obj_info):
        images = episode[-1]['rgb_record']
        depths = episode[-1]['depth_record']
        done = False
        
        for i in range(len(images)):
            img = images[i]
            depth = depths[i]
            target_detections = []
            boxes, logits = self.detect(img, obj_info)

            if len(boxes) > 0:
                for i, point in enumerate(boxes):
                    point = list(map(int, point))
                    center_point = (int((point[0] + point[2]) / 2), int((point[1] + point[3]) / 2))
                    depth_data = int(depth[center_point[1], center_point[0]] / 2.55)
                    if depth_data < 18:
                        target_detections.append((float(logits[i]), depth_data))

            if len(target_detections) > 0:
                done = True
                break

        return done

    def get_dino_results_test(self, episode, obj_info, save_dir="/home/gentoo/asus/liusq/UAV_VLN/TravelUAV/data/eval_test/detection_results"):
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        images = episode[-1]['rgb_record']
        depths = episode[-1]['depth_record']
        done = False
        saved_files = []  # 存储保存的文件路径

        for img_idx in range(len(images)):
            img = images[img_idx]
            depth = depths[img_idx]
            target_detections = []
            boxes, logits = self.detect(img, obj_info)

            if len(boxes) > 0:
                # 创建副本用于绘制边界框，避免修改原图
                img_with_boxes = img.copy()

                for box_idx, box in enumerate(boxes):
                    # 将边界框坐标转换为整数
                    box = list(map(int, box))
                    x1, y1, x2, y2 = box

                    # 计算中心点
                    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                    # 获取深度信息
                    depth_data = int(depth[center_point[1], center_point[0]] / 2.55)

                    # 绘制边界框 (绿色，线宽2)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 绘制中心点 (红色)
                    cv2.circle(img_with_boxes, center_point, 5, (0, 0, 255), -1)

                    # 添加置信度和深度信息标签
                    label = f"Conf: {logits[box_idx]:.2f}, Depth: {depth_data}"
                    cv2.putText(img_with_boxes, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if depth_data < 18:
                        target_detections.append((float(logits[box_idx]), depth_data))

                # 如果检测到目标，保存图片
                if len(target_detections) > 0:
                    filename = f"{os.path.basename(episode[-1]['trajectory_dir'])}_{obj_info}_{len(episode)}.jpg"
                    save_path = os.path.join(save_dir, filename)

                    # 保存图片
                    #cv2.imwrite(save_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path, img_with_boxes)
                    saved_files.append(save_path)
                    #print(f"保存检测结果到: {save_path}")

                    done = True
                    break

        return done
    
    def detect(self, img, prompt):
        import groundingdino.datasets.transforms as T
        from groundingdino.util import box_ops
        
        img_src = copy.deepcopy(np.array(img))
        img = Image.fromarray(img_src)
        transform = T.Compose(
        [   T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_transformed, _ = transform(img, None)
        boxes, logits, phrases = self.dino_model(
            image=image_transformed,
            caption=prompt,
            box_threshold=0.6,
            text_threshold=0.40
        )
        logits = logits.detach().cpu().numpy()
        H, W, _ = img_src.shape
        boxes_xyxy = (box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])).cpu().numpy()
        boxes = []
        for box in boxes_xyxy:
            if (box[2] - box[0]) / W > 0.6 or (box[3] - box[1]) / H > 0.5:
                continue
            boxes.append(box)
        return boxes, logits
    
dino_monitor = DinoMonitor.get_instance()
