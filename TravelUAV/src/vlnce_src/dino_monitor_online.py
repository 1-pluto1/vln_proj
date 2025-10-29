import copy
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
