import os, io, base64, numpy as np, requests
from PIL import Image
from src.model_wrapper.base_model import BaseModelWrapper
from src.vlnce_src.dino_monitor_online import DinoMonitor

FIS_SERVER_URL = os.environ.get("FIS_SERVER_URL", "http://127.0.0.1:5000/predict")
FIS_TIMEOUT = float(os.environ.get("FIS_TIMEOUT", "20.0"))
SESSION = requests.Session()
JPEG_QUALITY = int(os.environ.get("FIS_JPEG_QUALITY", "85"))
WAYPOINT_EXPECTED_LEN = int(os.environ.get("FIS_WAYPOINT_EXPECTED_LEN", "7"))

def _encode_image(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")

class FiSIPCModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        self.model_args = model_args
        self.data_args = data_args
        self.dino_moinitor = None

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        batch_ctx = []
        for i in range(len(episodes)):
            obs = episodes[i][-1] if len(episodes[i]) > 0 else {}
            img_b64 = None
            depth_b64 = None
            try:
                if isinstance(obs.get("rgb"), list) and len(obs["rgb"]) > 0:
                    img_b64 = _encode_image(obs["rgb"][0])
            except Exception:
                img_b64 = None
            try:
                if isinstance(obs.get("depth"), list) and len(obs["depth"]) > 0:
                    depth_b64 = _encode_image(obs["depth"][0])
            except Exception:
                depth_b64 = None
            sensors = obs.get("sensors") or {}
            state = sensors.get("state", {})
            pos = list(map(float, state.get("position", [0.0, 0.0, 0.0])))[:3]
            orient_quat = list(map(float, state.get("orientation", [0.0, 0.0, 0.0, 1.0])))[:4]
            imu_rot = (sensors.get("imu") or {}).get("rotation")
            has_collided = bool((state.get("collision") or {}).get("has_collided", False))
            tgt = target_positions[i] if i < len(target_positions) else pos

            instr_ep = obs.get("instruction")
            instr_assist = (assist_notices[i] if assist_notices is not None and i < len(assist_notices) else None)
            batch_ctx.append({
                "schema": "uav-vln/fis-ipc-v1",
                "version": "1.0",
                "image_rgb_b64": img_b64,
                "image_depth_b64": depth_b64,
                "position": pos,
                "orientation_quat": orient_quat,
                "imu_rotation": imu_rot,
                "has_collided": has_collided,
                "instruction_ep": instr_ep,
                "assist_instruction": (str(instr_assist) if instr_assist not in [None, ""] else None),
                "target_position": tgt,
            })
        return {"batch_ctx": batch_ctx}, None

    def eval(self):
        pass

    def run(self, inputs, episodes, rot_to_targets):
        batch_ctx = inputs["batch_ctx"]
        waypoints_batch = []
        for item in batch_ctx:
            payload = {
                "instruction_ep": item.get("instruction_ep"),
                "assist_instruction": item.get("assist_instruction"),
                "image_rgb_b64": item.get("image_rgb_b64"),
                "image_depth_b64": item.get("image_depth_b64"),
                "position": item.get("position"),
                "orientation_quat": item.get("orientation_quat") or item.get("orientation"),
                "imu_rotation": item.get("imu_rotation"),
                "has_collided": item.get("has_collided"),
                "target_position": item.get("target_position"),
                "schema": item.get("schema"),
                "version": item.get("version"),
            }
            try:
                resp = SESSION.post(FIS_SERVER_URL, json=payload, timeout=FIS_TIMEOUT, headers={"X-IPC-Version": "v1"})
                resp.raise_for_status()
                data = resp.json()
                wp = np.array(data.get("waypoints", []), dtype=np.float64)
                if wp.size == 0:
                    raise requests.exceptions.RequestException()
                if wp.ndim == 1:
                    wp = wp.reshape(1, -1)
                if wp.shape[1] > 3:
                    wp = wp[:, :3]
                if wp.shape[0] < WAYPOINT_EXPECTED_LEN:
                    last = wp[-1] if wp.shape[0] > 0 else np.array(item["position"], dtype=np.float64)
                    pad = np.stack([last] * (WAYPOINT_EXPECTED_LEN - wp.shape[0]), axis=0)
                    wp = np.concatenate([wp, pad], axis=0)
                elif wp.shape[0] > WAYPOINT_EXPECTED_LEN:
                    wp = wp[:WAYPOINT_EXPECTED_LEN]
                waypoints_batch.append(wp)
            except requests.exceptions.RequestException:
                cur_pos = np.array(item["position"], dtype=np.float32)
                waypoints_batch.append(np.stack([cur_pos] * 5, axis=0))
        return waypoints_batch

    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results_test(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones