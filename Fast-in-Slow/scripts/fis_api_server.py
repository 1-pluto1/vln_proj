from flask import Flask, request, jsonify
import base64, io, numpy as np
from PIL import Image
import os
import argparse
import torch
import time
import uuid
import logging
from models import load_vla
import threading
import sys
import logging
import os
import sys
import colorlog 
from PIL import Image as PILImage

app = Flask(__name__)
BIND_HOST = "127.0.0.1"
BIND_PORT = 5000
PREDICT_MODE = "diff"
VLA = None
WAYPOINT_COUNT=7
UNNORMALIZE_KEY = "uav_dataset"
DEVICE = torch.device('cuda:0')

# --- 日志配置 ---
log_level_str = os.environ.get("LOG_LEVEL", "DEBUG").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# 定义颜色和格式
log_colors = {
    'DEBUG':    'bold_blue',
    'INFO':     'bold_green',
    'WARNING':  'bold_yellow',
    'ERROR':    'bold_red',
    'CRITICAL': 'bold_white,bg_red',
}

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(levelname)s]%(reset)s[%(name)s:%(funcName)s:%(lineno)d][%(threadName)s] %(message_log_color)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors=log_colors,
    secondary_log_colors={
        'message': {
            'DEBUG':    'bold_blue',
            'INFO':     'bold_green',
            'WARNING':  'bold_yellow',
            'ERROR':    'bold_red',
            'CRITICAL': 'bold_white,bg_red',
        }
    },
    style='%'
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter) 

root_logger = logging.getLogger()
root_logger.setLevel(log_level)

for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('flask').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)

# --- 日志配置结束 ---

def run_slow_system_background():

    global g_slow_output_cache, g_slow_input_cache, g_is_slow_system_busy, g_lock

    logger.info("[Slow Thread]: Activated! Checking for work...")

    g_lock.acquire()
    prompt = g_slow_input_cache["prompt"]
    slow_image = g_slow_input_cache["slow_image"]

    g_is_slow_system_busy = True 
    g_lock.release()

    if prompt is None or slow_image is None:
        logger.warning("[Slow Thread]: Slow system input cache is empty, skipping this run.")
        g_lock.acquire() 
        g_is_slow_system_busy = False 
        g_lock.release() 
        return 

    try:
        logger.info(f"[Slow Thread]: Starting System 2 (slow) inference... (this will take a while)")

        real_input_ids, real_slow_embedding = model_predict_slow_latent_embedding(
            VLA, prompt, slow_image 
        )

        logger.info(f"[Slow Thread]: System 2 (slow) inference completed.")
        g_lock.acquire()
        g_slow_output_cache["input_ids"] = real_input_ids.tolist()
        g_slow_output_cache["slow_latent_embedding"] = real_slow_embedding.tolist()
        g_is_slow_system_busy = False 
        g_lock.release()

        logger.info(f"[Slow Thread]: System 2 cache updated.")

    except Exception as e:
        logger.error(f"[Slow Thread]: System 2 inference failed: {e}")

        g_lock.acquire()
        g_is_slow_system_busy = False 
        g_lock.release()

def model_load(args):
    model = load_vla(
            args.model_path,
            load_for_training=False,
            future_action_window_size=int(args.model_action_steps),
            hf_token=args.hf_token,
            use_diff = 1,
            diffusion_steps = args.training_diffusion_steps,
            llm_middle_layer = args.llm_middle_layer,
            training_mode = args.training_mode,
            load_pointcloud = args.load_pointcloud,
            pointcloud_pos=args.pointcloud_pos,
            action_chunk=args.action_chunk,
            load_state=args.use_robot_state,
            lang_subgoals_exist=args.lang_subgoals_exist,
            action_dim=args.action_dim,
            )
    model.to(f'cuda:{args.cuda}').eval()
    return model

def model_predict(args, predict_mode, model, image, prompt, cur_robot_state=None, slow_image=None, point_cloud=None, input_ids = None, slow_latent_embedding=None):
    if predict_mode == 'ar' or predict_mode == 'diff+ar':
        output = model.predict_action(
                image_head_slow = slow_image,
                image_head_fast = image,
                point_cloud = point_cloud,
                instruction = prompt,
                unnorm_key=UNNORMALIZE_KEY,
                cfg_scale = float(args.cfg_scale), 
                use_ddim = True,
                num_ddim_steps = int(args.ddim_steps),
                cur_robot_state = cur_robot_state,
                action_dim = args.action_dim,
                predict_mode = predict_mode,
                )
    elif predict_mode == 'diff':
        output = model.fast_system_forward(
                image_head_fast = image,
                point_cloud=point_cloud,
                slow_latent_embedding = slow_latent_embedding,
                input_ids = input_ids,
                unnorm_key = UNNORMALIZE_KEY,
                cur_robot_state = cur_robot_state,
                cfg_scale = float(args.cfg_scale), 
                use_ddim = True,
                num_ddim_steps = int(args.ddim_steps),
                action_dim = args.action_dim,
                predict_mode = predict_mode,
                )
    return output

def model_predict_slow_latent_embedding(model, prompt, slow_image):
    input_ids, slow_latent_embedding = model.slow_system_forward(
        image_head_slow = slow_image,
        instruction = prompt,
        unnorm_key = UNNORMALIZE_KEY,
        )
    return input_ids, slow_latent_embedding


def quat_to_rpy(quat):
    """
    Convert quaternion [x, y, z, w] to RPY (roll, pitch, yaw) in radians.
    Returns in order: roll, pitch, yaw
    """
    try:
        x, y, z, w = map(float, quat)

        n = np.sqrt(x*x + y*y + z*z + w*w)
        if n == 0:
            return 0.0, 0.0, 0.0
        x, y, z, w = x/n, y/n, z/n, w/n

        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)  
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw], dtype=np.float32)
    except Exception:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def quat_to_rot(q):
    try:
        x, y, z, w = [float(v) for v in q]
        n = x*x + y*y + z*z + w*w
        if n == 0:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / n
        xx, yy, zz = x*x*s, y*y*s, z*z*s
        xy, xz, yz = x*y*s, x*z*s, y*z*s
        wx, wy, wz = w*x*s, w*y*s, w*z*s
        R = np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)]
        ], dtype=np.float32)
        return R
    except Exception:
        return np.eye(3, dtype=np.float32)

def build_waypoints(pos, deltas, R=None, count=7):
    wp = []
    R = np.eye(3, dtype=np.float32) if R is None else R
    for i in range(min(len(deltas), count)):
        d = np.asarray(deltas[i], dtype=np.float32)
        world_d = R @ d
        wp.append((pos + world_d).tolist())
    if len(wp) == 0:
        wp.append(pos.tolist())
    while len(wp) < count:
        wp.append(wp[-1])
    return wp

def _decode_image(b64):
    if not b64:
        return None
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)

@app.route("/health", methods=["GET"]) 
def health():
    logger.info("Health check requested")
    return jsonify({"status": "ok", "model_loaded": VLA is not None})

@app.route("/predict", methods=["POST"])
def predict():

    global g_fast_step_counter, g_slow_output_cache, g_slow_input_cache
    global g_is_slow_system_busy, g_lock, g_slow_fast_ratio

    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    data = request.json or {}
    schema = data.get("schema")
    version = data.get("version")
    pos = np.array(data.get("position", [0, 0, 0]), dtype=np.float32)
    tgt = np.array(data.get("target_position", pos.tolist()), dtype=np.float32)
    img_b64 = data.get("image_rgb_b64")
    depth_b64 = data.get("image_depth_b64")
    instruction_ep = data.get("instruction_ep")
    assist_instruction = data.get("assist_instruction")
    quat = data.get("orientation_quat")
    imu_rotation = data.get("imu_rotation")
    has_collided = data.get("has_collided")
    
    try:
        logger.info(f"[{request_id}] Received request keys: {list(data.keys())}")
        logger.info(f"[{request_id}] Schema: {schema}, Version: {version}")
        logger.debug(f"[{request_id}] Position: {pos.tolist()}, shape: {pos.shape}")
        logger.debug(f"[{request_id}] Target: {tgt.tolist()}, shape: {tgt.shape}")
        logger.info(f"[{request_id}] Image data lengths - RGB: {len(img_b64) if isinstance(img_b64, str) else 0}, Depth: {len(depth_b64) if isinstance(depth_b64, str) else 0}")
        logger.debug(f"[{request_id}] Quaternion length: {len(quat) if isinstance(quat, list) else 0}")
        logger.debug(f"[{request_id}] IMU rotation shape: {(np.array(imu_rotation).shape if isinstance(imu_rotation, list) else None)}")
        logger.info(f"[{request_id}] Collision detected: {bool(has_collided)}")
    except Exception as e:
        logger.error(f"[{request_id}] Error logging request data: {e}")

    img_np = _decode_image(img_b64)
    depth_np = _decode_image(depth_b64) if depth_b64 else None
    if img_np is not None:
        try:
            logger.debug(f"[{request_id}] RGB image - shape: {img_np.shape}, dtype: {img_np.dtype}, range: [{img_np.min():.3f}, {img_np.max():.3f}]")
        except Exception:
            logger.warning(f"[{request_id}] Failed to log RGB image properties")
    if depth_np is not None:
        try:
            logger.debug(f"[{request_id}] Depth image - shape: {depth_np.shape}, dtype: {depth_np.dtype}, range: [{depth_np.min():.3f}, {depth_np.max():.3f}]")
        except Exception:
            logger.warning(f"[{request_id}] Failed to log depth image properties")

    if pos.shape != (3,) or tgt.shape != (3,):
        return jsonify({"error": "invalid_position"}), 400

    if VLA is None or img_np is None:
        fallback_reason = "model_not_loaded" if VLA is None else "no_image"
        direction = tgt - pos
        norm = float(np.linalg.norm(direction)) + 1e-6
        step = direction / norm * min(1.0, norm)
        waypoints = [(pos + step * k).tolist() for k in range(1, WAYPOINT_COUNT + 1)]
        elapsed = time.time() - start_time
        logger.warning(f"[{request_id}] Fallback triggered - reason: {fallback_reason}, elapsed: {elapsed:.3f}s")
        logger.debug(f"[{request_id}] Fallback waypoints: {waypoints}")
        return jsonify({"waypoints": waypoints, "schema": schema, "version": version, "fallback": True, "fallback_reason": fallback_reason})

    image = PILImage.fromarray(img_np)
    logger.debug(f"[{request_id}] Image converted to PIL format - size: {image.size}, mode: {image.mode}")

    prompt = data.get("instruction_ep") or data.get("assist_instruction")

    if quat and len(quat) == 4:
        rpy = quat_to_rpy(quat)
    else:
        logger.warning(f"[{request_id}] No valid quat received, RPY will use [0,0,0].")
        rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    cur_robot_state = np.concatenate([pos, rpy]).astype(np.float32)

    # if args.load_depth and args.depth_pos == 'slow' and slow_cnt % int(args.slow_fast_ratio) == 0:
    #     depth_image = depth_np if args.load_depth else None
    # elif args.load_depth and args.depth_pos == 'fast':
    #     depth_image = depth_np if args.load_depth else None
    # else:
    #     depth_image=None

    point_cloud=None

    g_lock.acquire()

    g_fast_step_counter += 1

    g_slow_input_cache['prompt'] = prompt
    g_slow_input_cache['slow_image'] = image

    current_counter = g_fast_step_counter
    is_busy = g_is_slow_system_busy
    current_slow_output = g_slow_output_cache.copy()

    g_lock.release()

    if current_counter == 1 or current_slow_output['slow_latent_embedding'] is None:
        logger.info(f"[{request_id}] Cold start: first call, running System 2 synchronously...")

        run_slow_system_background()

        g_lock.acquire()
        current_slow_output = g_slow_output_cache.copy()
        g_lock.release()
        logger.info(f"[{request_id}] Cold start complete.")

    elif (current_counter % g_slow_fast_ratio == 0) and not is_busy:
        logger.info(f"[{request_id}] Step {current_counter}: triggering System 2 update asynchronously...")

        thread = threading.Thread(target=run_slow_system_background, daemon=True)
        thread.start()

    try:
        logger.debug(f"[{request_id}] Starting VLA prediction with params:")
        logger.debug(f"[{request_id}]   - unnorm_key: {UNNORMALIZE_KEY}")
        logger.debug(f"[{request_id}]   - cfg_scale: 1.0")
        logger.debug(f"[{request_id}]   - use_ddim: True")
        logger.debug(f"[{request_id}]   - num_ddim_steps: 4")
        logger.debug(f"[{request_id}]   - action_dim: 6")
        logger.debug(f"[{request_id}]   - predict_mode: {PREDICT_MODE}")
        
        logger.debug(f"[{request_id}] Starting System 1 (fast) inference...")
        inference_start = time.time()

        cached_input_ids_list = current_slow_output['input_ids']
        cached_embedding_list = current_slow_output['slow_latent_embedding']

        if cached_embedding_list is None:
            logger.error(f"[{request_id}] Fatal error: slow system cache is empty, unable to run fast system.")
            raise Exception("Slow system cache is empty")

        input_ids_tensor = torch.tensor(cached_input_ids_list, device=DEVICE)
        slow_latent_embedding_tensor = torch.tensor(cached_embedding_list, device=DEVICE)

        try:
            logger.debug(f"[{request_id}] Pre-infer params:")
            logger.debug(f"[{request_id}]   image_size={image.size} image_mode={image.mode}")
            logger.debug(f"[{request_id}]   prompt_len={len(prompt) if prompt else 0}")
            logger.debug(f"[{request_id}]   cur_robot_state shape={(tuple(cur_robot_state.shape) if torch.is_tensor(cur_robot_state) else 'None')} dtype={(cur_robot_state.dtype if torch.is_tensor(cur_robot_state) else 'None')} device={(cur_robot_state.device if torch.is_tensor(cur_robot_state) else 'None')}")
            logger.debug(f"[{request_id}]   point_cloud={'None' if point_cloud is None else type(point_cloud).__name__}")
            logger.debug(f"[{request_id}]   input_ids shape={tuple(input_ids_tensor.shape)} dtype={input_ids_tensor.dtype} device={input_ids_tensor.device}")
            logger.debug(f"[{request_id}]   slow_embed shape={tuple(slow_latent_embedding_tensor.shape)} dtype={slow_latent_embedding_tensor.dtype} device={slow_latent_embedding_tensor.device}")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to log pre-infer params: {e}")

        output = model_predict(
            args,
            PREDICT_MODE,
            VLA,
            image,
            prompt,
            cur_robot_state,
            None,
            point_cloud,
            input_ids_tensor,
            slow_latent_embedding_tensor
        )

        actions = np.asarray(output)

        inference_time = time.time() - inference_start
        logger.debug(f"[{request_id}] VLA prediction completed in {inference_time:.3f}s")
        actions = np.asarray(actions).reshape(-1, 6)
        logger.debug(f"[{request_id}] Raw actions shape: {actions.shape}, dtype: {actions.dtype}")
        logger.debug(f"[{request_id}] Actions range: [{actions.min():.6f}, {actions.max():.6f}]")
        
        deltas = actions[:, :3]
        logger.debug(f"[{request_id}] Deltas shape: {deltas.shape}")
        logger.debug(f"[{request_id}] Deltas range: [{deltas.min():.6f}, {deltas.max():.6f}]")
        
        quat = data.get("orientation_quat")
        logger.debug(f"[{request_id}] Input quaternion: {quat}")
        
        R = quat_to_rot(quat) if isinstance(quat, list) and len(quat) == 4 else np.eye(3, dtype=np.float32)
        logger.debug(f"[{request_id}] Rotation matrix:\n{R}")
        
        waypoints = build_waypoints(pos, deltas, R=R, count=WAYPOINT_COUNT)
        logger.debug(f"[{request_id}] Built waypoints with position: {pos}, deltas count: {len(deltas)}")
        elapsed = time.time() - start_time
        
        logger.info(f"[{request_id}] Inference successful - actions: {actions.shape}, deltas: {deltas.shape}, waypoints: {len(waypoints)}, elapsed: {elapsed:.3f}s")
        logger.debug(f"[{request_id}] Actions sample: {actions[0].tolist() if len(actions) > 0 else []}")
        logger.debug(f"[{request_id}] Deltas sample: {deltas[0].tolist() if len(deltas) > 0 else []}")
        logger.debug(f"[{request_id}] Generated waypoints: {waypoints}")
        
        return jsonify({
            "waypoints": waypoints, 
            "schema": schema, 
            "version": version, 
            "request_id": request_id, 
            "inference_time": elapsed,
            "actions_shape": actions.shape,
            "deltas_shape": deltas.shape
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        logger.error(f"[{request_id}] Inference failed: {str(e)}")
        logger.error(f"[{request_id}] Error context - pos: {pos.tolist()}, tgt: {tgt.tolist()}, quat_len: {(len(data.get('orientation_quat')) if isinstance(data.get('orientation_quat'), list) else 0)}")
        logger.debug(f"[{request_id}] Full error traceback:\n{error_trace}")
        
        direction = tgt - pos
        norm = float(np.linalg.norm(direction)) + 1e-6
        step = direction / norm * min(1.0, norm)
        waypoints = [(pos + step * k).tolist() for k in range(1, WAYPOINT_COUNT + 1)]
        elapsed = time.time() - start_time
        
        logger.warning(f"[{request_id}] Using fallback waypoints - count: {len(waypoints)}, elapsed: {elapsed:.3f}s")
        
        return jsonify({
            "waypoints": waypoints, 
            "schema": schema, 
            "version": version, 
            "fallback": True, 
            "error": str(e), 
            "request_id": request_id, 
            "inference_time": elapsed
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--replay-or-predict', type=str, default='predict')
    parser.add_argument('--training_mode', type=str, default='async')
    parser.add_argument('--slow-fast-ratio', type=int, default=4)
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--training-diffusion-steps', type=int, default=100)
    parser.add_argument('--llm_middle_layer', type=int, default=32)
    parser.add_argument('--use-diff', type=int, default=1)
    parser.add_argument('--use-ar', type=int, default=1)
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--model-action-steps', type=str, default='0')
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--load-pointcloud', type=int, default=0)
    parser.add_argument('--pointcloud-pos', type=str, default='slow')
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--sparse', type=int, default=0)
    parser.add_argument('--angle_delta', type=int, default=0)
    parser.add_argument('--lang_subgoals_exist', type=int, default=0)
    parser.add_argument('--ddim-steps', type=int, default=10)
    parser.add_argument('--cfg-scale', type=str, default='0')
    parser.add_argument('--threshold', type=str, default='5.8')
    parser.add_argument('--hf-token', type=str, default='')
    parser.add_argument('--action-dim', type=int, default=6)

    args=parser.parse_args()
    
    BIND_HOST = args.host
    BIND_PORT = args.port

    DEVICE = torch.device(f'cuda:{args.cuda}')
    if int(args.use_diff)==1 and int(args.use_ar)==0:
        PREDICT_MODE = 'diff'
    elif int(args.use_diff)==0 and int(args.use_ar)==1:
        PREDICT_MODE = 'ar'
    elif int(args.use_diff)==1 and int(args.use_ar)==1:
        PREDICT_MODE = 'diff+ar'

    VLA = model_load(args)

    g_lock = threading.Lock() 

    g_slow_output_cache = {
        "input_ids": None,
        "slow_latent_embedding": None
    }

    g_slow_input_cache = {
        "prompt": None,
        "slow_image": None 
    }

    g_fast_step_counter = 0 
    g_is_slow_system_busy = False 
    g_slow_fast_ratio = args.slow_fast_ratio  

    logger.info(f"Starting server on {BIND_HOST}:{BIND_PORT}")
    logger.info(f"Waypoint count: {WAYPOINT_COUNT}")
    logger.info(f"Predict mode: {PREDICT_MODE}")
    logger.info(f"Model loaded: {VLA is not None}")
    
    app.run(host=BIND_HOST, port=BIND_PORT, debug=False)
