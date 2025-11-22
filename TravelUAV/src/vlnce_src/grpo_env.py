from scipy.spatial.transform import Rotation as R
import numpy as np
import logging
from env_uav import AirVLNENV

logger = logging.getLogger(__name__)

class GRPO_UAVEnv:
    def __init__(self, dataset_path, save_path, eval_json_path, seed, success_reward, collision_penalty, step_penalty, activate_maps):
        self._env = AirVLNENV(batch_size=1, dataset_path=dataset_path, save_path=save_path, eval_json_path=eval_json_path, seed=seed, activate_maps=activate_maps)
        self._last_dist = None
        self.success_reward = float(success_reward)
        self.collision_penalty = float(collision_penalty)
        self.step_penalty = float(step_penalty)
        self._initialized = False
        self.observation_space = {'rgb': (5, 256, 256, 3), 'depth': (5, 256, 256), 'pose': (7,)}
        self.action_space_shape = (3,)

    def _ensure_batch_ready(self):
        try:
            if not hasattr(self._env, 'batch') or self._env.batch is None:
                logger.debug('Ensuring batch via next_minibatch')
                self._env.next_minibatch()
            if not hasattr(self._env, 'batch') or self._env.batch is None:
                raise RuntimeError('AirVLNENV.next_minibatch did not produce batch')
        except Exception as e:
            logger.exception(f'_ensure_batch_ready failed: {e}')
            raise

    def set_state(self, state_dict: dict):
        try:
            ready = hasattr(self._env, 'sim_states') and isinstance(self._env.sim_states, list) and len(self._env.sim_states) > 0 and (self._env.sim_states[0] is not None)
            if not ready:
                logger.warning("set_state called before initialization; performing reset")
                self.reset()
            s = self._env.sim_states[0]
            if 'position' in state_dict:
                s.pose[:3] = state_dict['position']
            if 'orientation' in state_dict:
                r, p, y = np.radians(state_dict['orientation'])
                quat = R.from_euler('xyz', [r, p, y], degrees=False).as_quat()
                s.pose[3:7] = quat
            if 'battery' in state_dict:
                s.battery = float(state_dict['battery'])
            if 'target_coords' in state_dict and hasattr(self._env, 'batch') and isinstance(self._env.batch, list) and len(self._env.batch) > 0:
                self._env.batch[0]['object_position'] = state_dict['target_coords']
            self._last_dist = self._distance_to_target()
            s.is_collisioned = False
            s.is_end = False
            s.oracle_success = False
            self._initialized = True
            logger.debug("set_state applied")
        except Exception as e:
            logger.exception(f"set_state failed: {e}")
            raise


    def reset(self, seed=None, options=None):
        logger.debug("reset called")
        self._ensure_batch_ready()
        obs_batch = self._env.reset()
        obs = self._to_obs_dict(obs_batch[0])
        self._last_dist = self._distance_to_target()
        self._initialized = True
        return obs, {}

    def step(self, action):
        try:
            self._ensure_batch_ready()
            if not getattr(self, "_initialized", False):
                logger.warning("step called before initialization; performing reset")
                self.reset()
            if not hasattr(self._env, 'machines_info'):
                logger.warning("machines_info missing; performing reset")
                self.reset()
            if isinstance(action, (list, tuple, np.ndarray)):
                waypoint = [float(action[0]), float(action[1]), float(action[2])]
            else:
                waypoint = list(action)
            self._env.makeActions(waypoints_list=[[waypoint]])
            obs_batch = self._env.get_obs()
            obs = self._to_obs_dict(obs_batch[0])
            dist = self._distance_to_target()
            reward = (self._last_dist - dist) - self.step_penalty
            self._last_dist = dist
            s = self._env.sim_states[0]
            if s.is_collisioned:
                reward -= self.collision_penalty
            terminated = bool(s.oracle_success)
            if terminated:
                reward += self.success_reward
            truncated = bool(s.is_end and not terminated)
            info = {"distance_to_target": dist, "collision": bool(s.is_collisioned)}
            logger.info(f"step: waypoint={waypoint}, reward={reward}, dist={dist}, collision={bool(s.is_collisioned)}, terminated={terminated}, truncated={truncated}")
            return obs, reward, terminated, truncated, info
        except Exception as e:
            logger.exception(f"step failed: {e}")
            raise

    def _to_obs_dict(self, obs_tuple):
        observations, _, _, _ = obs_tuple
        last = observations[-1]
        rgb = last['rgb']
        depth = last['depth']
        pose = np.array(self._env.sim_states[0].pose[:7], dtype=np.float32)
        rgb_stack = np.stack(rgb, axis=0)
        depth_stack = np.stack(depth, axis=0)
        return {'rgb': rgb_stack, 'depth': depth_stack, 'pose': pose}

    def _distance_to_target(self):
        s = self._env.sim_states[0]
        cur = np.array(s.pose[0:3])
        tgt = np.array(self._env.batch[0]['object_position'])
        return float(np.linalg.norm(cur - tgt))




