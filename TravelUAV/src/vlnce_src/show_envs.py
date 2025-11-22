import gymnasium as gym

# 列出所有已注册的环境（带详细信息）
print("已注册的Gymnasium环境:")
print("=" * 50)

for env_id in gym.envs.registry:
    spec = gym.envs.registry[env_id]
    print(f"ID: {env_id}")
    print(f"  入口点: {spec.entry_point}")
    print(f"  最大步数: {spec.max_episode_steps}")
    print(f"  奖励阈值: {spec.reward_threshold}")
    print("-" * 30)