# 数据说明

- 本文描述闭环评估过程中三个核心变量的类型、结构与语义：`batch_state.episodes`、`batch_state.target_positions`、`assist_notices`。
- 结合日志样例，提供一个可阅读的 JSON 结构示例，便于快速理解。

## 变量概览
- `batch_state.episodes`
  - 类型：`list`
  - 结构：`episodes[batch_index] -> list -> dict`
  - 语义：每个 batch 的场景与传感器上下文，包含传感器状态、IMU、任务指令、数据目录、以及当前/历史图像与深度帧
- `batch_state.target_positions`
  - 类型：`list`
  - 结构：`[ [x, y, z] ]`
  - 语义：每个智能体的目标位置（世界坐标/局部坐标，视具体实现而定）
- `assist_notices`
  - 类型：`list`
  - 结构：`[ str, ... ]`
  - 语义：辅助提示/建议（如起飞、保持高度等），供生成或策略模块参考

## 详细结构说明
- `episodes[0][0]`（字典）典型键：
  - `sensors.state`
    - `position`: `[x, y, z]`
    - `linear_velocity`: `[vx, vy, vz]`
    - `linear_acceleration`: `[ax, ay, az]`
    - `orientation`: 四元数 `[qx, qy, qz, qw]`
    - `angular_velocity`: `[wx, wy, wz]`
    - `angular_acceleration`: `[awx, awy, awz]`
    - `collision`: `{has_collided: bool, object_name: str}`
    - `gps_location`: `[lat, lon, alt]`
    - `timestamp`: `int`
  - `sensors.imu`
    - `time_stamp`: `int`
    - `rotation`: `3x3` 旋转矩阵
    - `orientation`: 四元数 `[qx, qy, qz, qw]`
    - `linear_acceleration`: `[ax, ay, az]`
    - `angular_velocity`: `[wx, wy, wz]`
  - `instruction`: 任务与场景的自然语言描述（可含 `<image>` 标记）
  - `trajectory_dir`: 数据目录
  - `teacher_action`: 教师动作（可能为 `None`）
  - `rgb` / `depth` / `rgb_record` / `depth_record`: 列表，元素为 `numpy.ndarray(dtype=uint8)` 图像/深度帧
    - `rgb` 帧形状：`H x W x 3`
    - `depth` 帧形状：`H x W`

## JSON 示例（基于日志样例）
```json
{
  "episodes": [
    [
      {
        "sensors": {
          "state": {
            "position": [678.7479858398438, 775.114013671875, -121.80349731445312],
            "linear_velocity": [0.0, 0.0, 0.01307726837694645],
            "linear_acceleration": [0.0, 0.0, 8.717992782592773],
            "orientation": [-0.0, 0.0, 0.7131236791610718, -0.7010382413864136],
            "angular_velocity": [0.0, 0.0, 0.0],
            "angular_acceleration": [0.0, 0.0, 0.0],
            "collision": {"has_collided": true, "object_name": ""},
            "gps_location": [47.64757206245155, -122.1298487236341, 243.88674926757812],
            "timestamp": 1763731582732888576
          },
          "imu": {
            "time_stamp": 1763731582732888576,
            "rotation": [
              [-0.017090763560446476, 0.9998539398601736, -0.0],
              [-0.9998539398601736, -0.017090763560446476, 0.0],
              [0.0, 0.0, 1.0]
            ],
            "orientation": [-0.0, 0.0, 0.7131236791610718, -0.7010382413864136],
            "linear_acceleration": [0.01952425390481949, -0.0515897199511528, -1.0558462142944336],
            "angular_velocity": [-0.0028537039179354906, -0.002337362617254257, 0.0016809632070362568]
          }
        },
        "instruction": "<image> ... Please control the drone and find the target.",
        "trajectory_dir": "/home/gentoo/docker_shared/asus/liusq/UAV_VLN/TravelUAV/data/TravelUAV_unzip/Carla_Town15/33e66ebb-b2de-4c4e-aa65-e3501d6fae40",
        "teacher_action": null,
        "rgb": ["uint8[H][W][3]", "..."],
        "depth": ["uint8[H][W]", "..."],
        "rgb_record": ["uint8[H][W][3]", "..."],
        "depth_record": ["uint8[H][W]", "..."]
      }
    ]
  ],
  "target_positions": [
    [648.2244262695312, 423.89605712890625, -117.1862564086914]
  ],
  "assist_notices": [
    "take off"
  ]
}
```

## 说明
- 图像/深度帧在日志中以完整 `numpy` 数组形式输出；为节省空间，示例 JSON 中以占位字符串标注其类型与形状。
- `orientation` 为四元数表示；`rotation` 为 3×3 旋转矩阵。
- `target_positions` 为三元坐标；不同时间步会随环境更新而变化（日志中可见多组样例）。
