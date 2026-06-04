# 四方向通道 + 通道感知 heading 设计

## 问题

当前梯形通道训练存在"去 Y 轴泛化"失败：策略部署到 pillar 场景（360 度随机出生朝向）后，所有机器人转向 +Y 方向行走，无视 heading 命令。

**根因**：通道始终沿世界 +Y 方向，出生朝向固定 pi/2（面朝 +Y）。策略学到基于 proprioceptive 观测（重力方向等）的"朝北走"捷径，而非真正根据 LiDAR 感知和 heading 命令导航。

## 方案

方案 B：四方向离散通道（北/东/南/西），通道方向与 `num_cols=4` 对齐。heading 命令围绕通道方向 +/-20 度采样。

## 改动清单

### 1. 地形生成 (`legged_gym/utils/terrain.py`)

**`trapezoid_corridor_terrain` 函数改造**：

- 新增参数 `direction`（int, 0/1/2/3），由 `make_terrain` 根据 `j % 4` 传入
- 原始 polyline waypoints（沿 +Y 生成）通过 rot(angle) 变换到目标方向
- 墙壁高度场在旋转后的坐标上构建，90 度整数倍旋转保证网格对齐

**方向映射表**：

| direction | 通道走向 | spawn_angle | forward 向量 | 旋转角 theta |
|-----------|----------|-------------|-------------|-------------|
| 0         | +Y (北)  | pi/2        | (0, 1)      | 0           |
| 1         | +X (东)  | 0           | (1, 0)      | pi/2        |
| 2         | -Y (南)  | -pi/2       | (0, -1)     | pi          |
| 3         | -X (西)  | pi          | (-1, 0)     | 3pi/2       |

- `goal_offset_x/y` 根据旋转后的终点位置计算：
  - 原始终点（direction 0, +Y 方向）：`(0, terrain_len - corridor_width - 2*end_margin - goal_forward_margin)`
  - 旋转后：`(x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta))`
- `goal_radius` 保持不变

**`make_terrain` 改造**：

```python
direction = j % 4  # col 索引即通道方向
trapezoid_corridor_terrain(terrain, difficulty, self.cfg, direction=direction)
```

### 2. 环境配置 (`go2_lidar_pd_risknet_config.py`)

- heading 命令范围：围绕通道方向 +/-20 度（约 +/-0.35 rad），采样时由环境代码处理中心偏移
- `heading_command = True`（保持）
- 移除 `y_progress` reward scale（已为 0.0，清理冗余声明）

### 3. 通道方向感知 (`go2_lidar_pd_risknet.py`)

**新增 buffer**：`_channel_forward` — `(num_envs, 2)`，每个环境的通道前进方向单位向量。

**方向查找表**：

```python
# terrain_type (col 索引) -> (forward_x, forward_y), spawn_angle
_FORWARD_LOOKUP = [
    (0.0, 1.0),    # direction 0: +Y, spawn_angle = pi/2
    (1.0, 0.0),    # direction 1: +X, spawn_angle = 0
    (0.0, -1.0),   # direction 2: -Y, spawn_angle = -pi/2
    (-1.0, 0.0),   # direction 3: -X, spawn_angle = pi
]
```

初始化：
```python
self._channel_forward = torch.tensor(
    [_FORWARD_LOOKUP[t.item()] for t in self.terrain_types],
    device=self.device, dtype=torch.float
)
```

**`_resample_commands` override**（Go2LidarPDRiskNet 新增）：

```python
def _resample_commands(self, env_ids):
    # 调用父类逻辑获取基础速度命令
    super()._resample_commands(env_ids)
    # heading 围绕通道方向采样
    spawn_angles = [pi/2, 0, -pi/2, pi]
    channel_angle = torch.tensor(
        [spawn_angles[t.item()] for t in self.terrain_types[env_ids]],
        device=self.device, dtype=torch.float
    )
    spread = 0.35  # +/-20 degrees
    self.commands[env_ids, 3] = channel_angle + torch_rand_float(
        -spread, spread, (len(env_ids), 1), device=self.device
    ).squeeze(1)
```

**删除 `_reward_y_progress`**：函数体、reward scale 声明、`self.last_y` buffer 一并移除。

**保留但无需改动**：`_reward_move_distance`、`check_termination`、`_reward_goal`（已通过 `goal_offset_x/y` 支持二维）。

### 4. 地形课程 (`_update_terrain_curriculum`)

`forward_dist` 从 Y 轴投影改为通道方向投影：

```python
# 原来
forward_dist = self.root_states[env_ids, 1] - self.env_origins[env_ids, 1]

# 改为
delta_xy = self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]
forward_dist = torch.sum(delta_xy * self._channel_forward[env_ids], dim=1)
```

`goal_dist` 对应修改为 `goal_offset` 沿通道方向的投影长度。

注意：`_channel_forward` 在 `_get_env_origins` 之后初始化一次即可，无需在 `reset_idx` 中更新——`terrain_types`（col 索引/通道方向）在环境生命周期内不变，只有 `terrain_levels`（难度行）会随课程升降级改变。

### 5. 测试

- 更新 `test_go2_lidar_pd_risknet_math.py` 中的 `spawn_angle` 测试：四方向分别验证
- 新增方向旋转后 goal_offset 的几何测试

## 不修改的设计

- 不改动 `_reward_move_distance`（距离范数本身无方向性）
- 不改动 `check_termination` / `_reward_goal`（已通过 `goal_offset_x/y` 支持任意方向）
- 不改动 LiDAR 传感器配置（传感器是本体的，与通道方向无关）
- 不删除 heading 命令机制（保留为操控接口）

## 预期效果

1. 训练后策略在 pillar 场景 360 度随机出生下，跟随 heading 命令前进而非转向 +Y
2. heading 命令 +/-20 度范围内精确跟踪，LiDAR 处理局部避障
3. 后续添加任意方向场景时，方向查找表机制可直接复用
