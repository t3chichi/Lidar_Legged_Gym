# 四方向通道 + 通道感知 heading 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将梯形通道从固定 +Y 方向改造为四方向（+Y/+X/-Y/-X），添加通道感知 heading 采样，删除 y_progress 奖励

**Architecture:** terrain.py 的 `trapezoid_corridor_terrain` 新增 `direction` 参数，通过 90 度整数倍旋转生成不同方向的通道几何。go2_lidar_pd_risknet.py 新增 `_channel_forward` buffer 和 `_resample_commands` override，将课程逻辑中的 Y 轴投影改为通道方向投影。

**Tech Stack:** Python, NumPy, PyTorch, Isaac Gym terrain_utils

---

### Task 1: 创建功能分支

**Files:**
- No file changes — git branch creation

- [ ] **Step 1: 创建分支**

```bash
cd /home/t3chichi/Lidar_legged_gym
git checkout -b feat/four-direction-corridor
```

---

### Task 2: terrain.py — `trapezoid_corridor_terrain` 支持 direction 参数

**Files:**
- Modify: `legged_gym/legged_gym/utils/terrain.py:501-651`

- [ ] **Step 1: 修改函数签名和 docstring，添加 direction 参数**

在 `trapezoid_corridor_terrain` 函数签名中添加 `direction=0`，在 cfg 参数之前（函数签名为 `trapezoid_corridor_terrain(terrain, difficulty, cfg)` → 改为 `trapezoid_corridor_terrain(terrain, difficulty, cfg, direction=0)` 或更安全的 `direction=None`）。

由于 `make_terrain` 调用处已有 `trapezoid_corridor_terrain(terrain, difficulty, self.cfg)`，直接加默认值 `direction=None` 向下兼容，内部转 `int`。

在函数体开头（参数提取后、坐标计算前）添加旋转逻辑。完整改动如下：

```python
# 将 line 501 的 def trapezoid_corridor_terrain(terrain, difficulty, cfg):
# 改为:
def trapezoid_corridor_terrain(terrain, difficulty, cfg, direction=None):
    # ... 现有参数提取保持不变, 直到 line 532 ...

    # === direction: 旋转整个通道到目标方向 ===
    # direction (int): 0=+Y(default), 1=+X, 2=-Y, 3=-X
    # 90 度整数倍旋转，网格完美对齐
    dir_idx = int(direction) if direction is not None else 0
    rot_angle = dir_idx * np.pi / 2.0
    # 标准旋转矩阵 (绕 Z 轴, 仅影响 X-Y 平面)
    cos_r = np.cos(rot_angle)
    sin_r = np.sin(rot_angle)

    # ... 继续现有的走廊几何计算、墙壁绘制到 line 638 ...
```

- [ ] **Step 2: 对 waypoints 和 goal_offset 做旋转变换**

在 `terrain.height_field_raw[in_corridor] = 0` (line 638) **之后**、`cfg.goal_offset_x = 0.0` (line 641) **之前**，插入：

```python
    # === 应用方向旋转到高度场 ===
    if dir_idx != 0:
        # 将地形中心设定为旋转中心 (以 terrain patch 中点为准)
        cx = (size_x - 1) / 2.0
        cy = (size_y - 1) / 2.0

        # 用标准旋转矩阵重新采样: 对于每个目标像素 (x_dst, y_dst),
        # 找到其在源像素坐标 (x_src, y_src) 的位置并取最近邻
        src_field = terrain.height_field_raw.copy()

        # 构建目标坐标网格
        x_dst, y_dst = np.meshgrid(
            np.arange(size_x, dtype=np.float64),
            np.arange(size_y, dtype=np.float64),
            indexing='ij',
        )

        # 逆变换: 目标坐标 -> 源坐标
        x_rel = x_dst - cx
        y_rel = y_dst - cy
        x_src = cx + x_rel * cos_r + y_rel * sin_r   # rotate by -angle
        y_src = cy - x_rel * sin_r + y_rel * cos_r

        # 最近邻插值
        x_src_idx = np.clip(np.round(x_src).astype(int), 0, size_x - 1)
        y_src_idx = np.clip(np.round(y_src).astype(int), 0, size_y - 1)
        terrain.height_field_raw = src_field[x_src_idx, y_src_idx]
```

- [ ] **Step 3: 修改 spawn_angle 和 goal_offset 跟随方向旋转**

替换 lines 640-649 (goal 和 spawn_angle 设置):

```python
    # Goal info — 原始沿 +Y 方向的终点，旋转后跟随方向
    raw_goal_y = float(terrain_len) - corridor_width - 2.0 * end_margin
    goal_forward_margin = float(getattr(cfg, "goal_forward_margin", 0.0))
    if goal_forward_margin > 0:
        raw_goal_y -= goal_forward_margin

    # 将原本 (0, raw_goal_y) 旋转 dir_idx * pi/2
    cfg.goal_offset_x = 0.0 - sin_r * raw_goal_y
    cfg.goal_offset_y = cos_r * raw_goal_y
    cfg.goal_radius = float(getattr(cfg, "goal_radius", corridor_width / 2.0))

    # Spawn angle: 跟随方向旋转
    _SPAWN_ANGLES = [np.pi / 2, 0.0, -np.pi / 2, np.pi]
    terrain.spawn_angle = _SPAWN_ANGLES[dir_idx]
```

- [ ] **Step 4: 修改 `make_terrain` 调用处传入 direction**

在 `make_terrain` 函数中 (line 163-164)，增加 `direction` 参数：

```python
        elif choice < self.proportions[7]:
            direction = j % 4
            trapezoid_corridor_terrain(terrain, difficulty, self.cfg, direction=direction)
```

---

### Task 3: 更新现有梯形通道测试

**Files:**
- Modify: `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py:354-483`

- [ ] **Step 1: 更新 `test_trapezoid_corridor_geometry` — 多方向 spawn_angle 和 goal_offset 验证**

替换该测试函数（保留 name 和 docstring），使其参数化验证四方向：

```python
def test_trapezoid_corridor_geometry():
    """Verify trapezoid corridor spawn_angle and goal_offset for all 4 directions."""
    import numpy as np
    import math
    from legged_gym.utils.terrain import trapezoid_corridor_terrain
    from isaacgym import terrain_utils

    hs = 0.1
    vs = 1.0
    size = 150

    expected = [
        # dir, spawn_angle, goal_ox_sign (0=zero, +/-1=sign), goal_oy_sign
        (0,  math.pi / 2,  0, +1),
        (1,  0.0,          +1,  0),
        (2, -math.pi / 2,   0, -1),
        (3,  math.pi,      -1,  0),
    ]

    for direction, exp_spawn, goal_ox_sign, goal_oy_sign in expected:
        class Cfg:
            corridor_width = 3.0
            wall_height = 1.5
            wall_thickness = 2.0
            turn_angle_deg_max = 55.0
            diagonal_length = 3.0
            terrain_length = 15.0
            terrain_width = 15.0
            end_margin = 0.5
            goal_forward_margin = 0.6
            goal_radius = 1.6
            curriculum = False
            _first_turn_left = True

        cfg = Cfg()
        terrain = terrain_utils.SubTerrain(f"test_dir{direction}", width=size, length=size,
                             vertical_scale=vs, horizontal_scale=hs)
        trapezoid_corridor_terrain(terrain, difficulty=0.5, cfg=cfg, direction=direction)

        # spawn_angle
        assert abs(terrain.spawn_angle - exp_spawn) < 1e-6, \
            f"dir={direction}: expected spawn_angle={exp_spawn}, got {terrain.spawn_angle}"

        # goal_offset sign checks
        if goal_ox_sign == 0:
            assert abs(cfg.goal_offset_x) < 1e-6, \
                f"dir={direction}: expected goal_offset_x=0, got {cfg.goal_offset_x}"
        elif goal_ox_sign > 0:
            assert cfg.goal_offset_x > 1.0, \
                f"dir={direction}: expected goal_offset_x > 0, got {cfg.goal_offset_x}"
        else:
            assert cfg.goal_offset_x < -1.0, \
                f"dir={direction}: expected goal_offset_x < 0, got {cfg.goal_offset_x}"

        if goal_oy_sign == 0:
            assert abs(cfg.goal_offset_y) < 1e-6, \
                f"dir={direction}: expected goal_offset_y=0, got {cfg.goal_offset_y}"
        elif goal_oy_sign > 0:
            assert cfg.goal_offset_y > 1.0, \
                f"dir={direction}: expected goal_offset_y > 0, got {cfg.goal_offset_y}"
        else:
            assert cfg.goal_offset_y < -1.0, \
                f"dir={direction}: expected goal_offset_y < 0, got {cfg.goal_offset_y}"
```

- [ ] **Step 2: 运行测试验证失败（新测试需要 direction 参数）**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_trapezoid_corridor_geometry -v
```

- [ ] **Step 3: 运行全部梯形测试确认通过**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_trapezoid_corridor_geometry legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_trapezoid_corridor_lr_rl_mirror legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_trapezoid_corridor_level0_straight -v
```

- [ ] **Step 4: 提交 terrain.py 和测试改动**

```bash
git add legged_gym/legged_gym/utils/terrain.py legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py
git commit -m "feat: support four-direction trapezoid corridors

Add direction parameter (0=+Y, 1=+X, 2=-Y, 3=-X) to trapezoid_corridor_terrain.
Waypoints, height field, spawn_angle, and goal offsets rotate with direction.
Update terrain test to verify all four directions."
```

---

### Task 4: Config — 清理 y_progress, 更新 heading 范围

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py:185-187`

- [ ] **Step 1: 删除 y_progress reward scale 和相关注释**

```python
# 删除 line 185: y_progress = 0.0 ...
# 删除 line 65: y_backward_penalty_ratio = 0.1 ...

# line 65: 删除这一行
# line 185: 删除 "y_progress = 0.0   # 消融实验..."
```

- [ ] **Step 2: 更新 heading 范围注释和恢复训练值**

heading 保持 `[0.87, 2.27]`（北向约 ±40°，会在环境代码中围绕通道方向重采样），清理测试用固定值注释：

```python
# line 131-133 改为:
            heading = [0.87, 2.27]  # 基础范围; 实际会在 _resample_commands 中围绕通道方向重采样
```

- [ ] **Step 3: 提交 config 改动**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "cleanup: remove y_progress reward, restore training heading range"
```

---

### Task 5: go2_lidar_pd_risknet.py — 添加 `_channel_forward` buffer 和 `_resample_commands` override

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 在 `_init_pd_risknet_buffers` 末尾添加 `_channel_forward` 初始化**

在 `_init_pd_risknet_buffers` 方法末尾（`_consecutive_downgrade_count` 初始化之后）添加。`terrain_types` 在父类 init 链中已就绪：

```python
        # 通道前进方向单位向量 (每环境根据 terrain_type 即 col 索引确定)
        _FORWARD_LOOKUP_TABLE = torch.tensor([
            [0.0, 1.0],    # direction 0: +Y (北)
            [1.0, 0.0],    # direction 1: +X (东)
            [0.0, -1.0],   # direction 2: -Y (南)
            [-1.0, 0.0],   # direction 3: -X (西)
        ], device=self.device, dtype=torch.float)
        self._channel_forward = _FORWARD_LOOKUP_TABLE[self.terrain_types.long()]
```

- [ ] **Step 2: 添加 `_resample_commands` override**

在 `Go2LidarPDRiskNet` 类中新增方法，覆盖父类 `LeggedRobot._resample_commands`：

```python
    def _resample_commands(self, env_ids):
        import math
        # 父类采样 lin_vel_x, lin_vel_y
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)

        # heading 围绕通道方向采样 (向量化: 无 Python 循环)
        _SPAWN_ANGLES = torch.tensor(
            [math.pi / 2, 0.0, -math.pi / 2, math.pi],
            device=self.device, dtype=torch.float)
        channel_angle = _SPAWN_ANGLES[self.terrain_types[env_ids].long()]
        spread = 0.35  # +/-20 degrees
        self.commands[env_ids, 3] = channel_angle + torch_rand_float(
            -spread, spread, (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero (复用父类逻辑)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
```

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add channel-aware heading resampling and channel_forward buffer"
```

---

### Task 6: go2_lidar_pd_risknet.py — 删除 `_reward_y_progress` 和 `last_y` buffer

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 删除 `last_y` buffer 初始化和相关代码**

删除 `_init_pd_risknet_buffers` 中的 lines 105-109:
```python
        self.last_y = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
```

删除 `reset_idx` 中的 line 522:
```python
        self.last_y[env_ids] = self.base_pos[env_ids, 1]
```

- [ ] **Step 2: 删除 `_reward_y_progress` 函数体** (lines 569-575)

```python
# 删除整个 _reward_y_progress 方法 (共 7 行)
```

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "cleanup: remove _reward_y_progress and last_y buffer"
```

---

### Task 7: go2_lidar_pd_risknet.py — 课程逻辑改为通道方向投影

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:436-504`

- [ ] **Step 1: 修改 `_update_terrain_curriculum` 中的 forward_dist 和 goal_dist**

将 lines 447-453 中基于 Y 轴的 forward/ goal 计算改为通道方向投影：

```python
        if hasattr(self.cfg.terrain, "goal_offset_y"):
            # 沿通道方向的前进距离
            delta_xy = self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]
            forward_dist = torch.sum(delta_xy * self._channel_forward[env_ids], dim=1)

            # goal 沿通道方向的投影距离
            goal_x = torch.full_like(forward_dist, self.cfg.terrain.goal_offset_x)
            goal_y = torch.full_like(forward_dist, self.cfg.terrain.goal_offset_y)
            goal_xy = torch.stack([goal_x, goal_y], dim=1)
            goal_dist = torch.sum(goal_xy * self._channel_forward[env_ids], dim=1) - self.cfg.terrain.goal_radius
```

保持后续 move_up/move_down 逻辑不变（仍用 `forward_dist > goal_dist` 等判断）。

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "fix: use channel-forward projection for terrain curriculum"
```

---

### Task 8: 运行全部测试验证

**Files:**
- No file changes — validation only

- [ ] **Step 1: 运行梯形通道测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

- [ ] **Step 2: 运行基础环境测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_env.py -v
```

- [ ] **Step 3: Python 导入检查**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg
print('Import OK')
"
```

- [ ] **Step 4: 确认所有提交在分支上且无遗漏文件**

```bash
git log --oneline main..HEAD
git status
```
