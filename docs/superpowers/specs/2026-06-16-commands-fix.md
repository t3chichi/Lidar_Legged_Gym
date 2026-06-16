# commands[:,2] 补充与坐标系约定

日期: 2026-06-16 | 更新: 2026-06-17

## 修复: `_resample_commands` 补充 `commands[:,2]`

### 背景

`Go2LidarPDRiskNet._resample_commands` 覆写了基类方法，但缺少 `heading_command=False` 时对 `commands[:,2]` (ang_vel_yaw) 的赋值，导致该通道恒为 0。

### 实现

```python
# go2_lidar_pd_risknet.py _resample_commands
if self.cfg.commands.heading_command:
    # heading 围绕通道方向采样 (向量化)
    _SPAWN_ANGLES = torch.tensor(
        [math.pi / 2, 0.0, -math.pi / 2, math.pi],
        device=self.device, dtype=torch.float)
    if hasattr(self, "terrain_types"):
        channel_angle = _SPAWN_ANGLES[self.terrain_types[env_ids].long()]
        spread = float(getattr(self.cfg.pd_risknet, "heading_spread", 0.35))
        self.commands[env_ids, 3] = channel_angle + torch_rand_float(
            -spread, spread, (len(env_ids), 1), device=self.device).squeeze(1)
    else:
        self.commands[env_ids, 3] = 0.0
else:
    self.commands[env_ids, 2] = torch_rand_float(
        self.command_ranges["ang_vel_yaw"][0],
        self.command_ranges["ang_vel_yaw"][1],
        (len(env_ids), 1), device=self.device).squeeze(1)
```

ang_vel_yaw=[0,0] 的任务采样结果天然为 0，行为不变。

### 影响

| 任务 | heading_command | ang_vel_yaw | tracking_ang_vel scale | 修复前 commands[:,2] | 修复后 |
|------|:-:|------|:-:|------|------|
| 走廊 (lidar_pillar) | False | [-0, 0] | 0.0 | 0 | 0 (不变) |
| 软预训练 | False | [-1, 1] | 0.5 | 0 | 随机 [-1,1] |
| 旧预训练 | False | [-1, 1] | 0.5 | 0 | 随机 [-1,1] |
| 梅花桩 (lidar_pd_risknet) | False | [-0, 0] | ~ | 0 | 0 (不变) |

## 坐标系约定

`_reward_tracking_ang_vel` 中使用的 `base_ang_vel` 轴因机器人姿态而异，**非 bug，是两种有效约定**:

### 行走姿态

```
body X = 前进, body Y = 左, body Z = 上 = 世界 Z
偏航轴 = body Z → base_ang_vel[:, 2]
```

所有行走类使用此约定:
- `LeggedRobot` (基类, `legged_robot_rew_mixin.py:233`)
- `Go2LidarPDRiskNet` (未覆写，继承基类)
- `Go2LidarPDRiskNet` 其他偏航相关奖励: `_reward_rays`, `_reward_curvature`, `_reward_ang_vel_yaw_penalty` 均使用 `[:, 2]`

### 站立姿态

```
body X = 上 = 世界 Z, body Y = 左, body Z = 后
偏航轴 = body X → base_ang_vel[:, 0]
```

所有 Stand 变体使用此约定，三个方法自洽:

| 方法 | 轴 | 站立姿态含义 |
|------|:---:|------|
| `_reward_orientation` | `projected_gravity[:, 1:]` → 0 | body X 朝上 |
| `_reward_ang_vel_xy` | `base_ang_vel[:, 1:]` | 惩罚 body Y+Z，允许绕 X 旋转 |
| `_reward_tracking_ang_vel` | `base_ang_vel[:, 0]` | body X 追踪世界偏航 |

涉及的类 (均正确，无需修改):
- `StandGo2` (`go2.py:280`)
- `StandAnymal` (`anymal.py:285`)
- `StandElSpider` (`elspider.py:708`)

## 修改范围

| 文件 | 改动 | 状态 |
|------|------|:---:|
| `go2_lidar_pd_risknet.py` | `_resample_commands` 补充 else 分支 | ✅ 已应用 |
| 其余 `_reward_tracking_ang_vel` 覆写 | 无改动 (站立坐标系，[:,0] 正确) | — |
