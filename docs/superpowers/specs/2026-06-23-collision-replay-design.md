# Collision Replay Mechanism — Design Spec

Date: 2026-06-23
Source: 参考 SEA-Nav (SEA-Nav-Code) 的碰撞回放机制设计

## Overview

在 Go2 LiDAR PD-RiskNet 训练中引入碰撞回放机制：
- 区分软/硬碰撞（腿接触 vs 身体/头部接触）
- 硬碰撞触发终止惩罚
- 软碰撞保留"继续存活"机会，按概率触发回放（从碰撞前状态重新尝试）
- 引入 `terminate_buf` 将终止信号和重置信号分离

## Termination Signal Design

```
终止原因                  terminate_buf    reset_buf    惩罚 (scale=-10)
──────────────────────────────────────────────────────────────────
硬碰撞 (base/head撞墙)        ✅              ✅           -10
early_reset (腿碰撞概率触发)    ✅              ✅           -10
跌倒 (翻转或底盘过低)          ✅              ✅           -10
巨大冲击 (>50N任意部位)        ❌              ✅            0
到达目标                       ❌              ✅            0
静止超时                       ❌              ✅            0
episode超时                    ❌              ✅            0
spawn碰撞 (刚出生)             ❌              ✅            0  (bad_mask屏蔽)
```

## Collision Detection

### Body Part Classification

| 部位 | 配置来源 | 索引 | 含义 |
|------|---------|------|------|
| base, Head_upper, Head_lower | `terminate_after_contacts_on` | `termination_contact_indices` | 硬碰撞 → 直接终止 |
| thigh, calf, Head_upper, Head_lower, base | `penalize_contacts_on` | `penalised_contact_indices` | 所有障碍接触 → 回放碰撞记录 + 连续惩罚 |

检测条件: `||Force_xy|| > 1.0`（水平面接触力标量）

### Initial-Step Protection

- `initial_ = episode_length_buf <= 1`: spawn 后的前 2 步标记为 "初始化步骤"
- 这些步骤中不触发碰撞检测、不施加终止惩罚
- 通过 `extras["bad_masks"] = initial_` 让 PPO 在计算 loss 时跳过这些 transition

## Replay Mechanism

### Replay Trigger Conditions (in reset_idx)

```python
wants_replay = (
    enable_collision_replay          # 总开关
    & (rand() < replay_prob)         # 80% 概率 (20% 正常重置做探索)
    & collision_occurred             # episode 中有过任何碰撞
    & ~goal_reached                  # 不是成功到达目标 (成功=目标区域连续3秒)
    & ~time_out                      # 不是超时
)
```

### Early Reset (in check_termination)

- 仅在碰撞首帧 (`is_new_collision`) 触发，避免连续帧反复触发
- 概率随地形难度增长: `early_prob = 0.1 + (0.5 - 0.1) * (terrain_levels / max_terrain_level).clip(max=1.0)`
- 触发时: `terminate_buf |= 1`, `reset_buf |= 1`
- 效果: 腿碰到障碍物时有 10%~50% 概率直接终止+惩罚 (取决于难度)

### Replay State Buffer

- 滚动缓冲区: `[num_envs, replay_len=100, dim]`
- 存储: `root_states [13]`, `dof_pos [12]`, `dof_vel [12]`
- 每步 `_update_replay_buffer()` 追加当前状态
- 新 episode 前两步用广播填充，避免读到上一个 episode 的脏数据

### Replay Execution

- 回退步数: `randint(100, 150)` → min(当前 episode 长度, replay_len-1)
- 最低要求: `undo_steps > 20` (历史不够则走正常重置)
- 恢复: `root_states[idx, -undo_steps]`, `dof_pos`, `dof_vel` → 同步到 Isaac Gym
- episode_length_buf 减去 undo_steps (保持一致性)
- 回放后 LiDAR 历史通过下一帧 `_update_lidar_history` 自然重建

## Config Changes

```python
# go2_lidar_pd_risknet_config.py

class asset:
    terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]
    penalize_contacts_on = ["thigh", "calf", "Head_upper", "Head_lower", "base"]

class replay:                    # 新增
    enable_collision_replay = True
    replay_prob = 0.8
    early_reset_prob_range = [0.1, 0.5]
    undo_steps_range = [100, 150]
    max_collision_points = 10

class rewards:
    class scales:
        termination = -10.0       # 从 -0.0 改为 -10.0
```

## Implementation Checklist

### go2_lidar_pd_risknet_config.py
- [ ] 添加 `replay` 配置类
- [ ] 修改 `asset.terminate_after_contacts_on`
- [ ] 添加 `asset.penalize_contacts_on`
- [ ] 修改 `rewards.scales.termination`

### go2_lidar_pd_risknet.py

- [ ] `_init_replay_buffers()` — 新方法: 滚动缓冲区 + 碰撞标志
- [ ] `_update_replay_buffer()` — 新方法: 滚动更新状态缓冲区
- [ ] `_post_physics_step_callback()` — 首行追加 `self._update_replay_buffer()`
- [ ] `check_termination()` — 追加: 碰撞检测、early_reset、terminate_buf 管理、bad_masks
- [ ] `_reward_termination()` — 覆盖: 改为 `self.terminate_buf.float()`
- [ ] `_reward_collision()` — 改用 `self.penalised_contact_indices`
- [ ] `reset_idx()` — 覆盖: replay/normal 分流逻辑
- [ ] `_reset_collision_replay()` — 新方法: 从缓冲区恢复状态

### 不修改的文件
- `legged_robot.py` (基类)
- `go2.py`
- `ppo.py`
- `rsl_rl/modules/`
- `LidarSensor/`
