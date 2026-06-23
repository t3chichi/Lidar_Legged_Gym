# Stuck Detection — Design Spec

Date: 2026-06-23
Source: 参考 SEA-Nav `legged_robot_pos.py:606-625` 的卡住检测机制

## Overview

在 Go2 LiDAR PD-RiskNet 中添加卡住检测：当机器人长时间无法有效移动时，提前终止 episode 以止损，不施加终止惩罚。

## Design

### Parameters

| 参数 | 值 | 说明 |
|------|-----|------|
| `pos_hist` 长度 | 10 帧 | 位置历史缓冲区 |
| 更新频率 | 每 10 步 | 10×10=100 步 ≈ 2 秒回看窗口 |
| `v_low` 速度阈值 | `||v_xy|| < 0.1` 且 `|ω_z| < 0.1` | 几乎完全静止 |
| `d_low` 位移阈值 | `||pos - pos_100步前|| < 0.2m` | 2 秒内移动不到 20cm |
| `stay_timer` 阈值 | 150 步 = 3 秒 | 连续累计静止触发重置 |
| 终止惩罚 | 无 | 仅止损，不惩罚 |

### Stuck Criteria (any one is sufficient)

```python
v_low = ||base_lin_vel_xy|| < 0.1 AND |ang_vel_z| < 0.1   # 瞬时静止
d_low = ||current_pos - pos_100_steps_ago|| < 0.2           # 长期无位移
static = (v_low | d_low) AND episode_progress > 10%         # 排除刚出生
stay_timer += static
stand_still_flag = stay_timer >= 150
reset_buf |= stand_still_flag   # 仅重置，不设 terminate_buf
```

`v_low` 覆盖瞬时卡死（腿被夹立刻不动），`d_low` 覆盖原地打转（有速度无位移）。

### Termination Signal

```
终止原因                 terminate_buf    reset_buf    惩罚
────────────────────────────────────────────────────────────
卡住 (stand_still_flag)        ❌              ✅        0
```

## Implementation

### Files Modified

仅 `go2_lidar_pd_risknet.py`，三处改动：

1. **`_init_buffers`**: 添加 `pos_hist` [num_envs, 10, 2] + `stay_timer` [num_envs]
2. **`_post_physics_step_callback`**: 每 10 步滚动更新 `pos_hist`
3. **`check_termination`**: 追加卡住判定 + `reset_buf` 设置
4. **`reset_idx`**: 公共清理区追加 `stay_timer[env_ids] = 0` + `pos_hist[env_ids] = 0`

### 不修改的文件

- 配置文件（参数硬编码，不需要配置项）
- 基类、Go2 父类、PPO 算法
