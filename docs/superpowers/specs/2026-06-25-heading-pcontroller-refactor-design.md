# 设计文档：heading/P控制器重构

日期: 2026-06-25

## 目标

删除观测中的本体朝向(target_heading)和当前朝向(current_heading)，改为使用 P 控制器将 heading 命令转换为 ang_vel_yaw，完全对齐原版 legged_gym 的数据流。

## 背景

基类 `LeggedRobot._post_physics_step_callback` (legged_robot.py:400-405) 已包含 P 控制器逻辑：

```python
if self.cfg.commands.heading_command:
    forward = quat_apply(self.base_quat, self.forward_vec)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    p_gain = float(getattr(self.cfg.commands, "heading_p_gain", 0.5))
    self.commands[:, 2] = torch.clip(p_gain*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
```

当前仅需：(1) 开启 `heading_command = True`；(2) 清理观测中的 heading 相关代码；(3) 更新维度常量。

## 数据流（对齐后）

```
heading_command = True:
  _resample_commands → 采样 commands[:, 3] (目标 heading, 围绕通道方向)
  _post_physics_step_callback (每步, 基类) → P控制器:
      commands[:, 2] = clip(0.5 * wrap_to_pi(target - current), -1, 1)
  compute_observations → commands[:, :3] (lin_vel_x, lin_vel_y, ang_vel_yaw)

heading_command = False:
  _resample_commands → 直接采样 commands[:, 2] (ang_vel_yaw)
  compute_observations → commands[:, :3]
```

## 修改清单

### go2_lidar_pd_risknet_config.py

1. `HEADING_OBS_ENABLED = False` — **删除**
2. `PD_PROPRIO_DIM` — 改为 `48`
3. `PD_PRIV_CRITIC_DIM` — 改为 `48 + PD_PRIV_HEIGHT_DIM`
4. `pd_risknet` 类 — **删除** `heading_obs_enabled`, `heading_noise_enabled`, `heading_noise_std`
5. `commands.heading_command` — 改为 `True`
6. `commands.ranges` — 添加 `heading = [-3.14, 3.14]`
7. `normalization.obs_scales` — **删除** `heading = 1.0`
8. `env.num_observations` — 更新为 `48 + ...`
9. `env.num_privileged_obs` — 更新为 `48 + 187`
10. `policy.proprio_obs_dim` — 更新为 `48`
11. `policy.privileged_critic_dim` — 更新为 `48 + 187`

### go2_lidar_pd_risknet.py

1. `_get_noise_scale_vec` — 删除 heading 分支，统一 48 维
2. `compute_observations` — 删除 `heading_obs_enabled` 分支，统一 `commands[:, :3]` 格式

## 不需要修改的部分

- `_post_physics_step_callback`: 基类已实现 P 控制器
- `_resample_commands`: 已支持两种模式
- `wrap_to_pi`: 已存在于 `math_utils.py`
