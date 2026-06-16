# rays→ω_tracking 解耦设计

日期: 2026-06-11 | 状态: 已确认

## 概述

将 `_reward_rays` 从方向对齐奖励（cos θ）改为角速度跟踪奖励，与 `_reward_vel_avoid` 实现旋转/平移域的完全解耦。

参考来源: Hiking in the Wild (论文 III-D 节 PoseVelocityCommand 公式)

## 设计原则

- 奖励信号不入观测空间，仅作为训练监督
- 旋转域 (ω_z) 和 平移域 (v_x, v_y) 各自独立控制
- 所有内部信号 (ω_target, v_avoid) 不增加 sim-to-real gap

## 算法设计

### `_compute_rays_omega_target()` — 新增

```
输入: _smooth_dir_world (EMA 平滑后的 open_dir, 世界帧 2D 单位向量)
输出: ω_target (N,) 标量

1. 将 _smooth_dir_world 转到机体帧 → open_dir_body
2. heading_error = atan2(open_dir_body_y, open_dir_body_x)
3. ω_target = clip(k_ω × heading_error, ±ω_max)
```

### `_reward_rays()` — 重写

```
当前:  return cos(θ)  where θ = angle([1,0], open_dir_body)
新方案: ω_target = _compute_rays_omega_target()
       return exp(-|base_ang_vel[:, 2] - ω_target|²)
```

### `_reward_vel_avoid()` — 保持不变

合并形式 `exp(-β × ||v_actual - (v_cmd + v_avoid)||²)` 在平移域内语义正确，不拆分。

## 配置变更

### pd_risknet 新增参数

```python
rays_omega_gain = 0.5     # k_ω: heading_error → ω_target P 增益
rays_omega_max  = 0.5     # rad/s: 角速度指令上限
```

### rewards.scales 调整

避障地形 (go2_lidar_pd_risknet_config.py, go2_lidar_pillar_config.py):
```python
tracking_ang_vel = 0.0    # 原值 0.1，rays 接管角速度引导
```

预训练 (go2_pd_pretrain_config.py): 保持 `tracking_ang_vel = 0.5`，rays 权重为 0。

## 影响范围

| 文件 | 改动 |
|------|------|
| `go2_lidar_pd_risknet.py` | `_reward_rays` 重写 + 新增 `_compute_rays_omega_target` |
| `go2_lidar_pd_risknet_config.py` | pd_risknet 新增 2 参数 + tracking_ang_vel → 0 |
| `go2_lidar_pillar_config.py` | 同上 |
| `go2_pd_pretrain_config.py` | pd_risknet 新增 2 参数 (rays 权重保持 0) |

## 不变项

- `_compute_rays_target_dir()` — 计算逻辑不变
- `_smooth_dir_world` EMA — 保留
- `_reward_vel_avoid` — 保持合并形式
- `_compute_v_avoid` — 计算逻辑不变
- LiDAR 观测空间 — 不变
- 本体观测空间 — 不变 (48维)
- 指令空间 — 不变 (vx, vy, ω_z 3维)

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| ω_target 限制探索自由度 | 软约束 (exp 奖励)，策略可通过其他奖励项权衡 |
| 与 tracking_ang_vel 冲突 | 避障地形 tracking_ang_vel = 0 |
| 高频角速度振荡 | 现有 action_rate / curvature 惩罚抑制 |
| 开阔地形 open_dir 不稳定 | 不存在 — LiDAR 倾斜安装使前方射线天然更长 |
