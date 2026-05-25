# v_avoid 对齐论文设计

## 目标

将 `_compute_v_avoid` 和 `_reward_vel_avoid` 对齐 Omni-Perception 论文第 3.3.2 节公式。

## 论文公式

```
36 扇区 (10°/扇区)
扇区 j: d_j = min(该扇区所有 LiDAR 点距离)
if d_j < d_thresh (1.0m):
    V_j = exp(-d_j × α) × (-扇区_direction)
V_avoid = Σ_j V_j  (所有扇区向量和)

r_vel,avoid = exp(-β × ||v - (v_cmd + V_avoid)||²)
```

## 当前 vs 论文

| 维度 | 当前 | 改为 |
|------|------|------|
| 每扇区距离 | 第10百分位 | min() |
| 聚合方式 | 迭代取最大(3轮) | 全部扇区向量和 |
| v_cmd投影调制 | exp(-d×α) × C_i | exp(-d×α) |
| avoid_iters | 3 | 移除 |
| avoid_gain | 1.1 | 移除 |
| n_sectors (pillar) | 24 | 36 |
| avoid_distance_thresh (pillar) | 1.6 | 1.0 |

## 不改

- `_reward_vel_avoid` 公式不变（已对齐）
- `avoid_alpha = 1.5`, `avoid_beta = 1.0`
- `avoid_distances`（地面滤除、无域随机化）作为输入不变
- 扇区分割逻辑不变

## 新 `_compute_v_avoid` 伪代码

```python
def _compute_v_avoid(self):
    n_sec = 36
    sec_size = 2π / 36
    pts = lidar_points_base[..., :2]
    dist = avoid_distances  # 地面滤除，无域随机化

    angles = atan2(pts[..., 1], pts[..., 0])
    sec_ids = floor((angles + π) / sec_size).clamp(0, 35)

    # 每扇区取 min 距离
    inf = 1e9 * ones_like(dist)
    min_dist_per_sec = [min(where(sec_ids==s, dist, inf), dim=1) for s in range(36)]

    # 仅 d < 1.0m 的扇区产生避障速度
    active = min_dist < 1.0
    mag = exp(-min_dist * α) * active.float()

    # 扇区中心方向的反方向
    away_dirs = [(-cos(center), -sin(center)) for center in sec_centers]

    # 向量和
    v_avoid = Σ_j mag[:,j] * away_dirs[j]
```

## 配置改动

| 配置 | 改动 |
|------|------|
| pillar | n_sectors 24→36, avoid_distance_thresh 1.6→1.0, 移除 avoid_iters/avoid_gain |
| risknet | 移除 avoid_iters/avoid_gain |
| pretrain | 移除 avoid_iters/avoid_gain |

## 验证

- 机器人靠近墙壁时应产生远离墙壁的 V_avoid
- 开阔区域 V_avoid ≈ 0
- 两个方向同时有障碍时合力应指向远离两者的方向
