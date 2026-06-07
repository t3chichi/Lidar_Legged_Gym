# Rays 方向一致性奖励设计

## 动机

当前 rays 奖励（`go2_lidar_pd_risknet.py:581-612`）通过 36 扇区 top-25% 平均距离 × cos 权重，奖励"看向远处"。这本质上是静态姿态评分，导致两个局部最优：

1. **摆头刷分**：高频小角度左右摆头，让正面反复扫过通道深处奖励峰值
2. **原地哨兵**：正面稳对远方即可拿高分，无需真正前进

改进方向：将奖励从"视线质量"转向"朝向开阔方向的运动量"。

## 设计

### 整体流程

```
扇区距离提取 (机体帧) → 平方加权方向 (机体帧) → 转向世界帧 → EMA 平滑 → 方向一致性奖励 (机体帧)
```

### 1. 扇区距离提取

- 源数据：远端射线（仰角 < 20°，`_distal_mask`），滤除天空点（`dist < d_max - 0.001`）
- 扇区划分：36 个 10° 扇区，覆盖全 360° 方位角（保持现有扇区结构）
- 每扇区取 top 20% 最远有效点的平均值作为 `d_i`
- 无效扇区（无有效命中点）直接跳过，不参与后续加权
- 运行时通过预计算的扇区索引 + `gather` 操作向量化实现

### 2. 吸引力权重

```
w_i = d_i²
```

作用：远方开阔扇区自然获得高权重，无需手工 cos 偏好。直道中 8m vs 2m 权重比 16:1。

### 3. 加权平均方向（机体帧）

```
target_dir_body = normalize( Σ(w_i · sector_dir_i) / Σw_i )
```

`sector_dir_i` 为 36 个扇区中心的单位方向向量（机体帧 2D），初始化时预计算为 buffer。

### 4. 世界帧 EMA 平滑

```
target_dir_world = quat_rotate_yaw(base_quat, target_dir_body)
smooth_dir_world = normalize( α·target_dir_world + (1-α)·smooth_dir_world_prev )
```

- 平滑因子 α 默认 0.4，可配置
- 在世界帧下平滑，避免机器人转动导致坐标系不一致
- 新增 buffer `_smooth_dir_world` (N, 2)，reset 时初始化为首次 `target_dir_world`

### 5. 方向一致性奖励

```
r_rays = (v_body · smooth_dir_body) / max(|v_body|, ε)
```

- `smooth_dir_body = quat_rotate_inverse_yaw(base_quat, smooth_dir_world)`
- `ε = 0.01`
- 取值范围 [-1, 1]：完全对准 +1，垂直 0，反向 -1
- 与速度大小完全解耦，无法通过调节速度或原地转圈刷分

## 配置变更

`go2_lidar_pd_risknet_config.py` 中 `pd_risknet` 类新增/变更：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `n_sectors` | 36 | 保持不变 |
| `ray_max_distance` | 10.0 | 保持不变 |
| `rays_top_ratio` | 0.2 | 每扇区取前 20% 最远点（新增） |
| `rays_smoothing_alpha` | 0.4 | EMA 平滑因子（新增） |
| `rays_epsilon` | 0.01 | 速度分母软化（新增） |

可移除配置项：
- `avoid_distance_thresh` / `avoid_alpha` / `avoid_beta` / `avoid_speed_limit` — 属于 `vel_avoid`，不受本次改动影响，保留。

## Buffer 变更

`_init_pd_risknet_buffers` 新增：
- `_smooth_dir_world` (N, 2)：世界帧下的平滑方向向量
- `_sector_dirs` (36, 2)：36 扇区中心方向向量（预计算）

`_init_lidar_aux` 新增：
- `_distal_ray_per_sector_index`：每扇区对应的远端射线索引列表，用于快速 `gather`

`reset_idx` 中初始化 `_smooth_dir_world[env_ids]`。

## 不变量

- `vel_avoid` 奖励保持不变，负责局部避障
- `rays` 奖励负责全局朝向引导
- 两者形成"一远一近、一全向一局部"的完备导航系统
