# 软预训练地形网格布局修复

日期: 2026-06-16 | 状态: 已确认

## 问题

当前 sqrt-grid 布局（间距 3m）与正式 pillar config 的 15m 地块不兼容，
30 根柱子无法放入 1.275m 半径内。需要改用与正式 pillar 一致的地形网格布局。

## 设计

### 三层不变
- 物理: `terrain_type='plane'` — 平地，机器人自由穿越
- LiDAR: 自定义 `wp.Mesh` — 含方柱，与正式 pillar 参数一致
- 观测: 4548 维

### env 布局

plane + `soft_pretrain=True` 时，`_get_env_origins` 基于 terrain config 网格生成原点：

```
num_rows=4, num_cols=4 → 16 个地块
每地块: terrain_length=15m × terrain_width=15m
每地块中心 = env_origin
4096 envs / 16 地块 = 256 envs/地块
同地块的 256 个 env 共享同一中心点，随机朝向
```

设置 `terrain_types = col_index`，`terrain_levels = 0`（无课程）。

### 柱子网格

`generate_pillar_lidar_mesh` 回退到基于 terrain config（`num_rows, num_cols, terrain_length, terrain_width`），每地块一个柱子簇，参数与正式 pillar config 完全一致（`pillar_count=30, spawn_radius=9m, ...`）。

### 可视化

`pillar_boxes` 返回列表，`_draw_debug_vis` 中用 numpy float32 数组调用 `gym.add_lines`。

## 修改范围

| 文件 | 改动 |
|------|------|
| `legged_robot.py` | `_get_env_origins` plane 分支增加 terrain grid 路径 |
| `pillar_mesh.py` | `generate_pillar_lidar_mesh` 恢复 `terrain_cfg` 签名 |
| `go2_lidar_pd_risknet.py` | `_init_lidar_sensor` 调用签名 + `_draw_debug_vis` |
