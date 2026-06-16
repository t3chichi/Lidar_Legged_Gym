# 软预训练柱子网格修复

日期: 2026-06-16 | 状态: 已确认

## 问题

`generate_pillar_lidar_mesh` 使用 terrain 参数（`num_rows/cols/terrain_length/width`，4×4×15m=60m 网格）
与平面模式的实际 env 布局（`sqrt(N)×env_spacing`，32envs→5×7×3m=15m×21m）坐标系不匹配，
导致大部分 env 看不到柱子。

## 修复

### 1. 坐标对齐

`generate_pillar_lidar_mesh` 签名改为：
```python
def generate_pillar_lidar_mesh(num_envs, env_spacing, pd_cfg, device):
```

内部基于实际 env 布局：
```python
num_cols = int(np.floor(np.sqrt(num_envs)))
num_rows = int(np.ceil(num_envs / num_cols))
spacing = env_spacing

ground: [0, num_cols×spacing] × [0, num_rows×spacing]
per env: center = (col×spacing+spacing/2, row×spacing+spacing/2)
         spawn_radius = min(spacing/2 * 0.85, pillar_spawn_radius)
         clear_radius = min(pillar_center_clear_radius, spawn_radius * 0.15)
```

### 2. 柱子可视化

返回值新增 pillar_boxes 列表：
```python
return vertices, triangles_np, pillar_boxes
# pillar_boxes: list of (env_idx, cx, cy, sx, sy, h)
```

`_init_lidar_sensor` 存储 `self._pillar_boxes`。
`_draw_debug_vis` 中为当前 env 画线框（`gymutil.draw_lines`）。

### 3. 修改范围

| 文件 | 改动 |
|------|------|
| `pillar_mesh.py` | 重写 `generate_pillar_lidar_mesh` |
| `go2_lidar_pd_risknet.py` | 调用签名 + 存储 + debug_vis 线框 |
