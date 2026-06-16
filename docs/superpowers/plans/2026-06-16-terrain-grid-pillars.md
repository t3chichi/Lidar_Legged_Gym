# Terrain Grid Pillars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 软预训练改用 terrain 网格布局（4×4×15m），与正式 pillar config 柱子参数完全一致。

**Architecture:** 重写 `_get_env_origins` 为 soft_pretrain 生成 terrain-grid 原点（无需 terrain 对象），柱子网格回退 terrain_cfg 签名，可视化保留。

**Tech Stack:** Python, PyTorch, Isaac Gym, NumPy

---

### Task 1: 重写 `_get_env_origins` — terrain 网格布局

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 新增 `_get_env_origins` 覆写方法**

在类中插入（位置在 `_init_buffers` 之后）：

```python
    def _get_env_origins(self):
        pd_cfg = self.cfg.pd_risknet
        if getattr(pd_cfg, "soft_pretrain", False):
            num_rows = self.cfg.terrain.num_rows
            num_cols = self.cfg.terrain.num_cols
            t_len = self.cfg.terrain.terrain_length
            t_wid = self.cfg.terrain.terrain_width

            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)

            # terrain_types = col index per env, evenly distributed
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / num_cols), rounding_mode='floor'
            ).to(torch.long)
            self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long,
                                              device=self.device)
            self.max_terrain_level = 1

            # Build manual terrain_origins from grid cell centres
            origins = torch.zeros(num_rows, num_cols, 3, device=self.device)
            for r in range(num_rows):
                for c in range(num_cols):
                    origins[r, c, 0] = c * t_len + t_len / 2.0
                    origins[r, c, 1] = r * t_wid + t_wid / 2.0

            self.terrain_origins = origins
            self.env_origins[:] = self.terrain_origins[self.terrain_levels,
                                                       self.terrain_types]
            self._spawn_angles = None
        else:
            super()._get_env_origins()
```

- [ ] **Step 2: 语法检查**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: override _get_env_origins for soft_pretrain terrain grid layout"
```

---

### Task 2: 回退 `generate_pillar_lidar_mesh` 到 terrain_cfg 签名

**Files:**
- Modify: `legged_gym/legged_gym/utils/pillar_mesh.py`

- [ ] **Step 1: 替换函数签名和实现**

将 `generate_pillar_lidar_mesh` 完整替换为 terrain_cfg 版本：

```python
def generate_pillar_lidar_mesh(terrain_cfg, pd_cfg, device='cuda:0'):
    """Build ground plane + pillar clusters per sub-terrain cell.

    Uses terrain_cfg grid (num_rows, num_cols, terrain_length, terrain_width)
    so pillar placement matches the formal pillar config.

    Returns:
        (vertices, triangles_np, pillar_boxes)
    """
    num_rows = terrain_cfg.num_rows
    num_cols = terrain_cfg.num_cols
    t_len = terrain_cfg.terrain_length
    t_wid = terrain_cfg.terrain_width
    border = getattr(terrain_cfg, 'border_size', 0.0)

    total_x = num_cols * t_len
    total_y = num_rows * t_wid

    plane_verts = np.array([
        [-border,             -border,              0.0],
        [total_x + border,    -border,              0.0],
        [total_x + border,     total_y + border,    0.0],
        [-border,              total_y + border,     0.0],
    ], dtype=np.float32)
    plane_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    all_verts = [plane_verts]
    all_tris = [plane_tris]
    vert_offset = 4
    pillar_boxes = []

    rng = np.random.RandomState(42)

    for row in range(num_rows):
        for col in range(num_cols):
            center_x = col * t_len + t_len / 2.0
            center_y = row * t_wid + t_wid / 2.0

            pillars = generate_pillar_positions(
                center_x=center_x, center_y=center_y,
                spawn_radius=pd_cfg.pillar_spawn_radius,
                clear_radius=pd_cfg.pillar_center_clear_radius,
                min_separation=pd_cfg.pillar_min_separation,
                count=pd_cfg.pillar_count,
                size_x=None, size_y=None,
                size_x_range=[pd_cfg.pillar_size_x_min, pd_cfg.pillar_size_x_max],
                size_y_range=[pd_cfg.pillar_size_y_min, pd_cfg.pillar_size_y_max],
                height_min=pd_cfg.pillar_height_min,
                height_max=pd_cfg.pillar_height_max,
                allow_height_variation=pd_cfg.pillar_allow_height_variation,
                rng=rng,
            )

            for cx, cy, sx, sy, h in pillars:
                verts, tris = build_box_mesh(cx, cy, sx, sy, h)
                all_verts.append(verts)
                all_tris.append(tris + vert_offset)
                vert_offset += 8
                pillar_boxes.append((row * num_cols + col, cx, cy, sx, sy, h))

    vertices_np = np.concatenate(all_verts, axis=0).astype(np.float32)
    triangles_np = np.concatenate(all_tris, axis=0).astype(np.int32)
    vertices = torch.as_tensor(vertices_np, device=device, dtype=torch.float32)
    return vertices, triangles_np, pillar_boxes
```

- [ ] **Step 2: 语法检查 + 提交**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/utils/pillar_mesh.py').read()); print('Syntax OK')"
git add legged_gym/legged_gym/utils/pillar_mesh.py
git commit -m "fix: revert generate_pillar_lidar_mesh to terrain_cfg-based layout"
```

---

### Task 3: 更新 `_init_lidar_sensor` 调用签名

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 调用签名改回 terrain_cfg**

找到 line ~220:
```python
                vertices, triangles_i32, self._pillar_boxes = generate_pillar_lidar_mesh(
                    self.num_envs, self.cfg.env.env_spacing, pd_cfg, device=self.device)
```

改为：
```python
                vertices, triangles_i32, self._pillar_boxes = generate_pillar_lidar_mesh(
                    self.cfg.terrain, pd_cfg, device=self.device)
```

- [ ] **Step 2: 语法检查 + 提交**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('Syntax OK')"
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "fix: update pillar mesh call back to terrain_cfg signature"
```

---

### Task 4: 集成验证

- [ ] **Step 1: 全语法检查**

```bash
python3 -c "
import ast
for f in [
    'legged_gym/legged_gym/utils/pillar_mesh.py',
    'legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py',
]:
    ast.parse(open(f).read())
    print(f'{f}: Syntax OK')
"
```
