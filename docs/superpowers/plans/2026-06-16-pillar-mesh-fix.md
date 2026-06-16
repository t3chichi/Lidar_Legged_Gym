# Pillar Mesh Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于实际 env 布局重建柱子网格，并增加柱子线框可视化。

**Architecture:** `generate_pillar_lidar_mesh` 改用 `num_envs + env_spacing` 替代 terrain 参数，每个 env 一个柱子簇。新增 `pillar_boxes` 返回列表供 debug_vis 画线框。

**Tech Stack:** Python, NumPy, PyTorch, Isaac Gym gymutil

---

### Task 1: 重写 `generate_pillar_lidar_mesh`

**Files:**
- Modify: `legged_gym/legged_gym/utils/pillar_mesh.py`

- [ ] **Step 1: 替换函数签名和实现**

将 `generate_pillar_lidar_mesh` 完整替换为：

```python
def generate_pillar_lidar_mesh(num_envs, env_spacing, pd_cfg, device='cuda:0'):
    """Build a ground plane + per-env pillar clusters matching the actual env layout.

    Uses the same sqrt-grid layout as _get_env_origins for plane terrain.

    Args:
        num_envs: total number of environments.
        env_spacing: spacing between env origins (meters).
        pd_cfg: pd_risknet config with pillar_* parameters.
        device: torch device string.

    Returns:
        (vertices, triangles_np, pillar_boxes)
        pillar_boxes: list of (env_idx, cx, cy, sx, sy, h) in world meters.
    """
    num_cols = int(np.floor(np.sqrt(num_envs)))
    num_rows = int(np.ceil(num_envs / num_cols))
    spacing = env_spacing

    total_x = num_cols * spacing
    total_y = num_rows * spacing
    border = spacing  # extra margin so border envs still see ground

    # Ground plane
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

    spawn_r = float(pd_cfg.pillar_spawn_radius)
    clear_r = float(pd_cfg.pillar_center_clear_radius)
    min_sep = float(pd_cfg.pillar_min_separation)
    count = int(pd_cfg.pillar_count)

    # Clamp spawn/clear to env cell size
    spawn_radius = min(spacing / 2.0 * 0.85, spawn_r)
    clear_radius = min(clear_r, spawn_radius * 0.15) if clear_r > 0 else 0.0

    rng = np.random.RandomState(42)

    for row in range(num_rows):
        for col in range(num_cols):
            env_idx = row * num_cols + col
            if env_idx >= num_envs:
                break

            center_x = col * spacing + spacing / 2.0
            center_y = row * spacing + spacing / 2.0

            pillars = generate_pillar_positions(
                center_x=center_x, center_y=center_y,
                spawn_radius=spawn_radius,
                clear_radius=clear_radius,
                min_separation=min_sep,
                count=count,
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
                pillar_boxes.append((env_idx, cx, cy, sx, sy, h))

    vertices_np = np.concatenate(all_verts, axis=0).astype(np.float32)
    triangles_np = np.concatenate(all_tris, axis=0).astype(np.int32)
    vertices = torch.as_tensor(vertices_np, device=device, dtype=torch.float32)
    return vertices, triangles_np, pillar_boxes
```

- [ ] **Step 2: 语法检查**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/utils/pillar_mesh.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/utils/pillar_mesh.py
git commit -m "fix: align pillar mesh with actual env layout + return pillar_boxes for debug vis"
```

---

### Task 2: 更新 `_init_lidar_sensor` 调用 + 存储 pillar_boxes

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 更新调用签名**

找到 `generate_pillar_lidar_mesh(self.cfg.terrain, pd_cfg, device=self.device)` 这一行，替换为：

```python
                vertices, triangles_i32, self._pillar_boxes = generate_pillar_lidar_mesh(
                    self.num_envs, self.cfg.env.env_spacing, pd_cfg, device=self.device)
```

- [ ] **Step 2: 添加初始化默认值**

在 `_init_pd_risknet_buffers` 中，其他 self._xxx 初始化附近添加：

```python
        self._pillar_boxes = []  # [(env_idx, cx, cy, sx, sy, h), ...] for debug viz
```

- [ ] **Step 3: 语法检查 + 提交**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('Syntax OK')"
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "fix: update pillar mesh call signature + store pillar_boxes"
```

---

### Task 3: 增加柱子线框可视化

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: `_draw_debug_vis` 新增柱子线框**

在 `_draw_debug_vis` 方法末尾（`return` 之前）添加：

```python
        # Draw pillar bounding boxes for current viewed env
        if hasattr(self, '_pillar_boxes') and self._pillar_boxes:
            env_id = self.visual_env_id if hasattr(self, 'visual_env_id') else 0
            color = [0.0, 0.6, 0.8]  # cyan
            color_vec = gymapi.Vec3(*color)
            for env_idx, cx, cy, sx, sy, h in self._pillar_boxes:
                if env_idx != env_id:
                    continue
                x0, x1 = cx - sx / 2.0, cx + sx / 2.0
                y0, y1 = cy - sy / 2.0, cy + sy / 2.0
                # 12 edges of a box
                edges = [
                    ([x0, y0, 0.0], [x1, y0, 0.0]), ([x1, y0, 0.0], [x1, y1, 0.0]),
                    ([x1, y1, 0.0], [x0, y1, 0.0]), ([x0, y1, 0.0], [x0, y0, 0.0]),
                    ([x0, y0, h],   [x1, y0, h]),   ([x1, y0, h],   [x1, y1, h]),
                    ([x1, y1, h],   [x0, y1, h]),   ([x0, y1, h],   [x0, y0, h]),
                    ([x0, y0, 0.0], [x0, y0, h]),   ([x1, y0, 0.0], [x1, y0, h]),
                    ([x1, y1, 0.0], [x1, y1, h]),   ([x0, y1, 0.0], [x0, y1, h]),
                ]
                for a, b in edges:
                    p0 = gymapi.Vec3(*a)
                    p1 = gymapi.Vec3(*b)
                    self.gym.add_lines(self.viewer, self.envs[env_id], 1.0, [p0, p1], color_vec)
```

- [ ] **Step 2: 语法检查 + 提交**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('Syntax OK')"
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add pillar wireframe visualization in debug view"
```

---

### Task 4: 调整 soft_pretrain pillar 参数（可选）

柱子参数从正式梅花桩搬来（spawn_radius=9m, count=30），在新布局下每 env 只占 3m×3m 空间，需调整。

- [ ] **Step 1: 降低 count + radius（用户自己调 config）**

建议初始值：`pillar_count=5, pillar_spawn_radius=1.2`，按需增加。
