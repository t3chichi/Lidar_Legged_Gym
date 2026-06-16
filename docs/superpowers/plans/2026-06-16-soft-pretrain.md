# go2_soft_pretrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `go2_soft_pretrain` 任务：物理平地 + LiDAR 感知柱子，步态与避障同步学习。

**Architecture:** 物理层用 `terrain_type='plane'`，LiDAR 层用自定义 `wp.Mesh`（地面 + 随机方柱）。柱子网格生成复用 `pillar_field_terrain` 的极坐标采样逻辑，输出 3D 盒子顶点/三角面而非高度场像素。

**Tech Stack:** Python, PyTorch, Isaac Gym, NVIDIA Warp, NumPy

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|:---:|------|
| `legged_gym/legged_gym/utils/pillar_mesh.py` | 新建 | 柱子→3D 网格生成 |
| `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` | 修改 | `_init_lidar_sensor` plane 分支扩展 |
| `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_soft_pillar_pretrain.py` | 新建 | 配置文件 |
| `legged_gym/legged_gym/envs/__init__.py` | 修改 | 注册 `go2_soft_pretrain` |

---

### Task 1: 创建柱子网格生成工具

**Files:**
- Create: `legged_gym/legged_gym/utils/pillar_mesh.py`

- [ ] **Step 1: 创建 `pillar_mesh.py`**

```python
"""Generate 3D pillar meshes for LiDAR rendering on flat terrain."""

import numpy as np
import torch


def generate_pillar_positions(center_x, center_y, spawn_radius, clear_radius,
                               min_separation, count, size_x, size_y,
                               size_x_range, size_y_range,
                               height_min, height_max,
                               allow_height_variation, rng):
    """Sample pillar positions via polar coordinates with rejection sampling.

    Reuses the logic from pillar_field_terrain but returns world-frame
    position/size/height tuples instead of modifying a height field.

    Args:
        center_x, center_y: sub-terrain center in world XY (meters).
        spawn_radius, clear_radius, min_separation: meters.
        count: number of pillars to place.
        size_x, size_y: base pillar half-sizes. Overridden by ranges if given.
        size_x_range, size_y_range: [min, max] tuples.
        height_min, height_max: pillar height range (meters).
        allow_height_variation: if True, randomize height in [0.6*h, h].
        rng: np.random.RandomState for deterministic placement.

    Returns:
        list of (cx, cy, sx, sy, h) in world meters.
    """
    if size_x_range is not None:
        size_x = rng.uniform(*size_x_range)
    if size_y_range is not None:
        size_y = rng.uniform(*size_y_range)

    max_attempts = count * 100
    positions = []
    for _ in range(max_attempts):
        if len(positions) >= count:
            break
        r = rng.uniform(clear_radius, spawn_radius)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        cx = center_x + r * np.cos(theta)
        cy = center_y + r * np.sin(theta)

        if np.hypot(cx - center_x, cy - center_y) < clear_radius:
            continue
        valid = True
        for px, py in positions:
            if np.hypot(cx - px, cy - py) < min_separation:
                valid = False
                break
        if valid:
            positions.append((cx, cy))

    pillars = []
    for cx, cy in positions:
        if allow_height_variation:
            h = rng.uniform(height_min * 0.6, height_max)
        else:
            h = rng.uniform(height_min, height_max)
        sx = rng.uniform(size_x_range[0], size_x_range[1]) if size_x_range else size_x
        sy = rng.uniform(size_y_range[0], size_y_range[1]) if size_y_range else size_y
        pillars.append((cx, cy, sx, sy, h))
    return pillars


def build_box_mesh(cx, cy, sx, sy, h):
    """Build 8 vertices + 12 triangles for an axis-aligned box.

    Returns (verts_8x3, tris_12x3) in world frame.
    """
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = 0.0, h

    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)

    tris = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [2, 3, 7], [2, 7, 6],  # back
        [3, 0, 4], [3, 4, 7],  # left
    ], dtype=np.int32)

    return verts, tris


def generate_pillar_lidar_mesh(terrain_cfg, pd_cfg, device='cuda:0'):
    """Build a single wp.Mesh containing a ground plane + random pillars for all sub-terrains.

    Args:
        terrain_cfg: terrain config with num_rows, num_cols, terrain_length,
                     terrain_width, border_size, horizontal_scale.
        pd_cfg: pd_risknet config with pillar_* parameters.
        device: torch device string.

    Returns:
        vertices (torch.Tensor), triangles (np.ndarray) ready for wp.Mesh.
    """
    num_rows = terrain_cfg.num_rows
    num_cols = terrain_cfg.num_cols
    t_len = terrain_cfg.terrain_length
    t_wid = terrain_cfg.terrain_width
    border = getattr(terrain_cfg, 'border_size', 0.0)

    # Ground plane — covers the full terrain area.
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

    vertices_np = np.concatenate(all_verts, axis=0).astype(np.float32)
    triangles_np = np.concatenate(all_tris, axis=0).astype(np.int32)

    vertices = torch.as_tensor(vertices_np, device=device, dtype=torch.float32)
    return vertices, triangles_np
```

- [ ] **Step 2: 验证模块可导入**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/utils/pillar_mesh.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/utils/pillar_mesh.py
git commit -m "feat: add pillar mesh generation utility for LiDAR rendering"
```

---

### Task 2: 修改 `_init_lidar_sensor` 增加软预训练 mesh 路径

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 扩展 plane 分支**

在 `_init_lidar_sensor` 方法中，找到 `elif self.cfg.terrain.mesh_type == "plane":` 分支。在该分支内，plane_size 平面构建之前，增加软预训练检测：

```python
        elif self.cfg.terrain.mesh_type == "plane":
            pd_cfg = self.cfg.pd_risknet
            if getattr(pd_cfg, "soft_pretrain", False):
                from legged_gym.utils.pillar_mesh import generate_pillar_lidar_mesh
                vertices, triangles_i32 = generate_pillar_lidar_mesh(
                    self.cfg.terrain, pd_cfg, device=self.device)
            else:
                plane_size = 100.0
                vertices = torch.tensor(
                    [
                        [-plane_size, -plane_size, 0.0],
                        [plane_size, -plane_size, 0.0],
                        [plane_size, plane_size, 0.0],
                        [-plane_size, plane_size, 0.0],
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                triangles_i32 = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
```

- [ ] **Step 2: 语法检查**

```bash
python3 -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add soft_pretrain pillar mesh path to _init_lidar_sensor"
```

---

### Task 3: 创建 `go2_soft_pillar_pretrain.py` 配置文件

**Files:**
- Create: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_soft_pillar_pretrain.py`

- [ ] **Step 1: 创建配置文件**

```python
from legged_gym.envs.go2.flat.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO


OBS_HISTORY_LENGTH = 1
PROX_HISTORY_LENGTH = 10
DIST_HISTORY_LENGTH = 10
PD_SPHERICAL_AZIMUTH = 50
PD_SPHERICAL_ELEVATION = 30
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION
PD_PROXIMAL_POINTS = 512
PD_DISTAL_POINTS = 256
PD_PROXIMAL_FEATURE_DIM = 187
PD_DISTAL_FEATURE_DIM = 64
HEADING_OBS_ENABLED = False

PD_PROPRIO_DIM = 48 + (1 if HEADING_OBS_ENABLED else 0)
PD_THETA_DEG = 20.0
MEASURED_GRID_X_COUNT = 17
MEASURED_GRID_Y_COUNT = 11
MEASURED_GRID_X_RANGE = [-0.0, 2.0]
MEASURED_GRID_Y_RANGE = [-0.7, 0.7]
PD_PRIV_HEIGHT_DIM = MEASURED_GRID_X_COUNT * MEASURED_GRID_Y_COUNT
PD_PRIV_CRITIC_DIM = PD_PROPRIO_DIM + PD_PRIV_HEIGHT_DIM


class Go2SoftPillarPretrainCfg(Go2RoughCfg):
    class init_state(Go2RoughCfg.init_state):
        pos = [0.0, 0.0, 0.34]
        randomize_rot = True
        rot_randomization_range = [-3.1415, 3.1415]

    class pd_risknet:
        enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = PD_THETA_DEG

        heading_obs_enabled = HEADING_OBS_ENABLED
        heading_noise_enabled = True
        heading_noise_std = 0.05

        n_sectors = 36
        avoid_distance_thresh = 1.0
        avoid_alpha = 2.0
        avoid_beta = 1.0
        avoid_speed_limit = 1.0

        # rays → ω_target
        rays_omega_gain = 0.5
        rays_omega_max  = 0.5
        ray_max_distance = 10.0

        rays_top_ratio = 0.4
        rays_power = 4
        rays_smoothing_alpha = 0.2

        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        num_lidar_points = PD_NUM_LIDAR_POINTS

        # channel_forward
        channel_backward_ratio = 0.5

        # 软预训练标志
        soft_pretrain = True

        # 柱子参数（与正式梅花桩一致，可在 config 中调整）
        pillar_count = 30
        pillar_spawn_radius = 9.0
        pillar_size_x_min = 0.40
        pillar_size_x_max = 0.60
        pillar_size_y_min = 0.40
        pillar_size_y_max = 0.60
        pillar_height_min = 0.60
        pillar_height_max = 1.00
        pillar_min_separation = 2.5
        pillar_center_clear_radius = 1.6
        pillar_allow_height_variation = True

        collision_3d = False

    class env(Go2RoughCfg.env):
        num_observations = PD_PROPRIO_DIM + OBS_HISTORY_LENGTH * PD_NUM_LIDAR_POINTS * 3
        num_privileged_obs = PD_PRIV_CRITIC_DIM
        enable_fall_termination = False
        fall_projected_gravity_z_threshold = -0.1
        fall_base_height_threshold = 0.12

    class terrain(Go2RoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = True
        measured_grid_x_range = MEASURED_GRID_X_RANGE
        measured_grid_y_range = MEASURED_GRID_Y_RANGE
        measured_grid_x_count = MEASURED_GRID_X_COUNT
        measured_grid_y_count = MEASURED_GRID_Y_COUNT
        curriculum = False
        num_rows = 4
        num_cols = 4
        terrain_length = 15
        terrain_width = 15

    class asset(Go2RoughCfg.asset):
        self_collisions = 0

    class commands(Go2RoughCfg.commands):
        heading_command = False
        resampling_time = 4.
        class ranges(Go2RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]    # 无角速度指令

    class raycaster(Go2RoughCfg.raycaster):
        enable_raycast = True
        ray_pattern = "spherical"
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        max_distance = 50.0
        attach_yaw_only = False
        vertical_fov_deg_min = -2.0
        vertical_fov_deg_max = 57.0
        offset_pos = [0.28945, 0.0, -0.046825]
        sensor_offset_rpy = [0.0, -2.8782, 3.14]

    class rewards(Go2RoughCfg.rewards):
        base_height_target = 0.34
        class scales(Go2RoughCfg.rewards.scales):
            vel_avoid = 1.0
            rays = 0.5

            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -5.0
            torques = -0.000025
            dof_acc = -2.5e-7
            base_height = -5.0
            feet_air_time = 1.0
            collision = 0
            action_rate = -0.01
            gait_2_step = -1.0

    class normalization(Go2RoughCfg.normalization):
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            heading = 1.0

    class domain_rand(Go2RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        lidar_point_mask_ratio = 0.05
        lidar_point_mask_value_range = [2.0, 10.0]
        lidar_distance_noise_ratio = 0.02
        payload_mass_range = [-1.0, 3.0]
        com_shift_range = [[-0.1, -0.15, -0.2], [0.1, 0.15, 0.2]]
        restitution_range = [0.0, 1.0]
        motor_strength_range = [0.8, 1.2]
        joint_calib_offset_range = [-0.02, 0.02]
        gravity_offset_range = [-1.0, 1.0]
        proprio_delay_range = [0.005, 0.045]


class Go2SoftPillarPretrainCfgPPO(Go2RoughCfgPPO):
    class policy(Go2RoughCfgPPO.policy):
        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        perception_enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_history_length = PROX_HISTORY_LENGTH
        distal_history_length = DIST_HISTORY_LENGTH
        num_lidar_points = PD_NUM_LIDAR_POINTS
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = PD_THETA_DEG
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proprio_obs_dim = PD_PROPRIO_DIM
        privileged_height_dim = PD_PRIV_HEIGHT_DIM
        privileged_critic_dim = PD_PRIV_CRITIC_DIM
        privileged_supervision_coef = 1.0
        sensor_offset_rpy = [0.0, -2.8782, 3.14]
        sensor_offset_pos = [0.28945, 0.0, -0.046825]

    class algorithm(Go2RoughCfgPPO.algorithm):
        amp_enabled = True
        clip_param = 0.2
        lam = 0.95
        gamma = 0.99
        learning_rate = 1.0e-3
        schedule = "adaptive"
        entropy_coef = 0.01
        desired_kl = 0.01
        max_grad_norm = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(Go2RoughCfgPPO.runner):
        policy_class_name = "PDRiskNetActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        experiment_name = "go2_soft_pretrain"
        run_name = ""
        max_iterations = 1000
```

- [ ] **Step 2: 验证配置加载**

```bash
python3 -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_soft_pillar_pretrain import Go2SoftPillarPretrainCfg
cfg = Go2SoftPillarPretrainCfg()
print('soft_pretrain:', cfg.pd_risknet.soft_pretrain)
print('mesh_type:', cfg.terrain.mesh_type)
print('vel_avoid:', cfg.rewards.scales.vel_avoid)
print('rays:', cfg.rewards.scales.rays)
print('pillar_count:', cfg.pd_risknet.pillar_count)
print('num_observations:', cfg.env.num_observations)
"
```
Expected: `soft_pretrain: True`, `mesh_type: plane`, `vel_avoid: 1.0`, `rays: 0.5`, `pillar_count: 30`, `num_observations: 4548`

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_soft_pillar_pretrain.py
git commit -m "feat: add go2_soft_pretrain config"
```

---

### Task 4: 注册任务

**Files:**
- Modify: `legged_gym/legged_gym/envs/__init__.py`

- [ ] **Step 1: 注册 `go2_soft_pretrain`**

在 `__init__.py` 末尾，其他 go2 注册附近添加：

```python
from legged_gym.envs.go2.lidar_pd_risknet.go2_soft_pillar_pretrain import Go2SoftPillarPretrainCfg, Go2SoftPillarPretrainCfgPPO
from legged_gym.envs.go2.lidar_pd_risknet import Go2LidarPDRiskNet
task_registry.register("go2_soft_pretrain", Go2LidarPDRiskNet, Go2SoftPillarPretrainCfg(), Go2SoftPillarPretrainCfgPPO())
```

- [ ] **Step 2: 验证注册**

```bash
python3 -c "
from legged_gym.utils.task_registry import task_registry
import legged_gym.envs
print('go2_soft_pretrain' in task_registry.task_classes)
"
```
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/__init__.py
git commit -m "feat: register go2_soft_pretrain task"
```

---

### Task 5: 集成验证

- [ ] **Step 1: 语法全检查**

```bash
python3 -c "
import ast
for f in [
    'legged_gym/legged_gym/utils/pillar_mesh.py',
    'legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py',
    'legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_soft_pillar_pretrain.py',
    'legged_gym/legged_gym/envs/__init__.py',
]:
    ast.parse(open(f).read())
    print(f'{f}: Syntax OK')
"
```
Expected: 4 files `Syntax OK`

- [ ] **Step 2: 配置全加载**

```bash
python3 -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_soft_pillar_pretrain import Go2SoftPillarPretrainCfg, Go2SoftPillarPretrainCfgPPO
cfg = Go2SoftPillarPretrainCfg()
ppo = Go2SoftPillarPretrainCfgPPO()
print('Configs loaded OK')
print('PD_PROPRIO_DIM:', cfg.env.num_observations)
print('Critic dim:', cfg.env.num_privileged_obs)
"
```
Expected: `Configs loaded OK`, `PD_PROPRIO_DIM: 4548`, `Critic dim: 235`
