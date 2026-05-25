# r_rays 距离最大化奖励对齐论文 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `_reward_rays` 实现对齐 Omni-Perception 论文公式，使用全部远端射线（theta < split_theta_deg）截断距离的均值。

**Architecture:** 在 `_init_pd_risknet_buffers` 中预计算远端掩码（基于球形网格的固定仰角线），`_reward_rays` 中用掩码索引 `raycast_distances` 取远端距离，clip 到 d_max 后取均值归一化。配置中 d_max 从 6m 改为 10m 对齐物理探测距离。

**Tech Stack:** PyTorch, NumPy

---

### Task 1: 更新配置 — d_max 对齐物理探测距离，添加 FOV 字段

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py` (pd_risknet 和 raycaster 段落)

- [ ] **Step 1: 修改 ray_max_distance 并添加 vertical FOV 字段**

在 `go2_lidar_pillar_config.py` 中修改两处：

```python
# pd_risknet 段，约第 48 行
ray_max_distance = 6.0  # rays 奖励截断距离 (m)
```
改为：
```python
ray_max_distance = 10.0  # rays 奖励截断距离 (m)，对齐 raycaster.max_distance
```

```python
# raycaster 段，约第 132 行，在 max_distance = 10.0 之后添加
max_distance = 10.0
attach_yaw_only = False
```
改为：
```python
max_distance = 10.0
attach_yaw_only = False
vertical_fov_deg_min = -2.0   # sensor frame 垂直 FOV 下限 (deg)
vertical_fov_deg_max = 57.0   # sensor frame 垂直 FOV 上限 (deg)
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py
git commit -m "config: align ray_max_distance to 10m, add vertical FOV fields for distal mask"
```

---

### Task 2: 预计算远端掩码

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` (`_init_pd_risknet_buffers`)

- [ ] **Step 1: 在 `_init_pd_risknet_buffers` 末尾添加掩码预计算**

文件 `go2_lidar_pd_risknet.py`，在 `_init_pd_risknet_buffers` 方法末尾 (`self._consecutive_downgrade_count` 初始化后) 添加：

```python
        # Precompute distal ray mask from spherical grid geometry.
        # The LiDAR sensor uses a SIMPLE_GRID pattern: elevation × azimuth, flattened
        # row-major. In the sensor frame, theta = elevation angle. Rays with
        # theta < split_theta_deg are distal (far-field), the rest are proximal.
        num_azimuth = int(cfg.spherical_num_azimuth)
        num_elevation = int(cfg.spherical_num_elevation)
        v_min_rad = math.radians(float(self.cfg.raycaster.vertical_fov_deg_min))
        v_max_rad = math.radians(float(self.cfg.raycaster.vertical_fov_deg_max))
        split_rad = math.radians(float(cfg.split_theta_deg))

        # Elevation descends from v_max to v_min (matching _initialize_grid_rays ordering).
        elev_rad = torch.linspace(v_max_rad, v_min_rad, num_elevation, device=self.device)
        distal_lines = elev_rad < split_rad  # (num_elevation,)
        distal_mask_2d = distal_lines.unsqueeze(1).expand(num_elevation, num_azimuth)
        self._distal_mask = distal_mask_2d.reshape(-1)  # (num_lidar_points,)
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: precompute distal ray mask from spherical grid geometry"
```

---

### Task 3: 重写 `_reward_rays`

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` (line 511-517)

- [ ] **Step 1: 替换 `_reward_rays` 方法**

将当前实现（line 511-517）：
```python
    def _reward_rays(self):
        d_max = float(self.cfg.pd_risknet.ray_max_distance)
        dist = self.avoid_distances
        k = max(int(dist.shape[1] * 0.5), 1)
        closest = torch.topk(dist, k, dim=1, largest=False)[0]
        clipped = torch.clamp(closest, max=d_max)
        return torch.mean(clipped / d_max, dim=1)
```

替换为：
```python
    def _reward_rays(self):
        d_max = float(self.cfg.pd_risknet.ray_max_distance)
        dist = self.raycast_distances[:, self._distal_mask]
        clipped = torch.clamp(dist, max=d_max)
        return torch.mean(clipped / d_max, dim=1)
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: align _reward_rays with paper — mean of all distal ray truncated distances"
```

---

### Task 4: 更新数学测试

**Files:**
- Modify: `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py`

- [ ] **Step 1: 添加远端掩码计算函数和测试**

在测试文件中追加以下内容到文件末尾：

```python
def build_distal_mask(num_azimuth, num_elevation, v_fov_min_deg, v_fov_max_deg,
                       split_theta_deg, device="cpu"):
    """Replicate the distal mask logic used in _init_pd_risknet_buffers.

    Returns a bool tensor of shape (num_elevation * num_azimuth,).
    """
    import torch
    import math

    v_min_rad = math.radians(v_fov_min_deg)
    v_max_rad = math.radians(v_fov_max_deg)
    split_rad = math.radians(split_theta_deg)

    elev_rad = torch.linspace(v_max_rad, v_min_rad, num_elevation, device=device)
    distal_lines = elev_rad < split_rad  # (num_elevation,)
    distal_mask_2d = distal_lines.unsqueeze(1).expand(num_elevation, num_azimuth)
    return distal_mask_2d.reshape(-1)


def test_distal_mask_shape_and_count():
    import torch

    num_azimuth = 24
    num_elevation = 18
    v_fov_min_deg = -2.0
    v_fov_max_deg = 57.0
    split_theta_deg = 20.0

    mask = build_distal_mask(num_azimuth, num_elevation,
                              v_fov_min_deg, v_fov_max_deg, split_theta_deg)

    # Shape: full spherical grid
    assert mask.shape == (num_azimuth * num_elevation,), f"expected ({num_azimuth * num_elevation},), got {mask.shape}"
    # Must be bool
    assert mask.dtype == torch.bool

    distal_count = mask.sum().item()
    # With 18 lines from 57° down to -2°, lines < 20°: 11 through 17 = 7 lines × 24 = 168
    assert distal_count == 168, f"expected 168 distal points, got {distal_count}"

    # Verify specific lines: line 0 (57°) is NOT distal, line 17 (-2°) IS distal
    assert not mask[0].item()          # line 0, azimuth 0: elevation 57° → proximal
    assert mask[-1].item()             # line 17, azimuth 23: elevation -2° → distal


def test_distal_rays_reward_matches_paper():
    """Paper formula: mean(clip(d_i, d_max) / d_max) over n distal rays."""
    import torch

    num_azimuth = 24
    num_elevation = 18
    mask = build_distal_mask(num_azimuth, num_elevation,
                              -2.0, 57.0, 20.0)

    # Simulate 2 environments with random distances
    torch.manual_seed(42)
    all_distances = torch.rand(2, num_elevation * num_azimuth) * 15.0  # up to 15m
    distal_dist = all_distances[:, mask]  # only distal rays

    d_max = 10.0
    clipped = torch.clamp(distal_dist, max=d_max)
    reward = torch.mean(clipped / d_max, dim=1)

    # All rewards should be in (0, 1]
    assert torch.all(reward > 0.0)
    assert torch.all(reward <= 1.0)

    # Verify formula manually for env 0
    expected = torch.mean(torch.clamp(distal_dist[0], max=d_max) / d_max)
    assert torch.isclose(reward[0], expected, atol=1e-6)


def test_rays_formula_matches_paper():
    import torch

    distances = torch.tensor([[1.0, 2.0, 12.0]], dtype=torch.float32)
    d_max = 10.0
    rew = rays_reward(distances, d_max)
    expected = (1.0 / 10.0 + 2.0 / 10.0 + 10.0 / 10.0) / 3.0
    assert torch.isclose(rew[0], torch.tensor(expected), atol=1e-6)
```

注：`test_rays_formula_matches_paper` 保持不变——它验证论文公式的数学正确性，与掩码无关。

- [ ] **Step 2: 运行新增测试验证通过**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_distal_mask_shape_and_count legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_distal_rays_reward_matches_paper legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_rays_formula_matches_paper -v
```

Expected: 3 tests PASS

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py
git commit -m "test: add distal mask and paper-aligned rays reward tests"
```

---

### Task 5: 运行完整测试套件验证无回归

**Files:** 无修改

- [ ] **Step 1: 运行所有现有测试**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

Expected: 所有测试 PASS（包括原有和新增）

- [ ] **Step 2: 运行基础环境测试**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -m pytest legged_gym/legged_gym/tests/test_env.py -v
```

Expected: PASS
