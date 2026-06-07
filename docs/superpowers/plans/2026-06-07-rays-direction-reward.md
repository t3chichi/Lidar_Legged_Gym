# Rays 方向一致性奖励实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 rays 奖励从"静态视线质量评分"替换为"朝向开阔方向的运动一致性奖励"

**Architecture:** 扇区距离提取→平方加权方向（机体帧）→世界帧 EMA 平滑→方向一致性点积奖励。新增 `_compute_rays_target_dir()` 辅助方法分离方向计算逻辑。

**Tech Stack:** PyTorch, Isaac Gym `quat_apply_yaw` / `quat_apply_yaw_inverse`

---

### Task 1: 添加配置项

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py:51-55`

- [ ] **Step 1: 在 `pd_risknet` 类中添加新配置项**

在 `ray_max_distance = 10.0` 之后添加：

```python
        # Rays direction-consistency reward (replaces top-k distance scoring).
        rays_top_ratio = 0.2           # 每扇区取前 20% 最远点进行距离平均
        rays_smoothing_alpha = 0.4     # 世界帧方向 EMA 平滑因子
        rays_epsilon = 0.01            # 速度分母软化项
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "feat: add rays direction-consistency reward config entries"
```

---

### Task 2: 预计算扇区缓冲区

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:138-162` (`_init_lidar_aux`)
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:74-116` (`_init_pd_risknet_buffers`)

- [ ] **Step 1: 在 `_init_pd_risknet_buffers` 末尾添加新 buffer**

在 `self._channel_forward = _FORWARD_LOOKUP_TABLE[_safe_idx]` 之后：

```python
        # Rays direction reward: smoothed world-frame direction (N, 2)
        self._smooth_dir_world = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float, requires_grad=False)

        # Precomputed 36 sector center unit directions (body frame, 2D)
        sector_centers = torch.linspace(
            -math.pi + math.pi / 36, math.pi - math.pi / 36, 36, device=self.device)
        self._sector_dirs = torch.stack(
            (torch.cos(sector_centers), torch.sin(sector_centers)), dim=1)  # (36, 2)
```

- [ ] **Step 2: 在 `_init_lidar_aux` 末尾预计算扇区射线索引**

在 `self._distal_ray_sector_ids` 赋值之后：

```python
        # Precompute per-sector distal ray indices for fast gather in _reward_rays.
        self._sector_ray_indices = []
        for s in range(36):
            idx = torch.where(self._distal_ray_sector_ids == s)[0]
            self._sector_ray_indices.append(idx)
```

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: precompute sector buffers and ray indices for rays direction reward"
```

---

### Task 3: 实现 `_compute_rays_target_dir` 辅助方法

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` (insert before `_reward_rays`)

- [ ] **Step 1: 添加 `_compute_rays_target_dir` 方法**

在 `_reward_rays` 方法之前插入（约581行前）：

```python
    def _compute_rays_target_dir(self):
        """Compute world-frame weighted-average direction pointing toward open space.

        Steps 1-3 of the design:
        1. Per-sector top-20% farthest valid distal point average distance d_i
        2. Square weights: w_i = d_i²
        3. Weighted average of sector center directions → target_dir_body
        Returns target_dir_world (N, 2) in world frame.
        """
        cfg = self.cfg.pd_risknet
        d_max = float(cfg.ray_max_distance)
        top_ratio = float(cfg.rays_top_ratio)

        dist_all = self.raycast_distances[:, self._distal_mask]  # (N, num_distal)
        valid = dist_all < (d_max - 0.001)

        weighted_sum = torch.zeros(self.num_envs, 2, device=self.device)
        weight_total = torch.zeros(self.num_envs, device=self.device)

        for s in range(36):
            indices = self._sector_ray_indices[s]
            if len(indices) == 0:
                continue

            s_dist = dist_all[:, indices]                     # (N, rays_in_sector)
            s_valid = valid[:, indices]

            n_valid = s_valid.sum(dim=1)                       # (N,)
            # Mask invalid distances to zero so they sort last.
            s_dist = torch.where(s_valid, s_dist, torch.zeros_like(s_dist))
            k = torch.clamp((n_valid.float() * top_ratio).long(), min=1)  # (N,)
            k_max = int(k.max().item())

            # Take top-k farthest distances.
            top_vals, _ = torch.topk(s_dist, k=k_max, dim=1)   # (N, k_max)
            idx_mask = torch.arange(k_max, device=self.device).unsqueeze(0) < k.unsqueeze(1)
            d_i = (top_vals * idx_mask.float()).sum(dim=1) / k.float()  # (N,)

            w_i = d_i.square()

            # Exclude envs where this sector has zero valid rays.
            w_i = torch.where(n_valid > 0, w_i, torch.zeros_like(w_i))

            sec_dir = self._sector_dirs[s]                     # (2,)
            weighted_sum = weighted_sum + w_i.unsqueeze(1) * sec_dir.unsqueeze(0)
            weight_total = weight_total + w_i

        # Normalize to unit direction (body frame).
        target_norm = torch.norm(weighted_sum, dim=1, keepdim=True).clamp(min=1e-8)
        target_dir_body = weighted_sum / target_norm

        # Transform to world frame (yaw only).
        target_dir_body_3d = torch.cat(
            [target_dir_body, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        target_dir_world = quat_apply_yaw(self.base_quat, target_dir_body_3d)[:, :2]

        return target_dir_world
```

这完成了设计中的步骤 1-3。

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add _compute_rays_target_dir helper for sector-weighted direction"
```

---

### Task 4: 重写 `_reward_rays`

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:581-612`

- [ ] **Step 1: 替换 `_reward_rays` 方法**

将旧的 `_reward_rays`（581-612行）替换为：

```python
    def _reward_rays(self):
        """Direction-consistency reward: dot product of body velocity and smoothed open-space direction.

        Steps 4-5 of the design:
        4. World-frame EMA smoothing of target_dir
        5. r = (v_body · smooth_dir_body) / max(|v_body|, eps)
        Range [-1, 1]: +1 when moving exactly toward open space, -1 when moving away.
        """
        cfg = self.cfg.pd_risknet
        alpha = float(cfg.rays_smoothing_alpha)
        eps = float(cfg.rays_epsilon)

        # Step 1-3: raw target direction (world frame).
        target_dir_world = self._compute_rays_target_dir()  # (N, 2)

        # Step 4: EMA smooth in world frame.
        self._smooth_dir_world = (
            alpha * target_dir_world + (1.0 - alpha) * self._smooth_dir_world
        )
        smooth_norm = torch.norm(self._smooth_dir_world, dim=1, keepdim=True).clamp(min=1e-8)
        self._smooth_dir_world = self._smooth_dir_world / smooth_norm

        # Step 5: direction consistency reward in body frame.
        smooth_dir_world_3d = torch.cat(
            [self._smooth_dir_world, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        smooth_dir_body = quat_apply_yaw_inverse(self.base_quat, smooth_dir_world_3d)[:, :2]

        v_body = self.base_lin_vel[:, :2]
        v_norm = torch.norm(v_body, dim=1)
        dot = (v_body * smooth_dir_body).sum(dim=1)

        return dot / torch.clamp(v_norm, min=eps)
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: replace static rays distance reward with direction-consistency reward"
```

---

### Task 5: 更新 `reset_idx` 初始化平滑方向

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:561-573` (`reset_idx`)

- [ ] **Step 1: 在 `reset_idx` 中初始化 `_smooth_dir_world`**

在 `reset_idx` 方法的 `self._update_lidar_history()` 调用之后，添加平滑方向初始化：

现有的 `reset_idx`:
```python
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.lidar_points_base[env_ids] = 0.0
        self.raycast_distances[env_ids] = float(self.cfg.pd_risknet.ray_max_distance)
        self.v_avoid[env_ids] = 0.0
        self.last_dist[env_ids] = torch.norm(
            self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        if hasattr(self, 'last_last_actions'):
            self.last_last_actions[env_ids] = 0.
        self._update_lidar_history()
```

在文件末尾的 `self._update_lidar_history()` 后添加：

```python
        # Initialize rays smoothed direction to first target direction.
        # Must happen after _update_lidar_history() so LiDAR data is fresh.
        target_dir_world = self._compute_rays_target_dir()  # (N, 2)
        self._smooth_dir_world[env_ids] = target_dir_world[env_ids]
```

完整修改后的 `reset_idx`:

```python
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.lidar_points_base[env_ids] = 0.0
        self.raycast_distances[env_ids] = float(self.cfg.pd_risknet.ray_max_distance)
        self.v_avoid[env_ids] = 0.0
        self.last_dist[env_ids] = torch.norm(
            self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        if hasattr(self, 'last_last_actions'):
            self.last_last_actions[env_ids] = 0.
        self._update_lidar_history()
        # Initialize rays smoothed direction to first target direction.
        # Must happen after _update_lidar_history() so LiDAR data is fresh.
        target_dir_world = self._compute_rays_target_dir()  # (N, 2)
        self._smooth_dir_world[env_ids] = target_dir_world[env_ids]
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: initialize smooth_dir_world on env reset for rays reward"
```

---

### Task 6: 更新测试文件

**Files:**
- Modify: `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py`

- [ ] **Step 1: 替换旧的 `rays_reward` 测试函数，添加新的方向一致性奖励测试**

将 `test_rays_formula_matches_paper` 和 `test_distal_rays_reward_matches_paper` 替换为新的单元测试：

```python
def rays_direction_reward(v_body, smooth_dir_body, eps=0.01):
    """Direction-consistency reward: r = dot(v_body, smooth_dir) / max(|v_body|, eps)."""
    import torch
    v_norm = torch.norm(v_body, dim=-1)
    dot = (v_body * smooth_dir_body).sum(dim=-1)
    return dot / torch.clamp(v_norm, min=eps)


def test_rays_direction_perfect_alignment():
    """Moving exactly toward open space → reward near +1."""
    import torch
    v = torch.tensor([[1.0, 0.0], [0.5, 0.0], [2.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    assert torch.allclose(r, torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)


def test_rays_direction_opposite():
    """Moving away from open space → reward near -1."""
    import torch
    v = torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
    d = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    assert torch.allclose(r, torch.tensor([-1.0, -1.0]), atol=1e-6)


def test_rays_direction_orthogonal():
    """Moving perpendicular to open space → reward near 0."""
    import torch
    v = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6)


def test_rays_direction_speed_invariant():
    """Same direction, different speeds → same reward (speed-decoupled)."""
    import torch
    v = torch.tensor([[0.1, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    assert torch.allclose(r, torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)


def test_rays_direction_zero_velocity():
    """Zero velocity → reward near 0 (eps prevents division by zero)."""
    import torch
    v = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d, eps=0.01)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6)


def test_rays_direction_partial_alignment():
    """45° between velocity and direction → reward = cos(45°) ≈ 0.707."""
    import torch
    import math
    v = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    angle = math.radians(45)
    d = torch.tensor([[math.cos(angle), math.sin(angle)]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    expected = math.cos(angle)
    assert torch.allclose(r, torch.tensor([expected]), atol=1e-6)


def test_rays_target_dir_square_weights():
    """Verify square weighting: far sectors dominate direction.

    Two sectors: front (0°) at 8m, right (90°) at 2m.
    w_front = 64, w_right = 4. Weighted direction leans strongly forward.
    """
    import torch
    import math

    d_front = torch.tensor([8.0])
    d_right = torch.tensor([2.0])
    w_front = d_front.square()  # 64
    w_right = d_right.square()  # 4

    # Sector dirs: front = (1, 0), right = (0, 1)
    dir_front = torch.tensor([1.0, 0.0])
    dir_right = torch.tensor([0.0, 1.0])

    weighted_sum = w_front * dir_front + w_right * dir_right  # (64, 4)
    target_dir = weighted_sum / torch.norm(weighted_sum)

    # Expected angle: atan2(4, 64) ≈ 3.58°, almost entirely forward.
    expected_angle = math.atan2(4, 64)  # ≈ 0.0624 rad
    actual_angle = math.atan2(target_dir[1].item(), target_dir[0].item())
    assert abs(actual_angle - expected_angle) < 1e-4, \
        f"expected {math.degrees(expected_angle):.2f}°, got {math.degrees(actual_angle):.2f}°"

    # x component > 0.99
    assert target_dir[0].item() > 0.99


def test_rays_target_dir_bend_scenario():
    """In a bend: front at 3m, diagonal at 6m → direction shifts toward diagonal.

    Simulates a left turn: front (0°) wall at 3m, left-forward (30°) open at 6m.
    Square weights: w_front=9, w_diag=36 → 4:1 advantage for diagonal.
    """
    import torch
    import math

    d_front = torch.tensor([3.0])
    d_diag = torch.tensor([6.0])
    w_front = d_front.square()  # 9
    w_diag = d_diag.square()    # 36

    angle = math.radians(30)
    dir_front = torch.tensor([1.0, 0.0])
    dir_diag = torch.tensor([math.cos(angle), math.sin(angle)])

    weighted_sum = w_front * dir_front + w_diag * dir_diag
    target_dir = weighted_sum / torch.norm(weighted_sum)

    # The diagonal direction should pull the result leftward (> 10°).
    actual_angle = math.atan2(target_dir[1].item(), target_dir[0].item())
    assert actual_angle > math.radians(10), \
        f"bend should shift direction >10°, got {math.degrees(actual_angle):.2f}°"


def test_rays_ema_smoothing():
    """EMA smoothing: smooth = α * target + (1-α) * prev."""
    import torch

    alpha = 0.4
    prev = torch.tensor([1.0, 0.0])  # previous smooth direction
    target = torch.tensor([0.0, 1.0])  # new target (90° turn)

    raw = alpha * target + (1 - alpha) * prev
    smooth = raw / torch.norm(raw)

    # After one step with α=0.4, angle should be ~atan2(0.4, 0.6) ≈ 33.7°
    import math
    expected_angle = math.atan2(0.4, 0.6)
    actual_angle = math.atan2(smooth[1].item(), smooth[0].item())
    assert abs(actual_angle - expected_angle) < 1e-4, \
        f"expected {math.degrees(expected_angle):.2f}°, got {math.degrees(actual_angle):.2f}°"
```

保留 `rays_reward` 和 `test_rays_formula_matches_paper` 作为旧公式参考（标记 skip），或直接删除。建议**删除**旧测试函数 `rays_reward`、`test_rays_formula_matches_paper`、`test_distal_rays_reward_matches_paper`，因为奖励公式已完全替换。

- [ ] **Step 2: 运行测试验证**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v -k "rays"
```

预期：8 个新测试全部 PASS。

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py
git commit -m "test: replace rays reward tests with direction-consistency reward tests"
```

---

### Task 7: 最终验证

- [ ] **Step 1: 运行完整测试套件**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

预期：所有测试 PASS（含 PD-RiskNet 形状门、config 门等旧测试）。

- [ ] **Step 2: 检查导入和配置兼容性**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg
cfg = Go2LidarPDRiskNetCfg()
assert cfg.pd_risknet.rays_top_ratio == 0.2
assert cfg.pd_risknet.rays_smoothing_alpha == 0.4
assert cfg.pd_risknet.rays_epsilon == 0.01
print('Config OK')
"
```

- [ ] **Step 3: 最终提交**

```bash
# 无新文件，验证步骤不需要额外提交
```
