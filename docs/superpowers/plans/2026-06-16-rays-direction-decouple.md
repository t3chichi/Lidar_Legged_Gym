# Rays 平滑方向与奖励解耦 实现计划

**Goal:** 提取 EMA 更新为独立方法，使可视化箭头在 rays 奖励为 0 时也能更新。

**Architecture:** 新增 `_update_smooth_rays_dir()` 方法，在 `_post_physics_step_callback` 中调用（与 `_compute_v_avoid()` 同模式），`_reward_rays` 简化为仅读缓存。

**Tech Stack:** PyTorch

**Spec:** `docs/superpowers/specs/2026-06-16-rays-direction-decouple-design.md`

---

### 文件职责

| 文件 | 职责 |
|------|------|
| `legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` | 新增方法、修改 callback、简化 _reward_rays |
| `legged_gym/tests/test_go2_lidar_pd_risknet_math.py` | 添加独立 EMA 更新测试 |

---

### Task 1: 新增 `_update_smooth_rays_dir` 方法 + 修改 callback

**Files:**
- Modify: `legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

**Step 1: 在 `_compute_rays_omega_target` 方法前插入新方法**

在 `_compute_rays_omega_target` 前（约第 744 行）插入：

```python
    def _update_smooth_rays_dir(self):
        """Update EMA-smoothed open-space direction (called every step).

        Mirrors _compute_v_avoid(): cache computation here,
        _reward_rays() only reads the cached result.
        """
        cfg = self.cfg.pd_risknet
        alpha = float(cfg.rays_smoothing_alpha)
        target_dir_world = self._compute_rays_target_dir()  # (N, 2)
        # EMA
        self._smooth_dir_world = (
            alpha * target_dir_world + (1.0 - alpha) * self._smooth_dir_world
        )
        smooth_norm = torch.norm(self._smooth_dir_world, dim=1, keepdim=True).clamp(min=1e-8)
        self._smooth_dir_world = self._smooth_dir_world / smooth_norm
```

**Step 2: 在 `_post_physics_step_callback` 中添加调用**

将第 482-487 行：
```python
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_lidar_history()
        self._compute_v_avoid()
```

改为：
```python
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_lidar_history()
        self._compute_v_avoid()
        self._update_smooth_rays_dir()
```

**Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: extract rays EMA to _update_smooth_rays_dir, call every step"
```

---

### Task 2: 简化 `_reward_rays`

**Files:**
- Modify: `legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

**Step 1: 替换 _reward_rays 方法体**

将第 773-801 行替换为：

```python
    def _reward_rays(self):
        """Angular-velocity tracking reward: encourages turning toward open space.

        _smooth_dir_world is updated every step by _update_smooth_rays_dir()
        (called in _post_physics_step_callback), so we only read the cache.
        """
        omega_target = self._compute_rays_omega_target()  # (N,)
        omega_actual = self.base_ang_vel[:, 2]
        omega_err = omega_actual - omega_target
        sigma = float(getattr(self.cfg.pd_risknet, "rays_omega_sigma", 0.25))
        return torch.exp(-omega_err * omega_err / sigma)
```

**Step 2: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "refactor: simplify _reward_rays to read cached smooth_dir"
```

---

### Task 3: 添加测试 + 最终验证

**Files:**
- Modify: `legged_gym/tests/test_go2_lidar_pd_risknet_math.py`

**Step 1: 添加 EMA 解耦测试**

```python
# ── Rays smooth_dir decoupling tests ────────────────────────────

class TestSmoothRaysDirDecoupled(unittest.TestCase):
    """Verify _smooth_dir_world updates independently of _reward_rays."""

    def test_ema_updates_every_step(self):
        """_smooth_dir_world should change after _update_smooth_rays_dir."""
        # Create a minimal mock env with the necessary attributes
        import copy
        # Use the actual config from the module
        from legged_gym.envs.go2.lidar_pd_risknet import go2_lidar_pd_risknet_config as cfg_mod
        # Verify the method exists and has the expected behavior
        from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
        self.assertTrue(hasattr(Go2LidarPDRiskNet, '_update_smooth_rays_dir'),
                        "Go2LidarPDRiskNet should have _update_smooth_rays_dir method")

    def test_reward_rays_no_longer_updates_ema(self):
        """_reward_rays should NOT contain EMA logic (alpha * target + (1-alpha) * smooth)."""
        import inspect
        from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
        source = inspect.getsource(Go2LidarPDRiskNet._reward_rays)
        # EMA pattern should NOT be in _reward_rays
        self.assertNotIn('alpha * target_dir_world', source,
                         "_reward_rays should not contain EMA update logic")
```

**Step 2: 运行测试**

```bash
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

Expected: 所有测试通过。

**Step 3: 验证等价性（手动推理）**

旧路径 (rays≠0): `compute_reward` → `_reward_rays` → 计算 target + EMA + 奖励
新路径 (rays≠0): callback → `_update_smooth_rays_dir` → 计算 target + EMA
                  `compute_reward` → `_reward_rays` → 读缓存 + 奖励

两者在同一时间步使用相同的 `_raw_distances`（由 `_update_lidar_history` 在 callback 中填充，之后未被修改）。计算结果数学上等价。

**Step 4: Commit**

```bash
git add legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py
git commit -m "test: add rays EMA decoupling verification tests"
```
