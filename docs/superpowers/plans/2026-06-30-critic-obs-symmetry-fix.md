# Critic Observation Symmetry Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让对称函数正确处理 `obs_type="critic"` 的高度网格观测（Y 轴镜像），避免维度崩溃。

**Architecture:** 在对称函数顶部新增 `obs_type == "critic"` 分支，用 reshape→flip→flatten 对高度网格做 Y 轴镜像后提前返回。网格维度参数从 `env.cfg.terrain` 读取，通过 `partial` 绑定。

**Tech Stack:** PyTorch (`torch.flip`)

**Spec:** `docs/superpowers/specs/2026-06-30-critic-obs-symmetry-fix.md`

---

### Task 1: 新增 critic 观测测试

**Files:**
- Modify: `legged_gym/legged_gym/tests/test_go2_xsym.py`

- [ ] **Step 1: 在 TestInvariants 类中添加测试**

在文件末尾添加：

```python
class TestCriticObs:
    """Verify critic observation (height grid) symmetry."""

    import torch

    def test_critic_grid_shape(self):
        """Obs type critic should double the batch and preserve obs dim."""
        func = _make_func()
        obs = torch.randn(4, 187)  # 17x11 height grid
        obs_aug, act_aug = func(obs=obs, actions=None, obs_type="critic")
        assert obs_aug.shape == (8, 187)
        # Critic observations don't have actions, but PPO passes None
        assert act_aug is None

    def test_critic_grid_y_flip(self):
        """Each row of the height grid should be Y-reversed."""
        func = _make_func()
        # Build a test grid with known values: each cell = row*100 + col
        obs = torch.zeros(2, 187)
        y_count = 11
        for x in range(17):
            for y in range(y_count):
                obs[:, x * y_count + y] = float(x * 100 + y)
        obs_aug, _ = func(obs=obs, actions=None, obs_type="critic")
        mirrored = obs_aug[2:]
        # Check: mirrored row x, column y == original row x, column (y_count-1-y)
        for x in range(17):
            for y in range(y_count):
                orig_val = obs[0, x * y_count + y].item()
                mirr_val = mirrored[0, x * y_count + (y_count - 1 - y)].item()
                assert mirr_val == orig_val, (
                    f"x={x} y={y}: expected {orig_val}, got {mirr_val}"
                )

    def test_critic_grid_double_mirror_identity(self):
        """Double mirror of height grid must be exact identity."""
        func = _make_func()
        obs = torch.randn(4, 187)
        obs_aug, _ = func(obs=obs, actions=None, obs_type="critic")
        obs_m1 = obs_aug[4:]
        obs_aug2, _ = func(obs=obs_m1, actions=None, obs_type="critic")
        obs_m2 = obs_aug2[4:]
        diff = (obs_m2 - obs).abs()
        assert diff.max().item() == 0.0, f"Double mirror not identity, max diff={diff.max().item():.2e}"
```

- [ ] **Step 2: 运行测试，预期失败**

```bash
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_xsym.py::TestCriticObs -v
# 预期：test_critic_grid_shape / test_critic_grid_y_flip / test_critic_grid_double_mirror_identity 全部 FAIL
# (因为 obs_type="critic" 尚未实现)
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/tests/test_go2_xsym.py
git commit -m "test: add failing critic obs symmetry tests"
```

---

### Task 2: 在对称函数中添加 critic 分支

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py`

- [ ] **Step 1: 添加函数签名参数**

在第 1115 行 `) -> tuple:` 前，新增两个关键字参数：

```python
    height_grid_x_count: int = 17,
    height_grid_y_count: int = 11,
```

- [ ] **Step 2: 在 `obs is not None` 块顶部添加 critic 分支**

在第 1130-1131 行之间 (`if obs is not None:` 和 `obs_mirrored = obs.clone()` 之间) 插入：

```python
        # ── Critic observation: height grid, mirror Y axis only ──
        if obs_type == "critic":
            # Height grid layout: [B, x_count * y_count] with x as outer loop.
            # Mirror flips the y axis: for each x row, reverse y within it.
            obs_mirrored = torch.flip(
                obs.reshape(-1, height_grid_x_count, height_grid_y_count),
                dims=[2],
            ).reshape_as(obs)
            obs_augmented = torch.cat([obs, obs_mirrored], dim=0)
            if actions is not None:
                # Critic branch shouldn't normally reach here (PPO passes
                # None), but handle it consistently just in case.
                actions_augmented = torch.cat([actions, actions.clone()], dim=0)
            else:
                actions_augmented = None
            return obs_augmented, actions_augmented

        # ── Policy observation: full proprio + LiDAR mirror (existing) ──
```

- [ ] **Step 3: 运行测试，预期通过**

```bash
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_xsym.py -v
# 预期：全部通过（原有 15 + 新增 3 = 18 passed）
```

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py
git commit -m "fix: add critic observation symmetry branch for height grid

Handle obs_type=critic by reshaping the 187-dim height grid (17x11)
to [B,17,11], flipping the y axis, and flattening back. This prevents
the wrapped policy obs path from being applied to the critic's grid
observation, which was causing a dimension mismatch crash."
```

---

### Task 3: 在 _setup_symmetry 中绑定网格参数

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/on_policy_runner.py`

- [ ] **Step 1: 在 partial 绑定中添加网格维度参数**

在第 319 行 `)` 结束 partial 前，新增两行：

```python
                height_grid_x_count=int(getattr(self.env.cfg.terrain, 'measured_grid_x_count', 17)),
                height_grid_y_count=int(getattr(self.env.cfg.terrain, 'measured_grid_y_count', 11)),
```

完整修改后的 `func = partial(func, ...)` 块（第 312-320 行）：

```python
            func = partial(func,
                sensor_quat=self.env._sensor_offset_quat[0:1],
                sensor_trans=self.env._sensor_translation[0:1],
                proprio_dim=proprio_dim,
                proximal_points=proximal_points,
                distal_history_points=distal_history_points,
                distal_history_length=distal_history_length,
                height_grid_x_count=int(getattr(self.env.cfg.terrain, 'measured_grid_x_count', 17)),
                height_grid_y_count=int(getattr(self.env.cfg.terrain, 'measured_grid_y_count', 11)),
            )
```

- [ ] **Step 2: 验证集成链路**

```bash
conda run -n li_leggym python -c "
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe_config import Go2CmdSafeCfgPPO
from rsl_rl.utils import string_to_callable
from functools import partial
import torch

cfg = Go2CmdSafeCfgPPO()
p_cfg = cfg.policy
func = string_to_callable(cfg.algorithm.symmetry_cfg.data_augmentation_func)
func = partial(func,
    sensor_quat=torch.zeros(1,4), sensor_trans=torch.zeros(1,3),
    proprio_dim=int(p_cfg.proprio_obs_dim),
    proximal_points=int(p_cfg.proximal_points),
    distal_history_points=int(p_cfg.distal_points)*int(p_cfg.distal_history_length),
    distal_history_length=int(p_cfg.distal_history_length),
    height_grid_x_count=17, height_grid_y_count=11)

# Test policy obs (should work)
obs_pol = torch.randn(4, 2736)
obs_aug_p, _ = func(obs=obs_pol, actions=None, obs_type='policy')
assert obs_aug_p.shape == (8, 2736)
print('policy obs: OK')

# Test critic obs (should work now)
obs_crit = torch.randn(4, 187)
obs_aug_c, _ = func(obs=obs_crit, actions=None, obs_type='critic')
assert obs_aug_c.shape == (8, 187)
assert not torch.isnan(obs_aug_c).any()
print('critic obs: OK')

# Double mirror identity for critic
obs_aug_c2, _ = func(obs=obs_aug_c[4:], actions=None, obs_type='critic')
diff = (obs_aug_c2[4:] - obs_crit).abs()
assert diff.max().item() == 0.0, f'Not identity: {diff.max().item():.2e}'
print('critic double mirror: EXACT IDENTITY')
print('ALL PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add rsl_rl/rsl_rl/runners/on_policy_runner.py
git commit -m "feat: bind height grid dimensions for critic obs symmetry"
```

---

### Task 4: 全量回归测试

- [ ] **Step 1: 运行全部测试**

```bash
# Batch 1: 不依赖 isaacgym
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_pointcloud_geometry.py legged_gym/legged_gym/tests/test_cmd_safe_history_wrapper.py -v

# Batch 2: 依赖 isaacgym
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_xsym.py -v
```

- [ ] **Step 2: 确认 18/18 通过（15 原有 + 3 新增）**

```bash
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_xsym.py -v 2>&1 | grep -c PASSED
# Expected: 18
```
