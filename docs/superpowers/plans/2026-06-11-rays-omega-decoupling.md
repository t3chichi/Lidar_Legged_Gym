# rays→ω_tracking 解耦实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `_reward_rays` 从 cos(θ) 方向对齐奖励改为角速度跟踪奖励，与 `_reward_vel_avoid` 实现旋转/平移完全解耦。

**Architecture:** 新增 `_compute_rays_omega_target()` 方法将已 EMA 平滑的 open_dir 转换为目标角速度；重写 `_reward_rays()` 使用 exp(-ω_err²) 形式；五处配置文件新增 `rays_omega_gain`/`rays_omega_max` 参数并调整 `tracking_ang_vel`。

**Tech Stack:** Python, PyTorch, Isaac Gym

---

## 文件结构

| 文件 | 职责 | 操作 |
|------|------|:---:|
| `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py` | 新增 ω_target 计算方法 + 重写 rays 奖励 | 修改 |
| `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py` | 新增 rays_omega 参数 + tracking_ang_vel → 0 | 修改 |
| `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py` | 新增 rays_omega 参数 + tracking_ang_vel → 0 | 修改 |
| `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py` | 新增 rays_omega 参数 | 修改 |

---

### Task 1: 走廊配置新增 rays_omega 参数 + tracking_ang_vel 归零

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py`

- [ ] **Step 1: 在 pd_risknet 类中新增参数**

在 `class pd_risknet:` 中，`avoid_speed_limit = 1.0` 行之后，`ray_max_distance = 10.0` 行之前，插入：

```python
            # rays → ω_target 参数
            rays_omega_gain = 0.5     # k_ω: heading_error → ω_target P 增益
            rays_omega_max  = 0.5     # rad/s: 角速度指令上限
```

- [ ] **Step 2: tracking_ang_vel 归零**

将 `tracking_ang_vel = 0.1` 改为 `tracking_ang_vel = 0.0`。

- [ ] **Step 3: 验证配置一致性**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg, Go2LidarPDRiskNetCfgPPO
cfg = Go2LidarPDRiskNetCfg()
print('rays_omega_gain:', cfg.pd_risknet.rays_omega_gain)
print('rays_omega_max:', cfg.pd_risknet.rays_omega_max)
print('tracking_ang_vel:', cfg.rewards.scales.tracking_ang_vel)
print('PD_PROPRIO_DIM:', cfg.env.num_observations)
"
```
Expected: `rays_omega_gain: 0.5`, `rays_omega_max: 0.5`, `tracking_ang_vel: 0.0`, `PD_PROPRIO_DIM: 4548`

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "feat: add rays_omega params + disable tracking_ang_vel for corridor config"
```

---

### Task 2: 梅花桩配置新增 rays_omega 参数 + tracking_ang_vel 归零

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py`

- [ ] **Step 1: 在 pd_risknet 类中新增参数**

在 `class pd_risknet:` 中，`avoid_speed_limit = 1.0` 行之后，`ray_max_distance = 10.0` 行之前，插入：

```python
            # rays → ω_target 参数
            rays_omega_gain = 0.5     # k_ω: heading_error → ω_target P 增益
            rays_omega_max  = 0.5     # rad/s: 角速度指令上限
```

- [ ] **Step 2: tracking_ang_vel 归零**

将 `tracking_ang_vel = 2.0e-1` 改为 `tracking_ang_vel = 0.0`。

- [ ] **Step 3: 验证配置一致性**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pillar_config import Go2LidarPillarCfg, Go2LidarPillarCfgPPO
cfg = Go2LidarPillarCfg()
print('rays_omega_gain:', cfg.pd_risknet.rays_omega_gain)
print('rays_omega_max:', cfg.pd_risknet.rays_omega_max)
print('tracking_ang_vel:', cfg.rewards.scales.tracking_ang_vel)
"
```
Expected: `rays_omega_gain: 0.5`, `rays_omega_max: 0.5`, `tracking_ang_vel: 0.0`

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py
git commit -m "feat: add rays_omega params + disable tracking_ang_vel for pillar config"
```

---

### Task 3: 预训练配置新增 rays_omega 参数（保持 rays=0）

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py`

- [ ] **Step 1: 在 pd_risknet 类中新增参数**

在 `class pd_risknet:` 中，`avoid_speed_limit = 1.0` 行之后，`ray_max_distance = 10.0` 行之前，插入：

```python
            # rays → ω_target 参数
            rays_omega_gain = 0.5     # k_ω: heading_error → ω_target P 增益
            rays_omega_max  = 0.5     # rad/s: 角速度指令上限
```

- [ ] **Step 2: 确认 tracking_ang_vel 保持正常值**

验证 tracking_ang_vel 仍为 0.5（预训练时 rays 权重为 0，不受影响）。

- [ ] **Step 3: 验证配置一致性**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_pd_pretrain_config import Go2LidarPDRiskNetCfg, Go2LidarPDRiskNetCfgPPO
cfg = Go2LidarPDRiskNetCfg()
print('rays_omega_gain:', cfg.pd_risknet.rays_omega_gain)
print('rays_omega_max:', cfg.pd_risknet.rays_omega_max)
print('rays_scale:', cfg.rewards.scales.rays)
print('tracking_ang_vel:', cfg.rewards.scales.tracking_ang_vel)
"
```
Expected: `rays_omega_gain: 0.5`, `rays_omega_max: 0.5`, `rays_scale: 0`, `tracking_ang_vel: 0.5`

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py
git commit -m "feat: add rays_omega params to pretrain config"
```

---

### Task 4: 实现 `_compute_rays_omega_target()` 并重写 `_reward_rays()`

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 新增 `_compute_rays_omega_target()` 方法**

在 `_compute_rays_target_dir()` (line ~669) 和 `_reward_rays()` (line ~671) 之间插入：

```python
    def _compute_rays_omega_target(self):
        """Convert smoothed open-space direction to a target yaw angular velocity.

        open_dir_world (EMA smoothed) → body-frame heading_error →
        ω_target = clip(k_ω × heading_error, ±ω_max).

        Returns:
            omega_target (N,): target yaw angular velocity in rad/s.
        """
        cfg = self.cfg.pd_risknet
        k_omega = float(cfg.rays_omega_gain)
        omega_max = float(cfg.rays_omega_max)

        # open_dir_world → body frame
        smooth_dir_world_3d = torch.cat(
            [self._smooth_dir_world, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        open_dir_body = quat_apply_yaw_inverse(self.base_quat, smooth_dir_world_3d)[:, :2]

        # heading_error = signed angle from body-forward [1, 0] to open_dir_body
        heading_error = torch.atan2(open_dir_body[:, 1], open_dir_body[:, 0])

        omega_target = k_omega * heading_error
        omega_target = torch.clamp(omega_target, -omega_max, omega_max)

        return omega_target
```

- [ ] **Step 2: 重写 `_reward_rays()`**

将现有的 `_reward_rays()` 方法完整替换为：

```python
    def _reward_rays(self):
        """Angular-velocity tracking reward: encourages turning toward open space.

        Computes ω_target from the EMA-smoothed open-space direction, then
        rewards the robot for matching its actual yaw angular velocity to it.

        r = exp(-|ω_actual - ω_target|²)
        """
        cfg = self.cfg.pd_risknet
        alpha = float(cfg.rays_smoothing_alpha)

        # Step 1-3: raw target direction (world frame).
        target_dir_world = self._compute_rays_target_dir()  # (N, 2)

        # Step 4: EMA smooth in world frame.
        self._smooth_dir_world = (
            alpha * target_dir_world + (1.0 - alpha) * self._smooth_dir_world
        )
        smooth_norm = torch.norm(self._smooth_dir_world, dim=1, keepdim=True).clamp(min=1e-8)
        self._smooth_dir_world = self._smooth_dir_world / smooth_norm

        # Step 5: compute ω_target from smoothed direction.
        omega_target = self._compute_rays_omega_target()  # (N,)

        # Step 6: reward matching actual yaw angular velocity to ω_target.
        omega_actual = self.base_ang_vel[:, 2]
        omega_err = omega_actual - omega_target
        return torch.exp(-omega_err * omega_err)
```

- [ ] **Step 3: 验证 synthax 和 import**

```bash
python -c "
import torch
import sys
sys.path.insert(0, 'legged_gym')
# 只做语法检查，不创建完整环境
import ast
with open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py') as f:
    source = f.read()
ast.parse(source)
print('Syntax OK')
"
```
Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: rewrite _reward_rays as angular-velocity tracking + add _compute_rays_omega_target"
```

---

### Task 5: 更新记忆文件

**Files:**
- Modify: `.claude/projects/-home-t3chichi-Lidar-legged-gym/memory/reward-system-design.md`

- [ ] **Step 1: 更新 rays 奖励描述**

将文件内容更新为：

```markdown
---
name: reward-system-design
description: go2_lidar 避障奖励的当前实现和设计演进方向
metadata:
  type: reference
---

## 当前奖励实现 (2026-06-11 更新)

### rays 奖励（`_reward_rays`）
- **实质**：角速度跟踪奖励，鼓励机器人转向 LiDAR 感知到的"最开阔方向"
- **计算**：36 扇区加权平均 → body-frame open_dir → EMA 平滑 → heading_error =
  atan2(open_dir_y, open_dir_x) → ω_target = clip(k_ω × heading_error, ±ω_max) →
  r = exp(-|ω_actual - ω_target|²)
- k_ω = 0.5, ω_max = 0.5 rad/s

### vel_avoid 奖励（`_reward_vel_avoid`）
- **实质**：跟踪 (指令速度 + 避障偏移速度) 的合向量（合并形式，不拆分 x/y）
- v_avoid 计算：每扇区最小距离 → 指数权重 exp(-α·d) → 加权矢量求和（排斥方向），限制最大速度
- 奖励：exp(-β · ||v_actual - (v_cmd + v_avoid)||²)

### 设计决策
- rays 管旋转域 (ω_z)，vel_avoid 管平移域 (v_x, v_y)，完全解耦
- tracking_ang_vel 在避障地形中设为 0，rays 接管所有角速度引导
- 奖励信号不入观测空间，仅作为训练监督

## 设计演进

详见 [[position-based-command]]
```

- [ ] **Step 2: Commit**

```bash
git add .claude/projects/-home-t3chichi-Lidar-legged-gym/memory/reward-system-design.md
git commit -m "docs: update reward system memory after rays→ω decoupling"
```

---

### Task 6: 集成验证 — 运行环境初始化测试

**Files:**
- Test: `legged_gym/legged_gym/tests/test_env.py` (已有)

- [ ] **Step 1: 验证走廊配置环境可初始化**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg, Go2LidarPDRiskNetCfgPPO
cfg = Go2LidarPDRiskNetCfg()
cfg.env.num_envs = 4
cfg.terrain.num_rows = 1
cfg.terrain.num_cols = 1
cfg.terrain.curriculum = False
cfg.sim.dt = 0.005
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
from legged_gym.utils.task_registry import task_registry
task_registry.register('test_rays_omega', Go2LidarPDRiskNet, cfg, Go2LidarPDRiskNetCfgPPO())
env, _ = task_registry.make_env('test_rays_omega', args=None)
# 验证新方法和奖励可调用
env._compute_rays_omega_target()
r = env._reward_rays()
print(f'_reward_rays shape: {r.shape}, range: [{r.min().item():.4f}, {r.max().item():.4f}]')
r_v = env._reward_vel_avoid()
print(f'_reward_vel_avoid shape: {r_v.shape}, range: [{r_v.min().item():.4f}, {r_v.max().item():.4f}]')
print('Integration check PASSED')
"
```
Expected: 无崩溃，rays 奖励范围合理 (0~1)，vel_avoid 奖励正常输出。

- [ ] **Step 2: 运行已有测试确保无回归**

```bash
python -m pytest legged_gym/legged_gym/tests/test_env.py -v --timeout=120
```
Expected: 全部通过 (与改动无关，但确保环境仍可正常初始化)

- [ ] **Step 3: Commit (如有必要)**

如果测试产出修复，提交它们。
