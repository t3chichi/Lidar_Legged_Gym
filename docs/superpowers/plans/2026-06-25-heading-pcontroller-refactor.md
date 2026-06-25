# Heading/P控制器重构 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除观测中的朝向相关代码，开启 heading_command 用 P 控制器将 heading 转为 ang_vel_yaw。

**Architecture:** 基类 `LeggedRobot._post_physics_step_callback` 已包含 P 控制器逻辑，只需开启 `heading_command = True` 激活。观测统一使用 `commands[:, :3]` 48 维格式。主要改动在 config 配置常量和 `compute_observations`/`_get_noise_scale_vec` 方法。

**Tech Stack:** Python, PyTorch, Isaac Gym

---

### Task 1: 更新 config — 删除 heading_obs 开关和维度常量

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py`

- [ ] **Step 1: 删除 HEADING_OBS_ENABLED，更新 PD_PROPRIO_DIM 和 PD_PRIV_CRITIC_DIM**

将文件头部常量区的第 14、16-17、24 行修改为：

```python
# 删除 L14: HEADING_OBS_ENABLED = False

# L16 改为:
PD_PROPRIO_DIM = 48

# L24 改为:
PD_PRIV_CRITIC_DIM = PD_PROPRIO_DIM + PD_PRIV_HEIGHT_DIM
```

- [ ] **Step 2: 从 pd_risknet 类中删除 heading_obs 相关配置**

删除以下三行（在 `pd_risknet` 类中，当前约在第 47-49 行）：

```python
# 删除这三行:
heading_obs_enabled = HEADING_OBS_ENABLED
heading_noise_enabled = True
heading_noise_std = 0.05
```

- [ ] **Step 3: 开启 heading_command 并添加 heading 范围**

修改 `commands` 类（约第 141-149 行）：

```python
class commands(Go2RoughCfg.commands):
    heading_command = True        # 改为 True，启用 P 控制器
    resampling_time = 2.
    curriculum = False

    class ranges(Go2RoughCfg.commands.ranges):
        lin_vel_x = [0.5, 1.0]
        lin_vel_y = [-0.0, 0.0]
        ang_vel_yaw = [-0.0, 0.0]
        heading = [-3.14, 3.14]   # 新增
```

- [ ] **Step 4: 删除 obs_scales 中的 heading**

修改 `normalization` 类（约第 215-216 行）：

```python
class normalization(Go2RoughCfg.normalization):
    class obs_scales(Go2RoughCfg.normalization.obs_scales):
        pass  # 删除 heading = 1.0，不再有额外成员
```

- [ ] **Step 5: 更新 env.num_observations 和 env.num_privileged_obs**

修改 `env` 类（约第 99-103 行）：

```python
class env(Go2RoughCfg.env):
    num_observations = PD_PROPRIO_DIM + OBS_HISTORY_LENGTH * PD_NUM_LIDAR_POINTS * 3
    num_privileged_obs = PD_PRIV_CRITIC_DIM
```

（实际值: `num_observations = 48 + 1 * 1000 * 3 = 3048`, `num_privileged_obs = 235`）

- [ ] **Step 6: 更新 PPO 配置中的 proprio_obs_dim 和 privileged_critic_dim**

修改 `Go2LidarPDRiskNetCfgPPO.policy` 类（约第 259-261 行）：

```python
class policy(Go2RoughCfgPPO.policy):
    # ... 其他保持不变 ...
    proprio_obs_dim = PD_PROPRIO_DIM          # 48
    privileged_height_dim = PD_PRIV_HEIGHT_DIM  # 187
    privileged_critic_dim = PD_PRIV_CRITIC_DIM  # 235
```

- [ ] **Step 7: 验证 config 导入无语法错误**

```bash
python -c "from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg; print('PD_PROPRIO_DIM:', Go2LidarPDRiskNetCfg.pd_risknet.proximal_feature_dim); print('num_observations:', Go2LidarPDRiskNetCfg.env.num_observations)"
```

预期: 打印 `PD_PROPRIO_DIM` 和 `num_observations` 值，无错误。

- [ ] **Step 8: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "$(cat <<'EOF'
refactor: remove heading from observations, enable P-controller for heading→ang_vel conversion

- Delete heading_obs_enabled/noise config and HEADING_OBS_ENABLED constant
- Set heading_command=True with P gain 0.5 (base class already implements P controller)
- Fix PD_PROPRIO_DIM to 48 (was 49 when heading_obs_enabled)
- Update num_observations, num_privileged_obs, proprio_obs_dim, privileged_critic_dim
EOF
)"
```

---

### Task 2: 更新环境代码 — 删除观测中的 heading 分支

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 简化 _get_noise_scale_vec — 删除 heading 分支**

将 `_get_noise_scale_vec` 方法（约第 116-143 行）替换为：

```python
def _get_noise_scale_vec(self, cfg):
    """Proprio noise only; LiDAR channels are noise-free by default."""
    noise_vec = torch.zeros_like(self.obs_buf[0])
    self.add_noise = self.cfg.noise.add_noise
    noise_scales = self.cfg.noise.noise_scales
    noise_level = self.cfg.noise.noise_level

    noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
    noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    noise_vec[6:9] = noise_scales.gravity * noise_level
    noise_vec[9:12] = 0.0  # commands
    noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    noise_vec[36:48] = 0.0  # previous actions

    return noise_vec
```

- [ ] **Step 2: 简化 compute_observations — 删除 heading_obs_enabled 分支**

将 `compute_observations` 方法中从 `if self.cfg.pd_risknet.heading_obs_enabled:` 到 `self.obs_buf = torch.cat(...)` 的部分（约第 1071-1124 行）替换为：

```python
def compute_observations(self):
    # Base proprioception: 48-dim, matching LeggedRobot convention.
    proprio_obs = torch.cat((
        self.base_lin_vel * self.obs_scales.lin_vel,
        self.base_ang_vel * self.obs_scales.ang_vel,
        self.projected_gravity,
        self.commands[:, :3] * self.commands_scale,
        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        self.dof_vel * self.obs_scales.dof_vel,
        self.actions,
    ), dim=-1)

    self.obs_buf = torch.cat((
        proprio_obs,
        self.lidar_points_base.reshape(self.num_envs, -1),
    ), dim=-1)

    # Privileged channel for critic: proprio + terrain height samples.
    if self.privileged_obs_buf is not None:
        self.privileged_obs_buf = torch.cat((proprio_obs, self.measured_heights), dim=-1)

    if self.add_noise:
        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
```

- [ ] **Step 3: 清理导入（不再需要 math_utils 中的 heading 相关引用）**

检查文件开头的 import 区域。确认 `quat_apply_yaw`, `quat_apply_yaw_inverse` 等仍被其他方法使用（`_compute_rays_omega_target`、`_draw_debug_vis` 等），**不需要**修改 import。

- [ ] **Step 4: 验证 Python 语法**

```bash
python -c "import sys; sys.path.insert(0, 'legged_gym'); from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet; print('Import OK')"
```

预期: `Import OK`，无错误。

- [ ] **Step 5: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "$(cat <<'EOF'
refactor: remove heading_obs_enabled branch from observations and noise

Always use commands[:, :3] in observations (48-dim proprio). The P-controller
in the base class LeggedRobot._post_physics_step_callback converts heading
commands to ang_vel_yaw every step when heading_command=True.
EOF
)"
```

---

### Task 3: 验证

**Files:** (none created/modified)

- [ ] **Step 1: 运行数学测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

预期: 所有测试通过。

- [ ] **Step 2: 运行基础环境测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_env.py -v
```

预期: 所有测试通过。

- [ ] **Step 3: 验证任务注册正常**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet import Go2LidarPDRiskNet
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg, Go2LidarPDRiskNetCfgPPO
cfg = Go2LidarPDRiskNetCfg()
print(f'num_observations: {cfg.env.num_observations}')
print(f'num_privileged_obs: {cfg.env.num_privileged_obs}')
print(f'proprio_obs_dim: {cfg.policy.proprio_obs_dim}')
print(f'privileged_critic_dim: {cfg.policy.privileged_critic_dim}')
print(f'heading_command: {cfg.commands.heading_command}')
# 验证维度一致性
expected_obs = cfg.policy.proprio_obs_dim + 1 * 1000 * 3
assert cfg.env.num_observations == expected_obs, f'obs mismatch: {cfg.env.num_observations} != {expected_obs}'
assert cfg.env.num_privileged_obs == cfg.policy.privileged_critic_dim, 'privileged obs mismatch'
assert cfg.policy.proprio_obs_dim == 48, f'proprio_obs_dim should be 48, got {cfg.policy.proprio_obs_dim}'
print('All config checks passed!')
"
```

预期: 打印所有配置值，所有断言通过。
