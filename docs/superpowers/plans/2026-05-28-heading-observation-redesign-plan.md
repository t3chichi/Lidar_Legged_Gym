# Heading 观测重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Go2 PD-RiskNet 项目中添加 `HEADING_OBS_ENABLED` 开关，支持新观测（heading 目标 + current_heading）和旧观测（P 控制器角速度）两种模式，断开观测-动作反馈回路。

**Architecture:** 三个配置文件通过模块级开关 `HEADING_OBS_ENABLED` 控制 `PD_PROPRIO_DIM`（48/49）的动态派生。`go2_lidar_pd_risknet.py` 中 `compute_observations` 和 `_get_noise_scale_vec` 各加分支处理新旧布局。其他项目文件不变。

**Tech Stack:** Python, PyTorch, isaacgym

---

### Task 1: 修改 corridor 配置文件

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py:15`

- [ ] **Step 1: 添加模块级总开关，替换硬编码 PD_PROPRIO_DIM**

将 line 15:
```python
PD_PROPRIO_DIM = 48
```

替换为:
```python
HEADING_OBS_ENABLED = True

PD_PROPRIO_DIM = 48 + (1 if HEADING_OBS_ENABLED else 0)
```

- [ ] **Step 2: 在 pd_risknet 内部类添加开关和噪声参数**

在 `class pd_risknet:` 的 `split_theta_deg = PD_THETA_DEG` 之后（line 43 后）插入:

```python
        # 观测模式开关及朝向噪声配置
        heading_obs_enabled = HEADING_OBS_ENABLED
        heading_noise_enabled = True
        heading_noise_std = 0.05
```

- [ ] **Step 3: 在 obs_scales 中添加 heading 缩放系数**

将 line 189-190:
```python
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            pass
```

替换为:
```python
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            heading = 1.0
```

- [ ] **Step 4: 验证 Python 语法**

```bash
python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cfg','legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('PD_PROPRIO_DIM:', m.PD_PROPRIO_DIM); print('heading_obs_enabled:', m.Go2LidarPDRiskNetCfg.pd_risknet.heading_obs_enabled)"
```

Expected: `PD_PROPRIO_DIM: 49`, `heading_obs_enabled: True`

### Task 2: 修改 pillar 配置文件

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py`

- [ ] **Step 1: 添加模块级总开关，替换硬编码 PD_PROPRIO_DIM**

将 line 15 `PD_PROPRIO_DIM = 48` 替换为:

```python
HEADING_OBS_ENABLED = True

PD_PROPRIO_DIM = 48 + (1 if HEADING_OBS_ENABLED else 0)
```

- [ ] **Step 2: 在 pd_risknet 内部类添加开关和噪声参数**

在 `class pd_risknet:` 的 `split_theta_deg = PD_THETA_DEG` 之后插入:

```python
        # 观测模式开关及朝向噪声配置
        heading_obs_enabled = HEADING_OBS_ENABLED
        heading_noise_enabled = True
        heading_noise_std = 0.05
```

- [ ] **Step 3: 修改 heading_p_gain 1.0 → 0.5**

将 line 115 `heading_p_gain = 1.0` 替换为:

```python
        heading_p_gain = 0.5     # P 增益
```

- [ ] **Step 4: 在 obs_scales 中添加 heading 缩放系数**

将 pillar config 中 `class obs_scales(Go2RoughCfg.normalization.obs_scales): pass` 替换为:

```python
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            heading = 1.0
```

- [ ] **Step 5: 验证语法**

```bash
python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cfg','legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('PD_PROPRIO_DIM:', m.PD_PROPRIO_DIM); print('P gain:', m.Go2LidarPillarCfg.commands.heading_p_gain)"
```

Expected: `PD_PROPRIO_DIM: 49`, `P gain: 0.5`

### Task 3: 修改 pretrain 配置文件

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py`

- [ ] **Step 1: 添加模块级总开关，替换硬编码 PD_PROPRIO_DIM**

将 line 15 `PD_PROPRIO_DIM = 48` 替换为:

```python
HEADING_OBS_ENABLED = True

PD_PROPRIO_DIM = 48 + (1 if HEADING_OBS_ENABLED else 0)
```

- [ ] **Step 2: 在 pd_risknet 内部类添加开关和噪声参数**

在 `class pd_risknet:` 的 `split_theta_deg = PD_THETA_DEG` 之后插入:

```python
        # 观测模式开关及朝向噪声配置
        heading_obs_enabled = HEADING_OBS_ENABLED
        heading_noise_enabled = True
        heading_noise_std = 0.05
```

- [ ] **Step 3: 修改 heading_p_gain 1.0 → 0.5**

将 `heading_p_gain = 1.0` 替换为:

```python
        heading_p_gain = 0.5       # P 增益
```

- [ ] **Step 4: 在 obs_scales 中添加 heading 缩放系数**

将 `class obs_scales(Go2RoughCfg.normalization.obs_scales): pass` 替换为:

```python
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            heading = 1.0
```

- [ ] **Step 5: 验证语法**

```bash
python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cfg','legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('PD_PROPRIO_DIM:', m.PD_PROPRIO_DIM); print('P gain:', m.Go2LidarPDRiskNetCfg.commands.heading_p_gain)"
```

Expected: `PD_PROPRIO_DIM: 49`, `P gain: 0.5`

### Task 4: 修改 compute_observations 添加分支

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:598-627`

- [ ] **Step 1: 重写 compute_observations 方法**

将 line 598-627 整个 `compute_observations` 方法替换为:

```python
    def compute_observations(self):
        # Keep base proprio order identical to Go2/LeggedRobot, then append LiDAR history.
        if self.cfg.pd_risknet.heading_obs_enabled:
            # ── 新观测：heading 目标 + current_heading ──
            cmd_obs = torch.cat((
                self.commands[:, 0:1] * self.obs_scales.lin_vel,
                self.commands[:, 1:2] * self.obs_scales.lin_vel,
                self.commands[:, 3:4] * self.obs_scales.heading,
            ), dim=-1)

            forward = quat_apply(self.base_quat, self.forward_vec)
            current_heading = torch.atan2(forward[:, 1], forward[:, 0])
            if self.cfg.pd_risknet.heading_noise_enabled:
                current_heading = current_heading + torch.randn_like(current_heading) * self.cfg.pd_risknet.heading_noise_std

            proprio_obs = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                cmd_obs,
                current_heading.unsqueeze(1) * self.obs_scales.heading,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ), dim=-1)
        else:
            # ── 旧观测：P 控制器角速度（兼容已有 checkpoint）──
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

            # 临时诊断打印（确认后可注释)
            if not hasattr(self, '_printed_height_shape'):
                print(f"[INFO] measured_heights.shape: {self.measured_heights.shape}")
                self._printed_height_shape = True

            self.privileged_obs_buf = torch.cat((proprio_obs, self.measured_heights), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
```

### Task 5: 修改 _get_noise_scale_vec 添加分支

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:43-62`

- [ ] **Step 1: 重写 _get_noise_scale_vec 方法**

将 line 43-62 整个 `_get_noise_scale_vec` 方法替换为:

```python
    def _get_noise_scale_vec(self, cfg):
        """Use Go2 proprio noise only; keep LiDAR history channels noise-free by default.

        Base LeggedRobot assumes height-map observations at indices [48:235] when
        terrain.measure_heights=True. This task replaces that block with flattened
        LiDAR history, so we override the mapping to avoid injecting wrong noise.
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0  # commands

        if self.cfg.pd_risknet.heading_obs_enabled:
            noise_vec[12:13] = 0.0  # current_heading: 不额外加噪（已有独立噪声机制）
            noise_vec[13:25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[25:37] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[37:49] = 0.0  # previous actions
        else:
            noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[36:48] = 0.0  # previous actions

        return noise_vec
```

### Task 6: 运行测试验证

- [ ] **Step 1: 运行 PD-RiskNet 数学测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

Expected: 所有测试 PASS

- [ ] **Step 2: 运行环境测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_env.py -v
```

Expected: PASS（这些测试不涉及 PD-RiskNet，仅验证基类完整性）

- [ ] **Step 3: 快速 train.py 干跑验证（headless 模式，1 步退出）**

```bash
python legged_gym/legged_gym/scripts/train.py --task=go2_lidar_pd_risknet --num_envs=4 --headless --max_iterations=1
```

Expected: 成功启动，完成 1 步迭代后正常退出，无维度错误或 NaN。

### Task 7: 提交

- [ ] **Step 1: 提交全部修改**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py \
        legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py \
        legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py \
        legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py \
        docs/superpowers/specs/2026-05-28-heading-observation-redesign.md \
        docs/superpowers/plans/2026-05-28-heading-observation-redesign-plan.md
git commit -m "$(cat <<'EOF'
feat: replace P-controller yaw obs with heading target + current_heading

Add HEADING_OBS_ENABLED toggle to switch between new observation design
(heading target + current_heading, breaks feedback loop) and old design
(P-controller angular velocity output, compatible with existing checkpoints).

Changes:
- Three configs: dynamic PD_PROPRIO_DIM (48/49), new pd_risknet params
- compute_observations: branch on heading_obs_enabled
- _get_noise_scale_vec: adjust hardcoded indices for new layout
- Pillar/pretrain: heading_p_gain 1.0 → 0.5
EOF
)"
```
