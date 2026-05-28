# 观测空间 Heading 表示重构设计

## 问题

训练中出现两个异常行为：
- **训练后期**：机器人原地转圈
- **训练初期**：撞墙后猛拐弯

## 根本原因

观测空间中 `commands[:, 2]` 是 P 控制器的角速度输出：

```
commands[:, 2] = clip(P_gain * wrap(heading_target - current_heading), -1, 1)
```

该值依赖 `current_heading`——机器人自身状态。形成反馈回路：

```
策略动作 → 机器人转动 → heading 变化 → P 控制器输出变化
    → 观测变化 → 策略动作 → ...
```

- **转圈**：策略发现持续旋转 → P 控制器持续输出非零角速度 → `tracking_ang_vel` 奖励给正反馈 → 局部最优
- **猛拐**：早期策略读不懂 LiDAR → 撞墙 → P 控制器输出大幅修正 + `v_avoid` 骤增 → 观测突变 → 极端动作

## 修改方案

**核心思路**：将观测中的 P 控制器角速度输出替换为独立的 heading 目标 + 当前朝向，断开反馈回路。P 控制器退居奖励端作为软性辅助信号。

通过 `HEADING_OBS_ENABLED` 总开关控制新旧模式。关闭时完全保留旧行为，兼容已有 checkpoint。

### 修改 0：总开关（每个配置文件顶部）

```python
# 模块级常量：True=新观测，False=旧观测（兼容已有 checkpoint）
HEADING_OBS_ENABLED = True

# PD_PROPRIO_DIM 由开关自动派生
PD_PROPRIO_DIM = 48 + (1 if HEADING_OBS_ENABLED else 0)
```

涉及文件（替换现有的 `PD_PROPRIO_DIM = 48`）：
- `go2_lidar_pd_risknet_config.py` (line 15)
- `go2_lidar_pillar_config.py` (line 15)
- `go2_pd_pretrain_config.py` (line 15)

### 修改 1：`pd_risknet` 配置类新增参数

```python
class pd_risknet:
    # 总开关（引用模块级常量）
    heading_obs_enabled = HEADING_OBS_ENABLED
    # Sim2real 朝向噪声（仅 heading_obs_enabled=True 时生效）
    heading_noise_enabled = True
    heading_noise_std = 0.05
```

涉及文件：
- `go2_lidar_pd_risknet_config.py`（`Go2LidarPDRiskNetCfg.pd_risknet`，line 35）
- `go2_lidar_pillar_config.py`（`Go2LidarPillarCfg.pd_risknet`，line 35）
- `go2_pd_pretrain_config.py`（`Go2LidarPDRiskNetCfg.pd_risknet`，line 33）

### 修改 2：新增观测缩放

在三个配置各自的 `obs_scales` 中新增：

```python
class obs_scales(Go2RoughCfg.normalization.obs_scales):
    heading = 1.0
```

涉及文件：
- `go2_lidar_pd_risknet_config.py` (line 189)
- `go2_lidar_pillar_config.py` (line 176)
- `go2_pd_pretrain_config.py` (line 160)

### 修改 3：`compute_observations`（`go2_lidar_pd_risknet.py`）

根据 `heading_obs_enabled` 分支构建 `cmd_obs` 和 `proprio_obs`：

```python
def compute_observations(self):
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
            self.base_lin_vel * self.obs_scales.lin_vel,       # [0:3]
            self.base_ang_vel * self.obs_scales.ang_vel,       # [3:6]
            self.projected_gravity,                             # [6:9]
            cmd_obs,                                            # [9:12]
            current_heading.unsqueeze(1) * self.obs_scales.heading,  # [12] 新增
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # [13:25]
            self.dof_vel * self.obs_scales.dof_vel,             # [25:37]
            self.actions,                                       # [37:49]
        ), dim=-1)
    else:
        # ── 旧观测：P 控制器角速度（兼容已有 checkpoint）──
        proprio_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,       # [0:3]
            self.base_ang_vel * self.obs_scales.ang_vel,       # [3:6]
            self.projected_gravity,                             # [6:9]
            self.commands[:, :3] * self.commands_scale,         # [9:12]
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # [12:24]
            self.dof_vel * self.obs_scales.dof_vel,             # [24:36]
            self.actions,                                       # [36:48]
        ), dim=-1)

    self.obs_buf = torch.cat((
        proprio_obs,
        self.lidar_points_base.reshape(self.num_envs, -1),
    ), dim=-1)

    if self.privileged_obs_buf is not None:
        self.privileged_obs_buf = torch.cat((proprio_obs, self.measured_heights), dim=-1)

    if self.add_noise:
        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
```

### 修改 4：`_get_noise_scale_vec`（`go2_lidar_pd_risknet.py`）

索引需要适配新观测布局。根据开关分支：

```python
def _get_noise_scale_vec(self, cfg):
    noise_vec = torch.zeros_like(self.obs_buf[0])
    self.add_noise = self.cfg.noise.add_noise
    noise_scales = self.cfg.noise.noise_scales
    noise_level = self.cfg.noise.noise_level

    noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
    noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    noise_vec[6:9] = noise_scales.gravity * noise_level
    noise_vec[9:12] = 0.0  # commands

    if self.cfg.pd_risknet.heading_obs_enabled:
        noise_vec[12:13] = 0.0                         # current_heading: 不额外加噪（已有独立噪声）
        noise_vec[13:25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[25:37] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[37:49] = 0.0                         # previous actions
    else:
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.0

    return noise_vec
```

### 修改 5：P 控制器 P gain

| 文件 | 参数 | 旧值 | 新值 |
|------|------|------|------|
| `go2_lidar_pillar_config.py` | `heading_p_gain` | 1.0 | 0.5 |
| `go2_pd_pretrain_config.py` | `heading_p_gain` | 1.0 | 0.5 |

`go2_lidar_pd_risknet_config.py` 已经是 0.5，无需修改。

### 修改 6：奖励权重

保持当前值不变。

## 不改的内容

- **`legged_robot.py`** — 不碰基类
- **`go2_rough_config.py`** — 不碰 Go2 通用配置
- **`pd_risknet_actor_critic.py`** — 默认值保持 48。三个配置均显式传入 `proprio_obs_dim = PD_PROPRIO_DIM`，默认值仅用于防御性编程。且改动默认值会破坏独立测试
- P 控制器逻辑（`_post_physics_step_callback`）
- 命令重采样逻辑（`_resample_commands`）
- 避障奖励（`vel_avoid`、`rays`）
- LiDAR 观测、近端/远端感知路径
- `test_go2_lidar_pd_risknet_math.py` — 测试独立创建模型，不依赖配置常量

## 开关行为总结

| `HEADING_OBS_ENABLED` | `PD_PROPRIO_DIM` | cmd_obs 第三通道 | proprio_obs | 总 obs dim | checkpoint |
|-----------------------|------------------|-----------------|-------------|-----------|------------|
| `False` | 48 | P 控制器角速度 | 48 维 | 2640 | 兼容旧模型 |
| `True` | 49 | heading 目标 | 49 维（含 current_heading） | 2641 | 新模型 |

切换方式：修改对应配置文件顶部的 `HEADING_OBS_ENABLED = True/False`。

## 架构对比

```
旧模式 (HEADING_OBS_ENABLED=False):
  heading_target → P控制器 → yaw_cmd → 策略观测 → 动作 → heading变化 → 循环

新模式 (HEADING_OBS_ENABLED=True):
  heading_target ────────────────────→ 策略观测 → 动作 (无循环)
  current_heading ──(+独立噪声)─────→ 策略观测
  heading_target → P控制器 → yaw_cmd → tracking_ang_vel 奖励 (软建议)
```

## 影响范围

| 文件 | 改动 |
|------|------|
| `go2_lidar_pd_risknet.py` | `compute_observations` 分支 + `_get_noise_scale_vec` 分支 |
| `go2_lidar_pd_risknet_config.py` | `HEADING_OBS_ENABLED`，`PD_PROPRIO_DIM` 派生，`pd_risknet` 新增 3 参数，`obs_scales.heading` |
| `go2_lidar_pillar_config.py` | 同上 + `heading_p_gain` 1.0→0.5 |
| `go2_pd_pretrain_config.py` | 同上 + `heading_p_gain` 1.0→0.5 |

## 影响确认

以下项目**不受影响**：
- ANYmal C、ElSpider Air、Cyberdog2、Cassie、A1、Franka — 使用 `ActorCritic`，不涉及 `PDRiskNetActorCritic`
- Go2 基础任务（`go2_rough`, `go2_flat`）— 使用 `ActorCritic`，`Go2LidarPDRiskNet.compute_observations` 不会被执行
- `pd_risknet_actor_critic.py` — 模块不变
- `legged_robot.py` — 模块不变
- 已有测试 — 独立于配置，继续通过

## 潜在风险

1. **pillar / pretrain heading 范围 [-3.14, 3.14] 跨越 0/2π 边界**：raw heading 存在回绕。corridor 场景（[0.87, 2.27]）不受影响。后续可对 pillar/pretrain 改用 cos/sin 编码。

2. **旧 checkpoint 兼容**：`HEADING_OBS_ENABLED = False` 时完全兼容。新旧 checkpoint 互不兼容，切换开关后需重新训练。
