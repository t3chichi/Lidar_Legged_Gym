# 对称数据增强：基于 wrapped observation 的左右镜像

**日期**: 2026-07-01
**状态**: 已实现，已审计

## 目标

在 Go2 CmdSafe 策略训练中引入左右对称数据增强，使策略具备环境镜像不变性（equivariance）。核心约束：

- **机器人不对称性保留**：Go2 髋关节默认角度左右不对称（FL/RL = +0.1 rad, FR/RR = −0.1 rad），补偿机制必须正确抵消
- **环境镜像**：LiDAR 点云做 Y 翻转，模拟左右翻转后的地形/障碍物
- **策略 equivariance**：给定镜像观测，策略输出镜像动作

## 架构

```
                    CmdSafeHistoryWrapper
                    (split → FPS → downsample → angular sort)
                              │
                              ▼
                    wrapped_obs (2736 dims)
                              │
                ┌─────────────┼─────────────┐
                │                           │
           [48 proprio]          [768 prox pcd] + [1920 distal pcd]
                │                           │
        scalar flips + swap           Y-flip + angular-key re-sort
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    mirrored_obs (2736 dims)
                              │
                    cat([obs, mirrored_obs])
                              │
                              ▼
                    PPO update + mirror loss
```

对称增强在 **wrapped observation 层面** 操作，不需要重新跑 FPS/下采样 pipeline。点云仅做 Y 翻转 + angular key 重排，与 HistoryWrapper 共用 `sort_points_by_angular_key()`，保证处理流程一致。

## Observation 布局 (2736 dims)

```
[0:48]    proprio (48)
[48:816]   proximal LiDAR: 256 points × 3 dims = 768, angular-sorted
[816:2736] distal LiDAR:   640 points × 3 dims = 1920, angular-sorted
           (64 points/frame × 10-frame history, globally angular-sorted)
```

### Proprioceptive 48 维详细布局

| 索引 | 内容 | 维数 | 镜像变换 |
|------|------|------|----------|
| 0–2 | base_lin_vel × `lin_vel_scale` | 3 | vy (idx 1) 取反 |
| 3–5 | base_ang_vel × `ang_vel_scale` | 3 | wx (idx 3), wz (idx 5) 取反 |
| 6–8 | projected_gravity | 3 | gy (idx 7) 取反 |
| 9–11 | commands × `commands_scale` | 3 | cmd_vy (idx 10), cmd_wz (idx 11) 取反 |
| 12–23 | (dof_pos − default_dof_pos) × `dof_pos_scale` | 12 | FL↔FR, RL↔RR 交换 + 髋取反 |
| 24–35 | dof_vel × `dof_vel_scale` | 12 | FL↔FR, RL↔RR 交换 + 髋取反 |
| 36–47 | prev_actions (raw policy output) | 12 | FL↔FR, RL↔RR 交换 + 髋取反 |

### 标量取反的物理依据

左右镜像为 Y 轴反射（Y → −Y）。对于矢量 v 和赝矢量 ω（变换规则 ω′ = det(R)·R·ω，其中 R = diag(1, −1, 1), det(R) = −1）：

| 量 | 类型 | Y 反射变换 | 代码 idx |
|----|------|------------|----------|
| vy (线速度 Y 分量) | 矢量 | −vy | 1 |
| wx (角速度 X 分量) | 赝矢量 | −wx | 3 |
| wz (角速度 Z 分量) | 赝矢量 | −wz | 5 |
| gy (重力 Y 投影) | 矢量 | −gy | 7 |
| cmd_vy (侧向指令) | 矢量分量 | −cmd_vy | 10 |
| cmd_wz (偏航角速度指令) | 赝矢量分量 | −cmd_wz | 11 |

## Go2 关节 DOF 布局与镜像映射

Go2 关节 URDF 顺序：

```
FL(hip, thigh, calf)  FR(hip, thigh, calf)  RL(hip, thigh, calf)  RR(hip, thigh, calf)
 0    1      2         3    4      5          6    7      8          9   10     11
```

默认角度（`go2_rough_config.py`）：

| 关节 | FL | FR | RL | RR |
|------|-----|-----|-----|-----|
| hip | +0.1 | −0.1 | +0.1 | −0.1 |
| thigh | 0.8 | 0.8 | 0.8 | 0.8 |
| calf | −1.5 | −1.5 | −1.5 | −1.5 |

髋关节轴为 `[1,0,0]`（统一 X 方向），FL/RL 在 +Y 侧，FR/RR 在 −Y 侧。相同旋转角度在左右两侧产生相反的腿部运动方向，因此左右交换时髋关节必须取反。Thigh/calf 关节轴左右对称，直接交换即可。

## 对称算法细节

### 1. 标量取反

```python
obs_mirrored[:, 1]  = -obs[:, 1]   # vy
obs_mirrored[:, 3]  = -obs[:, 3]   # wx
obs_mirrored[:, 5]  = -obs[:, 5]   # wz
obs_mirrored[:, 7]  = -obs[:, 7]   # gy
obs_mirrored[:, 10] = -obs[:, 10]  # cmd_vy
obs_mirrored[:, 11] = -obs[:, 11]  # cmd_wz
```

### 2. DOF 位置镜像（含默认值补偿）

```python
# Decode: 恢复绝对关节角度
dof_raw = obs[:, 12:24] / dof_obs_scale + default_dof_pos

# Swap leg groups
dof_mirrored_raw[:, 0:3]  = dof_raw[:, 3:6]   # FL ← FR
dof_mirrored_raw[:, 3:6]  = dof_raw[:, 0:3]   # FR ← FL
dof_mirrored_raw[:, 6:9]  = dof_raw[:, 9:12]  # RL ← RR
dof_mirrored_raw[:, 9:12] = dof_raw[:, 6:9]   # RR ← RL

# Negate hip joints (index 0 in each group → global index 0, 3, 6, 9)
dof_mirrored_raw[:, 0] = -dof_mirrored_raw[:, 0]
dof_mirrored_raw[:, 3] = -dof_mirrored_raw[:, 3]
dof_mirrored_raw[:, 6] = -dof_mirrored_raw[:, 6]
dof_mirrored_raw[:, 9] = -dof_mirrored_raw[:, 9]

# Re-encode
obs_mirrored[:, 12:24] = (dof_mirrored_raw - default_dof_pos) * dof_obs_scale
```

**默认值补偿证明**：以 FL/FR 髋关节为例：

```
new_FL_obs = (−FR_raw − FL_default) × scale
           = (−(FR_obs/scale + FR_default) − FL_default) × scale
           = −FR_obs − (FR_default + FL_default) × scale
```

因为 `FL_default + FR_default = 0.1 + (−0.1) = 0`，`RL_default + RR_default = 0.1 + (−0.1) = 0`，默认值严格抵消：`new_FL_obs = −old_FR_obs`，`new_FR_obs = −old_FL_obs`。

### 3. DOF 速度镜像

DOF 速度无默认值偏移，直接交换 + 髋取反：

```python
obs_mirrored[:, 24:27] = obs[:, 27:30]  # FL ← FR
obs_mirrored[:, 27:30] = obs[:, 24:27]  # FR ← FL
obs_mirrored[:, 30:33] = obs[:, 33:36]  # RL ← RR
obs_mirrored[:, 33:36] = obs[:, 30:33]  # RR ← RL

obs_mirrored[:, 24] = -obs_mirrored[:, 24]  # FL hip
obs_mirrored[:, 27] = -obs_mirrored[:, 27]  # FR hip
obs_mirrored[:, 30] = -obs_mirrored[:, 30]  # RL hip
obs_mirrored[:, 33] = -obs_mirrored[:, 33]  # RR hip
```

### 4. 前一帧动作镜像

`obs[:, 36:48]` 存储策略上一帧的 raw output（未经 action_scale/default_dof_pos 变换）。因为 raw action 区间居中（≈ [−1, 1]），无默认值偏移，交换 + 髋取反即可：

```python
acts_mirrored_raw[:, 0:3]  = acts_raw[:, 3:6]   # 交换
acts_mirrored_raw[:, 3:6]  = acts_raw[:, 0:3]
acts_mirrored_raw[:, 6:9]  = acts_raw[:, 9:12]
acts_mirrored_raw[:, 9:12] = acts_raw[:, 6:9]
acts_mirrored_raw[:, 0] = -acts_mirrored_raw[:, 0]  # 髋取反
acts_mirrored_raw[:, 3] = -acts_mirrored_raw[:, 3]
acts_mirrored_raw[:, 6] = -acts_mirrored_raw[:, 6]
acts_mirrored_raw[:, 9] = -acts_mirrored_raw[:, 9]
obs_mirrored[:, 36:48] = acts_mirrored_raw
```

### 5. LiDAR 点云镜像

```python
# Proximal (indices 48:816): 256 points
prox_pts = obs_mirrored[:, 48:816].reshape(-1, 256, 3)
prox_pts[:, :, 1] = -prox_pts[:, :, 1]        # Y flip
prox_sorted = sort_points_by_angular_key(prox_pts, sensor_quat, sensor_trans)
obs_mirrored[:, 48:816] = prox_sorted.reshape(-1, 768)

# Distal (indices 816:2736): 640 points
dist_pts = obs_mirrored[:, 816:].reshape(-1, 640, 3)
dist_pts[:, :, 1] = -dist_pts[:, :, 1]        # Y flip
dist_sorted = sort_points_by_angular_key(dist_pts, sensor_quat, sensor_trans)
obs_mirrored[:, 816:] = dist_sorted.reshape(-1, 1920)
```

Y 翻转后每个点的 `azimuth = atan2(y, x)` 变号，`phi` 不变（r 中 y² 不变），angular key 改变。重排序恢复 angular key 递增的扫描方向，保证 GRU 输入约定不变。`sort_points_by_angular_key` 与 `CmdSafeHistoryWrapper` 共用同一函数，流程一致。

### 6. Action 增强（Stage 2）

```python
acts_mirrored = actions.clone()
acts_mirrored[:, 0:3]  = actions[:, 3:6]   # FL ← FR
acts_mirrored[:, 3:6]  = actions[:, 0:3]   # FR ← FL
acts_mirrored[:, 6:9]  = actions[:, 9:12]  # RL ← RR
acts_mirrored[:, 9:12] = actions[:, 6:9]   # RR ← RL
acts_mirrored[:, 0] = -acts_mirrored[:, 0]  # FL hip
acts_mirrored[:, 3] = -acts_mirrored[:, 3]  # FR hip
acts_mirrored[:, 6] = -acts_mirrored[:, 6]  # RL hip
acts_mirrored[:, 9] = -acts_mirrored[:, 9]  # RR hip
actions_augmented = torch.cat([actions, acts_mirrored], dim=0)
```

Actions 是策略原始输出（raw action），交换 + 髋取反即可，无需默认值补偿。

### 7. Auxiliary 高度网格镜像（critic obs）

```python
# 网格布局: [B, x_count × y_count], X-major (x_count=17, y_count=11)
obs_mirrored = torch.flip(
    obs.reshape(-1, height_grid_x_count, height_grid_y_count),
    dims=[2],  # 沿 Y 轴反转行内顺序
).reshape_as(obs)
```

## Mirror Loss

训练中额外施加 equivariance 约束：

```
MSE(policy(mirror(obs)), mirror(policy(obs)))
```

实现流程：

```python
# 1. 用当前策略预测所有 augmented obs (原始 + 镜像) 的 mean action
mean_actions_batch = self.policy.act_inference(obs_batch.detach())

# 2. 对原始样本的预测做 action 镜像
action_mean_orig = mean_actions_batch[:batch_size]
_, actions_mean_symm_batch = symmetry_func(obs=None, actions=action_mean_orig, ...)

# 3. MSE 损失
symmetry_loss = MSE(
    mean_actions_batch[batch_size:],                    # policy(mirror(obs))
    actions_mean_symm_batch.detach()[batch_size:]       # mirror(policy(obs))
)
```

## 数学一致性

PPO 使用 scalar std（12 维共享同一 σ）。对称变换为交换 + 部分维度取反，是正交变换（保范数），因此：

```
log_prob(mirror(a) | mirror(μ)) = −Σⱼ(mirror(a)ⱼ − mirror(μ)ⱼ)²/(2σ²) + const
                                = −Σⱼ(aₖ − μₖ)²/(2σ²) + const
                                = log_prob(a | μ)
```

Gaussian log_prob 在对称变换下严格不变。PPO 中 `old_log_prob_batch.repeat(num_aug)` 的近似实际上是精确的（当策略趋于对称时）。

## 二次镜像恒等性

对称变换是对合（involution）：

```
mirror(mirror(obs, actions)) = (obs, actions)
```

测试 `test_double_mirror_is_exact_identity` 验证 diff < 1e−5。逐项验证：

- 标量取反：两次 = 恒等
- DOF 位置 decode→swap→negate→encode：两次 = 恒等（默认值严格抵消）
- DOF 速度 swap→negate：两次 = 恒等
- LiDAR Y-flip→angular re-sort：两次 = 恒等（排序函数确定性的）

## 配置

```python
# go2_cmd_safe_config.py
class symmetry_cfg:
    use_data_augmentation = True    # 数据增强 (obs + actions)
    use_mirror_loss = True          # 镜像损失约束
    mirror_loss_coeff = 1.0         # 镜像损失权重
    data_augmentation_func = (
        "legged_gym.envs.go2.cmd_safe.go2_cmd_safe:get_go2_cmd_safe_xsym_obs_act"
    )
```

## 涉及文件

| 文件 | 职责 |
|------|------|
| `go2_cmd_safe.py:1098–1268` | `get_go2_cmd_safe_xsym_obs_act()` 对称函数 |
| `on_policy_runner.py:292–340` | `_setup_symmetry()` 参数绑定 |
| `ppo.py:258–402` | 数据增强 + mirror loss 调用 |
| `cmd_safe_history_wrapper.py` | wrapped_obs 构建（FPS + 下采样 + angular sort） |
| `pointcloud_geometry.py:58–70` | `sort_points_by_angular_key()` 共用排序函数 |
| `cmd_safe_actor_critic.py` | 双 GRU 编码，consumes 2736-dim obs |
| `go2_cmd_safe_config.py:285–291` | symmetry_cfg 配置 |
| `test_go2_xsym.py` | 对称函数回归测试 |

## 审计结论 (2026-07-01)

- 所有 48 维 proprioceptive 镜像变换均经物理和数学验证通过
- 默认值补偿在 `FL_default + FR_default = 0`, `RL_default + RR_default = 0` 前提下严格抵消
- LiDAR 处理与 CmdSafeHistoryWrapper 共用同一排序函数，流程一致
- 二次镜像恒等性测试通过 (diff < 1e−5)
- 无 bug 发现
