# Rays 平滑方向与奖励解耦

## 目标

将 rays 开阔方向 EMA 更新从 `_reward_rays` 中提取为独立方法，在 `_post_physics_step_callback` 中每步调用，使可视化箭头在 rays 奖励为 0 时也能正常更新。

## 动机

当前 `_smooth_dir_world` 只在 `_reward_rays` 内更新，而该方法在 rays 奖励 scale=0 时完全不被调用（`_prepare_reward_function` 删除了零 scale 项）。导致预训练等场景中方向箭头冻结。

## 设计

### 新增方法

```python
def _update_smooth_rays_dir(self):
    """Update EMA-smoothed open-space direction (called every step).

    Mirrors _compute_v_avoid(): cache computation happens here,
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

### 调用位置

`_post_physics_step_callback` 中，在 `_compute_v_avoid()` 后添加一行：

```python
def _post_physics_step_callback(self):
    super()._post_physics_step_callback()
    self._update_lidar_history()
    self._compute_v_avoid()
    self._update_smooth_rays_dir()      # ← 新增
```

调用顺序保证了 `_raw_distances` 已由 `_update_lidar_history` 填充。

### 简化的 `_reward_rays`

删除其中的 target_dir 计算和 EMA 更新，改为直接读缓存：

```python
def _reward_rays(self):
    # 缓存已由 _update_smooth_rays_dir 更新（每步，在 callback 中）
    omega_target = self._compute_rays_omega_target()  # reads _smooth_dir_world
    omega_actual = self.base_ang_vel[:, 2]
    omega_err = omega_actual - omega_target
    sigma = float(getattr(self.cfg.pd_risknet, "rays_omega_sigma", 0.25))
    return torch.exp(-omega_err * omega_err / sigma)
```

### Reset 路径不变

Reset 中保持直接赋值 `_smooth_dir_world[env_ids] = target_dir_world[env_ids]`，新方法在此路径中不会被额外调用（reset 不走 `_post_physics_step_callback` 的正常流程）。

## 等价性

| | 旧 (rays≠0) | 新 (rays≠0) | 新 (rays=0) |
|---|---|---|---|
| EMA 更新 | `_reward_rays` 内 | `_post_physics_step_callback` 内 | `_post_physics_step_callback` 内 |
| 更新时机 | compute_reward 阶段 | 同一步，提前于 compute_reward | 同一步 |
| `_reward_rays` 行为 | 计算 + EMA + 奖励 | 仅计算奖励（读缓存） | 不调用 |
| 计算结果 | — | **完全等价** | 新增功能 |

对于 rays≠0 的情况，数学上完全等价——同一组操作、同样的顺序、同样的数据。

## 性能

每步额外开销：36 扇区 × 每扇区 top-k 排序 + 加权求和 ≈ **5-15ms** GPU。对比 90s/轮训练时间可忽略。

## 不改的部分

- `_compute_rays_target_dir()` 逻辑
- `_compute_rays_omega_target()` 逻辑
- `_smooth_dir_world` 缓冲区和初始化
- Reset 初始化逻辑
- 可视化代码
- 所有配置
