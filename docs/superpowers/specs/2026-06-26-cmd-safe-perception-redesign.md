# CmdSafe 感知架构重构

## 目标

将 `go2_cmd_safe` 训练的 PD-RiskNet 感知架构改造为符合论文设计：近端/远端均使用零初始化 GRU，远端采用 10 帧显式滚动窗口代替跨步隐藏状态。仅影响 `go2_cmd_safe` 任务。

## 新建文件

### `rsl_rl/rsl_rl/modules/cmd_safe_actor_critic.py`

`CmdSafeActorCritic` — 新网络模块。

```python
class CmdSafeActorCritic(nn.Module):
    is_recurrent = False

    # ── 子模块 ──
    proximal_gru:       GRU(input_size=3, hidden_size=187, batch_first=True)
    distal_gru:         GRU(input_size=3, hidden_size=64,  batch_first=True)
    height_supervisor:  Linear(187, 187)
    actor:              Sequential(299→1024→512→256→128→12)
    critic:             Sequential(299→512→256→128→1)

    # ── 缓存 ──
    _cached_proximal_feature  # (B, 187) 近端 GRU 最终隐藏状态
```

**forward 流程：**

1. 拆分观测 → `proprio (B,48)`, `proximal (B,256,3)`, `distal (B,1280,3)`
2. 近端：球坐标排序 (256) → `proximal_gru` (zero-init) → 取 `h_n` → `(B, 187)`，缓存到 `_cached_proximal_feature`
3. 远端：`distal_gru` (zero-init, seq_len=1280) → 取 `h_n` → `(B, 64)`
4. 合并 → Actor / Critic MLP

**辅助损失 (`get_auxiliary_loss`)：**

```python
pred = self.height_supervisor(self._cached_proximal_feature)  # (B, 187)
target = privileged_heights[..., -187:]                        # measured_heights
loss = privileged_supervision_coef * MSE(pred, target)
```

- `pred[i]` 通过索引 `i` 与 `target[i]` (即 `measured_heights[i]`) 一一对应
- `measured_heights` 的 flatten 顺序固定 (row-major over 17×11 grid)
- 梯度从 RL loss + aux loss 双路回传到 `proximal_gru`

**分块处理：** 每 128 env 一块处理，和旧网络一致。

**无 PointNet、无 `distal_gru_hidden` 跨步状态、无 `_build_sampling_plan`。**

---

### `legged_gym/legged_gym/utils/cmd_safe_history_wrapper.py`

`CmdSafeHistoryWrapper` — 历史帧维护与点云预处理。

```python
class CmdSafeHistoryWrapper:
    def __init__(self, num_envs, cfg, device):
        # 远端滚动窗口: (num_envs, 10, 128, 3)
        # phi_threshold = cfg.pd_risknet.split_theta_deg  # 默认 20°

    def wrap_obs(self, obs_buf, lidar_points_base, dones):
        # 1. 拆分 proprio(48) + lidar_points (N,3)
        # 2. 转 sensor frame → cart_to_sphere → [r, azimuth, phi]
        # 3. 近端 (phi <= threshold):
        #      torch_fpsample FPS → 256点 → 球坐标排序 → (B,256,3)
        # 4. 远端 (phi > threshold):
        #      组合键排序 (phi*2π + azimuth) → 均匀采样128 → 球坐标排序
        #      → push 滚动窗口
        #      → 不满10帧: 广播首帧填充
        #      → 取出1280点 → 全局组合键排序 → (B,1280,3)
        # 5. dones → 清零对应 env 的远端窗口
        # 6. 返回 wrapped_obs (B, 48+256×3+1280×3) = (B, 4656)

    def reset(self, env_ids):
        # 远端窗口清零

    def _cart_to_sphere(self, points_sensor):
        # → [r, azimuth, phi]
```

**远端完整处理链：** 每帧候选点 → 组合键排序 → 均匀采样128 → push窗口 → 10帧拼接1280 → 全局组合键排序 → 输出

**`torch_fpsample.sample(padded, k)` 使用 `proximal_points_fps` 参考实现的批量化逻辑**（pad → 首点填充 → FPS → mask padding 采样点）。

**输出透传 `privileged_obs_buf`，不修改。**

---

## 修改文件

| 文件 | 改动 |
|------|------|
| `go2_cmd_safe_config.py` | `num_observations` 4656; `policy_class_name` `"CmdSafeActorCritic"` |
| `rsl_rl/rsl_rl/modules/__init__.py` | 导出 `CmdSafeActorCritic` |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` | env 创建后包裹 `CmdSafeHistoryWrapper`; `is_recurrent=False` 路径 |
| `legged_gym/setup.py` | 添加 `torch_fpsample` 依赖 |

## 数据流

```
Go2CmdSafe 环境
  obs_buf:                48 + N×3  (单帧原始点云 sensor frame)
  privileged_obs_buf:     48 + 187  (特权高度, 透传)
      │
      ▼
CmdSafeHistoryWrapper
  cart_to_sphere → phi 分离
      │
  ├─ 近端(phi≤20°): torch_fpsample → 256 → 球排序 → (B,256,3)
  │
  └─ 远端(phi>20°): 组合键排序 → 均匀128 → push 10帧窗口
                    广播首帧填充 → 全局组合键排序 → (B,1280,3)
      │
  wrapped_obs (B, 4656)
      │
      ▼
CmdSafeActorCritic
  近端 (B,256,3)   → 球排序 → GRU(3→187) → 187 ─┬─→ Actor [1024,512,256,128] → 12
                                          └→ Linear(187,187) → aux_loss
  远端 (B,1280,3)  → GRU(3→64)             → 64  ─┤
  proprio (B,48)                           → 48  ─┘  → Critic [512,256,128] → 1
```

## 与旧 PDRiskNetActorCritic 对照

| 项目 | 旧 | 新 |
|------|-----|-----|
| PointNet (近端/远端) | 各一个 MLP 3→16→32→64 | **无** |
| GRU input_size | 64 | **3** |
| 远端状态管理 | `distal_gru_hidden` 跨步 | **零初始化，每帧独立** |
| 远端输入 | 单帧128点，逐帧送入 | **10帧拼接1280点，一次送入** |
| 历史维护 | 网络内 GRU 隐藏状态 | **Wrapper 滚动窗口** |
| FPS 采样 | 首帧静态索引缓存 | **torch_fpsample 每帧动态批处理** |
| height_supervisor | Linear(187,187) | **Linear(187,187) 不变** |
| is_recurrent | True | **False** |
| 远端帧间排序 | 每帧独立排序 | **10帧拼接后全局排序** |
| 球坐标体系 | base frame, theta=atan2(z,√(x²+y²)) | **sensor frame, phi=asin(z/r)** |
| 近远端分离 | theta >= split_theta | **phi <= phi_threshold (config 默认20°)** |

## 维度表

| 组件 | 输入 | 输出 |
|------|------|------|
| Wrapper 输出 | — | (B, 4656) = 48 + 256×3 + 1280×3 |
| `proximal_gru` | (B, 256, 3) zero-init | h_n (1,B,187) |
| `distal_gru` | (B, 1280, 3) zero-init | h_n (1,B,64) |
| `height_supervisor` | (B, 187) | (B, 187) |
| `actor` | (B, 299) | (B, 12) |
| `critic` | (B, 299) | (B, 1) |
