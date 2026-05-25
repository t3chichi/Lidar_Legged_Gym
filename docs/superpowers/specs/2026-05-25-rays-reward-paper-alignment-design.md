# r_rays 距离最大化奖励对齐论文设计

## 目标

将 `_reward_rays` 实现与 Omni-Perception 论文公式对齐：

\[
r_{rays} = \sum_{i=1}^{n} \frac{\min(d_{t,i}, d_{max})}{n \cdot d_{max}}
\]

鼓励策略通过最大化截断的远端 LiDAR 射线距离来维持安全裕度并推向开阔区域。

## 当前实现 vs 论文

| 维度 | 当前实现 | 论文 | 改动 |
|------|---------|------|------|
| 选点范围 | 全部 LiDAR 点（仅地面滤除） | 远端射线 (theta < split_theta_deg) | 限定远端 |
| 聚合方式 | closest 50% (top-k) | 全部 n 条射线均值 | 取消 top-k |
| 数据源 | `avoid_distances` (地面滤除, 无域随机化) | 原始 LiDAR 距离 | 改用 `raycast_distances` |
| 地面滤除 | 有 | 不可用 | 不做地面滤除 |
| d_max | 6.0m | 未指定 | 对齐物理探测距离 10m |

## 设计决策

1. **不做地面滤除** — 远处地面点同样指示"该方向开阔"，与障碍物点一样对 r_rays 有正向贡献。r_rays 衡量开阔性，不是障碍物检测。
2. **全部远端射线取均值** — "代表性远端射线"指低俯角射线这一类别本身，不是从远端中再选子集。
3. **d_max = 10m** — 与 `raycaster.max_distance` 对齐，反映物理传感器探测范围。

## 实现方案

### 1. 预计算远端掩码 (`_init_pd_risknet_buffers`)

利用球形网格传感器固定仰角线，在初始化时一次性计算 sensor frame 下每条射线的 theta，与 `split_theta_deg`（20°）比较：

- elevation 范围：-2° ~ 57°，18 条线
- theta < 20° → distal (True)
- theta >= 20° → proximal (False)

结果存入 `self._distal_mask: Tensor[bool, (432,)]`，后续直接索引。

### 2. 重写 `_reward_rays`

```python
def _reward_rays(self):
    d_max = float(self.cfg.pd_risknet.ray_max_distance)
    dist = self.raycast_distances[:, self._distal_mask]
    clipped = torch.clamp(dist, max=d_max)
    return torch.mean(clipped / d_max, dim=1)
```

### 3. 配置改动

`go2_lidar_pillar_config.py` — `pd_risknet.ray_max_distance`: 6.0 → 10.0

## 不改动范围

- `_compute_v_avoid` 及相关逻辑保持不变
- `_reward_vel_avoid` 保持不变
- 其他奖励项不变
- 网络结构、观测空间不变
