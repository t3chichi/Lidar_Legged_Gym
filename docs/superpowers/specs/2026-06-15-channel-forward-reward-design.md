# channel_forward 奖励设计

日期: 2026-06-15 | 状态: 已确认

## 概述

新增 `_reward_channel_forward`：每步沿通道方向的位置增量奖励，前进正反馈、后退惩罚。
目标：打破圆周运动的奖励自洽性，提供每步即时的前进引力。

## 背景

训练 1000+ 轮后出现侧身圆周运动：贴墙绕圈，vel_avoid + rays 均为高分，goal 延迟折现无法打破。
现有的即时奖励 (vel_avoid, rays, collision) 均不区分"往前走"和"绕圈"——后者同样是合速度跟踪 + 面朝开阔处。
需要一个与通道方向绑定的即时前进信号。

## 算法

```python
def _reward_channel_forward(self):
    # 当 weight=0 时跳过计算
    cfg = self.cfg.pd_risknet
    if getattr(cfg, "channel_backward_ratio", 0.5) <= 0 and getattr(self.cfg, "channel_forward", 0.0) == 0.0:
        return zeros

    channel_pos = torch.sum(self.base_pos[:, :2] * self._channel_forward, dim=1)
    delta = channel_pos - self._last_channel_pos
    self._last_channel_pos[:] = channel_pos

    forward  = torch.clamp(delta, min=0.0)
    backward = torch.clamp(-delta, min=0.0)
    return forward - channel_backward_ratio * backward
```

奖励函数返回原始值，外部通过 `rewards.scales.channel_forward` 控制权重。
与 `_reward_vel_avoid`、`_reward_rays` 的 weight 管理模式一致。

## 配置参数

```python
class pd_risknet:
    channel_backward_ratio = 0.5    # 后退惩罚相对于前进的倍率

class rewards.scales:
    channel_forward = 10.0          # 走廊启用，外部 weight
```

| 配置 | pd_risknet.channel_backward_ratio | scales.channel_forward |
|------|:---:|:---:|
| 走廊 | 0.5 | 10.0 |
| 梅花桩 | 0.5 | 0 (未定义，默认 0) |
| 预训练 | 0.5 | 0 (未定义，默认 0) |

## 新增状态

- `_last_channel_pos` (N,): 上一步沿通道方向坐标，reset_idx 时初始化

## 影响范围

| 文件 | 改动 |
|------|------|
| `go2_lidar_pd_risknet.py` | `_init_pd_risknet_buffers` 新增 `_last_channel_pos`；`reset_idx` 初始化；新增 `_reward_channel_forward` 方法 |
| `go2_lidar_pd_risknet_config.py` | pd_risknet 新增 `channel_backward_ratio`；scales 新增 `channel_forward=10.0`；curvature 清零 |
| `go2_lidar_pillar_config.py` | pd_risknet 新增 `channel_backward_ratio` |
| `go2_pd_pretrain_config.py` | pd_risknet 新增 `channel_backward_ratio` |
