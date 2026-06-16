# Channel Forward Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `_reward_channel_forward` 奖励，每步沿通道方向位置增量奖励前进、惩罚后退。

**Architecture:** 利用已有的 `_channel_forward` 方向向量和 `base_pos`，新增 `_last_channel_pos` 缓冲区存储上一步通道坐标，计算每步位置增量 delta。奖励函数返回原始值 `(forward - ratio × backward)`，外部由 `rewards.scales.channel_forward = 10.0` 控制权重——与 `_reward_vel_avoid`、`_reward_rays` 的 weight 管理模式一致。

**Tech Stack:** Python, PyTorch, Isaac Gym

---

### Task 1: 走廊配置新增参数 + curvature 清零

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py`

- [ ] **Step 1: pd_risknet 新增 `channel_backward_ratio`**

在 `class pd_risknet:` 中，`collision_3d = False` 行之前插入：

```python
            # channel_forward 沿通道方向后退惩罚倍率
            channel_backward_ratio = 0.5    # 后退惩罚相对于前进的倍率
```

- [ ] **Step 2: scales 新增 `channel_forward = 10.0`**

在 `curvature = -0.005` 行之后插入：

```python
            channel_forward = 10.0  # 沿通道方向前进/后退奖励
```

- [ ] **Step 3: curvature 清零**

将 `curvature = -0.005` 改为 `curvature = -0.0`。

- [ ] **Step 4: 验证配置**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg
cfg = Go2LidarPDRiskNetCfg()
print('channel_backward_ratio:', cfg.pd_risknet.channel_backward_ratio)
print('channel_forward:', cfg.rewards.scales.channel_forward)
print('curvature:', cfg.rewards.scales.curvature)
"
```
Expected: `channel_backward_ratio: 0.5`, `channel_forward: 10.0`, `curvature: -0.0`

- [ ] **Step 5: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "feat: add channel_forward reward + zero curvature for corridor config"
```

---

### Task 2: 梅花桩和预训练配置新增 pd_risknet 参数

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py`
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py`

- [ ] **Step 1: 梅花桩 pd_risknet 新增参数**

在 `go2_lidar_pillar_config.py` 的 `class pd_risknet:` 中，`collision_3d = False` 行之前插入：

```python
            # channel_forward 沿通道方向后退惩罚倍率
            channel_backward_ratio = 0.5    # 后退惩罚相对于前进的倍率
```

- [ ] **Step 2: 预训练 pd_risknet 新增参数**

在 `go2_pd_pretrain_config.py` 的 `class pd_risknet:` 中，`collision_3d = True` 行之前插入：

```python
            # channel_forward 沿通道方向后退惩罚倍率
            channel_backward_ratio = 0.5    # 后退惩罚相对于前进的倍率
```

- [ ] **Step 3: 确认两个配置均无 `channel_forward` scale（不定义 = 默认 0）**

```bash
grep "channel_forward" legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py
grep "channel_forward" legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py
```
Expected: 只输出 `channel_backward_ratio`（pd_risknet 参数），无 scales 中的 `channel_forward`。

- [ ] **Step 4: 验证配置加载**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pillar_config import Go2LidarPillarCfg
cfg = Go2LidarPillarCfg()
print('channel_backward_ratio:', cfg.pd_risknet.channel_backward_ratio)
print('has channel_forward scale:', hasattr(cfg.rewards.scales, 'channel_forward'))
"
```
Expected: `channel_backward_ratio: 0.5`, `has channel_forward scale: False`

- [ ] **Step 5: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py
git commit -m "feat: add channel_backward_ratio to pillar and pretrain configs"
```

---

### Task 3: 实现 `_reward_channel_forward` 方法

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: `_init_pd_risknet_buffers` 新增 `_last_channel_pos`**

在 `self.last_dist = torch.zeros(...)` 之后插入：

```python
        self._last_channel_pos = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
```

- [ ] **Step 2: `reset_idx` 初始化 `_last_channel_pos`**

找到 `self.last_dist[env_ids] = torch.norm(...)` 这行。在该行之后插入：

```python
        self._last_channel_pos[env_ids] = torch.sum(
            self.base_pos[env_ids, :2] * self._channel_forward[env_ids], dim=1)
```

- [ ] **Step 3: 新增 `_reward_channel_forward` 方法**

在 `_reward_move_distance` 方法之前插入：

```python
    def _reward_channel_forward(self):
        cfg = self.cfg.rewards
        if getattr(cfg.scales, "channel_forward", 0.0) == 0.0:
            return torch.zeros(self.num_envs, device=self.device)
        p_cfg = self.cfg.pd_risknet
        ratio = float(getattr(p_cfg, "channel_backward_ratio", 0.5))
        channel_pos = torch.sum(self.base_pos[:, :2] * self._channel_forward, dim=1)
        delta = channel_pos - self._last_channel_pos
        self._last_channel_pos[:] = channel_pos
        forward  = torch.clamp(delta, min=0.0)
        backward = torch.clamp(-delta, min=0.0)
        return forward - ratio * backward
```

- [ ] **Step 4: 语法检查**

```bash
python -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: add _reward_channel_forward with per-step forward/backward reward"
```

---

### Task 4: 集成验证

- [ ] **Step 1: 语法验证**

```bash
conda run -n li_leggym python -c "
import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('Syntax OK')
"
```
Expected: `Syntax OK`
