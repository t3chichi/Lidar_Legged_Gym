# 曲率惩罚奖励实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Go2LidarPDRiskNet 中添加曲率惩罚奖励，惩罚原地转圈行为。

**Architecture:** 在 `Go2LidarPDRiskNet` 类中添加 `_reward_curvature()` 方法，基类的 `compute_reward()` 通过 `_prepare_reward_function()` 自动发现并调用它。配置中注册 `curvature = -0.05` 的奖励缩放。

**Tech Stack:** Python, PyTorch

---

### Task 1: 添加曲率惩罚奖励函数

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py`

- [ ] **Step 1: 添加 _reward_curvature 方法到 Go2LidarPDRiskNet**

在 `go2_lidar_pd_risknet.py` 的 `Go2LidarPDRiskNet` 类中，在 `_reward_ang_vel_yaw_penalty` 方法之后添加：

```python
def _reward_curvature(self):
    """惩罚瞬时路径曲率平方，抑制原地转圈行为。
    
    r = -lambda * omega_z^2 / (v_xy^2 + sigma^2)
    轨迹曲率 kappa = |omega_z| / v_xy，该项 = -lambda * kappa^2。
    sigma^2 软化项防止零线速度时惩罚爆炸。
    """
    v_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
    omega_z = self.base_ang_vel[:, 2]
    return omega_z.square() / (v_xy.square() + 0.49)
```

- [ ] **Step 2: 在配置中添加 curvature 奖励缩放**

在 `go2_lidar_pd_risknet_config.py` 的 `rewards.scales` 类中，`ang_vel_yaw_penalty` 注释行之后添加：

```python
curvature = -0.05  # 曲率惩罚：抑制 ω_z²/(v_xy²+σ²)，防止原地转圈
```

- [ ] **Step 3: 验证奖励函数被自动发现**

运行以下命令确认 `_reward_curvature` 能被 `_prepare_reward_function` 正确识别：

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg
cfg = Go2LidarPDRiskNetCfg()
print('curvature' in cfg.rewards.scales)
print('curvature scale:', cfg.rewards.scales.get('curvature', 'NOT FOUND'))
"
```

预期输出：
```
True
curvature scale: -0.00025
```

（scale 在 `_prepare_reward_function` 中会乘以 dt=0.005，所以实际值为 -0.05 × 0.005 = -0.00025）

- [ ] **Step 4: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "feat: add curvature penalty reward to suppress spinning-in-place behavior"
```
