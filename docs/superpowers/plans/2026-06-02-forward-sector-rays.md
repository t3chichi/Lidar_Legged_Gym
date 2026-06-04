# Forward-Sector Rays 奖励实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `_reward_rays()` 从 36 扇区等权平均改为仅取机器人前方 12 扇区，y_progress 清零做消融实验。

**Architecture:** 两文件改动 — config 新增扇区参数，env 修改 `_reward_rays()` 末尾两行。

**Tech Stack:** Python / PyTorch

---

### Task 1: 配置新增前向扇区参数

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py`

- [ ] **Step 1: 在 pd_risknet 类新增两个参数**

在 `ray_max_distance = 10.0` 行之后插入：

```python
ray_forward_sector_count = 12     # rays 奖励使用的前方扇区数（扇区18=正前方，12扇区=±60°）
ray_forward_sector_center = 18    # 前方扇区中轴索引（传感器+X→机器人+X）
```

同时将 `y_progress` scale 从 10.0 清零：

```python
# 改前：
y_progress = 10.0  # 世界坐标系 Y 进度奖励，鼓励沿走廊持续前进

# 改后：
y_progress = 0.0   # 消融实验：清零，验证 forward-sector rays 方向梯度是否足够
```

---

### Task 2: 修改 _reward_rays() 只取前向扇区

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:562-563`

- [ ] **Step 1: 改 _reward_rays() 最后两行**

```python
# 改前：
        sector_mean = torch.stack(sector_means, dim=1)  # (N, 36)
        return sector_mean.mean(dim=1) / d_max

# 改后：
        sector_mean = torch.stack(sector_means, dim=1)  # (N, 36)
        n_fwd = int(self.cfg.pd_risknet.ray_forward_sector_count)
        center = int(self.cfg.pd_risknet.ray_forward_sector_center)
        start = center - n_fwd // 2
        end = start + n_fwd
        return sector_mean[:, start:end].mean(dim=1) / d_max
```

---

### Task 3: 运行测试验证

- [ ] **Step 1: 运行数学测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

- [ ] **Step 2: 运行环境测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_env.py -v
```

---

### Task 4: 提交

- [ ] **Step 1: 提交所有改动**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py \
        legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py \
        docs/superpowers/specs/2026-06-02-forward-sector-rays-design.md \
        docs/superpowers/plans/2026-06-02-forward-sector-rays.md
git commit -m "feat: forward-sector rays reward, replace 36-sector mean with front 12 sectors (±60°)

- Add ray_forward_sector_count=12, ray_forward_sector_center=18 to config
- _reward_rays() now averages only forward sectors instead of all 36
- y_progress scale set to 0 for ablation study
- Design doc: docs/superpowers/specs/2026-06-02-forward-sector-rays-design.md"
```
