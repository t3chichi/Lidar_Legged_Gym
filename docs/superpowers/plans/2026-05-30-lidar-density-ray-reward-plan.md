# LiDAR 点云密度提升与 Ray 奖励改造 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 LiDAR 仿真密度对齐 Mid-360 真实参数（80×50），提升采样点数（512/256），重写 ray 奖励为扇区 top-p% 形式。

**Architecture:** 修改三个 config 文件的共享常量（密度/采样/距离），重写 `Go2LidarPDRiskNet._reward_rays` 为 d_max=50m + 滤除天空 + 36 扇区 top-25% 均值的等权平均。网络输入保持原样不滤除。`_init_pd_risknet_buffers` 新增扇区归属预计算。

**Tech Stack:** PyTorch, Isaac Gym, NVIDIA Warp

---

### Task 1: 修改走廊 config 参数 (`go2_lidar_pd_risknet_config.py`)

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py`

- [ ] **Step 1: 修改模块级常量**

```python
# 第 7-9 行，修改:
PD_SPHERICAL_AZIMUTH = 80
PD_SPHERICAL_ELEVATION = 50
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION  # 4000

# 第 11-12 行，修改:
PD_PROXIMAL_POINTS = 512
PD_DISTAL_POINTS = 256
```

- [ ] **Step 2: 修改 pd_risknet.ray_max_distance**

```python
# 第 56 行，修改:
ray_max_distance = 50.0  # rays 奖励截断距离 (m)
```

- [ ] **Step 3: 修改 raycaster.max_distance**

```python
# 第 152 行，max_distance 从 10.0 改为 50.0
max_distance = 50.0
```

- [ ] **Step 4: 验证 num_observations 自动推导正确**

确认 `env.num_observations = PD_PROPRIO_DIM + OBS_HISTORY_LENGTH * PD_NUM_LIDAR_POINTS * 3` 不需要手动改（引用常量，自动更新到 49 + 1×4000×3 = 12049）。

- [ ] **Step 5: 验证 policy 参数自动推导正确**

确认 `num_lidar_points`, `proximal_points`, `distal_points` 均引用模块常量，无需手动修改。

- [ ] **Step 6: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "feat: upgrade corridor lidar density to 80x50, sampling to 512/256, d_max to 50m"
```

---

### Task 2: 同步梅花桩 config 参数 (`go2_lidar_pillar_config.py`)

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py`

- [ ] **Step 1: 修改模块级常量**

```python
# 第 7-9 行，修改:
PD_SPHERICAL_AZIMUTH = 80
PD_SPHERICAL_ELEVATION = 50
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION

# 第 11-12 行，修改:
PD_PROXIMAL_POINTS = 512
PD_DISTAL_POINTS = 256
```

- [ ] **Step 2: 修改 ray_max_distance**

```python
# 第 55 行，修改:
ray_max_distance = 50.0
```

- [ ] **Step 3: 修改 raycaster.max_distance**

```python
# 约第 142 行，max_distance 从 10.0 改为 50.0
max_distance = 50.0
```

- [ ] **Step 4: 修改 raycaster 中的 spherical_num_***

确认 `spherical_num_azimuth = PD_SPHERICAL_AZIMUTH` 和 `spherical_num_elevation = PD_SPHERICAL_ELEVATION` 引用模块常量，无需手动修改。

- [ ] **Step 5: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py
git commit -m "feat: upgrade pillar lidar density to 80x50, sampling to 512/256, d_max to 50m"
```

---

### Task 3: 同步预训练 config 参数 (`go2_pd_pretrain_config.py`)

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py`

- [ ] **Step 1: 修改模块级常量**

```python
# 第 7-9 行，修改:
PD_SPHERICAL_AZIMUTH = 80
PD_SPHERICAL_ELEVATION = 50
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION

# 第 11-12 行，修改:
PD_PROXIMAL_POINTS = 512
PD_DISTAL_POINTS = 256
```

- [ ] **Step 2: 修改 ray_max_distance**

```python
# 第 53 行，修改:
ray_max_distance = 50.0
```

- [ ] **Step 3: 修改 raycaster.max_distance**

```python
# 约第 110 行，max_distance 从 10.0 改为 50.0
max_distance = 50.0
```

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py
git commit -m "feat: upgrade pretrain lidar density to 80x50, sampling to 512/256, d_max to 50m"
```

---

### Task 4: 新增扇区预计算 (`_init_pd_risknet_buffers`)

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py`

- [ ] **Step 1: 在 `_init_pd_risknet_buffers` 末尾新增扇区归属张量**

在 `self._ray_dirs_sensor = ...` (line 156) 之后添加:

```python
        # Precompute which 10° sector each distal ray belongs to.
        # Ray directions are in sensor frame; compute azimuth angle.
        ray_azimuth = torch.atan2(
            self._ray_dirs_sensor[:, 1],
            self._ray_dirs_sensor[:, 0],
        )  # (-π, π]
        ray_azimuth_0_2pi = ray_azimuth + math.pi  # [0, 2π)
        sector_size = 2.0 * math.pi / 36.0
        self._distal_ray_sector_ids = torch.floor(
            ray_azimuth_0_2pi[self._distal_mask] / sector_size
        ).long().clamp(min=0, max=35)  # (num_distal_raw,)
```

- [ ] **Step 2: 确认所有引用均有效**

- `self._distal_mask` 已在前面的代码中定义（line 137）
- `self._ray_dirs_sensor` 已在前面的代码中定义（line 156）
- `ray_azimuth_0_2pi` 在 `[0, 2π)` 范围内，除以 `sector_size` 后 clamp 到 `[0, 35]`

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: precompute distal ray sector ids for ray reward"
```

---

### Task 5: 重写 `_reward_rays`

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:544-548`

- [ ] **Step 1: 替换 `_reward_rays` 方法**

```python
    def _reward_rays(self):
        d_max = float(self.cfg.pd_risknet.ray_max_distance)
        dist_all = self.raycast_distances[:, self._distal_mask]  # (N, num_distal_raw)
        valid = dist_all < (d_max - 0.001)  # exclude sky / no-hit rays at d_max
        dist = torch.where(valid, dist_all, torch.zeros_like(dist_all))

        n_sectors = 36
        top_ratio = 0.25

        sector_means = []
        for s in range(n_sectors):
            s_mask = self._distal_ray_sector_ids == s  # (num_distal_raw,)
            s_dist = dist[:, s_mask]                    # (N, rays_in_sector)
            s_valid = valid[:, s_mask]

            # Number of valid rays in this sector per env
            n_valid = s_valid.sum(dim=1, keepdim=True).clamp(min=1).float()  # (N, 1)
            k = torch.clamp((n_valid * top_ratio).long(), min=1)           # (N, 1)
            k_max = int(k.max().item())

            top_vals, _ = torch.topk(s_dist, k=k_max, dim=1)  # (N, k_max)

            # Mask out entries beyond per-env k (set to 0, they're sorted so these are the smaller values)
            idx = torch.arange(k_max, device=s_dist.device).unsqueeze(0).expand_as(top_vals)
            keep = idx < k
            top_sum = (top_vals * keep.float()).sum(dim=1)  # (N,)
            sector_mean = top_sum / k.squeeze(1)            # (N,)
            sector_means.append(sector_mean)

        sector_mean = torch.stack(sector_means, dim=1)  # (N, 36)
        return sector_mean.mean(dim=1) / d_max
```

- [ ] **Step 2: 运行语法检查**

```bash
python -c "import ast; ast.parse(open('legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: rewrite ray reward to sector top-25% with sky filtering at d_max=50m"
```

---

### Task 6: 运行环境初始化冒烟测试

**Files:**
- Test: `legged_gym/legged_gym/tests/test_env.py`

- [ ] **Step 1: 运行环境创建测试**

```bash
cd /home/t3chichi/Lidar_legged_gym
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg, Go2LidarPDRiskNetCfgPPO
from legged_gym.utils import task_registry
# Verify config loads
cfg = task_registry.get_cfgs('go2_lidar_pd_risknet')
print(f'num_observations: {cfg.env.num_observations}')
print(f'num_lidar_points: {cfg.pd_risknet.num_lidar_points}')
print(f'proximal_points: {cfg.pd_risknet.proximal_points}')
print(f'distal_points: {cfg.pd_risknet.distal_points}')
print(f'ray_max_distance: {cfg.pd_risknet.ray_max_distance}')
print(f'raycaster.max_distance: {cfg.raycaster.max_distance}')
assert cfg.pd_risknet.num_lidar_points == 4000
assert cfg.pd_risknet.proximal_points == 512
assert cfg.pd_risknet.distal_points == 256
assert cfg.pd_risknet.ray_max_distance == 50.0
print('Config OK')
"
```

- [ ] **Step 2: 验证预训练和梅花桩 config 也能加载**

```bash
python -c "
from legged_gym.utils import task_registry
for task in ['go2_pd_pretrain', 'go2_lidar_pillar']:
    cfg = task_registry.get_cfgs(task)
    assert cfg.pd_risknet.num_lidar_points == 4000, f'{task} num_lidar_points mismatch'
    assert cfg.pd_risknet.proximal_points == 512, f'{task} proximal_points mismatch'
    assert cfg.pd_risknet.distal_points == 256, f'{task} distal_points mismatch'
    print(f'{task} config OK')
"
```

- [ ] **Step 3: Commit (如有小修正)**

---

### Task 7: 最终验证

- [ ] **Step 1: 运行现有测试套件**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
python -m pytest legged_gym/legged_gym/tests/test_env.py -v
```

- [ ] **Step 2: 检查 log 中的 DEBUG 输出**

如果启动环境，检查 theta distribution 日志输出确认新分辨率下的远端/近端点数量合理：

```
预期:
  [DEBUG] num proximal (theta >= 20.0°): ~2480
  [DEBUG] num distal (theta < 20.0°): ~1520
```

- [ ] **Step 3: Commit 验证结果**
