# v_avoid 对齐论文 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `_compute_v_avoid` 对齐论文公式——min() 取每扇区最近距离，全部危险扇区向量和，移除迭代取最大和 v_cmd 投影调制。

**Architecture:** 重写 `_compute_v_avoid` 为一个简单的向量化操作：36 扇区 × min() 距离 → 阈值激活 → exp 衰减 → 向量和。清理配置中的 `avoid_iters` 和 `avoid_gain`。

**Tech Stack:** PyTorch

---

### File Structure

| 文件 | 改动 |
|------|------|
| `go2_lidar_pd_risknet.py` | 重写 `_compute_v_avoid` |
| `go2_lidar_pillar_config.py` | n_sectors 24→36, threshold 1.6→1.0, 移除 avoid_iters/avoid_gain |
| `go2_lidar_pd_risknet_config.py` | 移除 avoid_iters/avoid_gain |
| `go2_pd_pretrain_config.py` | 移除 avoid_iters/avoid_gain |
| `test_go2_lidar_pd_risknet_math.py` | 新增 v_avoid 数学测试 |

---

### Task 1: 重写 `_compute_v_avoid`

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:312-369`

- [ ] **Step 1: 替换 `_compute_v_avoid` 方法**

将整个方法（lines 312-369）替换为：

```python
    def _compute_v_avoid(self):
        cfg = self.cfg.pd_risknet
        n_sec = int(cfg.n_sectors)
        sec_size = 2.0 * math.pi / n_sec

        pts = self.lidar_points_base[..., :2]
        dist = self.avoid_distances
        angles = torch.atan2(pts[..., 1], pts[..., 0])
        sec_ids = torch.floor((angles + math.pi) / sec_size).long().clamp(min=0, max=n_sec - 1)

        # Per-sector minimum distance. Non-sector points are set to a large
        # sentinel so they sort to the end, making min() correct.
        inf = torch.full_like(dist, 1.0e9)
        min_dist_per_sec = []
        for sec in range(n_sec):
            sec_vals = torch.where(sec_ids == sec, dist, inf)
            sec_min = torch.min(sec_vals, dim=1).values  # (num_envs,)
            min_dist_per_sec.append(sec_min)
        min_dist_per_sec = torch.stack(min_dist_per_sec, dim=1)  # (num_envs, n_sec)

        # Active sectors: d < threshold → exp(-d * alpha); inactive → 0
        active = min_dist_per_sec < float(cfg.avoid_distance_thresh)
        mag = torch.exp(-min_dist_per_sec * float(cfg.avoid_alpha)) * active.float()

        # Sector center directions; avoidance pushes AWAY from each sector
        sec_centers = torch.linspace(
            -math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size, n_sec, device=self.device
        )
        away_dirs = torch.stack(
            (-torch.cos(sec_centers), -torch.sin(sec_centers)), dim=-1
        ).unsqueeze(0)  # (1, n_sec, 2)

        # Vector sum over all sectors
        self.v_avoid = torch.sum(away_dirs * mag.unsqueeze(-1), dim=1)  # (num_envs, 2)
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: align _compute_v_avoid with paper — min() distance per sector, full vector sum"
```

---

### Task 2: 更新配置

**Files:**
- Modify: `go2_lidar_pillar_config.py`, `go2_lidar_pd_risknet_config.py`, `go2_pd_pretrain_config.py`

- [ ] **Step 1: pillar config — 对齐参数**

```python
# 改前 (go2_lidar_pillar_config.py pd_risknet section):
        n_sectors = 24
        avoid_distance_thresh = 1.6
        ...
        avoid_iters = 3
        avoid_gain = 1.1

# 改后:
        n_sectors = 36
        avoid_distance_thresh = 1.0
        ...
        (avoid_iters 和 avoid_gain 行删除)
```

- [ ] **Step 2: risknet config — 移除 avoid_iters/avoid_gain**

从 `go2_lidar_pd_risknet_config.py` 的 `pd_risknet` section 中删除:
```python
        avoid_iters = 3      # 迭代挑最大轮数
        avoid_gain = 1.1     # 避障速度增益
```

- [ ] **Step 3: pretrain config — 同上**

从 `go2_pd_pretrain_config.py` 的 `pd_risknet` section 中删除 `avoid_iters` 和 `avoid_gain`。

- [ ] **Step 4: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py \
       legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py \
       legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_pd_pretrain_config.py
git commit -m "config: align v_avoid params with paper — n_sectors=36, threshold=1.0, remove avoid_iters/gain"
```

---

### Task 3: 新增 v_avoid 数学测试

**Files:**
- Modify: `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py`

- [ ] **Step 1: 追加测试**

```python
def test_v_avoid_paper_formula():
    """Paper formula: V_j = exp(-d_j * alpha) * (-dir_j) if d_j < thresh."""
    import torch
    import math

    alpha = 1.5
    thresh = 1.0
    n_sec = 36
    sec_size = 2.0 * math.pi / n_sec

    # Simulate 2 envs: env0 has close obstacle NW, env1 is clear
    inf = torch.tensor(1e9)
    min_dist = torch.tensor([
        [inf, inf, 0.5, inf, inf] + [inf] * (n_sec - 5),   # env0: sector 2 at 0.5m
        [inf] * n_sec,                                        # env1: all clear
    ])

    active = min_dist < thresh
    mag = torch.exp(-min_dist * alpha) * active.float()
    sec_centers = torch.linspace(-math.pi + 0.5*sec_size, math.pi - 0.5*sec_size, n_sec)
    away_dirs = torch.stack((-torch.cos(sec_centers), -torch.sin(sec_centers)), dim=-1)

    v_avoid = torch.sum(away_dirs.unsqueeze(0) * mag.unsqueeze(-1), dim=1)

    # env1: zero avoidance
    assert torch.all(v_avoid[1] == 0.0)

    # env0: non-zero, pointing away from sector 2
    assert v_avoid[0].norm() > 0.0

    # Magnitude check: exp(-0.5 * 1.5) ≈ 0.4724
    expected_mag = math.exp(-0.5 * 1.5)
    assert abs(v_avoid[0].norm().item() - expected_mag) < 1e-4
```

- [ ] **Step 2: 运行测试**

```bash
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_v_avoid_paper_formula -v
```
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py
git commit -m "test: add v_avoid paper formula unit test"
```

---

### Task 4: 运行完整测试验证无回归

- [ ] **Step 1: 运行全部数学测试**

```bash
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v -k "not gate"
```
Expected: 6 passed (5 existing + 1 new)
