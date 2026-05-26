# vel_avoid 引导式避障奖励 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `_compute_v_avoid` 从激进推开式改为引导式，用指令方向投影区分威胁优先级，消除走廊场景的过度振荡。

**Architecture:** 重写 `_compute_v_avoid` 方法核心计算逻辑，保留 36 扇区的最小距离提取和向量求和框架，替换幅值计算部分。`_reward_vel_avoid` 和可视化代码零改动。

**Tech Stack:** PyTorch, 现有 legged_gym 基础设施

---

### Task 1: 添加 `avoid_c` 配置参数

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py:45-48`

- [ ] **Step 1: 新增 `avoid_c` 参数**

在 `class pd_risknet` 中 `avoid_beta` 之后新增一行：

```python
avoid_c = 0.15
```

当前上下文（约第 45-48 行）：
```python
            avoid_distance_thresh = 1.5
            avoid_alpha = 1.0
            avoid_beta = 1.0
            ray_max_distance = 10.0  # rays 奖励截断距离 (m)
```

修改为：
```python
            avoid_distance_thresh = 1.5
            avoid_alpha = 1.0
            avoid_beta = 1.0
            avoid_c = 0.15   # 侧面兜底推力比例，占指令速度的 15%
            ray_max_distance = 10.0  # rays 奖励截断距离 (m)
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py
git commit -m "feat: add avoid_c config parameter for guided avoidance"
```

---

### Task 2: 重写 `_compute_v_avoid` 方法

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py:312-345`

- [ ] **Step 1: 替换方法实现**

将现有 `_compute_v_avoid` 方法（第 312-345 行）替换为：

```python
    def _compute_v_avoid(self):
        cfg = self.cfg.pd_risknet
        n_sec = int(cfg.n_sectors)
        sec_size = 2.0 * math.pi / n_sec

        pts = self.lidar_points_base[..., :2]
        dist = self.avoid_distances
        angles = torch.atan2(pts[..., 1], pts[..., 0])
        sec_ids = torch.floor((angles + math.pi) / sec_size).long().clamp(min=0, max=n_sec - 1)

        # Per-sector minimum distance (unchanged).
        inf = torch.full_like(dist, 1.0e9)
        min_dist_per_sec = []
        for sec in range(n_sec):
            sec_vals = torch.where(sec_ids == sec, dist, inf)
            sec_min = torch.min(sec_vals, dim=1).values
            min_dist_per_sec.append(sec_min)
        min_dist_per_sec = torch.stack(min_dist_per_sec, dim=1)  # (num_envs, n_sec)

        # Sector center directions pointing AWAY from each sector.
        sec_centers = torch.linspace(
            -math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size, n_sec, device=self.device
        )
        away_dirs = torch.stack(
            (-torch.cos(sec_centers), -torch.sin(sec_centers)), dim=-1
        )  # (n_sec, 2)

        # Command direction projection per sector: cos_i = max(0, v_cmd_dir · u_i).
        v_cmd = self.commands[:, :2]                                              # (num_envs, 2)
        v_cmd_norm = torch.norm(v_cmd, dim=1, keepdim=True)                       # (num_envs, 1)
        # Guard against zero-length commands (stationary → no avoidance needed).
        nonzero = (v_cmd_norm.squeeze(1) > 1e-6)                                  # (num_envs,)

        if nonzero.any():
            v_cmd_dir = v_cmd[nonzero] / v_cmd_norm[nonzero]                      # (N, 2)
            # Sector center directions (inward, same as LiDAR sectors): u_i = -away_dirs.
            u = -away_dirs                                                        # (n_sec, 2)
            cos = torch.relu(torch.mm(v_cmd_dir, u.T))                            # (N, n_sec), [0, 1]

            # Weighted urgency: (cos_i + c) × exp(-alpha × d_i) × within_d_max.
            d = min_dist_per_sec[nonzero]                                          # (N, n_sec)
            d_max = float(cfg.avoid_distance_thresh)
            c_val = float(getattr(cfg, "avoid_c", 0.15))
            alpha = float(cfg.avoid_alpha)
            w = (cos + c_val) * torch.exp(-alpha * d) * (d < d_max).float()      # (N, n_sec)

            # Weighted sum: v_avoid = ||v_cmd|| × Σ w_i × (-u_i).
            v_avoid_nonzero = v_cmd_norm[nonzero] * (w @ away_dirs)               # (N, 2)

            self.v_avoid[nonzero] = v_avoid_nonzero

        # Stationary envs: v_avoid = 0.
        self.v_avoid[~nonzero] = 0.0
```

- [ ] **Step 2: 运行现有测试确认不破坏已有功能**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

预期：所有 7 个现有测试通过。其中 `test_v_avoid_paper_formula` 测试了旧公式，需要更新（见 Task 3）。

- [ ] **Step 3: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet.py
git commit -m "feat: rewrite v_avoid as guided avoidance with command projection

Replace push-away mechanism: each sector's contribution is now weighted
by command direction projection (cos_i + c), preventing lateral walls
from producing excessive opposing forces in corridors."
```

---

### Task 3: 更新测试文件

**Files:**
- Modify: `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py`

- [ ] **Step 1: 替换 `test_v_avoid_paper_formula` 为新的引导式公式测试**

将第 277-309 行的旧测试 `test_v_avoid_paper_formula` 替换为以下新测试。

**注意：** 36 扇区编号从 `-175°` 到 `+175°`，扇区 18 约在 `5°`（近似正前方），扇区 9 约在 `-85°`（近似左侧）。测试中使用这些正确的扇区编号。

```python
def test_v_avoid_guided_formula():
    """Guided formula: w_i = (cos_i + c) * exp(-alpha * d_i) * (d_i < d_max),
       v_avoid = ||v_cmd|| * sum(w_i * (-u_i))."""
    import torch
    import math

    c_val = 0.15
    alpha = 1.0
    d_max = 1.5
    n_sec = 36
    sec_size = 2.0 * math.pi / n_sec

    inf = torch.tensor(1e9)

    # --- Case 1: stationary command → v_avoid = 0 ---
    v_cmd_0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    # --- Case 2: forward command, obstacle ahead (cos ≈ 1) ---
    v_cmd_1 = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    v_cmd_norm_1 = torch.norm(v_cmd_1, dim=1, keepdim=True)
    v_cmd_dir_1 = v_cmd_1 / v_cmd_norm_1

    # Obstacle at sector 18 (≈5°, approximately forward), d=0.5m. Other sectors clear.
    d_1 = torch.full((1, n_sec), inf)
    d_1[0, 18] = 0.5  # forward obstacle

    # Compute sector center directions (inward u_i) and away_dirs.
    sec_centers = torch.linspace(-math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size, n_sec)
    u = torch.stack((torch.cos(sec_centers), torch.sin(sec_centers)), dim=-1)       # (n_sec, 2)
    away_dirs = -u                                                                   # (n_sec, 2)

    cos = torch.relu(torch.mm(v_cmd_dir_1, u.T))  # (1, n_sec)
    active = d_1 < d_max
    w = (cos + c_val) * torch.exp(-alpha * d_1) * active.float()
    v_avoid_1 = v_cmd_norm_1 * (w @ away_dirs)

    # Sector 18 center ≈ 5° → u ≈ [0.996, 0.087], away_dir ≈ [-0.996, -0.087].
    # cos[0, 18] ≈ 0.996, w ≈ (0.996+0.15)*exp(-0.5) ≈ 1.146*0.6065 ≈ 0.695
    # v_avoid ≈ 0.5*0.695*[-0.996,-0.087] ≈ [-0.346, -0.030]
    expected_w = (cos[0, 18].item() + c_val) * math.exp(-0.5)
    assert abs(w[0, 18].item() - expected_w) < 1e-4
    # Strong backward push (negative x)
    assert v_avoid_1[0, 0].item() < -0.3

    # --- Case 3: forward command, clear environment → v_avoid ≈ 0 ---
    d_3 = torch.full((1, n_sec), inf)  # all clear
    active_3 = d_3 < d_max
    # exp(-inf) = 0, and active_3 is all False → w_3 all zeros
    w_3 = (cos + c_val) * torch.exp(-alpha * d_3) * active_3.float()
    assert (w_3 == 0.0).all(), "clear env should have zero weights"
    v_avoid_3 = v_cmd_norm_1 * (w_3 @ away_dirs)
    assert torch.all(v_avoid_3.abs() < 1e-6)

    # --- Case 4: forward command, lateral wall on left (cos ≈ 0, d small) ---
    # Sector 27 center = 95° ≈ left, u ≈ [-0.087, 0.996], cos ≈ 0 (forward cmd ⊥ left)
    d_4 = torch.full((1, n_sec), inf)
    d_4[0, 27] = 0.3  # left wall close
    active_4 = d_4 < d_max
    w_4 = (cos + c_val) * torch.exp(-alpha * d_4) * active_4.float()
    v_avoid_4 = 0.5 * (w_4 @ away_dirs)

    # cos[27] ≈ max(0, [1,0]·[-0.087,0.996]) ≈ 0
    # w ≈ (0 + 0.15) * exp(-0.3) ≈ 0.111
    # away_dir[27] = -u[27] ≈ [0.087, -0.996], pushes right+forward (away from left wall)
    expected_w_27 = 0.15 * math.exp(-0.3)
    assert abs(w_4[0, 27].item() - expected_w_27) < 1e-4
    # Push should be rightward (negative y in body frame) since wall is on left
    assert v_avoid_4[0, 1].item() < 0.0, f"should push right away from left wall, got vy={v_avoid_4[0,1].item():.4f}"
    # Gentle push magnitude ≈ 0.5 * 0.111 ≈ 0.056
    mag_4 = v_avoid_4.norm(dim=1).item()
    assert 0.03 < mag_4 < 0.10, f"lateral push should be gentle, got {mag_4:.4f}"

    # --- Case 5: 10-sector lateral wall (cos ≈ 0) → should NOT explode ---
    # Sectors 6-10 (span -115° to -75°, right side). With forward command,
    # cos is near-zero for most of these, dominated by c.
    d_5 = torch.full((1, n_sec), inf)
    for i in range(6, 11):
        d_5[0, i] = 0.5
    active_5 = d_5 < d_max
    w_5 = (cos + c_val) * torch.exp(-alpha * d_5) * active_5.float()
    v_avoid_5 = 0.5 * (w_5 @ away_dirs)
    mag_5 = v_avoid_5.norm(dim=1).item()
    # With cos≈0 for most of these 5 sectors, each contributes ~c*exp(-0.5)≈0.091.
    # Actual magnitude ~0.32 (y-components constructively add).
    assert mag_5 < 0.4, \
        f"lateral wall should not cause large avoidance: {mag_5:.4f} >= 0.3"
```

- [ ] **Step 2: 运行测试**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py::test_v_avoid_guided_formula -v
```

预期：PASS

- [ ] **Step 3: 运行完整测试套件**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

预期：全部 7 个测试通过（`test_v_avoid_paper_formula` 被替换为 `test_v_avoid_guided_formula`）。

- [ ] **Step 4: 提交**

```bash
git add legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py
git commit -m "test: replace v_avoid paper-formula test with guided-formula test"
```

---

### Task 4: 最终验证

**Files:**
- 无新建或修改，仅运行测试

- [ ] **Step 1: 运行完整测试套件确认无回归**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

预期：全部 7 个测试 PASS。

- [ ] **Step 2: 配置导入测试**

```bash
python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import Go2LidarPDRiskNetCfg
cfg = Go2LidarPDRiskNetCfg()
assert hasattr(cfg.pd_risknet, 'avoid_c')
assert cfg.pd_risknet.avoid_c == 0.15
print('Config import OK, avoid_c =', cfg.pd_risknet.avoid_c)
"
```

预期：`Config import OK, avoid_c = 0.15`

---

### 改动总结

| 文件 | 改动类型 | 行数 |
|------|---------|------|
| `go2_lidar_pd_risknet_config.py` | 新增 1 行 | +1 |
| `go2_lidar_pd_risknet.py` `_compute_v_avoid` | 重写 | ~40 行替换 ~35 行 |
| `test_go2_lidar_pd_risknet_math.py` | 替换 1 个测试 | ~80 行替换 ~33 行 |
