# Critic Observation Symmetry Fix

**日期**: 2026-06-30
**状态**: 设计完成，待实现

## 问题

`get_go2_cmd_safe_xsym_obs_act` 无视 `obs_type` 参数，把 critic 观测（高度网格 187=17×11）当作 wrapped policy 观测（2736）处理，切片越界导致 reshape 崩溃。

PPO 调用链路：
```
policy obs:  obs_batch [M, 2736], obs_type="policy"   → 需完整 proprio + LiDAR 镜像
critic obs:  obs_batch [M, 187],  obs_type="critic"   → 需高度网格 Y 轴镜像
```

## 方案

在对称函数顶部新增 `obs_type == "critic"` 分支，对高度网格做 Y 轴镜像后提前返回。

### 高度网格镜像

```
[B, x_count * y_count] → reshape [B, x_count, y_count] → torch.flip dims=[2] → reshape_as
```

一行向量化操作，与 LiDAR 处理的 `reshape → 空间变换 → flatten` 模式一致。

### 参数传递

| 参数 | 来源 | 绑定方式 |
|------|------|---------|
| `height_grid_x_count` | `env.cfg.terrain.measured_grid_x_count` | `_setup_symmetry` partial 绑定 |
| `height_grid_y_count` | `env.cfg.terrain.measured_grid_y_count` | `_setup_symmetry` partial 绑定 |

### 涉及文件

| 文件 | 修改 |
|------|------|
| `go2_cmd_safe.py` | 对称函数新增 critic 分支 |
| `on_policy_runner.py` | `_setup_symmetry` 读取并绑定网格参数 |
| `test_go2_xsym.py` | 新增 critic 观测测试 |
