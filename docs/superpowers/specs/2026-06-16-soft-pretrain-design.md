# go2_soft_pretrain 软约束预训练设计

日期: 2026-06-16 | 状态: 设计中

## 概述

新建 `go2_soft_pretrain` 任务：物理世界为纯平地，LiDAR 感知世界包含随机方柱。
机器人在物理上可穿过柱子（无碰撞惩罚），但 LiDAR 始终能看到障碍物。
通过 vel_avoid + rays 奖励在步态学习阶段同步建立"感知有意义"的初步意识。

目标：消除当前"平地跟指令 → 转避障 → 崩塌"的相位转换问题。

## 背景

当前预训练（`go2_pd_pretrain`）在纯平地上用 tracking_lin_vel + tracking_ang_vel 学习指令跟随，
转入避障地形后策略需要同时遗忘指令跟随习惯并重新学习 LiDAR 驱动的避障行为。
两者在奖励空间中存在冲突（指令驱动 vs 感知驱动），导致 1500~2500 轮后行为崩塌。

新方案的思路：从第一步起，策略的所有经验都在"LiDAR 感知 + 前进"的分布里，
不需要任何行为模式转换。

## 核心设计

### 双世界分离

```
物理世界:  terrain_type='plane'
          → 纯平地，Isaac Gym 物理计算无柱子
          → 机器人可自由穿越任何位置，无碰撞惩罚

LiDAR 世界: 自定义 wp.Mesh
          → 平面 + 随机放置的方柱（3D 盒子网格）
          → 每环境独立柱子布局，4096 envs 4×16 变量
          → 柱子在高度场上不可见（无物理碰撞）
```

### 柱子生成

```python
# 复用 pillar_field_terrain 的放置逻辑
# 极坐标采样 + 排斥约束 → 输出 3D 顶点 + 三角面 → 合并到 wp.Mesh
```

柱子参数（`class pd_risknet:`，与正式梅花桩一致）：
```python
pillar_count       = 30      # 柱子数量
pillar_spawn_radius= 9.0     # 生成半径 (m)
pillar_size_x_min  = 0.40    # x 边长最小 (m)
pillar_size_x_max  = 0.60
pillar_size_y_min  = 0.40
pillar_size_y_max  = 0.60
pillar_height_min  = 0.60    # 高度最小 (m)
pillar_height_max  = 1.00
pillar_min_separation = 2.5  # 柱心最小间距 (m)
pillar_center_clear_radius = 1.6  # 中心净空 (m)
pillar_allow_height_variation = True
```

### 指令

```python
commands:
    heading_command = False
    lin_vel_x  = [-1.0, 1.0]
    lin_vel_y  = [0.0, 0.0]     # 不随机横向，v_avoid 已提供横向引导
    ang_vel_yaw = [0.0, 0.0]   # 无角速度指令
```

### 奖励

| 奖励 | weight | 说明 |
|------|:---:|------|
| vel_avoid | 1.0 | LiDAR 驱动的避障合速度跟踪 |
| rays | 0.5 | 角速度跟踪（朝向开阔方向） |
| tracking_lin_vel | 0 | — |
| tracking_ang_vel | 0 | — |
| channel_forward | 0 | 预训练无通道方向 |
| collision | 0 | 无物理碰撞惩罚 |
| feet_air_time | 1.0 | 步态节律 |
| gait_2_step | -0.5 | 两脚步态 |
| ang_vel_xy | -0.1 | 躯干稳定 |
| base_height | -5.0 | 高度保持 |
| orientation | -5.0 | 姿态保持 |
| torques | -0.000025 | 能耗 |
| dof_acc | -2.5e-7 | 平滑 |
| action_rate | -0.01 | 动作连贯 |

### 观测

与正式走廊/梅花桩一致：
- 本体 48 维（`HEADING_OBS_ENABLED=False`）
- LiDAR 1500 点 × 3 坐标 = 4500 维
- 总观测 4548 维

### 无课程

`curriculum = False`，所有 env 柱子数量和难度相同。

## 文件结构

| 文件 | 操作 | 职责 |
|------|:---:|------|
| `go2_soft_pillar_pretrain.py` | 新建 | 配置文件 |
| `go2_lidar_pd_risknet.py` | 修改 | `_init_lidar_sensor` 增加自定义 mesh 路径 |
| `__init__.py` | 修改 | 注册 `go2_soft_pretrain` 任务 |
| `terrain_utils.py` 或新工具文件 | 新建/修改 | 柱子 → 3D 网格生成函数 |

## 不变项

- PPO 参数
- PD-RiskNet 架构
- 本体观测空间
- LiDAR 传感器配置

## 风险

| 风险 | 缓解 |
|------|------|
| vel_avoid+rays 信号不足以驱动类人步态 | 步态正则全保留，物理上四肢交替摆动自然产生前进 |
| 无碰撞惩罚 → 策略学不到"避开"的价值 | rays+avoid 的方向性引导足够；离散柱子消灭了 v_avoid 自维持循环 |
