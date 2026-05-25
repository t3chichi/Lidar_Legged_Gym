# PD-RiskNet 架构对齐论文设计

## 目标

将 PD-RiskNet 感知模块对齐 Omni-Perception 论文中描述的架构：两条路径各用一个 GRU，3D 坐标直接输入，不做中间 MLP 编码。

## 当前架构 vs 论文

### 近端路径

| 步骤 | 当前 | 论文 | 改为 |
|------|------|------|------|
| Point Encoder | MLP(3→64→64) + ELU | 无 | 移除 |
| Spatial GRU | GRU(64→187) | GRU(3→187) | 改输入维度 |
| Temporal Memory | Memory GRU(187→187), 10帧 | 无（单帧） | 移除 |
| 每步隐藏态 | 跨步保持 | 每步零初始化 | 每步零初始化 |

### 远端路径

| 步骤 | 当前 | 论文 | 改为 |
|------|------|------|------|
| Point Encoder | MLP(3→64→64) + ELU | 无 | 移除 |
| Spatial GRU | GRU(64→64) | — | 重命名 distal_gru, GRU(3→64) |
| Temporal Memory | Memory GRU(64→64) | — | 移除 |
| 时空处理 | 分离（spatial + temporal） | 单个 GRU 跨步保持 | 单 GRU 跨步保持 |

### 参数变化

| 组件 | 当前 | 改为 |
|------|------|------|
| proximal_point_encoder | ~8K | 移除 |
| distal_point_encoder | ~8K | 移除 |
| proximal_gru | ~48K (64→187) | ~107K (3→187) |
| distal_gru | ~25K (64→64) | ~13K (3→64) |
| proximal_memory_a | ~175K | 移除 |
| distal_memory_a | ~25K | 移除 |
| **总计** | **~289K** | **~120K** (-58%) |

## 最终数据流

```
每 env step 输入: 当前帧 432 点 3D 坐标
                    ↓
         ┌── theta >= 20° (近端) ─────┐
         │  FPS → 192 点               │
         │  球坐标排序 (θ, φ)          │
         │  proximal_gru(3→187)        │  ← 每步零初始化，取最后隐藏态
         │  → 187D                     │
         │          ↓                  │
         │  height_supervisor          │  ← Linear(187→187), MSE, 辅助损失
         └─────────────────────────────┘
                    ↓
         ┌── theta < 20° (远端) ───────┐
         │  平均下采样 → 56 点          │
         │  球坐标排序 (θ, φ)          │
         │  distal_gru(3→64)           │  ← 隐藏态跨步保持，累积 ~10 帧历史
         │  → 64D                      │
         └─────────────────────────────┘
                    ↓
 [proprio(48) | prox_feat(187) | dist_feat(64)] = 299D
                    ↓
         Actor MLP  [1024, 512, 256, 128] → 12
         Critic MLP [512, 256, 128] → 1
```

## 关键设计决策

1. **移除 Point Encoder** — 论文将排序后的 3D 坐标直接送入 GRU。GRU 的输入门能自适应地学习坐标重映射，不需要显式 MLP 预处理。移除后梯度路径缩短两层。

2. **近端单帧** — 近端扫描密度高（theta >= 20°），单帧的 192 个 FPS 采样点已足够捕获近场几何结构。论文未描述近端时间建模。

3. **远端单 GRU 跨步** — 远端点稀疏（theta < 20°），依赖时间上下文感知远处环境变化。单个 GRU 每步处理当前帧的 56 个排序点，隐藏态自然累积历史信息——GRU 的门控机制比硬滑窗提供了更平滑的信息衰减。

4. **`is_recurrent = True`** — 远端 GRU 跨步隐藏态需要 PPO runner 管理。隐藏态接口只传递远端 GRU 状态，近端无关。

5. **监督器不变** — `height_supervisor`: Linear(187→187)，MSE 损失。当前网络已有充分表达能力，架构清理后近端特征质量提升，即使最简单的 Linear 也能获得更好的监督信号。

## 不改的部分

- Actor / Critic MLP 结构和维度
- 高度特权监督器
- FPS 采样、平均下采样、球坐标排序逻辑
- proximal/distal split (theta >= 20°)
- 观测空间（当前帧 432 点 3D 坐标）
- 配置中 `distal_history_length = 10` 保留，`proximal_history_length` 从 10 改为 1
