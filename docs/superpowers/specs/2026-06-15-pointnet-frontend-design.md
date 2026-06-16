# PD-RiskNet PointNet 前端设计

## 目标

在 PD-RiskNet 近端和远端 GRU 之前各插入一个轻量 PointNet（per-point 共享 MLP），将原始 3D 坐标映射为几何特征向量后再送入 GRU。与当前单 GRU 架构形成消融对照。

## 动机

当前架构：3D 坐标 → 球坐标排序 → GRU(3→hidden)。GRU 对 3D 坐标的第一步变换是纯线性（W_ih · [x,y,z]），无法在首层学习非线性几何量（距离、法向、曲率等）。插入浅层 PointNet 后，GRU 从"学习空间+时序映射"简化为"仅学习时序映射"。

## 架构

### 数据流

```
Proximal: FPS(192点) → sort(θ,φ) → PointNet_prox → GRU(64→187) → 187D
                                      ↑ per-point MLP   ↑ zero-init per step

Distal:   avg(56点)  → sort(θ,φ) → PointNet_dist → GRU(64→64)  → 64D
                                      ↑ per-point MLP  ↑ hidden跨步保持
```

### PointNet 规格

```
PointNet_prox:  Linear(3→16) → ELU → Linear(16→32) → ELU → Linear(32→64)
PointNet_dist:  Linear(3→16) → ELU → Linear(16→32) → ELU → Linear(32→64)
```

- **近端和远端使用独立权重**，不做参数共享
- **无 BatchNorm**（PPO mini-batch 统计量噪声大）
- **ELU 激活**（与项目 Actor/Critic 一致，避免 RL 早期 dead neuron）
- **无全局 Max Pooling**（保留点序列空间结构供 GRU 处理）
- 每层输出维度 16→32→64，三层浅网足够学习基本几何原语

### GRU 变更

| 组件 | 当前 | 改为 |
|------|------|------|
| `proximal_gru` | `nn.GRU(3, 187)` | `nn.GRU(64, 187)` |
| `distal_gru` | `nn.GRU(3, 64)` | `nn.GRU(64, 64)` |

GRU 后续逻辑不变：近端每步 zero-init，远端隐藏态跨步保持。

## 参数量

| 组件 | 当前 | 新方案 | 增量 |
|------|-----:|-----:|-----:|
| PointNet_prox | 0 | 2,608 | +2,608 |
| PointNet_dist | 0 | 2,608 | +2,608 |
| Proximal GRU (W_ih) | 3×187×3 = 1,683 | 64×187×3 = 35,904 | +34,221 |
| Distal GRU (W_ih) | 3×64×3 = 576 | 64×64×3 = 12,288 | +11,712 |
| GRU (W_hh) | 不变 | 不变 | 0 |
| Actor/Critic | 不变 | 不变 | 0 |
| **总计** | **~120K** | **~171K** | **~51K** |

51K 增量对比旧架构（PointNet + 双 GRU）的 289K 仍少 40%。

## 显存影响

- 参数：51K × 4 bytes = 200 KB（含 Adam ×3 ≈ 600 KB）
- 激活峰值（chunk_size=128, proximal 192 点 × 64D）：约 6 MB
- 远端激活：约 1.8 MB
- 总增量 < 10 MB，对比总训练显存（10-20 GB）可忽略
- gradient checkpointing 保持不变，PointNet 重算成本极低

## 不改的部分

- FPS 采样、平均下采样、球坐标排序逻辑
- Proximal/distal split (theta >= 20°)
- Height supervisor (Linear 187→187)
- Actor / Critic MLP 结构
- 感知输出维度（近端 187 + 远端 64 = 251 → Actor）
- 观测空间布局
- 检查点兼容性逻辑（需新增针对新架构的 guard）

## 检查点兼容性

新架构的 `proximal_gru` 和 `distal_gru` 的 `input_size` 从 3 变为 64，与当前 checkpoint 不兼容。`load_state_dict` 中的兼容逻辑需要更新：

- 检测 `proximal_gru.weight_ih_l0.shape[1]` 是否为旧值（3），若是则跳过所有感知模块权重
- PointNet 模块没有旧 checkpoint 对应项，由 `strict=False` 自然处理

## 消融验证逻辑

| 配置 | 感知架构 | 预期 |
|------|---------|------|
| 当前 baseline | 单 GRU (3→hidden) | 已观察：无主动避障 |
| 新方案 | PointNet + GRU (64→hidden) | 待实验 |

- 如果新方案效果显著改善 → PointNet 前端是有效增益，3D 坐标直接入 GRU 信息密度不够
- 如果新方案无明显变化 → 感知架构不是瓶颈，问题在奖励设计或训练稳定性
- 极端情况下如果新方案更差 → PointNet 增加的参数给 GRU 带来额外优化负担，需检查训练超参

## 实现文件

| 文件 | 变更 |
|------|------|
| `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` | 添加 `PerPointMLP` 类，修改 `__init__` 创建 PointNet 和更新 GRU 维度，修改 `_encode_proximal_points_chunked` 和 `_encode_distal_points_chunked` 在 GRU 前插入 PointNet，更新 `load_state_dict` 兼容逻辑 |
| 配置文件 | 无需变更（PointNet 不需要额外配置参数） |
