# PointNet+GRU 合并 Gradient Checkpointing

## 目标

将 PointNet 前向传播纳入 GRU 的 gradient checkpointing 区段，消除 PointNet 中间激活的存储开销，降低单轮训练时间（当前 ~90s，目标 ~35-40s）。

## 变更

### 近端路径

```python
# 旧：
chunk_seq = self.proximal_pointnet(chunk.reshape(c * T_prox, P, 3))
if self.training:
    _, chunk_h = checkpoint(self.proximal_gru, chunk_seq, use_reentrant=True)
else:
    _, chunk_h = self.proximal_gru(chunk_seq)

# 新：
chunk_input = chunk.reshape(c * T_prox, P, 3)
if self.training:
    _, chunk_h = checkpoint(
        lambda x: self.proximal_gru(self.proximal_pointnet(x)),
        chunk_input, use_reentrant=True
    )
else:
    _, chunk_h = self.proximal_gru(self.proximal_pointnet(chunk_input))
```

### 远端路径

```python
# 旧：
chunk_seq = self.distal_pointnet(chunk.reshape(c * T_dist, D, 3))
chunk_hidden = None
if hidden is not None:
    chunk_hidden = hidden[:, start:end, :]
if self.training:
    _, chunk_h = checkpoint(self.distal_gru, chunk_seq, chunk_hidden, use_reentrant=True)
else:
    _, chunk_h = self.distal_gru(chunk_seq, chunk_hidden)

# 新：
chunk_input = chunk.reshape(c * T_dist, D, 3)
chunk_hidden = None
if hidden is not None:
    chunk_hidden = hidden[:, start:end, :]
if self.training:
    _, chunk_h = checkpoint(
        lambda x, h: self.distal_gru(self.distal_pointnet(x), h),
        chunk_input, chunk_hidden, use_reentrant=True
    )
else:
    _, chunk_h = self.distal_gru(self.distal_pointnet(chunk_input), chunk_hidden)
```

## 数学等价性

`torch.utils.checkpoint.checkpoint(f, x)` 和 `f(x)` 的输出值和梯度完全相同。合并后：

- 输出值：`GRU(PointNet(x))` 不变
- 梯度：autograd 对同一复合函数求导，梯度路径不变
- PointNet 是确定性函数（固定权重、无 BN、无 dropout），重算结果按位一致

## 影响

| 维度 | 变化 | 原因 |
|------|:--:|------|
| 前向时间 | ↓ 减少 | PointNet 激活不再写入 autograd graph |
| 反向时间 | ↑ 微增 | PointNet 需要重算，但仅 2.6K 参数 |
| 激活内存 | ↓ >90% | 省去 16D/32D 中间层的存储 |
| 梯度值 | 不变 | 数学等价 |
| 推理时间 | 不变 | 推理时不走 checkpoint |

## 实现文件

`rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` — 修改 `_encode_proximal_points_chunked` 和 `_encode_distal_points_chunked`
