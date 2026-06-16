# PointNet+GRU 合并 Checkpointing 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 PointNet 纳入 GRU 的 gradient checkpointing 区段，消除中间激活存储，降低训练时间。

**Architecture:** 修改 `_encode_proximal_points_chunked` 和 `_encode_distal_points_chunked`，用 lambda 将 PointNet + GRU 合并为单一 checkpoint 调用。推理路径不变。

**Tech Stack:** PyTorch, torch.utils.checkpoint

**Spec:** `docs/superpowers/specs/2026-06-16-pointnet-gru-checkpoint-merge.md`

---

### 文件职责

| 文件 | 职责 |
|------|------|
| `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` | 修改两个 encode 方法的 checkpoint 边界 |

---

### Task 1: 合并近端 PointNet+GRU checkpointing

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py:447-472`

- [ ] **Step 1: 修改 `_encode_proximal_points_chunked`**

将当前代码 (行 458-471):
```python
        chunk_size = 128
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = prox_points[start:end]  # (c, T, P, 3)
            c = end - start
            # Reshape: (c*T, P, 3) -> batch_first GRU, seq_len=P
            # PointNet: per-point feature extraction -> (c*T, P, 64)
            chunk_seq = self.proximal_pointnet(chunk.reshape(c * T_prox, P, 3))
            if self.training:
                _, chunk_h = checkpoint(self.proximal_gru, chunk_seq, use_reentrant=True)
            else:
                _, chunk_h = self.proximal_gru(chunk_seq)
            # chunk_h: (1, c*T, 187) -> (c, T, 187)
            out[start:end] = chunk_h.squeeze(0).reshape(c, T_prox, -1)
```

替换为:
```python
        chunk_size = 128
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = prox_points[start:end]  # (c, T, P, 3)
            c = end - start
            chunk_input = chunk.reshape(c * T_prox, P, 3)
            if self.training:
                _, chunk_h = checkpoint(
                    lambda x: self.proximal_gru(self.proximal_pointnet(x)),
                    chunk_input, use_reentrant=True
                )
            else:
                _, chunk_h = self.proximal_gru(self.proximal_pointnet(chunk_input))
            # chunk_h: (1, c*T, 187) -> (c, T, 187)
            out[start:end] = chunk_h.squeeze(0).reshape(c, T_prox, -1)
```

- [ ] **Step 2: Commit**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "perf: merge proximal PointNet into GRU checkpoint region"
```

---

### Task 2: 合并远端 PointNet+GRU checkpointing

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py:474-504`

- [ ] **Step 1: 修改 `_encode_distal_points_chunked`**

将当前代码 (行 491-504):
```python
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = dist_points[start:end]  # (c, T, D, 3)
            c = end - start
            # PointNet: per-point feature extraction → (c*T, D, 64)
            chunk_seq = self.distal_pointnet(chunk.reshape(c * T_dist, D, 3))
            chunk_hidden = None
            if hidden is not None:
                chunk_hidden = hidden[:, start:end, :]  # (1, c, 64)
            if self.training:
                _, chunk_h = checkpoint(self.distal_gru, chunk_seq, chunk_hidden,
                                        use_reentrant=True)
            else:
                _, chunk_h = self.distal_gru(chunk_seq, chunk_hidden)
            out[start:end] = chunk_h.squeeze(0).reshape(c, T_dist, -1)
            final_hidden[:, start:end, :] = chunk_h
```

替换为:
```python
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = dist_points[start:end]  # (c, T, D, 3)
            c = end - start
            chunk_input = chunk.reshape(c * T_dist, D, 3)
            chunk_hidden = None
            if hidden is not None:
                chunk_hidden = hidden[:, start:end, :]  # (1, c, 64)
            if self.training:
                _, chunk_h = checkpoint(
                    lambda x, h: self.distal_gru(self.distal_pointnet(x), h),
                    chunk_input, chunk_hidden, use_reentrant=True
                )
            else:
                _, chunk_h = self.distal_gru(self.distal_pointnet(chunk_input),
                                             chunk_hidden)
            out[start:end] = chunk_h.squeeze(0).reshape(c, T_dist, -1)
            final_hidden[:, start:end, :] = chunk_h
```

- [ ] **Step 2: Commit**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "perf: merge distal PointNet into GRU checkpoint region"
```

---

### Task 3: 验证

- [ ] **Step 1: 运行所有测试确认无回归**

```bash
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

Expected: 32 passed

- [ ] **Step 2: 验证推理和前向仍可用**

```bash
conda run -n li_leggym python -c "
from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic
import torch
m = PDRiskNetActorCritic(48+1024*3, 235, 12, perception_enabled=True, num_lidar_points=1024)
m.eval()
obs = torch.randn(2, 48+1024*3)
with torch.no_grad():
    m.update_distribution(obs)
    a = m.act(obs)
print(f'Inference OK, action shape: {a.shape}')
# 训练模式
m.train()
obs = torch.randn(2, 48+1024*3)
m.update_distribution(obs)
a = m.act(obs)
print(f'Training OK, action shape: {a.shape}')
priv = torch.randn(2, 187)
loss = m.get_auxiliary_loss(priv)
loss.backward()
print(f'Backward OK, loss: {loss.item():.4f}')
"
```

Expected: 三段 OK 输出，无报错
