# PD-RiskNet 架构对齐论文 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 PD-RiskNet 感知模块的冗余 MLP 编码器和双 GRU 架构替换为论文中的单 GRU + 原始 3D 坐标直接输入。

**Architecture:** 移除 `proximal_point_encoder`、`distal_point_encoder`、`proximal_memory_a`、`distal_memory_a`。保留 `proximal_gru(3→187)`（每步零初始化）和 `distal_gru(3→64)`（隐藏态跨步保持）。近端单帧处理，远端单 GRU 同时编码空间和时间结构。

**Tech Stack:** PyTorch, `nn.GRU`

---

### Files to Modify

| 文件 | 改动类型 |
|------|---------|
| `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` | 大量重写 |
| `legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py` | 改一行 |

---

### Task 1: 配置 — proximal_history_length 改为 1

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py`

- [ ] **Step 1: 修改配置值**

将 `class policy` 段中第 214 行附近:
```python
        proximal_history_length = PROX_HISTORY_LENGTH
```
改为:
```python
        proximal_history_length = 1
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pillar_config.py
git commit -m "config: set proximal_history_length to 1 for single-frame GRU"
```

---

### Task 2: 重写 `__init__`、`get_hidden_states`、`reset`

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` (lines ~121-176, ~204-224)

- [ ] **Step 1: 重写 `__init__` 中的感知模块定义**

移除 lines 123-149 的四块定义（`proximal_point_encoder`、`distal_point_encoder`、`proximal_gru` 的旧定义、`distal_spatial_gru`、`proximal_memory_a`、`distal_memory_a`），替换为：

```python
        self.proximal_gru = nn.GRU(
            input_size=3,
            hidden_size=self.proximal_feature_dim,
            batch_first=True,
        )
        self.distal_gru = nn.GRU(
            input_size=3,
            hidden_size=self.distal_feature_dim,
            batch_first=True,
        )
```

注释掉 `from rsl_rl.networks import Memory` 导入（如果 noqa 保留也没关系，后续可清理）。

- [ ] **Step 2: 重写 `reset` 方法**

替换整个 `reset` 方法 (lines 204-207):

```python
    def reset(self, dones=None):
        if self.distal_gru_hidden is not None:
            mask = dones.bool() if dones is not None else None
            if mask is not None and mask.any():
                self.distal_gru_hidden[:, mask, :] = 0.0
        self._cached_actor_latent = None
```

- [ ] **Step 3: 重写 `get_hidden_states` 方法**

替换整个 `get_hidden_states` 方法 (lines 209-224):

```python
    def get_hidden_states(self):
        dist_hidden = self.distal_gru_hidden
        if dist_hidden is None:
            return (None, None)
        # Proximal GRU has no cross-step state; pad with zeros for compatibility.
        prox_pad = torch.zeros_like(dist_hidden)
        actor_hidden_states = (prox_pad, dist_hidden)
        critic_hidden_states = (prox_pad, dist_hidden)
        return actor_hidden_states, critic_hidden_states
```

- [ ] **Step 4: 在 `__init__` 末尾添加 `distal_gru_hidden` 属性**

在 `__init__` 的 `self._sampling_plan_ready = False` 之后添加:

```python
        self.distal_gru_hidden: torch.Tensor | None = None
```

- [ ] **Step 5: 更新 `_init_actor_hidden_like` 调用点**

在 `_build_actor_latent` 的推理分支中，处理远端隐藏态时用 `self.distal_gru_hidden` 替代 `self.distal_memory_a.hidden_states`。

- [ ] **Step 6: 提交**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "refactor: replace dual-GRU + point encoder with single GRU per path"
```

---

### Task 3: 重写近端编码路径

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` (`_encode_proximal_points_chunked`)

- [ ] **Step 1: 替换 `_encode_proximal_points_chunked` 方法**

将 lines 508-531 替换为：

```python
    def _encode_proximal_points_chunked(self, prox_points: torch.Tensor) -> torch.Tensor:
        """Encode sorted proximal 3D points through single GRU (zero-init per call).

        Args:
            prox_points: (B, T, P, 3) where T is 1 for inference or N for training batch.
        Returns:
            (B, T, proximal_feature_dim)
        """
        B, T_prox, P, _ = prox_points.shape
        out = torch.empty((B, T_prox, self.proximal_feature_dim),
                          device=prox_points.device, dtype=prox_points.dtype)
        chunk_size = 128
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = prox_points[start:end]  # (c, T, P, 3)
            c = end - start
            # Reshape: (c*T, P, 3) -> batch_first GRU, seq_len=P
            chunk_seq = chunk.reshape(c * T_prox, P, 3)
            if self.training:
                _, chunk_h = checkpoint(self.proximal_gru, chunk_seq, use_reentrant=True)
            else:
                _, chunk_h = self.proximal_gru(chunk_seq)
            # chunk_h: (1, c*T, 187) -> (c, T, 187)
            out[start:end] = chunk_h.squeeze(0).reshape(c, T_prox, -1)
        return out
```

- [ ] **Step 2: 提交**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "refactor: proximal path — remove point encoder, feed raw 3D to GRU"
```

---

### Task 4: 重写远端编码路径

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` (`_encode_distal_points_chunked`)

- [ ] **Step 1: 替换 `_encode_distal_points_chunked` 方法**

将 lines 533-554 替换为：

```python
    def _encode_distal_points_chunked(
        self, dist_points: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode sorted distal 3D points through single GRU with optional hidden state.

        Args:
            dist_points: (B, T, D, 3) where T is 1 for inference or N for training batch.
            hidden: (1, B, 64) optional initial hidden state. If None, zero-init.
        Returns:
            (output: (B, T, distal_feature_dim), final_hidden: (1, B, distal_feature_dim))
        """
        B, T_dist, D, _ = dist_points.shape
        out = torch.empty((B, T_dist, self.distal_feature_dim),
                          device=dist_points.device, dtype=dist_points.dtype)
        chunk_size = 128
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = dist_points[start:end]  # (c, T, D, 3)
            c = end - start
            chunk_seq = chunk.reshape(c * T_dist, D, 3)
            chunk_hidden = None
            if hidden is not None:
                chunk_hidden = hidden[:, start:end, :]  # (1, c, 64)
            if self.training:
                _, chunk_h = checkpoint(self.distal_gru, chunk_seq, chunk_hidden,
                                        use_reentrant=True)
            else:
                _, chunk_h = self.distal_gru(chunk_seq, chunk_hidden)
            out[start:end] = chunk_h.squeeze(0).reshape(c, T_dist, -1)
        final_hidden = chunk_h.reshape(1, B, -1)
        return out, final_hidden
```

- [ ] **Step 2: 提交**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "refactor: distal path — remove point encoder, feed raw 3D to GRU with hidden state"
```

---

### Task 5: 重写 `_encode_perception`、`_build_actor_latent` 和 `_build_replay_frame_features`

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` (`_encode_perception`, `_build_actor_latent`, `_build_replay_frame_features`)

- [ ] **Step 1: 替换 `_build_replay_frame_features` 方法**

将 lines 459-506 替换为：

```python
    def _build_replay_frame_features(self, lidar_points_seq: torch.Tensor, masks: torch.Tensor | None,
                                      init_dist_hidden: torch.Tensor | None = None):
        """Build per-frame proximal and distal features from training sequence.

        Proximal: process each frame independently through proximal_gru (zero-init each).
        Distal: process frames sequentially through distal_gru (hidden state carries across frames).

        Args:
            lidar_points_seq: (T, B, N, 3) point cloud sequence over env steps.
            masks: (T, B) validity masks.
            init_dist_hidden: (1, B, 64) initial distal GRU hidden state.
        Returns:
            prox_feat_seq: (T, B, 187), dist_feat_seq: (T, B, 64)
        """
        T_seq, B, N, _ = lidar_points_seq.shape

        if masks is not None:
            frame_any_valid = masks.any(dim=1)
            valid_frames = frame_any_valid.nonzero(as_tuple=False)
            effective_len = valid_frames[-1].item() + 1 if len(valid_frames) > 0 else T_seq
        else:
            effective_len = T_seq

        prox_feat_list = []
        dist_feat_list = []
        dist_hidden = init_dist_hidden  # from PPO runner's stored hidden state

        for t in range(effective_len):
            points_t = lidar_points_seq[t]  # (B, N, 3)
            prox_points_t, dist_points_t = self._compute_sampled_sorted_points_frame(points_t)
            # (B, P, 3) and (B, D, 3)

            # Proximal: single-frame, zero-init per call
            prox_feat_t = self._encode_proximal_points_chunked(
                prox_points_t.unsqueeze(1)
            ).squeeze(1)  # (B, 187)

            # Distal: hidden carries across frames
            dist_feat_t, dist_hidden = self._encode_distal_points_chunked(
                dist_points_t.unsqueeze(1), dist_hidden
            )
            dist_feat_t = dist_feat_t.squeeze(1)  # (B, 64)

            prox_feat_list.append(prox_feat_t)
            dist_feat_list.append(dist_feat_t)

        prox_feat_seq = torch.stack(prox_feat_list, dim=0)  # (T, B, 187)
        dist_feat_seq = torch.stack(dist_feat_list, dim=0)  # (T, B, 64)

        if effective_len < T_seq:
            pad_len = T_seq - effective_len
            z_prox = torch.zeros(pad_len, B, self.proximal_feature_dim,
                                 device=prox_feat_seq.device, dtype=prox_feat_seq.dtype)
            z_dist = torch.zeros(pad_len, B, self.distal_feature_dim,
                                 device=dist_feat_seq.device, dtype=dist_feat_seq.dtype)
            prox_feat_seq = torch.cat([prox_feat_seq, z_prox], dim=0)
            dist_feat_seq = torch.cat([dist_feat_seq, z_dist], dim=0)

        return prox_feat_seq, dist_feat_seq
```

- [ ] **Step 2: 替换 `_encode_perception` 方法**

将 lines 566-587 替换为：

```python
    def _encode_perception(self, observations: torch.Tensor, masks: torch.Tensor | None = None,
                           init_dist_hidden: torch.Tensor | None = None):
        """Split observations and encode LiDAR into frame features.

        For 2D inference: returns (proprio, prox_feat, dist_points).
          Distal encoding deferred to _build_actor_latent for hidden-state management.
        For 3D training: returns (proprio, prox_feat_seq, dist_feat_seq).
        """
        proprio, lidar_frame = self._split_obs(observations)

        if observations.dim() == 2:
            # Inference: sample, sort, encode proximal; return raw distal points.
            prox_points_t, dist_points_t = self._compute_sampled_sorted_points_frame(lidar_frame)
            # (B, P, 3) and (B, D, 3)
            prox_feat_t = self._encode_proximal_points_chunked(
                prox_points_t.unsqueeze(1)
            ).squeeze(1)  # (B, 187)
            return proprio, prox_feat_t, dist_points_t

        elif observations.dim() == 3:
            # Training: build full per-frame feature sequences.
            prox_frame_feat, dist_frame_feat = self._build_replay_frame_features(
                lidar_frame, masks, init_dist_hidden=init_dist_hidden
            )
            return proprio, prox_frame_feat, dist_frame_feat

        else:
            raise ValueError(f"Unsupported observations rank: {observations.dim()}")
```

- [ ] **Step 3: 替换 `_build_actor_latent` 方法**

将 lines 589-646 替换为：

```python
    def _build_actor_latent(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    ):
        _, dist_hidden_states = self._split_actor_hidden_states(hidden_states)

        if masks is not None:
            # Training: pass runner's stored distal hidden state as initial state.
            perception_out = self._encode_perception(
                observations, masks=masks, init_dist_hidden=dist_hidden_states
            )
            proprio, prox_frame_feat, dist_frame_feat = perception_out
            prox_feat = prox_frame_feat  # (T, B, 187)
            dist_feat = dist_frame_feat  # (T, B, 64)
            proprio = unpad_trajectories(proprio, masks)
            prox_feat = unpad_trajectories(prox_feat, masks)
            dist_feat = unpad_trajectories(dist_feat, masks)
        else:
            # Inference: _encode_perception returns (proprio, prox_feat, dist_points).
            perception_out = self._encode_perception(observations, masks=None)
            proprio, prox_feat, dist_points = perception_out
            prox_feat = prox_feat  # (B, 187) — already encoded

            # Distal: carry hidden state across steps.
            dist_feat_t, self.distal_gru_hidden = self._encode_distal_points_chunked(
                dist_points.unsqueeze(1), self.distal_gru_hidden
            )
            dist_feat = dist_feat_t.squeeze(0).squeeze(0)  # (B, 64)

        actor_latent = torch.cat((proprio, prox_feat, dist_feat), dim=-1)

        self._cached_proximal_feature = prox_feat
        self._cached_actor_latent = actor_latent
        return actor_latent
```

- [ ] **Step 4: 提交**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "refactor: simplify perception pipeline — proximal single-frame, distal hidden-state GRU"
```

---

### Task 6: 更新 `evaluate` 冷启动路径

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` (`evaluate` method, lines ~696-707)

- [ ] **Step 1: 替换 `evaluate` 中的冷启动远端处理**

将 lines 696-707 替换为：

```python
            proprio, prox_feat, dist_points = self._encode_perception(critic_observations, masks=None)
            # Proximal: already encoded by _encode_perception.
            # Distal: encode with hidden state for temporal consistency.
            dist_feat_t, _ = self._encode_distal_points_chunked(
                dist_points.unsqueeze(1), self.distal_gru_hidden
            )
            dist_feat = dist_feat_t.squeeze(0).squeeze(0)  # (B, 64)
            return self.critic(torch.cat((proprio, prox_feat, dist_feat), dim=-1))
```

- [ ] **Step 2: 提交**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "refactor: update evaluate cold-start to use single distal GRU"
```

---

### Task 7: 清理 — 移除未使用的导入和 `_run_actor_memory`

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py`

- [ ] **Step 1: 移除 `Memory` 导入**

将 line 10:
```python
from rsl_rl.networks import Memory
```
移除（或注释掉）。

- [ ] **Step 2: 移除 `_run_actor_memory` 方法（lines 289-323）和 `_frame_window_to_seq`、`_collapse_window_output`（lines 253-287）**

这三个方法与 `Memory` 类耦合，移除 Point Encoder + Memory 后不再需要。删除 lines 253-323。

- [ ] **Step 3: 提交**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "chore: remove unused Memory import and helper methods"
```

---

### Task 8: 运行测试验证无回归

**Files:** 无修改

- [ ] **Step 1: 运行数学测试**

```bash
conda run -n li_leggym python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

Expected: 所有测试 PASS（policy_shape_gate 等可能需要调整 mock）

- [ ] **Step 2: 验证配置导入无语法错误**

```bash
conda run -n li_leggym python -c "
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pillar_config import Go2LidarPillarCfg, Go2LidarPillarCfgPPO
cfg = Go2LidarPillarCfgPPO()
assert cfg.policy.proximal_history_length == 1
print('Config OK')
"
```

Expected: `Config OK`

- [ ] **Step 3: 验证网络初始化无错误**

```bash
conda run -n li_leggym python -c "
import torch
import sys
sys.path.insert(0, 'rsl_rl')
from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic

model = PDRiskNetActorCritic(
    num_actor_obs=48 + 432*3,
    num_critic_obs=235,
    num_actions=12,
    proximal_history_length=1,
    distal_history_length=10,
    num_lidar_points=432,
    proximal_points=192,
    distal_points=56,
    split_theta_deg=20.0,
    proximal_feature_dim=187,
    distal_feature_dim=64,
    proprio_obs_dim=48,
    privileged_height_dim=187,
)

# 模拟推理
obs = torch.randn(4, 48 + 432*3)
model.reset()
action = model.act_inference(obs)
assert action.shape == (4, 12), f'Expected (4, 12), got {action.shape}'
print('Network init and inference OK')
"
```

Expected: `Network init and inference OK`
