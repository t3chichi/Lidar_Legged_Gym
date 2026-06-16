# PointNet 前端架构改造 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PD-RiskNet 近端/远端 GRU 前插入轻量 per-point MLP（PointNet），将原始 3D 坐标映射为 64D 几何特征后送入 GRU。

**Architecture:** 添加 `PerPointMLP` 私有类（3→16→32→64, ELU, 无BN, 无max pool），近端和远端各自独立实例。GRU 的 `input_size` 从 3 改为 64。前向传播时在 GRU 前以相同 chunk 策略插入 PointNet。

**Tech Stack:** Python, PyTorch, torch.nn.GRU, torch.utils.checkpoint

**基线 Tag:** `baseline-single-gru` — 当前单GRU架构用于消融对比

**Spec:** `docs/superpowers/specs/2026-06-15-pointnet-frontend-design.md`

---

### 文件职责

| 文件 | 职责 |
|------|------|
| `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py` | 所有架构变更：`PerPointMLP` 类、GRU 维度修改、前向传播插入 PointNet、检查点兼容性 |
| `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py` | PointNet 单元测试、架构集成测试 |

---

### Task 1: 创建 PerPointMLP 类

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py:45-60`（在 PDRiskNetActorCritic 类定义前插入）

- [ ] **Step 1: 在 PDRiskNetActorCritic 类定义前添加 PerPointMLP**

```python
class PerPointMLP(nn.Module):
    """Per-point shared MLP for geometric feature extraction.

    Maps raw 3D coordinates to a feature vector through a shallow MLP.
    No BatchNorm (unstable under RL mini-batch noise), no global pooling
    (preserves spatial structure for downstream GRU)."""

    def __init__(self, in_dim: int = 3, hidden_dims: list[int] = [16, 32],
                 out_dim: int = 64, activation: str = "elu"):
        super().__init__()
        act_fn = resolve_nn_activation(activation)
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn)
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 3) → (..., out_dim)"""
        return self.mlp(x)
```

插入位置：在 `class PDRiskNetActorCritic(nn.Module):` (当前第 45 行) 之前。

- [ ] **Step 2: 运行现有测试确认插入不破坏任何功能**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v -x
```

Expected: 所有现有测试仍然通过（PerPointMLP 尚未被引用）。

- [ ] **Step 3: Commit**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "feat: add PerPointMLP class for per-point geometric feature extraction"
```

---

### Task 2: 修改 __init__ 创建 PointNet 实例并更新 GRU 维度

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py:128-137`

- [ ] **Step 1: 在 __init__ 中添加 PointNet 模块并修改 GRU input_size**

将当前代码 (行 128-137):
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

替换为:
```python
        self.proximal_pointnet = PerPointMLP(
            in_dim=3, hidden_dims=[16, 32], out_dim=64, activation=activation
        )
        self.distal_pointnet = PerPointMLP(
            in_dim=3, hidden_dims=[16, 32], out_dim=64, activation=activation
        )

        self.proximal_gru = nn.GRU(
            input_size=64,
            hidden_size=self.proximal_feature_dim,
            batch_first=True,
        )
        self.distal_gru = nn.GRU(
            input_size=64,
            hidden_size=self.distal_feature_dim,
            batch_first=True,
        )
```

- [ ] **Step 2: 运行现有测试，预期失败（GRU input_size 不匹配旧的 checkpoint 加载逻辑）**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v -x
```

测试中如果有创建 `PDRiskNetActorCritic` 实例的逻辑，会因为 GRU input_size 从 3 变成 64 而改变参数计数。确认哪些测试受到影响。

- [ ] **Step 3: Commit**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "feat: add proximal/distal PointNet instances, update GRU input_size 3→64"
```

---

### Task 3: 修改 _encode_proximal_points_chunked 在 GRU 前插入 PointNet

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py:415-439`

- [ ] **Step 1: 在当前 GRU 调用前插入 PointNet forward**

将当前代码 (行 415-439):
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

替换为:
```python
    def _encode_proximal_points_chunked(self, prox_points: torch.Tensor) -> torch.Tensor:
        """Encode sorted proximal 3D points through PointNet + GRU (zero-init per call).

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
            # PointNet: per-point feature extraction → (c*T, P, 64)
            chunk_seq = self.proximal_pointnet(chunk.reshape(c * T_prox, P, 3))
            if self.training:
                _, chunk_h = checkpoint(self.proximal_gru, chunk_seq, use_reentrant=True)
            else:
                _, chunk_h = self.proximal_gru(chunk_seq)
            # chunk_h: (1, c*T, 187) -> (c, T, 187)
            out[start:end] = chunk_h.squeeze(0).reshape(c, T_prox, -1)
        return out
```

变更要点：`chunk.reshape(c * T_prox, P, 3)` 后先过 `self.proximal_pointnet()`，得到 `(c*T, P, 64)` 再送入 GRU。

- [ ] **Step 2: Commit**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "feat: insert proximal PointNet before GRU in _encode_proximal_points_chunked"
```

---

### Task 4: 修改 _encode_distal_points_chunked 在 GRU 前插入 PointNet

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py:441-473`

- [ ] **Step 1: 在当前 GRU 调用前插入 PointNet forward**

将当前代码 (行 441-473):
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
        final_hidden = torch.zeros(1, B, self.distal_feature_dim,
                                   device=dist_points.device, dtype=dist_points.dtype)
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
            final_hidden[:, start:end, :] = chunk_h
        return out, final_hidden
```

替换为:
```python
    def _encode_distal_points_chunked(
        self, dist_points: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode sorted distal 3D points through PointNet + GRU with optional hidden state.

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
        final_hidden = torch.zeros(1, B, self.distal_feature_dim,
                                   device=dist_points.device, dtype=dist_points.dtype)
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
        return out, final_hidden
```

变更要点：`chunk.reshape(c * T_dist, D, 3)` 后先过 `self.distal_pointnet()`，得到 `(c*T, D, 64)` 再送入 GRU。

- [ ] **Step 2: Commit**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "feat: insert distal PointNet before GRU in _encode_distal_points_chunked"
```

---

### Task 5: 更新 load_state_dict 兼容逻辑

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py:656-690`

- [ ] **Step 1: 更新检查点兼容性守卫**

当前兼容逻辑 (行 669-690) 检测 `proximal_gru.weight_ih_l0` shape 不匹配时跳过旧感知权重。新架构的 `input_size=64`，旧 checkpoint 的 `input_size` 可能是 3（当前基线）或 64+（历史架构）。需要统一处理：

将当前兼容逻辑 (行 669-690):
```python
        # 兼容旧 PD-RiskNet checkpoint：proximal_gru input_size 从 64→3,
        # distal_gru 替代了 distal_spatial_gru。旧权重无法映射到新架构，
        # 移除感知模块的旧权重，让其随机初始化。
        if 'proximal_gru.weight_ih_l0' in state_dict:
            expected = self.proximal_gru.weight_ih_l0.shape
            actual = state_dict['proximal_gru.weight_ih_l0'].shape
            if expected != actual:
                print("[PDRiskNetActorCritic] Old checkpoint detected "
                      f"(proximal_gru input_size: checkpoint={actual[1]}, model={expected[1]}). "
                      "Perception modules will be randomly initialized.")
                prefix_blacklist = (
                    'proximal_gru.', 'distal_gru.',
                    'proximal_point_encoder.', 'distal_point_encoder.',
                    'proximal_memory_a.', 'distal_memory_a.',
                    'distal_spatial_gru.',
                )
                keys_to_remove = [k for k in state_dict
                                  if k.startswith(prefix_blacklist)]
                for k in keys_to_remove:
                    del state_dict[k]
```

替换为:
```python
        # 兼容旧 checkpoint：感知模块架构变更时跳过不兼容的权重。
        # 检测 trigger: proximal_gru.weight_ih_l0 的 input_size 维度不匹配。
        if 'proximal_gru.weight_ih_l0' in state_dict:
            expected = self.proximal_gru.weight_ih_l0.shape
            actual = state_dict['proximal_gru.weight_ih_l0'].shape
            if expected != actual:
                print("[PDRiskNetActorCritic] Perception architecture changed "
                      f"(proximal_gru input_size: checkpoint={actual[1]}, model={expected[1]}). "
                      "Perception modules will be randomly initialized.")
                prefix_blacklist = (
                    'proximal_gru.', 'distal_gru.',
                    'proximal_point_encoder.', 'distal_point_encoder.',
                    'proximal_memory_a.', 'distal_memory_a.',
                    'distal_spatial_gru.',
                    'proximal_pointnet.', 'distal_pointnet.',
                )
                keys_to_remove = [k for k in state_dict
                                  if k.startswith(prefix_blacklist)]
                for k in keys_to_remove:
                    del state_dict[k]
```

变更要点：在 blacklist 中添加 `'proximal_pointnet.'` 和 `'distal_pointnet.'`；更新日志信息去掉过时的 "64→3" 描述。

- [ ] **Step 2: Commit**

```bash
git add rsl_rl/rsl_rl/modules/pd_risknet_actor_critic.py
git commit -m "fix: extend checkpoint compat guard to cover pointnet modules"
```

---

### Task 6: 编写单元测试

**Files:**
- Modify: `legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py`

- [ ] **Step 1: 添加 PerPointMLP 单元测试**

在文件末尾追加:

```python
# ── PerPointMLP tests ──────────────────────────────────────────

class TestPerPointMLP(unittest.TestCase):

    def setUp(self):
        from rsl_rl.modules.pd_risknet_actor_critic import PerPointMLP
        self.mlp = PerPointMLP(in_dim=3, hidden_dims=[16, 32], out_dim=64)
        self.mlp.eval()

    def test_output_shape_single_point(self):
        """PerPointMLP maps (3,) → (64,)."""
        x = torch.randn(3)
        out = self.mlp(x)
        self.assertEqual(out.shape, (64,))

    def test_output_shape_batch_of_points(self):
        """PerPointMLP maps (B, N, 3) → (B, N, 64)."""
        x = torch.randn(4, 192, 3)
        out = self.mlp(x)
        self.assertEqual(out.shape, (4, 192, 64))

    def test_output_shape_flattened(self):
        """PerPointMLP maps (B*N, 3) → (B*N, 64) — the chunked call pattern."""
        x = torch.randn(256, 3)
        out = self.mlp(x)
        self.assertEqual(out.shape, (256, 64))

    def test_different_inputs_produce_different_outputs(self):
        """Distinct 3D points should map to distinct features."""
        x1 = torch.tensor([[1.0, 0.0, 0.0]])
        x2 = torch.tensor([[0.0, 1.0, 0.0]])
        out1 = self.mlp(x1)
        out2 = self.mlp(x2)
        self.assertFalse(torch.allclose(out1, out2, atol=1e-4))

    def test_same_input_produces_same_output(self):
        """Deterministic: same input → same output (no dropout, no BN)."""
        x = torch.randn(16, 3)
        out1 = self.mlp(x)
        out2 = self.mlp(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_two_instances_have_independent_weights(self):
        """Proximal and distal PointNets must not share parameters."""
        from rsl_rl.modules.pd_risknet_actor_critic import PerPointMLP
        mlp1 = PerPointMLP()
        mlp2 = PerPointMLP()
        # Different initializations should produce different outputs
        x = torch.randn(4, 192, 3)
        out1 = mlp1(x)
        out2 = mlp2(x)
        self.assertFalse(torch.allclose(out1, out2, atol=1e-4))


class TestPDRiskNetWithPointNet(unittest.TestCase):

    def setUp(self):
        from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic
        self.num_obs = 48 + 432 * 3  # proprio + 1 frame of 432 points × 3D
        self.model = PDRiskNetActorCritic(
            num_actor_obs=self.num_obs,
            num_critic_obs=235,
            num_actions=12,
            perception_enabled=True,
            history_length=1,
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
        self.model.eval()

    def test_has_pointnet_modules(self):
        """Model should have proximal_pointnet and distal_pointnet."""
        self.assertTrue(hasattr(self.model, 'proximal_pointnet'))
        self.assertTrue(hasattr(self.model, 'distal_pointnet'))

    def test_gru_input_size_is_64(self):
        """GRU input_size should be 64 (PointNet output dim), not 3."""
        self.assertEqual(self.model.proximal_gru.input_size, 64)
        self.assertEqual(self.model.distal_gru.input_size, 64)

    def test_forward_pass_does_not_crash(self):
        """Full forward pass with single-frame observation."""
        obs = torch.randn(2, self.num_obs)  # 2 envs
        with torch.no_grad():
            self.model.update_distribution(obs)
            actions = self.model.act(obs)
        self.assertEqual(actions.shape, (2, 12))

    def test_auxiliary_loss_returns_scalar(self):
        """Height supervision loss should return a scalar tensor."""
        obs = torch.randn(2, self.num_obs)
        priv = torch.randn(2, 187)
        with torch.no_grad():
            self.model.update_distribution(obs)
            loss = self.model.get_auxiliary_loss(priv)
        self.assertEqual(loss.dim(), 0)  # scalar

    def test_parameter_count_reasonable(self):
        """Total params should be ~171K (±5K)."""
        total = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total, 165_000)
        self.assertLess(total, 180_000)

    def test_checkpoint_compat_skips_mismatched_weights(self):
        """load_state_dict should skip perception weights when GRU input_size mismatches."""
        old_state = self.model.state_dict()
        # Simulate an old checkpoint with input_size=3 GRU
        old_state['proximal_gru.weight_ih_l0'] = torch.randn(187 * 3, 3)  # (3*187, 3)
        old_state['distal_gru.weight_ih_l0'] = torch.randn(64 * 3, 3)     # (3*64, 3)
        # Should not raise — compat logic should strip perception keys
        self.model.load_state_dict(old_state, strict=False)
```

- [ ] **Step 2: 运行新测试，预期全部通过**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v -k "TestPerPointMLP or TestPDRiskNetWithPointNet"
```

- [ ] **Step 3: 运行全部测试，确认无回归**

```bash
python -m pytest legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py -v
```

- [ ] **Step 4: Commit**

```bash
git add legged_gym/legged_gym/tests/test_go2_lidar_pd_risknet_math.py
git commit -m "test: add PerPointMLP unit tests and PointNet-GRU integration tests"
```

---

### Task 7: 最终验证

- [ ] **Step 1: 运行所有项目测试**

```bash
python -m pytest legged_gym/legged_gym/tests/ -v --timeout=120
```

- [ ] **Step 2: 确认 baseline tag 可切回**

```bash
git checkout baseline-single-gru && python -c "
from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic
m = PDRiskNetActorCritic(48+432*3, 235, 12, perception_enabled=True)
assert not hasattr(m, 'proximal_pointnet'), 'Baseline should not have pointnet'
assert m.proximal_gru.input_size == 3, 'Baseline GRU input_size should be 3'
print('Baseline verified: single-GRU architecture intact')
"
git checkout main
```

- [ ] **Step 3: 确认 main 上是新架构**

```bash
python -c "
from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic
m = PDRiskNetActorCritic(48+432*3, 235, 12, perception_enabled=True)
assert hasattr(m, 'proximal_pointnet'), 'Should have pointnet'
assert m.proximal_gru.input_size == 64, 'GRU input_size should be 64'
print('Verified: PointNet+GRU architecture active')
"
```

- [ ] **Step 5: Commit (如有测试修复)**

```bash
git add -A
git commit -m "test: add PointNet architecture validation script"
```
