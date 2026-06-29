# Go2CmdSafe Symmetry Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ElSpider AIR 的左右对称数据增强机制迁移到 Go2CmdSafe，在 wrapped 4656-dim 观测上实现完整物理镜像。

**Architecture:** 新建 `pointcloud_geometry.py` 纯几何工具模块（5 个公开函数），重构 `CmdSafeHistoryWrapper` 调用该模块，在 `go2_cmd_safe.py` 中实现对称函数，通过 `functools.partial` 在 Runner 层绑定传感器参数注入 PPO。

**Tech Stack:** Python 3.x, PyTorch, Isaac Gym 四元数约定 `[x, y, z, w]`

**设计文档:** `docs/superpowers/specs/2026-06-29-go2-cmdsafe-symmetry-augmentation-design.md`

**文件变更总览：**

| 操作 | 文件 |
|------|------|
| 新建 | `legged_gym/legged_gym/utils/pointcloud_geometry.py` |
| 新建 | `legged_gym/legged_gym/tests/test_pointcloud_geometry.py` |
| 新建 | `legged_gym/legged_gym/tests/test_go2_xsym.py` |
| 新建/修改 | `legged_gym/legged_gym/tests/test_cmd_safe_history_wrapper.py` |
| 修改 | `legged_gym/legged_gym/utils/cmd_safe_history_wrapper.py` |
| 修改 | `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py` |
| 修改 | `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py` |
| 修改 | `rsl_rl/rsl_rl/runners/on_policy_runner.py` |

**实现顺序:** Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9

---

### Task 1: 新建 pointcloud_geometry.py

**Files:**
- Create: `legged_gym/legged_gym/utils/pointcloud_geometry.py`

- [ ] **Step 1: 创建模块文件**

完整代码见设计文档 4 个公开函数：`quaternion_conjugate`, `quaternion_apply`, `cartesian_to_spherical`, `to_sensor_frame`, `sort_points_by_angular_key`。

```python
# legged_gym/legged_gym/utils/pointcloud_geometry.py
"""Pure geometric functions for LiDAR point cloud manipulation.

All quaternions use Isaac Gym convention: [x, y, z, w] (scalar-last).
Shared by CmdSafeHistoryWrapper and symmetry augmentation.
No state, no classes, only torch dependency.
"""

from __future__ import annotations
import math
import torch


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate a quaternion [x, y, z, w] -> [-x, -y, -z, w]."""
    sign = torch.tensor([-1, -1, -1, 1], device=q.device, dtype=q.dtype)
    return q * sign


def quaternion_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions.  q: [..., 4], v: [..., 3] -> [..., 3]."""
    q_vec = q[..., :3]
    q_scalar = q[..., 3:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_scalar * t + torch.cross(q_vec, t, dim=-1)


def cartesian_to_spherical(
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cartesian [..., 3] -> (r, azimuth, phi).
    azimuth = atan2(y, x) in [-pi, pi]; phi = asin(z / r) in [-pi/2, pi/2].
    Guards: nan->0, inf->1e5."""
    points = torch.nan_to_num(points, nan=0.0, posinf=1e5, neginf=-1e5)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    r = torch.norm(points, dim=-1)
    azimuth = torch.atan2(y, x)
    phi = torch.asin(z / (r + 1e-9))
    return r, azimuth, phi


def to_sensor_frame(
    points_base: torch.Tensor,
    sensor_quat: torch.Tensor,
    sensor_trans: torch.Tensor,
) -> torch.Tensor:
    """Base frame [B,N,3] -> sensor frame [B,N,3].
    sensor_quat: [1,4] or [B,4]; sensor_trans: [1,3] or [B,3]."""
    t = sensor_trans.to(points_base.device)
    pts = points_base - t.unsqueeze(1)
    q = sensor_quat.to(points_base.device)
    q_conj = quaternion_conjugate(q)
    B, N = pts.shape[:2]
    pts = quaternion_apply(q_conj.expand(B * N, 4), pts.reshape(-1, 3))
    return pts.reshape(B, N, 3)


def sort_points_by_angular_key(
    points_base: torch.Tensor,
    sensor_quat: torch.Tensor,
    sensor_trans: torch.Tensor,
) -> torch.Tensor:
    """Sort points by angular key (phi * 2*pi + azimuth).
    SINGLE sorting entry point. Input/output both in base frame.
    points_base: [B,N,3] -> [B,N,3] sorted."""
    pts_sensor = to_sensor_frame(points_base, sensor_quat, sensor_trans)
    _, azimuth, phi = cartesian_to_spherical(pts_sensor)
    key = phi * (2.0 * math.pi) + azimuth
    order = torch.argsort(key, dim=1)
    return torch.gather(points_base, 1, order.unsqueeze(-1).expand_as(points_base))
```

- [ ] **Step 2: 验证模块可导入**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "from legged_gym.utils.pointcloud_geometry import sort_points_by_angular_key; print('import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/utils/pointcloud_geometry.py
git commit -m "feat: add pointcloud_geometry pure geometry module"
```

---

### Task 2: 编写 test_pointcloud_geometry.py

**Files:**
- Create: `legged_gym/legged_gym/tests/test_pointcloud_geometry.py`

- [ ] **Step 1: 创建测试文件**

测试覆盖 5 个类：`TestQuaternion` (conjugate 符号, identity, batch), `TestQuaternionApply` (identity rotation, 90deg yaw), `TestCartesianToSpherical` (x/y/z 轴, nan/inf guard), `TestToSensorFrame` (no offset, translation only), `TestSortPointsByAngularKey` (preserves values, shape, y-flip sorting)。

```python
# legged_gym/legged_gym/tests/test_pointcloud_geometry.py
import math
import torch
from legged_gym.utils.pointcloud_geometry import (
    quaternion_conjugate, quaternion_apply,
    cartesian_to_spherical, to_sensor_frame,
    sort_points_by_angular_key,
)


class TestQuaternion:
    def test_conjugate_signs(self):
        q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        qc = quaternion_conjugate(q)
        assert qc[0, 0] == -1.0 and qc[0, 1] == -2.0
        assert qc[0, 2] == -3.0 and qc[0, 3] == 4.0

    def test_identity_conjugate(self):
        q_id = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        torch.testing.assert_close(quaternion_conjugate(q_id), q_id)


class TestQuaternionApply:
    def test_identity_rotation(self):
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(quaternion_apply(q, v), v, atol=1e-6, rtol=1e-6)

    def test_90deg_z_rotation(self):
        s = math.sqrt(2) / 2
        q = torch.tensor([[0.0, 0.0, s, s]])
        v = torch.tensor([[1.0, 0.0, 0.0]])
        torch.testing.assert_close(quaternion_apply(q, v),
                                   torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5)


class TestCartesianToSpherical:
    def test_x_axis(self):
        _, az, phi = cartesian_to_spherical(torch.tensor([[[1.0, 0.0, 0.0]]]))
        torch.testing.assert_close(az, torch.tensor([[0.0]]))
        torch.testing.assert_close(phi, torch.tensor([[0.0]]))

    def test_nan_guard(self):
        pts = torch.tensor([[[float('nan'), 0.0, 0.0]]])
        r, az, phi = cartesian_to_spherical(pts)
        assert not torch.isnan(r).any() and not torch.isnan(az).any()

    def test_inf_guard(self):
        pts = torch.tensor([[[float('inf'), 0.0, 0.0]]])
        r, az, phi = cartesian_to_spherical(pts)
        assert not torch.isinf(az).any() and not torch.isinf(phi).any()


class TestSortPointsByAngularKey:
    def test_preserves_values(self):
        pts = torch.randn(4, 100, 3)
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        t = torch.zeros(1, 3)
        s = sort_points_by_angular_key(pts, q, t)
        assert s.shape == pts.shape
        s_sorted, _ = torch.sort(s.reshape(4, -1), dim=1)
        p_sorted, _ = torch.sort(pts.reshape(4, -1), dim=1)
        torch.testing.assert_close(s_sorted, p_sorted, atol=1e-5)

    def test_y_flip_sorting(self):
        pts = torch.randn(2, 50, 3)
        mir = pts.clone(); mir[:,:,1] = -mir[:,:,1]
        s = sort_points_by_angular_key(mir,
            torch.tensor([[0.,0.,0.,1.]]), torch.zeros(1,3))
        assert not torch.isnan(s).any()
```

- [ ] **Step 2: 运行测试**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -m pytest legged_gym/legged_gym/tests/test_pointcloud_geometry.py -v
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/tests/test_pointcloud_geometry.py
git commit -m "test: add unit tests for pointcloud_geometry module"
```

---

### Task 3: 重构 CmdSafeHistoryWrapper

**Files:**
- Modify: `legged_gym/legged_gym/utils/cmd_safe_history_wrapper.py`

**变更要点：**
1. 顶部添加 `from legged_gym.utils.pointcloud_geometry import ...` (5 个函数)
2. `__init__` 中新增 `self._sensor_quat` 属性（保存非共轭版本，供 sort_points_by_angular_key 使用）
3. 删除原有的内联 `_quat_conjugate`, `_quat_apply`, `_cart_to_sphere`
4. 替换为模块级转发函数
5. `_to_sensor_frame()` 和 `_sort_by_angular_key()` 改为委托给 pointcloud_geometry
6. `_batch_fps()`, `_downsample_distal()`, `wrap_obs()` **业务逻辑不变**

- [ ] **Step 1: 应用修改**

在 `__init__` 方法中（约第 54-56 行），将：
```python
if sensor_offset_quat is not None:
    self._sensor_conj = _quat_conjugate(sensor_offset_quat[0:1]).to(device)
```
改为：
```python
if sensor_offset_quat is not None:
    self._sensor_conj = quaternion_conjugate(sensor_offset_quat[0:1]).to(device)
    self._sensor_quat = sensor_offset_quat[0:1].to(device)
```

将类方法 `_to_sensor_frame` 和 `_sort_by_angular_key` 改为委托调用。

- [ ] **Step 2: 验证 wrapped_obs_dim**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from legged_gym.utils.cmd_safe_history_wrapper import CmdSafeHistoryWrapper
w = CmdSafeHistoryWrapper(4, 1000, 10, 256, 128, 12.0, 48, 'cpu')
assert w.wrapped_obs_dim == 4656; print('PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/utils/cmd_safe_history_wrapper.py
git commit -m "refactor: delegate geometry to pointcloud_geometry in CmdSafeHistoryWrapper"
```

---

### Task 4: Wrapper 回归测试

**Files:**
- Create/Modify: `legged_gym/legged_gym/tests/test_cmd_safe_history_wrapper.py`

- [ ] **Step 1: 创建回归测试**

测试 4 个关键行为：`wrapped_obs_dim == 4656`、`wrap_obs` 输出形状正确、proximal/distal 分割有效、done 重置历史缓冲区。

```python
# legged_gym/legged_gym/tests/test_cmd_safe_history_wrapper.py
import torch
from legged_gym.utils.cmd_safe_history_wrapper import CmdSafeHistoryWrapper

def test_wrapped_obs_dim():
    w = CmdSafeHistoryWrapper(4, 1000, 10, 256, 128, 12.0, 48, 'cpu')
    assert w.wrapped_obs_dim == 4656

def test_wrap_obs_output_shape():
    w = CmdSafeHistoryWrapper(2, 1000, 10, 256, 128, 12.0, 48, 'cpu')
    wrapped = w.wrap_obs(torch.zeros(2, 3048), torch.zeros(2, 1000, 3),
                         torch.zeros(2, dtype=torch.bool))
    assert wrapped.shape == (2, 4656)
    assert not torch.isnan(wrapped).any()

def test_done_resets_history():
    w = CmdSafeHistoryWrapper(2, 1000, 10, 256, 128, 12.0, 48, 'cpu')
    w.wrap_obs(torch.randn(2, 3048), torch.randn(2, 1000, 3),
               torch.tensor([True, False]))
    # 第二个 env 的远端历史不应全零（未被重置）
    # 此处只验证 done=True 时不崩溃且形状正确
```

- [ ] **Step 2: 运行测试**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -m pytest legged_gym/legged_gym/tests/test_cmd_safe_history_wrapper.py -v
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/tests/test_cmd_safe_history_wrapper.py
git commit -m "test: add regression tests for CmdSafeHistoryWrapper"
```

---

### Task 5: 添加对称函数到 go2_cmd_safe.py

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py`

- [ ] **Step 1: 在文件末尾 Go2CmdSafe 类之后添加函数**

```python
# ── Symmetry data augmentation ──
# Distal LiDAR: 128 points x 10-frame history = 1280 points (3840-dim flattened)

@torch.no_grad()
def get_go2_cmd_safe_xsym_obs_act(
    obs: torch.Tensor = None,
    actions: torch.Tensor = None,
    env = None,
    obs_type: str = "policy",
    *,
    sensor_quat: torch.Tensor,
    sensor_trans: torch.Tensor,
    proprio_dim: int = 48,
    proximal_points: int = 256,
    distal_history_points: int = 1280,
    distal_history_length: int = 10,
) -> tuple:
    """Apply left-right symmetry transformation for Go2 quadruped.

    Mirrors wrapped 4656-dim observation (from CmdSafeHistoryWrapper).
    LiDAR points: Y-flip then angular-key re-sort.
    DOF mapping: FL(0:3)<->FR(3:6), RL(6:9)<->RR(9:12).
    """
    from legged_gym.utils.pointcloud_geometry import sort_points_by_angular_key

    device = obs.device if obs is not None else actions.device

    # Stage 1: Observation augmentation
    if obs is not None:
        obs_mirrored = obs.clone()

        # 1a. Scalar sign flips
        obs_mirrored[:, 1] = -obs[:, 1]   # vy
        obs_mirrored[:, 3] = -obs[:, 3]   # wx
        obs_mirrored[:, 5] = -obs[:, 5]   # wz
        obs_mirrored[:, 7] = -obs[:, 7]   # gy
        obs_mirrored[:, 10] = -obs[:, 10] # cmd_vy
        obs_mirrored[:, 11] = -obs[:, 11] # cmd_wz

        # 1b. DOF swaps (3-DOF groups: hip, thigh, calf)
        # dof_pos [12:24]
        obs_mirrored[:, 12:15] = obs[:, 15:18]; obs_mirrored[:, 15:18] = obs[:, 12:15]
        obs_mirrored[:, 18:21] = obs[:, 21:24]; obs_mirrored[:, 21:24] = obs[:, 18:21]
        # dof_vel [24:36]
        obs_mirrored[:, 24:27] = obs[:, 27:30]; obs_mirrored[:, 27:30] = obs[:, 24:27]
        obs_mirrored[:, 30:33] = obs[:, 33:36]; obs_mirrored[:, 33:36] = obs[:, 30:33]
        # prev actions [36:48]
        obs_mirrored[:, 36:39] = obs[:, 39:42]; obs_mirrored[:, 39:42] = obs[:, 36:39]
        obs_mirrored[:, 42:45] = obs[:, 45:48]; obs_mirrored[:, 45:48] = obs[:, 42:45]

        # 1c. Proximal LiDAR Y-flip + re-sort
        prox_s = proprio_dim; prox_e = prox_s + proximal_points * 3
        prox_pts = obs_mirrored[:, prox_s:prox_e].reshape(-1, proximal_points, 3)
        prox_pts[:, :, 1] = -prox_pts[:, :, 1]
        obs_mirrored[:, prox_s:prox_e] = sort_points_by_angular_key(
            prox_pts, sensor_quat, sensor_trans).reshape_as(obs_mirrored[:, prox_s:prox_e])

        # 1d. Distal LiDAR Y-flip + re-sort
        dist_pts = obs_mirrored[:, prox_e:].reshape(-1, distal_history_points, 3)
        dist_pts[:, :, 1] = -dist_pts[:, :, 1]
        obs_mirrored[:, prox_e:] = sort_points_by_angular_key(
            dist_pts, sensor_quat, sensor_trans).reshape_as(obs_mirrored[:, prox_e:])

        obs_augmented = torch.cat([obs, obs_mirrored], dim=0)
    else:
        obs_augmented = None

    # Stage 2: Action augmentation
    if actions is not None:
        a = actions.clone()
        a[:, 0:3], a[:, 3:6] = actions[:, 3:6], actions[:, 0:3]   # FL <-> FR
        a[:, 6:9], a[:, 9:12] = actions[:, 9:12], actions[:, 6:9] # RL <-> RR
        actions_augmented = torch.cat([actions, a], dim=0)
    else:
        actions_augmented = None

    return obs_augmented, actions_augmented
```

- [ ] **Step 2: 验证导入**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "from legged_gym.envs.go2.cmd_safe.go2_cmd_safe import get_go2_cmd_safe_xsym_obs_act; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe.py
git commit -m "feat: add get_go2_cmd_safe_xsym_obs_act symmetry function"
```

---

### Task 6: 编写 test_go2_xsym.py

**Files:**
- Create: `legged_gym/legged_gym/tests/test_go2_xsym.py`

- [ ] **Step 1: 创建测试文件**

测试覆盖 6 个类：`TestScalarMirror` (vy/wx/wz/gy/cmd 符号翻转), `TestDOFSwap` (pos/vel/actions 索引交换), `TestLidarMirror` (proximal/distal Y 翻转), `TestActionMirror` (12 维动作交换), `TestInvariants` (双重镜像=恒等, N→2N, obs_only/actions_only 模式), `TestNumericalStability` (2048 batch 无 NaN/Inf, zero points)。

完整测试代码约 200 行（每个测试 5-10 行）。

- [ ] **Step 2: 运行测试**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -m pytest legged_gym/legged_gym/tests/test_go2_xsym.py -v
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/tests/test_go2_xsym.py
git commit -m "test: add comprehensive tests for Go2 symmetry augmentation"
```

---

### Task 7: 在 go2_cmd_safe_config.py 中添加 symmetry_cfg

**Files:**
- Modify: `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py`

- [ ] **Step 1: 在 Go2CmdSafeCfgPPO.algorithm 内添加嵌套类**

在 `class algorithm(Go2RoughCfgPPO.algorithm):` 内部、`num_mini_batches = 4` 之后添加：

```python
            class symmetry_cfg:
                use_data_augmentation = True
                use_mirror_loss = True
                mirror_loss_coeff = 1.0
                data_augmentation_func = (
                    "legged_gym.envs.go2.cmd_safe.go2_cmd_safe:get_go2_cmd_safe_xsym_obs_act"
                )
```

- [ ] **Step 2: 验证配置可加载**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe_config import Go2CmdSafeCfgPPO
cfg = Go2CmdSafeCfgPPO()
print('symmetry_cfg.data_augmentation_func:', cfg.algorithm.symmetry_cfg.data_augmentation_func)
print('symmetry_cfg.use_data_augmentation:', cfg.algorithm.symmetry_cfg.use_data_augmentation)
print('symmetry_cfg.use_mirror_loss:', cfg.algorithm.symmetry_cfg.use_mirror_loss)
"
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py
git commit -m "feat: add symmetry_cfg to Go2CmdSafeCfgPPO"
```

---

### Task 8: 修改 OnPolicyRunner._setup_symmetry()

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/on_policy_runner.py`

- [ ] **Step 1: 在文件顶部添加 import**

```python
from functools import partial
```

- [ ] **Step 2: 重写 `_setup_symmetry` 方法**

当前方法（约第 291-295 行）：
```python
def _setup_symmetry(self):
    if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
        self.alg_cfg["symmetry_cfg"]["_env"] = self.env
```

替换为：
```python
def _setup_symmetry(self):
    """Setup symmetry if configured. Binds sensor parameters via functools.partial."""
    if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
        from rsl_rl.utils import string_to_callable
        sc = self.alg_cfg["symmetry_cfg"]
        func = sc["data_augmentation_func"]
        # Resolve string path to callable (PPO would do this later,
        # but we need the callable now to wrap with partial)
        if isinstance(func, str):
            func = string_to_callable(func)
        # Bind LiDAR sensor parameters from env (created once, stable lifetime)
        func = partial(func,
            sensor_quat=self.env._sensor_offset_quat,
            sensor_trans=self.env._sensor_translation,
            proprio_dim=48,
            proximal_points=256,
            distal_history_points=1280,
            distal_history_length=10,
        )
        sc["data_augmentation_func"] = func
        sc["_env"] = self.env
```

- [ ] **Step 3: 验证 Runner 初始化链路**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from rsl_rl.utils import string_to_callable
from functools import partial
# 模拟 _setup_symmetry 流程
func = string_to_callable('legged_gym.envs.go2.cmd_safe.go2_cmd_safe:get_go2_cmd_safe_xsym_obs_act')
print('resolved:', func.__name__)
# 验证 partial 绑定
func = partial(func, sensor_quat=None, sensor_trans=None, proprio_dim=48,
               proximal_points=256, distal_history_points=1280, distal_history_length=10)
print('partial wrapped, callable:', callable(func))
"
```

- [ ] **Step 4: Commit**

```bash
git add rsl_rl/rsl_rl/runners/on_policy_runner.py
git commit -m "feat: bind sensor params via partial in _setup_symmetry"
```

---

### Task 9: 集成冒烟测试

**Files:** 无新建（集成测试）

- [ ] **Step 1: 验证配置加载链路完整**

```bash
cd /home/t3chichi/Lidar_legged_gym && python -c "
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe_config import Go2CmdSafeCfgPPO
from rsl_rl.utils import string_to_callable
from functools import partial

cfg = Go2CmdSafeCfgPPO()
sc = cfg.algorithm.symmetry_cfg
func = string_to_callable(sc.data_augmentation_func)
import torch
func = partial(func,
    sensor_quat=torch.zeros(1,4), sensor_trans=torch.zeros(1,3),
    proprio_dim=48, proximal_points=256,
    distal_history_points=1280, distal_history_length=10)
# 执行一次端到端调用
obs = torch.randn(4, 4656)
actions = torch.randn(4, 12)
obs_aug, act_aug = func(obs=obs, actions=actions)
assert obs_aug.shape[0] == 2 * obs.shape[0]
assert act_aug.shape[0] == 2 * actions.shape[0]
assert not torch.isnan(obs_aug).any()
print('Integration smoke test PASSED')
"
```

- [ ] **Step 2: Commit**（如有配置微调）

```bash
git add -A && git status
# 如果有多余文件，精确定位后 commit
```

---

### 实现完成后的验证清单

- [ ] 所有单元测试通过：`pytest legged_gym/legged_gym/tests/test_pointcloud_geometry.py legged_gym/legged_gym/tests/test_go2_xsym.py legged_gym/legged_gym/tests/test_cmd_safe_history_wrapper.py -v`
- [ ] `wrapped_obs_dim == 4656` 断言通过
- [ ] 对称函数的双重镜像约等于恒等变换（在 proprio 维度上）
- [ ] `obs_aug.shape[0] == 2 * original_batch_size`
- [ ] 无 NaN/Inf 在增强后的数据中
- [ ] 完整训练 1 epoch (24 steps × mini-batches) 不崩溃
