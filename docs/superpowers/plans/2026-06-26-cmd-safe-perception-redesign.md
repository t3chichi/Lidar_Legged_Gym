# CmdSafe 感知架构重构 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `go2_cmd_safe` 训练新建感知架构：零初始化 GRU + 10 帧远端滚动窗口 + GPU 批量 FPS。

**Architecture:** `CmdSafeHistoryWrapper` 在环境外维护远端 10 帧窗口并做近端/远端分离和采样；`CmdSafeActorCritic` 消费预处理后的点云，双 GRU 零初始化，保留 `height_supervisor`。

**Tech Stack:** PyTorch, torch_fpsample, Isaac Gym, rsl_rl

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `legged_gym/legged_gym/utils/cmd_safe_history_wrapper.py` | 新建 |
| `rsl_rl/rsl_rl/modules/cmd_safe_actor_critic.py` | 新建 |
| `rsl_rl/rsl_rl/modules/__init__.py` | 修改 |
| `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py` | 修改 |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` | 修改 |
| `legged_gym/setup.py` | 修改 |

---

### Task 1: 新建 CmdSafeHistoryWrapper

**Files:**
- Create: `legged_gym/legged_gym/utils/cmd_safe_history_wrapper.py`

- [ ] **Step 1: 写入完整实现**

```python
"""CmdSafeHistoryWrapper — LiDAR 点云预处理与远端帧历史维护。

每步将单帧原始点云分离为近端/远端，经 FPS/平均下采样后拼接输出。
近端：torch_fpsample 动态 FPS → 球坐标排序 → 单帧 256 点。
远端：组合键排序 → 均匀采样 128 点 → 10 帧滚动窗口 → 全局排序 → 1280 点。
"""

from __future__ import annotations

import math
import torch

try:
    import torch_fpsample
except ImportError:
    torch_fpsample = None


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([-1, -1, -1, 1], device=q.device, dtype=q.dtype)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., :3]
    q_scalar = q[..., 3:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_scalar * t + torch.cross(q_vec, t, dim=-1)


class CmdSafeHistoryWrapper:
    """在环境与策略之间执行 LiDAR 点云预处理和历史维护。

    近端（phi <= threshold）：FPS → 256 点 → 球坐标排序。
    远端（phi > threshold）：组合键排序 → 均匀采样 128 点 → 10 帧滚动窗口
                              → 不满时广播首帧 → 全局排序 → 1280 点。
    """

    def __init__(
        self,
        num_envs: int,
        num_lidar_points: int,
        distal_history_length: int,
        proximal_points: int,
        distal_points: int,
        phi_threshold_deg: float,
        proprio_dim: int,
        device: torch.device,
        sensor_offset_quat: torch.Tensor | None = None,
        sensor_translation: torch.Tensor | None = None,
    ):
        if torch_fpsample is None:
            raise ImportError(
                "torch_fpsample is required. Install with: pip install torch-fpsample"
            )

        self.num_envs = num_envs
        self.num_lidar_points = num_lidar_points
        self.distal_history_length = distal_history_length
        self.proximal_points = proximal_points
        self.distal_points = distal_points
        self.phi_threshold_rad = math.radians(phi_threshold_deg)
        self.proprio_dim = proprio_dim
        self.device = device

        # Sensor offset (to transform base-frame points → sensor frame)
        if sensor_offset_quat is not None:
            self._sensor_conj = _quat_conjugate(sensor_offset_quat[0:1]).to(device)
        else:
            self._sensor_conj = None
        if sensor_translation is not None:
            self._sensor_t = sensor_translation[0:1].to(device)
        else:
            self._sensor_t = torch.zeros(1, 3, device=device)

        # Distal rolling window: (num_envs, distal_history_length, distal_points, 3)
        self._distal_window = torch.zeros(
            num_envs, distal_history_length, distal_points, 3,
            device=device, dtype=torch.float,
        )
        self._distal_frame_count = torch.zeros(num_envs, device=device, dtype=torch.long)

    @property
    def wrapped_obs_dim(self) -> int:
        return self.proprio_dim + self.proximal_points * 3 + self.distal_history_length * self.distal_points * 3

    def _cart_to_sphere(self, points_sensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cartesian (..., 3) → spherical [r, azimuth, phi]."""
        x, y, z = points_sensor[..., 0], points_sensor[..., 1], points_sensor[..., 2]
        r = torch.norm(points_sensor, dim=-1)
        azimuth = torch.atan2(y, x)
        phi = torch.asin(z / (r + 1e-9))
        return r, azimuth, phi

    def _sphere_to_cart(self, r: torch.Tensor, azimuth: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Spherical [r, azimuth, phi] → Cartesian (..., 3)."""
        x = r * torch.cos(phi) * torch.cos(azimuth)
        y = r * torch.cos(phi) * torch.sin(azimuth)
        z = r * torch.sin(phi)
        return torch.stack((x, y, z), dim=-1)

    def _sort_by_angular_key(self, points: torch.Tensor) -> torch.Tensor:
        """Sort points by combined angular key: phi * 2π + azimuth."""
        _, azimuth, phi = self._cart_to_sphere(points)
        key = phi * (2.0 * math.pi) + azimuth  # (B, N)
        order = torch.argsort(key, dim=1)
        return torch.gather(points, 1, order.unsqueeze(-1).expand_as(points))

    def _batch_fps(self, points: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
        """GPU-batched FPS via torch_fpsample.

        Args:
            points: (B, N_max, 3) padded points.
            mask:   (B, N_max) True where valid.
            k:      number of points to sample.

        Returns:
            (B, k, 3) sampled points, zero-padded where count < k.
        """
        B, N_max, _ = points.shape
        device = points.device

        counts = mask.sum(dim=1)  # (B,)
        if (counts == 0).all():
            return torch.zeros(B, k, 3, device=device)

        max_count = int(counts.max().item())

        # Pack valid points into padded tensor via scatter
        cumsum = mask.cumsum(dim=1) - 1  # (B, N_max)
        positions = torch.where(mask, cumsum, torch.zeros_like(cumsum))
        padded = torch.zeros(B, max_count, 3, device=device)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N_max)
        valid_batch = batch_idx[mask]
        valid_pos = positions[mask]
        valid_pts = points[mask]
        padded[valid_batch, valid_pos] = valid_pts

        # Fill padding with first valid point to keep distances finite
        fill_mask = torch.arange(max_count, device=device).unsqueeze(0) >= counts.unsqueeze(1)
        if fill_mask.any():
            first = padded[:, 0:1, :].expand(-1, max_count, -1)
            padded[fill_mask] = first[fill_mask]

        k_eff = min(k, max_count)
        sampled, sampled_idx = torch_fpsample.sample(padded, k_eff)  # (B, k_eff, 3), (B, k_eff)

        # Mask out samples that mapped to padding
        invalid = sampled_idx >= counts.unsqueeze(1)
        if invalid.any():
            sampled = sampled.masked_fill(invalid.unsqueeze(-1), 0.0)

        if k_eff < k:
            out = torch.zeros(B, k, 3, device=device)
            out[:, :k_eff] = sampled
            return out
        return sampled

    def _downsample_distal(self, points_sensor: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
        """Downsample distal points: sort by angular key, then uniformly pick k points.

        Args:
            points_sensor: (B, N, 3) distal candidate points in sensor frame.
            mask:          (B, N) True where valid distal.
            k:             number of output points.

        Returns:
            (B, k, 3) sorted downsampled points.
        """
        B, N, _ = points_sensor.shape
        device = points_sensor.device

        _, azimuth, phi = self._cart_to_sphere(points_sensor)

        # Sort by combined key
        sort_key = torch.where(
            mask,
            phi * (2.0 * math.pi) + azimuth,
            torch.full_like(phi, float("inf")),
        )
        sorted_idx = torch.argsort(sort_key, dim=1)  # (B, N)
        # Gather sorted points
        idx_exp = sorted_idx.unsqueeze(-1).expand(-1, -1, 3)
        sorted_pts = torch.gather(points_sensor, 1, idx_exp)

        counts = mask.sum(dim=1).clamp(min=1)  # (B,)
        k_eff = min(k, int(sorted_pts.shape[1]))

        # Uniformly spaced indices
        pos = torch.arange(k_eff, device=device).unsqueeze(0).expand(B, -1)  # (B, k_eff)
        denom = max(k_eff - 1, 1)
        uniform_pos = torch.round(
            pos.float() * (counts.unsqueeze(1).float() - 1.0) / float(denom)
        ).long()
        uniform_pos = torch.minimum(uniform_pos, (counts - 1).unsqueeze(1))

        selected = torch.gather(sorted_pts, 1, uniform_pos.unsqueeze(-1).expand(-1, -1, 3))

        if k_eff < k:
            out = torch.zeros(B, k, 3, device=device)
            out[:, :k_eff] = selected
            # Zero-out positions >= counts
            keep = pos < counts.unsqueeze(1)
            out = out * keep.unsqueeze(-1)
            return out
        return selected

    def _to_sensor_frame(self, points_base: torch.Tensor, env_ids_slice: slice | None = None) -> torch.Tensor:
        """Transform base-frame points to sensor frame."""
        pts = points_base - self._sensor_t
        if self._sensor_conj is not None:
            q = self._sensor_conj
            pts = _quat_apply(q.expand(pts.shape[0] * pts.shape[1], 4), pts.reshape(-1, 3))
            pts = pts.reshape(points_base.shape)
        return pts

    def wrap_obs(
        self,
        obs_buf: torch.Tensor,
        lidar_points_base: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Transform raw obs_buf into wrapped observation.

        Args:
            obs_buf:            (B, proprio_dim + N*3) raw observation from env.
            lidar_points_base:  (B, N, 3) LiDAR points in base frame.
            dones:              (B,) episode termination flags.

        Returns:
            wrapped_obs: (B, proprio_dim + proximal_points*3 + distal_history*points*3)
        """
        B = obs_buf.shape[0]

        # Split proprio and lidar from obs_buf
        proprio = obs_buf[:, :self.proprio_dim]                     # (B, 48)
        lidar_raw = obs_buf[:, self.proprio_dim:].reshape(B, -1, 3)  # (B, N, 3)

        # Sensor-frame points for theta split
        pts_sensor = self._to_sensor_frame(lidar_raw)  # (B, N, 3)
        _, _, phi = self._cart_to_sphere(pts_sensor)

        # Masks
        valid_mask = (lidar_raw.abs().sum(dim=-1) > 0)  # crude valid: non-zero
        proximal_mask = (phi <= self.phi_threshold_rad) & valid_mask
        distal_mask = (phi > self.phi_threshold_rad) & valid_mask

        # ── Proximal: FPS → sort ──
        prox_k = self.proximal_points
        prox_fps = self._batch_fps(lidar_raw, proximal_mask, prox_k)  # (B, prox_k, 3)
        prox_sorted = self._sort_by_angular_key(prox_fps)              # (B, prox_k, 3)

        # ── Distal: downsample → rolling window ──
        dist_k = self.distal_points
        dist_down = self._downsample_distal(pts_sensor, distal_mask, dist_k)  # (B, dist_k, 3)
        dist_sorted = self._sort_by_angular_key(dist_down)                     # (B, dist_k, 3)

        # Push to rolling window (circular buffer)
        write_idx = self._distal_frame_count % self.distal_history_length
        self._distal_window[torch.arange(B, device=self.device), write_idx] = dist_sorted
        self._distal_frame_count += 1

        # Broadcast first frame for unfilled windows
        fill_mask = self._distal_frame_count < self.distal_history_length  # (B,)
        if fill_mask.any():
            fill_envs = fill_mask.nonzero(as_tuple=False).squeeze(-1)
            first_frame = self._distal_window[fill_envs, 0:1, :, :]  # (n, 1, dist_k, 3)
            self._distal_window[fill_envs] = first_frame.expand(-1, self.distal_history_length, -1, -1)

        # Concatenate window → global sort
        distal_cat = self._distal_window.reshape(B, self.distal_history_length * dist_k, 3)  # (B, 1280, 3)
        distal_sorted = self._sort_by_angular_key(distal_cat)  # global sort

        # Reset done envs
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._distal_window[done_ids] = 0.0
            self._distal_frame_count[done_ids] = 0

        # Assemble
        wrapped = torch.cat([
            proprio,                           # (B, 48)
            prox_sorted.reshape(B, -1),        # (B, 256*3)
            distal_sorted.reshape(B, -1),      # (B, 1280*3)
        ], dim=-1)

        return wrapped
```

- [ ] **Step 2: 提交**

```bash
git add legged_gym/legged_gym/utils/cmd_safe_history_wrapper.py
git commit -m "feat: add CmdSafeHistoryWrapper for LiDAR point cloud preprocessing"
```

---

### Task 2: 新建 CmdSafeActorCritic 网络

**Files:**
- Create: `rsl_rl/rsl_rl/modules/cmd_safe_actor_critic.py`

- [ ] **Step 1: 写入完整网络实现**

```python
"""CmdSafeActorCritic — 零初始化双 GRU 感知 Actor-Critic。

近端 GRU: input=3, hidden=187, 每帧零初始化, seq_len=256。
远端 GRU: input=3, hidden=64, 每帧零初始化, seq_len=1280。
height_supervisor: Linear(187, 187) 近端特征 → 高度图, 辅助 MSE 监督。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


# --- Quaternion utilities (无 isaacgym 依赖) ---

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return torch.tensor([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([-1, -1, -1, 1], device=q.device)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[:3]
    q_scalar = q[3]
    q_vec_exp = q_vec.expand(*v.shape[:-1], 3)
    t = 2.0 * torch.cross(q_vec_exp, v, dim=-1)
    return v + q_scalar * t + torch.cross(q_vec_exp, t, dim=-1)


class CmdSafeActorCritic(nn.Module):
    """CmdSafe Actor-Critic with dual zero-init GRU perception.

    Observation layout (from CmdSafeHistoryWrapper):
      - proprio (48 dims)
      - proximal points (256 × 3 = 768 dims, sorted by spherical key)
      - distal points (1280 × 3 = 3840 dims, 10-frame concatenated + globally sorted)
      Total: 48 + 768 + 3840 = 4656

    Architecture:
      Proximal: (B, 256, 3) → GRU(3→187, zero-init) → h_n (B, 187)
      Distal:   (B, 1280, 3) → GRU(3→64, zero-init)  → h_n (B, 64)
      Actor:    (B, 48+187+64=299) → MLP → 12
      Critic:   (B, 299) → MLP → 1
      Aux:      Linear(187, 187) → MSE with privileged heights
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [1024, 512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        proximal_points: int = 256,
        distal_history_length: int = 10,
        distal_points: int = 128,
        proximal_feature_dim: int = 187,
        distal_feature_dim: int = 64,
        proprio_obs_dim: int = 48,
        privileged_height_dim: int = 187,
        privileged_critic_dim: int = 235,
        privileged_supervision_coef: float = 1.0,
        sensor_offset_rpy: list[float] | None = None,
        sensor_offset_pos: list[float] | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "CmdSafeActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.proximal_points = int(proximal_points)
        self.distal_history_length = int(distal_history_length)
        self.distal_points = int(distal_points)
        self.proximal_feature_dim = int(proximal_feature_dim)
        self.distal_feature_dim = int(distal_feature_dim)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.privileged_height_dim = int(privileged_height_dim)
        self.privileged_critic_dim = int(privileged_critic_dim)
        self.privileged_supervision_coef = float(privileged_supervision_coef)
        self.num_actions = num_actions

        # Sensor offset (for sorting in sensor frame)
        self._sensor_conj: torch.Tensor | None = None
        if sensor_offset_rpy is not None and any(v != 0.0 for v in sensor_offset_rpy):
            self._sensor_conj = _quat_conjugate(_euler_to_quat(*sensor_offset_rpy))
        if sensor_offset_pos is not None and any(v != 0.0 for v in sensor_offset_pos):
            sensor_t = torch.tensor(sensor_offset_pos, dtype=torch.float32)
        else:
            sensor_t = torch.zeros(3, dtype=torch.float32)
        self.register_buffer("_sensor_translation", sensor_t, persistent=False)

        # Validate input dimensions
        expected_obs = self.proprio_obs_dim + self.proximal_points * 3 + self.distal_history_length * self.distal_points * 3
        if num_actor_obs < expected_obs:
            raise ValueError(
                f"CmdSafeActorCritic expects at least {expected_obs} actor obs dims, got {num_actor_obs}"
            )

        act_fn = resolve_nn_activation(activation)

        # ── GRU encoders (no PointNet, raw xyz input) ──
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

        # ── Height supervisor (kept from original PDRiskNet) ──
        self.height_supervisor = nn.Linear(self.proximal_feature_dim, self.privileged_height_dim)

        # ── Actor / Critic heads ──
        actor_input_dim = self.proprio_obs_dim + self.proximal_feature_dim + self.distal_feature_dim
        self.actor = self._build_mlp(actor_input_dim, actor_hidden_dims, num_actions, act_fn)
        self.critic = self._build_mlp(actor_input_dim, critic_hidden_dims, 1, act_fn)

        # ── Noise parameterisation ──
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        # ── Cache for auxiliary loss ──
        self._cached_proximal_feature: torch.Tensor | None = None
        self._cached_actor_latent: torch.Tensor | None = None

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation: nn.Module) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    # ── Properties ──

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # ── Observation splitting ──

    def _split_obs(self, observations: torch.Tensor):
        """Split wrapped observation into proprio, proximal, distal.

        Returns:
            proprio:  (..., 48)
            proximal: (..., 256, 3)
            distal:   (..., 1280, 3)
        """
        if observations.numel() == 0:
            if observations.dim() == 2:
                return (
                    torch.empty(0, self.proprio_obs_dim, device=observations.device),
                    torch.empty(0, self.proximal_points, 3, device=observations.device),
                    torch.empty(0, self.distal_history_length * self.distal_points, 3, device=observations.device),
                )
            elif observations.dim() == 3:
                t, b, _ = observations.shape
                return (
                    torch.empty(t, b, self.proprio_obs_dim, device=observations.device),
                    torch.empty(t, b, self.proximal_points, 3, device=observations.device),
                    torch.empty(t, b, self.distal_history_length * self.distal_points, 3, device=observations.device),
                )
            raise ValueError(f"Unexpected observations dim: {observations.dim()}")

        prox_len = self.proximal_points * 3
        dist_len = self.distal_history_length * self.distal_points * 3

        if observations.dim() == 2:
            proprio = observations[:, :self.proprio_obs_dim]
            prox_flat = observations[:, self.proprio_obs_dim:self.proprio_obs_dim + prox_len]
            dist_flat = observations[:, self.proprio_obs_dim + prox_len:self.proprio_obs_dim + prox_len + dist_len]
            return proprio, prox_flat.reshape(-1, self.proximal_points, 3), dist_flat.reshape(-1, self.distal_history_length * self.distal_points, 3)

        # 3D: (T, B, dim)
        t, b, _ = observations.shape
        obs_flat = observations.reshape(t * b, -1)
        proprio = obs_flat[:, :self.proprio_obs_dim].reshape(t, b, self.proprio_obs_dim)
        prox_flat = obs_flat[:, self.proprio_obs_dim:self.proprio_obs_dim + prox_len]
        dist_flat = obs_flat[:, self.proprio_obs_dim + prox_len:self.proprio_obs_dim + prox_len + dist_len]
        return (
            proprio,
            prox_flat.reshape(t, b, self.proximal_points, 3),
            dist_flat.reshape(t, b, self.distal_history_length * self.distal_points, 3),
        )

    # ── Spherical sorting ──

    def _sort_by_spherical(self, points: torch.Tensor) -> torch.Tensor:
        """Sort points by combined angular key in sensor frame.

        Args:
            points: (B, P, 3) in base frame.

        Returns:
            Sorted points in same shape.
        """
        # Transform to sensor frame
        t = self._sensor_translation.to(device=points.device, dtype=points.dtype)
        pts = points - t.unsqueeze(1)
        if self._sensor_conj is not None:
            q = self._sensor_conj.to(device=points.device, dtype=points.dtype)
            pts = _quat_apply(q.unsqueeze(0).expand(pts.shape[0] * pts.shape[1], 4), pts.reshape(-1, 3))
            pts = pts.reshape(points.shape)

        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        r = torch.norm(pts, dim=-1)
        azimuth = torch.atan2(y, x)
        phi = torch.asin(z / (r + 1e-9))
        key = phi * (2.0 * math.pi) + azimuth

        order = torch.argsort(key, dim=1)
        return torch.gather(points, 1, order.unsqueeze(-1).expand_as(points))

    # ── GRU encoding (chunked, zero-init) ──

    def _encode_proximal_chunked(self, prox_points: torch.Tensor) -> torch.Tensor:
        """Encode proximal points through zero-init GRU.

        Args:
            prox_points: (B, T, 256, 3), where T is 1 for inference.

        Returns:
            (B, T, 187)
        """
        B, T_prox, P, _ = prox_points.shape
        out = torch.empty(B, T_prox, self.proximal_feature_dim,
                          device=prox_points.device, dtype=prox_points.dtype)
        chunk_size = 128
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = prox_points[start:end]  # (c, T, P, 3)
            c = end - start
            chunk_seq = chunk.reshape(c * T_prox, P, 3)  # (c*T, P, 3) — raw xyz
            _, h = self.proximal_gru(chunk_seq)           # zero-init, h: (1, c*T, 187)
            out[start:end] = h.squeeze(0).reshape(c, T_prox, -1)
        return out

    def _encode_distal_chunked(self, dist_points: torch.Tensor) -> torch.Tensor:
        """Encode distal points through zero-init GRU.

        Args:
            dist_points: (B, T, 1280, 3), where T is 1 for inference.

        Returns:
            (B, T, 64)
        """
        B, T_dist, D, _ = dist_points.shape
        out = torch.empty(B, T_dist, self.distal_feature_dim,
                          device=dist_points.device, dtype=dist_points.dtype)
        chunk_size = 128
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = dist_points[start:end]  # (c, T, D, 3)
            c = end - start
            chunk_seq = chunk.reshape(c * T_dist, D, 3)  # (c*T, D, 3) — raw xyz
            _, h = self.distal_gru(chunk_seq)             # zero-init, h: (1, c*T, 64)
            out[start:end] = h.squeeze(0).reshape(c, T_dist, -1)
        return out

    # ── Actor latent construction ──

    def _build_actor_latent(self, observations: torch.Tensor, masks: torch.Tensor | None = None):
        """Build the concatenated actor latent from wrapped observations.

        Args:
            observations: (B, 4656) or (T, B, 4656).
            masks:        (T, B) or None.

        Returns:
            actor_latent: (B, 299) or (T_h, 299) after unpad.
        """
        proprio, proximal, distal = self._split_obs(observations)

        if masks is not None:
            # Training: proximal (T, B, 256, 3), distal (T, B, 1280, 3)
            # Flatten to (T*B, ...) for zero-init GRU (each frame independent)
            T_seq, B = proprio.shape[:2]
            prox_flat = proximal.reshape(T_seq * B, self.proximal_points, 3)
            _, h_prox = self.proximal_gru(prox_flat)  # (1, T*B, 187)
            prox_feat = h_prox.squeeze(0).reshape(T_seq, B, self.proximal_feature_dim)  # (T, B, 187)

            dist_flat = distal.reshape(T_seq * B, self.distal_history_length * self.distal_points, 3)
            _, h_dist = self.distal_gru(dist_flat)  # (1, T*B, 64)
            dist_feat = h_dist.squeeze(0).reshape(T_seq, B, self.distal_feature_dim)  # (T, B, 64)

            # Unpad
            proprio = unpad_trajectories(proprio, masks)
            prox_feat = unpad_trajectories(prox_feat, masks)
            dist_feat = unpad_trajectories(dist_feat, masks)
        else:
            # Inference: (B, 4656)
            # Proximal: sort first
            prox_sorted = self._sort_by_spherical(proximal)  # (B, 256, 3)
            prox_feat_t = self._encode_proximal_chunked(prox_sorted.unsqueeze(1))  # (B, 1, 187)
            prox_feat = prox_feat_t.squeeze(1)  # (B, 187)

            # Distal: already globally sorted by wrapper
            dist_feat_t = self._encode_distal_chunked(distal.unsqueeze(1))  # (B, 1, 64)
            dist_feat = dist_feat_t.squeeze(1)  # (B, 64)

        actor_latent = torch.cat((proprio, prox_feat, dist_feat), dim=-1)

        self._cached_proximal_feature = prox_feat
        self._cached_actor_latent = actor_latent
        return actor_latent

    # ── Public API ──

    def update_distribution(self, observations, masks=None, hidden_states=None):
        actor_latent = self._build_actor_latent(observations, masks=masks)
        mean = self.actor(actor_latent)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        self.update_distribution(observations, masks=masks, hidden_states=hidden_states)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_latent = self._build_actor_latent(observations)
        return self.actor(actor_latent)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        if masks is not None:
            if self._cached_actor_latent is not None:
                return self.critic(self._cached_actor_latent)
            actor_latent = self._build_actor_latent(critic_observations, masks=masks)
            return self.critic(actor_latent)
        else:
            if self._cached_actor_latent is not None:
                return self.critic(self._cached_actor_latent)
            expected_dim = self.proprio_obs_dim + self.proximal_points * 3 + self.distal_history_length * self.distal_points * 3
            if critic_observations.shape[-1] < expected_dim:
                raise ValueError(
                    f"evaluate() cold-start expects >= {expected_dim}-dim actor observations, "
                    f"got {critic_observations.shape[-1]}."
                )
            actor_latent = self._build_actor_latent(critic_observations)
            return self.critic(actor_latent)

    def get_auxiliary_loss(self, privileged_heights: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        if self._cached_proximal_feature is None:
            return torch.zeros((), device=privileged_heights.device)
        if self._cached_proximal_feature.numel() == 0:
            return torch.zeros((), device=privileged_heights.device)

        if masks is not None and privileged_heights.dim() == 3:
            privileged_heights = unpad_trajectories(privileged_heights, masks)

        if self._cached_proximal_feature.dim() == 3:
            prox_feat = self._cached_proximal_feature[:, -1, :]
        else:
            prox_feat = self._cached_proximal_feature

        pred = self.height_supervisor(prox_feat)

        if privileged_heights.dim() == 3:
            priv_obs = privileged_heights[:, -1, :]
        else:
            priv_obs = privileged_heights

        actual_dim = priv_obs.shape[-1]
        if actual_dim == self.privileged_height_dim:
            height_target = priv_obs
        elif actual_dim == self.privileged_critic_dim:
            height_target = priv_obs[..., -self.privileged_height_dim:]
        else:
            if actual_dim >= self.privileged_height_dim:
                height_target = priv_obs[..., -self.privileged_height_dim:]
            else:
                print(f"[WARNING] Aux loss skip: actual dim {actual_dim}, expected {self.privileged_height_dim}")
                return torch.zeros((), device=privileged_heights.device)

        if pred.shape[-1] != height_target.shape[-1]:
            min_dim = min(pred.shape[-1], height_target.shape[-1])
            pred = pred[..., :min_dim]
            height_target = height_target[..., :min_dim]

        return self.privileged_supervision_coef * torch.mean(torch.square(pred - height_target))

    def load_state_dict(self, state_dict, strict=True):
        # Compatibility: skip if critic weight shape mismatches
        if 'critic.0.weight' in state_dict:
            expected = self.critic[0].weight.shape
            actual = state_dict['critic.0.weight'].shape
            if expected != actual:
                print(f"[CmdSafeActorCritic] Critic weight shape mismatch "
                      f"(checkpoint {list(actual)} -> model {list(expected)}). "
                      f"Critic will be randomly initialized.")
                keys_to_remove = [k for k in state_dict if k.startswith('critic.')]
                for k in keys_to_remove:
                    del state_dict[k]

        if 'proximal_gru.weight_ih_l0' in state_dict:
            expected = self.proximal_gru.weight_ih_l0.shape
            actual = state_dict['proximal_gru.weight_ih_l0'].shape
            if expected != actual:
                print("[CmdSafeActorCritic] Architecture changed "
                      f"(proximal_gru input_size: checkpoint={actual[1]}, model={expected[1]}). "
                      "Perception modules will be randomly initialized.")
                prefix_blacklist = (
                    'proximal_gru.', 'distal_gru.',
                    'proximal_pointnet.', 'distal_pointnet.',
                )
                keys_to_remove = [k for k in state_dict if k.startswith(prefix_blacklist)]
                for k in keys_to_remove:
                    del state_dict[k]

        return super().load_state_dict(state_dict, strict=False)
```

- [ ] **Step 2: 提交**

```bash
git add rsl_rl/rsl_rl/modules/cmd_safe_actor_critic.py
git commit -m "feat: add CmdSafeActorCritic with dual zero-init GRU perception"
```

---

### Task 3: 修改模块导出和训练配置

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/__init__.py`
- Modify: `legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py`

- [ ] **Step 1: 导出 CmdSafeActorCritic**

在 `rsl_rl/rsl_rl/modules/__init__.py` 添加导出：

```python
# 在现有 import 之后添加
from .cmd_safe_actor_critic import CmdSafeActorCritic

# 在 __all__ 中添加
    "CmdSafeActorCritic",
```

完整 diff：

```diff
 from .pd_risknet_actor_critic import PDRiskNetActorCritic
+from .cmd_safe_actor_critic import CmdSafeActorCritic

 __all__ = [
     ...
     "PDRiskNetActorCritic",
+    "CmdSafeActorCritic",
 ]
```

- [ ] **Step 2: 更新训练配置 `go2_cmd_safe_config.py`**

修改 `Go2CmdSafeCfg` 和 `Go2CmdSafeCfgPPO`：

```python
# go2_cmd_safe_config.py

# 修改 env.num_observations:
class env(Go2RoughCfg.env):
    num_observations = PD_PROPRIO_DIM + PD_PROXIMAL_POINTS * 3 + DIST_HISTORY_LENGTH * PD_DISTAL_POINTS * 3
    # = 48 + 256*3 + 10*128*3 = 48 + 768 + 3840 = 4656
    ...

# 修改 runner.policy_class_name:
class runner(Go2RoughCfgPPO.runner):
    policy_class_name = "CmdSafeActorCritic"
    ...
```

完整修改后的两个类（只展示变动部分）：

```python
class Go2CmdSafeCfg(Go2RoughCfg):
    class env(Go2RoughCfg.env):
        num_observations = 4656
        num_privileged_obs = PD_PRIV_CRITIC_DIM
        ...

class Go2CmdSafeCfgPPO(Go2RoughCfgPPO):
    class policy(Go2RoughCfgPPO.policy):
        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        proximal_points = 256
        distal_history_length = 10
        distal_points = 128
        proximal_feature_dim = 187
        distal_feature_dim = 64
        proprio_obs_dim = 48
        privileged_height_dim = 187
        privileged_critic_dim = 235
        privileged_supervision_coef = 1.0
        sensor_offset_rpy = [0.0, 0.0, 0.0]
        sensor_offset_pos = [0.0, 0.0, 0.0]
        # 移除 perception_enabled, history_length, proximal_history_length,
        # distal_history_length, num_lidar_points, split_theta_deg (不再需要)

    class runner(Go2RoughCfgPPO.runner):
        policy_class_name = "CmdSafeActorCritic"
        ...

    class algorithm(Go2RoughCfgPPO.algorithm):
        amp_enabled = True
        ...
```

- [ ] **Step 3: 提交**

```bash
git add rsl_rl/rsl_rl/modules/__init__.py legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py
git commit -m "feat: wire CmdSafeActorCritic into cmd_safe config and module exports"
```

---

### Task 4: 在 OnPolicyRunner 中集成 HistoryWrapper

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/on_policy_runner.py`

- [ ] **Step 1: 添加 HistoryWrapper 初始化逻辑**

在 `OnPolicyRunner.__init__` 中，`_initialize_old_interface()` 之后添加 HistoryWrapper 创建：

```python
# 在 _initialize_old_interface() 调用之后
self.history_wrapper = None
if self.use_old_interface and hasattr(self.env, 'cfg'):
    env_cfg = self.env.cfg
    if hasattr(env_cfg, 'pd_risknet') and getattr(env_cfg.pd_risknet, 'enabled', False):
        from legged_gym.utils.cmd_safe_history_wrapper import CmdSafeHistoryWrapper
        pd_cfg = env_cfg.pd_risknet
        ppo_policy_cfg = self.policy_cfg
        self.history_wrapper = CmdSafeHistoryWrapper(
            num_envs=self.env.num_envs,
            num_lidar_points=int(pd_cfg.num_lidar_points),
            distal_history_length=int(ppo_policy_cfg.get("distal_history_length", 10)),
            proximal_points=int(ppo_policy_cfg.get("proximal_points", 256)),
            distal_points=int(ppo_policy_cfg.get("distal_points", 128)),
            phi_threshold_deg=float(pd_cfg.split_theta_deg),
            proprio_dim=48,
            device=self.device,
        )
```

- [ ] **Step 2: 在 learn() 循环中插入 wrapper 调用**

在 `learn()` 的 rollout 阶段，`alg.act` 之前插入包装逻辑：

```python
# 在 rollout 循环中，env.step() 获取 obs 之后，alg.act() 之前:

# 现有代码:
obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
privileged_obs = privileged_obs.to(self.device)

# 插入 HistoryWrapper:
if self.history_wrapper is not None:
    obs = self.history_wrapper.wrap_obs(
        obs, self.env.lidar_points_base, dones,
    )
```

需要在 rollout 循环中的两个路径都插入（old interface 和 new interface 分支），且要在 `obs.to(self.device)` 之后（wrapper 内部操作已在正确 device 上）。

具体位置（在 `learn()` 中找到两处 `obs.to(self.device)` 之后）：

```python
# Old interface path (~line 423):
obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
privileged_obs = privileged_obs.to(self.device)

# >>> INSERT HERE <<<
if self.history_wrapper is not None:
    obs = self.history_wrapper.wrap_obs(obs, self.env.lidar_points_base, dones)

# New interface path (同样的插入点)
```

- [ ] **Step 3: 处理 learn() 开头的初始 obs 获取**

`learn()` 开头调用 `_get_observations()` 获取初始 obs。这发生在 rollout 循环之前。对此初始 obs 也要应用 wrapper：

```python
# 在 learn() 中，初始 obs 获取之后:
obs, privileged_obs = self._get_observations()
obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)

# >>> INSERT HERE <<<
if self.history_wrapper is not None:
    # 初始 obs 的 lidar_points_base 从 env 获取; dones 全部 False
    init_dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
    obs = self.history_wrapper.wrap_obs(obs, self.env.lidar_points_base, init_dones)
```

- [ ] **Step 4: 更新 num_obs 以匹配 wrapped obs dim**

Wrapper 改变了 obs 维度。`_setup_observations` 中读取 `num_obs` 时，如果 wrapper 已创建，应使用 wrapped 维度：

```python
# 在 _setup_observations 末尾，return 之前:
if self.history_wrapper is not None:
    num_obs = self.history_wrapper.wrapped_obs_dim
```

由于 wrapper 在 `__init__` 末尾创建，而 `_setup_observations` 在 `__init__` 早期调用，需要调整顺序或后期修正 `num_obs`。

最简单的方案：在 `__init__` 中，policy 创建使用原始 `num_obs`（env 报告的），然后在 wrapper 创建后用 `wrapped_obs_dim` 更新 storage 的 obs shape。但 storage 已在 `_initialize_storage` 中初始化。

更简洁方案：**在创建 wrapper 后重新初始化 storage**。在 `__init__` 末尾：

```python
# 在 __init__ 末尾，创建 wrapper 之后:
if self.history_wrapper is not None:
    # 用 wrapped obs dim 重建 storage
    wrapped_num_obs = self.history_wrapper.wrapped_obs_dim
    self.alg.init_storage(
        self.training_type,
        self.env.num_envs,
        self.num_steps_per_env,
        [wrapped_num_obs],
        [num_privileged_obs],
        [self.env.num_actions],
    )
```

同时 policy 在创建时传入的 `num_obs` 也需要是 wrapped 维度。所以最优顺序是：

1. 先确定 wrapped_num_obs
2. 用 wrapped_num_obs 创建 policy
3. 用 wrapped_num_obs 创建 storage

需要重构 `__init__` 中的顺序。在创建 policy 之前计算 wrapper 是否存在及 wrapped dim。

```python
# 在 _parse_config 和 _setup_observations 之后:
wrapped_num_obs = num_obs
# 预判是否需要 wrapper
if self.use_old_interface and hasattr(self.env, 'cfg'):
    env_cfg = self.env.cfg
    if hasattr(env_cfg, 'pd_risknet') and getattr(env_cfg.pd_risknet, 'enabled', False):
        from legged_gym.utils.cmd_safe_history_wrapper import CmdSafeHistoryWrapper
        pd_cfg = env_cfg.pd_risknet
        ppo_policy_cfg = self.policy_cfg
        wrapped_num_obs = (
            48
            + int(ppo_policy_cfg.get("proximal_points", 256)) * 3
            + int(ppo_policy_cfg.get("distal_history_length", 10))
            * int(ppo_policy_cfg.get("distal_points", 128)) * 3
        )

# 然后用 wrapped_num_obs 创建 policy 和 storage
# ...
```

- [ ] **Step 5: 提交**

```bash
git add rsl_rl/rsl_rl/runners/on_policy_runner.py
git commit -m "feat: integrate CmdSafeHistoryWrapper into OnPolicyRunner"
```

---

### Task 5: 添加 torch_fpsample 依赖

**Files:**
- Modify: `legged_gym/setup.py`

- [ ] **Step 1: 添加依赖声明**

在 `legged_gym/setup.py` 的 `install_requires` 中添加：

```python
install_requires=[
    ...
    "torch-fpsample",
],
```

- [ ] **Step 2: 安装依赖**

```bash
conda activate li_leggym && pip install torch-fpsample
```

- [ ] **Step 3: 提交**

```bash
git add legged_gym/setup.py
git commit -m "chore: add torch-fpsample dependency"
```

---

### Task 6: 端到端验证

- [ ] **Step 1: 验证导入**

```bash
cd /home/t3chichi/Lidar_legged_gym && conda activate li_leggym && python -c "
from rsl_rl.modules import CmdSafeActorCritic
from legged_gym.utils.cmd_safe_history_wrapper import CmdSafeHistoryWrapper
print('Imports OK')
print('CmdSafeActorCritic.is_recurrent:', CmdSafeActorCritic.is_recurrent)
"
```

预期：`Imports OK` + `is_recurrent: False`

- [ ] **Step 2: 验证网络 forward**

```bash
python -c "
import torch
from rsl_rl.modules import CmdSafeActorCritic

# 模拟 wrapped_obs 维度: 48 + 256*3 + 1280*3 = 4656
B = 4
obs = torch.randn(B, 4656)
priv = torch.randn(B, 235)

net = CmdSafeActorCritic(
    num_actor_obs=4656,
    num_critic_obs=235,
    num_actions=12,
)
net.eval()

with torch.no_grad():
    actions = net.act_inference(obs)
print('Actions shape:', actions.shape)
print('OK: forward pass succeeds')
"
```

预期：`Actions shape: torch.Size([4, 12])`

- [ ] **Step 3: 验证辅助损失**

```python
# 前向传播产生缓存
net.update_distribution(obs)
# 取辅助损失
loss = net.get_auxiliary_loss(priv)
print('Aux loss:', loss.item())
print('OK: aux loss computed')
```

预期：非零浮点数

- [ ] **Step 4: 验证 HistoryWrapper 维度**

```python
import torch
from legged_gym.utils.cmd_safe_history_wrapper import CmdSafeHistoryWrapper

B, N = 4, 1000
wrapper = CmdSafeHistoryWrapper(
    num_envs=B, num_lidar_points=N,
    distal_history_length=10, proximal_points=256, distal_points=128,
    phi_threshold_deg=20.0, proprio_dim=48, device='cpu',
)

obs_buf = torch.randn(B, 48 + N * 3)
lidar = torch.randn(B, N, 3)
dones = torch.zeros(B, dtype=torch.bool)

wrapped = wrapper.wrap_obs(obs_buf, lidar, dones)
print('Wrapped obs shape:', wrapped.shape)
assert wrapped.shape == (B, 4656), f'Expected (4, 4656), got {wrapped.shape}'
print('OK: wrapper dimension correct')
```

预期：`(4, 4656)` 通过

- [ ] **Step 5: 验证 is_recurrent=False 路径**

```bash
python -c "
from rsl_rl.modules import CmdSafeActorCritic
# PPO update 走 mini_batch_generator（非 recurrent）
print('is_recurrent:', CmdSafeActorCritic.is_recurrent)
print('OK: PPO will use mini_batch_generator (not recurrent)')
"
```

预期：`is_recurrent: False`

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "test: verify CmdSafeActorCritic and HistoryWrapper end-to-end"
```
