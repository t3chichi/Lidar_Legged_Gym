from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint

from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


# --- Quaternion utilities (no isaacgym dependency) ---

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Convert Euler angles (radians) to xyzw quaternion tensor."""
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
    """Conjugate of an xyzw quaternion."""
    return q * torch.tensor([-1, -1, -1, 1], device=q.device)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to vector(s). q: (4,), v: (..., 3)."""
    q_vec = q[:3]  # (3,)
    q_scalar = q[3]  # scalar
    q_vec_exp = q_vec.expand(*v.shape[:-1], 3)  # (..., 3)
    t = 2.0 * torch.cross(q_vec_exp, v, dim=-1)
    return v + q_scalar * t + torch.cross(q_vec_exp, t, dim=-1)


class PDRiskNetActorCritic(nn.Module):
    """PD-RiskNet actor-critic.

    Observation layout:
    - first proprio_obs_dim dims: proprio/command/action history-free state
    - remaining dims: lidar history points, flattened as
      [history_length, num_lidar_points, 3]
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        perception_enabled: bool = True,
        history_length: int = 1,
        proximal_history_length: int = 1,
        distal_history_length: int = 10,
        num_lidar_points: int = 1024,
        proximal_points: int = 512,
        distal_points: int = 512,
        split_theta_deg: float = 0.0,
        proximal_feature_dim: int = 187,
        distal_feature_dim: int = 64,
        proprio_obs_dim: int = 48,
        privileged_height_dim: int = 187,
        privileged_critic_dim: int | None = None,
        privileged_supervision_coef: float = 0.5,
        sensor_offset_rpy: list | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "PDRiskNetActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.perception_enabled = perception_enabled
        self.history_length = int(history_length)
        self.proximal_history_length = int(proximal_history_length)
        self.distal_history_length = int(distal_history_length)
        self.num_lidar_points = int(num_lidar_points)
        self.proximal_points = int(proximal_points)
        self.distal_points = int(distal_points)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.split_theta = float(split_theta_deg) * math.pi / 180.0
        self.proximal_feature_dim = int(proximal_feature_dim)
        self.distal_feature_dim = int(distal_feature_dim)

        # Sensor offset quaternion conjugate: transforms base-frame points back
        # to sensor frame so that the proximal/distal split uses the original
        # spherical-grid elevation angles.  Falls back to identity (base-frame
        # split) when no offset is specified.
        self._sensor_conj: torch.Tensor | None = None
        if sensor_offset_rpy is not None and any(v != 0.0 for v in sensor_offset_rpy):
            self._sensor_conj = _quat_conjugate(_euler_to_quat(*sensor_offset_rpy))
        self.privileged_height_dim = int(privileged_height_dim)
        self.privileged_critic_dim = int(privileged_critic_dim) if privileged_critic_dim is not None else self.privileged_height_dim
        self.privileged_supervision_coef = float(privileged_supervision_coef)
        self.num_actions = num_actions

        lidar_expected_dim = self.num_lidar_points * 3   # 仅单帧点云输入
        if num_actor_obs < self.proprio_obs_dim + lidar_expected_dim:
            raise ValueError(
                f"PDRiskNetActorCritic expects at least {self.proprio_obs_dim + lidar_expected_dim} actor obs dims, got {num_actor_obs}"
            )

        act_fn = resolve_nn_activation(activation)

        self.proximal_point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            act_fn,
            nn.Linear(64, 64),
            act_fn,
        )
        self.distal_point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            act_fn,
            nn.Linear(64, 64),
            act_fn,
        )

        self.proximal_gru = nn.GRU(input_size=64, hidden_size=self.proximal_feature_dim, batch_first=True)
        self.distal_spatial_gru = nn.GRU(input_size=64, hidden_size=self.distal_feature_dim, batch_first=True)
        self.proximal_memory_a = Memory(
            self.proximal_feature_dim,
            type="gru",
            num_layers=1,
            hidden_size=self.proximal_feature_dim,
        )
        self.distal_memory_a = Memory(
            self.distal_feature_dim,
            type="gru",
            num_layers=1,
            hidden_size=self.distal_feature_dim,
        )

        actor_input_dim = self.proprio_obs_dim + self.proximal_feature_dim + self.distal_feature_dim

        self.actor = self._build_mlp(actor_input_dim, actor_hidden_dims, num_actions, act_fn)
        # Critic 与 Actor 共享同一个 299 维感知表征
        self.critic = self._build_mlp(actor_input_dim, critic_hidden_dims, 1, act_fn)

        # Train-time proximal branch supervision head.
        self.height_supervisor = nn.Linear(self.proximal_feature_dim, self.privileged_height_dim)

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

        self._cached_proximal_feature = None
        self._cached_actor_latent = None
        # self._critic_hidden_state = None
        self._warned_missing_prox_hidden = False
        self._warned_missing_dist_hidden = False
        self._sampling_plan_ready = False
        self.register_buffer("_proximal_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_distal_sorted_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_distal_bin_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_distal_bin_counts", torch.empty(0, dtype=torch.float32), persistent=False)

    def _build_mlp(self, in_dim, hidden_dims, out_dim, activation):
        layers = [nn.Linear(in_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        self.proximal_memory_a.reset(dones)
        self.distal_memory_a.reset(dones)
        self._cached_actor_latent = None
    
    def get_hidden_states(self):
        prox_hidden = self.proximal_memory_a.hidden_states
        dist_hidden = self.distal_memory_a.hidden_states

        if prox_hidden is None and dist_hidden is None:
            return (None, None)
        # 若其中一个为 None，则用另一个的 batch/device 补全
        if prox_hidden is None:
            prox_hidden = torch.zeros((1, dist_hidden.shape[1], self.proximal_feature_dim), device=dist_hidden.device)
        elif dist_hidden is None:
            dist_hidden = torch.zeros((1, prox_hidden.shape[1], self.distal_feature_dim), device=prox_hidden.device)

        actor_hidden_states = (prox_hidden, dist_hidden)
        critic_hidden_states = (prox_hidden, dist_hidden)

        return actor_hidden_states, critic_hidden_states

    def _split_actor_hidden_states(self, hidden_states):
        if hidden_states is None:
            return None, None
        if isinstance(hidden_states, (list, tuple)):
            if len(hidden_states) >= 2:
                return hidden_states[0], hidden_states[1]
            if len(hidden_states) == 1:
                return hidden_states[0], None
        return hidden_states, None

    def _warn_missing_hidden_once(self, branch: str):
        if branch == "prox" and (not self._warned_missing_prox_hidden):
            print("[PDRiskNetActorCritic] Missing proximal hidden states in update; using zero initialization.")
            self._warned_missing_prox_hidden = True
        if branch == "dist" and (not self._warned_missing_dist_hidden):
            print("[PDRiskNetActorCritic] Missing distal hidden states in update; using zero initialization.")
            self._warned_missing_dist_hidden = True

    def _init_actor_hidden_like(self, frame_feat: torch.Tensor, feat_dim: int) -> torch.Tensor:
        if frame_feat.dim() == 4:
            batch_size = frame_feat.shape[1]
        elif frame_feat.dim() >= 2:
            batch_size = frame_feat.shape[0]
        else:
            raise ValueError(f"Unsupported frame feature rank for hidden initialization: {frame_feat.dim()}")
        return torch.zeros((1, batch_size, feat_dim), device=frame_feat.device, dtype=frame_feat.dtype)

    def _frame_window_to_seq(self, frame_feat: torch.Tensor):
        """Convert frame/window feature tensor into (seq,batch,feat) recurrent input.

        Accepted layouts:
        - (B, F) -> (1, B, F), grouped=False
        - (B, T_win, F) -> (T_win, B, F), grouped=True with per-step window T_win
        - (T_env, B, T_win, F) -> (T_env*T_win, B, F), grouped=True with env-step grouping
        """
        if frame_feat.dim() == 2:
            b, feat_dim = frame_feat.shape
            seq = frame_feat.unsqueeze(0)
            return seq, False, 1, b, feat_dim
        if frame_feat.dim() == 3:
            b, t_win, feat_dim = frame_feat.shape
            seq = frame_feat.permute(1, 0, 2)
            return seq, True, t_win, b, feat_dim
        if frame_feat.dim() == 4:
            t_env, b, t_win, feat_dim = frame_feat.shape
            seq = frame_feat.permute(0, 2, 1, 3).reshape(t_env * t_win, b, feat_dim)
            return seq, True, t_win, b, feat_dim
        raise ValueError(f"Unsupported frame feature rank: {frame_feat.dim()}")

    def _collapse_window_output(self, seq_out: torch.Tensor, grouped: bool, window_len: int, batch_size: int):
        if not grouped:
            return seq_out[-1]
        if seq_out.shape[0] % window_len != 0:
            raise ValueError(
                f"Sequence length {seq_out.shape[0]} is not divisible by window length {window_len}."
            )
        env_steps = seq_out.shape[0] // window_len
        feat_dim = seq_out.shape[-1]
        feat = seq_out.reshape(env_steps, window_len, batch_size, feat_dim)[:, -1, :, :]
        if env_steps == 1:
            return feat.squeeze(0)
        return feat

    def _run_actor_memory(
        self,
        memory: Memory,
        frame_feat: torch.Tensor,
        masks: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        feat_dim: int,
        branch_name: str,
    ) -> torch.Tensor:
        # 训练序列模式特判
        if masks is not None and frame_feat.dim() == 3:
            seq_in = frame_feat                 # (T, B, F)
            grouped = False
            window_len = 1
            batch_size = frame_feat.shape[1]
        # 推理分支单步序列特判：形状为 (1, B, F) 且 masks=None
        elif masks is None and frame_feat.dim() == 3 and frame_feat.shape[0] == 1:
            seq_in = frame_feat                 # (1, B, F)
            grouped = False
            window_len = 1
            batch_size = frame_feat.shape[1]
        else:
            seq_in, grouped, window_len, batch_size, _ = self._frame_window_to_seq(frame_feat)

        if masks is not None:
            if hidden_states is None:
                self._warn_missing_hidden_once(branch_name)
                hidden_states = self._init_actor_hidden_like(frame_feat, feat_dim)
            seq_out, _ = memory.rnn(seq_in, hidden_states)   # (T, B, F)
            # 训练分支直接返回序列，不再压缩时间维，由外部处理 unpad
            return seq_out

        # 推理分支
        seq_out, memory.hidden_states = memory.rnn(seq_in, memory.hidden_states)
        return self._collapse_window_output(seq_out, grouped, window_len, batch_size)

    def _split_obs(self, observations: torch.Tensor):
        # 观测布局：[proprio (48)] + [single frame point cloud (N*3)]
         # 处理空张量情况（例如 PPO 初始化阶段）
        if observations.numel() == 0:
            if observations.dim() == 2:
                return (torch.empty(0, self.proprio_obs_dim, device=observations.device),
                        torch.empty(0, self.num_lidar_points, 3, device=observations.device))
            elif observations.dim() == 3:
                t, b, _ = observations.shape
                return (torch.empty(t, b, self.proprio_obs_dim, device=observations.device),
                        torch.empty(t, b, self.num_lidar_points, 3, device=observations.device))
            else:
                raise ValueError(f"Unexpected observations dim: {observations.dim()}")
        if observations.dim() == 2:
            proprio = observations[:, :self.proprio_obs_dim]
            lidar_flat = observations[:, self.proprio_obs_dim:]
            # 单帧点云形状 (batch, N, 3)
            lidar_frame = lidar_flat.reshape(-1, self.num_lidar_points, 3)
            return proprio, lidar_frame
        if observations.dim() == 3:
            t, b, _ = observations.shape
            obs_flat = observations.reshape(t * b, -1)
            proprio = obs_flat[:, :self.proprio_obs_dim].reshape(t, b, self.proprio_obs_dim)
            lidar_flat = obs_flat[:, self.proprio_obs_dim:]
            # 训练序列单帧形状 (T, B, N, 3)
            lidar_frame = lidar_flat.reshape(t, b, self.num_lidar_points, 3)
            return proprio, lidar_frame

    def _fps_indices_single(self, points: torch.Tensor, k: int) -> torch.Tensor:
        n = points.shape[0]
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=points.device)
        if k >= n:
            return torch.arange(n, device=points.device, dtype=torch.long)

        selected = torch.empty(k, dtype=torch.long, device=points.device)
        distances = torch.full((n,), float("inf"), device=points.device)
        farthest = torch.argmax(torch.sum(points * points, dim=-1))
        for i in range(k):
            selected[i] = farthest
            centroid = points[farthest].unsqueeze(0)
            dist = torch.sum((points - centroid) ** 2, dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances)
        return selected

    def _build_sampling_plan(self, lidar_hist: torch.Tensor):
        # Build once from a representative scan: keeps runtime overhead low.
        ref_idx = torch.randint(0, lidar_hist.shape[0], (1,), device=lidar_hist.device).item()
        ref_points = lidar_hist[ref_idx, -1]

        # Rotate base-frame points back to sensor frame for correct
        # proximal/distal split.  The sensor may be mounted with a large
        # offset (e.g. upside-down), which would make a base-frame split
        # map most points to the wrong side of theta_threshold.
        if self._sensor_conj is not None:
            q = self._sensor_conj.to(device=lidar_hist.device, dtype=lidar_hist.dtype)
            ref_points = _quat_apply(q, ref_points)

        x = ref_points[:, 0]
        y = ref_points[:, 1]
        z = ref_points[:, 2]
        theta = torch.atan2(z, torch.sqrt(x * x + y * y + 1.0e-8))
        phi = torch.atan2(y, x)

        prox_candidates = torch.nonzero(theta >= self.split_theta, as_tuple=False).squeeze(-1)
        dist_candidates = torch.nonzero(theta < self.split_theta, as_tuple=False).squeeze(-1)

        if prox_candidates.numel() == 0:
            prox_candidates = torch.arange(self.num_lidar_points, device=lidar_hist.device, dtype=torch.long)
        if dist_candidates.numel() == 0:
            dist_candidates = torch.arange(self.num_lidar_points, device=lidar_hist.device, dtype=torch.long)

        prox_k = min(self.proximal_points, int(prox_candidates.numel()))
        prox_local = self._fps_indices_single(ref_points[prox_candidates], prox_k)
        self._proximal_indices = prox_candidates[prox_local]

        dist_key = theta[dist_candidates] * (2.0 * math.pi) + phi[dist_candidates]
        order = torch.argsort(dist_key)
        dist_sorted = dist_candidates[order]
        dist_m = int(dist_sorted.numel())
        dist_k = min(self.distal_points, dist_m)

        self._distal_sorted_indices = dist_sorted
        if dist_k > 0:
            # Evenly partition sorted distal points, then average each partition.
            bin_ids = torch.div(torch.arange(dist_m, device=lidar_hist.device) * dist_k, dist_m, rounding_mode="floor")
            counts = torch.bincount(bin_ids, minlength=dist_k).to(torch.float32)
        else:
            bin_ids = torch.empty(0, dtype=torch.long, device=lidar_hist.device)
            counts = torch.empty(0, dtype=torch.float32, device=lidar_hist.device)

        self._distal_bin_ids = bin_ids
        self._distal_bin_counts = counts
        self._sampling_plan_ready = True

    def _sample_proximal_fps(self, lidar_hist: torch.Tensor) -> torch.Tensor:
        prox_idx = self._proximal_indices.to(lidar_hist.device)
        return torch.index_select(lidar_hist, dim=2, index=prox_idx)

    def _sample_distal_avg(self, lidar_hist: torch.Tensor) -> torch.Tensor:
        dist_idx = self._distal_sorted_indices.to(lidar_hist.device)
        dist_points = torch.index_select(lidar_hist, dim=2, index=dist_idx)
        b, t, m, _ = dist_points.shape
        k = int(self._distal_bin_counts.numel())
        if k == 0:
            return dist_points

        bin_ids = self._distal_bin_ids.to(lidar_hist.device)
        out = torch.zeros((b, t, k, 3), device=lidar_hist.device, dtype=lidar_hist.dtype)
        scatter_idx = bin_ids.view(1, 1, m, 1).expand(b, t, m, 3)
        out.scatter_add_(2, scatter_idx, dist_points)

        counts = self._distal_bin_counts.to(lidar_hist.device).clamp(min=1.0).view(1, 1, k, 1)
        return out / counts

    def _compute_sampled_sorted_points_frame(self, lidar_points_frame: torch.Tensor):
        if lidar_points_frame.dim() != 3:
            raise ValueError(f"Expected lidar_points_frame shape (B, N, 3), got rank {lidar_points_frame.dim()}")

        lidar_frame_hist = lidar_points_frame.unsqueeze(1)
        if (not self._sampling_plan_ready) or (self._proximal_indices.numel() == 0) or (
            int(self._proximal_indices.max().item()) >= lidar_points_frame.shape[1]
        ):
            self._build_sampling_plan(lidar_frame_hist)

        prox_points = self._sample_proximal_fps(lidar_frame_hist).squeeze(1)
        dist_points = self._sample_distal_avg(lidar_frame_hist).squeeze(1)

        prox_points = self._sort_by_spherical(prox_points.unsqueeze(1)).squeeze(1)
        dist_points = self._sort_by_spherical(dist_points.unsqueeze(1)).squeeze(1)
        return prox_points, dist_points


    def _build_replay_frame_features(self, lidar_points_seq: torch.Tensor, masks: torch.Tensor | None):
        if lidar_points_seq.dim() != 4:
            raise ValueError(f"Expected lidar_points_seq shape (T, B, N, 3), got rank {lidar_points_seq.dim()}")

        seq_len, batch_size, num_points, _ = lidar_points_seq.shape

        if masks is not None:
            frame_any_valid = masks.any(dim=1)
            valid_frames = frame_any_valid.nonzero(as_tuple=False)
            effective_len = valid_frames[-1].item() + 1 if len(valid_frames) > 0 else seq_len
        else:
            effective_len = seq_len

        prox_feat_seq = []
        dist_feat_seq = []

        for t in range(effective_len):
            # 获取当前帧点云 [B, N, 3]
            points_t = lidar_points_seq[t]

            # 对当前帧进行采样、排序，得到近端和远端点云 [B, prox_points, 3] 和 [B, dist_points, 3]
            prox_points_t, dist_points_t = self._compute_sampled_sorted_points_frame(points_t)

            # 空间编码：需要将单帧点云扩展一个时间维度，因为现有编码函数期望输入 [B, T, points, 3]
            # 这里 T=1
            prox_points_t = prox_points_t.unsqueeze(1)  # [B, 1, prox_points, 3]
            dist_points_t = dist_points_t.unsqueeze(1)  # [B, 1, dist_points, 3]

            # 编码得到空间特征 [B, 1, feature_dim]，然后去掉时间维
            prox_feat_t = self._encode_proximal_points_chunked(prox_points_t).squeeze(1)  # [B, proximal_feature_dim]
            dist_feat_t = self._encode_distal_points_chunked(dist_points_t).squeeze(1)    # [B, distal_feature_dim]

            prox_feat_seq.append(prox_feat_t)
            dist_feat_seq.append(dist_feat_t)

        # 堆叠为序列 [T, B, F]；若跳过了纯 padding 尾部则补零以维持 seq_len
        prox_feat_seq = torch.stack(prox_feat_seq, dim=0)
        dist_feat_seq = torch.stack(dist_feat_seq, dim=0)
        if effective_len < seq_len:
            pad_len = seq_len - effective_len
            z_prox = torch.zeros(pad_len, batch_size, self.proximal_feature_dim,
                                 device=prox_feat_seq.device, dtype=prox_feat_seq.dtype)
            z_dist = torch.zeros(pad_len, batch_size, self.distal_feature_dim,
                                 device=dist_feat_seq.device, dtype=dist_feat_seq.dtype)
            prox_feat_seq = torch.cat([prox_feat_seq, z_prox], dim=0)
            dist_feat_seq = torch.cat([dist_feat_seq, z_dist], dim=0)

        return prox_feat_seq, dist_feat_seq

    def _encode_proximal_points_chunked(self, prox_points: torch.Tensor) -> torch.Tensor:
        flat_batch_size, t_prox, pn, _ = prox_points.shape
        prox_frame_feat = torch.empty(
            (flat_batch_size, t_prox, self.proximal_feature_dim),
            device=prox_points.device,
            dtype=prox_points.dtype,
        )
        # Chunk flat batch to keep GRU/MLP activations within small GPU memory budgets.
        chunk_size = 64
        for start in range(0, flat_batch_size, chunk_size):
            end = min(start + chunk_size, flat_batch_size)
            chunk = prox_points[start:end]
            chunk_enc = self.proximal_point_encoder(chunk.reshape((end - start) * t_prox * pn, 3)).reshape(
                end - start, t_prox, pn, -1
            )
            chunk_seq = chunk_enc.reshape((end - start) * t_prox, pn, -1)
            # 训练时使用 checkpoint 节省显存，推理时直接前向（且无需梯度）
            if self.training:
                _, chunk_h = checkpoint(self.proximal_gru, chunk_seq, use_reentrant=True)
            else:
                with torch.no_grad():
                    _, chunk_h = self.proximal_gru(chunk_seq)
            prox_frame_feat[start:end] = chunk_h.squeeze(0).reshape(end - start, t_prox, -1)
        return prox_frame_feat

    def _encode_distal_points_chunked(self, dist_points: torch.Tensor) -> torch.Tensor:
        flat_batch_size, t_dist, dn, _ = dist_points.shape
        dist_frame_feat = torch.empty(
            (flat_batch_size, t_dist, self.distal_feature_dim),
            device=dist_points.device,
            dtype=dist_points.dtype,
        )
        chunk_size = 64
        for start in range(0, flat_batch_size, chunk_size):
            end = min(start + chunk_size, flat_batch_size)
            chunk = dist_points[start:end]
            chunk_enc = self.distal_point_encoder(chunk.reshape((end - start) * t_dist * dn, 3)).reshape(
                end - start, t_dist, dn, -1
            )
            chunk_seq = chunk_enc.reshape((end - start) * t_dist, dn, -1)
            if self.training:
                _, chunk_h = checkpoint(self.distal_spatial_gru, chunk_seq, use_reentrant=True)
            else:
                with torch.no_grad():
                    _, chunk_h = self.distal_spatial_gru(chunk_seq)
            dist_frame_feat[start:end] = chunk_h.squeeze(0).reshape(end - start, t_dist, -1)
        return dist_frame_feat

    def _sort_by_spherical(self, points):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        theta = torch.atan2(z, torch.sqrt(x * x + y * y + 1.0e-8))
        phi = torch.atan2(y, x)
        order = torch.argsort(theta * (2.0 * math.pi) + phi, dim=-1)
        order_exp = order.unsqueeze(-1).expand_as(points)
        return torch.gather(points, dim=2, index=order_exp)

    def _encode_perception(self, observations: torch.Tensor, masks: torch.Tensor | None = None):
        proprio, lidar_frame = self._split_obs(observations)

        if observations.dim() == 2:
            # 推理/采样：仅处理当前帧，不维护点云缓存
            prox_points_t, dist_points_t = self._compute_sampled_sorted_points_frame(lidar_frame)
            prox_points_t = prox_points_t.unsqueeze(1)  # [B, 1, prox_points, 3]
            dist_points_t = dist_points_t.unsqueeze(1)  # [B, 1, dist_points, 3]
            prox_feat_t = self._encode_proximal_points_chunked(prox_points_t).squeeze(1)  # [B, proximal_feature_dim]
            dist_feat_t = self._encode_distal_points_chunked(dist_points_t).squeeze(1)    # [B, distal_feature_dim]
            prox_frame_feat = prox_feat_t.unsqueeze(0)   # [1, B, F]
            dist_frame_feat = dist_feat_t.unsqueeze(0)   # [1, B, F]
            return proprio, prox_frame_feat, dist_frame_feat

        elif observations.dim() == 3:
            # 训练更新：lidar_frame 形状为 (T, B, N, 3)
            # 返回 [T, B, F] 的特征序列
            prox_frame_feat, dist_frame_feat = self._build_replay_frame_features(lidar_frame, masks)
            return proprio, prox_frame_feat, dist_frame_feat

        else:
            raise ValueError(f"Unsupported observations rank: {observations.dim()}")

    def _build_actor_latent(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    ):
        proprio, prox_frame_feat, dist_frame_feat = self._encode_perception(observations, masks=masks)
        prox_hidden_states, dist_hidden_states = self._split_actor_hidden_states(hidden_states)

        if masks is not None:
            # 训练分支：保留时间维度，之后统一展平
            prox_feat = self._run_actor_memory(
                self.proximal_memory_a,
                prox_frame_feat,
                masks,
                prox_hidden_states,
                self.proximal_feature_dim,
                "prox",
            )  # (T, B, F)
            dist_feat = self._run_actor_memory(
                self.distal_memory_a,
                dist_frame_feat,
                masks,
                dist_hidden_states,
                self.distal_feature_dim,
                "dist",
            )  # (T, B, F)

            # 展平时间-批次维度
            proprio = unpad_trajectories(proprio, masks)      # (T*B, proprio_dim)
            prox_feat = unpad_trajectories(prox_feat, masks)  # (T*B, F)
            dist_feat = unpad_trajectories(dist_feat, masks)  # (T*B, F)

        else:
            # 推理分支：单步输入，输出二维
            prox_feat = self._run_actor_memory(
                self.proximal_memory_a,
                prox_frame_feat,
                masks=None,
                hidden_states=None,
                feat_dim=self.proximal_feature_dim,
                branch_name="prox",
            )  # (B, F)
            dist_feat = self._run_actor_memory(
                self.distal_memory_a,
                dist_frame_feat,
                masks=None,
                hidden_states=None,
                feat_dim=self.distal_feature_dim,
                branch_name="dist",
            )  # (B, F)
            # proprio 此时已经是二维 (B, proprio_dim)

        actor_latent = torch.cat((proprio, prox_feat, dist_feat), dim=-1)

        self._cached_proximal_feature = prox_feat
        self._cached_actor_latent = actor_latent
        return actor_latent

    def update_distribution(self, observations, masks=None, hidden_states=None):
        actor_latent = self._build_actor_latent(observations, masks=masks, hidden_states=hidden_states)
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
        # In Memory.forward with masks=None, hidden state is updated internally each step.
        actor_latent = self._build_actor_latent(observations)
        return self.actor(actor_latent)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        if masks is not None:
            # 训练路径：优先复用 act() 中 _build_actor_latent() 刚写入的缓存。
            # PPO.update() 总是先调 act(obs_batch) 再调 evaluate(obs_batch)，
            # 且传入同一个 obs_batch tensor，缓存一定是最新的。
            if self._cached_actor_latent is not None:
                return self.critic(self._cached_actor_latent)
            # 缓存未命中时走原路径（理论上不会触发，保留作为安全网）
            actor_latent = self._build_actor_latent(
                critic_observations, masks=masks, hidden_states=hidden_states
            )
            return self.critic(actor_latent)
        else:
            if self._cached_actor_latent is not None:
                return self.critic(self._cached_actor_latent)
            # compute_returns 冷启动：空间编码 + 只读 GRU（不推进内部隐藏状态，
            # 避免下一轮 rollout 的 act() 对同一观测重复处理导致时间偏移）
            expected_dim = self.proprio_obs_dim + self.num_lidar_points * 3
            if critic_observations.shape[-1] < expected_dim:
                raise ValueError(
                    f"evaluate() cold-start expects >= {expected_dim}-dim actor observations, "
                    f"got {critic_observations.shape[-1]}. "
                    f"Call act() first to populate the cached actor latent, "
                    f"or pass full actor observations (proprio + LiDAR)."
                )
            proprio, prox_frame_feat, dist_frame_feat = self._encode_perception(critic_observations, masks=None)
            if self.proximal_memory_a.hidden_states is not None:
                prox_out, _ = self.proximal_memory_a.rnn(prox_frame_feat, self.proximal_memory_a.hidden_states)
                prox_feat = prox_out[-1]
            else:
                prox_feat = prox_frame_feat[-1]
            if self.distal_memory_a.hidden_states is not None:
                dist_out, _ = self.distal_memory_a.rnn(dist_frame_feat, self.distal_memory_a.hidden_states)
                dist_feat = dist_out[-1]
            else:
                dist_feat = dist_frame_feat[-1]
            return self.critic(torch.cat((proprio, prox_feat, dist_feat), dim=-1))

    def get_auxiliary_loss(self, privileged_heights: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        if self._cached_proximal_feature is None:
            return torch.zeros((), device=privileged_heights.device)

        if self._cached_proximal_feature.numel() == 0:
            return torch.zeros((), device=privileged_heights.device)

        # 统一 unpad：训练时 _cached_proximal_feature 已被 _build_actor_latent unpad（保持 3D）
        if masks is not None and privileged_heights.dim() == 3:
            privileged_heights = unpad_trajectories(privileged_heights, masks)

        # 取序列最后一个时间步的特征进行监督（如果是训练分支的话），或者直接使用当前特征（如果是推理分支的话）
        if self._cached_proximal_feature.dim() == 3:
            # shape: [batch, seq_len, feat_dim] -> 取最后一步
            prox_feat = self._cached_proximal_feature[:, -1, :]
        else:
            prox_feat = self._cached_proximal_feature

        pred = self.height_supervisor(prox_feat)  # [batch, height_dim]

        # 处理特权观测，同样取最后一步
        if privileged_heights.dim() == 3:
            # [batch, seq_len, critic_dim] -> 取最后一步
            priv_obs = privileged_heights[:, -1, :]
        else:
            priv_obs = privileged_heights

        # 提取高度目标
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

        # 末维对齐（防止预测维度意外偏差）
        if pred.shape[-1] != height_target.shape[-1]:
            min_dim = min(pred.shape[-1], height_target.shape[-1])
            pred = pred[..., :min_dim]
            height_target = height_target[..., :min_dim]

        return self.privileged_supervision_coef * torch.mean(torch.square(pred - height_target))

    def load_state_dict(self, state_dict, strict=True):
        # 兼容旧 checkpoint：critic 第一层权重形状从 [*, 235] 变为 [*, 299]
        if 'critic.0.weight' in state_dict:
            expected = self.critic[0].weight.shape
            actual = state_dict['critic.0.weight'].shape
            if expected != actual:
                print(f"[PDRiskNetActorCritic] Critic weight shape mismatch "
                      f"(checkpoint {list(actual)} -> model {list(expected)}). "
                      f"Critic will be randomly initialized.")
                keys_to_remove = [k for k in state_dict if k.startswith('critic.')]
                for k in keys_to_remove:
                    del state_dict[k]
        return super().load_state_dict(state_dict, strict=False)
