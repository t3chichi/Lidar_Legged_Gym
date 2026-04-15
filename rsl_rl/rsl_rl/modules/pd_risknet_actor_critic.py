from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


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
        privileged_supervision_coef: float = 1.0,
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
        self.privileged_height_dim = int(privileged_height_dim)
        self.privileged_critic_dim = int(privileged_critic_dim) if privileged_critic_dim is not None else self.privileged_height_dim
        self.privileged_supervision_coef = float(privileged_supervision_coef)
        self.num_actions = num_actions

        lidar_expected_dim = self.history_length * self.num_lidar_points * 3
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
        critic_input_dim = actor_input_dim if num_critic_obs != self.privileged_critic_dim else self.privileged_critic_dim

        self.actor = self._build_mlp(actor_input_dim, actor_hidden_dims, num_actions, act_fn)
        self.critic = self._build_mlp(critic_input_dim, critic_hidden_dims, 1, act_fn)

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

        self._cached_actor_latent = None
        self._cached_proximal_feature = None
        self._critic_hidden_state = None
        self._warned_missing_prox_hidden = False
        self._warned_missing_dist_hidden = False
        self.register_buffer("_prox_points_cache", torch.empty(0), persistent=False)
        self.register_buffer("_dist_points_cache", torch.empty(0), persistent=False)
        self.register_buffer("_prox_points_valid_len", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_dist_points_valid_len", torch.empty(0, dtype=torch.long), persistent=False)
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
        if dones is None:
            self._critic_hidden_state = None
            self._prox_points_cache = torch.empty(0, device=self._proximal_indices.device)
            self._dist_points_cache = torch.empty(0, device=self._proximal_indices.device)
            self._prox_points_valid_len = torch.empty(0, dtype=torch.long, device=self._proximal_indices.device)
            self._dist_points_valid_len = torch.empty(0, dtype=torch.long, device=self._proximal_indices.device)
        elif self._critic_hidden_state is not None:
            self._critic_hidden_state[..., dones == 1, :] = 0.0
            if self._prox_points_valid_len.numel() > 0:
                self._prox_points_valid_len[dones == 1] = 0
            if self._dist_points_valid_len.numel() > 0:
                self._dist_points_valid_len[dones == 1] = 0
            if self._prox_points_cache.numel() > 0:
                self._prox_points_cache[dones == 1] = 0.0
            if self._dist_points_cache.numel() > 0:
                self._dist_points_cache[dones == 1] = 0.0

    def get_hidden_states(self):
        actor_hidden_states = (self.proximal_memory_a.hidden_states, self.distal_memory_a.hidden_states)
        if actor_hidden_states == (None, None) and self._critic_hidden_state is None:
            return (None, None)
        critic_hidden_states = (self._critic_hidden_state, self._critic_hidden_state)
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

    def _ensure_critic_hidden_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if self._critic_hidden_state is None or self._critic_hidden_state.shape[1] != batch_size:
            # Keep critic hidden-state shape aligned with recurrent latent size for storage/replay compatibility.
            self._critic_hidden_state = torch.zeros(
                (1, batch_size, self.distal_feature_dim),
                device=device,
                dtype=dtype,
            )

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
        seq_in, grouped, window_len, batch_size, _ = self._frame_window_to_seq(frame_feat)

        if masks is not None:
            if hidden_states is None:
                self._warn_missing_hidden_once(branch_name)
                hidden_states = self._init_actor_hidden_like(frame_feat, feat_dim)
            seq_out, _ = memory.rnn(seq_in, hidden_states)
            feat = self._collapse_window_output(seq_out, grouped, window_len, batch_size)
            if feat.dim() == 3:
                feat = unpad_trajectories(feat, masks)
                if feat.dim() == 3:
                    feat = feat.squeeze(0)
            return feat

        seq_out, memory.hidden_states = memory.rnn(seq_in, memory.hidden_states)
        return self._collapse_window_output(seq_out, grouped, window_len, batch_size)

    def _split_obs(self, observations: torch.Tensor):
        if observations.dim() == 2:
            proprio = observations[:, : self.proprio_obs_dim]
            lidar_flat = observations[
                :, self.proprio_obs_dim : self.proprio_obs_dim + self.history_length * self.num_lidar_points * 3
            ]
            lidar_hist = lidar_flat.reshape(-1, self.history_length, self.num_lidar_points, 3)
            return proprio, lidar_hist
        if observations.dim() == 3:
            t, b, _ = observations.shape
            obs_flat = observations.reshape(t * b, -1)
            proprio = obs_flat[:, : self.proprio_obs_dim].reshape(t, b, self.proprio_obs_dim)
            lidar_flat = obs_flat[
                :, self.proprio_obs_dim : self.proprio_obs_dim + self.history_length * self.num_lidar_points * 3
            ]
            lidar_hist = lidar_flat.reshape(t, b, self.history_length, self.num_lidar_points, 3)
            return proprio, lidar_hist
        raise ValueError(f"Unsupported observations rank: {observations.dim()}")

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
        ref_points = lidar_hist[0, -1]
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

    def _ensure_processed_point_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if (
            self._prox_points_cache.numel() == 0
            or self._prox_points_cache.shape[0] != batch_size
            or self._prox_points_cache.device != device
            or self._prox_points_cache.dtype != dtype
        ):
            self._prox_points_cache = torch.zeros(
                (batch_size, self.proximal_history_length, self.proximal_points, 3), device=device, dtype=dtype
            )
            self._prox_points_valid_len = torch.zeros((batch_size,), device=device, dtype=torch.long)

        if (
            self._dist_points_cache.numel() == 0
            or self._dist_points_cache.shape[0] != batch_size
            or self._dist_points_cache.device != device
            or self._dist_points_cache.dtype != dtype
        ):
            self._dist_points_cache = torch.zeros(
                (batch_size, self.distal_history_length, self.distal_points, 3), device=device, dtype=dtype
            )
            self._dist_points_valid_len = torch.zeros((batch_size,), device=device, dtype=torch.long)

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

    def _roll_points_cache_with_frame(
        self,
        cache: torch.Tensor,
        valid_len: torch.Tensor,
        frame_points: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        if cache.dim() != 4:
            raise ValueError(f"Expected cache rank 4, got {cache.dim()}")
        if frame_points.dim() != 3:
            raise ValueError(f"Expected frame_points rank 3, got {frame_points.dim()}")

        reset_mask = valid_mask & (valid_len == 0)
        continue_mask = valid_mask & (valid_len > 0)
        invalid_mask = ~valid_mask

        if torch.any(reset_mask):
            reset_points = frame_points[reset_mask].unsqueeze(1).repeat(1, cache.shape[1], 1, 1)
            cache[reset_mask] = reset_points
            valid_len[reset_mask] = 1

        if torch.any(continue_mask):
            rolled = torch.roll(cache[continue_mask], shifts=-1, dims=1)
            rolled[:, -1] = frame_points[continue_mask]
            cache[continue_mask] = rolled
            valid_len[continue_mask] = torch.clamp(valid_len[continue_mask] + 1, max=cache.shape[1])

        if torch.any(invalid_mask):
            cache[invalid_mask] = 0.0
            valid_len[invalid_mask] = 0

    def _build_online_points_windows(self, lidar_points_frame: torch.Tensor):
        batch_size = lidar_points_frame.shape[0]
        self._ensure_processed_point_cache(batch_size, lidar_points_frame.device, lidar_points_frame.dtype)

        prox_frame_points, dist_frame_points = self._compute_sampled_sorted_points_frame(lidar_points_frame)
        valid_mask = torch.ones((batch_size,), dtype=torch.bool, device=lidar_points_frame.device)
        self._roll_points_cache_with_frame(
            self._prox_points_cache, self._prox_points_valid_len, prox_frame_points, valid_mask
        )
        self._roll_points_cache_with_frame(
            self._dist_points_cache, self._dist_points_valid_len, dist_frame_points, valid_mask
        )
        return self._prox_points_cache, self._dist_points_cache

    def _build_replay_frame_features(self, lidar_points_seq: torch.Tensor, masks: torch.Tensor | None):
        if lidar_points_seq.dim() != 4:
            raise ValueError(f"Expected lidar_points_seq shape (T, B, N, 3), got rank {lidar_points_seq.dim()}")

        seq_len, batch_size, _, _ = lidar_points_seq.shape
        prox_cache = torch.zeros(
            (batch_size, self.proximal_history_length, self.proximal_points, 3),
            device=lidar_points_seq.device,
            dtype=lidar_points_seq.dtype,
        )
        dist_cache = torch.zeros(
            (batch_size, self.distal_history_length, self.distal_points, 3),
            device=lidar_points_seq.device,
            dtype=lidar_points_seq.dtype,
        )
        prox_valid_len = torch.zeros((batch_size,), device=lidar_points_seq.device, dtype=torch.long)
        dist_valid_len = torch.zeros((batch_size,), device=lidar_points_seq.device, dtype=torch.long)

        prox_feat_seq = []
        dist_feat_seq = []
        for t in range(seq_len):
            if masks is None:
                valid_mask = torch.ones((batch_size,), dtype=torch.bool, device=lidar_points_seq.device)
            else:
                valid_mask = masks[t].to(dtype=torch.bool)

            prox_frame_points, dist_frame_points = self._compute_sampled_sorted_points_frame(lidar_points_seq[t])
            self._roll_points_cache_with_frame(prox_cache, prox_valid_len, prox_frame_points, valid_mask)
            self._roll_points_cache_with_frame(dist_cache, dist_valid_len, dist_frame_points, valid_mask)

            # Encode the current cached windows immediately to avoid storing huge coordinate snapshots.
            prox_feat_seq.append(self._encode_proximal_points_chunked(prox_cache.clone()))
            dist_feat_seq.append(self._encode_distal_points_chunked(dist_cache.clone()))

        return torch.stack(prox_feat_seq, dim=0), torch.stack(dist_feat_seq, dim=0)

    def _encode_proximal_points_chunked(self, prox_points: torch.Tensor) -> torch.Tensor:
        flat_batch_size, t_prox, pn, _ = prox_points.shape
        prox_frame_feat = torch.empty(
            (flat_batch_size, t_prox, self.proximal_feature_dim),
            device=prox_points.device,
            dtype=prox_points.dtype,
        )
        # Chunk flat batch to keep GRU/MLP activations within small GPU memory budgets.
        chunk_size = 256
        for start in range(0, flat_batch_size, chunk_size):
            end = min(start + chunk_size, flat_batch_size)
            chunk = prox_points[start:end]
            chunk_enc = self.proximal_point_encoder(chunk.reshape((end - start) * t_prox * pn, 3)).reshape(
                end - start, t_prox, pn, -1
            )
            chunk_seq = chunk_enc.reshape((end - start) * t_prox, pn, -1)
            with torch.backends.cudnn.flags(enabled=False):
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
        chunk_size = 256
        for start in range(0, flat_batch_size, chunk_size):
            end = min(start + chunk_size, flat_batch_size)
            chunk = dist_points[start:end]
            chunk_enc = self.distal_point_encoder(chunk.reshape((end - start) * t_dist * dn, 3)).reshape(
                end - start, t_dist, dn, -1
            )
            chunk_seq = chunk_enc.reshape((end - start) * t_dist, dn, -1)
            with torch.backends.cudnn.flags(enabled=False):
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
        proprio, lidar_hist = self._split_obs(observations)

        if observations.dim() == 2:
            lidar_points_frame = lidar_hist[:, -1, :, :]
            prox_points, dist_points = self._build_online_points_windows(lidar_points_frame)
            seq_len = None
            batch_size = lidar_points_frame.shape[0]
            flat_batch_size = batch_size
        elif observations.dim() == 3:
            seq_len, batch_size, _, _, _ = lidar_hist.shape
            lidar_points_seq = lidar_hist[:, :, -1, :, :]
            prox_frame_feat, dist_frame_feat = self._build_replay_frame_features(lidar_points_seq, masks)
            return proprio, prox_frame_feat, dist_frame_feat
        else:
            raise ValueError(f"Unsupported observations rank: {observations.dim()}")

        _, t_prox, _, _ = prox_points.shape
        _, t_dist, _, _ = dist_points.shape

        # Proximal/Distal branches: encode with chunking to control activation memory.
        prox_frame_feat = self._encode_proximal_points_chunked(prox_points)
        dist_frame_feat = self._encode_distal_points_chunked(dist_points)

        if observations.dim() == 3:
            prox_frame_feat = prox_frame_feat.reshape(seq_len, batch_size, t_prox, -1)
            dist_frame_feat = dist_frame_feat.reshape(seq_len, batch_size, t_dist, -1)

        return proprio, prox_frame_feat, dist_frame_feat

    def _build_actor_latent(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    ):
        proprio, prox_frame_feat, dist_frame_feat = self._encode_perception(observations, masks=masks)
        prox_hidden_states, dist_hidden_states = self._split_actor_hidden_states(hidden_states)

        if masks is not None:
            prox_feat = self._run_actor_memory(
                self.proximal_memory_a,
                prox_frame_feat,
                masks,
                prox_hidden_states,
                self.proximal_feature_dim,
                "prox",
            )
            dist_feat = self._run_actor_memory(
                self.distal_memory_a,
                dist_frame_feat,
                masks,
                dist_hidden_states,
                self.distal_feature_dim,
                "dist",
            )

            proprio = unpad_trajectories(proprio, masks)
            if proprio.dim() == 3:
                proprio = proprio.squeeze(0)
        else:
            prox_feat = self._run_actor_memory(
                self.proximal_memory_a,
                prox_frame_feat,
                masks=None,
                hidden_states=None,
                feat_dim=self.proximal_feature_dim,
                branch_name="prox",
            )
            dist_feat = self._run_actor_memory(
                self.distal_memory_a,
                dist_frame_feat,
                masks=None,
                hidden_states=None,
                feat_dim=self.distal_feature_dim,
                branch_name="dist",
            )
            if observations.dim() == 2:
                self._ensure_critic_hidden_state(observations.shape[0], observations.device, observations.dtype)
            elif observations.dim() == 3:
                self._ensure_critic_hidden_state(observations.shape[1], observations.device, observations.dtype)

        actor_latent = torch.cat((proprio, prox_feat, dist_feat), dim=-1)

        self._cached_actor_latent = actor_latent
        self._cached_proximal_feature = prox_feat
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
        if critic_observations.shape[-1] == self.privileged_critic_dim:
            if masks is None:
                if critic_observations.dim() == 2:
                    self._ensure_critic_hidden_state(
                        critic_observations.shape[0], critic_observations.device, critic_observations.dtype
                    )
                elif critic_observations.dim() == 3:
                    self._ensure_critic_hidden_state(
                        critic_observations.shape[1], critic_observations.device, critic_observations.dtype
                    )
            elif critic_observations.dim() == 3:
                critic_observations = unpad_trajectories(critic_observations, masks).squeeze(0)
            return self.critic(critic_observations)

        actor_latent = self._build_actor_latent(critic_observations, masks=masks, hidden_states=hidden_states)
        return self.critic(actor_latent)

    def get_auxiliary_loss(self, privileged_heights: torch.Tensor) -> torch.Tensor:
        if self._cached_proximal_feature is None:
            return torch.zeros((), device=privileged_heights.device)

        if privileged_heights.shape[-1] == self.privileged_height_dim:
            height_target = privileged_heights
        elif privileged_heights.shape[-1] == self.privileged_critic_dim and self.privileged_critic_dim >= self.privileged_height_dim:
            # Critic privileged obs may be [proprio, heights]; supervise with trailing height channels.
            height_target = privileged_heights[..., -self.privileged_height_dim :]
        else:
            return torch.zeros((), device=privileged_heights.device)

        pred = self.height_supervisor(self._cached_proximal_feature)
        return self.privileged_supervision_coef * torch.mean(torch.square(pred - height_target))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
