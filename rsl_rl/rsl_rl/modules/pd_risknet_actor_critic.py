from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class PDRiskNetActorCritic(nn.Module):
    """PD-RiskNet actor-critic.

    Observation layout:
    - first proprio_obs_dim dims: proprio/command/action history-free state
    - remaining dims: lidar history points, flattened as
      [history_length, num_lidar_points, 3]
    """

    is_recurrent = False

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
        history_length: int = 10,
        num_lidar_points: int = 1024,
        proximal_points: int = 512,
        distal_points: int = 512,
        split_theta_deg: float = 0.0,
        proximal_feature_dim: int = 187,
        distal_feature_dim: int = 64,
        proprio_obs_dim: int = 48,
        privileged_height_dim: int = 187,
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
        self.num_lidar_points = int(num_lidar_points)
        self.proximal_points = int(proximal_points)
        self.distal_points = int(distal_points)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.split_theta = float(split_theta_deg) * math.pi / 180.0
        self.proximal_feature_dim = int(proximal_feature_dim)
        self.distal_feature_dim = int(distal_feature_dim)
        self.privileged_height_dim = int(privileged_height_dim)
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
        self.distal_gru = nn.GRU(input_size=64, hidden_size=self.distal_feature_dim, batch_first=True)

        actor_input_dim = self.proprio_obs_dim + self.proximal_feature_dim + self.distal_feature_dim
        critic_input_dim = actor_input_dim if num_critic_obs != self.privileged_height_dim else self.privileged_height_dim

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
        return

    def _split_obs(self, observations: torch.Tensor):
        proprio = observations[:, : self.proprio_obs_dim]
        lidar_flat = observations[:, self.proprio_obs_dim : self.proprio_obs_dim + self.history_length * self.num_lidar_points * 3]
        lidar_hist = lidar_flat.reshape(-1, self.history_length, self.num_lidar_points, 3)
        return proprio, lidar_hist

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

    def _sort_by_spherical(self, points):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        theta = torch.atan2(z, torch.sqrt(x * x + y * y + 1.0e-8))
        phi = torch.atan2(y, x)
        order = torch.argsort(theta * (2.0 * math.pi) + phi, dim=-1)
        order_exp = order.unsqueeze(-1).expand_as(points)
        return torch.gather(points, dim=2, index=order_exp)

    def _encode_perception(self, observations: torch.Tensor):
        proprio, lidar_hist = self._split_obs(observations)

        if (not self._sampling_plan_ready) or (self._proximal_indices.numel() == 0) or (
            int(self._proximal_indices.max().item()) >= lidar_hist.shape[2]
        ):
            self._build_sampling_plan(lidar_hist)

        prox_points = self._sample_proximal_fps(lidar_hist)
        dist_points = self._sample_distal_avg(lidar_hist)

        prox_points = self._sort_by_spherical(prox_points)
        dist_points = self._sort_by_spherical(dist_points)

        b, t, pn, _ = prox_points.shape
        _, _, dn, _ = dist_points.shape

        prox_enc = self.proximal_point_encoder(prox_points.reshape(b * t * pn, 3)).reshape(b, t, pn, -1).mean(dim=2)
        dist_enc = self.distal_point_encoder(dist_points.reshape(b * t * dn, 3)).reshape(b, t, dn, -1).mean(dim=2)

        _, prox_h = self.proximal_gru(prox_enc)
        _, dist_h = self.distal_gru(dist_enc)

        prox_feat = prox_h.squeeze(0)
        dist_feat = dist_h.squeeze(0)
        actor_latent = torch.cat((proprio, prox_feat, dist_feat), dim=-1)

        self._cached_actor_latent = actor_latent
        self._cached_proximal_feature = prox_feat
        return actor_latent, prox_feat

    def update_distribution(self, observations):
        actor_latent, _ = self._encode_perception(observations)
        mean = self.actor(actor_latent)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_latent, _ = self._encode_perception(observations)
        return self.actor(actor_latent)

    def evaluate(self, critic_observations, **kwargs):
        if critic_observations.shape[-1] == self.privileged_height_dim:
            return self.critic(critic_observations)

        actor_latent, _ = self._encode_perception(critic_observations)
        return self.critic(actor_latent)

    def get_auxiliary_loss(self, privileged_heights: torch.Tensor) -> torch.Tensor:
        if self._cached_proximal_feature is None:
            return torch.zeros((), device=privileged_heights.device)
        if privileged_heights.shape[-1] != self.privileged_height_dim:
            return torch.zeros((), device=privileged_heights.device)
        pred = self.height_supervisor(self._cached_proximal_feature)
        return self.privileged_supervision_coef * torch.mean(torch.square(pred - privileged_heights))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
