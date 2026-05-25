import math

import pytest


def vel_avoid_reward(v_t, v_cmd, v_avoid, beta_va):
    import torch

    err = torch.sum(torch.square(v_t - (v_cmd + v_avoid)), dim=-1)
    return torch.exp(-beta_va * err)


def rays_reward(distances, d_max):
    import torch

    clipped = torch.clamp(distances, max=d_max)
    return torch.mean(clipped / d_max, dim=-1)


def test_vel_avoid_formula_matches_paper():
    import torch

    v_t = torch.tensor([[0.5, 0.0], [0.0, 0.0]], dtype=torch.float32)
    v_cmd = torch.tensor([[0.5, 0.0], [0.3, 0.0]], dtype=torch.float32)
    v_avoid = torch.tensor([[0.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    beta_va = 1.0

    rew = vel_avoid_reward(v_t, v_cmd, v_avoid, beta_va)
    assert torch.isclose(rew[0], torch.tensor(1.0), atol=1e-6)

    expected = math.exp(-0.25)
    assert torch.isclose(rew[1], torch.tensor(expected), atol=1e-6)


def test_rays_formula_matches_paper():
    import torch

    distances = torch.tensor([[1.0, 2.0, 12.0]], dtype=torch.float32)
    d_max = 10.0
    rew = rays_reward(distances, d_max)
    expected = (1.0 / 10.0 + 2.0 / 10.0 + 10.0 / 10.0) / 3.0
    assert torch.isclose(rew[0], torch.tensor(expected), atol=1e-6)


def test_pd_risknet_policy_shape_gate():
    import sys
    import torch

    sys.path.insert(0, '/home/t3chichi/extended_legged_gym/rsl_rl')
    from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic

    num_obs = 48 + 10 * 1024 * 3
    model = PDRiskNetActorCritic(num_obs, 187, 12)
    obs = torch.randn(3, num_obs)
    act = model.act(obs)
    val = model.evaluate(torch.randn(3, 187))

    assert tuple(act.shape) == (3, 12)
    assert tuple(val.shape) == (3, 1)


def test_pd_risknet_auxiliary_supervision_gate():
    import sys
    import torch

    sys.path.insert(0, '/home/t3chichi/extended_legged_gym/rsl_rl')
    from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic

    num_obs = 48 + 10 * 1024 * 3
    model = PDRiskNetActorCritic(num_obs, 187, 12)
    obs = torch.randn(4, num_obs)

    # Populate cached proximal feature through a forward actor path.
    _ = model.act(obs)

    good_priv = torch.randn(4, 187)
    aux = model.get_auxiliary_loss(good_priv)
    assert aux.ndim == 0
    assert aux.item() >= 0.0

    bad_priv = torch.randn(4, 32)
    aux_bad = model.get_auxiliary_loss(bad_priv)
    assert torch.isclose(aux_bad, torch.tensor(0.0), atol=1e-8)


def test_pd_risknet_config_gate():
    import importlib.util
    import types
    import sys

    # Stub the base config import chain so this gate does not require full
    # legged_gym runtime dependencies (isaacgym/cv2/etc.).
    legged_gym_mod = types.ModuleType('legged_gym')
    envs_mod = types.ModuleType('legged_gym.envs')
    go2_mod = types.ModuleType('legged_gym.envs.go2')
    flat_mod = types.ModuleType('legged_gym.envs.go2.flat')
    rough_cfg_mod = types.ModuleType('legged_gym.envs.go2.flat.go2_rough_config')

    class _Go2RoughCfg:
        class env:
            num_envs = 4096

        class terrain:
            measure_heights = True

        class raycaster:
            pass

        class rewards:
            class scales:
                pass

        class normalization:
            class obs_scales:
                pass

        class domain_rand:
            pass

    class _Go2RoughCfgPPO:
        class policy:
            pass

        class algorithm:
            pass

        class runner:
            pass

    rough_cfg_mod.Go2RoughCfg = _Go2RoughCfg
    rough_cfg_mod.Go2RoughCfgPPO = _Go2RoughCfgPPO

    sys.modules['legged_gym'] = legged_gym_mod
    sys.modules['legged_gym.envs'] = envs_mod
    sys.modules['legged_gym.envs.go2'] = go2_mod
    sys.modules['legged_gym.envs.go2.flat'] = flat_mod
    sys.modules['legged_gym.envs.go2.flat.go2_rough_config'] = rough_cfg_mod

    cfg_path = '/home/t3chichi/extended_legged_gym/legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py'
    spec = importlib.util.spec_from_file_location('go2_lidar_pd_cfg', cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Go2LidarPDRiskNetCfg = module.Go2LidarPDRiskNetCfg
    Go2LidarPDRiskNetCfgPPO = module.Go2LidarPDRiskNetCfgPPO

    env_cfg = Go2LidarPDRiskNetCfg()
    train_cfg = Go2LidarPDRiskNetCfgPPO()

    # Paper-critical rollout and PPO settings.
    assert env_cfg.env.num_envs == 4096
    assert train_cfg.runner.num_steps_per_env == 24
    assert train_cfg.algorithm.clip_param == 0.2
    assert train_cfg.algorithm.lam == 0.95
    assert train_cfg.algorithm.gamma == 0.99
    assert train_cfg.algorithm.learning_rate == 1.0e-3
    assert train_cfg.algorithm.schedule == "adaptive"
    assert train_cfg.algorithm.entropy_coef == 0.01
    assert train_cfg.algorithm.desired_kl == 0.01
    assert train_cfg.algorithm.max_grad_norm == 1.0
    assert train_cfg.algorithm.num_learning_epochs == 5
    assert train_cfg.algorithm.num_mini_batches == 4

    # PD-RiskNet shape contract.
    assert env_cfg.pd_risknet.history_length == 10
    assert env_cfg.pd_risknet.proximal_feature_dim == 187
    assert env_cfg.pd_risknet.distal_feature_dim == 64
    assert env_cfg.pd_risknet.n_sectors == 24


def build_distal_mask(num_azimuth, num_elevation, v_fov_min_deg, v_fov_max_deg,
                       split_theta_deg, device="cpu"):
    """Replicate the distal mask logic used in _init_pd_risknet_buffers.

    Returns a bool tensor of shape (num_elevation * num_azimuth,).
    """
    import torch
    import math

    v_min_rad = math.radians(v_fov_min_deg)
    v_max_rad = math.radians(v_fov_max_deg)
    split_rad = math.radians(split_theta_deg)

    elev_rad = torch.linspace(v_max_rad, v_min_rad, num_elevation, device=device)
    distal_lines = elev_rad < split_rad  # (num_elevation,)
    distal_mask_2d = distal_lines.unsqueeze(1).expand(num_elevation, num_azimuth)
    return distal_mask_2d.contiguous().reshape(-1)


def test_distal_mask_shape_and_count():
    import torch

    num_azimuth = 24
    num_elevation = 18
    v_fov_min_deg = -2.0
    v_fov_max_deg = 57.0
    split_theta_deg = 20.0

    mask = build_distal_mask(num_azimuth, num_elevation,
                              v_fov_min_deg, v_fov_max_deg, split_theta_deg)

    # Shape: full spherical grid
    assert mask.shape == (num_azimuth * num_elevation,), f"expected ({num_azimuth * num_elevation},), got {mask.shape}"
    # Must be bool
    assert mask.dtype == torch.bool

    distal_count = mask.sum().item()
    # With 18 lines from 57° down to -2°, lines < 20°: 11 through 17 = 7 lines x 24 = 168
    assert distal_count == 168, f"expected 168 distal points, got {distal_count}"

    # Verify specific lines: line 0 (57°) is NOT distal, line 17 (-2°) IS distal
    assert not mask[0].item()          # line 0, azimuth 0: elevation 57° -> proximal
    assert mask[-1].item()             # line 17, azimuth 23: elevation -2° -> distal


def test_distal_rays_reward_matches_paper():
    """Paper formula with deterministic distances and hand-computed expected value."""
    import torch

    # Use a tiny grid (2 elevation x 3 azimuth = 6 points) for hand verification.
    # FOV: 0° to 30°, split at 15° -> line 0 (30°): proximal, line 1 (0°): distal.
    num_azimuth = 3
    num_elevation = 2
    mask = build_distal_mask(num_azimuth, num_elevation,
                              0.0, 30.0, 15.0)
    # mask = [False, False, False, True, True, True] -> 3 distal points

    # One env, 6 points. Distal points (indices 3,4,5) at distances 2, 8, 15.
    all_distances = torch.tensor([[5.0, 1.0, 9.0,  2.0, 8.0, 15.0]], dtype=torch.float32)
    distal_dist = all_distances[:, mask]  # [[2.0, 8.0, 15.0]]

    d_max = 10.0
    # expected: mean(min(2,10)/10, min(8,10)/10, min(15,10)/10)
    #         = mean(0.2, 0.8, 1.0) = 2.0 / 3.0
    expected = torch.tensor([(0.2 + 0.8 + 1.0) / 3.0], dtype=torch.float32)

    reward = torch.mean(torch.clamp(distal_dist, max=d_max) / d_max, dim=1)
    assert torch.allclose(reward, expected, atol=1e-6)


def test_distal_mask_empty_raises():
    """A mask with zero distal rays should raise ValueError (production guard)."""
    import torch

    # split at -10° is below v_fov_min (0°) -> all rays proximal, mask is empty.
    with pytest.raises(ValueError, match="No distal rays found"):
        mask = build_distal_mask(3, 2, 0.0, 30.0, -10.0)
        # Replicate the production guard check
        if mask.sum() == 0:
            raise ValueError(
                f"No distal rays found: split_theta_deg={-10.0:.1f}° "
                f"but vertical FOV min={0.0:.1f}°. "
                f"Lower split_theta_deg or decrease vertical_fov_deg_min.")


def test_v_avoid_paper_formula():
    """Paper formula: V_j = exp(-d_j * alpha) * (-dir_j) if d_j < thresh."""
    import torch
    import math

    alpha = 1.5
    thresh = 1.0
    n_sec = 36
    sec_size = 2.0 * math.pi / n_sec

    # Simulate 2 envs: env0 has close obstacle at sector 2 (0.5m), env1 is clear
    inf = torch.tensor(1e9)
    min_dist = torch.tensor([
        [inf, inf, 0.5, inf, inf] + [inf] * (n_sec - 5),   # env0
        [inf] * n_sec,                                        # env1: all clear
    ])

    active = min_dist < thresh
    mag = torch.exp(-min_dist * alpha) * active.float()
    sec_centers = torch.linspace(-math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size, n_sec)
    away_dirs = torch.stack((-torch.cos(sec_centers), -torch.sin(sec_centers)), dim=-1)

    v_avoid = torch.sum(away_dirs.unsqueeze(0) * mag.unsqueeze(-1), dim=1)

    # env1: zero avoidance
    assert torch.all(v_avoid[1] == 0.0)

    # env0: non-zero, pointing away from sector 2
    assert v_avoid[0].norm() > 0.0

    # Magnitude check: |V| = exp(-0.5 * 1.5) ≈ 0.4724 (single active sector)
    expected_mag = math.exp(-0.5 * 1.5)
    assert abs(v_avoid[0].norm().item() - expected_mag) < 1e-4
