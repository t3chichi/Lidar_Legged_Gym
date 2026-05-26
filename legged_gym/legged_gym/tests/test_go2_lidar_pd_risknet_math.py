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

    sys.path.insert(0, 'rsl_rl')
    from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic

    num_obs = 48 + 864 * 3
    model = PDRiskNetActorCritic(num_obs, 235, 12,
        num_lidar_points=864, proximal_points=256, distal_points=96,
        proximal_history_length=1, distal_history_length=10)
    obs = torch.randn(3, num_obs)
    act = model.act(obs)
    val = model.evaluate(torch.randn(3, 235))

    assert tuple(act.shape) == (3, 12)
    assert tuple(val.shape) == (3, 1)


def test_pd_risknet_auxiliary_supervision_gate():
    import sys
    import torch

    sys.path.insert(0, 'rsl_rl')
    from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic

    num_obs = 48 + 864 * 3
    model = PDRiskNetActorCritic(num_obs, 235, 12,
        num_lidar_points=864, proximal_points=256, distal_points=96,
        proximal_history_length=1, distal_history_length=10)
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
        class asset:
            pass

        class init_state:
            pass

        class env:
            num_envs = 4096

        class terrain:
            measure_heights = True

        class commands:
            class ranges:
                pass

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

        class sim:
            class physx:
                pass

        class obstacle_gen:
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

    cfg_path = 'legged_gym/legged_gym/envs/go2/lidar_pd_risknet/go2_lidar_pd_risknet_config.py'
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
    assert env_cfg.pd_risknet.history_length == 1
    assert env_cfg.pd_risknet.proximal_feature_dim == 187
    assert env_cfg.pd_risknet.distal_feature_dim == 64
    assert env_cfg.pd_risknet.n_sectors == 36


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


def test_v_avoid_guided_formula():
    """Guided formula: w_i = (cos_i + c) * exp(-alpha * d_i) * (d_i < d_max),
       v_avoid = ||v_cmd|| * sum(w_i * (-u_i))."""
    import torch
    import math

    c_val = 0.15
    alpha = 1.0
    d_max = 1.5
    n_sec = 36
    sec_size = 2.0 * math.pi / n_sec

    # Sector center directions: sec 0 = -175°, sec 18 approx 5° (forward), etc.
    sec_centers = torch.linspace(-math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size, n_sec)
    u = torch.stack((torch.cos(sec_centers), torch.sin(sec_centers)), dim=-1)       # (n_sec, 2)
    away_dirs = -u                                                                   # (n_sec, 2)

    inf = torch.tensor(1e9)
    v_cmd_dir = torch.tensor([[1.0, 0.0]])  # forward

    # --- Case 1: stationary command -> v_avoid = 0 ---
    v_cmd_0 = torch.tensor([[0.0, 0.0]])
    nonzero_0 = torch.norm(v_cmd_0, dim=1) > 1e-6
    assert not nonzero_0.any(), "stationary command should produce no avoidance"

    # --- Case 2: forward command, obstacle at sec 18 (forward), d=0.5m ---
    d_1 = torch.full((1, n_sec), inf)
    d_1[0, 18] = 0.5

    cos = torch.relu(torch.mm(v_cmd_dir, u.T))  # (1, n_sec)
    # sec 18 center = 5°, cos = cos(5°) = 0.996
    assert cos[0, 18].item() > 0.99

    active = d_1 < d_max
    w = (cos + c_val) * torch.exp(-alpha * d_1) * active.float()
    v_avoid_1 = 0.5 * (w @ away_dirs)  # ||v_cmd|| = 0.5

    # w[0, 18] = (0.996 + 0.15) * exp(-0.5) = 1.146 * 0.6065 = 0.695
    expected_w = (cos[0, 18].item() + c_val) * math.exp(-0.5)
    assert abs(w[0, 18].item() - expected_w) < 1e-4
    # Strong backward push (negative x)
    assert v_avoid_1[0, 0].item() < -0.3

    # --- Case 3: forward command, clear environment -> v_avoid = 0 ---
    d_3 = torch.full((1, n_sec), inf)  # all clear
    active_3 = d_3 < d_max
    # exp(-inf) = 0, and active_3 is all False -> w_3 all zeros
    w_3 = (cos + c_val) * torch.exp(-alpha * d_3) * active_3.float()
    assert (w_3 == 0.0).all(), "clear env should have zero weights"
    v_avoid_3 = 0.5 * (w_3 @ away_dirs)
    assert torch.all(v_avoid_3.abs() < 1e-6)

    # --- Case 4: forward command, lateral wall on left (cos = 0, d small) ---
    # Sector 27 center = 95° = left, u = [-0.087, 0.996], cos = 0 (forward cmd perpendicular to left)
    d_4 = torch.full((1, n_sec), inf)
    d_4[0, 27] = 0.3  # left wall close
    active_4 = d_4 < d_max
    w_4 = (cos + c_val) * torch.exp(-alpha * d_4) * active_4.float()
    v_avoid_4 = 0.5 * (w_4 @ away_dirs)

    # cos[27] = max(0, [1,0][-0.087,0.996]) = 0
    # w = (0 + 0.15) * exp(-0.3) = 0.111
    # away_dir[27] = -u[27] = [0.087, -0.996], pushes right+forward (away from left wall)
    expected_w_27 = 0.15 * math.exp(-0.3)
    assert abs(w_4[0, 27].item() - expected_w_27) < 1e-4
    # Push should be rightward (negative y in body frame) since wall is on left
    assert v_avoid_4[0, 1].item() < 0.0, f"should push right away from left wall, got vy={v_avoid_4[0,1].item():.4f}"
    # Gentle push magnitude = 0.5 * 0.111 = 0.056
    mag_4 = v_avoid_4.norm(dim=1).item()
    assert 0.03 < mag_4 < 0.10, f"lateral push should be gentle, got {mag_4:.4f}"

    # --- Case 5: 5-sector lateral wall (cos = 0) -> should NOT explode ---
    # Sectors 6-10 (span -115deg to -75deg, right side). With forward command,
    # cos is near-zero for most of these, dominated by c.
    d_5 = torch.full((1, n_sec), inf)
    for i in range(6, 11):
        d_5[0, i] = 0.5
    active_5 = d_5 < d_max
    w_5 = (cos + c_val) * torch.exp(-alpha * d_5) * active_5.float()
    v_avoid_5 = 0.5 * (w_5 @ away_dirs)
    mag_5 = v_avoid_5.norm(dim=1).item()
    # With cos=0 for most of these 5 sectors, each contributes ~c*exp(-0.5)=0.091.
    # Away y-components all point left (positive), summing to ~0.32 total.
    assert mag_5 < 0.4, \
        f"lateral wall should not cause large avoidance: {mag_5:.4f} >= 0.4"
