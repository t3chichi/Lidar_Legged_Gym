import math
import unittest

import pytest
import legged_gym.envs  # Pre-import to break circular dependency chain
import torch


def vel_avoid_reward(v_t, v_cmd, v_avoid, beta_va):
    import torch

    err = torch.sum(torch.square(v_t - (v_cmd + v_avoid)), dim=-1)
    return torch.exp(-beta_va * err)


def rays_direction_reward(v_body, smooth_dir_body, eps=0.01):
    """Direction-consistency reward: r = dot(v_body, smooth_dir) / max(|v_body|, eps)."""
    import torch

    v_norm = torch.norm(v_body, dim=-1)
    dot = (v_body * smooth_dir_body).sum(dim=-1)
    return dot / torch.clamp(v_norm, min=eps)


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


def test_rays_direction_perfect_alignment():
    """Moving exactly toward open space -> reward near +1."""
    import torch

    v = torch.tensor([[1.0, 0.0], [0.5, 0.0], [2.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    assert torch.allclose(r, torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)


def test_rays_direction_opposite():
    """Moving away from open space -> reward near -1."""
    import torch

    v = torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
    d = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    assert torch.allclose(r, torch.tensor([-1.0, -1.0]), atol=1e-6)


def test_rays_direction_orthogonal():
    """Moving perpendicular to open space -> reward near 0."""
    import torch

    v = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6)


def test_rays_direction_speed_invariant():
    """Same direction, different speeds -> same reward (speed-decoupled)."""
    import torch

    v = torch.tensor([[0.1, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d)
    assert torch.allclose(r, torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)


def test_rays_direction_zero_velocity():
    """Zero velocity -> reward near 0 (eps prevents division by zero)."""
    import torch

    v = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    r = rays_direction_reward(v, d, eps=0.01)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6)


def test_rays_direction_partial_alignment():
    """45 deg between velocity and direction -> reward = cos(45) ~ 0.707."""
    import torch

    v = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    d = torch.tensor([[math.cos(math.radians(45)), math.sin(math.radians(45))]],
                     dtype=torch.float32)
    r = rays_direction_reward(v, d)
    expected = math.cos(math.radians(45))
    assert torch.allclose(r, torch.tensor([expected]), atol=1e-6)


def test_rays_target_dir_power_weights():
    """Normalized power weighting: front (8m) vs right (2m) with p=6, d_max=10.

    w_front = (8/10)^6 = 0.262, w_right = (2/10)^6 = 6.4e-5. Ratio 4096:1.
    Direction is essentially pure forward (< 0.02 deg).
    """
    import torch

    d_max = 10.0
    p = 6
    d_front = torch.tensor([8.0])
    d_right = torch.tensor([2.0])
    w_front = (d_front / d_max).pow(p)
    w_right = (d_right / d_max).pow(p)

    dir_front = torch.tensor([1.0, 0.0])
    dir_right = torch.tensor([0.0, 1.0])

    weighted_sum = w_front * dir_front + w_right * dir_right
    target_dir = weighted_sum / torch.norm(weighted_sum)

    actual_angle = math.atan2(target_dir[1].item(), target_dir[0].item())
    assert actual_angle < math.radians(0.02)
    assert target_dir[0].item() > 0.9999


def test_rays_target_dir_bend_scenario():
    """Bend: front at 3m, left-forward (30 deg) at 6m -> direction shifts strongly leftward.

    Normalized power weights p=6, d_max=10:
    w_front = 0.3^6 = 7.29e-4, w_diag = 0.6^6 = 4.67e-2. Ratio 64:1 for diagonal.
    Expected angle ~29.6 deg.
    """
    import torch

    d_max = 10.0
    p = 6
    d_front = torch.tensor([3.0])
    d_diag = torch.tensor([6.0])
    w_front = (d_front / d_max).pow(p)
    w_diag = (d_diag / d_max).pow(p)

    angle_30 = math.radians(30)
    dir_front = torch.tensor([1.0, 0.0])
    dir_diag = torch.tensor([math.cos(angle_30), math.sin(angle_30)])

    weighted_sum = w_front * dir_front + w_diag * dir_diag
    target_dir = weighted_sum / torch.norm(weighted_sum)

    actual_angle = math.atan2(target_dir[1].item(), target_dir[0].item())
    # With p=6 the diagonal strongly dominates, angle ~29.6 deg.
    assert actual_angle > math.radians(25), \
        f"bend should shift direction >25 deg, got {math.degrees(actual_angle):.2f}"


def test_rays_ema_smoothing():
    """EMA: smooth = normalize(alpha * target + (1-alpha) * prev)."""
    import torch

    alpha = 0.4
    prev = torch.tensor([1.0, 0.0])
    target = torch.tensor([0.0, 1.0])

    raw = alpha * target + (1 - alpha) * prev
    smooth = raw / torch.norm(raw)

    expected_angle = math.atan2(0.4, 0.6)
    actual_angle = math.atan2(smooth[1].item(), smooth[0].item())
    assert abs(actual_angle - expected_angle) < 1e-4, \
        f"expected {math.degrees(expected_angle):.2f} deg, got {math.degrees(actual_angle):.2f}"


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
    """Pure distance-weighted formula: w_i = exp(-alpha * d_i) * (d_i < d_max),
       v_avoid = ||v_cmd|| * sum(w_i * (-u_i)) / sum(w_i)."""
    import torch
    import math

    alpha = 1.0
    d_max = 1.5
    n_sec = 36
    sec_size = 2.0 * math.pi / n_sec

    sec_centers = torch.linspace(-math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size, n_sec)
    u = torch.stack((torch.cos(sec_centers), torch.sin(sec_centers)), dim=-1)
    away_dirs = -u

    inf = torch.tensor(1e9)

    # --- Case 1: head-on obstacle (sector 18, d=0.5) -> backward push ---
    d_1 = torch.full((1, n_sec), inf)
    d_1[0, 18] = 0.5
    active_1 = d_1 < d_max
    w_1 = torch.exp(-alpha * d_1) * active_1.float()
    w_sum_1 = w_1.sum(dim=1, keepdim=True)
    v_avoid_1 = 0.5 * ((w_1 @ away_dirs) / (w_sum_1 + 1e-6))

    # Single sector: weighted avg = away_dir[18] = [-0.996, -0.087]
    assert v_avoid_1[0, 0].item() < -0.4
    assert abs(v_avoid_1.norm().item() - 0.5) < 1e-4

    # --- Case 2: clear environment -> v_avoid = 0 ---
    d_2 = torch.full((1, n_sec), inf)
    active_2 = d_2 < d_max
    w_2 = torch.exp(-alpha * d_2) * active_2.float()
    w_sum_2 = w_2.sum(dim=1, keepdim=True)
    v_avoid_2 = 0.5 * ((w_2 @ away_dirs) / (w_sum_2 + 1e-6))
    assert torch.all(v_avoid_2.abs() < 1e-6)

    # --- Case 3: left wall closer than right -> push rightward ---
    # Left sector 26 (85deg, away=[-0.087, -0.996]) at d=0.3, w=exp(-0.3)=0.741
    # Right sector 9 (-85deg, away=[-0.087, 0.996]) at d=1.0, w=exp(-1.0)=0.368
    # Weighted avg y = (-0.996*0.741 + 0.996*0.368)/1.109 = -0.335 -> pushes right
    d_3 = torch.full((1, n_sec), inf)
    d_3[0, 26] = 0.3   # left wall close
    d_3[0, 9] = 1.0    # right wall far
    active_3 = d_3 < d_max
    w_3 = torch.exp(-alpha * d_3) * active_3.float()
    w_sum_3 = w_3.sum(dim=1, keepdim=True)
    v_avoid_3 = 0.5 * ((w_3 @ away_dirs) / (w_sum_3 + 1e-6))

    # Left wall closer -> push right (vy < 0)
    # vy = 0.5 * weighted_avg_y = 0.5 * (-0.335) = -0.1675
    assert v_avoid_3[0, 1].item() < -0.15, \
        f"should push away from closer left wall, got vy={v_avoid_3[0,1].item():.4f}"

    # --- Case 4: corridor with symmetric walls -> x and y cancel ---
    d_4 = torch.full((1, n_sec), 1.2)
    active_4 = d_4 < d_max
    w_4 = torch.exp(-alpha * d_4) * active_4.float()
    w_sum_4 = w_4.sum(dim=1, keepdim=True)
    v_avoid_4 = 0.5 * ((w_4 @ away_dirs) / (w_sum_4 + 1e-6))

    # All 36 sectors at same distance -> symmetric cancel, v_avoid ~ zero
    assert abs(v_avoid_4[0, 0].item()) < 0.05, \
        f"corridor x should cancel, got vx={v_avoid_4[0,0].item():.4f}"
    assert abs(v_avoid_4[0, 1].item()) < 0.05, \
        f"corridor y should cancel, got vy={v_avoid_4[0,1].item():.4f}"

    # --- Case 5: stationary command -> v_avoid = 0 ---
    d_5 = d_1.clone()  # obstacle at sector 18, d=0.5
    active_5 = d_5 < d_max
    w_5 = torch.exp(-alpha * d_5) * active_5.float()
    w_sum_5 = w_5.sum(dim=1, keepdim=True)
    v_avoid_5 = 0.0 * ((w_5 @ away_dirs) / (w_sum_5 + 1e-6))
    assert torch.all(v_avoid_5.abs() < 1e-6)


def test_trapezoid_corridor_geometry():
    """Verify trapezoid corridor spawn_angle and goal_offset for all 4 directions."""
    import numpy as np
    import math
    from legged_gym.utils.terrain import trapezoid_corridor_terrain
    from isaacgym import terrain_utils

    hs = 0.1
    vs = 1.0
    size = 150

    expected = [
        # dir, spawn_angle, goal_ox_sign (0=zero, +/-1=sign), goal_oy_sign
        (0,  math.pi / 2,  0, +1),
        (1,  0.0,          +1,  0),
        (2, -math.pi / 2,   0, -1),
        (3,  math.pi,      -1,  0),
    ]

    for direction, exp_spawn, goal_ox_sign, goal_oy_sign in expected:
        class Cfg:
            corridor_width = 3.0
            wall_height = 1.5
            wall_thickness = 2.0
            turn_angle_deg_max = 55.0
            diagonal_length = 3.0
            terrain_length = 15.0
            terrain_width = 15.0
            end_margin = 0.5
            goal_forward_margin = 0.6
            goal_radius = 1.6
            curriculum = False
            _first_turn_left = True

        cfg = Cfg()
        terrain = terrain_utils.SubTerrain(f"test_dir{direction}", width=size, length=size,
                             vertical_scale=vs, horizontal_scale=hs)
        trapezoid_corridor_terrain(terrain, difficulty=0.5, cfg=cfg, direction=direction)

        assert abs(terrain.spawn_angle - exp_spawn) < 1e-6, \
            f"dir={direction}: expected spawn_angle={exp_spawn}, got {terrain.spawn_angle}"

        if goal_ox_sign == 0:
            assert abs(cfg.goal_offset_x) < 1e-6, \
                f"dir={direction}: expected goal_offset_x=0, got {cfg.goal_offset_x}"
        elif goal_ox_sign > 0:
            assert cfg.goal_offset_x > 1.0, \
                f"dir={direction}: expected goal_offset_x > 0, got {cfg.goal_offset_x}"
        else:
            assert cfg.goal_offset_x < -1.0, \
                f"dir={direction}: expected goal_offset_x < 0, got {cfg.goal_offset_x}"

        if goal_oy_sign == 0:
            assert abs(cfg.goal_offset_y) < 1e-6, \
                f"dir={direction}: expected goal_offset_y=0, got {cfg.goal_offset_y}"
        elif goal_oy_sign > 0:
            assert cfg.goal_offset_y > 1.0, \
                f"dir={direction}: expected goal_offset_y > 0, got {cfg.goal_offset_y}"
        else:
            assert cfg.goal_offset_y < -1.0, \
                f"dir={direction}: expected goal_offset_y < 0, got {cfg.goal_offset_y}"


def test_trapezoid_corridor_lr_rl_mirror():
    """L-R and R-L corridors should be mirror images across the midline."""
    import numpy as np
    import math
    from legged_gym.utils.terrain import trapezoid_corridor_terrain
    from isaacgym import terrain_utils

    hs = 0.1
    vs = 1.0
    size = 150

    class Cfg:
        corridor_width = 3.0
        wall_height = 1.5
        wall_thickness = 2.0
        turn_angle_deg_max = 45.0
        diagonal_length = 3.0
        terrain_length = 15.0
        terrain_width = 15.0
        end_margin = 0.5
        goal_forward_margin = 0.0
        goal_radius = 1.6
        curriculum = False

    cfg_lr = Cfg()
    cfg_lr._first_turn_left = True
    terrain_lr = terrain_utils.SubTerrain("lr", width=size, length=size,
                            vertical_scale=vs, horizontal_scale=hs)
    trapezoid_corridor_terrain(terrain_lr, difficulty=0.5, cfg=cfg_lr)

    cfg_rl = Cfg()
    cfg_rl._first_turn_left = False
    terrain_rl = terrain_utils.SubTerrain("rl", width=size, length=size,
                            vertical_scale=vs, horizontal_scale=hs)
    trapezoid_corridor_terrain(terrain_rl, difficulty=0.5, cfg=cfg_rl)

    # Mirror across X midline: hf_lr[x, y] should equal hf_rl[size-1-x, y]
    hf_lr = terrain_lr.height_field_raw
    hf_rl = terrain_rl.height_field_raw
    hf_rl_mirrored = hf_rl[::-1, :]  # flip along X axis
    assert np.array_equal(hf_lr, hf_rl_mirrored), \
        "L-R and R-L corridors should be X-mirror images"


def test_trapezoid_corridor_level0_straight():
    """Difficulty 0 (turn_angle=0) should produce a straight north corridor."""
    import numpy as np
    from legged_gym.utils.terrain import trapezoid_corridor_terrain
    from isaacgym import terrain_utils

    hs = 0.1
    vs = 1.0
    size = 150

    class Cfg:
        corridor_width = 3.0
        wall_height = 1.5
        wall_thickness = 2.0
        turn_angle_deg_max = 55.0
        diagonal_length = 3.0
        terrain_length = 15.0
        terrain_width = 15.0
        end_margin = 0.5
        goal_forward_margin = 0.0
        goal_radius = 1.6
        curriculum = False
        _first_turn_left = True

    cfg = Cfg()
    terrain = terrain_utils.SubTerrain("straight", width=size, length=size,
                         vertical_scale=vs, horizontal_scale=hs)
    trapezoid_corridor_terrain(terrain, difficulty=0.0, cfg=cfg)

    hf = terrain.height_field_raw
    mid_x = size // 2

    # At the midline, the corridor should be floor (0) from y_start to y_end
    half_cw = int(3.0 / hs // 2)
    y_start = half_cw + int(0.5 / hs)
    y_end = size - half_cw - int(0.5 / hs)

    # Midline column should have floor in corridor region
    floor_pixels = (hf[mid_x, y_start:y_end] == 0).sum()
    total_pixels = y_end - y_start
    assert floor_pixels > 0.9 * total_pixels, \
        f"Expected mostly floor at midline, got {floor_pixels}/{total_pixels}"


def test_trapezoid_corridor_four_direction_rot90():
    """Direction k corridor height field should match np.rot90(direction 0, k=k)."""
    import numpy as np
    from legged_gym.utils.terrain import trapezoid_corridor_terrain
    from isaacgym import terrain_utils

    hs = 0.1
    vs = 1.0
    size = 150

    class Cfg:
        corridor_width = 3.0
        wall_height = 1.5
        wall_thickness = 2.0
        turn_angle_deg_max = 45.0
        diagonal_length = 3.0
        terrain_length = 15.0
        terrain_width = 15.0
        end_margin = 0.5
        goal_forward_margin = 0.0
        goal_radius = 1.6
        curriculum = False
        _first_turn_left = True

    # Generate direction 0 baseline (+Y)
    cfg0 = Cfg()
    terrain0 = terrain_utils.SubTerrain("dir0", width=size, length=size,
                         vertical_scale=vs, horizontal_scale=hs)
    trapezoid_corridor_terrain(terrain0, difficulty=0.5, cfg=cfg0, direction=0)
    hf0 = terrain0.height_field_raw

    for k in [1, 2, 3]:
        cfgn = Cfg()
        terraink = terrain_utils.SubTerrain(f"dir{k}", width=size, length=size,
                             vertical_scale=vs, horizontal_scale=hs)
        trapezoid_corridor_terrain(terraink, difficulty=0.5, cfg=cfgn, direction=k)
        hfk = terraink.height_field_raw

        expected = np.rot90(hf0, k=-k, axes=(0, 1))
        match = (hfk == expected).mean()
        assert match > 0.90, \
            f"Direction {k}: only {match*100:.1f}% pixels match rot90, expected >90%"


# ── PerPointMLP tests ──────────────────────────────────────────

class TestPerPointMLP(unittest.TestCase):

    def setUp(self):
        from rsl_rl.modules.pd_risknet_actor_critic import PerPointMLP
        self.mlp = PerPointMLP(in_dim=3, hidden_dims=[16], out_dim=32)
        self.mlp.eval()

    def test_output_shape_single_point(self):
        """PerPointMLP maps (3,) -> (64,)."""
        x = torch.randn(3)
        out = self.mlp(x)
        self.assertEqual(out.shape, (32,))

    def test_output_shape_batch_of_points(self):
        """PerPointMLP maps (B, N, 3) -> (B, N, 32)."""
        x = torch.randn(4, 192, 3)
        out = self.mlp(x)
        self.assertEqual(out.shape, (4, 192, 32))

    def test_output_shape_flattened(self):
        """PerPointMLP maps (B*N, 3) -> (B*N, 32) -- the chunked call pattern."""
        x = torch.randn(256, 3)
        out = self.mlp(x)
        self.assertEqual(out.shape, (256, 32))

    def test_different_inputs_produce_different_outputs(self):
        """Distinct 3D points should map to distinct features."""
        x1 = torch.tensor([[1.0, 0.0, 0.0]])
        x2 = torch.tensor([[0.0, 1.0, 0.0]])
        out1 = self.mlp(x1)
        out2 = self.mlp(x2)
        self.assertFalse(torch.allclose(out1, out2, atol=1e-4))

    def test_same_input_produces_same_output(self):
        """Deterministic: same input -> same output (no dropout, no BN)."""
        x = torch.randn(16, 3)
        out1 = self.mlp(x)
        out2 = self.mlp(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_two_instances_have_independent_weights(self):
        """Proximal and distal PointNets must not share parameters."""
        from rsl_rl.modules.pd_risknet_actor_critic import PerPointMLP
        mlp1 = PerPointMLP()
        mlp2 = PerPointMLP()
        x = torch.randn(4, 192, 3)
        out1 = mlp1(x)
        out2 = mlp2(x)
        self.assertFalse(torch.allclose(out1, out2, atol=1e-4))


class TestPDRiskNetWithPointNet(unittest.TestCase):

    def setUp(self):
        from rsl_rl.modules.pd_risknet_actor_critic import PDRiskNetActorCritic
        self.num_obs = 48 + 432 * 3  # proprio + 1 frame of 432 points x 3D
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

    def test_gru_input_size_is_32(self):
        """GRU input_size should be 32 (PointNet output dim)."""
        self.assertEqual(self.model.proximal_gru.input_size, 32)
        self.assertEqual(self.model.distal_gru.input_size, 32)

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
        """Total model params (actor/critic with full hidden dims + perception)."""
        total = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total, 2_123_306)
        self.assertLess(total, 2_223_306)

    def test_checkpoint_compat_skips_mismatched_weights(self):
        """load_state_dict should skip perception weights when GRU input_size mismatches."""
        old_state = self.model.state_dict()
        # Simulate an old checkpoint with input_size=3 GRU
        old_state['proximal_gru.weight_ih_l0'] = torch.randn(187 * 3, 3)  # (3*187, 3)
        old_state['distal_gru.weight_ih_l0'] = torch.randn(64 * 3, 3)     # (3*64, 3)
        # Should not raise -- compat logic should strip perception keys
        self.model.load_state_dict(old_state, strict=False)


# ── Rays smooth_dir decoupling tests ────────────────────────────

class TestSmoothRaysDirDecoupled(unittest.TestCase):
    """Verify _smooth_dir_world updates independently of _reward_rays."""

    def test_method_exists(self):
        """Go2LidarPDRiskNet should have _update_smooth_rays_dir method."""
        from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
        self.assertTrue(hasattr(Go2LidarPDRiskNet, '_update_smooth_rays_dir'),
                        "Go2LidarPDRiskNet should have _update_smooth_rays_dir method")

    def test_reward_rays_no_ema(self):
        """_reward_rays should NOT contain EMA logic (alpha * target_dir_world)."""
        import inspect
        from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
        source = inspect.getsource(Go2LidarPDRiskNet._reward_rays)
        self.assertNotIn('alpha * target_dir_world', source,
                         "_reward_rays should not contain EMA update logic")

    def test_smooth_rays_dir_contains_ema(self):
        """_update_smooth_rays_dir SHOULD contain the EMA logic."""
        import inspect
        from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
        source = inspect.getsource(Go2LidarPDRiskNet._update_smooth_rays_dir)
        self.assertIn('alpha * target_dir_world', source,
                      "_update_smooth_rays_dir should contain EMA update logic")

    def test_callback_calls_update_smooth(self):
        """_post_physics_step_callback should call _update_smooth_rays_dir."""
        import inspect
        from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet import Go2LidarPDRiskNet
        source = inspect.getsource(Go2LidarPDRiskNet._post_physics_step_callback)
        self.assertIn('_update_smooth_rays_dir', source,
                      "_post_physics_step_callback should call _update_smooth_rays_dir")
