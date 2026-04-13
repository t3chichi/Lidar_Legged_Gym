import math


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
    assert env_cfg.pd_risknet.n_sectors == 36
