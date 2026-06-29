# legged_gym/legged_gym/tests/test_go2_xsym.py
"""Tests for get_go2_cmd_safe_xsym_obs_act symmetry function."""

# isaacgym must be imported before torch (it guards against torch being
# already loaded).  We import it at the top so that subsequent module-level
# code (including the exec_module call below) will not trigger its guard.
import importlib.util
import os
import sys

# The go2_cmd_safe module imports isaacgym.torch_utils at its top-level,
# which will trigger the isaacgym import guard.  Explicitly importing
# isaacgym first ensures the guard passes.
import isaacgym  # noqa: F401, E402

import torch
from functools import partial


def _load_sym_func():
    """Load the symmetry function by file path."""
    pkg = "legged_gym.envs.go2.cmd_safe.go2_cmd_safe"
    if pkg in sys.modules:
        return sys.modules[pkg]
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.normpath(os.path.join(
        test_dir, "..", "envs", "go2", "cmd_safe", "go2_cmd_safe.py"))
    spec = importlib.util.spec_from_file_location(pkg, mod_path)
    mod = importlib.util.module_from_spec(spec)
    # The module imports torch at module level, which is fine
    spec.loader.exec_module(mod)
    return mod


_mod = _load_sym_func()
_raw_func = _mod.get_go2_cmd_safe_xsym_obs_act


def _make_func(device="cpu"):
    """Create a partial-bound symmetry function with identity sensor params."""
    return partial(
        _raw_func,
        sensor_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device),
        sensor_trans=torch.zeros(1, 3, device=device),
        proprio_dim=48, proximal_points=256,
        distal_history_points=1280, distal_history_length=10,
    )


class TestScalarMirror:
    """Verify 6 scalar sign flips."""

    def test_vy_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 1], -obs[:, 1])

    def test_wx_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 3], -obs[:, 3])

    def test_wz_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 5], -obs[:, 5])

    def test_gy_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 7], -obs[:, 7])

    def test_cmd_vy_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 10], -obs[:, 10])

    def test_cmd_wz_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 11], -obs[:, 11])


class TestDOFSwap:
    """Verify left-right DOF group swaps."""

    def test_dof_pos_fl_fr(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs[:, 12:15] = 100.0  # FL
        obs[:, 15:18] = 200.0  # FR
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 12:15], torch.full((4, 3), 200.0))
        torch.testing.assert_close(obs_aug[4:, 15:18], torch.full((4, 3), 100.0))

    def test_dof_pos_rl_rr(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs[:, 18:21] = 300.0  # RL
        obs[:, 21:24] = 400.0  # RR
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 18:21], torch.full((4, 3), 400.0))
        torch.testing.assert_close(obs_aug[4:, 21:24], torch.full((4, 3), 300.0))


class TestActionMirror:
    def test_action_swap(self):
        func = _make_func()
        actions = torch.zeros(4, 12)
        actions[:, 0:3] = torch.tensor([1.0, 2.0, 3.0])
        actions[:, 3:6] = torch.tensor([4.0, 5.0, 6.0])
        actions[:, 6:9] = torch.tensor([7.0, 8.0, 9.0])
        actions[:, 9:12] = torch.tensor([10.0, 11.0, 12.0])
        _, aug = func(obs=None, actions=actions)
        torch.testing.assert_close(aug[4:, 0:3], torch.tensor([[4., 5., 6.]]).expand(4, 3))
        torch.testing.assert_close(aug[4:, 3:6], torch.tensor([[1., 2., 3.]]).expand(4, 3))


class TestInvariants:
    def test_batch_doubles(self):
        func = _make_func()
        obs = torch.randn(8, 4656)
        actions = torch.randn(8, 12)
        obs_aug, act_aug = func(obs=obs, actions=actions)
        assert obs_aug.shape[0] == 16
        assert act_aug.shape[0] == 16

    def test_obs_only_mode(self):
        func = _make_func()
        obs = torch.randn(4, 4656)
        obs_aug, act_aug = func(obs=obs, actions=None)
        assert obs_aug.shape == (8, 4656)
        assert act_aug is None

    def test_actions_only_mode(self):
        func = _make_func()
        actions = torch.randn(4, 12)
        obs_aug, act_aug = func(obs=None, actions=actions)
        assert obs_aug is None
        assert act_aug.shape == (8, 12)


class TestNumericalStability:
    def test_large_batch_no_nan(self):
        func = _make_func()
        torch.manual_seed(42)
        obs = torch.randn(2048, 4656)
        actions = torch.randn(2048, 12)
        obs_aug, act_aug = func(obs=obs, actions=actions)
        assert not torch.isnan(obs_aug).any()
        assert not torch.isinf(obs_aug).any()

    def test_zero_lidar_points(self):
        func = _make_func()
        obs = torch.zeros(4, 4656)
        obs[:, :48] = torch.randn(4, 48)
        obs_aug, _ = func(obs=obs, actions=None)
        assert not torch.isnan(obs_aug).any()
