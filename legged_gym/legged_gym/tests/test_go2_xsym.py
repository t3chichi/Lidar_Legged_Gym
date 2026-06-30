# legged_gym/legged_gym/tests/test_go2_xsym.py
"""Tests for get_go2_cmd_safe_xsym_obs_act symmetry function.

All dimension parameters are read from the production config file
(go2_cmd_safe_config.py) to guarantee the tests match the actual training
observation layout produced by CmdSafeHistoryWrapper.
"""

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

# ── Read production parameters from the actual config ──
# _load_sym_func() already triggered the full legged_gym package import
# chain, so we can safely import the config module here.
from legged_gym.envs.go2.cmd_safe.go2_cmd_safe_config import (  # noqa: E402
    PD_PROPRIO_DIM, PD_PROXIMAL_POINTS, PD_DISTAL_POINTS,
    DIST_HISTORY_LENGTH, MEASURED_GRID_X_COUNT, MEASURED_GRID_Y_COUNT,
    PD_PRIV_HEIGHT_DIM,
)

# Production values – any change to the config constants above will
# automatically flow into these and into every test that uses them.
PROD_PROPRIO_DIM = PD_PROPRIO_DIM
PROD_PROXIMAL_POINTS = PD_PROXIMAL_POINTS
PROD_DISTAL_POINTS = PD_DISTAL_POINTS
PROD_DISTAL_HISTORY_LENGTH = DIST_HISTORY_LENGTH
PROD_DISTAL_HISTORY_POINTS = PD_DISTAL_POINTS * DIST_HISTORY_LENGTH
PROD_GRID_X = MEASURED_GRID_X_COUNT
PROD_GRID_Y = MEASURED_GRID_Y_COUNT
PROD_POLICY_OBS_DIM = (
    PROD_PROPRIO_DIM
    + PROD_PROXIMAL_POINTS * 3
    + PROD_DISTAL_HISTORY_POINTS * 3
)
PROD_CRITIC_OBS_DIM = PD_PRIV_HEIGHT_DIM  # x_count * y_count

# Go2 default joint angles [rad]: FL(hip,thigh,calf), FR(hip,thigh,calf),
#                                RL(hip,thigh,calf), RR(hip,thigh,calf)
GO2_DEFAULT_DOF_POS = torch.tensor(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5]
)


def _make_func(device="cpu"):
    """Create a partial-bound symmetry function with identity sensor params.

    All dimension parameters are bound to the production config values so
    that the tests validate exactly what the training loop sees.
    """
    return partial(
        _raw_func,
        sensor_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device),
        sensor_trans=torch.zeros(1, 3, device=device),
        proprio_dim=PROD_PROPRIO_DIM,
        proximal_points=PROD_PROXIMAL_POINTS,
        distal_history_points=PROD_DISTAL_HISTORY_POINTS,
        distal_history_length=PROD_DISTAL_HISTORY_LENGTH,
        height_grid_x_count=PROD_GRID_X,
        height_grid_y_count=PROD_GRID_Y,
        default_dof_pos=GO2_DEFAULT_DOF_POS,
        dof_obs_scale=1.0,
        action_scale=0.3,
    )


class TestScalarMirror:
    """Verify 6 scalar sign flips."""

    def test_vy_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 1], -obs[:, 1])

    def test_wx_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 3], -obs[:, 3])

    def test_wz_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 5], -obs[:, 5])

    def test_gy_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 7], -obs[:, 7])

    def test_cmd_vy_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 10], -obs[:, 10])

    def test_cmd_wz_sign_flip(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs_aug, _ = func(obs=obs, actions=None)
        torch.testing.assert_close(obs_aug[4:, 11], -obs[:, 11])


class TestDOFSwap:
    """Verify left-right DOF group swaps."""

    def test_dof_pos_fl_fr(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs[:, 12:15] = 100.0  # FL
        obs[:, 15:18] = 200.0  # FR
        obs_aug, _ = func(obs=obs, actions=None)
        # Hip: corrected for asymmetric defaults (±0.2 rad), thigh/calf: direct copy
        expected_fl = torch.tensor([[199.8, 200.0, 200.0]]).expand(4, 3)
        expected_fr = torch.tensor([[100.2, 100.0, 100.0]]).expand(4, 3)
        torch.testing.assert_close(obs_aug[4:, 12:15], expected_fl)
        torch.testing.assert_close(obs_aug[4:, 15:18], expected_fr)

    def test_dof_pos_rl_rr(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs[:, 18:21] = 300.0  # RL
        obs[:, 21:24] = 400.0  # RR
        obs_aug, _ = func(obs=obs, actions=None)
        expected_rl = torch.tensor([[399.8, 400.0, 400.0]]).expand(4, 3)
        expected_rr = torch.tensor([[300.2, 300.0, 300.0]]).expand(4, 3)
        torch.testing.assert_close(obs_aug[4:, 18:21], expected_rl)
        torch.testing.assert_close(obs_aug[4:, 21:24], expected_rr)


class TestActionMirror:
    def test_action_swap(self):
        func = _make_func()
        actions = torch.zeros(4, 12)
        actions[:, 0:3] = torch.tensor([1.0, 2.0, 3.0])
        actions[:, 3:6] = torch.tensor([4.0, 5.0, 6.0])
        actions[:, 6:9] = torch.tensor([7.0, 8.0, 9.0])
        actions[:, 9:12] = torch.tensor([10.0, 11.0, 12.0])
        _, aug = func(obs=None, actions=actions)
        # Hip: corrected for asymmetric defaults (±0.667 action units), thigh/calf: unchanged
        exp_fl = torch.tensor([[3.3333333, 5.0, 6.0]]).expand(4, 3)
        exp_fr = torch.tensor([[1.6666667, 2.0, 3.0]]).expand(4, 3)
        torch.testing.assert_close(aug[4:, 0:3], exp_fl)
        torch.testing.assert_close(aug[4:, 3:6], exp_fr)


class TestInvariants:
    def test_batch_doubles(self):
        func = _make_func()
        obs = torch.randn(8, PROD_POLICY_OBS_DIM)
        actions = torch.randn(8, 12)
        obs_aug, act_aug = func(obs=obs, actions=actions)
        assert obs_aug.shape[0] == 16
        assert act_aug.shape[0] == 16

    def test_obs_only_mode(self):
        func = _make_func()
        obs = torch.randn(4, PROD_POLICY_OBS_DIM)
        obs_aug, act_aug = func(obs=obs, actions=None)
        assert obs_aug.shape == (8, PROD_POLICY_OBS_DIM)
        assert act_aug is None

    def test_actions_only_mode(self):
        func = _make_func()
        actions = torch.randn(4, 12)
        obs_aug, act_aug = func(obs=None, actions=actions)
        assert obs_aug is None
        assert act_aug.shape == (8, 12)

    def test_double_mirror_is_exact_identity(self):
        """Mirror applied twice must recover the original EXACTLY.

        Pre-condition: LiDAR points must be angular-sorted (as the
        CmdSafeHistoryWrapper produces), matching production conditions.
        """
        func = _make_func()
        # Load sort_points_by_angular_key via importlib (bypasses __init__.py)
        mod_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "utils", "pointcloud_geometry.py"))
        spec = importlib.util.spec_from_file_location(
            "pgeo", mod_path, submodule_search_locations=[])
        pgeo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pgeo)
        sort_fn = pgeo.sort_points_by_angular_key

        torch.manual_seed(42)
        proprio = torch.randn(4, PROD_PROPRIO_DIM)
        prox_raw = torch.randn(4, PROD_PROXIMAL_POINTS, 3)
        dist_raw = torch.randn(4, PROD_DISTAL_HISTORY_POINTS, 3)
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device='cpu')
        t = torch.zeros(1, 3)
        prox_sorted = sort_fn(prox_raw, q, t)
        dist_sorted = sort_fn(dist_raw, q, t)

        obs = torch.cat([
            proprio,
            prox_sorted.reshape(4, -1),
            dist_sorted.reshape(4, -1),
        ], dim=1)

        # First mirror
        obs_aug, _ = func(obs=obs, actions=None)
        obs_m1 = obs_aug[4:]
        # Second mirror on the mirrored half
        obs_aug2, _ = func(obs=obs_m1, actions=None)
        obs_m2 = obs_aug2[4:]

        # Must be identity (allow FP epsilon from action_scale division)
        diff = (obs_m2 - obs).abs()
        assert diff.max().item() < 1e-5, (
            f"Double mirror not identity, max diff={diff.max().item():.2e}"
        )


class TestNumericalStability:
    def test_large_batch_no_nan(self):
        func = _make_func()
        torch.manual_seed(42)
        obs = torch.randn(2048, PROD_POLICY_OBS_DIM)
        actions = torch.randn(2048, 12)
        obs_aug, act_aug = func(obs=obs, actions=actions)
        assert not torch.isnan(obs_aug).any()
        assert not torch.isinf(obs_aug).any()

    def test_zero_lidar_points(self):
        func = _make_func()
        obs = torch.zeros(4, PROD_POLICY_OBS_DIM)
        obs[:, :PROD_PROPRIO_DIM] = torch.randn(4, PROD_PROPRIO_DIM)
        obs_aug, _ = func(obs=obs, actions=None)
        assert not torch.isnan(obs_aug).any()


class TestAuxiliaryObs:
    """Verify auxiliary observation (height grid) symmetry.

    The auxiliary observation (187-dim measured heights) is used for
    privileged height supervision (MSE loss), NOT for the critic value
    function.  The critic shares the actor's latent features.
    """

    def test_auxiliary_grid_shape(self):
        """Obs type auxiliary should double the batch and preserve obs dim."""
        func = _make_func()
        obs = torch.randn(4, PROD_CRITIC_OBS_DIM)
        obs_aug, act_aug = func(obs=obs, actions=None, obs_type="auxiliary")
        assert obs_aug.shape == (8, PROD_CRITIC_OBS_DIM)
        assert act_aug is None

    def test_auxiliary_grid_y_flip(self):
        """Each row of the height grid should be Y-reversed."""
        func = _make_func()
        obs = torch.zeros(2, PROD_CRITIC_OBS_DIM)
        for x in range(PROD_GRID_X):
            for y in range(PROD_GRID_Y):
                obs[:, x * PROD_GRID_Y + y] = float(x * 100 + y)
        obs_aug, _ = func(obs=obs, actions=None, obs_type="auxiliary")
        mirrored = obs_aug[2:]
        for x in range(PROD_GRID_X):
            for y in range(PROD_GRID_Y):
                orig_val = obs[0, x * PROD_GRID_Y + y].item()
                mirr_val = mirrored[0, x * PROD_GRID_Y + (PROD_GRID_Y - 1 - y)].item()
                assert mirr_val == orig_val, (
                    f"x={x} y={y}: expected {orig_val}, got {mirr_val}"
                )

    def test_auxiliary_grid_double_mirror_identity(self):
        """Double mirror of height grid must be exact identity."""
        func = _make_func()
        obs = torch.randn(4, PROD_CRITIC_OBS_DIM)
        obs_aug, _ = func(obs=obs, actions=None, obs_type="auxiliary")
        obs_m1 = obs_aug[4:]
        obs_aug2, _ = func(obs=obs_m1, actions=None, obs_type="auxiliary")
        obs_m2 = obs_aug2[4:]
        diff = (obs_m2 - obs).abs()
        assert diff.max().item() == 0.0, (
            f"Double mirror not identity, max diff={diff.max().item():.2e}"
        )


class TestHipAsymmetry:
    """Verify hip joint default angle correction in DOF/action swaps.

    Go2 has asymmetric hip defaults: FL/RL = +0.1, FR/RR = -0.1.
    The symmetry function must account for this when swapping DOFs.
    """

    def test_dof_pos_hip_correction(self):
        """Mirrored FL hip obs should subtract the default difference."""
        func = _make_func()
        obs = torch.zeros(4, PROD_POLICY_OBS_DIM)
        # FR_dof = 0.0 → obs[FR_hip] = (0 - (-0.1)) * 1.0 = 0.1
        obs[:, 15] = 0.1
        obs_aug, _ = func(obs=obs, actions=None)
        mirrored = obs_aug[4:]
        # Correct mirror: FL_hip obs = (FR_dof - FL_default) * scale
        #   = (0 - 0.1) * 1.0 = -0.1
        torch.testing.assert_close(mirrored[:, 12], torch.full((4,), -0.1))
        # Old bug produced +0.1
        assert not torch.allclose(mirrored[:, 12], torch.full((4,), 0.1)), (
            "Bug: hip default asymmetry not corrected"
        )

    def test_action_hip_correction(self):
        """Mirrored hip action accounts for asymmetric default."""
        func = _make_func()
        actions = torch.zeros(4, 12)
        # FL_hip action = 0.5 → PD target = 0.5*0.3 + 0.1 = 0.25 rad
        # Mirror FL→FR: FR target should = 0.25 rad
        #   FR_action * 0.3 + (-0.1) = 0.25 → FR_action = 1.167
        actions[:, 0] = 0.5
        _, aug = func(obs=None, actions=actions)
        torch.testing.assert_close(aug[4:, 3], torch.full((4,), 1.1666666666666667))
        # Old bug produced 0.5 (direct copy)

    def test_thigh_calf_no_correction(self):
        """Thigh and calf defaults are symmetric, no spurous correction."""
        func = _make_func()
        obs = torch.zeros(4, PROD_POLICY_OBS_DIM)
        obs[:, 13] = 0.5  # FL_thigh
        obs[:, 16] = 0.8  # FR_thigh
        obs_aug, _ = func(obs=obs, actions=None)
        mirrored = obs_aug[4:]
        # Symmetric defaults (both 0.8) → direct copy
        torch.testing.assert_close(mirrored[:, 13], torch.full((4,), 0.8))
        torch.testing.assert_close(mirrored[:, 16], torch.full((4,), 0.5))
