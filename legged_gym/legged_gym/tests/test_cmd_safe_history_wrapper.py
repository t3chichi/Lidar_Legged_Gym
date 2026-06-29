"""Regression tests for CmdSafeHistoryWrapper after geometry extraction."""
import importlib.util
import os
import sys
import torch


def _load_module(pkg_name, rel_path, submodule_search=True):
    """Load a module by file path relative to this test file's directory.

    Parameters
    ----------
    pkg_name : str
        Fully-qualified package name to register in sys.modules
        (e.g. "legged_gym.utils.pointcloud_geometry").
    rel_path : str
        Path relative to this test's parent directory ("..").
    submodule_search : bool
        Whether to allow submodule searches from the loaded module.
    """
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.normpath(os.path.join(test_dir, rel_path))
    kwargs = {}
    if not submodule_search:
        kwargs["submodule_search_locations"] = []
    spec = importlib.util.spec_from_file_location(pkg_name, mod_path, **kwargs)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Ensure pointcloud_geometry is available as a legged_gym.utils submodule
# before loading cmd_safe_history_wrapper which imports from it.
_load_module(
    "legged_gym.utils.pointcloud_geometry",
    os.path.join("..", "utils", "pointcloud_geometry.py"),
    submodule_search=False,
)

# Now load cmd_safe_history_wrapper as a legged_gym.utils submodule.
_mod = _load_module(
    "legged_gym.utils.cmd_safe_history_wrapper",
    os.path.join("..", "utils", "cmd_safe_history_wrapper.py"),
    submodule_search=False,
)

CmdSafeHistoryWrapper = _mod.CmdSafeHistoryWrapper


def test_wrapped_obs_dim():
    w = CmdSafeHistoryWrapper(
        num_envs=4, num_lidar_points=1000, distal_history_length=10,
        proximal_points=256, distal_points=128, phi_threshold_deg=12.0,
        proprio_dim=48, device='cpu',
    )
    assert w.wrapped_obs_dim == 4656


def test_wrap_obs_output_shape():
    w = CmdSafeHistoryWrapper(
        num_envs=2, num_lidar_points=1000, distal_history_length=10,
        proximal_points=256, distal_points=128, phi_threshold_deg=12.0,
        proprio_dim=48, device='cpu',
    )
    obs_raw = torch.zeros(2, 48 + 1000 * 3)
    lidar_base = torch.zeros(2, 1000, 3)
    dones = torch.zeros(2, dtype=torch.bool)
    wrapped = w.wrap_obs(obs_raw, lidar_base, dones)
    assert wrapped.shape == (2, 4656)
    assert not torch.isnan(wrapped).any()


def test_wrap_obs_with_random_data():
    w = CmdSafeHistoryWrapper(
        num_envs=2, num_lidar_points=1000, distal_history_length=10,
        proximal_points=256, distal_points=128, phi_threshold_deg=12.0,
        proprio_dim=48, device='cpu',
    )
    obs_raw = torch.randn(2, 48 + 1000 * 3)
    lidar_base = torch.randn(2, 1000, 3)
    lidar_base[:, :, 2] = torch.randn(2, 1000) * 0.5
    dones = torch.zeros(2, dtype=torch.bool)
    wrapped = w.wrap_obs(obs_raw, lidar_base, dones)
    assert wrapped.shape == (2, 4656)


def test_done_resets_history():
    w = CmdSafeHistoryWrapper(
        num_envs=2, num_lidar_points=1000, distal_history_length=10,
        proximal_points=256, distal_points=128, phi_threshold_deg=12.0,
        proprio_dim=48, device='cpu',
    )
    obs_raw = torch.randn(2, 48 + 1000 * 3)
    lidar_base = torch.randn(2, 1000, 3)
    dones = torch.tensor([True, False])
    wrapped = w.wrap_obs(obs_raw, lidar_base, dones)
    # Just verify no crash and shape correct for done reset
    assert wrapped.shape == (2, 4656)
