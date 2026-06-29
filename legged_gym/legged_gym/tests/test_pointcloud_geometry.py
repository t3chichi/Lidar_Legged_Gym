# legged_gym/legged_gym/tests/test_pointcloud_geometry.py
"""Unit tests for pointcloud_geometry module (pure torch, no IsaacGym)."""
import importlib.util
import math
import os
import sys
import torch


def _load_module():
    """Load pointcloud_geometry.py directly by file path, bypassing the
    legged_gym.utils package __init__.py (which pulls in isaacgym via
    helpers.py)."""
    pkg = "legged_gym.utils.pointcloud_geometry"
    if pkg in sys.modules:
        return sys.modules[pkg]
    # Resolve the file path relative to this test file.
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.normpath(os.path.join(test_dir, "..", "utils", "pointcloud_geometry.py"))
    spec = importlib.util.spec_from_file_location(pkg, mod_path,
                                                  submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
quaternion_conjugate = _mod.quaternion_conjugate
quaternion_apply = _mod.quaternion_apply
cartesian_to_spherical = _mod.cartesian_to_spherical
to_sensor_frame = _mod.to_sensor_frame
sort_points_by_angular_key = _mod.sort_points_by_angular_key


class TestQuaternion:
    def test_conjugate_signs(self):
        q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        qc = quaternion_conjugate(q)
        assert qc[0, 0] == -1.0 and qc[0, 1] == -2.0
        assert qc[0, 2] == -3.0 and qc[0, 3] == 4.0

    def test_identity_conjugate(self):
        q_id = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        torch.testing.assert_close(quaternion_conjugate(q_id), q_id)

    def test_conjugate_batched(self):
        q = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        qc = quaternion_conjugate(q)
        assert qc.shape == q.shape
        assert (qc[:, :3] == -q[:, :3]).all()
        assert (qc[:, 3] == q[:, 3]).all()


class TestQuaternionApply:
    def test_identity_rotation(self):
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(quaternion_apply(q, v), v, atol=1e-6, rtol=1e-6)

    def test_90deg_z_rotation(self):
        s = math.sqrt(2) / 2
        q = torch.tensor([[0.0, 0.0, s, s]])  # 90 deg around z
        v = torch.tensor([[1.0, 0.0, 0.0]])
        torch.testing.assert_close(quaternion_apply(q, v),
                                   torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5, rtol=0)

    def test_batched(self):
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        rotated = quaternion_apply(q, v)
        assert rotated.shape == (2, 3)


class TestCartesianToSpherical:
    def test_x_axis(self):
        pts = torch.tensor([[[1.0, 0.0, 0.0]]])
        r, az, phi = cartesian_to_spherical(pts)
        torch.testing.assert_close(r, torch.tensor([[1.0]]), atol=1e-5, rtol=0)
        torch.testing.assert_close(az, torch.tensor([[0.0]]), atol=1e-5, rtol=0)
        torch.testing.assert_close(phi, torch.tensor([[0.0]]), atol=1e-5, rtol=0)

    def test_y_axis(self):
        pts = torch.tensor([[[0.0, 1.0, 0.0]]])
        r, az, phi = cartesian_to_spherical(pts)
        torch.testing.assert_close(az, torch.tensor([[math.pi / 2]]), atol=1e-5, rtol=0)

    def test_z_above(self):
        pts = torch.tensor([[[0.0, 0.0, 1.0]]])
        r, az, phi = cartesian_to_spherical(pts)
        torch.testing.assert_close(phi, torch.tensor([[math.pi / 2]]), atol=1e-5, rtol=0)

    def test_nan_guard(self):
        pts = torch.tensor([[[float('nan'), 0.0, 0.0]]])
        r, az, phi = cartesian_to_spherical(pts)
        assert not torch.isnan(r).any()
        assert not torch.isnan(az).any()
        assert not torch.isnan(phi).any()

    def test_inf_guard(self):
        pts = torch.tensor([[[float('inf'), 0.0, 0.0]]])
        r, az, phi = cartesian_to_spherical(pts)
        assert not torch.isinf(az).any()
        assert not torch.isinf(phi).any()


class TestToSensorFrame:
    def test_no_offset(self):
        pts = torch.randn(2, 10, 3)
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        t = torch.zeros(1, 3)
        out = to_sensor_frame(pts, q, t)
        torch.testing.assert_close(out, pts, atol=1e-5, rtol=1e-5)

    def test_translation_only(self):
        pts = torch.tensor([[[5.0, 0.0, 0.0]]])
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        t = torch.tensor([[2.0, 0.0, 0.0]])
        out = to_sensor_frame(pts, q, t)
        torch.testing.assert_close(out, torch.tensor([[[3.0, 0.0, 0.0]]]))


class TestSortPointsByAngularKey:
    def test_preserves_values(self):
        torch.manual_seed(42)
        pts = torch.randn(4, 100, 3)
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        t = torch.zeros(1, 3)
        s = sort_points_by_angular_key(pts, q, t)
        assert s.shape == pts.shape
        s_sorted, _ = torch.sort(s.reshape(4, -1), dim=1)
        p_sorted, _ = torch.sort(pts.reshape(4, -1), dim=1)
        torch.testing.assert_close(s_sorted, p_sorted, atol=1e-5, rtol=0)

    def test_y_flip_sorting(self):
        torch.manual_seed(42)
        pts = torch.randn(2, 50, 3)
        mir = pts.clone()
        mir[:, :, 1] = -mir[:, :, 1]
        s = sort_points_by_angular_key(mir,
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.zeros(1, 3))
        assert s.shape == pts.shape
        assert not torch.isnan(s).any()
