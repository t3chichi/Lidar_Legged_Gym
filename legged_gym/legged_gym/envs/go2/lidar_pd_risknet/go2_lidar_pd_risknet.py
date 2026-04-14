import torch
import math
import numpy as np
import warp as wp

from isaacgym.torch_utils import quat_rotate_inverse, quat_apply, quat_mul, quat_from_euler_xyz
from isaacgym import gymapi, gymutil

from legged_gym.envs.go2.go2 import Go2
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType


class Go2LidarPDRiskNet(Go2):
    """Go2 environment extension for the LiDAR + PD-RiskNet training pipeline.

    This class is compatibility-first: it preserves the parent Go2 behavior and only
    appends dedicated observation channels required by the new task.
    """

    def _init_buffers(self):
        super()._init_buffers()
        # Enable per-step debug drawing for this task when viewer is available.
        self.debug_viz = True
        self._init_pd_risknet_buffers()
        self._init_lidar_sensor()

    def _get_noise_scale_vec(self, cfg):
        """Use Go2 proprio noise only; keep LiDAR history channels noise-free by default.

        Base LeggedRobot assumes height-map observations at indices [48:235] when
        terrain.measure_heights=True. This task replaces that block with flattened
        LiDAR history, so we override the mapping to avoid injecting wrong noise.
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.0
        return noise_vec

    def _init_pd_risknet_buffers(self):
        cfg = self.cfg.pd_risknet
        self.lidar_history = torch.zeros(
            self.num_envs,
            int(cfg.history_length),
            int(cfg.num_lidar_points),
            3,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self.lidar_points_base = torch.zeros(
            self.num_envs,
            int(cfg.num_lidar_points),
            3,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self.raycast_distances = torch.full(
            (self.num_envs, int(cfg.num_lidar_points)),
            float(cfg.ray_max_distance),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self.v_avoid = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float, requires_grad=False)

    def _init_lidar_sensor(self):
        if not getattr(self.cfg.raycaster, "enable_raycast", False):
            self.lidar_sensor = None
            return

        wp.init()

        # Build Warp mesh and mesh_ids (official sample-compatible path).
        if hasattr(self, "terrain") and hasattr(self.terrain, "vertices") and hasattr(self.terrain, "triangles"):
            vertices = torch.as_tensor(self.terrain.vertices, device=self.device, dtype=torch.float32).clone()
            if hasattr(self.cfg.terrain, "border_size"):
                vertices[:, 0] -= self.cfg.terrain.border_size
                vertices[:, 1] -= self.cfg.terrain.border_size
            triangles_i32 = np.asarray(self.terrain.triangles, dtype=np.int32)
        elif self.cfg.terrain.mesh_type == "plane":
            # Plane terrain does not expose mesh buffers by default, so build a simple ground mesh here.
            plane_size = 100.0
            vertices = torch.tensor(
                [
                    [-plane_size, -plane_size, 0.0],
                    [plane_size, -plane_size, 0.0],
                    [plane_size, plane_size, 0.0],
                    [-plane_size, plane_size, 0.0],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            triangles_i32 = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        else:
            raise ValueError("go2_lidar_pd_risknet requires trimesh terrain vertices/triangles or plane terrain for lidar rendering")

        self._wp_mesh = wp.Mesh(
            points=wp.from_torch(vertices, dtype=wp.vec3),
            indices=wp.from_numpy(triangles_i32.flatten(), dtype=wp.int32, device=self.device),
        )
        self.mesh_ids = wp.array([self._wp_mesh.id], dtype=wp.uint64)

        self.sensor_pos_tensor = torch.zeros_like(self.base_pos)
        self.sensor_quat_tensor = torch.zeros_like(self.base_quat)

        ray_cfg = self.cfg.raycaster
        lidar_cfg = LidarConfig(
            sensor_type=LidarType.SIMPLE_GRID,
            dt=float(self.dt),
            update_frequency=max(1.0, 1.0 / float(self.dt)),
            max_range=float(ray_cfg.max_distance),
            min_range=0.05,
            num_sensors=1,
            horizontal_line_num=int(ray_cfg.spherical_num_azimuth),
            vertical_line_num=int(ray_cfg.spherical_num_elevation),
            horizontal_fov_deg_min=-180.0,
            horizontal_fov_deg_max=180.0,
            vertical_fov_deg_min=-90.0,
            vertical_fov_deg_max=90.0,
            return_pointcloud=True,
            pointcloud_in_world_frame=False,
            randomize_placement=False,
            enable_sensor_noise=False,
        )

        lidar_env = {
            "device": self.device,
            "num_envs": self.num_envs,
            "num_sensors": 1,
            "sensor_pos_tensor": self.sensor_pos_tensor,
            "sensor_quat_tensor": self.sensor_quat_tensor,
            "mesh_ids": self.mesh_ids,
        }
        self.lidar_sensor = LidarSensor(lidar_env, None, lidar_cfg, num_sensors=1, device=self.device)
        self._sensor_translation = torch.tensor(list(ray_cfg.offset_pos), dtype=torch.float32, device=self.device).view(1, 3).repeat(self.num_envs, 1)
        rpy = getattr(ray_cfg, "sensor_offset_rpy", [0.0, 0.0, 0.0])
        offset_q = quat_from_euler_xyz(
            torch.tensor(float(rpy[0]), device=self.device),
            torch.tensor(float(rpy[1]), device=self.device),
            torch.tensor(float(rpy[2]), device=self.device),
        )
        self._sensor_offset_quat = offset_q.view(1, 4).repeat(self.num_envs, 1)

    def _update_lidar_history(self):
        if self.lidar_sensor is None:
            return

        # Keep sensor attached to base with configurable translation offset.
        self.sensor_quat_tensor.copy_(quat_mul(self.base_quat, self._sensor_offset_quat))
        self.sensor_pos_tensor.copy_(self.base_pos + quat_apply(self.base_quat, self._sensor_translation))
        
        lidar_points, lidar_dist = self.lidar_sensor.update()
        points_sensor = lidar_points.view(self.num_envs, -1, 3)
        n_points = points_sensor.shape[1]
        sensor_quat_repeat = self._sensor_offset_quat.unsqueeze(1).repeat(1, n_points, 1).reshape(-1, 4)
        points_base = quat_apply(sensor_quat_repeat, points_sensor.reshape(-1, 3)).reshape(self.num_envs, n_points, 3)
        points_base = points_base + self._sensor_translation.unsqueeze(1)
        dist = lidar_dist.view(self.num_envs, -1)
        valid = dist > 0.0

        # LiDAR domain randomization from paper appendix.
        mask_ratio = float(getattr(self.cfg.domain_rand, "lidar_point_mask_ratio", 0.0))
        if mask_ratio > 0.0:
            rand_mask = torch.rand_like(dist) < mask_ratio
            lo, hi = self.cfg.domain_rand.lidar_point_mask_value_range
            fake_dist = torch.rand_like(dist) * (hi - lo) + lo
            dir_norm = torch.linalg.norm(points_base, dim=-1, keepdim=True).clamp(min=1.0e-6)
            dir_unit = points_base / dir_norm
            points_base = torch.where(rand_mask.unsqueeze(-1), dir_unit * fake_dist.unsqueeze(-1), points_base)
            dist = torch.where(rand_mask, fake_dist, dist)

        noise_ratio = float(getattr(self.cfg.domain_rand, "lidar_distance_noise_ratio", 0.0))
        if noise_ratio > 0.0:
            scale = 1.0 + (2.0 * torch.rand_like(dist) - 1.0) * noise_ratio
            points_base = points_base * scale.unsqueeze(-1)
            dist = dist * scale

        # If a ray is invalid, keep max distance endpoint behavior.
        dist = torch.where(valid, dist, torch.full_like(dist, float(self.cfg.pd_risknet.ray_max_distance)))

        self.lidar_points_base.copy_(points_base)
        self.raycast_distances.copy_(dist)
        # Use roll to avoid overlapping in-place memory writes during history shift.
        self.lidar_history = torch.roll(self.lidar_history, shifts=-1, dims=1)
        self.lidar_history[:, -1].copy_(self.lidar_points_base)

    def _compute_v_avoid(self):
        cfg = self.cfg.pd_risknet
        n_sec = int(cfg.n_sectors)
        sec_size = 2.0 * math.pi / n_sec

        pts = self.lidar_points_base[..., :2]
        dist = torch.linalg.norm(pts, dim=-1)
        angles = torch.atan2(pts[..., 1], pts[..., 0])
        sec_ids = torch.floor((angles + math.pi) / sec_size).long().clamp(min=0, max=n_sec - 1)

        inf = torch.full_like(dist, 1.0e6)
        min_dist_per_sec = []
        for sec in range(n_sec):
            sec_mask = sec_ids == sec
            sec_min = torch.where(sec_mask, dist, inf).min(dim=1).values
            min_dist_per_sec.append(sec_min)
        min_dist_per_sec = torch.stack(min_dist_per_sec, dim=1)

        sec_centers = torch.linspace(-math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size, n_sec, device=self.device)
        away_dirs = torch.stack((-torch.cos(sec_centers), -torch.sin(sec_centers)), dim=-1).unsqueeze(0)

        active = min_dist_per_sec < float(cfg.avoid_distance_thresh)
        mag = torch.exp(-min_dist_per_sec * float(cfg.avoid_alpha)) * active.float()
        self.v_avoid = torch.sum(away_dirs * mag.unsqueeze(-1), dim=1)

    def _compute_pd_risknet_features(self):
        self._compute_v_avoid()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_lidar_history()
        # Reward is computed before compute_observations in LeggedRobot.post_physics_step,
        # so V_avoid must be refreshed here to avoid one-step lag.
        self._compute_v_avoid()
    
    def check_termination(self):
        super().check_termination()
        if not getattr(self.cfg.env, "enable_fall_termination", False):
            return

        # projected_gravity[:, 2] is close to -1 when upright and increases as the robot flips.
        g_thresh = float(getattr(self.cfg.env, "fall_projected_gravity_z_threshold", -0.1))
        h_thresh = float(getattr(self.cfg.env, "fall_base_height_threshold", 0.12))

        flipped = self.projected_gravity[:, 2] > g_thresh
        low_base = self.base_pos[:, 2] < h_thresh
        self.reset_buf |= (flipped | low_base)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.lidar_history[env_ids] = 0.0
        self.lidar_points_base[env_ids] = 0.0
        self.raycast_distances[env_ids] = float(self.cfg.pd_risknet.ray_max_distance)
        self.v_avoid[env_ids] = 0.0

    def _reward_vel_avoid(self):
        cfg = self.cfg.pd_risknet
        vel_target = self.commands[:, :2] + self.v_avoid
        vel_err = torch.sum(torch.square(self.base_lin_vel[:, :2] - vel_target), dim=1)
        return torch.exp(-float(cfg.avoid_beta) * vel_err)

    def _reward_rays(self):
        d_max = float(self.cfg.pd_risknet.ray_max_distance)
        clipped = torch.clamp(self.raycast_distances, max=d_max)
        return torch.mean(clipped / d_max, dim=1)

    def _reward_action_rate2(self):
        if not hasattr(self, "last_last_actions"):
            self.last_last_actions = torch.zeros_like(self.actions)
        rate2 = self.actions - 2.0 * self.last_actions + self.last_last_actions
        self.last_last_actions[:] = self.last_actions
        return torch.sum(torch.square(rate2), dim=1)

    def compute_observations(self):
        # Keep base proprio order identical to Go2/LeggedRobot, then append LiDAR history.
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.lidar_history.reshape(self.num_envs, -1),
        ), dim=-1)
        self._compute_pd_risknet_features()

        # Privileged channel: terrain height samples for train-time-only supervision.
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = self.measured_heights

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _draw_debug_vis(self):
        """Draw LiDAR points and velocity vectors for quick policy debugging."""
        if self.viewer is None:
            return

        self.gym.clear_lines(self.viewer)
        env_id = 0

        # Draw a lightweight subset of LiDAR points in world frame.
        pts_base = self.lidar_points_base[env_id]
        num_pts = pts_base.shape[0]
        max_draw = min(256, num_pts)
        if max_draw > 0:
            step = max(1, num_pts // max_draw)
            idx = torch.arange(0, num_pts, step, device=self.device)[:max_draw]
            pts_sel = pts_base[idx]
            base_pos = self.base_pos[env_id].unsqueeze(0).repeat(pts_sel.shape[0], 1)
            base_quat = self.base_quat[env_id].unsqueeze(0).repeat(pts_sel.shape[0], 1)
            pts_world = base_pos + quat_apply(base_quat, pts_sel)

            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
            pts_np = pts_world.detach().cpu().numpy()
            for p in pts_np:
                sphere_pose = gymapi.Transform(gymapi.Vec3(float(p[0]), float(p[1]), float(p[2])), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

        # Draw command direction (green) and avoidance direction (yellow).
        start = self.base_pos[env_id].detach().cpu().numpy()
        cmd_xy = self.commands[env_id, :2].detach().cpu().numpy()
        avoid_xy = self.v_avoid[env_id].detach().cpu().numpy()

        cmd_vec = np.array([cmd_xy[0], cmd_xy[1], 0.0], dtype=np.float32)
        avoid_vec = np.array([avoid_xy[0], avoid_xy[1], 0.0], dtype=np.float32)

        cmd_norm = np.linalg.norm(cmd_vec[:2])
        avoid_norm = np.linalg.norm(avoid_vec[:2])
        if cmd_norm > 1.0e-6:
            self.vis.draw_arrow(env_id, start.tolist(), (start + 0.6 * cmd_vec / cmd_norm).tolist(), width=0.01, color=(0, 1, 0))
        if avoid_norm > 1.0e-6:
            self.vis.draw_arrow(env_id, start.tolist(), (start + 0.6 * avoid_vec / avoid_norm).tolist(), width=0.01, color=(1, 1, 0))
