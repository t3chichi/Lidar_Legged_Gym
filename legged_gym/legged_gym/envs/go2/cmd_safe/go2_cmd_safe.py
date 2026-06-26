import torch
import math
import numpy as np
import warp as wp

from isaacgym.torch_utils import quat_rotate_inverse, quat_apply, quat_mul, quat_from_euler_xyz, quat_from_angle_axis, torch_rand_float
from isaacgym import gymapi, gymtorch, gymutil

from legged_gym.envs.go2.go2 import Go2
from legged_gym.utils.math_utils import quat_apply_yaw
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType


class Go2CmdSafe(Go2):
    """Go2 environment with command-safe velocity reward.

    LiDAR at body centre [0,0,0] with horizontal orientation.
    Replaces vel_avoid/rays with cmd_safe_vel + sector_dist_penalty.
    Inherits collision replay, stuck detection, terrain curriculum from
    the copied Go2LidarPDRiskNet code.
    """

    def _init_buffers(self):
        if not getattr(self.cfg.pd_risknet, "soft_pretrain", False):
            self.cfg.env.episode_length_s = 15
        super()._init_buffers()
        self.debug_viz = True
        self._init_pd_risknet_buffers()
        self._init_cmd_safe_buffers()
        self._init_lidar_sensor()
        self._init_replay_buffers()
        if not hasattr(self, '_spawn_angles'):
            self._spawn_angles = None

    def _init_pd_risknet_buffers(self):
        cfg = self.cfg.pd_risknet
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
        self._raw_distances = torch.full(
            (self.num_envs, int(cfg.num_lidar_points)),
            float(cfg.ray_max_distance),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

        self.last_dist = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self._last_channel_pos = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self._pillar_boxes = []
        self._consecutive_upgrade_count = torch.zeros(
            self.num_envs, device=self.device,
            dtype=torch.int32, requires_grad=False)
        self._consecutive_downgrade_count = torch.zeros(
            self.num_envs, device=self.device,
            dtype=torch.int32, requires_grad=False)

        # Channel forward direction per env (based on terrain_type / col index)
        _FORWARD_LOOKUP_TABLE = torch.tensor([
            [0.0, 1.0],    # direction 0: +Y (north)
            [1.0, 0.0],    # direction 1: +X (east)
            [0.0, -1.0],   # direction 2: -Y (south)
            [-1.0, 0.0],   # direction 3: -X (west)
        ], device=self.device, dtype=torch.float)
        if hasattr(self, "terrain_types"):
            _safe_idx = self.terrain_types.long().clamp(0, 3)
            self._channel_forward = _FORWARD_LOOKUP_TABLE[_safe_idx]
        else:
            self._channel_forward = torch.zeros(
                self.num_envs, 2, device=self.device, dtype=torch.float)

        # Per-cell goal offsets table (world frame, relative to env_origin)
        if hasattr(self, "terrain") and hasattr(self.terrain, "goal_offsets") and np.any(self.terrain.goal_offsets):
            self._goal_offsets_table = torch.from_numpy(
                self.terrain.goal_offsets).to(self.device).to(torch.float)
        else:
            self._goal_offsets_table = None

    def _init_replay_buffers(self):
        """滚动状态缓冲区 + 碰撞标志，供碰撞回放机制使用。"""
        self.replay_len = 100
        self.replay_root_states = torch.zeros(
            self.num_envs, self.replay_len, 13, device=self.device)
        self.replay_dof_pos = torch.zeros(
            self.num_envs, self.replay_len, self.num_dof, device=self.device)
        self.replay_dof_vel = torch.zeros(
            self.num_envs, self.replay_len, self.num_dof, device=self.device)

        self.collision_occurred = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.last_collision_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.is_replay = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.pos_hist = torch.zeros(
            self.num_envs, 10, 2, device=self.device)
        self.stay_timer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int)

    def _init_cmd_safe_buffers(self):
        cd_cfg = self.cfg.cmd_safe
        n_sec = int(self.cfg.pd_risknet.n_sectors)

        # ── Precompute body radius per sector (ellipse) ──
        a = float(cd_cfg.body_semi_length)
        b = float(cd_cfg.body_semi_width)
        sec_size = 2.0 * math.pi / n_sec
        angles = torch.linspace(
            -math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size,
            n_sec, device=self.device,
        )
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        self._body_radius = a * b / torch.sqrt(
            b * b * cos_a * cos_a + a * a * sin_a * sin_a
        )  # (36,)

        self._sector_centers = torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=1,
        )  # (36, 2)

        # ── Runtime buffers ──
        self._sector_dists = torch.zeros(
            self.num_envs, n_sec, device=self.device, dtype=torch.float,
        )
        self._sector_safe = torch.zeros(
            self.num_envs, n_sec, device=self.device, dtype=torch.float,
        )
        self._safe_distances = torch.full(
            (self.num_envs, int(self.cfg.pd_risknet.num_lidar_points)),
            float(self.cfg.pd_risknet.ray_max_distance),
            device=self.device, dtype=torch.float,
        )
        self._clean_points_base = torch.zeros(
            self.num_envs, int(self.cfg.pd_risknet.num_lidar_points), 3,
            device=self.device, dtype=torch.float, requires_grad=False,
        )

    def _compute_sector_safety(self):
        """Compute per-sector z-filtered effective distances and safety factors.

        Replaces _compute_v_avoid() + _update_smooth_rays_dir().
        Called every step from _post_physics_step_callback.
        """
        cd_cfg = self.cfg.cmd_safe
        n_sec = int(self.cfg.pd_risknet.n_sectors)
        sec_size = 2.0 * math.pi / n_sec
        d_max = float(self.cfg.pd_risknet.ray_max_distance)

        dist = self._raw_distances.clone()
        pts = self._clean_points_base

        # ── Step 1: z-filtering ──
        z = pts[..., 2]
        z_mask = (z > cd_cfg.z_thresh_high) | (z < cd_cfg.z_thresh_low)
        dist = torch.where(z_mask, torch.full_like(dist, d_max), dist)
        self._safe_distances.copy_(dist)

        # ── Step 2: per-sector min distance ──
        body_azimuth = torch.atan2(pts[..., 1], pts[..., 0])
        sec_ids = torch.floor(
            (body_azimuth + math.pi) / sec_size
        ).long().clamp(min=0, max=n_sec - 1)

        min_dist = torch.full(
            (dist.shape[0], n_sec), 1e9, device=dist.device, dtype=dist.dtype,
        )
        min_dist.scatter_reduce_(
            1, sec_ids, dist, reduce='amin', include_self=False,
        )

        # ── Step 3: body radius compensation → effective distance ──
        d_eff = torch.clamp(min_dist - self._body_radius.unsqueeze(0), min=0.0)
        self._sector_dists.copy_(d_eff)

        # ── Step 4: safety factor ──
        d_safety = float(cd_cfg.d_safety)
        d_safe_max = float(cd_cfg.d_safe_max)
        safe = torch.clamp(
            (d_eff - d_safety) / (d_safe_max - d_safety), 0.0, 1.0,
        )
        self._sector_safe.copy_(safe)

    def _update_replay_buffer(self):
        """每步滚动更新回放缓冲区。新 episode 前两步用广播填充避免读到脏数据。"""
        self.replay_root_states = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            self.root_states.unsqueeze(1).expand(-1, self.replay_len, -1),
            torch.cat([self.replay_root_states[:, 1:],
                       self.root_states.unsqueeze(1)], dim=1))
        self.replay_dof_pos = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            self.dof_pos.unsqueeze(1).expand(-1, self.replay_len, -1),
            torch.cat([self.replay_dof_pos[:, 1:],
                       self.dof_pos.unsqueeze(1)], dim=1))
        self.replay_dof_vel = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            self.dof_vel.unsqueeze(1).expand(-1, self.replay_len, -1),
            torch.cat([self.replay_dof_vel[:, 1:],
                       self.dof_vel.unsqueeze(1)], dim=1))

    def _get_env_origins(self):
        pd_cfg = self.cfg.pd_risknet
        if getattr(pd_cfg, "soft_pretrain", False):
            from math import ceil

            num_rows = self.cfg.terrain.num_rows
            num_cols = self.cfg.terrain.num_cols
            t_len = self.cfg.terrain.terrain_length
            t_wid = self.cfg.terrain.terrain_width
            cells = num_rows * num_cols
            envs_per_cell = int(ceil(self.num_envs / cells))

            # Flat cell index: [0]*256, [1]*256, ..., [15]*256
            cell_idx = torch.div(
                torch.arange(self.num_envs, device=self.device),
                envs_per_cell, rounding_mode='floor'
            ).to(torch.long).clamp(0, cells - 1)

            # Split into (row, col) for 2D grid indexing.
            self.terrain_levels = torch.div(cell_idx, num_cols, rounding_mode='floor')
            self.terrain_types = torch.fmod(cell_idx, num_cols)
            self.max_terrain_level = 1

            # Build manual terrain_origins from grid cell centres.
            origins = torch.zeros(num_rows, num_cols, 3, device=self.device)
            for r in range(num_rows):
                for c in range(num_cols):
                    origins[r, c, 0] = c * t_len + t_len / 2.0
                    origins[r, c, 1] = r * t_wid + t_wid / 2.0

            self.custom_origins = True
            self.terrain_origins = origins
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels,
                                                       self.terrain_types]
            self._spawn_angles = None
        else:
            super()._get_env_origins()

    def _get_noise_scale_vec(self, cfg):
        """Proprio noise only; LiDAR channels are noise-free by default."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.0  # previous actions

        return noise_vec

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
            pd_cfg = self.cfg.pd_risknet
            if getattr(pd_cfg, "soft_pretrain", False):
                from legged_gym.utils.pillar_mesh import generate_pillar_lidar_mesh
                vertices, triangles_i32, self._pillar_boxes = generate_pillar_lidar_mesh(
                    self.cfg.terrain, pd_cfg, device=self.device)
            else:
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
            min_range=0.2,
            num_sensors=1,
            horizontal_line_num=int(ray_cfg.spherical_num_azimuth),
            vertical_line_num=int(ray_cfg.spherical_num_elevation),
            horizontal_fov_deg_min=-180.0,
            horizontal_fov_deg_max=180.0,
            vertical_fov_deg_min=float(getattr(ray_cfg, "vertical_fov_deg_min", -2.0)),
            vertical_fov_deg_max=float(getattr(ray_cfg, "vertical_fov_deg_max", 57.0)),
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

    def _init_height_points(self):
        """Override: support range+count (linspace) and legacy explicit-point-list configs.

        Stored grid metadata (self._height_grid_x/y) is used later by
        _draw_debug_vis to render the grid boundary.
        """
        t = self.cfg.terrain
        if hasattr(t, "measured_grid_x_range"):
            x_min, x_max = t.measured_grid_x_range
            y_min, y_max = t.measured_grid_y_range
            x = torch.linspace(x_min, x_max, int(t.measured_grid_x_count), device=self.device)
            y = torch.linspace(y_min, y_max, int(t.measured_grid_y_count), device=self.device)
        else:
            # Legacy path: explicit point lists from the parent config.
            x = torch.tensor(t.measured_points_x, device=self.device)
            y = torch.tensor(t.measured_points_y, device=self.device)

        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        # Cache grid vectors for the boundary visualisation in _draw_debug_vis.
        self._height_grid_x = x
        self._height_grid_y = y
        return points

    def _update_lidar_history(self):
        if self.lidar_sensor is None:
            return

        self.sensor_quat_tensor.copy_(quat_mul(self.base_quat, self._sensor_offset_quat))
        self.sensor_pos_tensor.copy_(
            self.base_pos + quat_apply(self.base_quat, self._sensor_translation))

        lidar_points, lidar_dist = self.lidar_sensor.update()
        points_sensor = lidar_points.view(self.num_envs, -1, 3)
        n_points = points_sensor.shape[1]
        dist = lidar_dist.view(self.num_envs, -1)

        # Store raw distances (no ground filter, no noise) for cmd_safe computation.
        self._raw_distances.copy_(dist)

        # Sensor frame → base frame transform.
        quat_1x4 = self._sensor_offset_quat[0:1]
        n_total = int(points_sensor.numel() // 3)
        points_base = quat_apply(quat_1x4.expand(n_total, 4), points_sensor.reshape(-1, 3))
        points_base = points_base.reshape(self.num_envs, n_points, 3) + \
            self._sensor_translation.unsqueeze(1)

        # Save clean points (before domain rand) for reward computation.
        self._clean_points_base.copy_(points_base)

        # ── Domain randomization for network input ──
        mask_ratio = float(getattr(self.cfg.domain_rand, "lidar_point_mask_ratio", 0.0))
        if mask_ratio > 0.0:
            rand_mask = torch.rand_like(dist) < mask_ratio
            lo, hi = self.cfg.domain_rand.lidar_point_mask_value_range
            fake_dist = torch.rand_like(dist) * (hi - lo) + lo
            dir_norm = torch.linalg.norm(points_base, dim=-1, keepdim=True).clamp(min=1.0e-6)
            dir_unit = points_base / dir_norm
            points_base = torch.where(
                rand_mask.unsqueeze(-1), dir_unit * fake_dist.unsqueeze(-1), points_base)
            dist = torch.where(rand_mask, fake_dist, dist)

        noise_ratio = float(getattr(self.cfg.domain_rand, "lidar_distance_noise_ratio", 0.0))
        if noise_ratio > 0.0:
            scale = 1.0 + (2.0 * torch.rand_like(dist) - 1.0) * noise_ratio
            points_base = points_base * scale.unsqueeze(-1)
            dist = dist * scale

        self.lidar_points_base.copy_(points_base)
        self.raycast_distances.copy_(dist)


    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_replay_buffer()
        self._update_lidar_history()
        self._compute_sector_safety()
        # pos_hist update for stuck detection
        update_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()
        if len(update_ids) > 0:
            self.pos_hist[update_ids] = torch.cat([
                self.pos_hist[update_ids, 1:],
                self.root_states[update_ids, :2].unsqueeze(1),
            ], dim=1)

    def check_termination(self):
        """终止检测 + 碰撞追踪 + early_reset 概率触发。"""
        # ── 初始化标志 ──
        self.initial_ = self.episode_length_buf <= 1
        self.extras["bad_masks"] = self.initial_

        # ── 硬碰撞 + timeout（check_termination 基类逻辑内联）──
        # 使用 :2（水平面力），与 SEA-Nav 一致
        hard_collision = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :2],
                       dim=-1) > 1.0, dim=1)
        hard_collision &= (~self.initial_)
        self.terminate_buf = hard_collision
        self.reset_buf = hard_collision.clone()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        # ── 翻转/跌落终止（保持原有逻辑）──
        if getattr(self.cfg.env, "enable_fall_termination", False):
            g_thresh = float(getattr(self.cfg.env, "fall_projected_gravity_z_threshold", -0.1))
            h_thresh = float(getattr(self.cfg.env, "fall_base_height_threshold", 0.12))
            flipped = self.projected_gravity[:, 2] > g_thresh
            low_base = self.base_pos[:, 2] < h_thresh
            self.reset_buf |= (flipped | low_base)
            self.terminate_buf |= (flipped | low_base)

        # ── 通道终点到达检测（保持原有逻辑）──
        pd_cfg = self.cfg.pd_risknet
        if self._goal_offsets_table is not None and getattr(pd_cfg, "goal_enabled", False) \
                and hasattr(self, 'terrain_levels') and hasattr(self, 'terrain_types'):
            off = self._goal_offsets_table[self.terrain_levels, self.terrain_types]
            gx = self.env_origins[:, 0] + off[:, 0]
            gy = self.env_origins[:, 1] + off[:, 1]
            gr = self.cfg.terrain.goal_radius
            dist = torch.sqrt(
                (self.base_pos[:, 0] - gx) ** 2 +
                (self.base_pos[:, 1] - gy) ** 2
            )
            reached = dist < gr
            self.reset_buf |= reached

        # ── 碰撞回放：碰撞追踪 + early_reset ──
        enable_replay = getattr(self.cfg.replay, 'enable_collision_replay', False)
        if enable_replay:
            # 检测新碰撞：penalised_contact_indices 中任意部位水平力 > 1.0
            new_collisions = torch.any(
                torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2],
                           dim=-1) > 1.0, dim=1)
            new_collisions &= (~self.initial_)

            # 只取碰撞首帧
            is_new_collision = new_collisions & (~self.last_collision_active)

            # early_reset 概率随地形难度线性增长
            prob_range = getattr(self.cfg.replay, 'early_reset_prob_range', [0.1, 0.5])
            if hasattr(self, 'terrain_levels') and hasattr(self, 'max_terrain_level'):
                early_prob = prob_range[0] + (prob_range[1] - prob_range[0]) * \
                    (self.terrain_levels.float() / max(1, self.max_terrain_level - 1)).clamp(max=1.0)
            else:
                early_prob = prob_range[0]
            trigger_early = is_new_collision & \
                (torch.rand(self.num_envs, device=self.device) < early_prob)

            self.reset_buf |= trigger_early
            self.terminate_buf |= trigger_early

            self.collision_occurred |= new_collisions
            self.last_collision_active = new_collisions

        # ── 卡住检测：瞬时静止 or 长期无位移 ──
        v_low = (torch.norm(self.base_lin_vel[:, :2], dim=-1) < 0.1) & \
                (torch.abs(self.base_ang_vel[:, 2]) < 0.1)
        d_low = torch.norm(
            self.root_states[:, :2] - self.pos_hist[:, 0, :2], dim=-1) < 0.2
        not_just_reset = (self.episode_length_buf.float() /
                          self.max_episode_length) > 0.1
        self.static = (v_low | d_low) & not_just_reset
        self.stay_timer += self.static.int()
        stand_still_flag = self.stay_timer >= 150
        self.reset_buf |= stand_still_flag

    def _reward_termination(self):
        """终止惩罚：仅对 terminate_buf==True 的环境施加（硬碰撞、early_reset、跌倒）。"""
        if hasattr(self, 'terminate_buf'):
            return self.terminate_buf.float()
        return self.reset_buf * ~self.time_out_buf

    def _reward_goal(self):
        pd_cfg = self.cfg.pd_risknet
        if self._goal_offsets_table is None or not getattr(pd_cfg, "goal_enabled", False):
            return torch.zeros(self.num_envs, device=self.device)
        off = self._goal_offsets_table[self.terrain_levels, self.terrain_types]
        gx = self.env_origins[:, 0] + off[:, 0]
        gy = self.env_origins[:, 1] + off[:, 1]
        gr = self.cfg.terrain.goal_radius
        dist = torch.sqrt(
            (self.base_pos[:, 0] - gx) ** 2 +
            (self.base_pos[:, 1] - gy) ** 2
        )
        reached = dist < gr
        return reached.float() * pd_cfg.goal_reward

    def _reward_ang_vel_yaw_penalty(self):
      return torch.square(self.base_ang_vel[:, 2])

    def _reward_curvature(self):
        """惩罚瞬时路径曲率平方，抑制原地转圈行为。

        r = -lambda * omega_z^2 / (v_xy^2 + sigma^2)
        轨迹曲率 kappa = |omega_z| / v_xy，该项 = -lambda * kappa^2。
        sigma^2 软化项防止零线速度时惩罚爆炸。
        """
        v_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        omega_z = self.base_ang_vel[:, 2]
        return omega_z.square() / (v_xy.square() + 0.49)

    def _reward_cmd_safe_vel(self):
        """Sector-constrained safe velocity tracking.

        v_safe_2d = sum_j(safe_j * align(cmd, sector_j) * sector_j)
        r = exp(-||v_actual - v_safe_2d||^2 / sigma^2)
        """
        cd_cfg = self.cfg.cmd_safe
        cmd_2d = self.commands[:, :2]
        cmd_norm = torch.norm(cmd_2d, dim=1, keepdim=True).clamp(min=1e-8)

        safe = self._sector_safe           # (N, 36)
        centers = self._sector_centers     # (36, 2)

        align = torch.matmul(cmd_2d, centers.T)        # (N, 36)
        align = torch.clamp(align, min=0.0)

        weighted = safe * align                        # (N, 36)
        v_safe = torch.matmul(weighted, centers)       # (N, 2)

        v_safe_norm = torch.norm(v_safe, dim=1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(cmd_norm / v_safe_norm, max=1.0)
        v_safe = v_safe * scale

        v_actual = self.base_lin_vel[:, :2]
        vel_err = torch.sum(torch.square(v_actual - v_safe), dim=1)
        sigma = float(cd_cfg.cmd_safe_sigma)
        return torch.exp(-vel_err / sigma)

    def _reward_sector_dist_penalty(self):
        """Omnidirectional background distance penalty.

        r = -sum_j(relu(d_thresh - d_eff_j)^2)
        Scaled by config weight (no internal alpha).
        """
        cd_cfg = self.cfg.cmd_safe
        d_thresh = float(cd_cfg.dist_penalty_thresh)
        penalty = torch.relu((d_thresh - self._sector_dists) / d_thresh).square()
        return -penalty.sum(dim=1)

    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            spawn_range = float(getattr(self.cfg.init_state, "spawn_offset_range", 0.5))
            self.root_states[env_ids, :2] += torch_rand_float(-spawn_range, spawn_range, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # 随机初始速度
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        # 随机初始朝向：切线方向 ± config 范围
        if self._spawn_angles is not None:
            base = self._spawn_angles[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            r0, r1 = self.cfg.init_state.rot_randomization_range
            rand_yaw = base + torch_rand_float(r0, r1, (len(env_ids), 1), device=self.device).squeeze(1)
        elif self.cfg.init_state.randomize_rot:
            r0, r1 = self.cfg.init_state.rot_randomization_range
            rand_yaw = torch_rand_float(r0, r1, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.root_states[env_ids, 3:7] = torch.tensor(self.cfg.init_state.rot, device=self.device)
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            return

        axis = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)
        self.root_states[env_ids, 3:7] = quat_from_angle_axis(rand_yaw, axis)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_terrain_curriculum(self, env_ids):
        """地形课程升降级，支持两种模式（向上兼容）。

        走廊地形（存在 goal_offset_y）：终点导向 — 到达终点区域升级，前进不足目标比例降级。
        非走廊地形：距离导向 — 行走距离 > env_length/2 升级（原始逻辑不变）。

        本方法在 reset_idx 之前调用，因此 root_states 仍是 episode 结束时的位置。
        """
        if not self.init_done:
            return

        if self._goal_offsets_table is not None:
            delta_xy = self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]
            forward_dist = torch.sum(delta_xy * self._channel_forward[env_ids], dim=1)
            off = self._goal_offsets_table[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            goal_dist = torch.sum(off * self._channel_forward[env_ids], dim=1) - self.cfg.terrain.goal_radius

            move_up_raw = forward_dist > goal_dist
            move_down_ratio = float(getattr(self.cfg.pd_risknet, "move_down_ratio", 0.5))
            move_down_raw = (forward_dist < move_down_ratio * goal_dist) & ~move_up_raw

            # 升级：连续 N 回合到达终点
            cons_up = int(getattr(self.cfg.pd_risknet, "consecutive_upgrade_episodes", 3))
            self._consecutive_upgrade_count[env_ids] = torch.where(
                move_up_raw,
                self._consecutive_upgrade_count[env_ids] + 1,
                torch.zeros_like(self._consecutive_upgrade_count[env_ids]))
            move_up = self._consecutive_upgrade_count[env_ids] >= cons_up

            # 降级：连续 N 回合未达阈值
            cons_down = int(getattr(self.cfg.pd_risknet, "consecutive_downgrade_episodes", 5))
            self._consecutive_downgrade_count[env_ids] = torch.where(
                move_down_raw,
                self._consecutive_downgrade_count[env_ids] + 1,
                torch.zeros_like(self._consecutive_downgrade_count[env_ids]))
            self._consecutive_downgrade_count[env_ids] = torch.where(
                move_up_raw,
                torch.zeros_like(self._consecutive_downgrade_count[env_ids]),
                self._consecutive_downgrade_count[env_ids])
            move_down = self._consecutive_downgrade_count[env_ids] >= cons_down

            # 升级将降级计数也归零
            self._consecutive_downgrade_count[env_ids] = torch.where(
                move_up,
                torch.zeros_like(self._consecutive_downgrade_count[env_ids]),
                self._consecutive_downgrade_count[env_ids])

            old_level = self.terrain_levels[env_ids].clone()
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

            # 防遗忘回退：在最高级完成 episode 后，随机回退到低级
            if self.max_terrain_level > 1:
                was_at_max = old_level >= self.max_terrain_level - 1
                fallback = was_at_max & move_up
                self.terrain_levels[env_ids] = torch.where(
                    fallback,
                    torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level - 1),
                    self.terrain_levels[env_ids],
                )
                self._consecutive_upgrade_count[env_ids] = torch.where(
                    fallback,
                    torch.zeros_like(self._consecutive_upgrade_count[env_ids]),
                    self._consecutive_upgrade_count[env_ids],
                )

            self.terrain_levels[env_ids] = torch.clip(
                self.terrain_levels[env_ids], 0, self.max_terrain_level - 1)

            self._consecutive_upgrade_count[env_ids] = torch.where(
                move_up | move_down,
                torch.zeros_like(self._consecutive_upgrade_count[env_ids]),
                self._consecutive_upgrade_count[env_ids])
            self._consecutive_downgrade_count[env_ids] = torch.where(
                move_down,
                torch.zeros_like(self._consecutive_downgrade_count[env_ids]),
                self._consecutive_downgrade_count[env_ids])
        else:
            # 非走廊地形：原始距离导向（向后兼容，逻辑与原 override 一致）
            distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
            move_up = distance > self.terrain.env_length / 2
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * 4.0) * ~move_up
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
            self.terrain_levels[env_ids] = torch.where(
                self.terrain_levels[env_ids] >= self.max_terrain_level,
                torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                torch.clip(self.terrain_levels[env_ids], 0))

        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _reset_collision_replay(self, env_ids):
        """从滚动缓冲区回退机器人状态，模拟"重新尝试"当前场景。"""
        undo_range = getattr(self.cfg.replay, 'undo_steps_range', [100, 150])
        undo_steps = torch.randint(
            undo_range[0], undo_range[1], (len(env_ids),), device=self.device)

        current_len = self.episode_length_buf[env_ids]
        undo_steps = torch.min(undo_steps.long(), current_len.long())
        undo_steps = torch.clamp(undo_steps, max=self.replay_len - 1)

        valid_replay = undo_steps > 20
        replay_ids = env_ids[valid_replay]
        fallback_ids = env_ids[~valid_replay]

        # 历史不够 → 走完整正常重置链
        if len(fallback_ids) > 0:
            super().reset_idx(fallback_ids)
            self.lidar_points_base[fallback_ids] = 0.0
            self._clean_points_base[fallback_ids] = 0.0
            self.raycast_distances[fallback_ids] = float(self.cfg.pd_risknet.ray_max_distance)
            self._raw_distances[fallback_ids] = float(self.cfg.pd_risknet.ray_max_distance)
            if hasattr(self, 'last_last_actions'):
                self.last_last_actions[fallback_ids] = 0.
            self._update_lidar_history()

        if len(replay_ids) == 0:
            return

        self.is_replay[replay_ids] = True
        indices = -undo_steps[valid_replay]

        self.root_states[replay_ids] = self.replay_root_states[replay_ids, indices]
        self.dof_pos[replay_ids] = self.replay_dof_pos[replay_ids, indices]
        self.dof_vel[replay_ids] = self.replay_dof_vel[replay_ids, indices]

        env_ids_int32 = replay_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.episode_length_buf[replay_ids] -= undo_steps[valid_replay]
        self.last_actions[replay_ids] = 0.
        self.last_dof_vel[replay_ids] = 0.

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        enable_replay = getattr(self.cfg.replay, 'enable_collision_replay', False)

        # ── 依赖 time_out_buf（check_termination 中设置）──
        time_out = self.time_out_buf if hasattr(self, 'time_out_buf') \
            else torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if enable_replay:
            is_collision = self.collision_occurred[env_ids]
            prob = getattr(self.cfg.replay, 'replay_prob', 0.8)
            wants_replay = (
                (torch.rand(len(env_ids), device=self.device) < prob)
                & is_collision
                & (~time_out[env_ids])
            )

            replay_ids = env_ids[wants_replay]
            normal_ids = env_ids[~wants_replay]

            if len(replay_ids) > 0:
                self._reset_collision_replay(replay_ids)
            if len(normal_ids) > 0:
                super().reset_idx(normal_ids)
        else:
            super().reset_idx(env_ids)
            normal_ids = env_ids
            replay_ids = torch.tensor([], device=self.device, dtype=torch.long)

        # ── LiDAR 专用重置：仅对非回放 env ──
        non_replay_ids = normal_ids if enable_replay else env_ids
        if len(non_replay_ids) > 0:
            self.lidar_points_base[non_replay_ids] = 0.0
            self._clean_points_base[non_replay_ids] = 0.0
            self.raycast_distances[non_replay_ids] = float(self.cfg.pd_risknet.ray_max_distance)
            self._raw_distances[non_replay_ids] = float(self.cfg.pd_risknet.ray_max_distance)
            if hasattr(self, 'last_last_actions'):
                self.last_last_actions[non_replay_ids] = 0.
            self._update_lidar_history()

        # ── 公共清理：所有 env ──
        self.collision_occurred[env_ids] = False
        self.last_collision_active[env_ids] = False
        self.is_replay[env_ids] = False
        self.stay_timer[env_ids] = 0
        self.pos_hist[env_ids, :, :] = 0.

    def _reward_channel_forward(self):
        cfg = self.cfg.rewards
        if getattr(cfg.scales, "channel_forward", 0.0) == 0.0:
            return torch.zeros(self.num_envs, device=self.device)
        p_cfg = self.cfg.pd_risknet
        ratio = float(getattr(p_cfg, "channel_backward_ratio", 0.5))
        channel_pos = torch.sum(self.base_pos[:, :2] * self._channel_forward, dim=1)
        delta = channel_pos - self._last_channel_pos
        self._last_channel_pos[:] = channel_pos
        forward  = torch.clamp(delta, min=0.0)
        backward = torch.clamp(-delta, min=0.0)
        return forward - ratio * backward

    def _reward_move_distance(self):
        dist = torch.norm(self.base_pos[:, :2] - self.env_origins[:, :2], dim=1)
        delta = dist - self.last_dist
        self.last_dist[:] = dist
        forward  = torch.clamp(delta, min=0.0)
        backward = torch.clamp(-delta, min=0.0)
        return forward - 0.1 * backward

    # --- 对齐论文 Table 5 的奖励机制覆盖 ---

    # def _reward_base_height(self):
    #     base_height = torch.mean(
    #         self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     return torch.abs(base_height - self.cfg.rewards.base_height_target)

    def _reward_collision(self):
        if getattr(self.cfg.pd_risknet, "collision_3d", False):
            # 原版 legged_gym: 全向 3D 二值检测
            return torch.sum(
                (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1).float(),
                dim=1)
        # 论文公式：水平 2D 连续平方
        forces_xy = torch.stack([
            torch.norm(self.contact_forces[:, idx, :2], dim=1)
            for idx in self.penalised_contact_indices
        ], dim=1)
        return torch.sum(torch.square(forces_xy), dim=1)

    def _reward_feet_stumble(self):
        """ 论文公式：||Force_xy||² × -0.02（连续平方力，仅在磕绊状态下触发）"""
        forces_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        forces_z = torch.abs(self.contact_forces[:, self.feet_indices, 2])
        stumbling = forces_xy > 5.0 * forces_z
        return torch.sum(torch.square(forces_xy) * stumbling.float(), dim=1)

    def _reward_dof_pos_limits(self):
        """ 论文公式：1_{q>qmax or q<qmin} × -0.2（二值越界指示，替代距离比例）"""
        out_of_limits = (self.dof_pos < self.dof_pos_limits[:, 0]).float()
        out_of_limits += (self.dof_pos > self.dof_pos_limits[:, 1]).float()
        return torch.sum(out_of_limits, dim=1)

    def _reward_action_rate2(self):
        if not hasattr(self, "last_last_actions"):
            self.last_last_actions = torch.zeros_like(self.actions)
        rate2 = self.actions - 2.0 * self.last_actions + self.last_last_actions
        self.last_last_actions[:] = self.last_actions
        return torch.sum(torch.square(rate2), dim=1)

    def compute_observations(self):
        # Base proprioception: 48-dim, matching LeggedRobot convention.
        proprio_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        self.obs_buf = torch.cat((
            proprio_obs,
            self.lidar_points_base.reshape(self.num_envs, -1),
        ), dim=-1)

        # Privileged channel for critic: proprio + terrain height samples.
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = self.measured_heights

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _draw_debug_vis(self):
        """Draw LiDAR points (proximal yellow, distal red) and velocity vectors."""
        if self.viewer is None:
            return

        self.gym.clear_lines(self.viewer)
        env_id = 0

        # Draw color-coded LiDAR points: proximal (theta >= split_theta) in yellow,
        # distal (theta < split_theta) in red.
        # Compute theta in sensor frame to match the network's _build_sampling_plan,
        # which correctly splits along the sensor's native spherical grid.
        pts_base = self.lidar_points_base[env_id]
        sensor_q = self._sensor_offset_quat[env_id]  # (4,)
        # Subtract sensor translation before inverse rotation for correct sensor-frame coordinates.
        pts_centered = pts_base - self._sensor_translation[env_id].unsqueeze(0)
        # Manual inverse rotation: conjugate and apply (isaacgym's torch_utils
        # requires matched batch dimensions, which is awkward for (4,) @ (N, 3)).
        conj = sensor_q * torch.tensor([-1, -1, -1, 1], device=self.device)
        conj_vec = conj[:3].unsqueeze(0).expand(pts_centered.shape[0], 3)  # (N, 3)
        cross = 2.0 * torch.cross(conj_vec, pts_centered, dim=-1)
        pts_base_sensor = pts_centered + conj[3] * cross + torch.cross(conj_vec, cross, dim=-1)
        eps = 1e-8
        theta = torch.atan2(pts_base_sensor[:, 2], torch.linalg.norm(pts_base_sensor[:, :2], dim=1) + eps)
        split_rad = float(self.cfg.pd_risknet.split_theta_deg) * math.pi / 180.0
        prox_mask = theta >= split_rad
        dist_mask = ~prox_mask

        base_pos = self.base_pos[env_id].unsqueeze(0).repeat(pts_base.shape[0], 1)
        base_quat = self.base_quat[env_id].unsqueeze(0).repeat(pts_base.shape[0], 1)
        pts_world = base_pos + quat_apply(base_quat, pts_base)

        # Diagnostic: print theta distribution once (env 0, first frame only).
        if not hasattr(self, '_printed_theta_dist'):
            self._printed_theta_dist = True
            theta_deg = theta * 180.0 / math.pi
            print(f"[DEBUG] theta range: [{theta_deg.min().item():.2f}°, {theta_deg.max().item():.2f}°]")
            print(f"[DEBUG] theta mean: {theta_deg.mean().item():.2f}°, split_at: {self.cfg.pd_risknet.split_theta_deg}°")
            print(f"[DEBUG] num proximal (theta >= {self.cfg.pd_risknet.split_theta_deg}°): {prox_mask.sum().item()}")
            print(f"[DEBUG] num distal (theta < {self.cfg.pd_risknet.split_theta_deg}°): {dist_mask.sum().item()}")
            print(f"[DEBUG] z range: [{pts_base[:, 2].min().item():.3f}, {pts_base[:, 2].max().item():.3f}]")
            print(f"[DEBUG] pts with z>0: {(pts_base[:, 2] > 0).sum().item()} / {pts_base.shape[0]}")

        # Downsample to ~256 points total for performance.
        num_pts = pts_base.shape[0]
        max_draw = min(4000, num_pts)
        step = max(1, num_pts // max_draw)

        # Exclude sky / no-hit points (d == d_max) from visualization.
        d_max = float(self.cfg.pd_risknet.ray_max_distance)
        valid_mask = self.raycast_distances[env_id] < (d_max - 0.001)

        # Draw proximal points (yellow)
        prox_mask_draw = prox_mask & valid_mask
        prox_pts = pts_world[prox_mask_draw].cpu().numpy()
        if len(prox_pts) > 0:
            prox_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(0, len(prox_pts), step):
                p = prox_pts[i]
                sphere_pose = gymapi.Transform(gymapi.Vec3(float(p[0]), float(p[1]), float(p[2])), r=None)
                gymutil.draw_lines(prox_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

        # Draw distal points (red)
        dist_mask_draw = dist_mask & valid_mask
        dist_pts = pts_world[dist_mask_draw].cpu().numpy()
        if len(dist_pts) > 0:
            dist_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
            for i in range(0, len(dist_pts), step):
                p = dist_pts[i]
                sphere_pose = gymapi.Transform(gymapi.Vec3(float(p[0]), float(p[1]), float(p[2])), r=None)
                gymutil.draw_lines(dist_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

        # --- 高度测量网格边界（绿色） ---
        # 在机体坐标系中沿网格最外圈走一圈，用 quat_apply_yaw 转到世界坐标。
        hx = self._height_grid_x
        hy = self._height_grid_y
        nx = len(hx)
        ny = len(hy)
        boundary = []
        # 下边: y=hy[0], x 从小到大
        boundary.extend([(hx[i].item(), hy[0].item(), 0.0) for i in range(nx)])
        # 右边: x=hx[-1], y 从小到大（跳过起点）
        boundary.extend([(hx[-1].item(), hy[j].item(), 0.0) for j in range(1, ny)])
        # 上边: y=hy[-1], x 从大到小（跳过起点）
        boundary.extend([(hx[i].item(), hy[-1].item(), 0.0) for i in range(nx - 2, -1, -1)])
        # 左边: x=hx[0], y 从大到小（跳过起点和终点）
        boundary.extend([(hx[0].item(), hy[j].item(), 0.0) for j in range(ny - 2, 0, -1)])
        b_pts = torch.tensor(boundary, device=self.device, dtype=torch.float)  # (N, 3)
        b_quat = self.base_quat[env_id].unsqueeze(0).expand(b_pts.shape[0], 4)
        b_pos = self.base_pos[env_id, :3].unsqueeze(0).expand(b_pts.shape[0], 3)
        b_world = quat_apply_yaw(b_quat, b_pts) + b_pos
        b_world[:, 2] = 0.0
        b_list = b_world.cpu().numpy().tolist()
        # 只取 4 个角点：底边 nx 点 → 右边 ny-1 点 → 上边 nx-1 点 → 左边 ny-2 点
        idx_bl = 0                              # 左下
        idx_br = nx - 1                         # 右下
        idx_tr = nx - 1 + ny - 1                # 右上
        idx_tl = nx - 1 + ny - 1 + nx - 1       # 左上
        corners = [b_list[idx_bl], b_list[idx_br], b_list[idx_tr], b_list[idx_tl]]
        for i in range(4):
            self.vis.draw_boldline(env_id, [corners[i], corners[(i + 1) % 4]],
                                   rad=0.01, resolution=6, color=(0, 1, 0))

        # Draw pillar wireframes (soft-pretrain debug).
        if hasattr(self, '_pillar_boxes') and self._pillar_boxes:
            verts = []
            colors = []
            for p_cell_id, cx, cy, sx, sy, h in self._pillar_boxes:
                x0, x1 = cx - sx / 2.0, cx + sx / 2.0
                y0, y1 = cy - sy / 2.0, cy + sy / 2.0
                z0, z1 = 0.0, h
                edges = [
                    ([x0, y0, z0], [x1, y0, z0]), ([x1, y0, z0], [x1, y1, z0]),
                    ([x1, y1, z0], [x0, y1, z0]), ([x0, y1, z0], [x0, y0, z0]),
                    ([x0, y0, z1], [x1, y0, z1]), ([x1, y0, z1], [x1, y1, z1]),
                    ([x1, y1, z1], [x0, y1, z1]), ([x0, y1, z1], [x0, y0, z1]),
                    ([x0, y0, z0], [x0, y0, z1]), ([x1, y0, z0], [x1, y0, z1]),
                    ([x1, y1, z0], [x1, y1, z1]), ([x0, y1, z0], [x0, y1, z1]),
                ]
                for a, b in edges:
                    verts.extend(a + b)
                    colors.extend([0.0, 0.6, 0.8] * 2)  # cyan
            if verts:
                verts_np = np.array(verts, dtype=np.float32)
                colors_np = np.array(colors, dtype=np.float32)
                num_lines = len(verts) // 3 // 2
                self.gym.add_lines(self.viewer, self.envs[env_id], num_lines,
                                   verts_np, colors_np)
