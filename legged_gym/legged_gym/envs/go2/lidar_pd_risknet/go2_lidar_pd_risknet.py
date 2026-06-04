import torch
import math
import numpy as np
import warp as wp

from isaacgym.torch_utils import quat_rotate_inverse, quat_apply, quat_mul, quat_from_euler_xyz, quat_from_angle_axis, torch_rand_float
from isaacgym import gymapi, gymtorch, gymutil

from legged_gym.envs.go2.go2 import Go2
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType
from legged_gym.utils.math_utils import quat_apply_yaw


class Go2LidarPDRiskNet(Go2):
    """Go2 environment extension for the LiDAR + PD-RiskNet training pipeline.

    This class is compatibility-first: it preserves the parent Go2 behavior and only
    appends dedicated observation channels required by the new task.
    """

    def _init_buffers(self):
        # Override: longer episodes give robots more walking time per episode.
        self.cfg.env.episode_length_s = 30
        super()._init_buffers()
        # Enable per-step debug drawing for this task when viewer is available.
        self.debug_viz = True
        # Body indices for obstacle collision penalty.
        self.collision_body_indices = [
            self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name)
            for name in (
                "base", "Head_upper",
                "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                "FL_calf",  "FR_calf",  "RL_calf",  "RR_calf",
            )
        ]
        self._init_pd_risknet_buffers()
        self._init_lidar_sensor()
        if self.lidar_sensor is not None:
            self._init_lidar_aux()
        if not hasattr(self, '_spawn_angles'):
            self._spawn_angles = None

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
        noise_vec[9:12] = 0.0  # commands

        if self.cfg.pd_risknet.heading_obs_enabled:
            noise_vec[12:13] = 0.0  # current_heading: 不额外加噪（已有独立噪声机制）
            noise_vec[13:25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[25:37] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[37:49] = 0.0  # previous actions
        else:
            noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[36:48] = 0.0  # previous actions

        return noise_vec

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
        self.avoid_distances = torch.full(
            (self.num_envs, int(cfg.num_lidar_points)),
            float(cfg.ray_max_distance),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self.v_avoid = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self._consecutive_upgrade_count = torch.zeros(
            self.num_envs, device=self.device,
            dtype=torch.int32, requires_grad=False)
        self._consecutive_downgrade_count = torch.zeros(
            self.num_envs, device=self.device,
            dtype=torch.int32, requires_grad=False)

        # 通道前进方向单位向量 (每环境根据 terrain_type 即 col 索引确定)
        _FORWARD_LOOKUP_TABLE = torch.tensor([
            [0.0, 1.0],    # direction 0: +Y (北)
            [1.0, 0.0],    # direction 1: +X (东)
            [0.0, -1.0],   # direction 2: -Y (南)
            [-1.0, 0.0],   # direction 3: -X (西)
        ], device=self.device, dtype=torch.float)
        _safe_idx = self.terrain_types.long().clamp(0, 3)
        self._channel_forward = _FORWARD_LOOKUP_TABLE[_safe_idx]

        # Per-cell goal offsets table (world frame, relative to env_origin)
        if hasattr(self.terrain, "goal_offsets") and np.any(self.terrain.goal_offsets):
            self._goal_offsets_table = torch.from_numpy(
                self.terrain.goal_offsets).to(self.device).to(torch.float)
        else:
            self._goal_offsets_table = None

        # Precompute distal ray mask and sector ids from sensor ray directions.
        # This is deferred to _init_lidar_aux() after _init_lidar_sensor().

    def _init_lidar_aux(self):
        """Post-init: compute auxiliary structures from LidarSensor ray directions."""
        cfg = self.cfg.pd_risknet
        split_rad = math.radians(float(cfg.split_theta_deg))

        self._ray_dirs_sensor = self.lidar_sensor.get_ray_directions()
        theta = torch.atan2(
            self._ray_dirs_sensor[:, 2],
            torch.linalg.norm(self._ray_dirs_sensor[:, :2], dim=1) + 1e-8,
        )
        self._distal_mask = theta < split_rad
        if self._distal_mask.sum() == 0:
            raise ValueError(
                f"No distal rays found: split_theta_deg={float(cfg.split_theta_deg):.1f}°."
            )

        ray_azimuth = torch.atan2(
            self._ray_dirs_sensor[:, 1],
            self._ray_dirs_sensor[:, 0],
        )
        ray_azimuth_0_2pi = ray_azimuth + math.pi
        sector_size = 2.0 * math.pi / 36.0
        self._distal_ray_sector_ids = torch.floor(
            ray_azimuth_0_2pi[self._distal_mask] / sector_size
        ).long().clamp(min=0, max=35)

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
        self.sensor_pos_tensor.copy_(self.base_pos + quat_apply(self.base_quat, self._sensor_translation))

        lidar_points, lidar_dist = self.lidar_sensor.update()
        points_sensor = lidar_points.view(self.num_envs, -1, 3)
        n_points = points_sensor.shape[1]
        dist = lidar_dist.view(self.num_envs, -1)
        max_dist = float(self.cfg.pd_risknet.ray_max_distance)

        # Sensor frame → base frame transform (expand avoids 98 MB repeat alloc).
        quat_1x4 = self._sensor_offset_quat[0:1]
        n_total = int(points_sensor.numel() // 3)
        points_base = quat_apply(quat_1x4.expand(n_total, 4), points_sensor.reshape(-1, 3))
        points_base = points_base.reshape(self.num_envs, n_points, 3) + self._sensor_translation.unsqueeze(1)

        # 通路 A: 避障 — 干净数据，仅做地面滤除。
        env_ids_per_point = torch.arange(self.num_envs, device=self.device).repeat_interleave(n_points)
        quat_per_point = self.base_quat[env_ids_per_point]
        clean_base_flat = points_base.clone().view(-1, 3)
        clean_world_flat = quat_apply(quat_per_point, clean_base_flat) + self.base_pos[env_ids_per_point]
        clean_world = clean_world_flat.view(self.num_envs, n_points, 3)
        clean_is_ground = torch.abs(clean_world[..., 2]) < 0.05
        self.avoid_distances.copy_(torch.where(clean_is_ground, torch.full_like(dist, max_dist), dist))

        # 通路 B: 网络观测 — 域随机化。
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

        self.lidar_points_base.copy_(points_base)
        self.raycast_distances.copy_(dist)

    def _compute_v_avoid(self):
        cfg = self.cfg.pd_risknet
        n_sec = int(cfg.n_sectors)
        sec_size = 2.0 * math.pi / n_sec

        pts = self.lidar_points_base[..., :2]
        dist = self.avoid_distances
        angles = torch.atan2(pts[..., 1], pts[..., 0])
        sec_ids = torch.floor((angles + math.pi) / sec_size).long().clamp(min=0, max=n_sec - 1)

        # Per-sector minimum: single scatter_reduce replaces 36× for loop + where.
        min_dist_per_sec = torch.full(
            (dist.shape[0], n_sec), 1e9, device=dist.device, dtype=dist.dtype)
        min_dist_per_sec.scatter_reduce_(
            1, sec_ids, dist, reduce='amin', include_self=False)

        # Sector center directions pointing AWAY from each sector.
        sec_centers = torch.linspace(
            -math.pi + 0.5 * sec_size, math.pi - 0.5 * sec_size, n_sec, device=self.device
        )
        away_dirs = torch.stack(
            (-torch.cos(sec_centers), -torch.sin(sec_centers)), dim=-1
        )  # (n_sec, 2)


        # Pure distance-weighted vector sum (paper V1 formula):
        # w_i = exp(-alpha * d_i) if d_i < d_max.
        # v_avoid = sum(w_i * away_dir_i) over all sectors.
        d_max = float(cfg.avoid_distance_thresh)
        alpha = float(cfg.avoid_alpha)
        exp_max = math.exp(-alpha * d_max)           # Python float，只算一次
        w = torch.relu(torch.exp(-alpha * min_dist_per_sec) - exp_max)

        self.v_avoid = (w.unsqueeze(-1) * away_dirs.unsqueeze(0)).sum(dim=1)  # (num_envs, 2)

        # Clamp avoidance speed to configurable limit.
        max_avoid = float(getattr(cfg, "avoid_speed_limit", 1.0))
        avoid_norm = torch.norm(self.v_avoid, dim=1, keepdim=True)
        self.v_avoid = torch.where(
            avoid_norm > max_avoid,
            self.v_avoid * max_avoid / avoid_norm,
            self.v_avoid,
        )

    def _compute_pd_risknet_features(self):
        self._compute_v_avoid()

    def _resample_commands(self, env_ids):
        import math
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)

        # heading 围绕通道方向采样 (向量化)
        _SPAWN_ANGLES = torch.tensor(
            [math.pi / 2, 0.0, -math.pi / 2, math.pi],
            device=self.device, dtype=torch.float)
        channel_angle = _SPAWN_ANGLES[self.terrain_types[env_ids].long()]
        spread = 0.35  # +/-20 degrees
        self.commands[env_ids, 3] = channel_angle + torch_rand_float(
            -spread, spread, (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_lidar_history()
        # Reward is computed before compute_observations in LeggedRobot.post_physics_step,
        # so V_avoid must be refreshed here to avoid one-step lag.
        self._compute_v_avoid()

    def check_termination(self):
        super().check_termination()
        # 翻转/跌落终止
        if getattr(self.cfg.env, "enable_fall_termination", False):
            g_thresh = float(getattr(self.cfg.env, "fall_projected_gravity_z_threshold", -0.1))
            h_thresh = float(getattr(self.cfg.env, "fall_base_height_threshold", 0.12))
            flipped = self.projected_gravity[:, 2] > g_thresh
            low_base = self.base_pos[:, 2] < h_thresh
            self.reset_buf |= (flipped | low_base)

        # 通道终点到达检测
        pd_cfg = self.cfg.pd_risknet
        if self._goal_offsets_table is not None and getattr(pd_cfg, "goal_enabled", False):
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

            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
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

    def update_command_curriculum(self, env_ids):
        """Override: use vel_avoid instead of tracking_lin_vel for curriculum check."""
        if torch.mean(self.episode_sums["vel_avoid"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["vel_avoid"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # self.lidar_history[env_ids] = 0.0
        self.lidar_points_base[env_ids] = 0.0
        self.raycast_distances[env_ids] = float(self.cfg.pd_risknet.ray_max_distance)
        self.v_avoid[env_ids] = 0.0
        self.last_dist[env_ids] = torch.norm(
            self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        if hasattr(self, 'last_last_actions'):
            self.last_last_actions[env_ids] = 0.
        self._update_lidar_history()

    def _reward_vel_avoid(self):
        cfg = self.cfg.pd_risknet
        vel_target = self.commands[:, :2] + self.v_avoid
        vel_err = torch.sum(torch.square(self.base_lin_vel[:, :2] - vel_target), dim=1)
        return torch.exp(-float(cfg.avoid_beta) * vel_err)

    def _reward_rays(self):
        d_max = float(self.cfg.pd_risknet.ray_max_distance)
        dist_all = self.raycast_distances[:, self._distal_mask]  # (N, num_distal_raw)
        valid = dist_all < (d_max - 0.001)  # exclude sky / no-hit rays at d_max
        dist = torch.where(valid, dist_all, torch.zeros_like(dist_all))

        n_sectors = 36
        top_ratio = 0.25

        sector_means = []
        for s in range(n_sectors):
            s_mask = self._distal_ray_sector_ids == s  # (num_distal_raw,)
            s_dist = dist[:, s_mask]                    # (N, rays_in_sector)
            s_valid = valid[:, s_mask]

            n_valid = s_valid.sum(dim=1, keepdim=True).clamp(min=1).float()  # (N, 1)
            k = torch.clamp((n_valid * top_ratio).long(), min=1)            # (N, 1)
            k_max = int(k.max().item())

            top_vals, _ = torch.topk(s_dist, k=k_max, dim=1)  # (N, k_max)

            idx = torch.arange(k_max, device=s_dist.device).unsqueeze(0).expand_as(top_vals)
            keep = idx < k
            top_sum = (top_vals * keep.float()).sum(dim=1)  # (N,)
            sector_mean = top_sum / k.squeeze(1)            # (N,)
            sector_means.append(sector_mean)

        sector_mean = torch.stack(sector_means, dim=1)  # (N, 36)
        n_fwd = self.cfg.pd_risknet.ray_forward_sector_count
        center = self.cfg.pd_risknet.ray_forward_sector_center
        start = max(0, center - n_fwd // 2)
        end = min(36, start + n_fwd)
        return sector_mean[:, start:end].mean(dim=1) / d_max

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
                (torch.norm(self.contact_forces[:, self.collision_body_indices, :], dim=-1) > 0.1).float(),
                dim=1)
        # 论文公式：水平 2D 连续平方
        forces_xy = torch.stack([
            torch.norm(self.contact_forces[:, idx, :2], dim=1)
            for idx in self.collision_body_indices
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
        # Keep base proprio order identical to Go2/LeggedRobot, then append LiDAR history.
        if self.cfg.pd_risknet.heading_obs_enabled:
            # ── 新观测：heading 目标 + current_heading ──
            cmd_obs = torch.cat((
                self.commands[:, 0:1] * self.obs_scales.lin_vel,
                self.commands[:, 1:2] * self.obs_scales.lin_vel,
                self.commands[:, 3:4] * self.obs_scales.heading,
            ), dim=-1)

            forward = quat_apply(self.base_quat, self.forward_vec)
            current_heading = torch.atan2(forward[:, 1], forward[:, 0])
            if self.cfg.pd_risknet.heading_noise_enabled:
                current_heading = current_heading + torch.randn_like(current_heading) * self.cfg.pd_risknet.heading_noise_std

            proprio_obs = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                cmd_obs,
                current_heading.unsqueeze(1) * self.obs_scales.heading,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ), dim=-1)
        else:
            # ── 旧观测：P 控制器角速度（兼容已有 checkpoint）──
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

            # 临时诊断打印（确认后可注释)
            if not hasattr(self, '_printed_height_shape'):
                print(f"[INFO] measured_heights.shape: {self.measured_heights.shape}")
                self._printed_height_shape = True

            self.privileged_obs_buf = torch.cat((proprio_obs, self.measured_heights), dim=-1)

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

        # Draw avoidance direction (yellow) and combined velocity (blue).
        start = self.base_pos[env_id].detach().cpu().numpy()
        base_quat = self.base_quat[env_id]            # 保留在 GPU 上用于旋转

        # 获取机体坐标系下的避障速度（保持为 torch 张量）
        avoid_xy = self.v_avoid[env_id].detach()

        # 构造三维机体向量，并用四元数旋转到世界坐标系
        avoid_body = torch.tensor([avoid_xy[0].item(), avoid_xy[1].item(), 0.0], device=self.device)

        avoid_world = quat_apply(base_quat, avoid_body).cpu().numpy()

        avoid_vec = avoid_world.astype(np.float32)

        # Scale factor: 1m arrow = 1 m/s, clamp to avoid extreme-length arrows.
        max_display_len = 3.0
        avoid_norm = np.linalg.norm(avoid_vec[:2])
        if avoid_norm > 1.0e-6:
            display_len = min(avoid_norm, max_display_len)
            self.vis.draw_arrow(env_id, start.tolist(),
                                (start + display_len * avoid_vec / avoid_norm).tolist(),
                                width=0.01, color=(1, 1, 0))

        # 绘制合成速度 (蓝色)
        combined_xy = (self.commands[env_id, :2] + self.v_avoid[env_id]).detach()
        combined_body = torch.tensor([combined_xy[0].item(), combined_xy[1].item(), 0.0], device=self.device)
        combined_world = quat_apply(base_quat, combined_body).cpu().numpy()
        combined_vec = combined_world.astype(np.float32)
        combined_norm = np.linalg.norm(combined_vec[:2])
        if combined_norm > 1.0e-6:
            display_len = min(combined_norm, max_display_len)
            self.vis.draw_arrow(env_id, start.tolist(),
                                (start + display_len * combined_vec / combined_norm).tolist(),
                                width=0.01, color=(0, 0, 1))  # 蓝色
