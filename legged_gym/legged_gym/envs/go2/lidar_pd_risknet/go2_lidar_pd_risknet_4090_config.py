from legged_gym.envs.go2.flat.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO


OBS_HISTORY_LENGTH = 1
PROX_HISTORY_LENGTH = 10
DIST_HISTORY_LENGTH = 10
PD_SPHERICAL_AZIMUTH = 80
PD_SPHERICAL_ELEVATION = 50
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION
# Prefer denser near-field sampling for collision avoidance cues.
PD_PROXIMAL_POINTS = 512
PD_DISTAL_POINTS = 256
PD_PROXIMAL_FEATURE_DIM = 187
PD_DISTAL_FEATURE_DIM = 64
PD_PROPRIO_DIM = 48
MEASURED_GRID_X_COUNT = 17
MEASURED_GRID_Y_COUNT = 11
MEASURED_GRID_X_RANGE = [-1.0, 4.6]
MEASURED_GRID_Y_RANGE = [-1.8, 1.8]
PD_PRIV_HEIGHT_DIM = MEASURED_GRID_X_COUNT * MEASURED_GRID_Y_COUNT
PD_PRIV_CRITIC_DIM = PD_PROPRIO_DIM + PD_PRIV_HEIGHT_DIM


class Go2LidarPDRiskNetCfg(Go2RoughCfg):
    class init_state(Go2RoughCfg.init_state):
        randomize_rot = True
        rot_randomization_range = [-0.5236, 0.5236]   # 相对切线方向的偏航随机范围 (rad)
        spawn_offset_range = 0.3                       # 出生点 XY 随机偏移范围 (m)

    class sim(Go2RoughCfg.sim):
        class physx(Go2RoughCfg.sim.physx):
            num_threads = 24  # AutoDL使用，4096环境(原10线程)
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            found_lost_aggregate_pairs_capacity = 2**24  # trimesh terrain (walls+pillars) 碰撞体远超默认值

    class raycaster(Go2RoughCfg.raycaster):
        enable_raycast = True
        ray_pattern = "spherical"
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        max_distance = 50.0
        attach_yaw_only = False
        # Match unitree_go2.py lidar mount translation (base frame, meters).
        offset_pos = [0.28945, 0.0, -0.046825]
        # Match unitree_go2.py lidar mount fixed rotation (roll, pitch, yaw in radians).
        sensor_offset_rpy = [0.0, -2.8782, 3.14]

    class pd_risknet:
        enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = 20.0

        n_sectors = 36
        avoid_distance_thresh = 1.5
        avoid_alpha = 2.0
        avoid_beta = 1.0
        ray_max_distance = 10.0

        # Spherical ray pattern used as raw LiDAR point cloud source.
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        num_lidar_points = spherical_num_azimuth * spherical_num_elevation

        avoid_speed_scale = 0.6

        # 通道终点奖励
        goal_enabled = True
        goal_reward = 1.0

    class env(Go2RoughCfg.env):
        # Base Go2 proprio obs + raw LiDAR history points (N_hist * N_points * xyz).
        num_observations = PD_PROPRIO_DIM + OBS_HISTORY_LENGTH * PD_NUM_LIDAR_POINTS * 3
        # Critic input uses proprio (48) + privileged heights (187).
        num_privileged_obs = PD_PRIV_CRITIC_DIM
        # Anti-flip termination gates to avoid upside-down reward exploitation.
        enable_fall_termination = False
        # In body frame, projected_gravity[:, 2] is near -1 when upright and near +1 when upside-down.
        fall_projected_gravity_z_threshold = -0.1
        # Terminate when base height is unrealistically low (meters).
        fall_base_height_threshold = 0.12

    class terrain(Go2RoughCfg.terrain):
        horizontal_scale = 0.1
        # Keep heights enabled for privileged supervision channel.
        measure_heights = True
        # Grid auto-generated from range + count via linspace in _init_height_points.
        measured_grid_x_range = MEASURED_GRID_X_RANGE
        measured_grid_y_range = MEASURED_GRID_Y_RANGE
        measured_grid_x_count = MEASURED_GRID_X_COUNT
        measured_grid_y_count = MEASURED_GRID_Y_COUNT
        mesh_type = 'trimesh'
        curriculum = True
        max_init_terrain_level = 0  # 所有机器人从 level 0 (直线通道) 开始

        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 1.0]
        terrain_length = 12.
        terrain_width = 12.
        num_rows = 6
        num_cols = 4

        # 弯曲通道地形配置
        corridor_width = 3.0       # 通道宽度 (m)
        wall_height = 0.8          # 墙壁高度 (m)
        wall_thickness = 0.4       # 墙壁厚度 (m)
        amplitude = 1.5            # 正弦波振幅 (m)
        num_cycles = 1.5           # 正弦波周期数
        alternate_sign = True      # 按地块索引交替反转振幅符号
        end_margin = 0.5           # 通道两端与地块边缘的间距 (m)
        straight_length = 2.0      # 起点直线段长度 (m)，先补齐两壁长度差再延伸此长度

        # 通道内随机方柱
        pillar_count = 3           # 每通道柱子数量
        pillar_half_width = 0.15   # 柱子半宽 (m), 全宽=0.3m
        pillar_min_separation = 1.0  # 柱子间最小净距 (m)
        pillar_wall_margin = 0.5   # 柱子与墙最小净距 (m)
        pillar_centerline_margin = 0.3  # 柱子与中心线最小距离 (m)
        pillar_margin_end = 1.5    # 柱子距两端半圆圆心最小距离 (m)

    class commands(Go2RoughCfg.commands):
        heading_command = False
        resampling_time = 2.
        curriculum = True
        class ranges(Go2RoughCfg.commands.ranges):
            lin_vel_x = [0.5, 1.0]  # min max [m/s]
            lin_vel_y = [-0.2, 0.2]  # min max [m/s]
            ang_vel_yaw = [-0.2, 0.2]    # min max [rad/s]

    class obstacle_gen(Go2RoughCfg.obstacle_gen):
        # Keep actor-based obstacle generator disabled for now.
        # Current base pipeline assumes one actor per env and needs a larger refactor
        # for multi-actor root-state bookkeeping.
        enable_obstacles = False

    class rewards(Go2RoughCfg.rewards):
        base_height_target = 0.33
        class scales:
            # Paper main rewards.
            vel_avoid = 2.0  # 速度跟踪+避障奖励：鼓励跟踪 (v_cmd + v_avoid)
            goal = 1.0  # 通道终点到达奖励
            rays = 1.5  # 距离最大化奖励：鼓励与障碍保持更大安全间距
            y_progress = 0.5  # 世界坐标系 Y 进度奖励

            # Auxiliary rewards from appendix Table 5.
            lin_vel_z = -3.0e-4  # 惩罚机体 z 方向线速度，抑制上下抖动/跳动
            feet_stumble = -2.0e-2  # 惩罚脚部绊碰（足端受到异常横向冲击）
            collision = -1.0  # 二元碰撞惩罚：任一检测部位 >0.1N → 全罚
            dof_pos_limits = -0.2  # 惩罚关节接近或超过位置限位
            torques = -1.0e-6  # 惩罚关节力矩过大，降低能耗和电机负担
            dof_vel = -1.0e-6  # 惩罚关节速度过大，抑制过激动作
            dof_acc = -2.5e-7  # 惩罚关节加速度过大，提升动作平滑性
            action_rate = -5.0e-3  # 一阶动作平滑惩罚：限制相邻时刻动作变化
            action_rate2 = -5.0e-3  # 二阶动作平滑惩罚：限制动作“抖动/顿挫”
            
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            feet_air_time = 1.0
            base_height = -0.3

            # Overrides
            lin_vel_z = -3.3e-4
            
    class normalization(Go2RoughCfg.normalization):
        # LiDAR points are raw geometric values; keep unscaled.
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            pass

    class domain_rand(Go2RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]

        # Paper-specific LiDAR randomization.
        lidar_point_mask_ratio = 0.05
        lidar_point_mask_value_range = [2.0, 10.0]
        lidar_distance_noise_ratio = 0.02

        # Remaining parameters are declared for parity and can be consumed by future hooks.
        payload_mass_range = [-1.0, 3.0]
        com_shift_range = [[-0.1, -0.15, -0.2], [0.1, 0.15, 0.2]]
        restitution_range = [0.0, 1.0]
        motor_strength_range = [0.8, 1.2]
        joint_calib_offset_range = [-0.02, 0.02]
        gravity_offset_range = [-1.0, 1.0]
        proprio_delay_range = [0.005, 0.045]
        
class Go2LidarPDRiskNetCfgPPO(Go2RoughCfgPPO):
    class policy(Go2RoughCfgPPO.policy):
        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [1024, 512, 256, 128]
        perception_enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_history_length = PROX_HISTORY_LENGTH
        distal_history_length = DIST_HISTORY_LENGTH
        num_lidar_points = PD_NUM_LIDAR_POINTS
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = 5.0
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proprio_obs_dim = PD_PROPRIO_DIM
        privileged_height_dim = PD_PRIV_HEIGHT_DIM
        privileged_critic_dim = PD_PRIV_CRITIC_DIM
        privileged_supervision_coef = 1.0

    class algorithm(Go2RoughCfgPPO.algorithm):
        amp_enabled = True   # 启用混合精度
        clip_param = 0.2
        lam = 0.95
        gamma = 0.99
        learning_rate = 1.0e-3
        schedule = "adaptive"
        entropy_coef = 0.01
        desired_kl = 0.01
        max_grad_norm = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(Go2RoughCfgPPO.runner):
        policy_class_name = "PDRiskNetActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        experiment_name = "go2_lidar_pd_risknet_4090"
        run_name = ""
        max_iterations = 1000
