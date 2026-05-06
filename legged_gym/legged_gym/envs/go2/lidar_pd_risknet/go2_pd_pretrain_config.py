from legged_gym.envs.go2.flat.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO


OBS_HISTORY_LENGTH = 1
PROX_HISTORY_LENGTH = 10
DIST_HISTORY_LENGTH = 10
PD_SPHERICAL_AZIMUTH = 24
PD_SPHERICAL_ELEVATION = 18
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION
# Prefer denser near-field sampling for collision avoidance cues.
PD_PROXIMAL_POINTS = 192
PD_DISTAL_POINTS = 56
PD_PROXIMAL_FEATURE_DIM = 187
PD_DISTAL_FEATURE_DIM = 64
PD_PROPRIO_DIM = 48
PD_THETA_DEG = 20.0
# Height measurement grid: auto-generated from range + count via linspace.
# Counts must match the main risknet config (17×11=187) for weight transfer.
MEASURED_GRID_X_COUNT = 17
MEASURED_GRID_Y_COUNT = 11
MEASURED_GRID_X_RANGE = [-0.3, 1.8]
MEASURED_GRID_Y_RANGE = [-0.8, 0.8]
PD_PRIV_HEIGHT_DIM = MEASURED_GRID_X_COUNT * MEASURED_GRID_Y_COUNT
PD_PRIV_CRITIC_DIM = PD_PROPRIO_DIM + PD_PRIV_HEIGHT_DIM


class Go2LidarPDRiskNetCfg(Go2RoughCfg):
    class init_state(Go2RoughCfg.init_state):
        randomize_rot = True
        rot_randomization_range = [-3.1415, 3.1415]

    class pd_risknet:
        enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = PD_THETA_DEG

        n_sectors = 24
        avoid_distance_thresh = 1.6
        avoid_alpha = 1.6
        avoid_beta = 1.1
        ray_max_distance = 10.0

        # Spherical ray pattern used as raw LiDAR point cloud source.
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        num_lidar_points = spherical_num_azimuth * spherical_num_elevation

        avoid_speed_scale = 0.6

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
        # True flat terrain for gait pretraining.
        mesh_type = 'plane'
        measure_heights = True
        # Grid auto-generated from range + count via linspace in _init_height_points.
        measured_grid_x_range = MEASURED_GRID_X_RANGE
        measured_grid_y_range = MEASURED_GRID_Y_RANGE
        measured_grid_x_count = MEASURED_GRID_X_COUNT
        measured_grid_y_count = MEASURED_GRID_Y_COUNT
        curriculum = False

    class asset(Go2RoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class commands(Go2RoughCfg.commands):
        heading_command = False
        resampling_time = 4.
        class ranges(Go2RoughCfg.commands.ranges):
            ang_vel_yaw = [-1.5, 1.5]

    class obstacle_gen(Go2RoughCfg.obstacle_gen):
        # Keep actor-based obstacle generator disabled for now.
        # Current base pipeline assumes one actor per env and needs a larger refactor
        # for multi-actor root-state bookkeeping.
        enable_obstacles = False

    class raycaster(Go2RoughCfg.raycaster):
        enable_raycast = True
        ray_pattern = "spherical"
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        max_distance = 10.0
        attach_yaw_only = False
        # Match unitree_go2.py lidar mount translation (base frame, meters).
        offset_pos = [0.28945, 0.0, -0.046825]
        # Match unitree_go2.py lidar mount fixed rotation (roll, pitch, yaw in radians).
        sensor_offset_rpy = [0.0, -2.8782, 3.14]

    class rewards(Go2RoughCfg.rewards):
        class scales(Go2RoughCfg.rewards.scales):
            # Paper main rewards.
            vel_avoid = 0 # 速度跟踪+避障奖励：鼓励跟踪 (v_cmd + v_avoid)
            rays = 0  # 距离最大化奖励：鼓励与障碍保持更大安全间距
            
            lin_vel_z = 0 # 惩罚机体 z 方向线速度，抑制上下抖动/跳动
            feet_stumble = 0  # 惩罚脚部绊碰（足端受到异常横向冲击）
            collision = 0  # 惩罚机体/连杆非期望碰撞
            dof_pos_limits = 0  # 惩罚关节接近或超过位置限位
            torques = 0  # 惩罚关节力矩过大，降低能耗和电机负担
            dof_vel = 0  # 惩罚关节速度过大，抑制过激动作
            dof_acc = 0  # 惩罚关节加速度过大，提升动作平滑性
            action_rate = 0  # 一阶动作平滑惩罚：限制相邻时刻动作变化
            action_rate2 = 0  # 二阶动作平滑惩罚：限制动作“抖动/顿挫”


            #flat_reward
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05 
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.

            # Overrides
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 1.0
            # feet_contact_forces = -0.01
            # gait_scheduler = -3


            
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
        critic_hidden_dims = [512, 256, 128]
        perception_enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_history_length = PROX_HISTORY_LENGTH
        distal_history_length = DIST_HISTORY_LENGTH
        num_lidar_points = PD_NUM_LIDAR_POINTS
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = PD_THETA_DEG
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proprio_obs_dim = PD_PROPRIO_DIM
        privileged_height_dim = PD_PRIV_HEIGHT_DIM
        privileged_critic_dim = PD_PRIV_CRITIC_DIM
        privileged_supervision_coef = 1.0
        sensor_offset_rpy = [0.0, -2.8782, 3.14]

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
        experiment_name = "go2_pd_pretrain"
        run_name = ""
        max_iterations = 300
