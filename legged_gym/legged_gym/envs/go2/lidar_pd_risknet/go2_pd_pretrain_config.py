from legged_gym.envs.go2.flat.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO


PD_HISTORY_LENGTH = 10
PD_SPHERICAL_AZIMUTH = 24
PD_SPHERICAL_ELEVATION = 18
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION
# Prefer denser near-field sampling for collision avoidance cues.
PD_PROXIMAL_POINTS = 288
PD_DISTAL_POINTS = 144
PD_PROXIMAL_FEATURE_DIM = 187
PD_DISTAL_FEATURE_DIM = 64
PD_PROPRIO_DIM = 48
PD_PRIV_HEIGHT_DIM = 187


class Go2LidarPDRiskNetCfg(Go2RoughCfg):
    class pd_risknet:
        enabled = True
        history_length = PD_HISTORY_LENGTH
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = 5.0

        n_sectors = 24
        avoid_distance_thresh = 1.0
        avoid_alpha = 1.0
        avoid_beta = 1.0
        ray_max_distance = 10.0

        # Spherical ray pattern used as raw LiDAR point cloud source.
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        num_lidar_points = spherical_num_azimuth * spherical_num_elevation

    class env(Go2RoughCfg.env):
        # Base Go2 proprio obs + raw LiDAR history points (N_hist * N_points * xyz).
        num_observations = PD_PROPRIO_DIM + PD_HISTORY_LENGTH * PD_NUM_LIDAR_POINTS * 3
        # Keep critic/aux-supervision dimension aligned with formal training.
        num_privileged_obs = PD_PRIV_HEIGHT_DIM
        # Anti-flip termination gates to avoid upside-down reward exploitation.
        enable_fall_termination = True
        # In body frame, projected_gravity[:, 2] is near -1 when upright and near +1 when upside-down.
        fall_projected_gravity_z_threshold = -0.1
        # Terminate when base height is unrealistically low (meters).
        fall_base_height_threshold = 0.12

    class terrain(Go2RoughCfg.terrain):
        # True flat terrain for gait pretraining.
        mesh_type = 'plane'
        measure_heights = True
        curriculum = False

    class commands(Go2RoughCfg.commands):
        class ranges(Go2RoughCfg.commands.ranges):
            lin_vel_x = [-0.5, 1.0]  # min max [m/s]
            lin_vel_y = [-0.2, 0.2]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]

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

            # Auxiliary rewards from appendix Table 5.
            lin_vel_z = -3.0e-4  # 惩罚机体 z 方向线速度，抑制上下抖动/跳动
            feet_stumble = -2.0e-2  # 惩罚脚部绊碰（足端受到异常横向冲击）
            collision = -2.0e-2  # 惩罚机体/连杆非期望碰撞
            dof_pos_limits = -0.2  # 惩罚关节接近或超过位置限位
            torques = -1.0e-6  # 惩罚关节力矩过大，降低能耗和电机负担
            dof_vel = -1.0e-6  # 惩罚关节速度过大，抑制过激动作
            dof_acc = -2.5e-7  # 惩罚关节加速度过大，提升动作平滑性
            action_rate = -5.0e-3  # 一阶动作平滑惩罚：限制相邻时刻动作变化
            action_rate2 = -5.0e-3  # 二阶动作平滑惩罚：限制动作“抖动/顿挫”

            termination = -0.5  # 显式终止惩罚：翻倒/触地后重置时给予负奖励
            
            #overrides
            lin_vel_z = -6.0e-4
            action_rate = -6.0e-3
            action_rate2 = -6.0e-3


            
    class normalization(Go2RoughCfg.normalization):
        # LiDAR points are raw geometric values; keep unscaled.
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            pass

    class domain_rand(Go2RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 5.0]

        # Paper-specific LiDAR randomization.
        lidar_point_mask_ratio = 0.10
        lidar_point_mask_value_range = [0.0, 0.3]
        lidar_distance_noise_ratio = 0.10

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
        history_length = PD_HISTORY_LENGTH
        num_lidar_points = PD_NUM_LIDAR_POINTS
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = 20.0
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proprio_obs_dim = PD_PROPRIO_DIM
        privileged_height_dim = PD_PRIV_HEIGHT_DIM
        privileged_supervision_coef = 1.0

    class algorithm(Go2RoughCfgPPO.algorithm):
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
        num_steps_per_env = 16
        experiment_name = "go2_pd_pretrain"
        run_name = ""
        max_iterations = 600
