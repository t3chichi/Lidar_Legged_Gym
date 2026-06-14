from legged_gym.envs.go2.flat.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO


OBS_HISTORY_LENGTH = 1
PROX_HISTORY_LENGTH = 10
DIST_HISTORY_LENGTH = 10
PD_SPHERICAL_AZIMUTH = 50
PD_SPHERICAL_ELEVATION = 30
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION
PD_PROXIMAL_POINTS = 512
PD_DISTAL_POINTS = 256
PD_PROXIMAL_FEATURE_DIM = 187
PD_DISTAL_FEATURE_DIM = 64
HEADING_OBS_ENABLED = False

PD_PROPRIO_DIM = 48 + (1 if HEADING_OBS_ENABLED else 0)
PD_THETA_DEG = 20.0
MEASURED_GRID_X_COUNT = 17
MEASURED_GRID_Y_COUNT = 11
MEASURED_GRID_X_RANGE = [0.1, 1.5]
MEASURED_GRID_Y_RANGE = [-0.6, 0.6]
PD_PRIV_HEIGHT_DIM = MEASURED_GRID_X_COUNT * MEASURED_GRID_Y_COUNT
PD_PRIV_CRITIC_DIM = PD_PROPRIO_DIM + PD_PRIV_HEIGHT_DIM


class Go2SoftPillarPretrainCfg(Go2RoughCfg):
    class init_state(Go2RoughCfg.init_state):
        pos = [0.0, 0.0, 0.34]
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

        heading_obs_enabled = HEADING_OBS_ENABLED
        heading_noise_enabled = True
        heading_noise_std = 0.05

        n_sectors = 36
        avoid_distance_thresh = 1.0
        avoid_alpha = 2.0
        avoid_beta = 1.0
        avoid_speed_limit = 1.0

        # rays → ω_target
        rays_omega_gain = 0.5
        rays_omega_max  = 0.5
        ray_max_distance = 10.0

        rays_top_ratio = 0.4
        rays_power = 4
        rays_smoothing_alpha = 0.2

        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        num_lidar_points = PD_NUM_LIDAR_POINTS

        # channel_forward
        channel_backward_ratio = 0.5

        # 软预训练标志
        soft_pretrain = True

        # 柱子参数（与正式梅花桩一致，可在 config 中调整）
        pillar_count = 20
        pillar_spawn_radius = 7.5
        pillar_size_x_min = 0.40
        pillar_size_x_max = 0.60
        pillar_size_y_min = 0.40
        pillar_size_y_max = 0.60
        pillar_height_min = 1.00
        pillar_height_max = 1.50
        pillar_min_separation = 3.5
        pillar_center_clear_radius = 2.5
        pillar_allow_height_variation = True

        collision_3d = False

    class env(Go2RoughCfg.env):
        episode_length_s = 15
        num_observations = PD_PROPRIO_DIM + OBS_HISTORY_LENGTH * PD_NUM_LIDAR_POINTS * 3
        num_privileged_obs = PD_PRIV_CRITIC_DIM
        enable_fall_termination = False
        fall_projected_gravity_z_threshold = -0.1
        fall_base_height_threshold = 0.12

    class terrain(Go2RoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = True
        measured_grid_x_range = MEASURED_GRID_X_RANGE
        measured_grid_y_range = MEASURED_GRID_Y_RANGE
        measured_grid_x_count = MEASURED_GRID_X_COUNT
        measured_grid_y_count = MEASURED_GRID_Y_COUNT
        curriculum = False
        num_rows = 4
        num_cols = 4
        terrain_length = 15
        terrain_width = 15

    class asset(Go2RoughCfg.asset):
        self_collisions = 0

    class commands(Go2RoughCfg.commands):
        heading_command = False
        resampling_time = 4.
        class ranges(Go2RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]    # 无角速度指令

    class raycaster(Go2RoughCfg.raycaster):
        enable_raycast = True
        ray_pattern = "spherical"
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        max_distance = 50.0
        attach_yaw_only = False
        vertical_fov_deg_min = -2.0
        vertical_fov_deg_max = 57.0
        offset_pos = [0.28945, 0.0, -0.046825]
        sensor_offset_rpy = [0.0, -2.8782, 3.14]

    class rewards(Go2RoughCfg.rewards):
        base_height_target = 0.34
        class scales(Go2RoughCfg.rewards.scales):
            vel_avoid = 1.0
            rays = 0.5

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
            termination = -0.0  # 终止惩罚：在环境终止时给予负奖励，避免策略利用终止状态
            tracking_lin_vel = 0.0  # 线速度跟踪奖励：鼓励按指令线速度前进
            tracking_ang_vel = 0.0  # 角速度跟踪奖励：鼓励按指令角速度跟踪朝向
            lin_vel_z = -2.0  # 垂直速度惩罚：抑制机体在 z 方向的上下抖动或跳动
            ang_vel_xy = -0.05  # 横向角速度惩罚：抑制 roll/pitch 方向过大角速度，保持机体稳定
            orientation = -0.  # 姿态偏差惩罚：惩罚与目标姿态的偏离，鼓励保持期望姿态
            torques = -0.00001  # 关节力矩惩罚：减少能耗并限制电机过载
            dof_vel = -0.  # 关节速度惩罚：抑制关节速度过大，避免剧烈或不稳定动作
            dof_acc = -2.5e-7  # 关节加速度惩罚：提升动作平滑性，减少瞬时加速度带来的冲击
            base_height = -0.  # 基座高度惩罚：鼓励保持目标基座高度，防止过低或过高
            feet_air_time = 1.0  # 足端离地时间权重：影响步态周期与接触模式，鼓励合理的离地时间
            collision = -1.  # 碰撞惩罚：对机体或连杆发生非期望碰撞时给予负奖励
            feet_stumble = -0.0  # 足端绊碰惩罚：惩罚脚部异常冲击或失稳事件
            action_rate = -0.01  # 动作变化率惩罚：限制相邻动作变化，平滑控制信号
            stand_still = -0.  # 静止惩罚：惩罚长时间静止，防止策略不移动以“获利”

            # Overrides
            orientation = -5.0  # 覆盖的姿态惩罚权重：更强烈地惩罚姿态偏差
            torques = -0.000025  # 覆盖的力矩惩罚权重：更严格地限制关节力矩
            feet_air_time = 1.0  # 覆盖的足端离地时间权重：保持/强调期望步态空中时间
            ang_vel_xy = -0.1
            base_height = -5.0
            # feet_contact_forces = -0.01
            # gait_scheduler = -3

            gait_2_step = -1.0  # 步态相关惩罚/奖励：用于鼓励或抑制特定步态（如两步周期）

    class normalization(Go2RoughCfg.normalization):
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            heading = 1.0

    class domain_rand(Go2RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        lidar_point_mask_ratio = 0.05
        lidar_point_mask_value_range = [2.0, 10.0]
        lidar_distance_noise_ratio = 0.02
        payload_mass_range = [-1.0, 3.0]
        com_shift_range = [[-0.1, -0.15, -0.2], [0.1, 0.15, 0.2]]
        restitution_range = [0.0, 1.0]
        motor_strength_range = [0.8, 1.2]
        joint_calib_offset_range = [-0.02, 0.02]
        gravity_offset_range = [-1.0, 1.0]
        proprio_delay_range = [0.005, 0.045]


class Go2SoftPillarPretrainCfgPPO(Go2RoughCfgPPO):
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
        sensor_offset_pos = [0.28945, 0.0, -0.046825]

    class algorithm(Go2RoughCfgPPO.algorithm):
        amp_enabled = True
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
        experiment_name = "go2_soft_pretrain"
        run_name = ""
        max_iterations = 1000
