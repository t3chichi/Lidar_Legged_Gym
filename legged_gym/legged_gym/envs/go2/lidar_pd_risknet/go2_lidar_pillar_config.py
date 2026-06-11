from legged_gym.envs.go2.flat.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO


OBS_HISTORY_LENGTH = 1
PROX_HISTORY_LENGTH = 10
DIST_HISTORY_LENGTH = 10
PD_SPHERICAL_AZIMUTH = 50
PD_SPHERICAL_ELEVATION = 30
PD_NUM_LIDAR_POINTS = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION
# Prefer denser near-field sampling for collision avoidance cues.
PD_PROXIMAL_POINTS = 512
PD_DISTAL_POINTS = 256
PD_PROXIMAL_FEATURE_DIM = 187
PD_DISTAL_FEATURE_DIM = 64
HEADING_OBS_ENABLED = False

PD_PROPRIO_DIM = 48 + (1 if HEADING_OBS_ENABLED else 0)
PD_THETA_DEG = 20.0
# Height measurement grid: auto-generated from range + count via linspace.
MEASURED_GRID_X_COUNT = 17
MEASURED_GRID_Y_COUNT = 11
MEASURED_GRID_X_RANGE = [-0.0, 2.0]
MEASURED_GRID_Y_RANGE = [-0.7, 0.7]
PD_PRIV_HEIGHT_DIM = MEASURED_GRID_X_COUNT * MEASURED_GRID_Y_COUNT
PD_PRIV_CRITIC_DIM = PD_PROPRIO_DIM + PD_PRIV_HEIGHT_DIM


class Go2LidarPillarCfg(Go2RoughCfg):
    class asset(Go2RoughCfg.asset):
        terminate_after_contacts_on = []

    class init_state(Go2RoughCfg.init_state):
        randomize_rot = True
        rot_randomization_range = [-3.14, 3.14]   # 全向随机出生朝向（无切线参考）
        spawn_offset_range = 0.2                 # 出生点 XY 随机偏移范围 (m)

    class pd_risknet:
        enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = PD_THETA_DEG

        # 观测模式开关及朝向噪声配置
        heading_obs_enabled = HEADING_OBS_ENABLED
        heading_noise_enabled = True
        heading_noise_std = 0.05

        n_sectors = 36
        avoid_distance_thresh = 1.0
        avoid_alpha = 1.0
        avoid_beta = 1.0
        avoid_speed_limit = 1.0  # 避障速度上界 (m/s)

        # rays → ω_target 参数
        rays_omega_gain = 0.5     # k_ω: heading_error → ω_target P 增益
        rays_omega_max  = 0.5     # rad/s: 角速度指令上限
        ray_max_distance = 10.0  # rays 奖励截断距离 (m)，对齐 raycaster.max_distance

        # Rays direction-consistency reward (replaces top-k distance scoring).
        rays_top_ratio = 0.4           # 每扇区取前 40% 最远点进行距离平均
        rays_power = 4                 # 距离归一化权重幂次: w_i = (d_i / d_max)^p
        rays_smoothing_alpha = 0.2     # 世界帧方向 EMA 平滑因子

        # Spherical ray pattern used as raw LiDAR point cloud source.
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        num_lidar_points = spherical_num_azimuth * spherical_num_elevation

        collision_3d = False             # 正式训练：2D 水平连续平方

        # 地形课程升降级
        move_down_ratio = 0.4                 # 降级阈值：forward_dist / goal_dist < 此比例
        consecutive_upgrade_episodes = 3      # 连续 N 回合到达终点才触发升级
        consecutive_downgrade_episodes = 3    # 连续 N 回合未达降级阈值才触发降级

    class env(Go2RoughCfg.env):
        # Base Go2 proprio obs + raw LiDAR history points (N_hist * N_points * xyz).
        num_observations = PD_PROPRIO_DIM + OBS_HISTORY_LENGTH * PD_NUM_LIDAR_POINTS * 3
        # Critic input uses proprio (48) + privileged heights (187).
        num_privileged_obs = PD_PRIV_CRITIC_DIM
        # Anti-flip termination gates to avoid upside-down reward exploitation.
        enable_fall_termination = True
        # In body frame, projected_gravity[:, 2] is near -1 when upright and near +1 when upside-down.
        fall_projected_gravity_z_threshold = -0.1
        # Terminate when base height is unrealistically low (meters).
        fall_base_height_threshold = 0.1

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
        curriculum = False
        max_init_terrain_level = 0  # 所有机器人从 level 0 (直线通道) 开始

        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        terrain_length = 15
        terrain_width = 15
        num_rows = 4
        num_cols = 4

        # 柱子参数（pillar_field_terrain 已通过 getattr 读取）
        pillar_count_min = 30
        pillar_count_max = 30
        pillar_size_x_min = 0.40
        pillar_size_x_max = 0.60
        pillar_size_y_min = 0.40
        pillar_size_y_max = 0.60
        pillar_height_min = 0.60
        pillar_height_max = 1.00
        pillar_min_separation = 2.5  
        pillar_center_clear_radius = 1.6
        pillar_spawn_radius = 9.0
        pillar_allow_height_variation = True


    class commands(Go2RoughCfg.commands):
        heading_command = False
        resampling_time = 2.
        curriculum = False
        class ranges(Go2RoughCfg.commands.ranges):
            lin_vel_x = [0.4, 0.8]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]

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
        max_distance = 50.0
        attach_yaw_only = False
        vertical_fov_deg_min = -2.0   # sensor frame 垂直 FOV 下限 (deg)
        vertical_fov_deg_max = 57.0   # sensor frame 垂直 FOV 上限 (deg)
        # Match unitree_go2.py lidar mount translation (base frame, meters).
        offset_pos = [0.28945, 0.0, -0.046825]
        # Match unitree_go2.py lidar mount fixed rotation (roll, pitch, yaw in radians).
        sensor_offset_rpy = [0.0, -2.8782, 3.14]

    class rewards(Go2RoughCfg.rewards):
        base_height_target = 0.34
        class scales:
            # Paper main rewards.
            vel_avoid = 1.0  # 速度跟踪+避障奖励：鼓励跟踪 (v_cmd + v_avoid)
            rays = 1.5  # 距离最大化奖励：鼓励与障碍保持更大安全间距

            # Auxiliary rewards from appendix Table 5.
            lin_vel_z = -3.0e-4  # 惩罚机体 z 方向线速度，抑制上下抖动/跳动
            feet_stumble = -2.0e-2  # 惩罚脚部绊碰（足端受到异常横向冲击）
            collision = -2.0e-2  # 连续 ||Force_xy||²（对齐论文 “连杆碰撞”）
            dof_pos_limits = -0.2  # 二值越界指示（对齐论文 1_{q>q_max or q<q_min}）
            torques = -1.0e-6  # 惩罚关节力矩过大，降低能耗和电机负担
            dof_vel = -1.0e-6  # 惩罚关节速度过大，抑制过激动作
            dof_acc = -2.5e-7  # 惩罚关节加速度过大，提升动作平滑性
            action_rate = -5.0e-3  # 一阶动作平滑惩罚：限制相邻时刻动作变化
            action_rate2 = -5.0e-3  # 二阶动作平滑惩罚：限制动作”抖动/顿挫”

            # tracking_lin_vel = 5.0e-1   
            tracking_ang_vel = 0.0
            feet_air_time = 1.0      
            gait_2_step = -5.0e-1    
            ang_vel_xy = -5.0e-2
            base_height = -2.0
            orientation = -0.0
            move_distance = 10.0  # pillar 地形：鼓励远离出生点
            
            #override
            # lin_vel_z = -1.0e-3
            tracking_lin_vel = 0.0
      

    class normalization(Go2RoughCfg.normalization):
        # LiDAR points are raw geometric values; keep unscaled.
        class obs_scales(Go2RoughCfg.normalization.obs_scales):
            heading = 1.0

    class domain_rand(Go2RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]

        # Paper-specific LiDAR randomization.
        lidar_point_mask_ratio = 0.02
        lidar_point_mask_value_range = [0, 0.3]
        lidar_distance_noise_ratio = 0.02

        # Remaining parameters are declared for parity and can be consumed by future hooks.
        payload_mass_range = [-1.0, 3.0]
        com_shift_range = [[-0.1, -0.15, -0.2], [0.1, 0.15, 0.2]]
        restitution_range = [0.0, 1.0]
        motor_strength_range = [0.8, 1.2]
        joint_calib_offset_range = [-0.02, 0.02]
        gravity_offset_range = [-1.0, 1.0]
        proprio_delay_range = [0.005, 0.045]

    class sim(Go2RoughCfg.sim):
        class physx(Go2RoughCfg.sim.physx):
            num_threads = 10  # AutoDL使用，4096环境(原10线程)
            max_gpu_contact_pairs = 2**25  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 10

        
class Go2LidarPillarCfgPPO(Go2RoughCfgPPO):
    class policy(Go2RoughCfgPPO.policy):
        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        perception_enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_history_length = 1
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
        experiment_name = "go2_lidar_pillar"
        run_name = ""
        max_iterations = 4000
