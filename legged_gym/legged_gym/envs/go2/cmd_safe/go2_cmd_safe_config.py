# legged_gym/legged_gym/envs/go2/cmd_safe/go2_cmd_safe_config.py

from legged_gym.envs.go2.flat.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO
from legged_gym.envs.go2.lidar_pd_risknet.go2_lidar_pd_risknet_config import (
    OBS_HISTORY_LENGTH,
    PROX_HISTORY_LENGTH,
    DIST_HISTORY_LENGTH,
    PD_SPHERICAL_AZIMUTH,
    PD_SPHERICAL_ELEVATION,
    PD_NUM_LIDAR_POINTS,
    PD_PROXIMAL_POINTS,
    PD_DISTAL_POINTS,
    PD_PROXIMAL_FEATURE_DIM,
    PD_DISTAL_FEATURE_DIM,
    PD_PROPRIO_DIM,
    PD_THETA_DEG,
    MEASURED_GRID_X_COUNT,
    MEASURED_GRID_Y_COUNT,
    MEASURED_GRID_X_RANGE,
    MEASURED_GRID_Y_RANGE,
    PD_PRIV_HEIGHT_DIM,
    PD_PRIV_CRITIC_DIM,
)


class Go2CmdSafeCfg(Go2RoughCfg):
    """Command-safe velocity reward config.

    Extends Go2RoughCfg directly (not Go2LidarPDRiskNetCfg).
    Incorporates selected fields from go2_lidar_pd_risknet_config.
    """

    class asset(Go2RoughCfg.asset):
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]
        penalize_contacts_on = ["thigh", "calf", "Head_upper", "Head_lower", "base"]

    class init_state(Go2RoughCfg.init_state):
        randomize_rot = True
        rot_randomization_range = [-0.2, 0.2]
        spawn_offset_range = 0.4

    class pd_risknet:
        enabled = True
        history_length = OBS_HISTORY_LENGTH
        proximal_feature_dim = PD_PROXIMAL_FEATURE_DIM
        distal_feature_dim = PD_DISTAL_FEATURE_DIM
        proximal_points = PD_PROXIMAL_POINTS
        distal_points = PD_DISTAL_POINTS
        split_theta_deg = PD_THETA_DEG
        n_sectors = 36
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        num_lidar_points = PD_SPHERICAL_AZIMUTH * PD_SPHERICAL_ELEVATION
        ray_max_distance = 10.0
        goal_enabled = True
        goal_reward = 20.0
        move_down_ratio = 0.5
        consecutive_upgrade_episodes = 5
        consecutive_downgrade_episodes = 3
        collision_3d = False
        channel_backward_ratio = 0.5

    class cmd_safe:
        body_semi_length = 0.188   # a: collision box L/2 (0.3762/2)
        body_semi_width  = 0.047   # b: collision box W/2 (0.0935/2)
        z_thresh_high = 0.10       # body top + 3.3cm clearance
        z_thresh_low  = -0.20      # body bottom + 14cm leg clearance
        d_safety   = 0.10          # additional safety gap (m)
        d_safe_max = 1.0           # distance where safe=1
        cmd_safe_sigma = 0.25      # gaussian kernel width
        dist_penalty_thresh = 0.5  # penalty activates below this (m)
        dist_penalty_alpha  = 0.5  # penalty scale factor

    class env(Go2RoughCfg.env):
        num_observations = PD_PROPRIO_DIM + OBS_HISTORY_LENGTH * PD_NUM_LIDAR_POINTS * 3
        num_privileged_obs = PD_PRIV_CRITIC_DIM
        enable_fall_termination = True
        fall_projected_gravity_z_threshold = -0.1
        fall_base_height_threshold = 0.1

    class terrain(Go2RoughCfg.terrain):
        horizontal_scale = 0.1
        measure_heights = True
        measured_grid_x_range = MEASURED_GRID_X_RANGE
        measured_grid_y_range = MEASURED_GRID_Y_RANGE
        measured_grid_x_count = MEASURED_GRID_X_COUNT
        measured_grid_y_count = MEASURED_GRID_Y_COUNT
        mesh_type = 'trimesh'
        curriculum = True
        max_init_terrain_level = 0
        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 1.0]
        terrain_length = 15
        terrain_width = 15
        num_rows = 5
        num_cols = 4
        corridor_width = 2.4
        wall_height = 1.5
        wall_thickness = 2
        turn_angle_deg_max = 55.0
        diagonal_length = 3.0
        end_margin = 0.5
        goal_forward_margin = 1.0
        goal_radius = 1.8

    class commands(Go2RoughCfg.commands):
        heading_command = True
        resampling_time = 2.
        curriculum = False

        class ranges(Go2RoughCfg.commands.ranges):
            lin_vel_x = [0.5, 1.0]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]
            heading = [0, 0]

    class obstacle_gen(Go2RoughCfg.obstacle_gen):
        enable_obstacles = False

    class raycaster(Go2RoughCfg.raycaster):
        enable_raycast = True
        ray_pattern = "spherical"
        spherical_num_azimuth = PD_SPHERICAL_AZIMUTH
        spherical_num_elevation = PD_SPHERICAL_ELEVATION
        max_distance = 10.0
        attach_yaw_only = False
        vertical_fov_deg_min = -2.0
        vertical_fov_deg_max = 57.0
        offset_pos = [0.0, 0.0, 0.0]          # LiDAR at body centre
        sensor_offset_rpy = [0.0, 0.0, 0.0]   # horizontal, no tilt

    class rewards(Go2RoughCfg.rewards):
        base_height_target = 0.34

        class scales:
            # ── New core rewards ──
            cmd_safe_vel        = 2.0
            sector_dist_penalty = 0.5

            # ── Task-specific rewards ──
            goal            = 20.0
            channel_forward = 10.0

            # ── Safety ──
            collision    = -2.0e-2
            termination  = -10.0

            # ── Auxiliary (paper Table 5) ──
            lin_vel_z    = -3.0e-4
            feet_stumble = -2.0e-2
            dof_pos_limits = -0.2
            torques      = -1.0e-6
            dof_vel      = -1.0e-6
            dof_acc      = -2.5e-7
            action_rate  = -5.0e-3
            action_rate2 = -5.0e-3

            # ── Explicitly zeroed ──
            vel_avoid       = 0.0
            rays            = 0.0
            base_height     = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            ang_vel_xy      = 0.0
            orientation     = 0.0
            feet_air_time   = 0.0
            gait_2_step     = 0.0
            curvature       = 0.0
            ang_vel_yaw_penalty = 0.0
            stand_still     = 0.0

    class normalization(Go2RoughCfg.normalization):
        pass

    class domain_rand(Go2RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        lidar_point_mask_ratio = 0.02
        lidar_point_mask_value_range = [0, 0.3]
        lidar_distance_noise_ratio = 0.02
        payload_mass_range = [-1.0, 3.0]
        com_shift_range = [[-0.1, -0.15, -0.2], [0.1, 0.15, 0.2]]
        restitution_range = [0.0, 1.0]
        motor_strength_range = [0.8, 1.2]
        joint_calib_offset_range = [-0.02, 0.02]
        gravity_offset_range = [-1.0, 1.0]
        proprio_delay_range = [0.005, 0.045]

    class sim(Go2RoughCfg.sim):
        class physx(Go2RoughCfg.sim.physx):
            num_threads = 10
            max_gpu_contact_pairs = 2 ** 25
            default_buffer_size_multiplier = 10

    class replay:
        enable_collision_replay = True
        replay_prob = 0.8
        early_reset_prob_range = [0.1, 0.5]
        undo_steps_range = [100, 150]
        max_collision_points = 10


class Go2CmdSafeCfgPPO(Go2RoughCfgPPO):
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
        sensor_offset_rpy = [0.0, 0.0, 0.0]
        sensor_offset_pos = [0.0, 0.0, 0.0]

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
        experiment_name = "go2_cmd_safe"
        run_name = ""
        max_iterations = 4000
