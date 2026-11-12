from .go2_config import GO2RoughCfg, GO2RoughCfgPPO


class GO2TerrainCfg(GO2RoughCfg):
    """Blind Go2 locomotion: flat qualification, then independent terrain levels."""

    class env(GO2RoughCfg.env):
        num_envs = 4096

    class terrain(GO2RoughCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        max_init_terrain_level = 0
        flat_rows = 1
        measure_heights = False  # Terrain geometry does not add policy inputs.
        num_rows = 10
        num_cols = 20
        terrain_length = 8.0
        terrain_width = 8.0
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25.0
        # Smooth slope, rough slope, descending stairs, ascending stairs, obstacles.
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_range = [0.04, 0.32]
        step_height_range = [0.04, 0.18]
        obstacle_height_range = [0.03, 0.16]
        roughness_range = [0.01, 0.04]
        boundary_margin = 0.5
        # Metrics use actual simulated time, independent of runner episode phases.
        flat_success_episodes = 3
        flat_min_episode_s = 3.0
        flat_min_command_distance = 2.0
        flat_tracking_threshold = 0.75
        terrain_tracking_threshold = 0.60
        min_progress_ratio = 0.5

    class commands(GO2RoughCfg.commands):
        curriculum = False  # Fixed inherited ranges, no speed curriculum.

    class rewards(GO2RoughCfg.rewards):
        only_positive_rewards = False
        base_height_target = 0.30
        # Height error is normalized by 0.1 m, as in the Lain reference.
        height_error_scale = 10.0
        air_time_target = 0.25

        class scales(GO2RoughCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -0.5
            orientation = -0.2
            base_height = -0.5
            torques = -0.0001
            joint_power = -0.00002
            action_rate = -0.01
            smoothness = -0.005
            feet_air_time = 0.2
            feet_slip = -0.05
            feet_stumble = -0.1
            termination = -2.0

    class viewer(GO2RoughCfg.viewer):
        pos = [13.0, -9.0, 11.0]
        lookat = [4.0, 12.0, 0.0]


class GO2TerrainCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = 'go2_terrain'
