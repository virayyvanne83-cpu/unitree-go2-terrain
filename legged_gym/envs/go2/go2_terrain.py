"""Go2 terrain loading and reset-based difficulty curriculum.

The layout/origin scheme follows upstream legged_gym, with a flat qualification
gate and bounded episodes added here. Plane Go2 continues to use LeggedRobot
directly, including its original observations.
"""
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain


class Go2Terrain(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.last_last_actions = torch.zeros_like(self.actions)
        self.rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        from isaacgym import gymtorch
        self.rigid_body_states = gymtorch.wrap_tensor(self.rigid_body_states).view(self.num_envs, -1, 13)
        self.flat_success_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # elapsed seconds, xy tracking, yaw tracking, commanded distance, forward progress
        self.curriculum_stats = torch.zeros(self.num_envs, 5, device=self.device)
        self.curriculum_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terrain_boundary = torch.zeros_like(self.curriculum_failure)
        offsets = torch.tensor([-0.2, 0.0, 0.2], device=self.device)
        xx, yy = torch.meshgrid(offsets, offsets, indexing='ij')
        self.ground_offsets = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        self.local_ground_height = torch.zeros(self.num_envs, device=self.device)
        self.last_curriculum_update = dict(promoted=0.0, demoted=0.0)
        # Local inference/reporting hooks; the supplied curriculum remains below.
        self.freeze_curriculum = False
        self.fixed_command = None
        self.capture_evaluation = False
        self.reward_terms = {}

    def _post_physics_step_callback(self):
        # Evaluate the command that produced this transition, before command resampling.
        cmd = self.commands[:, :2]
        speed = torch.norm(cmd, dim=1)
        progress = torch.sum(self.base_lin_vel[:, :2] * cmd, dim=1) / speed.clamp(min=1e-6)
        self.curriculum_stats[:, 0] += self.dt
        self.curriculum_stats[:, 1] += self._reward_tracking_lin_vel() * self.dt
        self.curriculum_stats[:, 2] += self._reward_tracking_ang_vel() * self.dt
        self.curriculum_stats[:, 3] += speed * self.dt
        self.curriculum_stats[:, 4] += progress.clamp(min=0) * self.dt
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        points = self.root_states[:, None, :2] + self.ground_offsets[None]
        self.local_ground_height[:] = self._sample_ground_height(points).mean(dim=1)
        if self.capture_evaluation:
            self.evaluation_state = torch.stack((self.base_lin_vel[:, 0], self.base_lin_vel[:, 1],
                self.base_ang_vel[:, 2], self.relative_base_height(),
                torch.acos((-self.projected_gravity[:, 2]).clamp(-1., 1.))), dim=1).clone()
            self.evaluation_command = self.commands[:, :3].clone()
        super()._post_physics_step_callback()

    def _sample_ground_height(self, points):
        """Bilinear height lookup for rewards only; not an actor/critic input."""
        pixels = (points + self.cfg.terrain.border_size) / self.cfg.terrain.horizontal_scale
        x = pixels[..., 0].clamp(0, self.height_samples.shape[0] - 1.001)
        y = pixels[..., 1].clamp(0, self.height_samples.shape[1] - 1.001)
        ix, iy = x.long(), y.long()
        fx, fy = x - ix, y - iy
        h = ((1-fx) * (1-fy) * self.height_samples[ix, iy] +
             fx * (1-fy) * self.height_samples[ix+1, iy] +
             (1-fx) * fy * self.height_samples[ix, iy+1] +
             fx * fy * self.height_samples[ix+1, iy+1])
        return h * self.cfg.terrain.vertical_scale

    def check_termination(self):
        super().check_termination()
        self.curriculum_failure[:] = (
            torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices], dim=-1) > 1, dim=1) |
            (self.rpy[:, 0].abs() > .8) | (self.rpy[:, 1].abs() > 1.0))
        displacement = (self.root_states[:, :2] - self.env_origins[:, :2]).abs()
        half_size = torch.tensor([self.cfg.terrain.terrain_length, self.cfg.terrain.terrain_width],
                                 device=self.device) / 2 - self.cfg.terrain.boundary_margin
        self.terrain_boundary[:] = torch.any(displacement >= half_size, dim=1)
        # Crossing the tile boundary ends an episode before touching a different level.
        # It is a truncation (PPO bootstraps), unless the robot also physically failed.
        self.time_out_buf |= self.terrain_boundary & ~self.curriculum_failure
        self.time_out_buf &= ~self.curriculum_failure
        self.reset_buf |= self.terrain_boundary
        if self.capture_evaluation:
            self.evaluation_failure = self.curriculum_failure.clone()
            self.evaluation_boundary = (self.terrain_boundary & ~self.curriculum_failure).clone()
            self.evaluation_timeout = (self.time_out_buf & ~self.terrain_boundary).clone()

    def post_physics_step(self):
        previous_actions = self.last_actions.clone()
        super().post_physics_step()
        self.last_last_actions[:] = previous_actions
        self.last_last_actions[self.reset_buf.bool()] = 0

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type not in ('heightfield', 'trimesh'):
            raise ValueError('go2_terrain requires heightfield or trimesh; use task go2 for plane')
        self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'heightfield':
            self._create_heightfield()
        else:
            self._create_trimesh()
        self.height_samples = torch.from_numpy(self.terrain.heightsamples).to(self.device)
        self._create_envs()

    def _create_heightfield(self):
        params = gymapi.HeightFieldParams()
        params.column_scale = self.cfg.terrain.horizontal_scale
        params.row_scale = self.cfg.terrain.horizontal_scale
        params.vertical_scale = self.cfg.terrain.vertical_scale
        # Isaac's binding names these axes opposite to the NumPy map axes.
        params.nbRows = self.terrain.tot_cols
        params.nbColumns = self.terrain.tot_rows
        self._set_contact_params(params)
        # Preview 4 consumes x-fast samples: transpose the [x,y] NumPy map's
        # memory order, not its world coordinates. Verified with non-square ramps
        # and first-contact probes; C order here produces striped/misaligned ground.
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.ravel(order='F'), params)

    def _create_trimesh(self):
        params = gymapi.TriangleMeshParams()
        params.nb_vertices = len(self.terrain.vertices)
        params.nb_triangles = len(self.terrain.triangles)
        self._set_contact_params(params)
        self.gym.add_triangle_mesh(
            self.sim, self.terrain.vertices.ravel(order='C'),
            self.terrain.triangles.ravel(order='C'), params)

    def _set_contact_params(self, params):
        params.transform.p.x = -self.cfg.terrain.border_size
        params.transform.p.y = -self.cfg.terrain.border_size
        params.static_friction = self.cfg.terrain.static_friction
        params.dynamic_friction = self.cfg.terrain.dynamic_friction
        params.restitution = self.cfg.terrain.restitution

    def _get_env_origins(self):
        self.custom_origins = True
        cfg = self.cfg.terrain
        max_initial = cfg.max_init_terrain_level if cfg.curriculum else cfg.num_rows - 1
        if not 0 <= max_initial < cfg.num_rows:
            raise ValueError('max_init_terrain_level must index an existing terrain row')
        self.terrain_levels = torch.randint(0, max_initial + 1, (self.num_envs,), device=self.device)
        # Integer arithmetic supports 4096/20 and small previews without divisibility constraints.
        self.terrain_types = torch.arange(self.num_envs, device=self.device) * cfg.num_cols // self.num_envs
        self.max_terrain_level = cfg.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device, dtype=torch.float)
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        super().reset_idx(env_ids)
        self.last_last_actions[env_ids] = 0
        self.last_contacts[env_ids] = False
        self.curriculum_stats[env_ids] = 0
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        episode = self.extras['episode']
        episode['terrain_level'] = self.terrain_levels.float().mean()
        episode['flat_fraction'] = (self.terrain_levels == 0).float().mean()
        episode['flat_success_streak'] = self.flat_success_streak.float().mean()
        episode['terrain_promoted_fraction'] = self.last_curriculum_update['promoted']
        episode['terrain_demoted_fraction'] = self.last_curriculum_update['demoted']
        for level in range(self.max_terrain_level):
            episode['terrain_level_{}_fraction'.format(level)] = (self.terrain_levels == level).float().mean()

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done or len(env_ids) == 0 or getattr(self, 'freeze_curriculum', False):
            return
        cfg = self.cfg.terrain
        stats = self.curriculum_stats[env_ids]
        elapsed = stats[:, 0]
        xy_tracking = stats[:, 1] / elapsed.clamp(min=self.dt)
        yaw_tracking = stats[:, 2] / elapsed.clamp(min=self.dt)
        informative = (elapsed >= cfg.flat_min_episode_s) & (stats[:, 3] >= cfg.flat_min_command_distance)
        progress_ok = stats[:, 4] >= cfg.min_progress_ratio * stats[:, 3]
        failed = self.curriculum_failure[env_ids] & (elapsed > 0)
        flat = self.terrain_levels[env_ids] == 0
        flat_success = (informative & ~failed & progress_ok &
                        (xy_tracking >= cfg.flat_tracking_threshold) & (yaw_tracking >= cfg.flat_tracking_threshold))
        streak = self.flat_success_streak[env_ids]
        streak = torch.where(flat & flat_success, streak + 1, streak)
        # Short initial runner phases / pure turning do not count as walking failures.
        streak = torch.where(flat & (failed | (informative & ~flat_success)), 0, streak)
        move_up_flat = flat & (streak >= cfg.flat_success_episodes)
        move_up_terrain = (~flat & informative & ~failed & progress_ok & self.terrain_boundary[env_ids] &
                           (xy_tracking >= cfg.terrain_tracking_threshold) & (yaw_tracking >= cfg.terrain_tracking_threshold))
        move_up = move_up_flat | move_up_terrain
        move_down = ~flat & (failed | (informative & ~progress_ok)) & ~move_up
        self.flat_success_streak[env_ids] = torch.where(move_up | move_down, 0, streak)
        levels = self.terrain_levels[env_ids] + move_up.long() - move_down.long()
        self.terrain_levels[env_ids] = torch.where(
            levels >= self.max_terrain_level,
            torch.randint_like(levels, low=1, high=self.max_terrain_level), levels.clamp(min=0))
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.last_curriculum_update = dict(promoted=move_up.float().mean().item(),
                                           demoted=move_down.float().mean().item())

    def _reward_base_height(self):
        height = self.root_states[:, 2] - self.local_ground_height
        return ((height - self.cfg.rewards.base_height_target) * self.cfg.rewards.height_error_scale).square()

    def _reward_joint_power(self):
        return (self.dof_vel * self.torques).abs().sum(dim=1)

    def _reward_smoothness(self):
        return (self.actions - 2 * self.last_actions + self.last_last_actions).square().sum(dim=1)

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1
        velocity = self.rigid_body_states[:, self.feet_indices, 7:9]
        return (velocity.square().sum(dim=-1) * contact).sum(dim=1)

    def _reward_feet_stumble(self):
        return self._reward_stumble().float()

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1
        filtered = contact | self.last_contacts
        first_contact = (self.feet_air_time > 0) & filtered
        self.feet_air_time += self.dt
        reward = ((self.feet_air_time - self.cfg.rewards.air_time_target) * first_contact).sum(dim=1)
        reward *= torch.norm(self.commands[:, :2], dim=1) > .1
        self.feet_air_time *= ~filtered
        self.last_contacts[:] = contact
        return reward

    def relative_base_height(self):
        points = self.root_states[:, None, :2] + self.ground_offsets[None]
        return self.root_states[:, 2] - self._sample_ground_height(points).mean(dim=1)

    def _resample_commands(self, env_ids):
        if getattr(self, 'fixed_command', None) is None:
            super()._resample_commands(env_ids)
        else:
            self.commands[env_ids, :3] = self.fixed_command

    def compute_reward(self):
        # Evaluate stateful rewards (air time) exactly once; retain terms for reports.
        self.rew_buf[:] = 0.
        for name, fn in zip(self.reward_names, self.reward_functions):
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if self.capture_evaluation:
                self.reward_terms[name] = rew.clone()
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf.clamp_min_(0.)
        if 'termination' in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales['termination']
            self.rew_buf += rew
            self.episode_sums['termination'] += rew
            if self.capture_evaluation:
                self.reward_terms['termination'] = rew.clone()


# Retain the local public name used by task registration and verification tools.
GO2Terrain = Go2Terrain
