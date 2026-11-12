"""Bounded terrain atlas/start distribution viewer using standing PD, not a policy."""
import argparse
import json
from pathlib import Path
import sys
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.terrain_playback import place_environments, configure_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['atlas', 'start'], default='atlas')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--mesh-type', choices=['trimesh', 'heightfield'], default='trimesh')
    parser.add_argument('--duration', type=float, default=10.)
    parser.add_argument('--report')
    parser.add_argument('--screenshot', help='Save the native viewer image; requires graphics mode')
    options, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    args.task = 'go2_terrain'
    args.num_envs = options.num_envs or args.num_envs or (200 if options.mode == 'atlas' else 4096)
    cfg, _ = task_registry.get_cfgs(args.task)
    configure_evaluation(cfg)
    cfg.terrain.mesh_type = options.mesh_type
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        ids = torch.arange(env.num_envs, device=env.device)
        if options.mode == 'atlas':
            env.terrain_types[:] = (ids // cfg.terrain.num_rows) % cfg.terrain.num_cols
        place_environments(env, -1 if options.mode == 'atlas' else 0)
        env.fixed_command = torch.zeros(3, device=env.device)
        env.capture_evaluation = True
        env.reset_idx(ids)
        initial_levels = torch.bincount(env.terrain_levels, minlength=10).tolist()
        env.compute_observations()
        center = env.env_origins[0].cpu().numpy()
        if env.viewer:
            env.set_camera(center + [-8., -10., 7.], center)
        min_relative_z = float('inf')
        steps = max(1, round(options.duration / env.dt))
        for _ in range(steps):
            obs, _, rew, _, _ = env.step(torch.zeros(env.num_envs, 12, device=env.device))
            assert obs.shape == (env.num_envs, 48)
            assert torch.isfinite(obs).all() and torch.isfinite(rew).all() and torch.isfinite(env.root_states).all()
            assert torch.isfinite(env.evaluation_state).all()
            min_relative_z = min(min_relative_z, float(env.evaluation_state[:, 3].min()))
        assert min_relative_z > -0.5, 'Robot fell below the sampled terrain: {}'.format(min_relative_z)
        report = dict(mode=options.mode, mesh_type=options.mesh_type, num_envs=env.num_envs,
            initial_level_counts=initial_levels, observation_shape=list(obs.shape), steps=steps,
            gpu_tensor_device=str(obs.device), min_relative_base_height=min_relative_z,
            weights_updated=False, note='Standing PD preview; not locomotion performance.')
        if options.screenshot:
            if env.viewer is None:
                raise ValueError('--screenshot requires graphics mode')
            dest = Path(options.screenshot).resolve(); dest.parent.mkdir(parents=True, exist_ok=True)
            env.gym.write_viewer_image_to_file(env.viewer, str(dest))
        if options.report:
            dest = Path(options.report); dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        if env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
