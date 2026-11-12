"""Frozen ordered-map evaluation; never constructs a PPO optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict, get_args, export_policy_as_jit
from legged_gym.utils.task_registry import task_registry


def terrain_family(column):
    if column < 2:
        return 'smooth_slope'
    if column < 4:
        return 'rough_slope'
    if column < 11:
        return 'stairs_negative_center'
    if column < 16:
        return 'stairs_positive_center'
    return 'obstacles'


def checkpoint_path(args, train_cfg):
    root = Path(LEGGED_GYM_ROOT_DIR) / 'logs' / (args.experiment_name or train_cfg.runner.experiment_name)
    if args.load_run and args.load_run != '-1':
        run = root / args.load_run
    else:
        runs = [p for p in root.glob('*') if p.is_dir() and list(p.glob('model_*.pt'))]
        if not runs:
            raise FileNotFoundError('No training checkpoints in {}. Train first or import the original run.'.format(root))
        run = max(runs, key=lambda p: p.stat().st_mtime)
    if args.checkpoint is not None and args.checkpoint >= 0:
        path = run / 'model_{}.pt'.format(args.checkpoint)
    else:
        models = list(run.glob('model_*.pt'))
        if not models:
            raise FileNotFoundError('No checkpoints in ' + str(run))
        path = max(models, key=lambda p: int(p.stem.split('_')[-1]))
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path.resolve()


def load_actor_critic(path, train_cfg, device):
    checkpoint = torch.load(str(path), map_location=device)
    model = ActorCritic(48, 48, 12, **class_to_dict(train_cfg.policy)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite checkpoint weights')
    return model, checkpoint.get('iter')


def configure_evaluation(cfg):
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.push_robots = False
    cfg.commands.heading_command = False
    # Keep curriculum=True during construction to preserve the ordered 10x20 map.


def place_environments(env, level, column=None):
    env.freeze_curriculum = True
    ids = torch.arange(env.num_envs, device=env.device)
    if column is not None:
        env.terrain_types[:] = column
    env.terrain_levels[:] = ids % env.max_terrain_level if level == -1 else level
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]


def play_terrain(args, options):
    cfg, train_cfg = task_registry.get_cfgs('go2_terrain')
    if args.seed is not None:
        cfg.seed = args.seed
    configure_evaluation(cfg)
    cfg.env.num_envs = args.num_envs or 100
    if options.terrain_level < -1 or options.terrain_level >= cfg.terrain.num_rows:
        raise ValueError('terrain-level must be -1 or 0..9')
    if options.terrain_column is not None and not 0 <= options.terrain_column < cfg.terrain.num_cols:
        raise ValueError('terrain-column must be 0..19')
    if options.duration <= 0:
        raise ValueError('duration must be positive')
    path = checkpoint_path(args, train_cfg)  # fail before allocating GPU simulation
    model, iteration = load_actor_critic(path, train_cfg, args.rl_device)
    export_dir = path.parent / 'exported' / path.stem
    export_policy_as_jit(model, str(export_dir))
    env = None
    try:
        env, _ = task_registry.make_env('go2_terrain', args=args, env_cfg=cfg)
        place_environments(env, options.terrain_level, options.terrain_column)
        env.fixed_command = torch.tensor(options.cmd, device=env.device)
        env.capture_evaluation = True
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        env.compute_observations()
        obs = env.get_observations()
        if env.viewer:
            center = env.env_origins[0].cpu().numpy()
            env.set_camera(center + [-3., -3., 2.], center + [0., 0., 0.3])
        if options.fast:
            env.render = lambda *a, **k: type(env).render(env, sync_frame_time=False)
        n_groups = cfg.terrain.num_rows * cfg.terrain.num_cols
        groups = env.terrain_levels * cfg.terrain.num_cols + env.terrain_types
        counts = torch.bincount(groups, minlength=n_groups)
        totals = torch.zeros(n_groups, 11, device=env.device)
        reward_totals = {name: torch.zeros(n_groups, device=env.device) for name in env.reward_scales}
        steps = int(round(options.duration / env.dt))
        with torch.inference_mode():
            for _ in range(steps):
                obs, _, reward, _, _ = env.step(model.act_inference(obs.to(args.rl_device)).to(env.device))
                state = env.evaluation_state
                error = state[:, :3] - env.evaluation_command
                values = torch.cat((state, error**2, env.evaluation_failure[:, None],
                        env.evaluation_boundary[:, None], env.evaluation_timeout[:, None]), dim=1)
                totals.index_add_(0, groups, values)
                for name, values in env.reward_terms.items():
                    reward_totals[name].index_add_(0, groups, values)
                if not torch.isfinite(obs).all() or not torch.isfinite(state).all() or not torch.isfinite(reward).all():
                    raise RuntimeError('Nonfinite simulation state, observation or reward')
        rows = []
        for g in (counts > 0).nonzero().flatten().tolist():
            count = int(counts[g])
            data = totals[g].cpu().numpy()
            rows.append(dict(level=g // cfg.terrain.num_cols, column=g % cfg.terrain.num_cols,
                family='flat' if g // cfg.terrain.num_cols == 0 else terrain_family(g % cfg.terrain.num_cols),
                num_envs=count, mean_vx_vy_yaw_relative_height_tilt=(data[:5] / (count*steps)).tolist(),
                rmse_vx_vy_yaw=np.sqrt(data[5:8] / (count*steps)).tolist(),
                body_failures=int(data[8]), boundary_completions=int(data[9]), timeouts=int(data[10]),
                reward_per_sim_second={k: float(v[g]) / (count*steps*env.dt) for k,v in reward_totals.items()}))
        report = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint_iter=iteration, seed=cfg.seed, duration=steps*env.dt, num_envs=env.num_envs,
            command=options.cmd, map_shape=[cfg.terrain.num_rows, cfg.terrain.num_cols],
            reset_randomness=True, weights_updated=False, groups=rows,
            note='Events include repeated episodes and startup/reset transients; not independent pass rates.')
        if options.report:
            dest = Path(options.report); dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != 'groups'}, indent=2))
        return report
    finally:
        if env is not None:
            if env.viewer:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


def main():
    parser = argparse.ArgumentParser(description='Go2 ordered terrain playback; standard train CLI options are also accepted.')
    parser.add_argument('--terrain-level', type=int, default=1)
    parser.add_argument('--terrain-column', type=int)
    parser.add_argument('--cmd', type=float, nargs=3, default=[0.4, 0., 0.])
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--report')
    options, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    play_terrain(args, options)
