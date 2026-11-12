import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    if args.task == 'go2_terrain' and args.seed is not None:
        cfg, train_cfg = task_registry.get_cfgs(args.task)
        cfg.seed = train_cfg.seed = args.seed
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    if args.task == 'go2_terrain':
        import json
        from pathlib import Path
        from legged_gym.utils.helpers import class_to_dict
        destination = Path(ppo_runner.log_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'config_snapshot.json').write_text(json.dumps(
            {'env': class_to_dict(env_cfg), 'train': class_to_dict(train_cfg),
             'dof_names': env.dof_names,
             'provenance': 'Reconstructed from supplied notes; fresh training, no original weights.'}, indent=2))
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
