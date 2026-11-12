"""Behavioral regression checks against the documented curriculum rules."""
import json
from pathlib import Path
import isaacgym
import numpy as np
import torch
from legged_gym.envs.go2.go2_terrain import GO2Terrain
from legged_gym.envs.go2.go2_terrain_config import GO2TerrainCfg
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.utils.terrain import Terrain


def main():
    cfg = GO2TerrainCfg()
    cfg.terrain.mesh_type = 'heightfield'
    np.random.seed(1)
    terrain = Terrain(cfg.terrain, 4096)
    assert terrain.heightsamples.shape == (1300, 2100)
    b = terrain.border
    assert not terrain.heightsamples[b:b+80,b:b+1600].any()
    assert terrain.env_origins[0,:,2].max() == 0
    assert terrain.env_origins[1,5,2] < 0 < terrain.env_origins[1,12,2]
    # Non-square terrain dimensions and selected mode do not mutate user kwargs.
    cfg.terrain.terrain_width = 6.
    cfg.terrain.num_rows = cfg.terrain.num_cols = 2
    terrain_rect = Terrain(cfg.terrain, 7)
    assert terrain_rect.heightsamples.shape == (660,620)
    cfg.terrain.curriculum = False
    cfg.terrain.selected = True
    cfg.terrain.terrain_kwargs = {'type':'pyramid_stairs_terrain', 'step_width':0.31, 'step_height':0.04}
    Terrain(cfg.terrain,7); Terrain(cfg.terrain,7)
    assert 'type' in cfg.terrain.terrain_kwargs
    cfg = GO2TerrainCfg()
    # Exercise the same vectorized rule without constructing physics or network weights.
    env = GO2Terrain.__new__(GO2Terrain)
    env.cfg, env.device, env.dt, env.num_envs = cfg, 'cpu', 0.02, 8
    env.max_terrain_level = 10
    env.terrain_levels = torch.tensor([0,0,0,0,1,4,9,4])
    env.terrain_types = torch.arange(8)
    env.terrain_origins = torch.tensor(terrain.env_origins, dtype=torch.float32)
    env.env_origins = env.terrain_origins[env.terrain_levels,env.terrain_types].clone()
    env.flat_success_streak = torch.tensor([2,2,2,2,0,0,0,0])
    # Columns are duration, integrated xy score, yaw score, requested and actual progress.
    env.curriculum_stats = torch.tensor([[5,5,5,3,3],[1,1,1,0.2,0.2],
        [5,5,5,0,0],[5,5,5,3,3],[5,4,4,3,3],[5,4,4,3,0.2],
        [5,4,4,3,3],[5,2,2,3,3]], dtype=torch.float32)
    env.curriculum_failure = torch.tensor([False,False,False,True,False,False,False,False])
    env.terrain_boundary = torch.ones(8,dtype=torch.bool)
    env.init_done = True
    env._update_terrain_curriculum(torch.arange(8))
    assert env.terrain_levels[:6].tolist() == [1,0,0,0,2,3]
    assert 1 <= env.terrain_levels[6] <= 9
    assert env.terrain_levels[7] == 4
    assert env.flat_success_streak[1:4].tolist() == [2,2,0]
    assert torch.equal(env.env_origins, env.terrain_origins[env.terrain_levels,env.terrain_types])
    # Failure wins over simultaneous boundary/timeout, so PPO does not bootstrap failures.
    env.rpy = torch.zeros(8,3); env.rpy[0,0] = 1.
    env.contact_forces = torch.zeros(8,1,3); env.termination_contact_indices = torch.tensor([0])
    env.root_states = torch.zeros(8,13); env.root_states[:,:3] = env.env_origins
    env.root_states[:,0] += 3.6
    env.episode_length_buf = torch.ones(8,dtype=torch.long) * 1001
    env.max_episode_length = 1000; env.capture_evaluation = False
    env.check_termination()
    assert env.curriculum_failure[0] and not env.time_out_buf[0]
    assert env.time_out_buf[1:].all()
    # Relative height is invariant under matching terrain/base vertical shifts.
    env.height_samples = torch.tensor(terrain.heightsamples,dtype=torch.float32)
    offsets = torch.tensor([-0.2,0.,0.2]); xx,yy = torch.meshgrid(offsets,offsets,indexing='ij')
    env.ground_offsets = torch.stack((xx.flatten(),yy.flatten()),dim=-1)
    height = env.relative_base_height()
    env.height_samples += 200.; env.root_states[:,2] += 1.
    assert torch.allclose(height,env.relative_base_height(),atol=1e-5)
    original, training = GO2RoughCfg(), GO2RoughCfgPPO()
    assert cfg.control.stiffness == original.control.stiffness == {'joint':20.}
    assert cfg.control.damping == original.control.damping == {'joint':0.5}
    assert cfg.commands.ranges.lin_vel_x == [-1.,1.] and not cfg.commands.curriculum
    assert training.policy.actor_hidden_dims == training.policy.critic_hidden_dims == [512,256,128]
    report = dict(passed=True,checks=['exact flat row and 10x20 map','terrain signs',
        'non-square and repeatable selected terrain','three flat successes',
        'short/turn-only episodes preserve streak','failure clears streak','terrain promotion/demotion',
        'top level revisits 1..9','insufficient tracking holds level','updated origins before reset',
        'failure overrides timeout','relative-height translation invariance','original PD/network/commands'])
    Path('verification/curriculum-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
