"""Export one tile of the ordered Isaac terrain atlas for MuJoCo playback.

Only geometry is generated; no simulator or training process is started. The
MuJoCo heightfield approximates Isaac's vertical-edge-corrected triangle mesh.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import isaacgym  # Required by terrain_utils; import before legged_gym/torch.
import numpy as np
import mujoco
import yaml

from legged_gym.envs.go2.go2_terrain_config import GO2TerrainCfg
from legged_gym.utils.terrain import Terrain

ROOT = Path(__file__).resolve().parents[2]


def export_terrain(output, level, column, seed, policy):
    cfg = GO2TerrainCfg()
    policy = Path(policy).resolve()
    if not policy.is_file():
        raise FileNotFoundError('Export the TorchScript policy first: '+str(policy))
    if not 0 <= level < cfg.terrain.num_rows or not 0 <= column < cfg.terrain.num_cols:
        raise ValueError('level/column must index the training atlas (0..9 / 0..19)')
    cfg.terrain.mesh_type = 'heightfield'  # Same raw map, skip the unused full triangle conversion.
    np.random.seed(seed)
    terrain = Terrain(cfg.terrain, 4096)
    nx, ny, b = terrain.length_per_env_pixels, terrain.width_per_env_pixels, terrain.border
    height = terrain.heightsamples[b+level*nx:b+(level+1)*nx, b+column*ny:b+(column+1)*ny].astype(np.float64) * cfg.terrain.vertical_scale
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    low, high = float(height.min()), float(height.max())
    scale = max(high-low, 1e-6)
    normalized = ((height-low) / scale).astype('<f4').T.copy()
    # MuJoCo rows run along y, columns along x; the input map is indexed [x,y].
    with (output/'heightfield.bin').open('wb') as f:
        f.write(struct.pack('<ii', ny, nx))
        f.write(normalized.tobytes(order='C'))
    model = ET.parse(ROOT/'resources/robots/go2/mujoco/go2.xml').getroot()
    model.set('model', 'go2_terrain_tile')
    model.find('compiler').set('meshdir', os.path.relpath(ROOT/'resources/robots/go2/mujoco/assets', output))
    scene = ET.parse(ROOT/'resources/robots/go2/mujoco/scene.xml').getroot()
    for name in ['visual']:
        model.append(copy.deepcopy(scene.find(name)))
    assets = model.find('asset')
    for element in scene.find('asset'):
        assets.append(copy.deepcopy(element))
    half_x, half_y = (nx-1)*cfg.terrain.horizontal_scale/2, (ny-1)*cfg.terrain.horizontal_scale/2
    ET.SubElement(assets, 'hfield', name='terrain',
                  file=os.path.relpath(output/'heightfield.bin', ROOT/'resources/robots/go2/mujoco/assets'),
                  size=f'{half_x} {half_y} {scale} 0.5')
    world = model.find('worldbody')
    world.append(copy.deepcopy(scene.find('worldbody/light')))
    # Shift the training tile center (4,4) to robot xy=(0,0), retaining its z origin.
    offset_x = half_x - cfg.terrain.terrain_length/2
    offset_y = half_y - cfg.terrain.terrain_width/2
    ET.SubElement(world, 'geom', name='floor', type='hfield', hfield='terrain',
                  pos=f'{offset_x} {offset_y} {low}', material='groundplane',
                  friction='1 0 0', contype='0', conaffinity='1', group='1')
    # Below all terrain: never cover the negative-center stairs with a plane at z=0.
    ET.SubElement(world, 'geom', name='apron', type='plane', size='0 0 .05',
                  pos=f'0 0 {low-.5}', rgba='.2 .2 .2 1', friction='1 0 0',
                  contype='0', conaffinity='1')
    ET.ElementTree(model).write(output/'scene.xml', encoding='unicode')
    deployment = yaml.safe_load((ROOT/'deploy/deploy_mujoco/configs/go2.yaml').read_text())
    deployment.update(xml_path=str(output/'scene.xml'), policy_path=str(policy),
                      init_height=float(terrain.env_origins[level, column, 2]) + cfg.init_state.pos[2])
    metadata = dict(level=level, column=column, seed=seed, xy_center_shift=[4,4],
                    source_origin=terrain.env_origins[level, column].tolist(),
                    height_range=[low,high], grid_shape=[nx,ny], horizontal_scale=cfg.terrain.horizontal_scale,
                    boundary_half_size=cfg.terrain.terrain_length/2-cfg.terrain.boundary_margin,
                    limitation='Heightfield approximation; no Isaac vertical-edge correction or identical contact solver.')
    deployment['terrain'] = metadata
    (output/'go2.yaml').write_text(yaml.safe_dump(deployment,sort_keys=False))
    metadata['map_sha256'] = hashlib.sha256(height.tobytes()).hexdigest()
    (output/'terrain.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Verify normalization and x/y ordering after native MuJoCo compilation.
    compiled = mujoco.MjModel.from_xml_path(str(output/'scene.xml'))
    actual = compiled.hfield_data.reshape(ny,nx).T * scale + low
    np.testing.assert_allclose(actual, height, atol=1e-6)
    data = mujoco.MjData(compiled)
    mujoco.mj_forward(compiled, data)
    ray_group = np.array([0,1,0,0,0,0], dtype=np.uint8)
    errors = []
    for ix in np.linspace(2,nx-3,8).astype(int):
        for iy in np.linspace(3,ny-4,7).astype(int):
            point = np.array([ix*cfg.terrain.horizontal_scale-cfg.terrain.terrain_length/2,
                              iy*cfg.terrain.horizontal_scale-cfg.terrain.terrain_width/2, high+3])
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(compiled, data, point, np.array([0.,0.,-1.]), ray_group, 1, -1, hit)
            if distance < 0:
                raise ValueError('Exported MuJoCo terrain has a missing surface')
            errors.append(abs(point[2]-distance-height[ix,iy]))
    metadata['max_native_height_error'] = max(errors)
    if max(errors) > 1e-5:
        raise ValueError('Exported ground axes/height do not match the training tile: '+str(max(errors)))
    (output/'terrain.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('EXPORTED_TERRAIN '+str(output/'go2.yaml'), flush=True)
    return output/'go2.yaml'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--column', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    export_terrain(args.output, args.level, args.column, args.seed, args.policy)
