"""Reproducible post-training log review, Isaac fixed-map evaluation and MuJoCo checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def evaluate(run, checkpoint=1500, num_envs=4096, duration=20., sim_device='cuda:0', output=None):
    root=Path(__file__).resolve().parents[1]
    run=Path(run).resolve()
    checkpoint_file=run/'model_{}.pt'.format(checkpoint)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(str(checkpoint_file))
    output=Path(output).resolve() if output else run/'evaluation'/checkpoint_file.stem
    output.mkdir(parents=True,exist_ok=True)
    status={'run':str(run),'checkpoint':str(checkpoint_file),'completed':False,'stages':[]}
    status_file=output/'status.json'
    def execute(name,arguments):
        print('Running '+name,flush=True)
        with (output/(name+'.log')).open('w') as log:
            result=subprocess.run([str(root/'bin/run'),'python']+arguments,cwd=str(root),stdout=log,stderr=subprocess.STDOUT)
        status['stages'].append({'stage':name,'exit_code':result.returncode})
        status_file.write_text(json.dumps(status,indent=2))
        if result.returncode:
            raise RuntimeError('{} failed: see {}'.format(name,output/(name+'.log')))
    execute('training_review',['verification/analyze_run.py','--run',str(run),'--output',str(output)])
    policy=run/'exported'/checkpoint_file.stem/'policy_1.pt'
    execute('policy_export',['deploy/deploy_mujoco/export_go2_policy.py','--checkpoint',str(checkpoint_file),'--output',str(policy)])
    for velocity in [0.4,0.8]:
        suffix='forward04' if velocity==0.4 else 'forward08'
        execute('isaac_'+suffix,['legged_gym/scripts/play.py','--task','go2_terrain','--load_run',run.name,
            '--checkpoint',str(checkpoint),'--num_envs',str(num_envs),'--terrain-level','-1','--cmd',str(velocity),'0','0',
            '--duration',str(duration),'--headless','--fast','--report',str(output/('isaac_'+suffix+'.json')),
            '--sim_device',sim_device,'--rl_device',sim_device,'--pipeline','cpu' if sim_device=='cpu' else 'gpu'])
    for level,column in [(0,5),(1,5),(4,0),(4,5),(4,12),(4,16)]:
        name='level{}_col{}'.format(level,column)
        scene=output/'mujoco'/name
        execute('export_'+name,['deploy/deploy_mujoco/export_go2_terrain.py','--level',str(level),'--column',str(column),
            '--seed','1','--policy',str(policy),'--output',str(scene)])
        execute('mujoco_'+name,['deploy/deploy_mujoco/deploy_go2.py','--config',str(scene/'go2.yaml'),
            '--headless','--cmd','0.4','0','0','--duration','6','--fast','--report',str(output/('mujoco_'+name+'.json'))])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for i,suffix in enumerate(['forward04','forward08']):
        report=json.loads((output/('isaac_'+suffix+'.json')).read_text())
        vx=np.full((10,20),np.nan);fail=np.full((10,20),np.nan)
        for group in report['groups']:
            index=(group['level'],group['column'])
            vx[index]=group['mean_vx_vy_yaw_relative_height_tilt'][0]
            fail[index]=group['body_failures']
        for j,(values,title) in enumerate([(vx,'Actual body vx [m/s]'),(fail,'Body failure events')]):
            ax=axes[i,j];im=ax.imshow(values,origin='lower',aspect='auto');fig.colorbar(im,ax=ax)
            ax.set_title('{} / {}'.format(suffix,title));ax.set_xlabel('Terrain column');ax.set_ylabel('Level')
    fig.savefig(str(output/'fixed_terrain_evaluation.png'),dpi=150);plt.close(fig)
    status['completed']=True
    status_file.write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--checkpoint',type=int,default=1500)
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--duration',type=float,default=20.)
    parser.add_argument('--sim-device',default='cuda:0')
    parser.add_argument('--output')
    args=parser.parse_args();evaluate(args.run,args.checkpoint,args.num_envs,args.duration,args.sim_device,args.output)
