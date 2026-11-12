"""Summarize real TensorBoard scalars and checkpoint identity; never updates weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    accumulator = EventAccumulator(str(run), size_guidance={'scalars':0}).Reload()
    summaries, series = {}, {}
    for tag in accumulator.Tags()['scalars']:
        events = accumulator.Scalars(tag)
        values = np.array([e.value for e in events])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite TensorBoard scalar: ' + tag)
        summaries[tag] = dict(count=len(events), first_step=events[0].step, last_step=events[-1].step,
            first100_mean=float(values[:100].mean()), last100_mean=float(values[-100:].mean()),
            last=float(values[-1]))
        series[tag] = ([e.step for e in events], values)
    paths = sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    checkpoints = []
    for path in ([paths[0],paths[-1]] if len(paths)>1 else paths):
        data = torch.load(str(path),map_location='cpu')
        state = data['model_state_dict']
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite checkpoint: ' + str(path))
        checkpoints.append(dict(path=str(path),iter=data.get('iter'),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            actor_weight_shapes={k:list(v.shape) for k,v in state.items() if k.startswith('actor.') and k.endswith('weight')}))
    changed = None
    if len(paths)>1:
        a = torch.load(str(paths[0]),map_location='cpu')['model_state_dict']
        b = torch.load(str(paths[-1]),map_location='cpu')['model_state_dict']
        changed = sum(not torch.equal(a[k],b[k]) for k in a if k.startswith('actor.'))
    report = dict(run=str(run),scalars=summaries,checkpoints=checkpoints,
        changed_actor_tensors=changed,finite=True,
        provenance='Fresh local reconstruction. Original model_1500 weights and raw events were not provided.')
    (output/'training_summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    tags=['Train/mean_reward','Episode/terrain_level','Episode/flat_fraction',
          'Train/mean_episode_length','Loss/value_function','Episode/terrain_level_9_fraction']
    for ax,tag in zip(axes.flat,tags):
        if tag in series:
            x,y=series[tag];ax.plot(x,y,linewidth=0.8,alpha=0.6)
            width=min(50,len(y))
            if width>1:
                ax.plot(x[width-1:],np.convolve(y,np.ones(width)/width,mode='valid'),linewidth=1.5)
        ax.set_title(tag);ax.set_xlabel('PPO iteration');ax.grid(alpha=0.25)
    fig.savefig(str(output/'training_curves.png'),dpi=150)
    plt.close(fig)
    print(json.dumps({'run':str(run),'last_iteration':checkpoints[-1]['iter'] if checkpoints else None,
        'changed_actor_tensors':changed,'scalar_count':len(summaries),'finite':True},indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();analyze(args.run,args.output)
