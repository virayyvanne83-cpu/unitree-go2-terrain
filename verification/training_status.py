"""Show the current supervised reconstruction run without importing Isaac Gym."""
import json
import os
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
directory=root/'logs/reconstruction'
pidfile=directory/'train.pid'
pid=int(pidfile.read_text()) if pidfile.exists() else None
alive=False
if pid is not None:
    try:
        cmdline=Path('/proc/{}/cmdline'.format(pid)).read_bytes()
        alive=Path('/proc/{}/cwd'.format(pid)).resolve() == root and b'train-go2-terrain' in cmdline
    except OSError:
        pass
log=directory/'train.log'
text=log.read_text(errors='replace') if log.exists() else ''
text=re.sub(r'\x1b\[[0-9;]*m','',text)
iterations=re.findall(r'Learning iteration\s+(\d+)/(\d+)',text)
runs=sorted((root/'logs/go2_terrain').glob('*reconstructed_flat_to_terrain'),key=lambda p:p.stat().st_mtime)
run=runs[-1] if runs else None
models=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1])) if run else []
report=dict(supervisor_pid=pid,running=alive,
    training_disabled_by_user=(directory/'cancelled_by_user').exists(),
    completed_iterations=int(iterations[-1][0])+1 if iterations else 0,
    requested_iterations=int(iterations[-1][1]) if iterations else None,
    run=str(run) if run else None,latest_checkpoint=str(models[-1]) if models else None,
    exit_code=int((directory/'train.exit').read_text()) if (directory/'train.exit').exists() else None,
    log=str(log))
print(json.dumps(report,indent=2))
