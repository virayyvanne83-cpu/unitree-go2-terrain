#!/usr/bin/env bash
# Usage: source /home/clt/project_rl/unitree_rl_gym/activate.sh
_unitree_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/clt/legged_gym/.tools/miniforge3/etc/profile.d/conda.sh
conda activate "$_unitree_root/.venv"
unset PYTHONHOME
# Runtime variables are managed by .venv/conda-meta/state and restored on deactivate.
unset _unitree_root
