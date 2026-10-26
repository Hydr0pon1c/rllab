#!/usr/bin/env bash
# Use an activated compatible RL environment, or an explicitly selected venv.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
rl_python=${RLLAB_PYTHON:-python}
if [[ -n ${RLLAB_RL_ENV_ROOT:-} ]]; then
  rl_python="$RLLAB_RL_ENV_ROOT/venv/bin/python"
  export PATH="$RLLAB_RL_ENV_ROOT/venv/bin:$PATH"
  export TRITON_CACHE_DIR="$RLLAB_RL_ENV_ROOT/cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$RLLAB_RL_ENV_ROOT/cache/torchinductor"
fi
cd "$repo_root"
exec "$rl_python" "$@"
