#!/usr/bin/env bash
# One driver controls the colocated native actor/rollout worker group.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
: "${RLLAB_RUN_TAG:?RLLAB_RUN_TAG is required}"
: "${RLLAB_MATH_PLAN:?RLLAB_MATH_PLAN is required}"
: "${RLLAB_CREDIT:?RLLAB_CREDIT must be orm or prm}"
[[ "$RLLAB_RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$RLLAB_CREDIT" == orm || "$RLLAB_CREDIT" == prm ]] || exit 2
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n $(git ls-files --others --exclude-standard) ]]; then
  echo 'Training requires a clean frozen checkout' >&2
  exit 2
fi
execution_context=${RLLAB_EXECUTION_CONTEXT:-task}
[[ "$execution_context" == task || "$execution_context" == attached ]] || exit 2
# A multi-node Ray cluster is managed by the allocation; submit this driver once.
# Single-node execution creates and closes its own Ray runtime.
nnodes=${NNODES:-1}
[[ ${NODE_RANK:-0} == 0 ]] || { echo 'Run the math driver only on Ray head/rank 0' >&2; exit 2; }
if [[ "$nnodes" != 1 ]]; then
  : "${RAY_ADDRESS:?Multi-node math requires the allocated Ray cluster address}"
fi
nproc=${RLLAB_NPROC_PER_NODE:-8}
math_storage=${RLLAB_STORAGE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/rllab}
run_root="$math_storage/task-logs/$RLLAB_RUN_TAG"
mkdir -p "$run_root/launches"
exec 9>"$run_root/.${RLLAB_CREDIT}.lock"
flock -n 9 || { echo 'This credit arm already has a driver' >&2; exit 2; }
launch_id="${RLLAB_CREDIT}-$(date -u +%Y%m%dT%H%M%S)-$$"
exec > >(tee "$run_root/launches/$launch_id.log") 2>&1
export PYTHONUNBUFFERED=1
export RLLAB_EXECUTION_CONTEXT="$execution_context"
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-$([[ "$nnodes" == 1 ]] && echo 1 || echo 0)}
args=(--plan "$RLLAB_MATH_PLAN" --credit "$RLLAB_CREDIT" --run-dir "$run_root" --devices "$nproc" --nodes "$nnodes")
[[ ${RLLAB_RESUME:-0} == 1 ]] && args+=(--resume)
[[ -n ${RLLAB_STOP_AFTER:-} ]] && args+=(--stop-after "$RLLAB_STOP_AFTER")
[[ -n ${RLLAB_TIME_LIMIT:-} ]] && args+=(--time-limit "$RLLAB_TIME_LIMIT")
[[ -n ${RLLAB_RESUME_MAX_WALL_SECONDS:-} ]] && args+=(--resume-max-wall-seconds "$RLLAB_RESUME_MAX_WALL_SECONDS")
[[ -n ${RLLAB_ROLLOUT_MAX_NUM_SEQS:-} ]] && args+=(--rollout-max-num-seqs "$RLLAB_ROLLOUT_MAX_NUM_SEQS")
[[ -n ${RLLAB_ROLLOUT_GPU_MEMORY_UTILIZATION:-} ]] && args+=(--rollout-gpu-memory-utilization "$RLLAB_ROLLOUT_GPU_MEMORY_UTILIZATION")
utilization="$run_root/launches/$launch_id-ppu.csv"
echo "context=$execution_context commit=$(git rev-parse HEAD) credit=$RLLAB_CREDIT nodes=$nnodes devices=$nproc plan=$RLLAB_MATH_PLAN resume=${RLLAB_RESUME:-0} stop_after=${RLLAB_STOP_AFTER:-end}"
bash scripts/monitor_ppu.sh "$utilization" &
math_monitor_pid=$!
cleanup() {
  kill "$math_monitor_pid" 2>/dev/null || true
  wait "$math_monitor_pid" 2>/dev/null || true
  python scripts/summarize_ppu.py "$utilization" > "$run_root/launches/$launch_id-utilization.json" || true
}
trap cleanup EXIT
bash scripts/real_rl_python.sh -m rllab.math_train "${args[@]}"
