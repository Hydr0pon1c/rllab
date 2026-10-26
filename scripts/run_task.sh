#!/usr/bin/env bash
# Single Task entry for formal runs; the immutable plan owns scientific controls.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
case "${RLLAB_TASK_MODE:?Set RLLAB_TASK_MODE=math_train or puzzlemdp}" in
  math_train) exec bash scripts/run_math_task.sh ;;
  puzzlemdp)
    : "${RLLAB_PUZZLE_CONFIG:?Set the experiment YAML path}"
    if ! git diff --quiet || ! git diff --cached --quiet || [[ -n $(git ls-files --others --exclude-standard) ]]; then
      echo 'Training requires a clean frozen checkout' >&2
      exit 2
    fi
    export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
    nproc=${RLLAB_NPROC_PER_NODE:-8}
    puzzle_storage=${RLLAB_STORAGE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/rllab}
    puzzle_tag=${RLLAB_RUN_TAG:?Set a unique RLLAB_RUN_TAG}
    [[ "$puzzle_tag" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
    puzzle_log="$puzzle_storage/task-logs/$puzzle_tag"
    mkdir -p "$(dirname "$puzzle_log")"
    mkdir "$puzzle_log"
    exec > >(tee "$puzzle_log/task.log") 2>&1
    bash scripts/monitor_ppu.sh "$puzzle_log/devices.csv" &
    monitor_pid=$!
    cleanup() {
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
      "${RLLAB_PYTHON:-python}" scripts/summarize_ppu.py "$puzzle_log/devices.csv" > "$puzzle_log/utilization.json" || true
    }
    trap cleanup EXIT
    "${RLLAB_PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
      -m rllab.experiment.cli train --config "$RLLAB_PUZZLE_CONFIG" "$@"
    ;;
  *) echo 'Supported Task modes: puzzlemdp, math_train' >&2; exit 2 ;;
esac
