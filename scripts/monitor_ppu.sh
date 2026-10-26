#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: monitor_ppu.sh OUTPUT.csv" >&2
  exit 2
fi

output=$1
interval=${RLLAB_PPU_MONITOR_INTERVAL:-30}
if [[ ! "$interval" =~ ^[1-9][0-9]*$ ]]; then
  echo "RLLAB_PPU_MONITOR_INTERVAL must be a positive integer" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"
printf 'timestamp_utc,index,name,utilization_percent,memory_used_mib,memory_total_mib\n' >"$output"

while true; do
  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  while IFS= read -r row; do
    printf '%s,%s\n' "$timestamp" "$row" >>"$output"
  done < <(
    nvidia-smi \
      --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
  )
  sleep "$interval"
done
