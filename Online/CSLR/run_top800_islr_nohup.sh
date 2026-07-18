#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/haojun/projects/SLRT/Online/CSLR"
LOG_DIR="$ROOT_DIR/run_logs"
LOG_FILE="$LOG_DIR/top800_islr.log"
PID_FILE="$LOG_DIR/top800_islr.pid"
MASTER_PORT="${MASTER_PORT:-29999}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate slrt_legacy

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

VISIBLE_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [[ -z "$VISIBLE_GPU_COUNT" || "$VISIBLE_GPU_COUNT" -lt 1 ]]; then
  echo "No visible CUDA devices found" >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$VISIBLE_GPU_COUNT}"
if [[ "$NPROC_PER_NODE" -gt "$VISIBLE_GPU_COUNT" ]]; then
  echo "Requested NPROC_PER_NODE=$NPROC_PER_NODE but only $VISIBLE_GPU_COUNT CUDA devices are visible" >&2
  exit 1
fi

nohup python -u -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  training.py \
  --config=/home/haojun/projects/SLRT/Online/CSLR/configs/slide_csl-daily-top-800.yaml \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
printf 'PID=%s\nLOG=%s\nVISIBLE_GPU_COUNT=%s\nNPROC_PER_NODE=%s\nMASTER_PORT=%s\n' "$(cat "$PID_FILE")" "$LOG_FILE" "$VISIBLE_GPU_COUNT" "$NPROC_PER_NODE" "$MASTER_PORT"