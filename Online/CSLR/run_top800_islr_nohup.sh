#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/haojun/projects/SLRT/Online/CSLR"
LOG_DIR="$ROOT_DIR/run_logs"
LOG_FILE="$LOG_DIR/top800_islr.log"
PID_FILE="$LOG_DIR/top800_islr.pid"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate slrt_legacy

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

nohup python -u -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port 29999 \
  training.py \
  --config=/home/haojun/projects/SLRT/Online/CSLR/configs/slide_csl-daily-top-800.yaml \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
printf 'PID=%s\nLOG=%s\n' "$(cat "$PID_FILE")" "$LOG_FILE"