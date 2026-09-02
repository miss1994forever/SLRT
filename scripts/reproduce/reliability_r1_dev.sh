#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CSLR_DIR="$ROOT_DIR/Online/CSLR"
PYTHON_BIN="${PYTHON:-python}"
GPU_UUID="${CUDA_VISIBLE_DEVICES:-}"

case "$GPU_UUID" in
  GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a|GPU-06afe121-c4ce-b981-bb86-399e4a85ae83)
    echo "Refusing known faulty GPU UUID: $GPU_UUID" >&2
    exit 2
    ;;
  GPU-*) ;;
  *) echo "Set CUDA_VISIBLE_DEVICES to one healthy GPU UUID (GPU-...)." >&2; exit 2 ;;
esac

cd "$CSLR_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"$PYTHON_BIN" tools/reliability_dev_diagnostic.py \
  --config configs/csl-daily-top-800_ISLR_full_stable.yaml \
  --ckpt results/csl-daily-top-800_ISLR_full_stable/ckpts/best.ckpt \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --output-dir results/csl-daily-top-800_ISLR_full_stable/diagnostics/reliability_r1_dev
