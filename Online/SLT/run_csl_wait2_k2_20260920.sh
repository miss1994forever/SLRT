#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/mnt/workspace/conda_envs/haojun/envs/slrt_legacy/bin/python"
CONFIG="configs/g2t_wait2_csl_retrain_k2_20260920.yaml"
GPU_LIST="GPU-91f9e38c-0a59-15c0-bd08-6a4f89da07ed,GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631,GPU-4f5bee72-bad2-f8c9-2656-58e4d94c3c98,GPU-1c25de34-7fce-3f81-158c-c4d82d3ac729,GPU-8c92fc9f-aaa8-cd49-7196-a08cca1c6c7b"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export OMP_NUM_THREADS=1
export enable_pbar=0
export WANDB_MODE=disabled

exec "$PYTHON_BIN" -m torch.distributed.launch \
  --nproc_per_node 5 \
  --master_port 29620 \
  --use_env \
  training.py \
  --config "$CONFIG"
