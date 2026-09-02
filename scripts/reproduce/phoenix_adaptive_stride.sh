#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CSLR_DIR="$ROOT_DIR/Online/CSLR"
MODE="${1:-adaptive}"
SPLIT="${2:-dev}"
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

if [[ "$SPLIT" == "test" && "${ALLOW_TEST:-0}" != "1" ]]; then
  echo "Test is frozen. Set ALLOW_TEST=1 only for the single final evaluation." >&2
  exit 2
fi
if [[ "$SPLIT" != "dev" && "$SPLIT" != "test" ]]; then
  echo "Split must be dev or test." >&2
  exit 2
fi

cd "$CSLR_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

COMMON=(
  prediction_slide.py
  --config configs/slide_phoenix-2014t.yaml
  --split "$SPLIT"
  --split_size 16
  --pred_src ensemble
  --save_fea 0
)

case "$MODE" in
  fixed)
    "$PYTHON_BIN" "${COMMON[@]}" \
      --save_subdir "reproduction_fixed_stride1_${SPLIT}" \
      --adaptive_stride 0 \
      --span_weighted_voting 0
    ;;
  adaptive)
    "$PYTHON_BIN" "${COMMON[@]}" \
      --save_subdir "reproduction_adaptive_span15_${SPLIT}" \
      --adaptive_stride 1 \
      --adaptive_min_stride 1 \
      --adaptive_max_stride 3 \
      --adaptive_ema_decay 0.4 \
      --adaptive_confidence_threshold 0.2 \
      --adaptive_quantile_low 0.2 \
      --adaptive_quantile_high 0.7 \
      --span_weighted_voting 1 \
      --vote_span_frames 15 \
      --span_min_weight 0.05
    ;;
  *) echo "Mode must be fixed or adaptive." >&2; exit 2 ;;
esac
