#!/usr/bin/env bash
set -euo pipefail

CSLR_DIR=/home/haojun/projects/SLRT/Online/CSLR
CONFIG=configs/csl-daily-top-800_ISLR_full_lr1e-4.yaml
SLIDE_CONFIG=configs/slide_csl-daily-top-800_full_lr1e-4.yaml
RESULTS=/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_lr1e-4
CKPT_DIR="${RESULTS}/ckpts"
WATCH_LOG="${RESULTS}/lr1e4_epoch_eval_watcher.log"

mkdir -p "${RESULTS}/diagnostics"
echo "[$(date -Is)] watcher started" | tee -a "${WATCH_LOG}"

cd "${CSLR_DIR}"

for epoch in 00 01 02; do
  ckpt="${CKPT_DIR}/epoch_${epoch}.ckpt"
  echo "[$(date -Is)] waiting for ${ckpt}" | tee -a "${WATCH_LOG}"
  while [ ! -f "${ckpt}" ]; do
    sleep 300
  done

  echo "[$(date -Is)] found ${ckpt}; start ISLR diagnostic" | tee -a "${WATCH_LOG}"
  CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python tools/islr_dev_diagnostic.py \
    --config "${CONFIG}" \
    --ckpt "${ckpt}" \
    --output-dir "${RESULTS}/diagnostics/dev_epoch_${epoch}" \
    > "${RESULTS}/diagnostics/dev_epoch_${epoch}.stdout.log" 2>&1
  echo "[$(date -Is)] diagnostic done for epoch_${epoch}" | tee -a "${WATCH_LOG}"

  echo "[$(date -Is)] start prediction_slide keypoint dev for epoch_${epoch}" | tee -a "${WATCH_LOG}"
  CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
    --config "${SLIDE_CONFIG}" \
    --split dev \
    --ckpt_name "epoch_${epoch}.ckpt" \
    --eval_setting "dev_keypoint_epoch_${epoch}_lr1e4" \
    --save_subdir "prediction_slide_dev_keypoint_epoch_${epoch}" \
    --pred_src keypoint \
    > "${RESULTS}/prediction_slide_dev_keypoint_epoch_${epoch}.stdout.log" 2>&1
  echo "[$(date -Is)] prediction_slide done for epoch_${epoch}" | tee -a "${WATCH_LOG}"
done

echo "[$(date -Is)] watcher finished all epochs" | tee -a "${WATCH_LOG}"
