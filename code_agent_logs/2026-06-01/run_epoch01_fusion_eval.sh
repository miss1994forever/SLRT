#!/usr/bin/env bash
set -euo pipefail

CSLR_DIR=/home/haojun/projects/SLRT/Online/CSLR
RESULTS=/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_lr1e-4
CONFIG=configs/csl-daily-top-800_ISLR_full_lr1e-4.yaml
SLIDE_CONFIG=configs/slide_csl-daily-top-800_full_lr1e-4.yaml
CKPT_NAME=epoch_01.ckpt
CKPT_PATH="${RESULTS}/ckpts/${CKPT_NAME}"
RUN_LOG="${RESULTS}/epoch01_fusion_eval.log"
GRID_DIR="${RESULTS}/diagnostics/dev_epoch_01_weight_grid"

cd "${CSLR_DIR}"
echo "[$(date -Is)] epoch01 fusion eval started" | tee -a "${RUN_LOG}"

if [[ -f "${GRID_DIR}/dev_weight_grid.csv" ]]; then
  echo "[$(date -Is)] isolated weighted grid already exists, skip rerun" | tee -a "${RUN_LOG}"
else
  echo "[$(date -Is)] start isolated weighted grid" | tee -a "${RUN_LOG}"
  CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python tools/islr_dev_diagnostic.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT_PATH}" \
    --weight-grid-search \
    --output-dir "${GRID_DIR}" \
    > "${RESULTS}/diagnostics/dev_epoch_01_weight_grid.stdout.log" 2>&1
  echo "[$(date -Is)] isolated weighted grid done" | tee -a "${RUN_LOG}"
fi

read -r RGB_W KEYPOINT_W FUSE_W < <(awk -F, 'NR==2 {print $3, $4, $5}' "${GRID_DIR}/dev_weight_grid.csv")
echo "[$(date -Is)] best grid weights rgb=${RGB_W} keypoint=${KEYPOINT_W} fuse=${FUSE_W}" | tee -a "${RUN_LOG}"

echo "[$(date -Is)] start prediction_slide fuse" | tee -a "${RUN_LOG}"
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config "${SLIDE_CONFIG}" \
  --split dev \
  --ckpt_name "${CKPT_NAME}" \
  --eval_setting dev_fuse_epoch_01_lr1e4 \
  --save_subdir prediction_slide_dev_fuse_epoch_01 \
  --pred_src fuse \
  > "${RESULTS}/prediction_slide_dev_fuse_epoch_01.stdout.log" 2>&1
echo "[$(date -Is)] prediction_slide fuse done" | tee -a "${RUN_LOG}"

echo "[$(date -Is)] start prediction_slide ensemble" | tee -a "${RUN_LOG}"
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config "${SLIDE_CONFIG}" \
  --split dev \
  --ckpt_name "${CKPT_NAME}" \
  --eval_setting dev_ensemble_epoch_01_lr1e4 \
  --save_subdir prediction_slide_dev_ensemble_epoch_01 \
  --pred_src ensemble \
  > "${RESULTS}/prediction_slide_dev_ensemble_epoch_01.stdout.log" 2>&1
echo "[$(date -Is)] prediction_slide ensemble done" | tee -a "${RUN_LOG}"

echo "[$(date -Is)] start prediction_slide weighted" | tee -a "${RUN_LOG}"
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config "${SLIDE_CONFIG}" \
  --split dev \
  --ckpt_name "${CKPT_NAME}" \
  --eval_setting dev_weighted_epoch_01_lr1e4 \
  --save_subdir prediction_slide_dev_weighted_epoch_01 \
  --pred_src weighted \
  --rgb_weight "${RGB_W}" \
  --keypoint_weight "${KEYPOINT_W}" \
  --fuse_weight "${FUSE_W}" \
  > "${RESULTS}/prediction_slide_dev_weighted_epoch_01.stdout.log" 2>&1
echo "[$(date -Is)] prediction_slide weighted done" | tee -a "${RUN_LOG}"

echo "[$(date -Is)] epoch01 fusion eval finished" | tee -a "${RUN_LOG}"
