#!/usr/bin/env bash
set -euo pipefail

CSLR_DIR=/home/haojun/projects/SLRT/Online/CSLR
RESULTS=/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily_ISLR
CONFIG=configs/slide_csl-daily.yaml
CKPT_NAME=cslr_best.ckpt
MAX_SAMPLES=200
GPU_ID=${GPU_ID:-2}
RUN_LOG="${RESULTS}/adaptive_stride_benchmark_max${MAX_SAMPLES}.log"

cd "${CSLR_DIR}"

echo "[$(date -Is)] adaptive stride benchmark started" | tee "${RUN_LOG}"
echo "[$(date -Is)] gpu=${GPU_ID} config=${CONFIG} ckpt=${CKPT_NAME} max_samples=${MAX_SAMPLES}" | tee -a "${RUN_LOG}"

run_eval() {
  local name="$1"
  local adaptive="$2"
  local start_ts end_ts elapsed
  start_ts=$(date +%s)
  echo "[$(date -Is)] start ${name}" | tee -a "${RUN_LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n slrt_legacy python prediction_slide.py \
    --config "${CONFIG}" \
    --split dev \
    --ckpt_name "${CKPT_NAME}" \
    --eval_setting "${name}" \
    --save_subdir "prediction_slide_${name}" \
    --pred_src ensemble \
    --adaptive_stride "${adaptive}" \
    --max_samples "${MAX_SAMPLES}" \
    > "${RESULTS}/prediction_slide_${name}.stdout.log" 2>&1
  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  echo "[$(date -Is)] done ${name} elapsed_sec=${elapsed}" | tee -a "${RUN_LOG}"
}

run_eval "dev_fixed_stride1_max${MAX_SAMPLES}" 0
run_eval "dev_adaptive_stride_max${MAX_SAMPLES}" 1

conda run -n slrt_legacy python - <<'PY' | tee -a "/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily_ISLR/adaptive_stride_benchmark_max200.log"
import os
import pickle

base = "/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily_ISLR"
runs = {
    "fixed": "prediction_slide_dev_fixed_stride1_max200/dev",
    "adaptive": "prediction_slide_dev_adaptive_stride_max200/dev",
}

def load_eval(run_dir):
    return pickle.load(open(os.path.join(base, run_dir, "dev_evaluation_results.pkl"), "rb"))

def load_logits(run_dir):
    return pickle.load(open(os.path.join(base, run_dir, "dev_logits.pkl"), "rb"))

def best_wer(eval_results):
    rows = []
    for key, value in eval_results.items():
        if key.startswith("wer_"):
            rows.append((key, float(value["wer"]), float(value["del"]), float(value["ins"]), float(value["sub"])))
    rows.sort(key=lambda x: x[1])
    return rows[0], rows

def window_stats(logits):
    counts = [int(v.shape[0]) for v in logits.values()]
    return {
        "samples": len(counts),
        "total_windows": sum(counts),
        "avg_windows": sum(counts) / max(len(counts), 1),
        "min_windows": min(counts) if counts else 0,
        "max_windows": max(counts) if counts else 0,
    }

summaries = {}
for name, run_dir in runs.items():
    eval_results = load_eval(run_dir)
    logits = load_logits(run_dir)
    best, rows = best_wer(eval_results)
    stats = window_stats(logits)
    summaries[name] = {"best": best, "stats": stats}
    print(f"### {name}")
    print(f"best={best[0]} WER={best[1]:.2f} DEL={best[2]:.2f} INS={best[3]:.2f} SUB={best[4]:.2f}")
    print(
        f"windows total={stats['total_windows']} avg={stats['avg_windows']:.2f} "
        f"min={stats['min_windows']} max={stats['max_windows']} samples={stats['samples']}"
    )
    for row in rows:
        print(f"{row[0]} WER={row[1]:.2f} DEL={row[2]:.2f} INS={row[3]:.2f} SUB={row[4]:.2f}")

fixed_w = summaries["fixed"]["stats"]["total_windows"]
adaptive_w = summaries["adaptive"]["stats"]["total_windows"]
fixed_wer = summaries["fixed"]["best"][1]
adaptive_wer = summaries["adaptive"]["best"][1]
reduction = 100.0 * (1.0 - adaptive_w / fixed_w) if fixed_w else 0.0
wer_delta = adaptive_wer - fixed_wer
print("### comparison")
print(f"window_reduction_percent={reduction:.2f}")
print(f"wer_delta_adaptive_minus_fixed={wer_delta:.2f}")
PY

echo "[$(date -Is)] adaptive stride benchmark finished" | tee -a "${RUN_LOG}"
