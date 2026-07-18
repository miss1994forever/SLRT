# Code Agent Summary - 2026-05-31

## Implemented

Added inference-side source selection and weighted fusion for the top-800 ISLR checkpoint.

Changed files:

- `Online/CSLR/tools/islr_dev_diagnostic.py`
- `Online/CSLR/prediction_slide.py`
- `Online/CSLR/configs/slide_csl-daily-top-800_full_stable.yaml`

New diagnostic options:

- `--weighted-fusion`
- `--rgb-weight`
- `--keypoint-weight`
- `--fuse-weight`
- `--weight-grid-search`

New `prediction_slide.py` options:

- `--pred_src {ensemble,rgb,keypoint,fuse,weighted}`
- `--rgb_weight`
- `--keypoint_weight`
- `--fuse_weight`
- `--max_samples`

Weighted fusion formula:

```python
weighted_prob = rgb_w * rgb_prob + keypoint_w * keypoint_prob + fuse_w * fuse_prob
weighted_logits = log(weighted_prob.clamp_min(1e-12))
```

Default behavior remains unchanged: `prediction_slide.py` still uses `pred_src=ensemble` unless explicitly overridden.

## Smoke Tests

Passed syntax/argument checks:

```bash
python3 - <<'PY'
import ast
for path in [
    '/home/haojun/projects/SLRT/Online/CSLR/tools/islr_dev_diagnostic.py',
    '/home/haojun/projects/SLRT/Online/CSLR/prediction_slide.py',
]:
    ast.parse(open(path, encoding='utf-8').read(), filename=path)
    print('AST OK', path)
PY
```

Passed `prediction_slide.py --help`; new args are visible.

Passed 1-sample continuous smoke tests:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config configs/slide_csl-daily-top-800_full_stable.yaml \
  --split dev \
  --ckpt_name best.ckpt \
  --eval_setting smoke_weighted \
  --save_subdir prediction_slide_smoke_weighted \
  --pred_src weighted \
  --rgb_weight 0.10 \
  --keypoint_weight 0.70 \
  --fuse_weight 0.20 \
  --max_samples 1
```

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config configs/slide_csl-daily-top-800_full_stable.yaml \
  --split dev \
  --ckpt_name best.ckpt \
  --eval_setting smoke_keypoint \
  --save_subdir prediction_slide_smoke_keypoint \
  --pred_src keypoint \
  --max_samples 1
```

## Isolated Dev Weight Grid

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python tools/islr_dev_diagnostic.py \
  --weight-grid-search \
  --output-dir /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best_weight_grid
```

Outputs:

- `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best_weight_grid/dev_diagnostic_summary.json`
- `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best_weight_grid/dev_diagnostic_streams.csv`
- `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best_weight_grid/dev_weight_grid.csv`

Best weighted candidate on isolated dev:

```text
rgb=0.0, keypoint=0.9, fuse=0.1
nonblank_top1_acc=47.55
signness_acc=77.88
top1_acc=66.32
gloss_to_blank_rate=31.43
```

However, keypoint-only is still stronger on isolated dev:

```text
keypoint-only nonblank_top1_acc=48.32
keypoint-only signness_acc=78.27
keypoint-only top1_acc=66.28
keypoint-only gloss_to_blank_rate=29.87
```

## Continuous Dev Slide Evaluation

All runs used:

- Config: `Online/CSLR/configs/slide_csl-daily-top-800_full_stable.yaml`
- Checkpoint: `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/ckpts/best.ckpt`
- Split: `dev`

### Keypoint-Only

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config configs/slide_csl-daily-top-800_full_stable.yaml \
  --split dev \
  --ckpt_name best.ckpt \
  --eval_setting dev_keypoint_fullstable \
  --save_subdir prediction_slide_dev_keypoint \
  --pred_src keypoint
```

Best WER:

```text
window_greedy_5 WER=48.29 DEL=17.52 INS=6.66 SUB=24.11
```

### Equal Ensemble

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config configs/slide_csl-daily-top-800_full_stable.yaml \
  --split dev \
  --ckpt_name best.ckpt \
  --eval_setting dev_ensemble_fullstable \
  --save_subdir prediction_slide_dev_ensemble \
  --pred_src ensemble
```

Best WER:

```text
window_greedy_3 WER=52.84 DEL=29.20 INS=4.00 SUB=19.64
```

### Weighted Fusion

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config configs/slide_csl-daily-top-800_full_stable.yaml \
  --split dev \
  --ckpt_name best.ckpt \
  --eval_setting dev_weighted_0_0_9_0_1_fullstable \
  --save_subdir prediction_slide_dev_weighted_0_0_9_0_1 \
  --pred_src weighted \
  --rgb_weight 0.0 \
  --keypoint_weight 0.9 \
  --fuse_weight 0.1
```

Best WER:

```text
window_greedy_5 WER=48.18 DEL=18.93 INS=5.96 SUB=23.28
```

## Interpretation

Weighted fusion slightly improves continuous dev WER over keypoint-only:

```text
weighted best WER: 48.18
keypoint best WER: 48.29
absolute gain: 0.11
```

This gain is very small. It is not enough to claim a robust fusion improvement yet.

Equal ensemble is clearly worse:

```text
equal ensemble best WER: 52.84
```

This confirms the earlier isolated diagnostic: the RGB and fused streams still drag down the strongest keypoint stream, mostly through blank-heavy behavior.

Current practical default for top-800 online CSLR should be one of:

1. `pred_src=keypoint`
2. `pred_src=weighted --rgb_weight 0.0 --keypoint_weight 0.9 --fuse_weight 0.1`

Because the weighted improvement is tiny, use keypoint-only as the safer baseline and weighted as the candidate to compare further on test.

## Next Steps

1. Run test split for keypoint-only and weighted if dev conclusion is accepted.
2. Add a small result-summary script for `prediction_slide` outputs so WER/non-empty/average-length/sample quality can be compared in one command.
3. Do not spend much more time hand-tuning static fusion weights.
4. Move toward signness/boundary-aware ISLR if continuous WER remains around 48.
