# Code Agent Summary - 2026-06-01

## Started Experiment: top-800 ISLR low-LR fine-tuning

Goal:

- Fine-tune from the previous `full_stable` best checkpoint.
- Use lower LR to test whether epoch 3 degradation was caused by too aggressive continued training.
- After each epoch, run ISLR dev diagnostic and continuous `prediction_slide dev` with `pred_src=keypoint`.

## Configs Added

Training config:

- `Online/CSLR/configs/csl-daily-top-800_ISLR_full_lr1e-4.yaml`

Important settings:

```yaml
training:
  model_dir: /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_lr1e-4
  load_ckpt: /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/ckpts/best.ckpt
  from_ckpt: false
  total_epoch: 3
  batch_size: 4
  optimization:
    learning_rate:
      default: 1.0e-4
    scheduler: cosineannealing
    t_max: 3
```

Slide config:

- `Online/CSLR/configs/slide_csl-daily-top-800_full_lr1e-4.yaml`

This points continuous `prediction_slide.py` evaluation to:

```text
/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_lr1e-4
```

## Training Session

Started in tmux:

```bash
tmux new-session -d -s slrt_lr1e4 \
  "cd /home/haojun/projects/SLRT/Online/CSLR && \
   CUDA_VISIBLE_DEVICES=3,4,5,6 \
   conda run -n slrt_legacy python -m torch.distributed.launch \
     --nproc_per_node 4 \
     --master_port 29998 \
     --use_env \
     training.py \
     --config=configs/csl-daily-top-800_ISLR_full_lr1e-4.yaml"
```

Initial log confirmed:

```text
Load ckpt /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/ckpts/best.ckpt
learning rate recognition_network=0.0001
Epoch 0, Training examples 186247
```

The first logged optimizer LR was `7.5e-05`, which is expected from the scheduler after initialization.

## Per-Epoch Evaluation Watcher

Watcher script:

- `code_agent_logs/2026-06-01/lr1e4_epoch_eval_watcher.sh`

Started in tmux:

```bash
tmux new-session -d -s slrt_lr1e4_eval \
  /home/haojun/projects/SLRT/code_agent_logs/2026-06-01/lr1e4_epoch_eval_watcher.sh
```

It waits for:

```text
epoch_00.ckpt
epoch_01.ckpt
epoch_02.ckpt
```

For each epoch, it runs:

1. ISLR dev diagnostic:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python tools/islr_dev_diagnostic.py \
  --config configs/csl-daily-top-800_ISLR_full_lr1e-4.yaml \
  --ckpt .../ckpts/epoch_XX.ckpt \
  --output-dir .../diagnostics/dev_epoch_XX
```

2. Continuous dev slide evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slrt_legacy python prediction_slide.py \
  --config configs/slide_csl-daily-top-800_full_lr1e-4.yaml \
  --split dev \
  --ckpt_name epoch_XX.ckpt \
  --eval_setting dev_keypoint_epoch_XX_lr1e4 \
  --save_subdir prediction_slide_dev_keypoint_epoch_XX \
  --pred_src keypoint
```

## Monitoring Commands

Check tmux sessions:

```bash
tmux ls
```

Attach to training:

```bash
tmux attach -t slrt_lr1e4
```

Attach to watcher:

```bash
tmux attach -t slrt_lr1e4_eval
```

Tail training log:

```bash
tail -f /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_lr1e-4/train.rank0.log
```

Tail watcher log:

```bash
tail -f /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_lr1e-4/lr1e4_epoch_eval_watcher.log
```

Check GPU:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

## Decision Criteria

Compare each epoch against the previous baseline:

- Isolated keypoint nonblank Top-1 baseline: `48.32`
- Continuous dev keypoint-only WER baseline: `48.29`
- Continuous dev weighted 0/0.9/0.1 WER baseline: `48.18`

If epoch 0/1/2 improves keypoint nonblank Top-1 and continuous WER:

- Continue with `lr=5e-5` or class-balanced/nonblank-focused experiment.

If WER remains around `48`:

- Stop stacking epochs.
- Move to signness/boundary/duration-aware ISLR.

## Final Low-LR Results

The 3-epoch low-LR run finished, and the epoch watcher completed all diagnostics and continuous dev evaluations.

### Training/Evaluation Summary

| Epoch | Ensemble Instance Top-1 | Ensemble Class Top-1 | Keypoint Instance Top-1 | Keypoint Class Top-1 | Best Score |
|---|---:|---:|---:|---:|---:|
| 0 | 75.44 | 59.23 | 73.88 | 56.67 | 75.44 |
| 1 | 75.91 | 60.44 | 74.32 | 58.49 | 75.91 |
| 2 | 75.35 | 60.91 | 73.96 | 58.91 | 75.91 |

### Diagnostic Nonblank Summary

| Epoch | Stream | Nonblank Top-1 | Signness Acc | Gloss -> Blank | Pred Blank |
|---|---|---:|---:|---:|---:|
| 0 | keypoint | 66.15 | 83.67 | 16.05 | 46.33 |
| 0 | ensemble | 68.29 | 85.03 | 14.27 | 45.74 |
| 1 | keypoint | 66.12 | 83.43 | 17.32 | 47.49 |
| 1 | ensemble | 67.86 | 84.69 | 16.18 | 47.50 |
| 2 | keypoint | 65.66 | 83.12 | 17.69 | 47.59 |
| 2 | ensemble | 67.71 | 84.30 | 16.02 | 46.94 |

### Continuous Dev Keypoint WER

| Epoch | Best Decode | Best WER | DEL | INS | SUB |
|---|---|---:|---:|---:|---:|
| 0 | window_greedy_9 | 33.83 | 16.62 | 4.31 | 12.90 |
| 1 | window_greedy_7 | 33.40 | 13.21 | 6.15 | 14.03 |
| 2 | window_greedy_7 | 34.03 | 13.76 | 6.35 | 13.92 |

Conclusion:

- Epoch 1 is the best checkpoint for continuous keypoint-only dev.
- Epoch 2 improves some per-class isolated metrics, but continuous WER gets worse.
- Do not continue stacking more epochs with this schedule.
- Next best action is to evaluate epoch 1 with ensemble/weighted fusion, because ensemble/fuse are now much stronger than in the old full-stable checkpoint.

Recommended checkpoint:

```text
/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_lr1e-4/ckpts/epoch_01.ckpt
```
