# GPU and Training Command Cheatsheet

Repository: `/home/haojun/projects/SLRT`

## Check Available GPUs

Basic GPU status:

```bash
nvidia-smi
```

Compact live view:

```bash
watch -n 2 nvidia-smi
```

Show GPU index, name, memory, and utilization:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

Show GPU processes:

```bash
nvidia-smi pmon -c 1
```

Check training processes:

```bash
ps -ef | grep training.py | grep -v grep
```

Check top-800 CSLR/ISLR related processes:

```bash
ps -ef | grep -E 'training.py|prediction_slide.py|csl-daily-top-800' | grep -v grep
```

## Start Top-800 ISLR Training

Resume the current 32768-example top-800 ISLR run:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR

CUDA_VISIBLE_DEVICES=3,4,5,6 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
conda run -n slrt_legacy python -m torch.distributed.launch \
  --nproc_per_node 4 \
  --master_port 29999 \
  --use_env \
  training.py \
  --config=configs/csl-daily-top-800_ISLR_subset32768.yaml
```

Expected log signs:

```text
Sucessfully resume training from ...
Epoch 4, Training examples 32768
lr=0.00059...
```

Note: `csl-daily-top-800_ISLR_subset32768.yaml` currently has `data.transform_cfg.color_jitter: false` because resume training hit CUDA OOM inside `ColorJitter.adjust_hue` on 2026-05-26. If OOM still appears, reduce `training.batch_size` from `4` to `2`.

If starting fresh on a new result directory, make sure the config does not point at an old `model_dir`.

Start the full 186247-example top-800 ISLR run:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR

CUDA_VISIBLE_DEVICES=3,4,5,6 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
conda run -n slrt_legacy python -m torch.distributed.launch \
  --nproc_per_node 4 \
  --master_port 29999 \
  --use_env \
  training.py \
  --config=configs/csl-daily-top-800_ISLR_full_stable.yaml
```

Full-train logs:

```bash
tail -f /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/train.rank0.log
```

## View Training Logs

Rank 0 main log:

```bash
tail -f /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/train.rank0.log
```

All rank logs:

```bash
tail -f /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/train.rank*.log
```

Extract epoch and accuracy summary:

```bash
grep -E 'Epoch [0-9]+, Training|ensemble_last_Per-instance ACC Top-1|best_score|Current Best Epoch' \
  /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/train.rank0.log
```

Check for errors:

```bash
grep -E 'Traceback|RuntimeError|Error|Killed|OOM|FORWARD_FAIL|OPTIM_FAIL' \
  /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/train.rank*.log
```

## Stop Training

Preferred: press `Ctrl+C` once in the terminal that launched training, then wait 10-30 seconds.

If processes remain, list them:

```bash
ps -ef | grep training.py | grep csl-daily-top-800 | grep -v grep
```

Gracefully terminate by PID:

```bash
kill <PID>
```

If a process does not exit after a short wait:

```bash
kill -9 <PID>
```

Kill only the current top-800 training processes after reviewing the printed PIDs:

```bash
ps -ef | grep training.py | grep csl-daily-top-800 | grep -v grep
```

## Generate Slide Predictions After ISLR Training

Generate dev predictions from the subset32768 checkpoint:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR

CUDA_VISIBLE_DEVICES=3 \
conda run -n slrt_legacy python -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port 29997 \
  prediction_slide.py \
  --config=configs/slide_csl-daily-top-800_subset32768.yaml \
  --save_fea 0 \
  --split dev
```

Check non-empty dev predictions:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR

conda run -n slrt_legacy python -c "import pickle; r=pickle.load(open('results/csl-daily-top-800_ISLR_subset32768/prediction_slide/dev/dev_results.pkl','rb')); print(sum(1 for k in r if str(r[k].get('window_greedy_13_gls_hyp','')).strip()))"
```

Generate test predictions only after dev is non-empty:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR

CUDA_VISIBLE_DEVICES=3 \
conda run -n slrt_legacy python -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port 29997 \
  prediction_slide.py \
  --config=configs/slide_csl-daily-top-800_subset32768.yaml \
  --save_fea 0 \
  --split test
```

## Accuracy Experiment Notes

Detailed notes about possible post-baseline accuracy experiments are kept in:

```bash
/home/haojun/projects/SLRT/code_agent_logs/2026-05-26/summary.md
```

Search for:

```bash
grep -n "Possible Accuracy Improvements" /home/haojun/projects/SLRT/code_agent_logs/2026-05-26/summary.md
```
