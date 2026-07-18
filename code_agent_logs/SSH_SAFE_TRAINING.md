# SSH-Safe Training Commands

Repository: `/home/haojun/projects/SLRT`

This document records ways to start long SLRT training jobs so they keep running after the SSH connection closes.

## Recommended: tmux

Create a persistent terminal session:

```bash
tmux new -s slrt_full800
```

Inside tmux, start the full top-800 ISLR run:

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

Detach from tmux without stopping training:

```text
Ctrl+B, then D
```

Reconnect later:

```bash
tmux attach -t slrt_full800
```

List active tmux sessions:

```bash
tmux ls
```

Kill the session only when the training has already stopped or you intentionally want to close that terminal:

```bash
tmux kill-session -t slrt_full800
```

## Alternative: nohup

Use this when you do not need an interactive terminal after launch:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR

nohup bash -lc 'CUDA_VISIBLE_DEVICES=3,4,5,6 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 conda run -n slrt_legacy python -m torch.distributed.launch --nproc_per_node 4 --master_port 29999 --use_env training.py --config=configs/csl-daily-top-800_ISLR_full_stable.yaml' \
  > /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/nohup.full_stable.log 2>&1 &
```

The command prints a shell job id. The actual training logs are still written by SLRT under the result directory.

## Monitor Logs

Main rank log:

```bash
tail -f /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/train.rank0.log
```

nohup wrapper log:

```bash
tail -f /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/nohup.full_stable.log
```

Check epoch, score, and best checkpoint messages:

```bash
grep -E 'Epoch [0-9]+, Training|ensemble_last_Per-instance ACC Top-1|best_score|Current Best Epoch' \
  /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/train.rank0.log
```

Check errors:

```bash
grep -E 'Traceback|RuntimeError|Error|Killed|OOM|FORWARD_FAIL|OPTIM_FAIL' \
  /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/train.rank*.log
```

## Check and Stop Processes

Check active training processes:

```bash
ps -ef | grep -E 'training.py|csl-daily-top-800_ISLR_full_stable' | grep -v grep
```

Preferred stop if running inside tmux:

```text
Ctrl+C
```

If the launcher is detached and a stop is required, inspect PIDs first, then terminate only the matching top-800 full-stable training processes:

```bash
ps -ef | grep -E 'training.py|csl-daily-top-800_ISLR_full_stable' | grep -v grep
kill <PID>
```

Use `kill -9 <PID>` only if a process does not exit after a normal `kill`.

## Notes

- A plain foreground SSH command will usually stop when the SSH connection is lost.
- `tmux` is preferred because it preserves the live terminal and makes `Ctrl+C` safe and simple.
- `nohup` is acceptable for fire-and-forget jobs, but stopping and debugging are less convenient.
- The current full-stable config uses GPUs `3,4,5,6`; it does not use physical GPU `1`.
- The full-stable config keeps `color_jitter: false` because the subset resume hit OOM in `ColorJitter.adjust_hue`.
