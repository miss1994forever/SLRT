# CSL-Daily top-800 training guide

This 800-word setup is a three-stage pipeline, not a single model:

1. Stage 1 trains an Online ISLR model on the top-800 closed vocabulary.
2. Stage 2 trains a sentence-level S2G model, which is the CSLR step in this repo.
3. Stage 3 trains a fused S2G model that uses both raw video streams and the Stage 1 online features.

In other words, if you ask "are we training ISLR or CSLR", the precise answer is: both.

- Stage 1 is ISLR: classify frames/clips into one of the 800 target glosses.
- Stage 2 is CSLR/S2G: recognize gloss sequences at sentence level on the 800-word subset.
- Stage 3 is a stronger fused CSLR/S2G model built on top of Stage 1 and Stage 2 outputs.

Configs used by this guide:

- `/home/haojun/projects/SLRT/Online/CSLR/configs/slide_csl-daily-top-800.yaml`
- `/home/haojun/projects/SLRT/Online/CTC_fusion/configs/csl-daily-top-800_s2g.yaml`
- `/home/haojun/projects/SLRT/Online/CTC_fusion/configs/csl-daily-top-800_fuse_online.yaml`

Required subset directory:

- `/home/haojun/projects/SLRT/data/csl-daily-top-800-all/`

Required training assets:

- `/home/haojun/projects/SLRT/data/csl-daily/csl-daily-frames-512x512.tar.gz`
- `/home/haojun/projects/SLRT/data/csl-daily/keypoints_hrnet_dark_coco_wholebody.pkl`
- `/home/haojun/projects/SLRT/data/csl-daily/keypoints_hrnet_dark_coco_wholebody_iso.pkl`

The loaders in this workspace now read frames directly from the CSL-Daily `tar.gz` archive. You do not need to repack a zip.

## Step 0. Verify assets and subset

Run this first. It checks the 800-word subset and the three training assets.

```bash
bash /home/haojun/projects/SLRT/Online/prepare_csl_daily_top_800.sh
```

If you need to rebuild the subset files:

```bash
bash /home/haojun/projects/SLRT/Online/prepare_csl_daily_top_800.sh --force
```

Expected subset summary:

- train: 9185 samples
- dev: 360 samples
- test: 400 samples

## Step 1. Activate the environment

Use the same environment for all training stages.

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate slrt_legacy
export OMP_NUM_THREADS=1
```

## Step 2. Train Stage 1 Online ISLR

This stage trains the word-level online recognizer on the 800-word closed vocabulary.

Output directory:

- `/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR`

Command:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR
python -m torch.distributed.run --nproc_per_node 1 --master_port 29999 training.py \
  --config=/home/haojun/projects/SLRT/Online/CSLR/configs/slide_csl-daily-top-800.yaml
```

What to check after Stage 1:

- logs appear under the ISLR result directory
- checkpoints are written under `ckpts/`
- best checkpoint is produced before moving on

## Step 3. Export Stage 1 online features

This step is required for the fused Stage 3 model.

Expected feature outputs:

- `/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR/prediction_slide/train/train_features.pkl`
- `/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR/prediction_slide/dev/dev_features.pkl`
- `/home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR/prediction_slide/test/test_features.pkl`

Command:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR
python -m torch.distributed.run --nproc_per_node 1 --master_port 29997 prediction_slide.py \
  --config=/home/haojun/projects/SLRT/Online/CSLR/configs/slide_csl-daily-top-800.yaml \
  --save_fea 1
```

## Step 4. Train Stage 2 S2G / CSLR

This is the sentence-level gloss recognition stage.

Output directory:

- `/home/haojun/projects/SLRT/Online/CTC_fusion/results/csl-daily-top-800_s2g`

Command:

```bash
cd /home/haojun/projects/SLRT/Online/CTC_fusion
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29999 --use_env training.py \
  --config=/home/haojun/projects/SLRT/Online/CTC_fusion/configs/csl-daily-top-800_s2g.yaml
```

What to check after Stage 2:

- checkpoints are written under the S2G result directory
- `best.ckpt` exists before moving to Stage 3

## Step 5. Train Stage 3 fused online-boosted S2G

This stage combines:

- Stage 1 online features
- Stage 2 S2G checkpoint
- raw RGB and keypoint streams

Output directory:

- `/home/haojun/projects/SLRT/Online/CTC_fusion/results/csl-daily-top-800_fuse_online`

Command:

```bash
cd /home/haojun/projects/SLRT/Online/CTC_fusion
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29999 --use_env training.py \
  --config=/home/haojun/projects/SLRT/Online/CTC_fusion/configs/csl-daily-top-800_fuse_online.yaml
```

## Step 6. Recommended execution order

Run the stages in this exact order:

1. `prepare_csl_daily_top_800.sh`
2. Stage 1 ISLR training
3. Stage 1 feature export
4. Stage 2 S2G / CSLR training
5. Stage 3 fused S2G training

## Step 7. Recommended background execution

For long training jobs, use `tmux` so the process survives SSH disconnects.

Example for Stage 1:

```bash
tmux new -s top800_islr
cd /home/haojun/projects/SLRT/Online/CSLR
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate slrt_legacy
export OMP_NUM_THREADS=1
python -m torch.distributed.run --nproc_per_node 1 --master_port 29999 training.py \
  --config=/home/haojun/projects/SLRT/Online/CSLR/configs/slide_csl-daily-top-800.yaml
```

Detach without stopping the job:

```bash
Ctrl+b then d
```

Reattach later:

```bash
tmux attach -t top800_islr
```

## Step 8. What model are we training right now

If you are about to start from scratch now, the next model to train is Stage 1 Online ISLR.

That means:

- the immediate next command is the ISLR command in Step 2
- after that you export online features
- then you train the sentence-level CSLR/S2G models