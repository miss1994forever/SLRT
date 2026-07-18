# Code Agent Daily Log

Date: 2026-05-27
Repository: `/home/haojun/projects/SLRT`

## Goal
- Validate whether the current top-800 subset checkpoint can feed the G2T pipeline.
- Prepare a stronger full-data top-800 ISLR training path.
- Record SSH-safe launch commands for long training jobs.

## Changes Made
- Generated top-800 slide dev predictions from:
  - `Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/ckpts/best.ckpt`
- Added and validated a G2T smoke-test path:
  - `Online/SLT/data_smoke/top800_g2t_from_subset32768/`
  - `Online/SLT/configs/g2t_wait2_csl_top800_smoke.yaml`
  - `Online/SLT/results/g2t_wait2_csl_top800_smoke_debug/`
- Added full-data stable ISLR config:
  - `Online/CSLR/configs/csl-daily-top-800_ISLR_full_stable.yaml`
- Added SSH-safe training guide:
  - `code_agent_logs/SSH_SAFE_TRAINING.md`

## Current State
- The subset32768 ISLR checkpoint is usable for format/process smoke testing, but its accuracy is weak:
  - best isolated dev Top-1: `20.47`
  - best checkpoint: `Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/ckpts/best.ckpt`
- Dev slide prediction file:
  - `Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/prediction_slide/dev/dev_results.pkl`
- Best checked dev decode was `window_greedy_13`:
  - WER: `87.22`
  - non-empty rate: `360/360`
  - average predicted gloss length: `7.57`
- The G2T smoke test completed end-to-end, so the data format and basic train/eval loop are workable.
- The smoke BLEU is not meaningful because it used official `transformers` fallback and random gloss embeddings.

## Verification
- Ran `prediction_slide.py` on dev with GPU `3`, avoiding GPU `1`.
- Loaded `dev_results.pkl` and summarized non-empty rate, average length, examples, and WER.
- Built smoke train/dev/test G2T pickle files from top-800 text and predicted glosses.
- Ran SLT/G2T smoke training and evaluation successfully in `slrt_legacy`.
- Confirmed the full-data top-800 train set is the closed 800-gloss + `<blank>` vocabulary set:
  - `data/csl-daily-top-800-all/csl_iso_top800_center_label_bag2items_halfblk.train`
  - 186247 samples
  - 801 classes

## Important Fixes / Compatibility Notes
- `Online/CSLR/configs/csl-daily-top-800_ISLR_subset32768.yaml` keeps `color_jitter: false` because previous training OOM happened in `ColorJitter.adjust_hue`.
- `Online/CSLR/configs/csl-daily-top-800_ISLR_full_stable.yaml` also keeps `color_jitter: false` for the first full-data baseline.
- SLT smoke required these compatibility fixes:
  - `Online/SLT/transformers_cust/utils/versions.py` handles compound version specifiers.
  - `Online/SLT/transformers_cust/dependency_versions_table.py` accepts installed `tokenizers==0.12.1`.
  - `Online/SLT/modelling/translation.py` falls back to official `transformers` when vendored `transformers_cust.models` is missing.
  - `Online/SLT/utils/external_metrics/sacrebleu.py` has a no-op fallback if `portalocker` is unavailable.
  - `sentencepiece` was installed into the `slrt_legacy` conda environment.

## SSH-Safe Training
- Plain foreground training is affected by SSH disconnect.
- Use the new document for persistent launch options:
  - `code_agent_logs/SSH_SAFE_TRAINING.md`
- Recommended full top-800 launch pattern:
  - start `tmux new -s slrt_full800`
  - run full train inside tmux
  - detach with `Ctrl+B`, then `D`
  - reattach with `tmux attach -t slrt_full800`

## Risks / Blockers
- The old `Online/CSLR/results/csl-daily-top-800_ISLR` checkpoint should still be avoided because it came from the wrong continuous/slide config path.
- The subset32768 checkpoint is not strong enough for final 800-word translation quality.
- Formal wait-k G2T is not fully clean yet:
  - vendored custom mBART files/assets still need cleanup
  - current smoke used official `transformers` fallback
  - current smoke used random gloss embeddings
- Large smoke/debug checkpoints under `Online/SLT/results/g2t_wait2_csl_top800_smoke_debug/` may consume substantial disk space.

## Next Steps
- Start full top-800 ISLR training with:
  - `Online/CSLR/configs/csl-daily-top-800_ISLR_full_stable.yaml`
- Use tmux or nohup from `code_agent_logs/SSH_SAFE_TRAINING.md` so training survives SSH disconnects.
- After full ISLR training:
  - run `prediction_slide.py` on dev/test with the full-stable checkpoint
  - check non-empty rate, blank rate, repetition, WER, and examples
  - only then move from G2T smoke to stronger formal G2T training.
