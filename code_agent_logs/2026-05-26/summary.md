# Code Agent Daily Log

Date: 2026-05-26
Repository: /home/haojun/projects/SLRT

## Goal
- Create a daily log location for code-agent work.
- Reassess whether the current SLRT checkout can proceed to CSL-Daily top-800 translation.

## Changes Made
- Created this daily log.
- Added the reusable template at `code_agent_logs/TEMPLATE.md`.
- Added GPU/training operations cheatsheet at `code_agent_logs/GPU_TRAINING_COMMANDS.md`.
- Updated `Online/CSLR/configs/csl-daily-top-800_ISLR_subset32768.yaml` so future runs resume instead of overwriting:
  - `overwrite: false`
  - `from_ckpt: true`

## Current State
- Top-800 isolated CSLR data is present and readable:
  - Full train: `data/csl-daily-top-800-all/csl_iso_top800_center_label_bag2items_halfblk.train`, 186247 dict entries, bag/list samples.
  - Balanced subset train: `data/csl-daily-top-800-all/csl_iso_top800_balanced32768.train`, 32768 dict entries, covers 801 labels.
  - Isolated dev/test: `csl_iso_top800_with_blank.dev` has 13077 entries; `csl_iso_top800_with_blank.test` has 14741 entries.
  - Continuous CSL-Daily top-800 dev/test are gzip pickles and load correctly through the existing fallback in `Dataset.load_annotations`.
- Vocab state:
  - `csl_iso_with_blank.vocab` has 801 entries and `<blank>` at index 0.
  - `gloss2ids.pkl` has 804 entries with extra special tokens (`<si>`, `<unk>`, `<pad>`, `</s>`); this mismatch should be kept in mind, but the current ISLR labels are generated from `vocab_file`.
- Current useful CSLR checkpoint:
  - `Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/ckpts/best.ckpt`.
  - Best observed isolated dev score in `train.rank0.log`: epoch 2, `ensemble_last_Per-instance ACC Top-1: 20.47`.
  - Training appears interrupted during epoch 4; no active training process was found.
- No usable continuous slide prediction files were found under `results/csl-daily-top-800_ISLR_subset32768`.
- `train_csl_wait_k_full.sh` still points at the old `results/csl-daily-top-800_ISLR` and `configs/slide_csl-daily-top-800.yaml`, so it is not safe to run as-is.

## Verification
- Ran repository status inspection with `git status --short`.
- Listed top-800 CSLR configs and result artifacts.
- Read recent rank logs for `csl-daily-top-800_ISLR_subset32768` and `csl-daily-top-800_ISLR_v2`.
- Loaded and summarized top-800 pickle/gzip-pickle files with the `slrt_legacy` conda env.
- Checked for active `training.py` / `prediction_slide.py` processes; none were active.
- Checked rank logs for explicit Python/OOM/error traces; none were found.

## 2026-05-26 OOM On subset32768 Resume
- User restarted `csl-daily-top-800_ISLR_subset32768` from `epoch_03.ckpt`; training resumed at epoch 4 but failed at step 11 with CUDA OOM.
- The fatal stack trace is inside RGB augmentation: `ColorJitter -> adjust_hue -> _hsv2rgb -> torch.einsum`.
- The repeated pretrained-weight messages are not the fatal issue. Classifier head mismatches are expected when adapting a checkpoint to the 800-word + `<blank>` output space.
- With `batch_size: 4` and `bag_size/num_instance: 6`, each rank can temporarily process 24 cropped instances. Hue jitter creates large intermediate tensors and can exceed 24 GB GPU memory.
- Applied config fix:
  - `Online/CSLR/configs/csl-daily-top-800_ISLR_subset32768.yaml`
  - changed `data.transform_cfg.color_jitter` from `true` to `false`
  - kept `training.batch_size: 4` to preserve the current resume setup and epoch step count.
- Recommended restart adds allocator fragmentation mitigation:
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
- If OOM still occurs after disabling color jitter, the next fallback is changing `training.batch_size` from `4` to `2`. That should reduce memory more reliably, but each epoch will take more steps.

## 2026-05-27 G2T Smoke From subset32768 Slide Predictions
- Generated top-800 dev slide predictions from `Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/ckpts/best.ckpt`.
- Main prediction file:
  - `Online/CSLR/results/csl-daily-top-800_ISLR_subset32768/prediction_slide/dev/dev_results.pkl`
- Best decoding among tested settings was `window_greedy_13`:
  - WER: `87.22`
  - non-empty rate: `360/360`
  - average predicted gloss length: `7.57`
- Created G2T smoke data aligned with top-800 CSL-Daily text:
  - `Online/SLT/data_smoke/top800_g2t_from_subset32768/csl_pred_smoke.train`
  - `Online/SLT/data_smoke/top800_g2t_from_subset32768/csl_pred_smoke.dev`
  - `Online/SLT/data_smoke/top800_g2t_from_subset32768/csl_pred_smoke.test`
  - `Online/SLT/data_smoke/top800_g2t_from_subset32768/csl_pred_smoke.dev_pred_all`
- Added smoke config:
  - `Online/SLT/configs/g2t_wait2_csl_top800_smoke.yaml`
- G2T smoke completed end-to-end:
  - data loading passed
  - text/gloss tokenization passed
  - model forward/backward passed
  - step and epoch validation passed
  - dev/test evaluation artifacts were written under `Online/SLT/results/g2t_wait2_csl_top800_smoke_debug`
- Environment fixes needed to run smoke:
  - installed `sentencepiece` in `slrt_legacy`
  - made `transformers_cust` version parsing handle compound specifiers
  - relaxed vendored `tokenizers` requirement to accept installed `0.12.1`
  - added a `portalocker` no-op fallback for local sacrebleu evaluation
  - made SLT translation fallback to official `transformers` when local `transformers_cust.models` is missing
- Smoke caveat:
  - This smoke uses official `transformers` fallback and random gloss embeddings, so BLEU is not meaningful.
  - Formal wait-k G2T training still needs the custom MBart path and valid/pruned mBART assets to be cleaned up.

## 2026-05-27 Full Top-800 ISLR Config
- Added full-data stable ISLR config:
  - `Online/CSLR/configs/csl-daily-top-800_ISLR_full_stable.yaml`
- It uses the full top-800 closed-vocabulary ISLR train set:
  - `data/csl-daily-top-800-all/csl_iso_top800_center_label_bag2items_halfblk.train`
  - 186247 samples
  - 801 labels = 800 glosses + `<blank>`
- It writes to a separate result directory:
  - `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable`
- It keeps `color_jitter: false` to avoid the previous OOM path in `ColorJitter.adjust_hue`.

## Risks / Blockers
- Cannot proceed to 800-word translation yet because the required continuous CSLR prediction outputs have not been generated and validated.
- The current 32768-subset checkpoint is promising but incomplete; isolated top-1 reached 20.47%, then training later degraded and was interrupted.
- Downstream wait-k/G2T config expects converted files at `Online/CSLR/results/online_slr_csl/prediction_slide/csl_pred.{dev,test}`; those files are not currently produced from the subset32768 checkpoint.
- Running the existing wait-k shell script without edits would reuse the old top-800 CSLR path.

## Next Steps
- Resume `csl-daily-top-800_ISLR_subset32768` to finish the planned 10 epochs:
  ```bash
  cd /home/haojun/projects/SLRT/Online/CSLR
  CUDA_VISIBLE_DEVICES=3,4,5,6 \
  conda run -n slrt_legacy python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 29999 \
    --use_env \
    training.py \
    --config=configs/csl-daily-top-800_ISLR_subset32768.yaml
  ```
- After subset32768 finishes, generate dev slide predictions with `configs/slide_csl-daily-top-800_subset32768.yaml` and check non-empty `window_greedy_13_gls_hyp`.
- If slide predictions are non-empty and reasonable, either:
  - continue with G2T conversion as a smoke/quality check, or
  - start full 186247-example ISLR training for a production-quality 800-word CSLR checkpoint.

## Model / Task Notes
- The immediate next training is `task: ISLR`, not full SLT.
- It is an isolated sign language recognition model in `Online/CSLR`, trained on top-800 isolated samples.
- Its checkpoint is used as the recognizer for later continuous CSLR sliding-window prediction.
- The later SLT/G2T step uses predicted gloss sequences from CSLR; it should not start until slide predictions have been generated and checked.
- Architecture assessment on 2026-05-26:
  - The current architecture matches the Online CSLR paper direction: RGB + keypoint two-stream S3D, gloss-level sampling, classification losses, and sliding-window inference.
  - No direct evidence was found that architecture changes are needed before completing training.
  - The immediate bottleneck is training completeness and data scale: the current subset32768 run reached `ensemble_last_Per-instance ACC Top-1: 20.47` at epoch 2, then was interrupted during epoch 4.
- The original paper reports CSLR WER rather than top-800 ISLR classification accuracy. Reported CSL-Daily online CSLR WER for "Ours" is 30.2 dev / 29.3 test, and boosted offline CSLR WER is 24.8 dev / 24.4 test.
- The paper reports a 93.4% sign-segmentor human-evaluation accuracy for CTC forced alignment, but that is not ISLR model Top-1 accuracy.

## Possible Accuracy Improvements After Full Training
- Do not apply these changes before completing a clean full-data baseline. They are second-stage experiments if full 186247-example training plus slide prediction still underperforms.

### 1. Stricter Saliency / Foreground Alignment
- Goal:
  - Make the ISLR model focus on the true sign portion inside a fixed 16-frame window and ignore co-articulation/background frames.
  - This matters because online inference slides a short window through continuous video, so many windows contain both a sign and transition motion.
- Current code support:
  - Data samples already carry `start`, `end`, `base_start`, `base_end`, and `aug`.
  - `Dataloader.collate_fn_` computes `temp_idx` when `base_start` exists.
  - `VisualHead` supports `split_setting: split_nonblk`, using `temp_idx` to separate foreground/non-blank regions.
  - `recognition.py` applies extra split/cam/attention losses when `head_split_setting` contains `split` and `temp_idx` is present.
- Possible changes:
  - Verify that `temp_idx` is non-null for full train batches and that foreground spans look sane.
  - Try stronger split settings already hinted by code, such as variants containing `cam` or `att`, only after confirming supported syntax in `VisualHead`.
  - Add logging for foreground span length distribution to catch bad segment boundaries.
  - If needed, increase the loss contribution from foreground/saliency terms, but only after inspecting current loss names and magnitudes.
- Expected benefit:
  - Better rejection of `<blank>` and transition windows.
  - Better continuous `prediction_slide` output, even if isolated Top-1 changes only modestly.
- Risks:
  - If pseudo boundaries are noisy, stronger foreground supervision can overfit wrong segments.
  - Can improve isolated dev while hurting continuous slide prediction, so always evaluate both.
- Suggested trigger:
  - Full-data ISLR Top-1 is acceptable, but slide predictions are sparse, blank-heavy, or repetitive.

### 2. NLA-SLR Language-Aware Label Smoothing
- Goal:
  - Replace uniform label smoothing with semantic similarity-aware smoothing so visually/semantically related glosses get softer targets.
  - Useful for confusing signs and large vocabularies where many classes are visually close.
- Current code support:
  - `recognition.py` already supports string `label_smooth` values containing `word_emb_sim`.
  - NLA-SLR configs use patterns like `label_smooth: word_emb_sim_softmax_0.2_0.5`.
  - This requires `word_emb_tab` to be available via `data.word_emb_file`; otherwise the loss cannot construct the word similarity matrix.
- Possible changes:
  - Build or locate a 801-entry word embedding table aligned exactly to `csl_iso_with_blank.vocab`.
  - Add `data.word_emb_file: ...` to the top-800 ISLR config.
  - Change `model.RecognitionNetwork.label_smooth` from `0.2` to a string experiment, for example `word_emb_sim_softmax_0.2_0.5`.
  - Start with a copied config, not the baseline config.
- Expected benefit:
  - More stable learning for ambiguous classes.
  - Potentially better Top-5/Top-10 and downstream post-processing robustness.
- Risks:
  - Bad or misaligned embeddings will silently teach the wrong class relations.
  - Extra special-token mismatch exists: `csl_iso_with_blank.vocab` has 801 entries, while `gloss2ids.pkl` has 804 entries. Any embedding file must follow the 801-class vocab order used for ISLR labels.
- Suggested trigger:
  - Full-data training plateaus with low Top-1 but reasonable Top-5/Top-10, indicating confusion among related classes.

### 3. RGB / Keypoint Fusion Adjustments
- Goal:
  - Improve how RGB, keypoint, and fused heads contribute to the final prediction.
  - Current logs show keypoint and fuse heads can behave differently by epoch; a fixed fusion may not always be optimal.
- Current code support:
  - Current config uses `fuse_method: triplehead_cat_bilateral`.
  - `recognition.py` has per-head losses for `rgb`, `keypoint`, and `fuse`.
  - There is a `visual_head.weighted` path that learns probability weights via `WeightLearner`.
  - Lateral two-stream fusion can be adjusted with `lateral.pose2rgb`, `lateral.rgb2pose`, and `lateral.fusion_features`.
- Possible changes:
  - First analyze per-head dev metrics across epochs to see whether RGB, keypoint, or fuse dominates.
  - Try `visual_head.weighted: true` in a copied config if supported by the current path.
  - Try reducing fusion complexity if fuse underperforms both individual heads.
  - Try keypoint-only or RGB-only diagnostic configs to isolate which stream is weak.
  - Adjust `fusion_features` only after a diagnostic run; this changes feature exchange depth and can affect stability.
- Expected benefit:
  - More stable ensemble predictions.
  - Better use of keypoint stream when RGB is noisy or signer/background variation is high.
- Risks:
  - More fusion parameters can overfit the subset.
  - Weighted fusion can hide a broken stream instead of fixing it.
  - Changing lateral fusion is more invasive than changing loss settings.
- Suggested trigger:
  - Per-head metrics diverge strongly, e.g. keypoint is consistently much better than RGB/fuse, or fuse is worse than both streams.

### Recommended Experiment Order
1. Finish the current baseline path: subset32768 to 10 epochs, then full 186247-example ISLR.
2. Generate slide predictions and inspect non-empty rate, blank rate, duplicate/repetition rate, and qualitative examples.
3. If continuous predictions are blank-heavy, investigate saliency/foreground alignment first.
4. If isolated Top-1 is low but Top-5/Top-10 are healthy, try language-aware label smoothing with verified 801-class embeddings.
5. If per-head behavior is unstable, run fusion diagnostics and then try weighted RGB/keypoint/fuse prediction.
