# Code Agent Daily Log

Date: 2026-05-30
Repository: /home/haojun/projects/SLRT

## Goal

- Diagnose the top-800 full ISLR training after epoch 4/5 degradation.
- Confirm whether the current training data and task setup are fundamentally correct.
- Decide whether to continue training or switch to evaluation/diagnostics.

## Current Training Result

Run:
- Config: `Online/CSLR/configs/csl-daily-top-800_ISLR_full_stable.yaml`
- Log: `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/train.rank0.log`
- Result dir: `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable`

Best epoch so far:
- Epoch 3
- `ensemble_last_Per-instance ACC Top-1: 63.10`
- `ensemble_last_Per-class ACC Top-1: 23.01`
- `ensemble_last_Per-class ACC Top-5: 64.77`
- `ensemble_last_Per-class ACC Top-10: 75.54`
- `total_loss Average: 9.86`

Epoch 4 degraded:
- `ensemble_last_Per-instance ACC Top-1: 55.86`
- `ensemble_last_Per-class ACC Top-1: 14.45`
- `ensemble_last_Per-class ACC Top-5: 70.37`
- `ensemble_last_Per-class ACC Top-10: 80.39`
- `total_loss Average: 10.52`

Epoch 5 degraded further:
- `ensemble_last_Per-instance ACC Top-1: 55.13`
- `ensemble_last_Per-class ACC Top-1: 11.43`
- `ensemble_last_Per-class ACC Top-5: 58.89`
- `ensemble_last_Per-class ACC Top-10: 71.16`
- `total_loss Average: 10.57`

Interpretation:
- Training after epoch 3 is not improving the selected dev metric.
- Epoch 4 looked like the correct class was often still in Top-5/Top-10, but final Top-1 calibration became worse.
- Epoch 5 confirms this is not just a one-epoch fluctuation.
- The current best checkpoint should remain `best.ckpt` / epoch 3.

## Data And Task Sanity Check

The current config is an ISLR setup:

```yaml
task: ISLR
data:
  dataset_name: csl_iso
  train: /home/haojun/projects/SLRT/data/csl-daily-top-800-all/csl_iso_top800_center_label_bag2items_halfblk.train
  dev: /home/haojun/projects/SLRT/data/csl-daily-top-800-all/csl_iso_top800_with_blank.dev
  test: /home/haojun/projects/SLRT/data/csl-daily-top-800-all/csl_iso_top800_with_blank.test
  vocab_file: /home/haojun/projects/SLRT/data/csl-daily-top-800-all/csl_iso_with_blank.vocab
```

Training data inspection:
- Train file type: `dict`
- Train groups: `186247`
- First group is a bag/list of isolated samples.
- Example sample:
  - `name: 你们_S000000_P0000_T00_[16:29]`
  - `label: 你们`
  - `seq_len: 13`
  - `start: 16`
  - `end: 29`
  - `base_start: 16`
  - `base_end: 29`

Dev data inspection:
- Dev file type: `list`
- Dev examples: `13077`
- Example sample:
  - `video_file: S000020_P0000_T00`
  - `name: 他_S000020_P0000_T00_[0:9]`
  - `label: 他`
  - `seq_len: 9`
  - `start: 0`
  - `end: 9`
- Dev classes present: `800`
- Dev `<blank>` count: `5887`

Conclusion:
- The current training data is not raw full-sentence CSLR training.
- It is top-800 closed-vocabulary CSL-Daily data converted into isolated/pre-segmented word clips, with `<blank>`/non-sign clips included.
- The model being trained is an ISLR classifier, not a continuous CTC CSLR model and not an SLT model.
- There is no evidence from the config/data check that the wrong dataset branch is being used.

## Main Suspected Issue

The strongest warning sign is not a data format error, but model/optimization behavior after epoch 3:

- Keypoint stream is consistently stronger than RGB/fuse in later epochs.
- Equal ensemble can be worse than the keypoint stream alone.
- RGB and fuse streams may be dragging down `ensemble_last`.

Epoch 5 stream metrics:
- RGB Top-1: `48.46`
- Keypoint Top-1: `60.95`
- Fuse Top-1: `55.07`
- Equal ensemble Top-1: `55.13`

This suggests:
- The keypoint stream remains the most reliable stream.
- Equal RGB/keypoint/fuse probability summation is probably not optimal.
- Continuing the same training recipe may not fix the issue.

## Decision

Do not continue the same full training run blindly.

Use epoch 3 / `best.ckpt` as the current best model and switch to diagnostics:

1. Run `prediction_slide` dev with `best.ckpt`.
2. Evaluate continuous online metrics:
   - WER
   - non-empty rate
   - blank rate
   - duplicate/repetition rate
   - average gloss length
3. Compare decoding variants:
   - equal ensemble
   - keypoint-only if easy to expose
   - later: weighted/reliability-aware ensemble
4. Only restart training after diagnostics show whether the bottleneck is:
   - isolated recognition quality,
   - stream fusion,
   - blank handling,
   - or sliding-window decoding.

## Next Recommended Work

Immediate next step:
- Generate dev slide predictions from:
  - `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/ckpts/best.ckpt`

Then:
- Compare output quality with the previous subset32768 slide predictions.
- If best full model improves continuous dev WER, use it as the current recognizer baseline.
- If continuous dev WER is still poor despite 63.10 isolated Top-1, prioritize online decoding improvements:
  - boundary-aware decoding
  - reliability-aware fusion
  - constrained gloss reranking

Training continuation policy:
- Continue training only if using a modified experiment config.
- Candidate modified configs:
  - lower learning rate from `6e-4`
  - keypoint-only diagnostic run
  - weighted fusion / reliability-aware fusion
  - stronger handling of `<blank>` and foreground spans
  - shorter early-stop schedule around epoch 3

## Open Questions

- Does `best.ckpt` point to epoch 3 as expected after epoch 5? It should, because `best_score` remained `63.10`.
- Does continuous `prediction_slide` benefit from the epoch 3 full model compared with subset32768?
- Is equal ensemble consistently worse than keypoint-only during continuous sliding-window prediction?

## ISLR Dev Diagnostic Script

Added:

- `Online/CSLR/tools/islr_dev_diagnostic.py`
- Dedicated analysis doc:
  - `code_agent_logs/2026-05-30/islr_dev_diagnostic_analysis.md`

Purpose:

- Diagnose isolated dev behavior before running continuous `prediction_slide`.
- Separate 801-class Top-1 from signness/blank behavior.
- Compare RGB, keypoint, fuse, and equal ensemble.
- Save machine-readable JSON/CSV results and mistake examples.

Metrics produced per stream:

- Overall Top-1 / Top-5 / Top-10
- Binary signness accuracy
- Blank recall
- Blank-to-gloss error rate
- Nonblank sign recall
- Gloss-to-blank error rate
- Nonblank-only Top-1 / Top-5 / Top-10
- Predicted blank rate
- Average max probability
- Average entropy
- Top predicted classes
- Top confusions
- Worst/best classes with enough support

Smoke test command:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR

CUDA_VISIBLE_DEVICES=0 \
conda run -n slrt_legacy python tools/islr_dev_diagnostic.py \
  --max-batches 1 \
  --output-dir /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best_smoke
```

Smoke test passed and wrote:

- `diagnostics/dev_best_smoke/dev_diagnostic_summary.json`
- `diagnostics/dev_best_smoke/dev_diagnostic_streams.csv`
- `diagnostics/dev_best_smoke/dev_diagnostic_examples.pkl`

Full dev command:

```bash
cd /home/haojun/projects/SLRT/Online/CSLR

CUDA_VISIBLE_DEVICES=0 \
conda run -n slrt_legacy python tools/islr_dev_diagnostic.py \
  --output-dir /home/haojun/projects/SLRT/Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best
```

Use another visible GPU by changing `CUDA_VISIBLE_DEVICES`.

## Preprocessing And Gloss Mapping Check

Checked on 2026-05-30 after the epoch 5 degradation.

### Vocab / Label Mapping

The ISLR vocab file is a JSON list, not a plain one-gloss-per-line text file:

- File: `data/csl-daily-top-800-all/csl_iso_with_blank.vocab`
- Length: `801`
- Index 0: `<blank>`
- Duplicate vocab entries: `0`

The training loader uses the same logic:

```python
vocab = json.load(open(vocab_file, "rb"))
labels = [vocab.index(sample["label"]) for sample in batch]
```

So the active ISLR label mapping is:

- `<blank>` -> 0
- 800 target glosses -> 1..800

The separate `gloss2ids.pkl` has `804` entries because it contains SLT/G2T special tokens:

- `<si>`, `<unk>`, `<pad>`, `</s>`
- then the same 800 target glosses

This mismatch is expected and is not used for ISLR class labels in `Dataloader.py`.

### Isolated Top-800 Files

Checked files:

- `csl_iso_top800_center_label_bag2items_halfblk.train`
- `csl_iso_top800_with_blank.dev`
- `csl_iso_top800_with_blank.test`
- `csl_iso_top800_balanced32768.train`
- `csl_iso_top800_balanced8192.train`

Results:

- Missing labels from ISLR vocab: `0`
- Bad `name` prefix vs `label`: `0`
- Bad `end - start != seq_len`: `0`
- Bad negative/empty spans: `0`

Full top-800 train:

- Raw type: `dict`
- Groups/bags: `186247`
- Expanded samples: `1949264`
- Classes: `801`
- `<blank>` samples: `737450`
- Augmented samples:
  - `aug=0`: `186247`
  - `aug=1`: `1763017`
- `seq_len` min/mean/median/max: `5 / 15.05 / 16 / 249`

Dev:

- Examples: `13077`
- Classes present: `800`
- `<blank>` samples: `5887`
- `seq_len` min/mean/median/max: `5 / 9.26 / 9 / 97`

Test:

- Examples: `14741`
- Classes present: `801`
- `<blank>` samples: `6826`
- `seq_len` min/mean/median/max: `5 / 9.49 / 9 / 81`

Conclusion:

- No evidence was found that the top-800 isolated sample labels, spans, or vocab mapping are fundamentally wrong.
- The current samples are correctly represented as pre-segmented isolated CSL-Daily gloss clips, with `<blank>` clips included.

### Top-800 Subset Construction

The subset builder is `/home/haojun/projects/build_csl_daily_subset.py`.

The top-800 subset was built with:

- `--match-mode all`
- `--write-derived-vocabs`

`match-mode all` means a continuous CSL-Daily sentence is kept only if every gloss token in the sentence belongs to the target 800-gloss set. Therefore the top-800 continuous split is a closed-vocabulary subset, not a loose partial match subset.

### Data Volume Compared With Original CSL-Daily ISLR

Original local CSL-Daily ISLR data:

- Vocab size: `2001`
- Train groups/bags: `200530`
- Expanded train samples: `2073040`
- Dev examples: `14059`
- Test examples: `15826`

Top-800 ISLR data:

- Vocab size: `801`
- Train groups/bags: `186247`
- Expanded train samples: `1949264`
- Dev examples: `13077`
- Test examples: `14741`

Interpretation:

- Top-800 training is not much smaller than the original local CSL-Daily ISLR training set.
- Train groups are reduced by about 7.1%.
- Expanded train samples are reduced by about 6.0%.
- Therefore the current degradation after epoch 3 is unlikely to be caused only by insufficient data volume.

More likely causes:

- Optimization/over-training after epoch 3.
- Equal ensemble being dragged down by weaker RGB/fuse streams.
- Difference from original training recipe, especially `color_jitter: false`.
- Initial checkpoint may not be the original final ISLR checkpoint for this exact 801-class task.
- Strong `<blank>` and class imbalance effects.
- Per-class metric instability because many classes have few dev/test examples.
