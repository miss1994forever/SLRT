# ISLR Dev Diagnostic Analysis

Date: 2026-05-30
Repository: /home/haojun/projects/SLRT

## Context

This document records the diagnosis of the current top-800 ISLR checkpoint before moving to continuous `prediction_slide`.

Model/result path:

- Config: `Online/CSLR/configs/csl-daily-top-800_ISLR_full_stable.yaml`
- Checkpoint: `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/ckpts/best.ckpt`
- Diagnostic output:
  - `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best/dev_diagnostic_summary.json`
  - `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best/dev_diagnostic_streams.csv`
  - `Online/CSLR/results/csl-daily-top-800_ISLR_full_stable/diagnostics/dev_best/dev_diagnostic_examples.pkl`

Diagnostic script:

- `Online/CSLR/tools/islr_dev_diagnostic.py`

The diagnostic was run on isolated dev samples, not on continuous sliding-window prediction.

## Data Reminder

Top-800 ISLR dev:

- Examples seen: `13077`
- Vocab size: `801`
- Blank id: `0`
- Classes: `800 glosses + <blank>`

The dev set contains many blank samples:

- Dev `<blank>` samples: `5887 / 13077`
- Blank ratio: about `45.02%`

Therefore overall Top-1 can be misleading. A model can look acceptable by predicting `<blank>` well while still missing many real glosses.

## Main Results

| Stream | Overall Top-1 | Signness Acc | Nonblank Top-1 | Gloss -> Blank | Pred Blank |
|---|---:|---:|---:|---:|---:|
| RGB | 52.48 | 57.91 | 15.16 | 74.97 | 85.36 |
| Keypoint | 66.28 | 78.27 | 48.32 | 29.87 | 56.14 |
| Fuse | 62.11 | 70.73 | 35.30 | 49.01 | 69.65 |
| Ensemble | 63.11 | 70.68 | 36.80 | 49.43 | 70.05 |

Additional Top-k results:

| Stream | Overall Top-5 | Overall Top-10 | Nonblank Top-5 | Nonblank Top-10 |
|---|---:|---:|---:|---:|
| RGB | 76.75 | 84.02 | 57.73 | 70.93 |
| Keypoint | 87.57 | 91.59 | 77.66 | 84.71 |
| Fuse | 86.32 | 90.41 | 75.13 | 82.56 |
| Ensemble | 87.15 | 90.97 | 76.63 | 83.57 |

## Findings

### 1. Keypoint Is The Strongest Stream

Keypoint is best on nearly every important metric:

- Overall Top-1: `66.28`
- Nonblank Top-1: `48.32`
- Signness accuracy: `78.27`
- Gloss-to-blank error: `29.87`

This suggests that the keypoint stream is currently the most reliable recognizer for this checkpoint.

### 2. RGB Is Overly Blank-Biased

RGB predicts `<blank>` for `85.36%` of dev samples.

Its nonblank behavior is weak:

- Nonblank sign recall: `25.03`
- Gloss-to-blank error: `74.97`
- Nonblank Top-1: `15.16`

RGB is not just slightly worse; it strongly rejects real gloss samples as `<blank>`.

### 3. Equal Ensemble Is Worse Than Keypoint

The equal ensemble underperforms keypoint:

- Keypoint overall Top-1: `66.28`
- Ensemble overall Top-1: `63.11`
- Keypoint nonblank Top-1: `48.32`
- Ensemble nonblank Top-1: `36.80`

This means equal fusion is dragging down the strongest stream.

The current ensemble behavior is not reliable enough to be the default for continuous online CSLR.

### 4. The Reported 63.11% Is Not True Gloss Recognition Strength

The ensemble overall Top-1 is `63.11`, but nonblank Top-1 is only `36.80`.

Reason:

- `<blank>` is frequent in dev.
- Ensemble blank recall is high: `95.24`.
- Many real gloss samples are still predicted as `<blank>`.

Therefore isolated overall Top-1 overstates the useful gloss recognition ability.

### 5. The Dominant Error Is Gloss -> Blank

Top ensemble confusions are mostly real glosses predicted as `<blank>`:

- `你 -> <blank>`
- `我 -> <blank>`
- `这 -> <blank>`
- `他 -> <blank>`
- `1 -> <blank>`
- `有 -> <blank>`
- `可以 -> <blank>`
- `人 -> <blank>`

The model is not simply confusing arbitrary glosses. It is too conservative and often rejects valid signs.

## Interpretation

The current issue is probably not a basic data-label mapping error.

More likely problems:

- `<blank>` and gloss classes are competing in a single 801-way classifier.
- RGB stream has learned an overly conservative blank decision.
- Fuse and equal ensemble inherit part of RGB's blank bias.
- The training objective optimizes overall classification but does not explicitly separate:
  - whether a sign is present
  - which gloss is present
  - where the sign is located inside the window

This supports the future research direction:

- Signness-aware ISLR
- Boundary/duration-aware ISLR
- Reliability-aware multimodal fusion

## What This Means For Prediction Slide

Directly using equal ensemble in `prediction_slide` is risky.

Expected risk:

- Output may be blank-heavy.
- Real gloss windows may be suppressed.
- Continuous WER may be worse than expected from isolated overall Top-1.

Better immediate tests:

1. Keypoint-only continuous prediction.
2. Heuristic weighted fusion with higher keypoint weight.
3. Equal ensemble only as a baseline, not as the trusted default.

## 2026-05-31 Follow-Up: Weighted Fusion And Slide Evaluation

Inference-side selection/fusion support has been implemented.

Changed files:

- `Online/CSLR/tools/islr_dev_diagnostic.py`
- `Online/CSLR/prediction_slide.py`
- `Online/CSLR/configs/slide_csl-daily-top-800_full_stable.yaml`

`prediction_slide.py` now supports:

```text
--pred_src ensemble|rgb|keypoint|fuse|weighted
--rgb_weight
--keypoint_weight
--fuse_weight
--max_samples
```

The isolated dev weight grid selected this best weighted candidate:

```text
rgb=0.0
keypoint=0.9
fuse=0.1
```

But isolated dev still favored keypoint-only on the most important nonblank metrics:

| Strategy | Nonblank Top-1 | Signness Acc | Overall Top-1 | Gloss -> Blank |
|---|---:|---:|---:|---:|
| Keypoint-only | 48.32 | 78.27 | 66.28 | 29.87 |
| Best weighted | 47.55 | 77.88 | 66.32 | 31.43 |

Continuous dev `prediction_slide` results:

| Strategy | Best Decode | WER | DEL | INS | SUB |
|---|---|---:|---:|---:|---:|
| Keypoint-only | window_greedy_5 | 48.29 | 17.52 | 6.66 | 24.11 |
| Equal ensemble | window_greedy_3 | 52.84 | 29.20 | 4.00 | 19.64 |
| Weighted 0/0.9/0.1 | window_greedy_5 | 48.18 | 18.93 | 5.96 | 23.28 |

Interpretation:

- Equal ensemble is clearly worse than keypoint-only.
- Weighted fusion is the best continuous dev number, but only by `0.11` WER.
- The improvement is too small to treat static weighted fusion as a strong result.
- For now, `pred_src=keypoint` is the safer baseline; `pred_src=weighted --rgb_weight 0.0 --keypoint_weight 0.9 --fuse_weight 0.1` is a candidate for test comparison.

Detailed command/results log:

- `code_agent_logs/2026-05-31/summary.md`

## Keypoint-Only And Weighted Fusion

These can first be implemented as inference/diagnostic strategies without retraining.

Current checkpoint already outputs:

- `rgb_gloss_logits`
- `keypoint_gloss_logits`
- `fuse_gloss_logits`
- `ensemble_last_gloss_logits`

Therefore one checkpoint can support multiple decoding strategies:

- RGB only
- Keypoint only
- Fuse only
- Equal ensemble
- Heuristic weighted fusion

Suggested heuristic starting point:

```text
weighted_prob = 0.10 * rgb_prob + 0.70 * keypoint_prob + 0.20 * fuse_prob
```

The exact weights should be tuned on dev.

## Model Architecture Impact

### Minimal Change: Inference-Only Strategy

Files likely to change:

- `Online/CSLR/tools/islr_dev_diagnostic.py`
- `Online/CSLR/prediction_slide.py`

Expected changes:

- Add a `--logit_source` or similar argument:
  - `rgb`
  - `keypoint`
  - `fuse`
  - `ensemble`
  - `weighted`
- Add weight arguments for weighted fusion:
  - `--rgb_weight`
  - `--keypoint_weight`
  - `--fuse_weight`

This does not require retraining and does not change the model architecture.

### Medium Change: Learned Weighted Fusion

The code already has partial support:

- `modelling/recognition.py`
- `visual_head.weighted`
- `WeightLearner`

Possible config experiment:

```yaml
model:
  RecognitionNetwork:
    visual_head:
      weighted: true
```

This should be treated as a separate training experiment.

### Larger Change: Signness / Boundary / Duration Heads

Files likely to change:

- `Online/CSLR/dataset/Dataloader.py`
- `Online/CSLR/modelling/Visualhead.py`
- `Online/CSLR/modelling/recognition.py`
- `Online/CSLR/prediction_slide.py`
- `Online/CSLR/tools/islr_dev_diagnostic.py`

New targets:

- `signness_target`: `0` for `<blank>`, `1` for real gloss
- `center_target`: normalized center of the base sign span
- `duration_target`: normalized duration/span of the base sign
- mask boundary losses for `<blank>` samples

New outputs:

- gloss logits
- signness logits
- center prediction
- duration prediction

This is a research contribution direction and should not be mixed with all other changes at once.

## Recommended Next Steps

1. Do not continue the same full ISLR training recipe blindly.
2. Add inference-side selection/fusion support to diagnostics and `prediction_slide`.
3. Run isolated dev diagnostics for:
   - keypoint only
   - equal ensemble
   - weighted fusion
4. Run continuous `prediction_slide dev` with:
   - keypoint only
   - equal ensemble
   - weighted fusion
5. If keypoint/weighted improves continuous WER, make reliability-aware fusion a formal module.
6. Then start a separate signness/boundary-aware ISLR experiment.

## Current Decision

The most urgent issue is not "train more epochs".

The most urgent issue is:

- reduce gloss-to-blank errors,
- stop weak RGB/fuse streams from suppressing keypoint,
- and separate sign presence from gloss identity.
