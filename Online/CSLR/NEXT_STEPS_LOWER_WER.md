# Next Steps To Lower WER

## Current Status

The latest adaptive stride search shows:

- Fixed `stride=1` is still the strongest overall baseline with test WER around `21.86`.
- The best adaptive configuration found so far is:
  - `min_stride=1`
  - `max_stride=3`
  - `quantile_low=0.20`
  - `quantile_high=0.70`
  - `ema_decay=0.40`
  - best decode method: `window_greedy_7`
  - dev WER: `22.10`
  - test WER: `22.33`
- This means the best adaptive setting is only about `0.47` WER behind the fixed `stride=1` baseline.

## Recommended Default Parameters

If adaptive stride must be enabled by default, use this configuration first:

```yaml
data:
  stride: 1
  adaptive_stride:
    enabled: true
    min_stride: 1
    max_stride: 3
    confidence_threshold: 0.2
    ema_decay: 0.4
    quantile_low: 0.2
    quantile_high: 0.7
```

Recommended decode method priority:

1. `window_greedy_7`
2. `window_greedy_5`
3. `window_greedy_3`

Do not use `max_stride >= 4` as the default adaptive setting. Current results show that larger stride ranges degrade temporal voting stability.

## Best Next Moves Without Training

These should be tried before any parameter retraining.

### 1. Use a Hybrid Runtime Policy

Keep fixed `stride=1` as the quality-first baseline and expose adaptive stride as a latency-reduction mode.

Suggested policy:

- default offline evaluation: fixed `stride=1` + `window_greedy_7`
- default adaptive runtime mode: adaptive `1/3`, `q=0.2/0.7`, `ema=0.4`, `window_greedy_7`
- if adaptive window count becomes too sparse for a sample, fall back to fixed `stride=1`

### 2. Replace Uniform Window Voting With Span-Weighted Voting

Current `window_greedy_*` still averages logits uniformly within the selected time-span neighborhood.

Next improvement:

- weight neighboring clips by temporal distance to the center clip
- use a triangular or Gaussian weight over clip-center distance
- keep the same real-time-span neighborhood, but avoid giving far-edge clips the same importance as near-center clips

Why this is promising:

- adaptive stride creates irregular clip densities
- uniform averaging can still over-count sparse long-span windows
- distance-weighted averaging should better preserve local evidence

### 3. Add Confidence-Gated Hybrid Decoding

Instead of always using one decode method globally:

- use `window_greedy_7` when per-window confidence is high enough
- fall back to `window_greedy_5` or even `naive_greedy` when confidence collapses or window counts are too small

This is useful because some samples prefer aggressive smoothing while others are harmed by it.

### 4. Analyze Failure Cases By Sample Type

Before training anything, split the current errors by:

- short sentences vs long sentences
- low-motion vs high-motion signing
- dense fingerspelling / fast transitions vs steady signing

Goal:

- determine whether adaptive stride is hurting because of under-sampling fast transitions
- or because the decoder still oversmooths low-density clip sequences

## When Training Starts Making Sense

Retraining is not required just because `window_greedy` became time-aware. That change is in decoding, not in model parameters.

Retraining becomes worthwhile only if you want adaptive stride to fully replace fixed `stride=1` as the default mode.

At that point, the training plan should be:

### 1. Train With Stride Mismatch Reduction

Introduce training-time temporal subsampling or stride augmentation:

- randomly simulate `stride=1/2/3`
- keep `win_size` fixed
- expose the model to variable clip density during training

This aligns the recognition logits with the adaptive inference regime.

### 2. Train With Temporal Density Augmentation

Augment by randomly dropping or duplicating frames in a controlled way:

- mild frame drop
- mild local time warping
- local duplication near pauses

This helps the model become robust to non-uniform temporal evidence.

### 3. Fine-Tune Only, Do Not Start From Scratch

The gap to fixed `stride=1` is already small. A short fine-tuning phase is the right first training step.

Suggested priority:

1. start from `cslr_best.ckpt`
2. fine-tune with adaptive-style temporal augmentation
3. evaluate fixed `stride=1` and adaptive `1/3` side by side

## Recommended Execution Order

1. Keep fixed `stride=1 + window_greedy_7` as the accuracy baseline.
2. Promote adaptive `1/3, q=0.2/0.7, ema=0.4` as the current best adaptive preset.
3. Implement span-weighted logit voting on top of the current real-time-span neighborhood.
4. Add confidence-gated fallback between `window_greedy_7`, `window_greedy_5`, and fixed `stride=1`.
5. Run error slicing to identify which sample categories still regress.
6. Only then decide whether adaptive-specific fine-tuning is necessary.

## Decision Summary

- If the goal is maximum recognition quality today: stay with fixed `stride=1`.
- If the goal is lower latency with minimal quality loss: use adaptive `1/3, q=0.2/0.7, ema=0.4`.
- If the goal is to make adaptive the permanent default: the next most valuable change is not full retraining yet; it is span-weighted voting plus confidence-gated fallback.