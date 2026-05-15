# Adaptive Stride Project Results

## 1. Project Goal

The goal of this phase was to improve Online CSLR inference with adaptive stride while keeping the recognition pipeline practical for deployment.

The work focused on three pieces:

1. adaptive stride window generation
2. span-weighted window voting
3. confidence-gated fallback decoding

This is an inference-time optimization effort. The core recognition checkpoint is reused; the main changes are in how clips are sampled and how window-level hypotheses are merged.

## 2. Experimental Setup

The final tested configuration uses:

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
postprocess:
  span_weighted_voting:
    enabled: true
    kernel: triangular
    min_weight: 0.05
```

For decoding, the best fixed method found in the latest evaluation is `window_greedy_13`.

## 3. Recognition Results

### Best adaptive result

On the Phoenix test set, the best adaptive configuration with span-weighted voting achieved:

- `window_greedy_13` test WER: **24.18**

Source log:
- [adaptive_best_span.test.log](/root/autodl-tmp/SLRT/Online/CSLR/results/phoenix-2014t_ISLR/next_round_20260512/logs/adaptive_best_span.test.log)

### Best original baseline for comparison

The strongest fixed-stride baseline used for comparison is the original `stride=1` setup, whose test WER is about:

- **21.86**

This means the best adaptive result is still worse in WER, but it provides a substantial compute reduction.

### Adaptive runtime note

The dynamic fallback experiments showed that the best dynamic candidate was `7 -> 13`, but it still did not beat the fixed `window_greedy_13` result.

## 4. Compute Saving Calculation

The compute saving is estimated by comparing the number of generated clips in test inference.

Because the model’s main compute cost is approximately proportional to the number of clips processed, clip count is a reasonable proxy for inference cost.

### Measured clip counts on the test split

From the test split:

- number of samples: **642**
- fixed `stride=1` total clips: **64627**
- adaptive stride total clips: **39353**

### Formula

Let:

$$\text{saving} = 1 - \frac{N_{adaptive}}{N_{fixed}}$$

where:

- $N_{fixed} = 64627$
- $N_{adaptive} = 39353$

### Substitution

$$\frac{N_{adaptive}}{N_{fixed}} = \frac{39353}{64627} \approx 0.608925$$

$$\text{saving} = 1 - 0.608925 = 0.391075$$

### Final result

$$\text{compute saving} \approx 39.11\%$$

This corresponds to an adaptive over fixed clip ratio of about **60.89%**, or roughly **1.64x** clip throughput improvement in the clip-count proxy.

## 5. Summary of Outcomes

- Adaptive stride is effective for reducing inference compute.
- The final measured compute reduction on the test set is **39.11%**.
- In the current evaluation setup, the best accuracy comes from fixed `window_greedy_13` with test WER **24.18**.
- The dynamic fallback experiments helped identify better fallback windows, but they did not surpass the best fixed `window_greedy_13` decode.

## 6. Practical Conclusion

If the priority is recognition quality, the current best choice is fixed `window_greedy_13`.

If the priority is lower compute cost with acceptable quality trade-off, adaptive stride provides a clear efficiency gain and should be reported as the main deployment-oriented improvement from this project phase.