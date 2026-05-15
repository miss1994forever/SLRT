# Span-Weighted Voting And Confidence Fallback Design

## Scope

This document refines the next-step decoding changes proposed in `NEXT_STEPS_LOWER_WER.md` into an implementation design that fits the current Online CSLR inference path.

This is a design-only document. It does not require model retraining and does not change model architecture.

## Current Control Path

The current decoding path already has the key primitive needed for adaptive-aware post-processing:

1. `sliding_windows(...)` returns `clip_centers`.
2. `evaluation_slide(...)` uses `clip_centers` to build a real-time-span neighborhood for each `window_greedy_*` decode size.
3. Inside that neighborhood, logits are still aggregated uniformly.
4. The decode method is still chosen globally, not per sample or per local segment.

That means the remaining mismatch is no longer neighborhood selection. The mismatch is the aggregation rule and the lack of selective fallback.

## Working Hypothesis

The remaining WER gap between fixed `stride=1` and the best adaptive setting is mainly caused by two local effects:

1. irregularly spaced adaptive clips are still treated as if every neighbor should contribute equally inside a time-span window
2. a single global decode method such as `window_greedy_7` is too aggressive for some sparse or low-confidence segments

This hypothesis is falsifiable.

The cheapest later check is:

- record per-sample confidence and fallback trigger statistics
- compare whether the adaptive regressions are concentrated in low-confidence or sparse-neighborhood samples

If the regressions are not concentrated there, then fallback logic is targeting the wrong failure mode.

## Design Goals

The design should satisfy all of the following:

1. Preserve the current real-time-span neighborhood definition based on `clip_centers`.
2. Improve adaptive decoding without changing model weights.
3. Avoid doubling latency in the default path.
4. Keep fixed-stride behavior stable when the new logic is disabled.
5. Expose enough diagnostics to understand when fallback helps or hurts.

## Proposed Change Summary

Introduce two decoding layers on top of the current neighborhood selection.

### Layer 1: Span-Weighted Voting

Replace uniform averaging inside each time-span neighborhood with distance-aware weighting over clip-center distance.

### Layer 2: Confidence Fallback

Choose between `window_greedy_7`, `window_greedy_5`, and `naive_greedy` based on local or sample-level confidence signals, instead of locking one method for every sample.

## Integration Points

The intended implementation surface is confined to the current decoding block in `prediction_slide.py`.

Primary touch points:

- `sliding_windows(...)`: no behavioral change required beyond continuing to return `clip_centers`
- `evaluation_slide(...)`: replace uniform aggregation and add fallback routing
- results serialization: add chosen-method and confidence diagnostics

No model checkpoint format changes are needed.

## Layer 1: Span-Weighted Voting

### Current Behavior

For each target position `t_idx` and decode size `decode_win_size`:

1. compute `half_span = 0.5 * decode_win_size * stride`
2. use `clip_centers` to find `[left:right)` within that time span
3. aggregate neighbors uniformly

Under adaptive stride, this still has a bias:

- a far-away sparse neighbor gets the same weight as a near-center neighbor
- edge clips in wide uneven neighborhoods can wash out local evidence

### Proposed Aggregation Rule

Keep the same `[left:right)` neighborhood, but weight each neighbor by distance to the target clip center.

Definitions:

- target center: $c_t$
- neighbor centers: $c_j$
- half span: $h = \max(0.5 \cdot \text{decode\_win\_size} \cdot \text{stride}, 1.0)$
- normalized distance: $d_j = \frac{|c_j - c_t|}{h}$

Preferred default kernel:

$$
w_j = \max(\epsilon, 1 - d_j)
$$

where $\epsilon$ is a small floor such as $0.05$ to avoid zeroing all edge evidence.

Then normalize:

$$
\hat{w}_j = \frac{w_j}{\sum_k w_k}
$$

Why triangular first:

- simpler than Gaussian
- no extra sigma tuning required for the first implementation
- directly tied to the existing span boundary
- easy to debug from logged weights

### Weighted Probability Aggregation

For the default path that currently does:

- `prob = logits.softmax(dim=-1).mean(dim=0)`

change the design to:

$$
p_t = \sum_{j=left}^{right-1} \hat{w}_j \cdot \text{softmax}(\ell_j)
$$

where $\ell_j$ is the class-logit vector for clip `j`.

Recommendation:

- weight probabilities, not raw logits, in the first version
- this preserves the semantics of the current implementation and reduces scale-sensitivity

### Weighted Token Voting

The current `win_index` path uses unweighted majority vote over `index[left:right]`.

For consistency, this should also become weighted.

For each token id `k` in the local window:

$$
V_k = \sum_{j \in W,\; index_j = k} \hat{w}_j
$$

Then choose:

$$
\text{vote\_token}_t = \arg\max_k V_k
$$

This keeps the existing token-vote interpretation but aligns it with the new span-aware weighting.

### Blank Handling

Do not introduce blank-specific weighting in the first version.

Reason:

- blank suppression is already handled later through repeat removal and blank removal
- adding blank penalties at the same time would blur attribution if WER changes

If needed later, blank penalty can be a second-stage extension.

## Layer 2: Confidence Fallback

### Problem

`window_greedy_7` is the best current default for adaptive mode, but not every sample benefits from the same smoothing strength.

Failure patterns likely include:

- sparse adaptive windows in fast transitions
- low-confidence windows where smoothing amplifies the wrong token
- short sequences where larger window smoothing is unnecessarily strong

### Fallback Strategy Levels

There are two possible fallback levels.

#### Level A: Local Decode Fallback

For each sample, compute outputs for a small decode ladder from the same logits:

1. `window_greedy_7`
2. `window_greedy_5`
3. `naive_greedy`

Then choose the final sample hypothesis based on confidence and sparsity statistics.

This is the preferred first implementation because:

- no second forward pass is needed
- it only reuses already available logits and clip centers
- latency increase is small compared with re-running fixed `stride=1`

#### Level B: Sample Fallback To Fixed `stride=1`

If the sample is globally too sparse or unstable, rerun the sample with fixed `stride=1` and use the better-quality branch.

This should not be the default first implementation because:

- it requires an extra inference pass
- it complicates latency budgeting
- it is harder to roll out safely in online mode

Recommendation:

- implement Level A first
- keep Level B as an optional second-stage experiment behind a config flag

### Confidence Signals

The confidence fallback should use simple interpretable signals derived from existing tensors.

Per-position signals:

1. `top1_prob_t = max(p_t)`
2. `margin_t = top1_prob_t - top2_prob_t`
3. `window_clip_count_t = right - left`
4. `window_span_t = clip_centers[right-1] - clip_centers[left]` when `right > left + 1`, else `0`
5. `window_density_t = window_clip_count_t / max(window_span_t, 1)`

Per-sample aggregate signals:

1. `mean_top1_prob`
2. `p25_top1_prob`
3. `mean_margin`
4. `p25_margin`
5. `min_window_clip_count`
6. `mean_window_density`

These are enough for a first routing policy and easy to write into result files.

### Routing Policy

Use `window_greedy_7` as the primary method, then step down only when evidence says smoothing is too risky.

Recommended first policy:

1. start from `window_greedy_7`
2. if `p25_top1_prob < low_confidence_threshold`, fall back to `window_greedy_5`
3. if `p25_margin < low_margin_threshold`, fall back to `window_greedy_5`
4. if `min_window_clip_count < min_window_clips`, fall back to `window_greedy_5`
5. if the sample still fails a stricter low-confidence condition, fall back to `naive_greedy`

Suggested first thresholds:

- `low_confidence_threshold = 0.45`
- `low_margin_threshold = 0.15`
- `min_window_clips = 3`
- `very_low_confidence_threshold = 0.35`
- `very_low_margin_threshold = 0.08`

These values should be treated as starting points, not final tuned values.

### Why Sample-Level Routing First

The first implementation should choose one method per sample, not per token position.

Reason:

- simpler output semantics
- easier comparison against current result files
- easier debugging because each sample has a single chosen method
- lower risk of creating unstable token-level switching artifacts

Token-level routing can be considered later if sample-level routing is clearly beneficial but not sufficient.

## Proposed Configuration Surface

Add a post-processing config block rather than extending `adaptive_stride` itself.

Suggested YAML shape:

```yaml
postprocess:
  span_weighted_voting:
    enabled: false
    kernel: triangular
    min_weight: 0.05
  confidence_fallback:
    enabled: false
    primary_method: window_greedy_7
    secondary_method: window_greedy_5
    tertiary_method: naive_greedy
    low_confidence_threshold: 0.45
    very_low_confidence_threshold: 0.35
    low_margin_threshold: 0.15
    very_low_margin_threshold: 0.08
    min_window_clips: 3
    sample_fallback_to_fixed_stride1: false
```

Why keep it under `postprocess`:

- these are decoding choices, not window-generation choices
- it keeps adaptive stride config focused on clip placement only
- it allows the same post-processing to be evaluated on fixed stride later if desired

## Proposed Result Fields

The result writer should expose enough diagnostics to compare behavior before and after routing.

Per sample, add fields such as:

- `dynamic_greedy_gls_hyp`
- `dynamic_greedy_selected_method`
- `dynamic_greedy_mean_top1_prob`
- `dynamic_greedy_p25_top1_prob`
- `dynamic_greedy_mean_margin`
- `dynamic_greedy_p25_margin`
- `dynamic_greedy_min_window_clips`
- `dynamic_greedy_mean_window_density`
- `dynamic_greedy_trigger_flags`

If span-weighted voting is enabled, also store:

- `window_greedy_7_span_weighted_gls_hyp`
- `window_greedy_5_span_weighted_gls_hyp`

This keeps later error slicing simple.

## Execution Flow

The intended control flow inside `evaluation_slide(...)` is:

1. compute `clip_centers` from `sliding_windows(...)`
2. compute base per-clip logits as today
3. for each candidate method in `{window_greedy_7, window_greedy_5, naive_greedy}`:
   - build local neighborhoods using the current real-time-span rule
   - if span-weighted voting is enabled, aggregate with distance-aware weights
   - otherwise preserve current uniform behavior
   - produce hypothesis and confidence summary
4. apply sample-level fallback routing using the confidence summary
5. store both raw candidate outputs and the final chosen output

## Rollout Plan

### Phase 1: Span-Weighted Voting Only

Enable only span-weighted voting and leave fallback disabled.

Expected outcome:

- adaptive `window_greedy_7` should narrow the gap to fixed `stride=1`
- if not, then equal-weight averaging was not the main remaining issue

### Phase 2: Add Sample-Level Confidence Fallback

Keep span-weighted voting enabled and route between `window_greedy_7`, `window_greedy_5`, and `naive_greedy`.

Expected outcome:

- more robust handling of sparse or low-confidence samples
- reduced tail regressions even if mean improvement is modest

### Phase 3: Optional Fixed-Stride Safety Fallback

Only if adaptive still has a small set of severe regressions:

- detect globally risky samples
- rerun just those samples with fixed `stride=1`

This phase is optional because it adds real latency cost.

## Validation Plan

When implementation starts, validate in the following order.

### Functional Checks

1. With both features disabled, outputs must match current behavior.
2. With span-weighted voting enabled on fixed stride, output changes should be small and interpretable.
3. With adaptive stride enabled, logged weights should peak near the target center and decay toward edges.
4. Fallback selection should be deterministic for the same sample and config.

### Quality Checks

1. compare `window_greedy_7` uniform vs span-weighted on the current best adaptive preset
2. compare dynamic fallback vs fixed `window_greedy_7`
3. inspect the worst regression samples and verify whether trigger flags align with intuition

### Decision Criteria

Promote the design only if it achieves at least one of:

1. lowers adaptive test WER below the current `22.33`
2. reduces the worst-sample regressions without materially harming the mean
3. improves adaptive latency-quality tradeoff enough to justify default runtime use

## Risks And Non-Goals

Risks:

1. weighted voting may become too center-biased and lose useful context
2. fallback thresholds may overfit to Phoenix and not transfer cleanly
3. storing too many diagnostics may enlarge result files noticeably

Non-goals for the first implementation:

1. no retraining
2. no token-level dynamic method switching
3. no blank-specific penalties
4. no beam-search redesign

## Recommended First Implementation Order

1. add reusable helper functions for span weights and confidence summary
2. implement span-weighted probability aggregation for `window_greedy_7`
3. extend the same weighting rule to weighted token voting
4. expose confidence summaries in result outputs
5. add sample-level routing across `window_greedy_7`, `window_greedy_5`, and `naive_greedy`
6. only then evaluate whether fixed `stride=1` fallback is needed

## Expected Outcome

If the current hypothesis is correct, this design should recover part of the remaining `0.47` WER gap without retraining by:

- preserving stronger local evidence inside irregular adaptive neighborhoods
- reducing damage from over-smoothing on sparse or uncertain samples
- keeping the best current adaptive preset usable as the low-latency branch