# Future Research Directions for Top-800 Online CSLR

Date: 2026-05-29
Repository: /home/haojun/projects/SLRT

## Goal

This document records possible future research paths for the current top-800 online CSLR project. The focus is not to summarize recent papers, but to decide what can become our own project contribution.

Current baseline:
- Train an 800-gloss ISLR model on isolated/sign-centered top-800 data.
- Apply it to continuous videos with sliding-window online prediction.
- Use RGB stream, keypoint stream, fused stream, and an ensemble prediction.
- Current pipeline is practical and easy to debug, but the sliding-window stage can be redundant, blank-heavy, and weak at boundary handling.

## What A Good Main Contribution Should Do

A paper-level improvement should satisfy most of these:
- It improves both recognition quality and online efficiency, not just one isolated metric.
- It directly addresses a weakness of the current ISLR + sliding-window pipeline.
- It can be evaluated with the data we already have.
- It does not require expensive LLM fine-tuning or a complete rewrite before we have a strong baseline.
- It is different enough from existing online CSLR and LLM-context work.

## Direction A: Temporal Boundary-Aware ISLR For Online CSLR

Core idea:
- Keep the ISLR recognizer, but make it predict more than "which gloss is in this window".
- Add temporal awareness: whether the current window contains a valid sign, where the sign center is, and how long the sign probably lasts.
- Use these predictions during online decoding to decide when to emit a gloss, when to skip forward, and when to merge duplicate windows.

Why it fits our project:
- The current data already contains useful temporal fields such as `start`, `end`, `base_start`, `base_end`, and `temp_idx`.
- The current model already has a foreground/non-blank split path through `split_nonblk`.
- Sliding-window online CSLR suffers exactly from boundary ambiguity: many windows contain transitions, partial signs, or repeated views of the same sign.

Possible concrete changes:
- Add a small boundary/duration head on top of the fused feature.
- Train it to predict:
  - signness / non-blank probability
  - normalized center offset inside the window
  - normalized duration or valid span length
- During `prediction_slide.py`, use this output to:
  - suppress low-signness windows
  - merge windows that point to the same sign center
  - increase stride in low-motion or low-signness areas
  - reduce duplicate gloss emissions

Hard parts:
- Pseudo-boundaries may be noisy. If `base_start/base_end` are imperfect, stronger boundary supervision can teach wrong timing.
- A duration prediction that works on isolated training clips may not transfer perfectly to continuous online windows.
- If the decoder skips too aggressively, short signs can be missed.
- The evaluation must include both WER and online cost, otherwise the contribution becomes only a small auxiliary loss.

Feasibility:
- High.
- This is the strongest candidate for our main contribution because it uses existing data and directly improves the current online pipeline.

Best first experiment:
- Do not train a new model immediately.
- First log the distribution of `temp_idx`, sign span lengths, blank windows, repeated predictions, and window count on dev.
- Then add a lightweight boundary head and compare fixed stride vs boundary-aware decoding.

## Direction B: Context-Constrained Gloss Reranking Without LLM Fine-Tuning

Core idea:
- Keep the visual recognizer unchanged.
- Use top-k gloss hypotheses from each window or segment.
- Rerank them with lightweight context constraints from previous predicted glosses, without fine-tuning an LLM.

Why it fits our project:
- The current recognizer already produces top-k class probabilities.
- CSLR output is a gloss sequence, so local gloss context can help fix visually ambiguous predictions.
- This avoids the cost and risk of LLM fine-tuning while still adding a language/context dimension.

Possible concrete changes:
- Build a top-800 gloss n-gram or small neural gloss language model from the available training gloss sequences.
- During online decoding, combine:
  - visual score
  - blank penalty
  - repeat penalty
  - context score from recent gloss history
- Keep reranking constrained to the 800-word vocabulary, so the language model cannot hallucinate out-of-vocabulary words.

Hard parts:
- CSL-Daily sentence-level context is limited; a gloss language model may mostly learn common local phrases rather than deep meaning.
- If the visual model is weak, a language prior can over-correct and produce plausible but visually wrong glosses.
- Online decoding must avoid using future context if we want a fair streaming setting.
- If this is only an n-gram reranker, novelty may be too small unless combined with boundary/reliability mechanisms.

Feasibility:
- High for engineering.
- Medium as a standalone paper contribution.

Best first experiment:
- Add an offline dev reranker over existing `prediction_slide` outputs.
- Compare WER, blank rate, repeated-gloss rate, and examples before/after reranking.
- If the improvement is real, integrate it into online decoding with a limited history length.

## Direction C: Reliability-Aware Multimodal Fusion

Core idea:
- The current ensemble treats RGB, keypoint, and fuse probabilities too uniformly.
- Instead, estimate which stream is more reliable for each window and use that to weight the final prediction.

Why it fits our project:
- Current training logs show stream behavior can differ. Keypoint can outperform RGB, while the fused/ensemble result is not always best.
- RGB is sensitive to clothing, lighting, background, and camera quality.
- Keypoints are more stable against appearance changes, but can fail when pose extraction is noisy.
- A per-window reliability mechanism matches the real failure modes of online video.

Possible concrete changes:
- Start with zero-training reliability scores:
  - entropy of each stream prediction
  - max probability / confidence
  - blank probability
  - disagreement between streams
  - keypoint visibility or motion quality
- Then try a learned weight module if the heuristic version helps.
- Output:
  - weighted RGB/keypoint/fuse probability
  - optional confidence score used by the online decoder

Hard parts:
- A simple learned weighted average may be judged as too incremental.
- Confidence is often miscalibrated; high softmax confidence does not always mean correctness.
- If fusion hides a broken stream, it may improve metrics without improving understanding.
- Need careful ablation: RGB only, keypoint only, fuse only, equal ensemble, heuristic reliability, learned reliability.

Feasibility:
- High.
- Strong as a supporting contribution, especially when combined with Direction A.

Best first experiment:
- Compute per-stream correctness and confidence on dev.
- Check whether entropy/confidence/keypoint quality actually predicts correctness.
- If yes, implement reliability-weighted ensemble in `prediction_slide.py` before changing training.

## New Combined Direction: Reliability- And Boundary-Aware Online CSLR

This is currently the best research path for our project.

Proposed contribution:
- A lightweight online CSLR system that keeps the practical ISLR recognizer, but makes online decoding smarter with boundary prediction and reliability-aware fusion.

Main components:
- Boundary-aware ISLR:
  - predicts gloss, signness, center offset, and duration/span.
- Reliability-aware fusion:
  - dynamically weights RGB, keypoint, and fused predictions per window.
- Adaptive online decoding:
  - uses signness, duration, confidence, and stream disagreement to skip redundant windows and merge duplicates.
- Optional constrained gloss reranking:
  - uses top-k visual candidates and a small gloss-context model.
  - no LLM fine-tuning.

Why this is more original than small tweaks:
- It changes the unit of online decision-making from "classify every fixed window" to "detect reliable sign events in a stream".
- It targets the two main weaknesses of sliding-window ISLR:
  - too many redundant windows
  - weak boundary/duplicate handling
- It can report both accuracy and latency/computation curves.
- It avoids becoming a direct copy of heavy context-LLM methods.

## Lower-Priority High-Risk Directions

### Full CTC Streaming Model

Potential value:
- CTC is a natural fit for continuous gloss recognition without frame-level labels.
- It can model blank/repeat collapse more directly than ISLR classification.

Why not first:
- It would require turning the current ISLR pipeline into a continuous sequence model.
- It overlaps strongly with existing online CSLR work.
- Debugging CTC alignment, blank collapse, and streaming latency is harder than improving the current pipeline.

Possible use in our project:
- Use a CTC model as a teacher to generate pseudo-boundaries or duration labels.
- Do not replace the whole pipeline until the ISLR baseline and boundary-aware decoder are fully evaluated.

### Transducer-Style Streaming CSLR

Potential value:
- Transducer models are designed for streaming sequence prediction.
- Gloss order is mostly monotonic with video, so CSLR is more suitable than full SLT.

Why not first:
- This is a major architecture rewrite.
- There is less mature hand-sign recognition code to reuse.
- It needs a predictor/joiner training setup, new decoding, and new failure analysis.

Possible use in our project:
- Long-term research direction after the current online baseline is strong.
- Not recommended as the next implementation step.

### Discrete Action Tokenization

Potential value:
- Could create sub-gloss motion units and connect sign motion to language models more naturally.
- Could help with duration and sub-action boundaries.

Why not first:
- Learning stable discrete motion tokens is difficult.
- Token collapse and token interpretability are serious risks.
- Many related works target sign generation/production, not directly online CSLR.

Possible use in our project:
- A safer variant is to learn a small set of sub-action prototypes from keypoint/RGB features for boundary-aware decoding.
- Avoid building a full VQ tokenizer until we have stronger baselines.

## Recommended Roadmap

### Stage 0: Lock The Baseline

Tasks:
- Finish full top-800 ISLR training.
- Run continuous top-800 `prediction_slide` dev/test.
- Record:
  - WER
  - non-empty rate
  - blank rate
  - duplicate/repetition rate
  - average emitted gloss length
  - average number of windows per video
  - per-stream dev accuracy

Exit condition:
- We know exactly whether errors come from isolated recognition, online decoding, blank handling, or stream fusion.

### Stage 1: Zero-Training Decoding Experiments

Tasks:
- Try reliability-weighted ensemble using existing RGB/keypoint/fuse logits.
- Try adaptive stride using keypoint motion and confidence.
- Try constrained gloss reranking on saved prediction outputs.

Exit condition:
- At least one no-retrain method improves WER or reduces window count without hurting WER.

### Stage 2: Boundary-Aware ISLR

Tasks:
- Add signness/center/duration heads.
- Train with existing temporal labels or pseudo-labels.
- Modify online decoder to emit sign events rather than raw window predictions.

Exit condition:
- Better WER and fewer redundant windows than fixed sliding-window ISLR.

### Stage 3: Reliability-Aware Fusion

Tasks:
- Add learned or calibrated stream weights.
- Compare against equal ensemble and heuristic reliability.
- Report stream failure cases and qualitative examples.

Exit condition:
- Fusion improves robustness across videos where RGB and keypoint streams disagree.

### Stage 4: Combined System

Tasks:
- Combine boundary-aware decoding, reliability-aware fusion, adaptive stride, and optional context-constrained reranking.
- Report accuracy-latency tradeoff curves.

Exit condition:
- The system has a clear claim:
  - better online CSLR accuracy
  - less computation
  - lower duplicate/blank error
  - no LLM fine-tuning requirement

## Suggested Priority

1. Direction A: Temporal Boundary-Aware ISLR.
2. Direction C: Reliability-Aware Multimodal Fusion.
3. Direction B: Context-Constrained Gloss Reranking.
4. CTC teacher / pseudo-boundary support.
5. Transducer or discrete tokenization only as long-term exploration.

## Short Research Position

The project should not abandon ISLR + sliding-window immediately. Instead, the stronger paper path is to show that a lightweight ISLR-based online system can become much less naive by adding boundary awareness, reliability-aware multimodal fusion, and constrained context decoding.

This keeps the project feasible while moving the contribution beyond small engineering tweaks.
