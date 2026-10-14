# Online Sign Language Recognition for iOS

This repository builds on [FangyunWei/SLRT](https://github.com/FangyunWei/SLRT) and supports research on
real-time sign-language-video-to-text for iOS. Current development focuses on `Online/CSLR`: sliding-window
continuous recognition with Two-Stream S3D, equal-budget temporal scheduling, and the deployability of
decoder-aware window selection.

The original SLRT papers, citations, and subproject links are preserved in
[`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md). The changes and experimental findings in this repository
are not official results from the upstream authors.

## Current Status

Phoenix-2014T is the primary online CSLR track and uses WER as its metric. CSL-Daily Top-800 is an isolated
reliability diagnostic and uses accuracy/AUROC. Results from these two data tracks must not be mixed.

| Stage | Task | Frozen conclusion |
|---|---|---|
| A0 | Motion-only adaptive stride | Reduced clips by 32.56% relative to B0 on dev; did not significantly outperform equal-budget uniform sampling |
| P0 | Equal-budget scheduling diagnostics | Boundary/prediction-change: No-Go; label-derived sign center: offline Strong-Go |
| P1 | Minimal train-only causal center predictor | Failed the calibration gate: No-Go; scheduler not run and no new WER reported |
| P2 | 211-dimensional keypoints + 31-frame causal TCN | Detection improved substantially but failed the gate: No-Go; dev/test remained unopened |
| P3 | Robust decoder-utility oracle and deployable predictor | Strong full-train oracle upper bound, but skeleton, three RGB variants, decoder-conditioned value, and the information ladder all missed the OOF gate; work on the current per-window utility target stopped |

The P0 sign-center result is an alignment-derived offline oracle, not a deployable model. P1 and P2 both failed
the center gate. On the full training fit, P3 confirmed that the robust continuation oracle could eliminate 637
errors, but only 769 of 366,802 candidate rows were beneficial. None of the frozen skeleton, RGB-DCT, finger RGB,
face RGB, decoder-conditioned value, or staged information-enhancement variants approached the 249-error gate.
This does not mean that visual information is useless; it means that the existing future/reference/EOS-aware
per-window counterfactual utility target is too weakly coupled to information available at deployment time.
Consequently, closed-loop evaluation is not run, hyperparameters are not retuned on the same OOF set, and the
next stage first redesigns the target/decision decomposition to be more stable and observable.

## Start Here

- [Documentation index](docs/README.md): separates current entry points, frozen results, historical designs, and machine-readable indexes.
- [Next-stage handoff](docs/NEXT_RESEARCH_HANDOFF.md): the single current task entry point for a new work session.
- [Unified results](docs/RESULTS.md): cross-stage numbers and permitted claims.
- [Complete experiment ledger](docs/ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md): P0--P3 facts, risks, and machine-artifact indexes.
- [P7 temporal-input report](docs/P7_TEMPORAL_INPUT_SOLUTION.md): fixed-length input geometry, alignment fixes, preregistered comparisons, and hardware incidents.
- [Environment and reproducibility](docs/REPRODUCIBILITY.md): data, checkpoints, GPUs, and reproduction commands.
- [Repository layout policy](docs/REPOSITORY_LAYOUT.md): where files belong.
- [Experiment-log index](code_agent_logs/README.md): implementation and audit trail organized by date.

Frozen stage documents:

- [A0 adaptive stride v1](docs/ADAPTIVE_BASELINE_V1.md)
- [P0 scheduling diagnostics v1](docs/P0_SCHEDULE_DIAGNOSTICS_V1.md)
- [P1 causal center predictor v1](docs/P1_CAUSAL_CENTER_PREDICTOR_V1.md)
- [P2 causal center TCN v1](docs/P2_CAUSAL_CENTER_TCN_V1.md)
- [P3 complete experiment ledger](docs/ONLINE_CSLR_EXPERIMENT_LEDGER_20260920.md)
- [P7 temporal-input solution](docs/P7_TEMPORAL_INPUT_SOLUTION.md)

## Local Path

Use the canonical path on this machine:

```text
/mnt/workspace/projects/haojun/SLRT
```

`/home/haojun/projects` is a symbolic link to `/mnt/workspace/projects/haojun`, so
`/home/haojun/projects/SLRT` and the path above refer to the same repository, not two copies. Open the `SLRT`
directory directly in VS Code Remote. Do not use `/home/haojun` or the outer
`/mnt/workspace/projects/haojun` directory as the project root, because doing so mixes user configuration and
historical scripts outside the repository into the Explorer view.

## Minimal Verification

Run from the repository root:

```bash
cd /mnt/workspace/projects/haojun/SLRT/Online/CSLR
python -m unittest discover -s tests -p 'test_*.py' -v
```

All `nvidia-smi` queries, CUDA checks, and GPU experiments must run outside the sandbox. CUDA being invisible
inside the sandbox does not indicate a driver failure. Verify devices by PCI bus ID/UUID and bind them
explicitly. Avoid PCI `25:00.0`, `41:00.0`, and host GPU 0 (currently `01:00.0`); do not rely on logical GPU
indexes, which can change.

## Data and Artifact Boundaries

- Keep `data/`, videos, split keypoint pickles, checkpoints, logits, and large per-sample results local only.
- When freezing a stage, commit only compact configs, manifests, aggregates, required small models, and machine-readable indexes through precise `.gitignore` allowlists.
- Use train for fitting and internal calibration. Evaluate dev only after the training-side plan is frozen. The test split is currently prohibited.
- Use a new output directory for every experiment; never overwrite A0, P0, P1, or P2.

## Repository Overview

```text
SLRT/
├── Online/CSLR/       # Primary code, configs, tools, tests, and local results
├── docs/              # Current handoff, frozen conclusions, reproducibility, and machine indexes
├── scripts/reproduce/ # Stable reproduction scripts
├── code_agent_logs/   # Implementation and experiment audits organized by date
├── data/              # Local data; excluded from Git by default
└── CiCo, NLA-SLR, Spoken2Sign, TwoStreamNetwork, and other upstream subprojects
```
