#!/usr/bin/env python3
"""CPU-only selective residual predictor on the fresh terminal-block split.

The policy keeps the fixed-center bonus unless a source-group OOF-selected raw
score threshold says that offset 1 or 3 has positive terminal advantage.  The
evaluation partition is exploratory because earlier experiments opened it.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_fresh_terminal_listwise_exploratory_v1_49faacc3"
DATA = FRESH.with_name(FRESH.name + "_dataset")
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_selective_residual_exploratory_v1_49faacc3"
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
SEED = 261022
FOLDS = 5
EPOCHS = 20
BATCH = 128
FALSE_OVERRIDE_COST = 4.0
COVERAGES = (0.005, 0.01, 0.02, 0.05, 0.10)


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FRESHMOD = imp("analyze_phoenix_partial_fresh_terminal_listwise")
PREVIEW = FRESHMOD.PREVIEW
BLOCK, CHRON, BUILDER = FRESHMOD.BLOCK, FRESHMOD.CHRON, FRESHMOD.BUILDER


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_spec():
    return json.loads((FRESH / "resolved_config_preregistered.json").read_text())["split"]


def source_fold(source):
    value = hashlib.sha256(f"{SEED}\0{source}".encode()).hexdigest()
    return int(value[:16], 16) % FOLDS


def config():
    return {
        "experiment": "selective/abstaining residual terminal-advantage predictor",
        "classification": "exploratory partial32; evaluation was opened by earlier experiments",
        "created_before_loading_calibration_or_evaluation_policy_outcomes": True,
        "cpu_only": True,
        "split": split_spec(),
        "data": {
            "terminal_rows": str(DATA),
            "preview_archive": str(SEQUENCES),
            "fit_samples": 2541,
            "fit_sources": 235,
            "calibration_samples": 459,
            "calibration_sources": 50,
            "evaluation_samples": 432,
            "evaluation_sources": 46,
        },
        "protocol": {
            "block_offsets": [0, 1, 2, 3],
            "hard_skeleton_offset": 0,
            "default_bonus_offset": 2,
            "eligible_override_offsets": [1, 3],
            "exactly_one_bonus_per_complete_block": True,
            "decision": "after offset3 arrives",
            "bounded_lookahead_candidate_arrivals": 2,
            "unknown_EOS": True,
            "tail_topup": False,
            "target_rate": "approximately 50%",
            "max_gap": 3,
            "decoder": "fixed span15",
        },
        "inputs": {
            "allowed": "bookkeeping + past-selected decoder prefix + each candidate 31x17 causal pose/hand preview",
            "preview_visible_through": "candidate start + 8",
            "forbidden": ["full ISLR logits", "reference", "future", "EOS", "terminal outcome"],
        },
        "target": {
            "definition": "center terminal errors minus side-candidate terminal errors under fixed-center continuation",
            "values": "signed integer: negative, zero, or positive",
            "future_reference_EOS": "label and audit only",
        },
        "model": {
            "architecture": "shared old 3-layer causal TCN candidate scorer; residual is side score minus center score",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "epochs": EPOCHS,
            "batch": BATCH,
            "training_blocks": "terminal-informative blocks only; both side residuals",
            "loss": "SmoothL1 signed-advantage regression; multiply examples with true advantage <=0 and predicted advantage >0 by 4",
            "false_override_cost": FALSE_OVERRIDE_COST,
            "seed": SEED,
        },
        "oof_selection": {
            "folds": FOLDS,
            "group": "source_video_id",
            "assignment": "sha256(seed+NUL+source) mod 5",
            "candidate_override_coverages": list(COVERAGES),
            "action": "override with higher predicted side residual only if it exceeds the frozen raw score threshold and zero",
            "qualification": "aggregate terminal reward >0; every fold reward >=0; at least 20 overrides",
            "selection": "maximum aggregate reward; tie chooses lower nominal coverage",
            "empty_policy": "zero overrides, reward 0, fixed-center regret",
        },
        "calibration_gate": [
            "frozen cumulative terminal reward > 0",
            "frozen total terminal regret < fixed-center total regret",
            "override precision and count reported without threshold adjustment",
        ],
        "evaluation": "only after calibration gate; one frozen run versus fixed center",
        "strong_go": "evaluation delta WER <= -0.5 percentage points and paired bootstrap CI upper < 0",
        "forbidden": ["GPU/CUDA", "dev", "test", "evaluation-dependent tuning", "post-evaluation model/coverage changes", "git commit"],
    }


def preregister(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    value = config()
    (root / "resolved_config_preregistered.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def loadcfg(root):
    value = json.loads((root / "resolved_config_preregistered.json").read_text())
    if value != config():
        raise ValueError("preregistered configuration changed")
    return value


def load_rows():
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    out = {"fit": [], "calibration": []}
    for info in manifest["workers"]:
        path = DATA / "workers" / f"worker-{info['worker']:02d}-of-{info['workers']:02d}.jsonl.gz"
        if sha(path) != info["sha256"]:
            raise ValueError("terminal dataset hash mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                out[row["partition"]].append(row)
    return out, manifest


def raw_arrays(rows, sequences):
    base, visual, errors, samples, sources, identities = [], [], [], [], [], []
    for row in rows:
        base.append(np.asarray([f["bookkeeping"] + f["prefix"] for f in row["features"]], np.float32))
        visual.append(np.stack([sequences.history(row["sample_id"], s) for s in row["candidate_starts"]]))
        errors.append(row["label"]["terminal_errors"])
        samples.append(row["sample_id"])
        sources.append(row["source_video_id"])
        identities.append(f"{row['sample_id']}:{row['block_start']}")
    errors = np.asarray(errors, np.float32)
    return {
        "base": np.stack(base),
        "visual": np.stack(visual).astype(np.float16),
        "errors": errors,
        "advantage": errors[:, 1:2] - errors[:, [0, 2]],
        "informative": np.any(errors != errors[:, 1:2], axis=1),
        "sample": np.asarray(samples),
        "source": np.asarray(sources),
        "identity": np.asarray(identities),
    }


def normalization(data, mask):
    base = data["base"][mask]
    bm = base.mean((0, 1), dtype=np.float64).astype(np.float32)
    bs = base.std((0, 1), dtype=np.float64).astype(np.float32)
    bs[bs < 1e-6] = 1
    visual = data["visual"][mask]
    vm = visual.mean((0, 1, 2), dtype=np.float64).astype(np.float32)
    vs = visual.std((0, 1, 2), dtype=np.float64).astype(np.float32)
    vs[vs < 1e-6] = 1
    return {"base_mean": bm, "base_scale": bs, "visual_mean": vm, "visual_scale": vs}


def batch_tensors(data, idx, stats):
    base = (data["base"][idx] - stats["base_mean"]) / stats["base_scale"]
    visual = (data["visual"][idx] - stats["visual_mean"]) / stats["visual_scale"]
    return torch.from_numpy(base).float(), torch.from_numpy(visual).float()


def residual_scores(model, base, visual):
    scores = model(base, visual)
    return scores[:, [0, 2]] - scores[:, 1:2]


def train_one(data, train_mask, seed):
    stats = normalization(data, train_mask)
    idx = np.flatnonzero(train_mask & data["informative"])
    if not len(idx):
        raise ValueError("no informative blocks")
    torch.manual_seed(seed)
    model = PREVIEW.Selector(data["base"].shape[2], data["visual"].shape[3], True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(EPOCHS):
        order = idx[torch.randperm(len(idx), generator=generator).numpy()]
        total = 0.0
        for left in range(0, len(order), BATCH):
            batch = order[left:left + BATCH]
            base, visual = batch_tensors(data, batch, stats)
            pred = residual_scores(model, base, visual)
            target = torch.from_numpy(data["advantage"][batch]).float()
            per = F.smooth_l1_loss(pred, target, reduction="none")
            false_override = (target <= 0) & (pred > 0)
            weights = torch.where(false_override, torch.full_like(per, FALSE_OVERRIDE_COST), torch.ones_like(per))
            loss = (per * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        history.append(total / len(order))
    model.eval()
    return model, stats, history, int(len(idx))


def predict(model, data, mask, stats):
    idx = np.flatnonzero(mask)
    out = []
    with torch.no_grad():
        for left in range(0, len(idx), 1024):
            part = idx[left:left + 1024]
            base, visual = batch_tensors(data, part, stats)
            out.append(residual_scores(model, base, visual).numpy())
    return idx, np.concatenate(out) if out else np.empty((0, 2), np.float32)


def select_actions(scores, threshold):
    side = np.argmax(scores, axis=1)
    confidence = scores[np.arange(len(scores)), side]
    override = (confidence >= threshold) & (confidence > 0)
    chosen = np.where(override, np.where(side == 0, 0, 2), 1)
    return chosen, override, confidence


def policy_metrics(data, scores, threshold, folds=None):
    chosen, override, confidence = select_actions(scores, threshold)
    errors = data["errors"]
    picked = errors[np.arange(len(errors)), chosen]
    center = errors[:, 1]
    minimum = errors.min(1)
    reward = center - picked
    positive = reward[override] > 0
    result = {
        "blocks": int(len(errors)),
        "threshold": float(threshold),
        "overrides": int(override.sum()),
        "realized_override_coverage": float(override.mean()),
        "override_precision": float(positive.mean()) if override.any() else None,
        "positive_overrides": int(positive.sum()),
        "harmful_overrides": int(np.sum(reward[override] < 0)),
        "neutral_overrides": int(np.sum(reward[override] == 0)),
        "cumulative_terminal_reward": int(reward.sum()),
        "total_terminal_regret": int(np.sum(picked - minimum)),
        "fixed_center_total_terminal_regret": int(np.sum(center - minimum)),
        "choice_counts": {str(k): int(v) for k, v in zip(*np.unique(chosen, return_counts=True))},
        "confidence_selected_min": float(confidence[override].min()) if override.any() else None,
    }
    if folds is not None:
        result["fold_reward"] = {str(f): int(reward[folds == f].sum()) for f in range(FOLDS)}
        result["fold_overrides"] = {str(f): int(override[folds == f].sum()) for f in range(FOLDS)}
    return result


def empty_metrics(data, folds=None):
    scores = np.full((len(data["errors"]), 2), -np.inf, np.float32)
    return policy_metrics(data, scores, math.inf, folds)


def candidate_threshold(scores, coverage):
    confidence = scores.max(1)
    positive = np.sort(confidence[confidence > 0])[::-1]
    target = max(1, int(math.ceil(coverage * len(confidence))))
    return float(positive[target - 1]) if len(positive) >= target else math.inf


def oof_select(data):
    folds = np.asarray([source_fold(s) for s in data["source"]], np.int64)
    oof = np.full((len(data["errors"]), 2), np.nan, np.float32)
    fold_training = {}
    histories = {}
    for fold in range(FOLDS):
        train_mask = folds != fold
        held = folds == fold
        model, stats, history, informative = train_one(data, train_mask, SEED + fold)
        idx, score = predict(model, data, held, stats)
        oof[idx] = score
        histories[str(fold)] = history
        fold_training[str(fold)] = {
            "train_sources": int(len(set(data["source"][train_mask]))),
            "heldout_sources": int(len(set(data["source"][held]))),
            "train_blocks": int(train_mask.sum()),
            "heldout_blocks": int(held.sum()),
            "informative_train_blocks": informative,
        }
    if np.isnan(oof).any():
        raise RuntimeError("incomplete OOF predictions")
    candidates = []
    for coverage in COVERAGES:
        threshold = candidate_threshold(oof, coverage)
        report = policy_metrics(data, oof, threshold, folds)
        qualifies = (
            report["cumulative_terminal_reward"] > 0
            and all(v >= 0 for v in report["fold_reward"].values())
            and report["overrides"] >= 20
        )
        candidates.append({"nominal_coverage": coverage, "qualifies": qualifies, **report})
    passing = [x for x in candidates if x["qualifies"]]
    selected = sorted(passing, key=lambda x: (-x["cumulative_terminal_reward"], x["nominal_coverage"]))[0] if passing else None
    return selected, candidates, oof, folds, fold_training, histories


def train(root):
    cfg = loadcfg(root)
    if (root / "calibration_started.marker").exists() or (root / "evaluation_started.marker").exists():
        raise RuntimeError("later partition already opened")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    rows, manifest = load_rows()
    sequences = PREVIEW.PreviewArchive(SEQUENCES)
    fit = raw_arrays(rows["fit"], sequences)
    selected, candidates, _, folds, fold_training, histories = oof_select(fit)
    result = {
        "stage": "fit-only source-group 5-fold OOF",
        "dataset_manifest": manifest,
        "fit": {"samples": int(len(set(fit["sample"]))), "sources": int(len(set(fit["source"]))), "blocks": len(fit["errors"]), "informative_blocks": int(fit["informative"].sum())},
        "empty_policy": empty_metrics(fit, folds),
        "candidates": candidates,
        "selected": selected,
        "fold_training": fold_training,
        "training_history": histories,
    }
    (root / "oof_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    if selected is None:
        status = {
            "status": "stopped_at_fit_oof_gate",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "no preregistered coverage met positive aggregate reward, nonnegative reward in every fold, and >=20 overrides",
            "calibration_outcomes_read": False,
            "evaluation_outcomes_read": False,
            "oof_metrics_sha256": sha(root / "oof_metrics.json"),
        }
        (root / "metrics.json").write_text(json.dumps({**result, "calibration": None, "evaluation": None, "decision": status["reason"]}, indent=2) + "\n")
        (root / "manifest.json").write_text(json.dumps(status, indent=2) + "\n")
        return result
    model, stats, history, informative = train_one(fit, np.ones(len(fit["errors"]), bool), SEED + 100)
    path = root / "residual_model.pt"
    torch.save({
        "state_dict": model.state_dict(), "stats": stats, "base_width": fit["base"].shape[2],
        "visual_width": fit["visual"].shape[3], "threshold": selected["threshold"],
        "coverage": selected["nominal_coverage"], "history": history,
    }, path)
    frozen = {
        "status": "model_and_threshold_frozen_before_calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha(path),
        "oof_metrics_sha256": sha(root / "oof_metrics.json"),
        "selected_coverage": selected["nominal_coverage"],
        "selected_threshold": selected["threshold"],
        "full_fit_informative_blocks": informative,
        "calibration_outcomes_read": False,
        "evaluation_outcomes_read": False,
    }
    (root / "training_manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return result


def load_model(root):
    manifest = json.loads((root / "training_manifest.json").read_text())
    path = root / "residual_model.pt"
    if manifest["status"] != "model_and_threshold_frozen_before_calibration" or sha(path) != manifest["model_sha256"]:
        raise ValueError("model not frozen")
    saved = torch.load(path, map_location="cpu")
    model = PREVIEW.Selector(saved["base_width"], saved["visual_width"], True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def calibrate(root):
    loadcfg(root)
    if (root / "evaluation_started.marker").exists():
        raise RuntimeError("evaluation already opened")
    model, saved = load_model(root)
    (root / "calibration_started.marker").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    rows, _ = load_rows()
    sequences = PREVIEW.PreviewArchive(SEQUENCES)
    cal = raw_arrays(rows["calibration"], sequences)
    idx, scores = predict(model, cal, np.ones(len(cal["errors"]), bool), saved["stats"])
    if len(idx) != len(cal["errors"]):
        raise RuntimeError("calibration prediction coverage mismatch")
    frozen = policy_metrics(cal, scores, saved["threshold"])
    empty = empty_metrics(cal)
    checks = {
        "cumulative_reward_gt_0": frozen["cumulative_terminal_reward"] > 0,
        "regret_lt_center": frozen["total_terminal_regret"] < frozen["fixed_center_total_terminal_regret"],
    }
    result = {"stage": "calibration with frozen model and raw threshold", "frozen_policy": frozen, "empty_policy": empty, "gate": {"passed": all(checks.values()), "checks": checks}}
    (root / "calibration_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    training = json.loads((root / "training_manifest.json").read_text())
    training["status"] = "calibration_passed_model_frozen" if result["gate"]["passed"] else "stopped_at_calibration_gate"
    training["calibration_outcomes_read"] = True
    training["calibration_metrics_sha256"] = sha(root / "calibration_metrics.json")
    (root / "training_manifest.json").write_text(json.dumps(training, indent=2) + "\n")
    if not result["gate"]["passed"]:
        oof = json.loads((root / "oof_metrics.json").read_text())
        (root / "metrics.json").write_text(json.dumps({**oof, "calibration": result, "evaluation": None, "decision": "calibration gate failed; evaluation not read"}, indent=2) + "\n")
        (root / "manifest.json").write_text(json.dumps({"status": "stopped_at_calibration_gate", "created_utc": datetime.now(timezone.utc).isoformat(), "evaluation_outcomes_read": False}, indent=2) + "\n")
    return result


def live_feature(name, probabilities, selected, candidates, blank, sequences, stats):
    fake = {
        "sample_id": name,
        "candidate_starts": candidates,
        "features": BLOCK.candidate_features(probabilities, selected, candidates, blank),
        "label": {"terminal_errors": [0, 0, 0]},
    }
    raw = raw_arrays([fake], sequences)
    base, visual = batch_tensors(raw, np.asarray([0]), stats)
    return base, visual


def run_sample(name, result, logits, model, saved, sequences, vocab, blank):
    probabilities = BUILDER.ORACLE.softmax_rows(np.asarray(logits))
    reference = BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    selected, audit = [], []
    threshold = saved["threshold"]
    for start in range(0, len(probabilities), 4):
        selected.append(start)
        if start + 3 >= len(probabilities):
            continue
        candidates = [start + 1, start + 2, start + 3]
        base, visual = live_feature(name, probabilities, selected, candidates, blank, sequences, saved["stats"])
        with torch.no_grad():
            score = residual_scores(model, base, visual).numpy()[0]
        side = int(np.argmax(score))
        override = bool(score[side] >= threshold and score[side] > 0)
        choice = (0 if side == 0 else 2) if override else 1
        true_reward = 0
        if override:
            future = BLOCK.center_continuation(len(probabilities), start + 4)
            side_error = BLOCK.decode_counts(probabilities, selected + [candidates[choice]] + future, reference, vocab, blank)["error"]
            center_error = BLOCK.decode_counts(probabilities, selected + [candidates[1]] + future, reference, vocab, blank)["error"]
            true_reward = int(center_error - side_error)
        selected.append(candidates[choice])
        audit.append({"override": override, "choice": choice, "score": float(score[side]), "true_terminal_reward": true_reward})
    return {
        "sample_id": name,
        "selected": selected,
        "counts": BLOCK.decode_counts(probabilities, selected, reference, vocab, blank),
        "dense_windows": len(probabilities),
        "coverage": CHRON.coverage_metrics(selected, len(probabilities)),
        "audit": audit,
    }


def evaluate(root):
    cfg = loadcfg(root)
    training = json.loads((root / "training_manifest.json").read_text())
    if training["status"] != "calibration_passed_model_frozen":
        raise RuntimeError("calibration gate failed; evaluation forbidden")
    model, saved = load_model(root)
    (root / "evaluation_started.marker").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    ids = set(cfg["split"]["partitions"]["evaluation"]["sample_ids"])
    sequences = PREVIEW.PreviewArchive(SEQUENCES)
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank = vocab.index("<blank>")
    predicted, uniform = [], []
    dense = CHRON.DEFAULT_DENSE_ROOT
    indices, count = CHRON.completed_shard_indices(dense)
    for shard in indices:
        results, logits, _ = BUILDER.BUILDER.validate_dense_shard(dense, shard, count, verify_hashes=False)
        for name in results:
            if name not in ids:
                continue
            predicted.append(run_sample(name, results[name], logits[name], model, saved, sequences, vocab, blank))
            base = BLOCK.oracle_sample(name, results[name], logits[name], "evaluation", vocab, blank, collect_rows=False)
            uniform.append({"sample_id": name, **base["uniform"], "dense_windows": base["dense_windows"]})
    if len(predicted) != len(ids):
        raise RuntimeError("evaluation coverage mismatch")
    by_name = {x["sample_id"]: x for x in uniform}
    summary = BLOCK.summarize(predicted, [by_name[x["sample_id"]] for x in predicted])
    audit = [a for row in predicted for a in row["audit"]]
    overrides = [a for a in audit if a["override"]]
    reward = sum(a["true_terminal_reward"] for a in overrides)
    precision = sum(a["true_terminal_reward"] > 0 for a in overrides) / len(overrides) if overrides else None
    delta = summary["delta_wer_pp_vs_structured_uniform"]
    ci = summary["paired_bootstrap_vs_structured_uniform"]["ci95"]
    result = {
        "stage": "one exploratory evaluation with frozen model and threshold",
        "samples": len(predicted),
        "predicted_policy": summary,
        "empty_policy_fixed_center": BLOCK.summarize(uniform, uniform),
        "override": {"blocks": len(audit), "count": len(overrides), "coverage": len(overrides) / len(audit), "precision": precision, "cumulative_terminal_reward": reward, "harmful": sum(a["true_terminal_reward"] < 0 for a in overrides), "neutral": sum(a["true_terminal_reward"] == 0 for a in overrides)},
        "budget_and_coverage": {"actual_window_rate": summary["actual_window_rate"], "max_gap": summary["max_gap"]},
        "errors_DIS": summary["metrics"],
        "decision": {"strong_go": delta <= -0.5 and ci[1] < 0, "delta_wer_pp": delta, "ci95": ci},
    }
    oof = json.loads((root / "oof_metrics.json").read_text())
    calibration = json.loads((root / "calibration_metrics.json").read_text())
    final = {**oof, "calibration": calibration, "evaluation": result, "limitations": ["32/56 non-random train shards", "evaluation partition previously opened", "bounded-lookahead-2", "cached preview excludes detector wall time"]}
    (root / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    (root / "manifest.json").write_text(json.dumps({"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(), "metrics_sha256": sha(root / "metrics.json"), "model_sha256": sha(root / "residual_model.pt")}, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preregister", "train", "calibrate", "evaluate"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = args.output_root.resolve()
    if args.mode == "preregister":
        result = preregister(root)
    elif args.mode == "train":
        result = train(root)
    elif args.mode == "calibrate":
        result = calibrate(root)
    else:
        result = evaluate(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
