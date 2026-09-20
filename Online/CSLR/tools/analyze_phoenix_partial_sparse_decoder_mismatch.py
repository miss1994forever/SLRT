#!/usr/bin/env python3
"""CPU-only sparse decoder mismatch diagnostic on the repeatedly used fixed-128 set.

This experiment deliberately stays inside the old fixed calibration subset.  It
never touches the fresh-disjoint replication and therefore has exploratory,
not confirmatory, status.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby, product
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_structured_block_smoke_v1_49faacc3"
BLOCK_DATA = BLOCK_ROOT.with_name(BLOCK_ROOT.name + "_dataset")
NONBLANK_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_nonblank_surrogate_closed_loop_exploratory_v1_49faacc3"
SEQUENCES = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
OUTPUT_ROOT = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_sparse_decoder_mismatch_diagnostic_v1_49faacc3"
SEED = 261021
SCHEDULES = ("fixed_center_uniform", "predicted_nonblank", "full_nonblank_oracle", "terminal_block_oracle")
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BLANK_BIASES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
SPAN_SENSITIVITY = (1.0, 7.0, 15.0, 23.0)


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BLOCK = imp("analyze_phoenix_partial_structured_block")
UPGRADE = imp("analyze_phoenix_partial_dense_upgrade_targets")
NONBLANK = imp("analyze_phoenix_partial_nonblank_closed_loop")
CHRON, BUILDER = BLOCK.CHRON, BLOCK.BUILDER


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_split(sample_ids):
    """Outcome-free, source-video-disjoint near-half split of the old 128."""
    groups = {}
    for sample_id in sample_ids:
        groups.setdefault(BUILDER.source_video(sample_id), []).append(sample_id)
    ordered = sorted(groups, key=lambda x: hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest())
    tune, audit = [], []
    for source in ordered:
        target = tune if len(tune) <= len(audit) else audit
        target.extend(sorted(groups[source]))
    return {"tune": sorted(tune), "audit": sorted(audit),
            "tune_source_videos": sorted({BUILDER.source_video(x) for x in tune}),
            "audit_source_videos": sorted({BUILDER.source_video(x) for x in audit})}


def config():
    block_config = json.loads((BLOCK_DATA / "resolved_config_preregistered.json").read_text())
    old_ids = block_config["subsets"]["calibration"]["sample_ids"]
    split = source_split(old_ids)
    return {
        "experiment": "exploratory fixed-128 sparse decoder mismatch diagnostic",
        "created_before_new_diagnostic_outcomes": True,
        "confirmatory": False,
        "cpu_only": True,
        "old_fixed128_only": True,
        "split": split,
        "schedules": list(SCHEDULES),
        "frozen_schedule_rule": "all four schedules are generated once with the existing policy/oracle definitions and never changed by decoder calibration",
        "decoder": {
            "baseline": {"temperature": 1.0, "blank_logit_bias": 0.0, "span": 15.0, "min_weight": 0.05},
            "temperature_grid": list(TEMPERATURES),
            "blank_logit_bias_grid": list(BLANK_BIASES),
            "selection": "minimize total tune edit errors; deterministic tie-break toward identity",
            "unified": "one parameter pair selected from pooled tune errors across all schedules",
            "schedule_specific": "one pair selected independently per schedule; diagnostic only, not a fair common-decoder comparison",
            "span_sensitivity": list(SPAN_SENSITIVITY),
        },
        "primary_questions": [
            "does one unified calibration improve both fixed-center uniform and predicted-nonblank on audit",
            "does calibration reduce predicted-nonblank minus oracle error gaps",
            "do schedule-specific optima materially differ, indicating decoder/schedule mismatch",
        ],
        "reported": ["WER", "deletions", "insertions", "substitutions", "output/reference length ratio",
                     "raw blank vote rate", "adjacent repeat rate", "singleton nonblank run rate",
                     "mean vote confidence", "mean vote margin", "mean vote entropy"],
        "forbidden": ["GPU/CUDA", "dev", "test", "fresh-disjoint replication IDs or outcomes",
                      "reference/future as deployed decoder input", "git commit"],
        "limitations": ["old fixed128 repeatedly inspected", "partial32 non-random shards", "exploratory only"],
    }


def preregister(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    value = config()
    (root / "resolved_config_preregistered.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def calibrated_probabilities(logits, temperature, blank_bias, blank_id):
    values = np.asarray(logits, dtype=np.float32) / np.float32(temperature)
    values = values.copy()
    values[:, blank_id] += np.float32(blank_bias)
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def detailed_decode(logits, starts, reference, vocab, blank_id, temperature=1.0,
                    blank_bias=0.0, span=15.0, min_weight=0.05):
    starts = np.asarray(sorted(int(x) for x in starts), dtype=np.int64)
    if len(starts) == 0:
        counts = CHRON.RANDOM.sentence_counts(reference, "")
        return {"hypothesis": "", "counts": counts, "stats": {"selected_windows": 0}}
    probabilities = calibrated_probabilities(logits, temperature, blank_bias, blank_id)
    distance = np.abs(starts[:, None] - starts[None, :]).astype(np.float32)
    radius = np.float32(span / 2.0)
    if span <= 1.0:
        weights = (distance == 0).astype(np.float32)
    else:
        mask = distance <= radius
        weights = np.where(mask, np.maximum(1.0 - distance / radius, min_weight), 0.0).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    scores = weights @ probabilities[starts]
    tokens = scores.argmax(axis=1)
    ordered = np.sort(scores, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -np.sum(scores * np.log(np.maximum(scores, 1e-12)), axis=1)
    runs = [(int(token), len(list(items))) for token, items in groupby(tokens.tolist())]
    nonblank_runs = [length for token, length in runs if token != blank_id]
    collapsed = [token for token, _ in runs if token != blank_id]
    raw_hypothesis = " ".join(BUILDER.REPLAY.map_phoenix_gloss(vocab[token]) for token in collapsed)
    hypothesis = BUILDER.REPLAY.clean_phoenix_2014_trans(raw_hypothesis)
    counts = CHRON.RANDOM.sentence_counts(reference, hypothesis)
    output = hypothesis.split() if hypothesis else []
    stats = {
        "selected_windows": int(len(starts)),
        "output_tokens": int(len(output)),
        "raw_blank_votes": int((tokens == blank_id).sum()),
        "raw_adjacent_repeats": int((tokens[1:] == tokens[:-1]).sum()),
        "raw_adjacencies": int(max(0, len(tokens) - 1)),
        "nonblank_runs": int(len(nonblank_runs)),
        "singleton_nonblank_runs": int(sum(length == 1 for length in nonblank_runs)),
        "output_adjacent_repeats": int(sum(left == right for left, right in zip(output, output[1:]))),
        "output_adjacencies": int(max(0, len(output) - 1)),
        "sum_confidence": float(confidence.sum()),
        "sum_margin": float(margin.sum()),
        "sum_entropy": float(entropy.sum()),
        "sum_selected_blank_probability": float(probabilities[starts, blank_id].sum()),
    }
    return {"hypothesis": hypothesis,
            "counts": {key: int(counts[key]) for key in ("error", "del", "ins", "sub", "ref_len")},
            "stats": stats}


def full_nonblank_schedule(probabilities, blank_id):
    chosen = []
    for start in range(0, len(probabilities), 4):
        chosen.append(start)
        if start + 3 >= len(probabilities):
            continue
        candidates = [start + 1, start + 2, start + 3]
        values = [1.0 - probabilities[candidate, blank_id] for candidate in candidates]
        best = max(range(3), key=lambda j: (values[j], j == 1, -j))
        chosen.append(candidates[best])
    return chosen


def load_nonblank_model():
    checkpoint = torch.load(NONBLANK_ROOT / "model.pt", map_location="cpu")
    width = int(len(checkpoint["stats"]["mean"]))
    model = UPGRADE.MLP(width)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["stats"]


def collect_samples(ids):
    blocks, samples, _ = BLOCK.load_rows(BLOCK_DATA)
    sample_by_name = {row["sample_id"]: row for row in samples["calibration"]}
    ids = set(ids)
    if ids != (ids & set(sample_by_name)):
        raise ValueError("old calibration IDs are missing")
    model, model_stats = load_nonblank_model()
    sequences = BLOCK.OLD.SequenceFeatures(SEQUENCES)
    vocab = json.loads(CHRON.DEFAULT_VOCAB.read_text())
    blank_id = vocab.index("<blank>")
    dense = CHRON.DEFAULT_DENSE_ROOT
    shard_indices, shard_count = CHRON.completed_shard_indices(dense)
    output = {}
    for shard in shard_indices:
        results, logits, _ = BUILDER.BUILDER.validate_dense_shard(dense, shard, shard_count, verify_hashes=False)
        for name in results:
            if name not in ids:
                continue
            raw_logits = np.asarray(logits[name], dtype=np.float32)
            probabilities = BUILDER.ORACLE.softmax_rows(raw_logits)
            reference = BUILDER.REPLAY.clean_phoenix_2014_trans(results[name]["gls_ref"])
            predicted = NONBLANK.run_sample(name, results[name], raw_logits, model, model_stats,
                                            sequences, vocab, blank_id)["selected"]
            stored = sample_by_name[name]
            output[name] = {
                "sample_id": name,
                "source_video_id": BUILDER.source_video(name),
                "reference": reference,
                "logits": raw_logits,
                "schedules": {
                    "fixed_center_uniform": stored["uniform"]["selected"],
                    "predicted_nonblank": predicted,
                    "full_nonblank_oracle": full_nonblank_schedule(probabilities, blank_id),
                    "terminal_block_oracle": stored["oracle"]["selected"],
                },
            }
    if set(output) != ids:
        raise RuntimeError(f"dense coverage mismatch: expected {len(ids)}, got {len(output)}")
    return output, vocab, blank_id


def aggregate(decoded):
    counts = Counter()
    stats = Counter()
    per_sample = []
    for sample_id, row in decoded:
        counts.update(row["counts"])
        stats.update(row["stats"])
        per_sample.append({"sample_id": sample_id, **row["counts"]})
    selected = max(1, stats["selected_windows"])
    ref_len = max(1, counts["ref_len"])
    nonblank_runs = max(1, stats["nonblank_runs"])
    return {
        "metrics": {"wer": 100.0 * counts["error"] / ref_len,
                    **{key: int(counts[key]) for key in ("error", "del", "ins", "sub", "ref_len")}},
        "sequence_stats": {
            "samples": len(per_sample),
            "selected_windows": int(stats["selected_windows"]),
            "output_tokens": int(stats["output_tokens"]),
            "output_to_reference_length_ratio": float(stats["output_tokens"] / ref_len),
            "raw_blank_vote_rate": float(stats["raw_blank_votes"] / selected),
            "raw_adjacent_repeat_rate": float(stats["raw_adjacent_repeats"] / max(1, stats["raw_adjacencies"])),
            "singleton_nonblank_run_rate": float(stats["singleton_nonblank_runs"] / nonblank_runs),
            "output_adjacent_repeat_rate": float(stats["output_adjacent_repeats"] / max(1, stats["output_adjacencies"])),
            "mean_vote_confidence": float(stats["sum_confidence"] / selected),
            "mean_vote_margin": float(stats["sum_margin"] / selected),
            "mean_vote_entropy": float(stats["sum_entropy"] / selected),
            "mean_selected_blank_probability": float(stats["sum_selected_blank_probability"] / selected),
        },
        "per_sample_counts": per_sample,
    }


def evaluate(data, names, schedule, params):
    rows = []
    for name in names:
        sample = data[name]
        decoded = detailed_decode(sample["logits"], sample["schedules"][schedule], sample["reference"],
                                  params["vocab"], params["blank_id"], params["temperature"],
                                  params["blank_bias"], params.get("span", 15.0), 0.05)
        rows.append((name, decoded))
    return aggregate(rows)


def identity_distance(temperature, blank_bias):
    return abs(float(np.log(temperature))) + abs(blank_bias)


def select_parameters(data, names, schedules, vocab, blank_id):
    grid = []
    for temperature, bias in product(TEMPERATURES, BLANK_BIASES):
        params = {"temperature": temperature, "blank_bias": bias, "span": 15.0,
                  "vocab": vocab, "blank_id": blank_id}
        errors = {schedule: evaluate(data, names, schedule, params)["metrics"]["error"] for schedule in schedules}
        grid.append({"temperature": temperature, "blank_bias": bias, "errors": errors,
                     "pooled_error": int(sum(errors.values()))})
    unified = min(grid, key=lambda x: (x["pooled_error"], identity_distance(x["temperature"], x["blank_bias"]),
                                       x["temperature"], x["blank_bias"]))
    specific = {}
    for schedule in schedules:
        specific[schedule] = min(grid, key=lambda x: (x["errors"][schedule],
                                                       identity_distance(x["temperature"], x["blank_bias"]),
                                                       x["temperature"], x["blank_bias"]))
    return unified, specific, grid


def strip_private(result):
    return {key: value for key, value in result.items() if key != "per_sample_counts"}


def paired_delta(candidate, baseline):
    by_candidate = {x["sample_id"]: x for x in candidate["per_sample_counts"]}
    by_baseline = {x["sample_id"]: x for x in baseline["per_sample_counts"]}
    names = sorted(by_candidate)
    delta_error = sum(by_candidate[n]["error"] - by_baseline[n]["error"] for n in names)
    ref_len = sum(by_candidate[n]["ref_len"] for n in names)
    return {"error_delta": int(delta_error), "wer_pp_delta": float(100.0 * delta_error / ref_len)}


def analyze(root):
    cfg = json.loads((root / "resolved_config_preregistered.json").read_text())
    if cfg != config():
        raise ValueError("preregistered configuration changed")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    all_ids = cfg["split"]["tune"] + cfg["split"]["audit"]
    data, vocab, blank_id = collect_samples(all_ids)
    unified, specific, grid = select_parameters(data, cfg["split"]["tune"], SCHEDULES, vocab, blank_id)
    base_params = {"temperature": 1.0, "blank_bias": 0.0, "span": 15.0, "vocab": vocab, "blank_id": blank_id}
    unified_params = {"temperature": unified["temperature"], "blank_bias": unified["blank_bias"],
                      "span": 15.0, "vocab": vocab, "blank_id": blank_id}
    audit = {"baseline": {}, "unified_calibration": {}, "schedule_specific_calibration": {}, "span_sensitivity": {}}
    private = {key: {} for key in ("baseline", "unified_calibration", "schedule_specific_calibration")}
    for schedule in SCHEDULES:
        private["baseline"][schedule] = evaluate(data, cfg["split"]["audit"], schedule, base_params)
        private["unified_calibration"][schedule] = evaluate(data, cfg["split"]["audit"], schedule, unified_params)
        chosen = specific[schedule]
        own_params = {"temperature": chosen["temperature"], "blank_bias": chosen["blank_bias"],
                      "span": 15.0, "vocab": vocab, "blank_id": blank_id}
        private["schedule_specific_calibration"][schedule] = evaluate(data, cfg["split"]["audit"], schedule, own_params)
        for family in private:
            audit[family][schedule] = strip_private(private[family][schedule])
        for span in SPAN_SENSITIVITY:
            span_params = {**base_params, "span": span}
            audit["span_sensitivity"].setdefault(str(span), {})[schedule] = strip_private(
                evaluate(data, cfg["split"]["audit"], schedule, span_params))
    effects = {}
    for family in ("unified_calibration", "schedule_specific_calibration"):
        effects[family] = {}
        for schedule in SCHEDULES:
            effects[family][schedule] = paired_delta(private[family][schedule], private["baseline"][schedule])
    gaps = {}
    for family in private:
        pred = private[family]["predicted_nonblank"]
        gaps[family] = {
            "predicted_minus_uniform": paired_delta(pred, private[family]["fixed_center_uniform"]),
            "predicted_minus_full_nonblank_oracle": paired_delta(pred, private[family]["full_nonblank_oracle"]),
            "predicted_minus_terminal_block_oracle": paired_delta(pred, private[family]["terminal_block_oracle"]),
        }
    base_gap = gaps["baseline"]
    unified_gap = gaps["unified_calibration"]
    decision = {
        "unified_improves_uniform": effects["unified_calibration"]["fixed_center_uniform"]["error_delta"] < 0,
        "unified_improves_predicted": effects["unified_calibration"]["predicted_nonblank"]["error_delta"] < 0,
        "unified_improves_both_uniform_and_predicted": all(
            effects["unified_calibration"][s]["error_delta"] < 0 for s in ("fixed_center_uniform", "predicted_nonblank")),
        "unified_shrinks_predictor_terminal_oracle_gap": abs(unified_gap["predicted_minus_terminal_block_oracle"]["error_delta"]) < abs(base_gap["predicted_minus_terminal_block_oracle"]["error_delta"]),
        "interpretation_rule": "common improvement means decoder calibration matters generally; only a materially smaller adaptive-oracle gap supports schedule-specific decoder mismatch",
    }
    result = {
        "scope": "exploratory old-fixed128 sparse decoder mismatch diagnostic",
        "confirmatory": False, "gpu_used": False, "dev_used": False, "test_used": False,
        "split_counts": {key: len(cfg["split"][key]) for key in ("tune", "audit")},
        "tuning": {
            "unified": {key: unified[key] for key in ("temperature", "blank_bias", "pooled_error", "errors")},
            "schedule_specific": {schedule: {"temperature": value["temperature"], "blank_bias": value["blank_bias"],
                                               "tune_error": value["errors"][schedule]} for schedule, value in specific.items()},
            "grid_cells": len(grid),
        },
        "audit": audit,
        "audit_calibration_effects_vs_baseline": effects,
        "audit_schedule_gaps": gaps,
        "decision": decision,
        "limitations": cfg["limitations"] + ["tune/audit are source-disjoint only within the already reused fixed128",
                                               "schedule-specific calibration is diagnostic and cannot be used as a fair common-decoder system comparison",
                                               "cached replay excludes real wall time and latency"],
    }
    (root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256_file(root / "resolved_config_preregistered.json"),
                "metrics_sha256": sha256_file(root / "metrics.json")}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preregister", "analyze"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = args.output_root.resolve()
    value = preregister(root) if args.mode == "preregister" else analyze(root)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
