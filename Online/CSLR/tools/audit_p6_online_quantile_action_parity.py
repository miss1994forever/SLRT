#!/usr/bin/env python3
"""CPU-only action-count parity using frozen train-fit K rows and exported folds.

Teacher-forced K rows contain each block's offset0 feature and the mean over
paid teacher windows. Consecutive means recover prior offset2 summaries for
audit input. This reconstruction uses later rows only to recover those input
summaries; the controller itself sees paid history and current offset0 only.
"""
import argparse
from array import array
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.online_quantile_controller import OnlineQuantileController, FrozenRidgeBundle, ridge_score_callback
from utils.online_quantile_controller import k_history_feature

BASE = ROOT / "results/phoenix-2014t_ISLR"
FIT = BASE / "p6_local_js_predictor_oof_v1_49faacc3/dataset.npz"
FROZEN = BASE / "p6_k_history_online_quantile_oof_v1_49faacc3/per_sample.jsonl.gz"
CONFIG = ROOT / "configs/p6_k_history_online_quantile_oof_v1_preregistered.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_array(archive, name, typecode):
    with zipfile.ZipFile(archive) as packed, packed.open(name + ".npy") as handle:
        magic = handle.read(8)
        length = int.from_bytes(handle.read(2 if magic[6] == 1 else 4), "little")
        handle.read(length)
        values = array(typecode)
        values.frombytes(handle.read())
    return values


def reconstruct_frame_features(krows):
    frames = {}
    for block, row in enumerate(krows):
        start = block * 4
        frames[start] = row[:36]
        if block:
            previous_bonus = start - 2
            visible = ([position for old in range(block) for position in (old * 4, old * 4 + 2)] + [start])[-16:]
            n = len(visible)
            known = [position for position in visible if position != previous_bonus]
            frames[previous_bonus] = [float(row[36 + j]) * n - sum(frames[position][j] for position in known)
                                      for j in range(36)]
    return frames


def frozen_reference_actions(frames, bundle, fold, block_count):
    """Direct transcription of the frozen schedules() loop, independent of class state."""
    selected, offsets, past = [], [], []
    balance = 0.0
    decisions = []
    for block in range(block_count):
        start = block * 4
        value = bundle.score(k_history_feature(frames, selected, start, offsets), fold)
        balance = min(2.0, balance + 0.5)
        execute = balance >= 2.0 or (balance >= 1.0 and past and value >= median(past))
        selected.append(start)
        if execute:
            selected.append(start + 2)
            offsets.append(2)
            balance -= 1.0
        past.append(value)
        decisions.append(bool(execute))
    return decisions


def audit(bundle_path):
    bundle = FrozenRidgeBundle(bundle_path, CONFIG)
    with gzip.open(FROZEN, "rt", encoding="utf-8") as handle:
        expected = {row["sample"]: row for row in (json.loads(line) for line in handle)}
    features = load_array(FIT, "K", "f")
    sample_ids = load_array(FIT, "sample_indices", "h")
    block_ids = load_array(FIT, "block_indices", "h")
    folds = load_array(FIT, "folds", "b")
    names_raw = load_array(FIT, "sample_names", "I")
    names = []
    for i in range(0, len(names_raw), 48):
        names.append("".join(chr(v) for v in names_raw[i:i + 48]).rstrip("\0"))
    if len(sample_ids) != 183401 or len(features) != 183401 * 153 or len(names) != 6378:
        raise RuntimeError("fit K dataset shape mismatch")
    if set(names) != set(expected):
        raise RuntimeError("fit sample identities differ from frozen results")
    mismatches = []
    totals = defaultdict(int)
    index = 0
    for sample_id, name in enumerate(names):
        rows = []
        first = index
        while index < len(sample_ids) and sample_ids[index] == sample_id:
            if block_ids[index] != len(rows):
                raise RuntimeError("block order mismatch")
            rows.append(features[index * 153:(index + 1) * 153])
            index += 1
        if not rows:
            raise RuntimeError("empty sample")
        fold = folds[first]
        if any(folds[j] != fold for j in range(first, index)):
            raise RuntimeError("sample crosses source folds")
        frames = reconstruct_frame_features(rows)
        teacher_selected = []
        for block, frozen_row in enumerate(rows):
            start = block * 4
            reconstructed = k_history_feature(frames, teacher_selected, start, [2] * block)
            totals["max_abs_teacher_feature_error"] = max(
                totals["max_abs_teacher_feature_error"],
                max(abs(float(a) - float(b)) for a, b in zip(reconstructed, frozen_row)))
            teacher_selected.extend((start, start + 2))
        policy = OnlineQuantileController(ridge_score_callback(bundle, fold, frames))
        decisions = []
        for block in range(len(rows)):
            policy.start_block(block * 4)
            decisions.append(policy.complete_block(block * 4))
        reference_decisions = frozen_reference_actions(frames, bundle, fold, len(rows))
        if decisions != reference_decisions:
            raise AssertionError(f"block action mismatch with frozen reference: {name}")
        if expected[name]["full_center_windows"] % 2:
            policy.start_block(len(rows) * 4)
        policy.finish()
        observed_bonus = policy.forced + policy.autonomous
        record = expected[name]
        expected_bonus = record["forced"] + record["autonomous"]
        totals["samples"] += 1
        totals["blocks"] += len(rows)
        totals["matched_block_actions"] += len(rows)
        totals["observed_bonus"] += observed_bonus
        totals["frozen_bonus"] += expected_bonus
        if (observed_bonus != expected_bonus or policy.forced != record["forced"] or
                policy.autonomous != record["autonomous"] or
                abs(policy.balance - record["unspent"]) > 1e-9):
            mismatches.append({"sample": name, "observed_bonus": observed_bonus,
                               "frozen_bonus": expected_bonus, "observed_forced": policy.forced,
                               "frozen_forced": record["forced"], "observed_unspent": policy.balance,
                               "frozen_unspent": record["unspent"]})
        if sample_id % 1000 == 0:
            print(json.dumps({"checked_samples": sample_id + 1, "mismatches": len(mismatches)}), flush=True)
    if index != len(sample_ids):
        raise RuntimeError("unconsumed fit rows")
    return {"passed": not mismatches, "checked_samples": totals["samples"],
            "checked_blocks": totals["blocks"], "observed_bonus": totals["observed_bonus"],
            "matched_block_actions": totals["matched_block_actions"],
            "max_abs_teacher_feature_error": totals["max_abs_teacher_feature_error"],
            "frozen_bonus": totals["frozen_bonus"], "mismatch_count": len(mismatches),
            "first_mismatches": mismatches[:20], "model_sha256": sha(bundle_path),
            "fit_dataset_sha256": sha(FIT), "frozen_per_sample_sha256": sha(FROZEN),
            "note": "Uses frozen K-row means to reconstruct paid offset2 summaries; compares actions only, no outcome replay."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(args.model)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
