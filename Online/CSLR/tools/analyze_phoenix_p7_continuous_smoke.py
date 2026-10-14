#!/usr/bin/env python3
"""Replay frozen P6 scheduling/decoding on P7 train-only continuous logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch


CSLR_ROOT = Path(__file__).resolve().parents[1]
SLRT_ROOT = CSLR_ROOT.parents[1]
EVAL_ROOT = CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_continuous_eval_v1"
OUTPUT = EVAL_ROOT / "analysis.json"
PREREG = CSLR_ROOT / "configs/p7_temporal_continuous_smoke_v1_preregistered.json"
P6_RULE = CSLR_ROOT / "configs/p6_k_history_online_quantile_oof_v1_preregistered.json"
P6_MODELS = CSLR_ROOT / "configs/frozen_models/p6_online_quantile_fold_models_v1.json"
VOCAB = SLRT_ROOT / "data/phoenix_2014t/phoenix_iso_with_blank.vocab"
CANDIDATES = ["A_current_fixed16", "B_aligned_fixed16", "C_inside_even16"]
SCHEDULES = ["quantile", "full_center", "dense"]

import sys
if str(CSLR_ROOT) not in sys.path:
    sys.path.insert(0, str(CSLR_ROOT))

from utils.adaptive_stride import span_weighted_predictions  # noqa: E402
from utils.metrics import wer_single  # noqa: E402
from utils.online_quantile_controller import (  # noqa: E402
    FrozenRidgeBundle,
    OnlineQuantileController,
    ridge_score_callback,
)
from utils.phoenix_cleanup import clean_phoenix_2014_trans  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_video(name: str) -> str:
    return name.rpartition("-")[0]


def source_fold(name: str) -> int:
    source = source_video(name)
    value = hashlib.sha256(f"robust-oof-v1\0{source}".encode()).digest()[:8]
    return int.from_bytes(value, "big") % 5


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def posterior_features(probabilities: np.ndarray, blank: int) -> list[list[float]]:
    rng = np.random.default_rng(261130)
    buckets = rng.integers(0, 32, size=probabilities.shape[1])
    signs = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=probabilities.shape[1])
    rows = []
    for probability in probabilities:
        top = np.argpartition(probability, -8)[-8:]
        feature = np.zeros(36, np.float32)
        feature[:32] = np.bincount(
            buckets[top], weights=probability[top] * signs[top], minlength=32
        )
        ordered = np.sort(probability[top])
        feature[32] = -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
        feature[33] = probability[blank]
        feature[34] = ordered[-1]
        feature[35] = ordered[-1] - ordered[-2]
        rows.append(feature.tolist())
    return rows


def frozen_schedules(logits: np.ndarray, name: str, blank: int, bundle: FrozenRidgeBundle) -> dict:
    features = posterior_features(softmax_rows(logits), blank)
    paid = {}
    controller = OnlineQuantileController(
        ridge_score_callback(bundle, source_fold(name), paid)
    )
    complete = list(range(0, max(0, len(logits) - 3), 4))
    for start in complete:
        paid[start] = features[start]
        controller.start_block(start)
        if controller.complete_block(start):
            paid[start + 2] = features[start + 2]
    quantile = controller.finish()
    skeleton = list(range(0, len(logits), 4))
    quantile = sorted(set(quantile + [start for start in skeleton if start not in complete]))
    full_center = sorted(set(skeleton + [start + 2 for start in complete]))
    return {
        "quantile": quantile,
        "full_center": full_center,
        "dense": list(range(len(logits))),
    }


def map_gloss(gloss: str) -> str:
    for prefix, offset in (("neg-", 4), ("poss-", 5), ("negalp-", 7), ("loc-", 7), ("cl-", 7)):
        if gloss.startswith(prefix):
            return prefix + gloss[offset:].upper()
    return gloss.upper()


def decode(logits: np.ndarray, starts: list[int], selected: list[int], vocab: list[str], blank: int) -> str:
    selected_logits = torch.from_numpy(np.asarray(logits[selected], np.float32))
    centers = [float(starts[index]) + 7.5 for index in selected]
    token_ids = span_weighted_predictions(
        selected_logits, centers, span=15.0, min_weight=0.05
    ).tolist()
    collapsed = [token for token, _ in groupby(token_ids) if token != blank]
    return " ".join(map_gloss(vocab[token]) for token in collapsed)


def sentence_row(name: str, reference: str, hypothesis: str, windows: int) -> dict:
    score = wer_single(reference, hypothesis)
    return {
        "sample": name,
        "source": source_video(name),
        "reference": reference,
        "hypothesis": hypothesis,
        "ref_len": int(score["num_ref"]),
        "error": int(score["num_err"]),
        "del": int(score["num_del"]),
        "ins": int(score["num_ins"]),
        "sub": int(score["num_sub"]),
        "windows": windows,
        "alignment": score["alignment"],
    }


def aggregate(rows: list[dict]) -> dict:
    ref = sum(row["ref_len"] for row in rows)
    counts = {key: sum(row[key] for row in rows) for key in ("error", "del", "ins", "sub", "windows")}
    return {
        "sentences": len(rows),
        "reference_glosses": ref,
        "errors": counts["error"],
        "wer": 100.0 * counts["error"] / ref,
        "del": counts["del"],
        "del_rate": 100.0 * counts["del"] / ref,
        "ins": counts["ins"],
        "ins_rate": 100.0 * counts["ins"] / ref,
        "sub": counts["sub"],
        "sub_rate": 100.0 * counts["sub"] / ref,
        "windows": counts["windows"],
        "exact_sentences": sum(row["error"] == 0 for row in rows),
        "sentences_with_2plus_errors": sum(row["error"] >= 2 for row in rows),
    }


def multiscale(rows: list[dict]) -> dict:
    words = Counter()
    word_errors = Counter()
    ngram_total = {2: Counter(), 3: Counter()}
    ngram_error = {2: Counter(), 3: Counter()}
    error_run_lengths = Counter()
    for row in rows:
        reference = row["reference"].split()
        alignment = row["alignment"]
        bad_positions = set()
        ref_position = 0
        current_run = 0
        for op in alignment["align_lst"]:
            if op in ("C", "D", "S"):
                if op in ("D", "S"):
                    bad_positions.add(ref_position)
                    current_run += 1
                else:
                    if current_run:
                        error_run_lengths[current_run] += 1
                    current_run = 0
                ref_position += 1
            elif current_run:
                error_run_lengths[current_run] += 1
                current_run = 0
        if current_run:
            error_run_lengths[current_run] += 1
        for position, token in enumerate(reference):
            words[token] += 1
            if position in bad_positions:
                word_errors[token] += 1
        for n in (2, 3):
            for start in range(len(reference) - n + 1):
                phrase = " ".join(reference[start:start + n])
                ngram_total[n][phrase] += 1
                if any(position in bad_positions for position in range(start, start + n)):
                    ngram_error[n][phrase] += 1
    highest_words = sorted(
        ({"gloss": token, "errors": count, "total": words[token], "rate": count / words[token]}
         for token, count in word_errors.items()),
        key=lambda row: (-row["errors"], row["gloss"]),
    )[:30]
    phrases = {}
    for n in (2, 3):
        total_occurrences = sum(ngram_total[n].values())
        affected = sum(ngram_error[n].values())
        phrases[str(n)] = {
            "occurrences": total_occurrences,
            "affected": affected,
            "affected_rate": affected / total_occurrences if total_occurrences else 0.0,
            "highest_error_phrases": sorted(
                ({"phrase": phrase, "affected": count, "total": ngram_total[n][phrase]}
                 for phrase, count in ngram_error[n].items()),
                key=lambda row: (-row["affected"], row["phrase"]),
            )[:20],
        }
    return {
        "highest_reference_consuming_error_words": highest_words,
        "reference_ngrams_with_at_least_one_del_or_sub": phrases,
        "consecutive_del_sub_run_lengths": dict(sorted(error_run_lengths.items())),
    }


def bootstrap_delta(baseline: list[dict], candidate: list[dict], repetitions: int, seed: int) -> dict:
    left = {row["sample"]: row for row in baseline}
    right = {row["sample"]: row for row in candidate}
    if set(left) != set(right):
        raise RuntimeError("candidate sample mismatch")
    clusters = defaultdict(lambda: [0, 0])
    for name in sorted(left):
        clusters[left[name]["source"]][0] += right[name]["error"] - left[name]["error"]
        clusters[left[name]["source"]][1] += left[name]["ref_len"]
    values = np.asarray(list(clusters.values()), np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, np.float64)
    for index in range(repetitions):
        sample = values[rng.integers(0, len(values), len(values))]
        draws[index] = 100.0 * sample[:, 0].sum() / sample[:, 1].sum()
    observed = 100.0 * values[:, 0].sum() / values[:, 1].sum()
    return {
        "delta_wer_pp_candidate_minus_A": float(observed),
        "source_clusters": len(values),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "repetitions": repetitions,
        "seed": seed,
    }


def changed_examples(baseline: list[dict], candidate: list[dict], limit: int = 20) -> dict:
    left = {row["sample"]: row for row in baseline}
    right = {row["sample"]: row for row in candidate}
    changes = []
    for name in sorted(left):
        delta = right[name]["error"] - left[name]["error"]
        if delta:
            changes.append({
                "sample": name,
                "reference": left[name]["reference"],
                "A_hypothesis": left[name]["hypothesis"],
                "candidate_hypothesis": right[name]["hypothesis"],
                "A_errors": left[name]["error"],
                "candidate_errors": right[name]["error"],
                "delta_errors": delta,
            })
    return {
        "improvements": sorted((row for row in changes if row["delta_errors"] < 0), key=lambda row: (row["delta_errors"], row["sample"]))[:limit],
        "regressions": sorted((row for row in changes if row["delta_errors"] > 0), key=lambda row: (-row["delta_errors"], row["sample"]))[:limit],
    }


def verify_preregistered_assets() -> dict:
    prereg = json.loads(PREREG.read_text())
    fixed = prereg["fixed_assets"]
    paths = {
        "continuous_assets_manifest_sha256": CSLR_ROOT / "results/phoenix-2014t_ISLR/p7_temporal_continuous_smoke_assets_v1/manifest.json",
        "A_config_sha256": CSLR_ROOT / "configs/p7_temporal_continuous_smoke_A_current_fixed16.yaml",
        "B_config_sha256": CSLR_ROOT / "configs/p7_temporal_continuous_smoke_B_aligned_fixed16.yaml",
        "C_config_sha256": CSLR_ROOT / "configs/p7_temporal_continuous_smoke_C_inside_even16.yaml",
        "p6_rule_sha256": P6_RULE,
        "p6_models_sha256": P6_MODELS,
        "p6_controller_sha256": CSLR_ROOT / "utils/online_quantile_controller.py",
        "span_decoder_implementation_sha256": CSLR_ROOT / "utils/adaptive_stride.py",
    }
    actual = {key: sha256(path) for key, path in paths.items()}
    if actual != fixed:
        raise RuntimeError({"preregistered": fixed, "actual": actual})
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = json.loads(args.output.read_text())
        for record in result["inputs"]["candidate_outputs"].values():
            for item in record.values():
                assert sha256(Path(item["path"])) == item["sha256"]
        assert result["scope"].endswith("official dev/test forbidden")
        print(json.dumps({"verified": True, "output": str(args.output.resolve())}, sort_keys=True))
        return
    if args.bootstrap_repetitions < 1000:
        raise SystemExit("--bootstrap-repetitions must be >= 1000")

    frozen_hashes = verify_preregistered_assets()
    vocab = json.loads(VOCAB.read_text())
    blank = vocab.index("<blank>")
    bundle = FrozenRidgeBundle(P6_MODELS, P6_RULE)
    all_rows = defaultdict(dict)
    inputs = {}
    runtimes = {}
    for candidate in CANDIDATES:
        directory = args.eval_root / candidate
        paths = {
            "results": directory / "train_results.pkl",
            "logits": directory / "train_logits.pkl",
            "runtime": directory / "train_runtime_profile.json",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        with paths["results"].open("rb") as handle:
            results = pickle.load(handle)
        with paths["logits"].open("rb") as handle:
            logits_by_name = pickle.load(handle)
        if set(results) != set(logits_by_name):
            raise RuntimeError(f"result/logit sample mismatch for {candidate}")
        runtimes[candidate] = json.loads(paths["runtime"].read_text())
        inputs[candidate] = {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in paths.items()
        }
        for name, result in results.items():
            logits = np.asarray(logits_by_name[name], np.float32)
            metadata = result.get("adaptive_stride_metadata")
            starts = [int(item["start"]) for item in metadata]
            if starts != list(range(len(starts))) or len(starts) != len(logits):
                raise RuntimeError(f"expected dense stride-1 starts for {name}")
            reference = clean_phoenix_2014_trans(result["gls_ref"])
            for schedule, selected in frozen_schedules(logits, name, blank, bundle).items():
                hypothesis = clean_phoenix_2014_trans(decode(logits, starts, selected, vocab, blank))
                all_rows[candidate][schedule] = all_rows[candidate].get(schedule, [])
                all_rows[candidate][schedule].append(
                    sentence_row(name, reference, hypothesis, len(selected))
                )

    summaries = {
        candidate: {
            schedule: {
                "aggregate": aggregate(rows),
                "multiscale": multiscale(rows),
            }
            for schedule, rows in schedules.items()
        }
        for candidate, schedules in all_rows.items()
    }
    comparisons = {}
    examples = {}
    baseline = all_rows[CANDIDATES[0]]["quantile"]
    for index, candidate in enumerate(CANDIDATES[1:], start=1):
        rows = all_rows[candidate]["quantile"]
        comparisons[f"{candidate}_minus_{CANDIDATES[0]}"] = bootstrap_delta(
            baseline, rows, args.bootstrap_repetitions, 73050 + index
        )
        examples[candidate] = changed_examples(baseline, rows)

    result = {
        "schema_version": 1,
        "scope": "frozen P6 scheduler/decoder on source-disjoint train-only continuous screen; official dev/test forbidden",
        "primary": {"schedule": "quantile", "decoder": "span_weighted_15"},
        "summaries": summaries,
        "paired_source_cluster_bootstrap": comparisons,
        "representative_sentence_changes": examples,
        "runtime_profiles": runtimes,
        "inputs": {
            "preregister": {"path": str(PREREG.resolve()), "sha256": sha256(PREREG)},
            "frozen_asset_hashes": frozen_hashes,
            "candidate_outputs": inputs,
            "vocab": {"path": str(VOCAB.resolve()), "sha256": sha256(VOCAB)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "candidates": CANDIDATES}, sort_keys=True))


if __name__ == "__main__":
    main()
