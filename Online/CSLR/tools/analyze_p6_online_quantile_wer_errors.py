#!/usr/bin/env python3
"""Read-only, multiscale WER analysis for the frozen P6 online controller.

This diagnostic reconstructs hypotheses from the already-frozen train-fit and
authorized retrospective-dev logits.  It does not run the recognizer, inspect
test, tune the controller, or change any frozen result.  The output is written
under a new diagnostic identity and is accepted only when every per-sample
error/window count and every aggregate WER count matches the frozen artifacts.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
TRAIN_DENSE = BASE / "train_dense_stride1_v1_49faacc3"
TRAIN_FIT = BASE / "p6_local_js_predictor_oof_v1_49faacc3/dataset.npz"
TRAIN_FROZEN = BASE / "p6_k_history_online_quantile_oof_v1_49faacc3"
DEV_FROZEN = BASE / "p6_k_online_quantile_dev_retrospective_v1_49faacc3"
DEV_INPUT = BASE / "baseline_matrix_v1_repaired_49faacc3/dev/B0_fixed1_window7/run_01"
FOLD_MODELS = BASE / "p6_online_quantile_release_v1/fold_models.json"
OUTPUT = BASE / "p6_online_quantile_wer_error_analysis_v1"
META = {
    "train_fit": ROOT.parents[1] / "data/phoenix_2014t/phoenix14t.train",
    "dev": ROOT.parents[1] / "data/phoenix_2014t/phoenix14t.dev",
}
POLICIES = ("quantile", "matched_uniform", "online_uniform", "full_center")
ERROR_TYPES = ("del", "ins", "sub")
BOOTSTRAP_SEED = 260929


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P6 = load_module(ROOT / "tools/analyze_phoenix_p6_k_online_quantile_oof.py", "p6_wer_frozen")
DEV = load_module(ROOT / "tools/evaluate_p6_k_online_quantile_dev_retrospective.py", "p6_wer_dev")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def frozen_rows(root):
    with gzip.open(root / "per_sample.jsonl.gz", "rt", encoding="utf-8") as handle:
        return {row["sample"]: row for row in map(json.loads, handle)}


def portable_fold_models():
    packed = json.loads(FOLD_MODELS.read_text(encoding="utf-8"))
    models = []
    for fold in packed["folds"]:
        models.append((
            np.asarray(fold["mean"], np.float64),
            np.asarray(fold["scale"], np.float64),
            np.asarray(fold["weights"], np.float64),
            float(fold["intercept"]),
        ))
    if len(models) != 5:
        raise RuntimeError("expected five frozen fold models")
    return models


def train_sample_folds():
    with np.load(TRAIN_FIT, allow_pickle=False) as packed:
        names = [str(value) for value in packed["sample_names"].tolist()]
        sample_indices = np.asarray(packed["sample_indices"], np.int64)
        folds = np.asarray(packed["folds"], np.int8)
    result = {}
    for index, name in enumerate(names):
        unique = np.unique(folds[sample_indices == index])
        if len(unique) != 1:
            raise RuntimeError(f"sample fold mismatch: {name}")
        result[name] = int(unique[0])
    return result


def metadata():
    result = {}
    for split, path in META.items():
        with gzip.open(path, "rb") as handle:
            rows = pickle.load(handle)
        result[split] = {row["name"]: row for row in rows}
    return result


def alignment(reference, hypothesis):
    result = P6.CHRON.REPLAY.wer_single(reference, hypothesis)
    aligned = result["alignment"]
    events = []
    ref_position = 0
    for op, ref_token, hyp_token in zip(
        aligned["align_lst"], aligned["align_ref_lst"], aligned["align_hyp_lst"]
    ):
        consumes_ref = op in ("C", "D", "S")
        position = ref_position if consumes_ref else max(0, ref_position - 1)
        if op != "C":
            events.append({
                "op": op,
                "ref": None if op == "I" else ref_token,
                "hyp": None if op == "D" else hyp_token,
                "ref_position": int(position),
                "insertion_boundary": int(ref_position) if op == "I" else None,
            })
        if consumes_ref:
            ref_position += 1
    return aligned, events


def decode_one(split, name, result, logits, model, frozen, meta, vocab, blank, buckets, signs):
    probabilities = P6.BUILDER.ORACLE.softmax_rows(np.asarray(logits))
    features = P6.RANK.dense_frame_features(probabilities, blank, buckets, signs)
    policies, trace = P6.schedules(probabilities, features, model)
    reference = P6.BUILDER.REPLAY.clean_phoenix_2014_trans(result["gls_ref"])
    row = {
        "split": split,
        "sample": name,
        "source": P6.BUILDER.source_video(name),
        "reference": reference,
        "ref_len": len(reference.split()),
        "signer": meta["signer"],
        "num_frames": int(meta["num_frames"]),
        "trace": trace,
        "policies": {},
    }
    expected = frozen[name]
    for policy in POLICIES:
        hypothesis, counts = P6.CHRON.decode_counts(
            probabilities, policies[policy], reference, vocab, blank
        )
        if int(counts["error"]) != int(expected[policy]):
            raise RuntimeError(
                f"frozen error mismatch {split} {name} {policy}: "
                f"{counts['error']} != {expected[policy]}"
            )
        if len(policies[policy]) != int(expected[f"{policy}_windows"]):
            raise RuntimeError(f"frozen window mismatch {split} {name} {policy}")
        aligned, events = alignment(reference, hypothesis)
        observed = Counter(event["op"] for event in events)
        if (observed["D"], observed["I"], observed["S"]) != (
            counts["del"], counts["ins"], counts["sub"]
        ):
            raise RuntimeError(f"alignment count mismatch {split} {name} {policy}")
        row["policies"][policy] = {
            "hypothesis": hypothesis,
            "windows": len(policies[policy]),
            "counts": counts,
            "alignment": aligned,
            "events": events,
        }
    return row


def reconstruct(output_path):
    train_frozen = frozen_rows(TRAIN_FROZEN)
    dev_frozen = frozen_rows(DEV_FROZEN)
    folds = train_sample_folds()
    meta = metadata()
    models = portable_fold_models()
    vocab = json.loads(P6.CHRON.DEFAULT_VOCAB.read_text(encoding="utf-8"))
    blank = vocab.index("<blank>")
    rng = np.random.default_rng(261130)
    buckets = rng.integers(0, 32, size=len(vocab))
    signs = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=len(vocab))
    count = 0
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        shard_indices, shard_count = P6.CHRON.completed_shard_indices(TRAIN_DENSE)
        for shard in shard_indices:
            results, logits, _ = P6.BUILDER.BUILDER.validate_dense_shard(
                TRAIN_DENSE, shard, shard_count, verify_hashes=False
            )
            for name in results:
                if name not in folds:
                    continue
                row = decode_one(
                    "train_fit", name, results[name], logits[name], models[folds[name]],
                    train_frozen, meta["train_fit"][name], vocab, blank, buckets, signs,
                )
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            print(json.dumps({"split": "train_fit", "shard": shard, "samples": count}), flush=True)
        if count != len(train_frozen):
            raise RuntimeError(f"train sample count mismatch: {count} != {len(train_frozen)}")

        model = DEV.fit_full_k()
        with (DEV_INPUT / "dev_results.pkl").open("rb") as source:
            results = pickle.load(source)
        with (DEV_INPUT / "dev_logits.pkl").open("rb") as source:
            logits = pickle.load(source)
        for name in sorted(results):
            row = decode_one(
                "dev", name, results[name], logits[name], model,
                dev_frozen, meta["dev"][name], vocab, blank, buckets, signs,
            )
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        if len(results) != len(dev_frozen):
            raise RuntimeError("dev sample count mismatch")
    return count, len(results)


def error_position(event, ref_len):
    if ref_len <= 1:
        return "single"
    position = event["insertion_boundary"] if event["op"] == "I" else event["ref_position"]
    ratio = float(position) / max(1, ref_len - 1)
    if ratio < 1.0 / 3.0:
        return "start"
    if ratio < 2.0 / 3.0:
        return "middle"
    return "end"


def length_bin(length):
    if length <= 3:
        return "1-3"
    if length <= 6:
        return "4-6"
    if length <= 9:
        return "7-9"
    return "10+"


def frame_bin(frames):
    if frames <= 64:
        return "1-64"
    if frames <= 96:
        return "65-96"
    if frames <= 128:
        return "97-128"
    return "129+"


def top_counter(counter, limit=30):
    return [{"key": key, "count": int(value)} for key, value in counter.most_common(limit)]


def concentration(values):
    ordered = sorted((int(value) for value in values), reverse=True)
    total = sum(ordered)
    result = {}
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(np.ceil(len(ordered) * fraction)))
        result[f"top_{int(fraction * 100)}pct"] = {
            "units": count,
            "errors": sum(ordered[:count]),
            "error_share": (sum(ordered[:count]) / total if total else 0.0),
        }
    return result


def bootstrap_rates(rows, policy, repetitions=4000):
    grouped = defaultdict(lambda: Counter())
    for row in rows:
        counts = row["policies"][policy]["counts"]
        bucket = grouped[row["source"]]
        for key in ("ref_len", "del", "ins", "sub", "error"):
            bucket[key] += int(counts[key])
    sources = list(grouped.values())
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if rows[0]["split"] == "train_fit" else 1))
    draws = {key: np.empty(repetitions, np.float64) for key in ("del", "ins", "sub", "error")}
    for index in range(repetitions):
        chosen = rng.integers(0, len(sources), len(sources))
        denominator = sum(sources[value]["ref_len"] for value in chosen)
        for key in draws:
            draws[key][index] = 100.0 * sum(sources[value][key] for value in chosen) / denominator
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in draws.items()
    }


def ngram_exposure(rows, n):
    total = Counter()
    affected = Counter()
    examples = defaultdict(list)
    for row in rows:
        tokens = row["reference"].split()
        events = row["policies"]["quantile"]["events"]
        bad_positions = {event["ref_position"] for event in events if event["op"] in ("D", "S")}
        for start in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[start:start + n])
            total[phrase] += 1
            if any(position in bad_positions for position in range(start, start + n)):
                affected[phrase] += 1
                if len(examples[phrase]) < 3:
                    examples[phrase].append(row["sample"])
    candidates = []
    for phrase, errors in affected.items():
        candidates.append({
            "phrase": phrase,
            "occurrences": total[phrase],
            "affected_occurrences": errors,
            "affected_rate": errors / total[phrase],
            "examples": examples[phrase],
        })
    by_count = sorted(candidates, key=lambda item: (-item["affected_occurrences"], -item["occurrences"], item["phrase"]))[:30]
    eligible = [item for item in candidates if item["occurrences"] >= 5]
    by_rate = sorted(eligible, key=lambda item: (-item["affected_rate"], -item["affected_occurrences"], item["phrase"]))[:30]
    return {"by_affected_count": by_count, "by_affected_rate_min5": by_rate}


def longest_common_contiguous(left, right):
    best = 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            value = previous[index - 1] + 1 if left_token == right_token else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


def insertion_neighbor_evidence(rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    neighbors = {}
    for source_rows in by_source.values():
        source_rows.sort(key=lambda row: int(row["sample"].rsplit("-", 1)[1]))
        for index, row in enumerate(source_rows):
            neighbors[row["sample"]] = (
                source_rows[index - 1] if index else None,
                source_rows[index + 1] if index + 1 < len(source_rows) else None,
            )
    run_lengths = Counter()
    total_tokens = matched_tokens = 0
    multi_runs = matched_multi_runs = 0
    direction = Counter()
    examples = []
    for row in rows:
        aligned = row["policies"]["quantile"]["alignment"]
        runs = []
        current = []
        for op, hyp in zip(aligned["align_lst"], aligned["align_hyp_lst"]):
            if op == "I":
                current.append(hyp)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        previous, following = neighbors[row["sample"]]
        candidates = {
            "previous": previous["reference"].split() if previous else [],
            "next": following["reference"].split() if following else [],
        }
        for run in runs:
            run_lengths[str(min(8, len(run))) + ("+" if len(run) >= 8 else "")] += 1
            total_tokens += len(run)
            if len(run) < 2:
                continue
            multi_runs += 1
            scores = {key: longest_common_contiguous(run, tokens) for key, tokens in candidates.items()}
            best_direction = max(scores, key=scores.get)
            best = scores[best_direction]
            if best >= 2:
                matched_multi_runs += 1
                matched_tokens += best
                direction[best_direction] += 1
                if len(examples) < 20:
                    examples.append({
                        "sample": row["sample"],
                        "inserted_run": " ".join(run),
                        "run_length": len(run),
                        "neighbor": best_direction,
                        "longest_neighbor_match": best,
                        "neighbor_reference": (
                            previous if best_direction == "previous" else following
                        )["reference"],
                    })
    return {
        "insertion_run_length_histogram": dict(run_lengths),
        "insertion_tokens": total_tokens,
        "multi_token_runs": multi_runs,
        "multi_token_runs_with_neighbor_match_len2plus": matched_multi_runs,
        "matched_neighbor_tokens_lower_bound": matched_tokens,
        "matched_neighbor_token_share_of_all_insertions": matched_tokens / total_tokens if total_tokens else 0.0,
        "match_direction": dict(direction),
        "examples": examples,
        "interpretation_limit": "lexical overlap with an adjacent segment; evidence of boundary leakage, not a causal proof",
    }


def summarize_split(rows):
    policies = {}
    for policy in POLICIES:
        totals = Counter()
        for row in rows:
            totals.update(row["policies"][policy]["counts"])
        errors = totals["error"]
        policies[policy] = {
            **{key: int(totals[key]) for key in ("ref_len", "error", "del", "ins", "sub")},
            "wer": 100.0 * errors / totals["ref_len"],
            "error_type_share": {
                key: (totals[key] / errors if errors else 0.0) for key in ERROR_TYPES
            },
            "exact_match_sentences": sum(
                row["policies"][policy]["counts"]["error"] == 0 for row in rows
            ),
        }

    ref_frequency = Counter()
    ref_error = Counter()
    deletions = Counter()
    insertions = Counter()
    substitutions = Counter()
    contexts = Counter()
    positions = Counter()
    error_runs = Counter()
    run_examples = defaultdict(list)
    sentence_histogram = Counter()
    length_stats = defaultdict(Counter)
    frame_stats = defaultdict(Counter)
    signer_stats = defaultdict(Counter)
    source_errors = Counter()
    exact_boundaries = Counter()
    run_shape = Counter()
    examples = []
    event_sets = defaultdict(dict)
    for row in rows:
        tokens = row["reference"].split()
        ref_frequency.update(tokens)
        quantile = row["policies"]["quantile"]
        counts = quantile["counts"]
        sentence_histogram[str(min(8, counts["error"])) + ("+" if counts["error"] >= 8 else "")] += 1
        stats = length_stats[length_bin(len(tokens))]
        stats["samples"] += 1
        stats["ref_len"] += len(tokens)
        for key in ("error", "del", "ins", "sub"):
            stats[key] += counts[key]
        for bucket in (frame_stats[frame_bin(row["num_frames"])], signer_stats[row["signer"]]):
            bucket["samples"] += 1
            bucket["ref_len"] += len(tokens)
            for key in ("error", "del", "ins", "sub"):
                bucket[key] += counts[key]
        source_errors[row["source"]] += counts["error"]
        aligned = quantile["alignment"]
        run = []
        all_runs = []
        for op, ref_token, hyp_token in zip(
            aligned["align_lst"], aligned["align_ref_lst"], aligned["align_hyp_lst"]
        ):
            if op == "C":
                if run:
                    all_runs.append(run)
                    ref_phrase = " ".join(item[1] for item in run if item[0] != "I") or "<EMPTY>"
                    if 2 <= len([item for item in run if item[0] != "I"]) <= 3:
                        signature = f"{''.join(item[0] for item in run)} | {ref_phrase}"
                        error_runs[signature] += 1
                        if len(run_examples[signature]) < 3:
                            run_examples[signature].append(row["sample"])
                    run = []
                continue
            run.append((op, ref_token, hyp_token))
        if run:
            all_runs.append(run)
            ref_phrase = " ".join(item[1] for item in run if item[0] != "I") or "<EMPTY>"
            if 2 <= len([item for item in run if item[0] != "I"]) <= 3:
                signature = f"{''.join(item[0] for item in run)} | {ref_phrase}"
                error_runs[signature] += 1
                if len(run_examples[signature]) < 3:
                    run_examples[signature].append(row["sample"])

        for error_run in all_runs:
            ref_words = sum(item[0] != "I" for item in error_run)
            key = "insertion_only" if ref_words == 0 else str(ref_words) if ref_words <= 3 else "4+"
            run_shape[f"runs_refwords_{key}"] += 1
            run_shape[f"events_refwords_{key}"] += len(error_run)

        for event in quantile["events"]:
            positions[error_position(event, len(tokens))] += 1
            pos = event["ref_position"]
            if event["op"] == "I":
                if event["insertion_boundary"] == 0:
                    exact_boundaries["insertion_before_first"] += 1
                if event["insertion_boundary"] == len(tokens):
                    exact_boundaries["insertion_after_last"] += 1
            else:
                if pos == 0:
                    exact_boundaries["first_reference_word"] += 1
                if pos == len(tokens) - 1:
                    exact_boundaries["last_reference_word"] += 1
            before = tokens[pos - 1] if pos > 0 and pos - 1 < len(tokens) else "<BOS>"
            after_index = pos + (1 if event["op"] in ("D", "S") else 0)
            after = tokens[after_index] if after_index < len(tokens) else "<EOS>"
            contexts[f"{event['op']} | {before} _ {after}"] += 1
            if event["op"] == "D":
                deletions[event["ref"]] += 1
                ref_error[event["ref"]] += 1
            elif event["op"] == "I":
                insertions[event["hyp"]] += 1
            else:
                substitutions[f"{event['ref']} -> {event['hyp']}"] += 1
                ref_error[event["ref"]] += 1
        if counts["error"]:
            examples.append({
                "sample": row["sample"],
                "source": row["source"],
                "reference": row["reference"],
                "hypothesis": quantile["hypothesis"],
                "counts": counts,
            })
        for policy in ("quantile", "matched_uniform", "online_uniform", "full_center"):
            signatures = set()
            for event in row["policies"][policy]["events"]:
                signatures.add((event["op"], event["ref_position"], event["ref"], event["hyp"]))
            event_sets[policy][row["sample"]] = signatures

    token_rows = []
    for token, errors in ref_error.items():
        token_rows.append({
            "token": token,
            "reference_occurrences": ref_frequency[token],
            "deletions": deletions[token],
            "substitution_sources": sum(
                count for pair, count in substitutions.items() if pair.startswith(token + " -> ")
            ),
            "reference_errors": errors,
            "reference_error_rate": errors / ref_frequency[token],
        })
    token_by_count = sorted(token_rows, key=lambda item: (-item["reference_errors"], -item["reference_occurrences"], item["token"]))[:40]
    token_by_rate = sorted(
        (item for item in token_rows if item["reference_occurrences"] >= 10),
        key=lambda item: (-item["reference_error_rate"], -item["reference_errors"], item["token"]),
    )[:40]

    comparisons = {}
    for other in ("matched_uniform", "online_uniform", "full_center"):
        sample_direction = Counter()
        type_delta = Counter()
        shared = unique_quantile = unique_other = 0
        for row in rows:
            left = row["policies"]["quantile"]["counts"]
            right = row["policies"][other]["counts"]
            difference = left["error"] - right["error"]
            sample_direction["quantile_better" if difference < 0 else "quantile_worse" if difference > 0 else "same"] += 1
            for key in ERROR_TYPES:
                type_delta[key] += left[key] - right[key]
            left_events = event_sets["quantile"][row["sample"]]
            right_events = event_sets[other][row["sample"]]
            shared += len(left_events & right_events)
            unique_quantile += len(left_events - right_events)
            unique_other += len(right_events - left_events)
        comparisons[other] = {
            "sample_direction": dict(sample_direction),
            "quantile_minus_other_by_type": dict(type_delta),
            "exact_error_event_overlap": {
                "shared": shared,
                "quantile_only": unique_quantile,
                "other_only": unique_other,
                "quantile_event_persistence": shared / (shared + unique_quantile) if shared + unique_quantile else 0.0,
            },
        }

    run_rows = [
        {"signature": key, "count": int(value), "examples": run_examples[key]}
        for key, value in error_runs.most_common(30)
    ]
    examples.sort(key=lambda item: (-item["counts"]["error"], item["sample"]))
    sample_errors = [row["policies"]["quantile"]["counts"]["error"] for row in rows]
    return {
        "samples": len(rows),
        "sources": len({row["source"] for row in rows}),
        "policies": policies,
        "cluster_bootstrap_ci95_wer_components": bootstrap_rates(rows, "quantile"),
        "single_word": {
            "deletion_tokens": top_counter(deletions, 40),
            "insertion_tokens": top_counter(insertions, 40),
            "substitution_pairs": top_counter(substitutions, 40),
            "reference_tokens_by_error_count": token_by_count,
            "reference_tokens_by_error_rate_min10": token_by_rate,
            "error_contexts": top_counter(contexts, 40),
            "error_positions": dict(positions),
            "exact_sentence_boundaries": dict(exact_boundaries),
        },
        "phrases": {
            "bigrams": ngram_exposure(rows, 2),
            "trigrams": ngram_exposure(rows, 3),
            "contiguous_2_3_reference_word_error_runs": run_rows,
            "error_run_shape": dict(run_shape),
            "adjacent_segment_insertion_evidence": insertion_neighbor_evidence(rows),
        },
        "sentences": {
            "error_count_histogram": dict(sentence_histogram),
            "length_bins": {
                key: {
                    **{field: int(value) for field, value in stats.items()},
                    "wer": 100.0 * stats["error"] / stats["ref_len"],
                }
                for key, stats in sorted(length_stats.items())
            },
            "frame_length_bins": {
                key: {
                    **{field: int(value) for field, value in stats.items()},
                    "wer": 100.0 * stats["error"] / stats["ref_len"],
                }
                for key, stats in sorted(frame_stats.items())
            },
            "signers": {
                key: {
                    **{field: int(value) for field, value in stats.items()},
                    "wer": 100.0 * stats["error"] / stats["ref_len"],
                }
                for key, stats in sorted(signer_stats.items())
            },
            "sample_error_concentration": concentration(sample_errors),
            "source_error_concentration": concentration(source_errors.values()),
            "highest_error_examples": examples[:30],
        },
        "policy_comparisons": comparisons,
    }


def train_frequency_effect(train_rows, dev_rows):
    frequency = Counter()
    for row in train_rows:
        frequency.update(row["reference"].split())
    bins = defaultdict(Counter)
    for row in dev_rows:
        tokens = row["reference"].split()
        bad = {
            event["ref_position"]
            for event in row["policies"]["quantile"]["events"]
            if event["op"] in ("D", "S")
        }
        for position, token in enumerate(tokens):
            count = frequency[token]
            label = "0" if count == 0 else "1-9" if count < 10 else "10-49" if count < 50 else "50-199" if count < 200 else "200+"
            bins[label]["dev_reference_occurrences"] += 1
            bins[label]["dev_del_or_sub"] += position in bad
    result = {}
    for label, values in bins.items():
        result[label] = {
            **{key: int(value) for key, value in values.items()},
            "dev_reference_error_rate": values["dev_del_or_sub"] / values["dev_reference_occurrences"],
        }
    result["train_vocabulary_types"] = len(frequency)
    return result


def validate_aggregates(summary):
    for split, root in (("train_fit", TRAIN_FROZEN), ("dev", DEV_FROZEN)):
        frozen = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        observed = summary["splits"][split]
        if observed["samples"] != frozen["samples"]:
            raise RuntimeError(f"{split} aggregate sample mismatch")
        if observed["policies"]["quantile"]["ref_len"] != frozen["reference_glosses"]:
            raise RuntimeError(f"{split} aggregate ref mismatch")
        for policy in POLICIES:
            if observed["policies"][policy]["error"] != frozen["errors"][policy]:
                raise RuntimeError(f"{split} aggregate error mismatch: {policy}")


def analyze(predictions_path):
    rows = defaultdict(list)
    with gzip.open(predictions_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["split"]].append(row)
    if set(rows) != {"train_fit", "dev"}:
        raise RuntimeError(f"unexpected reconstructed splits: {set(rows)}")
    summary = {
        "identity": "P6 online-quantile multiscale WER error analysis v1",
        "scope": {
            "train_fit": "6378 source-disjoint controller-OOF samples; recognizer training split",
            "dev": "519 authorized retrospective samples; non-confirmatory",
            "test_read": False,
            "recognizer_rerun": False,
            "controller_tuned": False,
        },
        "definitions": {
            "word": "weighted edit-distance alignment used by the project (DEL/INS cost 3, SUB cost 4)",
            "phrase": "reference bigram/trigram containing at least one aligned DEL or SUB; insertions are reported separately",
            "sentence": "aggregate corpus WER and per-sentence aligned error counts",
            "event_persistence": "same operation, reference position, reference token and hypothesis token in both policies",
        },
        "splits": {split: summarize_split(split_rows) for split, split_rows in rows.items()},
        "cross_split": {
            "dev_reference_error_by_train_token_frequency": train_frequency_effect(
                rows["train_fit"], rows["dev"]
            )
        },
    }
    validate_aggregates(summary)
    summary["validation"] = {
        "per_sample_error_counts_match_frozen": True,
        "per_sample_window_counts_match_frozen": True,
        "aggregate_error_counts_match_frozen": True,
        "aggregate_reference_counts_match_frozen": True,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--reuse-predictions", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if "test" in str(output).lower():
        raise ValueError("test paths are forbidden for this diagnostic")
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "per_sample_predictions.jsonl.gz"
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = {
            "script_sha256": sha256(Path(__file__)),
            "fold_models_sha256": sha256(FOLD_MODELS),
            "train_source_manifest_sha256": sha256(TRAIN_FROZEN / "manifest.json"),
            "train_metrics_sha256": sha256(TRAIN_FROZEN / "metrics.json"),
            "train_per_sample_sha256": sha256(TRAIN_FROZEN / "per_sample.jsonl.gz"),
            "dev_source_manifest_sha256": sha256(DEV_FROZEN / "manifest.json"),
            "dev_metrics_sha256": sha256(DEV_FROZEN / "metrics.json"),
            "dev_per_sample_sha256": sha256(DEV_FROZEN / "per_sample.jsonl.gz"),
            "train_metadata_sha256": sha256(META["train_fit"]),
            "dev_metadata_sha256": sha256(META["dev"]),
            "predictions_sha256": sha256(predictions),
            "summary_sha256": sha256(summary_path),
        }
        for key, value in actual.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"manifest mismatch: {key}")
        if manifest.get("test_read") is not False or not all(manifest.get("validation", {}).values()):
            raise RuntimeError("manifest scope or validation failed")
        print(json.dumps({"status": "verified", **actual}, indent=2))
        return
    if not args.reuse_predictions and any(path.exists() for path in (predictions, summary_path, manifest_path)):
        raise FileExistsError(f"diagnostic output already exists: {output}")
    if args.reuse_predictions:
        if not predictions.is_file():
            raise FileNotFoundError(f"reconstructed predictions missing: {predictions}")
    else:
        reconstruct(predictions)
    summary = analyze(predictions)
    train_count = summary["splits"]["train_fit"]["samples"]
    dev_count = summary["splits"]["dev"]["samples"]
    write_json(summary_path, summary)
    manifest = {
        "status": "complete",
        "train_samples": train_count,
        "dev_samples": dev_count,
        "test_read": False,
        "script_sha256": sha256(Path(__file__)),
        "fold_models_sha256": sha256(FOLD_MODELS),
        "train_source_manifest_sha256": sha256(TRAIN_FROZEN / "manifest.json"),
        "train_metrics_sha256": sha256(TRAIN_FROZEN / "metrics.json"),
        "train_per_sample_sha256": sha256(TRAIN_FROZEN / "per_sample.jsonl.gz"),
        "dev_source_manifest_sha256": sha256(DEV_FROZEN / "manifest.json"),
        "dev_metrics_sha256": sha256(DEV_FROZEN / "metrics.json"),
        "dev_per_sample_sha256": sha256(DEV_FROZEN / "per_sample.jsonl.gz"),
        "train_metadata_sha256": sha256(META["train_fit"]),
        "dev_metadata_sha256": sha256(META["dev"]),
        "predictions_sha256": sha256(predictions),
        "summary_sha256": sha256(summary_path),
        "validation": summary["validation"],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
