#!/usr/bin/env python3
"""Validate and aggregate a frozen adaptive-stride baseline matrix."""

import argparse
import csv
import hashlib
import json
import pickle
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CSLR_ROOT))

from utils.metrics import wer_list, wer_single
from utils.phoenix_cleanup import clean_phoenix_2014_trans


DEFAULT_PROTOCOL = CSLR_ROOT / "configs" / "experiments" / "phoenix_adaptive_baselines_v1.yaml"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else (CSLR_ROOT / path).resolve()


def load_protocol(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        return path, yaml.safe_load(handle)


def successful_run_dirs(output_root, split, variant_id):
    root = output_root / split / variant_id
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("run_*") if (path / "complete.json").is_file())


def runtime_stats(run_dirs, split):
    records = []
    profiles = []
    for run_dir in run_dirs:
        path = run_dir / "runtime.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("returncode") == 0:
                records.append(record)
        profile_path = run_dir / f"{split}_runtime_profile.json"
        if profile_path.is_file():
            profiles.append(json.loads(profile_path.read_text(encoding="utf-8")))
    values = [float(record["wall_time_seconds"]) for record in records]
    if not values:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    summary = {
        "repetitions": len(values),
        "mean_seconds": mean,
        "median_seconds": statistics.median(values),
        "std_seconds": std,
        "cv_percent": 100.0 * std / mean if mean else None,
        "values_seconds": values,
    }
    forward_values = [float(item["model_forward_seconds"]) for item in profiles]
    if forward_values:
        summary["model_forward_median_seconds"] = statistics.median(forward_values)
        summary["model_forward_values_seconds"] = forward_values
        summary["peak_cuda_memory_allocated_bytes"] = max(
            item["peak_cuda_memory_allocated_bytes"] or 0 for item in profiles
        )
    return summary


def validate_run_configuration(protocol, source_id, run_dir, split):
    config_path = run_dir / "resolved_config.yaml"
    command_path = run_dir / "command.json"
    if not config_path.is_file() or not command_path.is_file():
        raise FileNotFoundError(f"missing resolved config or command in {run_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    command = json.loads(command_path.read_text(encoding="utf-8"))
    specification = protocol["variants"][source_id]
    common = protocol["common_inference"]
    if config["data"]["sampling"] != specification["sampling"]:
        raise ValueError(f"sampling config differs from protocol in {run_dir}")
    span = config["postprocess"]["span_weighted_voting"]
    expected_span = {
        "enabled": bool(specification.get("span_weighted_voting", False)),
        "vote_span_frames": int(common["span_frames"]),
        "kernel": common["span_kernel"],
        "min_weight": float(common["span_min_weight"]),
    }
    for key, value in expected_span.items():
        if span[key] != value:
            raise ValueError(f"span config field {key} differs from protocol in {run_dir}")
    required_arguments = {
        "--split": split,
        "--pred_src": common["prediction_source"],
        "--blank_thr": str(common["blank_threshold"]),
        "--ckpt_name": Path(protocol["checkpoint"]).name,
    }
    for argument, expected in required_arguments.items():
        try:
            observed = command[command.index(argument) + 1]
        except (ValueError, IndexError) as error:
            raise ValueError(f"{argument} missing from command in {run_dir}") from error
        if str(observed) != str(expected):
            raise ValueError(f"{argument} differs from protocol in {run_dir}")


def load_variant(protocol, output_root, split, variant_id):
    variant = protocol["variants"][variant_id]
    source_id = variant.get("reuse_forward_from", variant_id)
    run_dirs = successful_run_dirs(output_root, split, source_id)
    if not run_dirs:
        raise FileNotFoundError(f"no completed run for {variant_id} (source {source_id})")
    run_dir = run_dirs[0]
    validate_run_configuration(protocol, source_id, run_dir, split)
    with (run_dir / f"{split}_results.pkl").open("rb") as handle:
        results = pickle.load(handle)
    with (run_dir / f"{split}_evaluation_results.pkl").open("rb") as handle:
        evaluation = pickle.load(handle)
    decoder = variant["report_decoder"]
    metric_key = f"wer_{decoder}"
    if metric_key not in evaluation:
        raise KeyError(f"{metric_key} missing for {variant_id} in {run_dir}")

    references = {}
    hypotheses = {}
    sentence_errors = {}
    sentence_ref_lengths = {}
    hypothesis_key = f"{decoder}_gls_hyp"
    clips_per_sample = []
    stride_counts = {}
    sampling_modes = {}
    for name, sample in results.items():
        if hypothesis_key not in sample:
            raise KeyError(f"{hypothesis_key} missing for {name} in {variant_id}")
        reference = clean_phoenix_2014_trans(sample["gls_ref"])
        hypothesis = clean_phoenix_2014_trans(sample[hypothesis_key])
        references[name] = reference
        hypotheses[name] = hypothesis
        sentence = wer_single(reference, hypothesis)
        sentence_errors[name] = int(sentence["num_err"])
        sentence_ref_lengths[name] = int(sentence["num_ref"])
        metadata = sample.get("adaptive_stride_metadata", [])
        clips_per_sample.append(len(metadata))
        for item in metadata:
            stride = str(int(item["stride"]))
            stride_counts[stride] = stride_counts.get(stride, 0) + 1
            mode = item.get("sampling_mode", "legacy")
            sampling_modes[mode] = sampling_modes.get(mode, 0) + 1

    metric = evaluation[metric_key]
    recomputed = wer_list(
        references=list(references.values()),
        hypotheses=list(hypotheses.values()),
    )
    for key in ("ref_len", "error"):
        if int(recomputed[key]) != int(metric[key]):
            raise ValueError(f"stored {key} differs from recomputation for {variant_id}")
    if abs(float(recomputed["wer"]) - float(metric["wer"])) > 1e-9:
        raise ValueError(f"stored WER differs from recomputation for {variant_id}")
    stripped_metric = {
        key: float(recomputed[key]) if key in ("wer", "del", "ins", "sub") else int(recomputed[key])
        for key in ("wer", "del", "ins", "sub", "ref_len", "error")
    }
    for repeat_dir in run_dirs[1:]:
        validate_run_configuration(protocol, source_id, repeat_dir, split)
        with (repeat_dir / f"{split}_results.pkl").open("rb") as handle:
            repeated_results = pickle.load(handle)
        with (repeat_dir / f"{split}_evaluation_results.pkl").open("rb") as handle:
            repeated_evaluation = pickle.load(handle)
        if list(repeated_results) != list(results):
            raise ValueError(f"sample order differs between repetitions for {variant_id}")
        for name, sample in results.items():
            repeated = repeated_results[name]
            if repeated["gls_ref"] != sample["gls_ref"] or repeated[hypothesis_key] != sample[hypothesis_key]:
                raise ValueError(f"prediction differs between repetitions for {variant_id}: {name}")
            original_schedule = [
                (item["start"], item["stride"]) for item in sample.get("adaptive_stride_metadata", [])
            ]
            repeated_schedule = [
                (item["start"], item["stride"]) for item in repeated.get("adaptive_stride_metadata", [])
            ]
            if repeated_schedule != original_schedule:
                raise ValueError(f"sampling schedule differs between repetitions for {variant_id}: {name}")
        repeated_metric = repeated_evaluation[metric_key]
        if any(repeated_metric[key] != metric[key] for key in metric):
            raise ValueError(f"stored metrics differ between repetitions for {variant_id}")
    return {
        "variant_id": variant_id,
        "source_variant_id": source_id,
        "decoder": decoder,
        "run_dir": str(run_dir),
        "sample_count": len(results),
        "metric": stripped_metric,
        "clip_count": sum(clips_per_sample),
        "clips_per_sample": {
            "mean": float(np.mean(clips_per_sample)),
            "p50": float(np.percentile(clips_per_sample, 50)),
            "p95": float(np.percentile(clips_per_sample, 95)),
            "min": int(min(clips_per_sample)),
            "max": int(max(clips_per_sample)),
        },
        "stride_counts": dict(sorted(stride_counts.items(), key=lambda item: int(item[0]))),
        "sampling_mode_counts": dict(sorted(sampling_modes.items())),
        "runtime": runtime_stats(run_dirs, split),
        "references": references,
        "hypotheses": hypotheses,
        "sentence_errors": sentence_errors,
        "sentence_ref_lengths": sentence_ref_lengths,
    }


def compare_references(variants):
    first = variants[0]
    expected_names = list(first["references"])
    expected = first["references"]
    for variant in variants[1:]:
        if list(variant["references"]) != expected_names:
            raise ValueError(f"sample order differs: {first['variant_id']} vs {variant['variant_id']}")
        mismatched = [name for name in expected_names if variant["references"][name] != expected[name]]
        if mismatched:
            raise ValueError(f"references differ for {variant['variant_id']}: {mismatched[:3]}")


def paired_bootstrap(candidate, baseline, samples=1000, seed=321):
    names = list(baseline["references"])
    candidate_error = np.asarray([candidate["sentence_errors"][name] for name in names], dtype=float)
    baseline_error = np.asarray([baseline["sentence_errors"][name] for name in names], dtype=float)
    ref_len = np.asarray([baseline["sentence_ref_lengths"][name] for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.integers(0, len(names), size=len(names))
        denominator = ref_len[chosen].sum()
        differences[index] = 100.0 * (
            candidate_error[chosen].sum() - baseline_error[chosen].sum()
        ) / denominator
    return {
        "samples": samples,
        "seed": seed,
        "wer_difference_pp": candidate["metric"]["wer"] - baseline["metric"]["wer"],
        "ci95_low_pp": float(np.percentile(differences, 2.5)),
        "ci95_high_pp": float(np.percentile(differences, 97.5)),
    }


def public_variant(variant):
    return {key: value for key, value in variant.items() if key not in {
        "references", "hypotheses", "sentence_errors", "sentence_ref_lengths"
    }}


def add_comparisons(variants):
    by_id = {variant["variant_id"]: variant for variant in variants}
    baseline_ids = [key for key in ("B0_fixed1_window7", "B1_fixed1_span15", "B2_uniform_rate_span15") if key in by_id]
    comparisons = {}
    for variant in variants:
        row = {}
        for baseline_id in baseline_ids:
            baseline = by_id[baseline_id]
            row[baseline_id] = {
                "wer_change_pp": variant["metric"]["wer"] - baseline["metric"]["wer"],
                "clip_reduction_percent": 100.0 * (1.0 - variant["clip_count"] / baseline["clip_count"]),
                "paired_bootstrap": paired_bootstrap(variant, baseline),
            }
            candidate_runtime = variant.get("runtime")
            baseline_runtime = baseline.get("runtime")
            if candidate_runtime and baseline_runtime:
                candidate_time = candidate_runtime["median_seconds"]
                baseline_time = baseline_runtime["median_seconds"]
                row[baseline_id]["wall_time_reduction_percent"] = 100.0 * (1.0 - candidate_time / baseline_time)
                row[baseline_id]["wall_time_speedup"] = baseline_time / candidate_time
        comparisons[variant["variant_id"]] = row
    return comparisons


def validate_budget_constraints(protocol, variants):
    """Check any pre-registered compute-budget match against observed clip counts."""
    by_id = {variant["variant_id"]: variant for variant in variants}
    checks = {}
    for variant_id, specification in protocol["variants"].items():
        reference_id = specification.get("budget_reference")
        if not reference_id:
            continue
        if variant_id not in by_id or reference_id not in by_id:
            raise ValueError(f"budget check is missing {variant_id} or {reference_id}")
        observed = by_id[variant_id]["clip_count"]
        reference = by_id[reference_id]["clip_count"]
        difference = 100.0 * abs(observed - reference) / reference
        maximum = float(specification["maximum_budget_difference_percent"])
        checks[variant_id] = {
            "reference_variant_id": reference_id,
            "observed_clips": observed,
            "reference_clips": reference,
            "absolute_difference_percent": difference,
            "maximum_difference_percent": maximum,
            "passed": difference <= maximum,
        }
        if difference > maximum:
            raise ValueError(
                f"{variant_id} budget differs from {reference_id} by {difference:.3f}%, "
                f"exceeding {maximum:.3f}%"
            )
    return checks


def render_markdown(summary):
    lines = [
        f"# {summary['experiment_id']} {summary['split']} results",
        "",
        "| Variant | Decoder | WER | DEL | INS | SUB | Clips | Runtime median (s) | Model forward median (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in summary["variants"]:
        metric = variant["metric"]
        runtime = variant.get("runtime")
        runtime_value = f"{runtime['median_seconds']:.2f}" if runtime else "—"
        forward_value = (
            f"{runtime['model_forward_median_seconds']:.2f}"
            if runtime and "model_forward_median_seconds" in runtime else "—"
        )
        lines.append(
            f"| {variant['variant_id']} | {variant['decoder']} | {metric['wer']:.4f}% | "
            f"{metric['del']:.4f}% | {metric['ins']:.4f}% | {metric['sub']:.4f}% | "
            f"{variant['clip_count']:,} | {runtime_value} | {forward_value} |"
        )
    lines.extend(["", "Generated by `tools/evaluate_adaptive_baseline_matrix.py`.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--frozen-manifest-sha256")
    args = parser.parse_args()
    protocol_path, protocol = load_protocol(args.protocol)
    output_root = (args.output_root or resolve_path(protocol["output_root"])).resolve()
    manifest_path = output_root / "protocol_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing protocol manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("protocol and manifest hashes differ")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=CSLR_ROOT, text=True
    ).strip()
    if manifest["git"]["commit"] != current_commit:
        raise ValueError("current Git commit differs from the frozen manifest")
    if args.split == "test":
        if not args.allow_test or not args.frozen_manifest_sha256:
            raise ValueError("test evaluation requires --allow-test and --frozen-manifest-sha256")
        actual = sha256_file(manifest_path)
        if actual != args.frozen_manifest_sha256:
            raise ValueError("frozen manifest hash mismatch")

    variants = [load_variant(protocol, output_root, args.split, variant_id) for variant_id in protocol["variants"]]
    compare_references(variants)
    policy_prefix = "development" if args.split == "dev" else "test"
    expected_samples = int(protocol["data_policy"][f"{policy_prefix}_samples"])
    expected_ref = int(protocol["data_policy"][f"{policy_prefix}_reference_glosses"])
    if not args.allow_partial:
        for variant in variants:
            if variant["sample_count"] != expected_samples:
                raise ValueError(f"{variant['variant_id']} sample count mismatch")
            if variant["metric"]["ref_len"] != expected_ref:
                raise ValueError(f"{variant['variant_id']} reference length mismatch")

    summary = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "split": args.split,
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
        "variants": [public_variant(variant) for variant in variants],
        "comparisons": add_comparisons(variants),
        "budget_checks": validate_budget_constraints(protocol, variants),
    }
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    json_path = aggregate_dir / f"{args.split}_summary.json"
    csv_path = aggregate_dir / f"{args.split}_summary.csv"
    markdown_path = aggregate_dir / f"{args.split}_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant_id", "source_variant_id", "decoder", "wer", "del", "ins", "sub",
                "ref_len", "error", "clips", "runtime_median_seconds",
                "model_forward_median_seconds", "peak_cuda_memory_allocated_bytes",
            ],
        )
        writer.writeheader()
        for variant in variants:
            metric = variant["metric"]
            writer.writerow(
                {
                    "variant_id": variant["variant_id"],
                    "source_variant_id": variant["source_variant_id"],
                    "decoder": variant["decoder"],
                    **metric,
                    "clips": variant["clip_count"],
                    "runtime_median_seconds": variant["runtime"]["median_seconds"] if variant["runtime"] else "",
                    "model_forward_median_seconds": (
                        variant["runtime"].get("model_forward_median_seconds", "")
                        if variant["runtime"] else ""
                    ),
                    "peak_cuda_memory_allocated_bytes": (
                        variant["runtime"].get("peak_cuda_memory_allocated_bytes", "")
                        if variant["runtime"] else ""
                    ),
                }
            )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json_path)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
