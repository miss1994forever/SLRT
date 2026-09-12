#!/usr/bin/env python3
"""Replay Phoenix dev schedules from frozen stride-1 logits on CPU only.

The tool deliberately accepts only ``dev`` artifacts.  It validates that the
B0 source contains one logit row for every integer window start, then indexes
those rows with saved B2/A0 starts and applies the frozen span-15 decoder.
"""

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import numpy as np
import torch


CSLR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CSLR_ROOT))

from utils.adaptive_stride import span_weighted_predictions
from utils.metrics import wer_list, wer_single
from utils.phoenix_cleanup import clean_phoenix_2014_trans


DEFAULT_MATRIX_ROOT = (
    CSLR_ROOT
    / "results/phoenix-2014t_ISLR/baseline_matrix_v1_repaired_49faacc3"
)
DEFAULT_OUTPUT_ROOT = (
    CSLR_ROOT
    / "results/phoenix-2014t_ISLR/p0_schedule_diagnostics_v1_49faacc3"
)
DEFAULT_VOCAB = CSLR_ROOT.parents[1] / "data/phoenix_2014t/phoenix_iso_with_blank.vocab"
VARIANTS = {
    "B0_fixed1_span15_replay": "B0_fixed1_window7",
    "B2_uniform_rate_span15_from_B0": "B2_uniform_rate_span15",
    "A0_adaptive_span15_from_B0": "A0_adaptive_span15",
}
EXPECTED_MATRIX_MANIFEST_SHA256 = "2e6889ec14777d938a3c7ac9265ab97f87118523e9258f5f93beba4b3e854b8f"
EXPECTED_PROTOCOL_SHA256 = "3d583b1430def1ae0b517c79bf66abc8dae9528118b325c58620fd83e22f9f65"
EXPECTED_FULL_DEV = {
    "B0_fixed1_span15_replay": {"wer": 22.6047504670403, "error": 847, "clips": 55775},
    "B2_uniform_rate_span15_from_B0": {"wer": 22.551374432879637, "error": 845, "clips": 37706},
    "A0_adaptive_span15_from_B0": {"wer": 22.41793434747798, "error": 840, "clips": 37615},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_test_path(path):
    """Fail closed if an input path appears to address a test split."""
    lowered = [part.lower() for part in Path(path).resolve().parts]
    if any(part == "test" or part.startswith("test_") or part.endswith("_test") for part in lowered):
        raise ValueError(f"test paths are forbidden: {path}")


def load_pickle(path):
    reject_test_path(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def map_phoenix_gloss(gloss):
    # Keep byte-for-byte parity with prediction_slide.index2token.
    for prefix, offset in (("neg-", 4), ("poss-", 5), ("negalp-", 7), ("loc-", 7), ("cl-", 7)):
        if gloss.startswith(prefix):
            return prefix + gloss[offset:].upper()
    return gloss.upper()


def decode_span15(logits, starts, vocab, blank_id, span=15.0, min_weight=0.05):
    if logits.ndim != 2 or len(logits) != len(starts):
        raise ValueError("logits must be [N, C] with one row per start")
    centers = [float(start) + 7.5 for start in starts]
    token_ids = span_weighted_predictions(
        torch.from_numpy(logits), centers, span=span, min_weight=min_weight
    ).tolist()
    collapsed = [token for token, _ in groupby(token_ids) if token != blank_id]
    return " ".join(map_phoenix_gloss(vocab[token]) for token in collapsed)


def starts_from_results(results, expected_names=None):
    if expected_names is not None and list(results) != list(expected_names):
        raise ValueError("sample order differs from the dense B0 source")
    starts = {}
    for name, sample in results.items():
        metadata = sample.get("adaptive_stride_metadata")
        if not isinstance(metadata, list) or not metadata:
            raise ValueError(f"missing schedule metadata for {name}")
        values = [int(item["start"]) for item in metadata]
        if values != sorted(set(values)):
            raise ValueError(f"starts are not strictly increasing and unique for {name}")
        starts[name] = values
    return starts


def dense_start_index(dense_results, dense_logits):
    starts = starts_from_results(dense_results)
    indices = {}
    for name, values in starts.items():
        if name not in dense_logits:
            raise KeyError(f"B0 logits missing sample {name}")
        if len(dense_logits[name]) != len(values):
            raise ValueError(f"B0 logit/metadata length mismatch for {name}")
        expected = list(range(len(values)))
        if values != expected:
            raise ValueError(
                f"B0 is not dense stride-1 for {name}: expected 0..{len(values) - 1}"
            )
        indices[name] = {start: index for index, start in enumerate(values)}
    if set(dense_logits) != set(dense_results):
        raise ValueError("B0 result/logit sample sets differ")
    return indices


def extract_schedule_logits(name, dense_logits, start_indices, target_starts):
    try:
        rows = [start_indices[name][start] for start in target_starts]
    except KeyError as error:
        raise ValueError(f"schedule start {error.args[0]} is absent from dense B0 for {name}") from error
    return dense_logits[name][np.asarray(rows, dtype=np.int64)]


def align_starts_to_dense_inputs(dense_starts, target_starts, window_frames=16):
    """Map padded-array starts to starts representing the same input frames.

    ``prediction_slide.sliding_windows`` centers padding from the final sampled
    start.  Consequently, identical saved ``start`` values can address input
    windows shifted by one frame when schedules have different final starts.
    B0's dense coordinates must be corrected by the padding-left difference.
    """
    total_frames = len(dense_starts)
    dense_pad_left = (dense_starts[-1] + window_frames - total_frames) // 2
    target_pad_left = (target_starts[-1] + window_frames - total_frames) // 2
    offset = dense_pad_left - target_pad_left
    aligned = [start + offset for start in target_starts]
    if aligned[0] < dense_starts[0] or aligned[-1] > dense_starts[-1]:
        raise ValueError("target schedule addresses a window outside dense B0 coverage")
    return aligned, dense_pad_left, target_pad_left


def compact_metrics(references, hypotheses):
    metric = wer_list(references, hypotheses)
    return {
        key: float(metric[key]) if key in ("wer", "del", "ins", "sub") else int(metric[key])
        for key in ("wer", "del", "ins", "sub", "ref_len", "error")
    }


def load_run(run_dir):
    reject_test_path(run_dir)
    results_path = run_dir / "dev_results.pkl"
    logits_path = run_dir / "dev_logits.pkl"
    if not results_path.is_file() or not logits_path.is_file():
        raise FileNotFoundError(f"incomplete dev run: {run_dir}")
    return load_pickle(results_path), load_pickle(logits_path), results_path, logits_path


def replay(matrix_root, vocab_path, max_samples=None):
    reject_test_path(matrix_root)
    reject_test_path(vocab_path)
    matrix_manifest_path = matrix_root / "protocol_manifest.json"
    observed_manifest_sha256 = sha256_file(matrix_manifest_path)
    if observed_manifest_sha256 != EXPECTED_MATRIX_MANIFEST_SHA256:
        raise ValueError(
            "baseline matrix manifest identity differs from frozen repaired dev: "
            f"{observed_manifest_sha256}"
        )
    matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    if matrix_manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("baseline matrix protocol hash differs from frozen v1")
    b0_dir = matrix_root / "dev/B0_fixed1_window7/run_01"
    b0_results, b0_logits, b0_results_path, b0_logits_path = load_run(b0_dir)
    all_names = list(b0_results)
    names = all_names if max_samples is None else all_names[:max_samples]
    if not names:
        raise ValueError("no dev samples selected")
    b0_results = {name: b0_results[name] for name in names}
    b0_logits = {name: b0_logits[name] for name in names}
    start_indices = dense_start_index(b0_results, b0_logits)

    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    blank_id = vocab.index("<blank>")
    references = [clean_phoenix_2014_trans(b0_results[name]["gls_ref"]) for name in names]
    per_sample = {name: {"name": name, "reference": references[index]} for index, name in enumerate(names)}
    summaries = {}
    input_paths = [matrix_manifest_path, b0_results_path, b0_logits_path, vocab_path]

    for replay_id, schedule_id in VARIANTS.items():
        if schedule_id == "B0_fixed1_window7":
            schedule_results = b0_results
            native_logits = b0_logits
            results_path, logits_path = b0_results_path, b0_logits_path
        else:
            run_dir = matrix_root / "dev" / schedule_id / "run_01"
            loaded_results, loaded_logits, results_path, logits_path = load_run(run_dir)
            if list(loaded_results) != all_names or list(loaded_logits) != all_names:
                raise ValueError(f"sample order differs for {schedule_id}")
            schedule_results = {name: loaded_results[name] for name in names}
            native_logits = {name: loaded_logits[name] for name in names}
            input_paths.extend([results_path, logits_path])

        for name in names:
            if schedule_results[name]["gls_ref"] != b0_results[name]["gls_ref"]:
                raise ValueError(f"reference differs for {schedule_id}: {name}")

        schedules = starts_from_results(schedule_results, names)
        replay_hypotheses = []
        native_replay_hypotheses = []
        stored_hypotheses = []
        exact_arrays = 0
        differing_values = 0
        maximum_abs_difference = 0.0
        clip_count = 0
        padding_offset_counts = {}
        for name in names:
            starts = schedules[name]
            dense_starts = list(start_indices[name])
            aligned_starts, dense_pad_left, target_pad_left = align_starts_to_dense_inputs(
                dense_starts, starts
            )
            selected = extract_schedule_logits(name, b0_logits, start_indices, aligned_starts)
            native = native_logits[name]
            if selected.shape != native.shape:
                raise ValueError(f"selected/native logit shape mismatch for {schedule_id}: {name}")
            exact = bool(np.array_equal(selected, native))
            exact_arrays += int(exact)
            if not exact:
                difference = np.abs(selected.astype(np.float64) - native.astype(np.float64))
                differing_values += int(np.count_nonzero(difference))
                maximum_abs_difference = max(maximum_abs_difference, float(difference.max()))

            replay_hyp = clean_phoenix_2014_trans(decode_span15(selected, starts, vocab, blank_id))
            native_hyp = clean_phoenix_2014_trans(decode_span15(native, starts, vocab, blank_id))
            stored_hyp = clean_phoenix_2014_trans(schedule_results[name]["span_weighted_15_gls_hyp"])
            replay_hypotheses.append(replay_hyp)
            native_replay_hypotheses.append(native_hyp)
            stored_hypotheses.append(stored_hyp)
            clip_count += len(starts)
            padding_offset = dense_pad_left - target_pad_left
            padding_offset_counts[str(padding_offset)] = padding_offset_counts.get(str(padding_offset), 0) + 1
            sentence = wer_single(per_sample[name]["reference"], replay_hyp)
            per_sample[name][replay_id] = {
                "clips": len(starts),
                "dense_input_start_offset": padding_offset,
                "hypothesis": replay_hyp,
                "errors": int(sentence["num_err"]),
                "native_logits_exact": exact,
                "matches_native_replay_hypothesis": replay_hyp == native_hyp,
                "matches_stored_hypothesis": replay_hyp == stored_hyp,
            }

        summaries[replay_id] = {
            "schedule_source": schedule_id,
            "extraction_coordinates": "same_input_frames_after_centered_padding_correction",
            "clip_count": clip_count,
            "dense_input_start_offset_sample_counts": dict(sorted(padding_offset_counts.items())),
            "metrics": compact_metrics(references, replay_hypotheses),
            "stored_metrics": compact_metrics(references, stored_hypotheses),
            "native_replay_metrics": compact_metrics(references, native_replay_hypotheses),
            "replay_vs_stored_hypotheses_equal": sum(
                left == right for left, right in zip(replay_hypotheses, stored_hypotheses)
            ),
            "native_replay_vs_stored_hypotheses_equal": sum(
                left == right for left, right in zip(native_replay_hypotheses, stored_hypotheses)
            ),
            "selected_vs_native_exact_logit_arrays": exact_arrays,
            "selected_vs_native_differing_logit_values": differing_values,
            "selected_vs_native_max_abs_difference": maximum_abs_difference,
        }

        if max_samples is None:
            expected = EXPECTED_FULL_DEV[replay_id]
            observed = summaries[replay_id]
            observed["frozen_reproduction_passed"] = (
                observed["metrics"]["error"] == expected["error"]
                and observed["clip_count"] == expected["clips"]
                and abs(observed["metrics"]["wer"] - expected["wer"]) <= 1e-12
                and observed["replay_vs_stored_hypotheses_equal"] == len(names)
            )

    return {
        "sample_count": len(names),
        "full_dev": max_samples is None,
        "span_frames": 15.0,
        "span_kernel": "triangular",
        "span_min_weight": 0.05,
        "window_frames": 16,
        "frozen_matrix_manifest_sha256": observed_manifest_sha256,
        "variants": summaries,
    }, list(per_sample.values()), sorted(set(input_paths))


def write_outputs(output_root, config, summary, per_sample, input_paths):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (aggregate_dir / "dev_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "per_sample_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in per_sample:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Phoenix-2014T repaired dev only; CPU logit replay; no model forward",
        "inputs": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        },
    }
    (output_root / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true", help="validate and print without writing")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    config = {
        "matrix_root": str(args.matrix_root.resolve()),
        "vocab": str(args.vocab.resolve()),
        "output_root": str(args.output_root.resolve()),
        "max_samples": args.max_samples,
        "device": "cpu",
    }
    summary, per_sample, input_paths = replay(args.matrix_root.resolve(), args.vocab.resolve(), args.max_samples)
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        write_outputs(args.output_root.resolve(), config, summary, per_sample, input_paths)


if __name__ == "__main__":
    main()
