#!/usr/bin/env python3
"""Convert online CSLR result dictionaries into G2T annotation files."""

import argparse
import gzip
import os
import pickle
import tempfile
from pathlib import Path


def load_pickle(path):
    """Load a plain or gzip-compressed pickle file."""
    path = Path(path)
    try:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    except (gzip.BadGzipFile, OSError):
        with path.open("rb") as handle:
            return pickle.load(handle)


def resolve_results_file(pred_dir, split, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        return path

    if not pred_dir:
        raise ValueError(f"Provide --{split}-results or --pred-dir")
    candidates = [
        Path(pred_dir) / f"{split}_results.pkl",
        Path(pred_dir) / split / f"{split}_results.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    attempted = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Prediction file not found. Tried:\n  - {attempted}")


def load_reference(path):
    annotation = load_pickle(path)
    if not isinstance(annotation, list):
        raise TypeError(f"Reference annotation must be a list, got {type(annotation).__name__}")
    by_name = {}
    for item in annotation:
        name = item.get("name")
        if not name:
            raise ValueError(f"Reference item has no name: {item}")
        if name in by_name:
            raise ValueError(f"Duplicate reference name: {name}")
        by_name[name] = item
    return annotation, by_name


def normalize_gloss(value, name):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(token) for token in value).strip()
    raise TypeError(f"Unexpected gloss type for {name}: {type(value).__name__}")


def convert_to_g2t_format(results, reference, decode_method):
    """Merge predicted glosses with reference text and sequence metadata."""
    if not isinstance(results, dict):
        raise TypeError(f"CSLR results must be a dictionary, got {type(results).__name__}")

    reference_list, reference_by_name = reference
    result_names = set(results)
    reference_names = set(reference_by_name)
    missing = reference_names - result_names
    extra = result_names - reference_names
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} predictions, e.g. {sorted(missing)[:3]}")
        if extra:
            details.append(f"found {len(extra)} unknown predictions, e.g. {sorted(extra)[:3]}")
        raise ValueError("Prediction/reference name mismatch: " + "; ".join(details))

    output = []
    empty_predictions = []
    for reference_item in reference_list:
        name = reference_item["name"]
        result = results[name]
        if decode_method not in result:
            available = sorted(key for key in result if "gls_hyp" in key)
            raise KeyError(
                f"{name} has no decode key {decode_method!r}; "
                f"available hypotheses: {available}"
            )
        gloss = normalize_gloss(result[decode_method], name)
        if not gloss:
            empty_predictions.append(name)
        output.append(
            {
                "name": name,
                "num_frames": int(reference_item["num_frames"]),
                "gloss": gloss,
                "text": reference_item.get("text", ""),
            }
        )

    if len(empty_predictions) == len(output):
        raise ValueError(
            "Every gloss prediction is empty; refusing to create a broken G2T input"
        )
    if empty_predictions:
        print(
            f"warning: {len(empty_predictions)}/{len(output)} gloss predictions are empty, "
            f"e.g. {empty_predictions[:3]}"
        )
    return output


def atomic_pickle_dump(value, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=output_path.name + ".", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        pickle.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", help="Directory containing dev/test result files")
    parser.add_argument("--dev-results", help="Explicit dev result pickle")
    parser.add_argument("--test-results", help="Explicit test result pickle")
    parser.add_argument("--dev-reference", required=True, help="Dev reference annotation")
    parser.add_argument("--test-reference", required=True, help="Test reference annotation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--decode-method", default="window_greedy_5_gls_hyp")
    return parser.parse_args()


def main():
    args = parse_args()
    for split in ("dev", "test"):
        results_file = resolve_results_file(
            args.pred_dir, split, getattr(args, f"{split}_results")
        )
        reference_file = getattr(args, f"{split}_reference")
        results = load_pickle(results_file)
        converted = convert_to_g2t_format(
            results,
            load_reference(reference_file),
            args.decode_method,
        )
        output_file = Path(args.output_dir) / f"{args.output_prefix}.{split}"
        atomic_pickle_dump(converted, output_file)
        print(
            f"{split}: wrote {len(converted)} entries to {output_file} "
            f"from {results_file}"
        )
        print(f"{split}: sample={converted[0]}")


if __name__ == "__main__":
    main()
