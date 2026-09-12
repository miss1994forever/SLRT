#!/usr/bin/env python3
"""Audit Phoenix P1 train-only sign-interior proxy targets.

This tool intentionally does not open the monolithic keypoint pickle because it
also contains the forbidden test split.  The previously published train-only
non-finite summary is incorporated as an explicitly attributed prior audit.
"""

import argparse
import gzip
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


KNOWN_NONFINITE_TRAIN = {
    "train/11September_2010_Saturday_tagesschau-5000": 1,
    "train/14October_2010_Thursday_tagesschau-288": 2,
    "train/10March_2011_Thursday_heute-50": 1,
    "train/06October_2011_Thursday_tagesschau-822": 2,
    "train/07December_2010_Tuesday_tagesschau-4151": 1,
    "train/01September_2010_Wednesday_tagesschau-5033": 2,
    "train/29August_2009_Saturday_tagesschau-5026": 1,
    "train/26November_2011_Saturday_tagesschau-5850": 2,
    "train/09August_2010_Monday_heute-5894": 1,
}


def reject_test_path(path):
    parts = [part.lower() for part in Path(path).resolve().parts]
    if any(part == "test" or part.startswith("test_") or part.endswith("_test") for part in parts):
        raise ValueError(f"test paths are forbidden: {path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_meta(path):
    reject_test_path(path)
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def load_pickle(path):
    reject_test_path(path)
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def normalize_gloss(label):
    return str(label).upper()


def audit(train_bags_path, train_meta_path, dev_center_path):
    bags = load_pickle(train_bags_path)
    train_meta = load_meta(train_meta_path)
    dev_center = load_pickle(dev_center_path)

    problems = Counter()
    base_records = []
    duplicate_base_records = 0
    bag_id_mismatch = 0
    for key, records in bags.items():
        if not records:
            problems["empty_bags"] += 1
            continue
        base = [item for item in records if int(item.get("aug", -1)) == 0]
        if len(base) != 1:
            problems["bags_without_exactly_one_base"] += 1
            continue
        item = base[0]
        base_records.append(item)
        if str(item.get("bag")) != str(key):
            bag_id_mismatch += 1
        identities = [(r.get("video_file"), r.get("label"), r.get("start"), r.get("end"), r.get("aug")) for r in records]
        duplicate_base_records += len(identities) - len(set(identities))

    meta_by_name = {item["name"]: item for item in train_meta}
    by_video = defaultdict(list)
    for item in base_records:
        by_video[item["video_file"]].append(item)

    out_of_range = empty_segments = base_field_mismatch = 0
    overlaps_all = overlaps_nonblank = 0
    overlap_videos_all = overlap_videos_nonblank = 0
    repeated_exact_segments = 0
    reference_mismatches = []
    label_counts = Counter()
    segment_identity_counts = Counter()
    for name, records in by_video.items():
        length = int(meta_by_name[name]["num_frames"])
        ordered = sorted(records, key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
        previous_all = previous_nonblank = None
        video_overlap_all = video_overlap_nonblank = False
        for item in ordered:
            start, end = int(item["start"]), int(item["end"])
            label = str(item["label"])
            label_counts[label] += 1
            segment_identity_counts[(name, label, start, end)] += 1
            empty_segments += start >= end
            out_of_range += start < 0 or end > length
            base_field_mismatch += int(item.get("base_start", start)) != start or int(item.get("base_end", end)) != end
            if previous_all is not None and start < previous_all:
                overlaps_all += 1
                video_overlap_all = True
            previous_all = max(previous_all or 0, end)
            if label != "<blank>":
                if previous_nonblank is not None and start < previous_nonblank:
                    overlaps_nonblank += 1
                    video_overlap_nonblank = True
                previous_nonblank = max(previous_nonblank or 0, end)
        nonblank = [normalize_gloss(item["label"]) for item in ordered if item["label"] != "<blank>"]
        reference = str(meta_by_name[name]["gloss"]).upper().split()
        if nonblank != reference:
            reference_mismatches.append({"name": name, "reference": reference, "segments": nonblank})
        overlap_videos_all += video_overlap_all
        overlap_videos_nonblank += video_overlap_nonblank

    repeated_exact_segments = sum(count - 1 for count in segment_identity_counts.values() if count > 1)
    midpoint_positive_rows = 0
    total_frame_rows = sum(int(item["num_frames"]) for item in train_meta)
    for name, records in by_video.items():
        length = int(meta_by_name[name]["num_frames"])
        positive = set()
        for item in records:
            if item["label"] == "<blank>":
                continue
            center = (int(item["start"]) + int(item["end"]) - 1) // 2
            positive.update(range(max(0, center - 1), min(length, center + 2)))
        midpoint_positive_rows += len(positive)
    train_names = set(meta_by_name)
    dev_names = {str(item["video_file"]) for item in dev_center}
    train_ids = {name.split("/", 1)[-1] for name in train_names}
    dev_ids = {name.split("/", 1)[-1] for name in dev_names}
    dev_bags = {str(item.get("bag")) for item in dev_center}
    train_bags = {str(item.get("bag")) for item in base_records}

    affected = {}
    for name, count in KNOWN_NONFINITE_TRAIN.items():
        segments = by_video.get(name, [])
        affected[name] = {
            "nonfinite_xy_scalars_from_prior_audit": count,
            "num_frames": int(meta_by_name[name]["num_frames"]),
            "base_segments": len(segments),
            "nonblank_segments": sum(item["label"] != "<blank>" for item in segments),
            "candidate_frame_rows_conservatively_affected_if_video_excluded": int(meta_by_name[name]["num_frames"]),
        }

    return {
        "schema_version": 1,
        "scope": "train target audit plus dev identity-only leakage check; test not opened",
        "proxy_identity": "alignment-derived segment proxy; not manual frame-level ground truth",
        "coverage": {
            "train_meta_records": len(train_meta),
            "train_unique_videos": len(train_names),
            "train_bags": len(bags),
            "train_base_segments": len(base_records),
            "train_nonblank_segments": sum(item["label"] != "<blank>" for item in base_records),
            "train_blank_segments": label_counts["<blank>"],
            "videos_with_segments": len(by_video),
            "train_frame_rows": total_frame_rows,
            "radius1_midpoint_positive_rows": midpoint_positive_rows,
            "radius1_midpoint_positive_fraction": midpoint_positive_rows / total_frame_rows,
            "missing_segment_videos": sorted(train_names - set(by_video)),
            "extra_segment_videos": sorted(set(by_video) - train_names),
        },
        "integrity": {
            "empty_bags": problems["empty_bags"],
            "bags_without_exactly_one_base": problems["bags_without_exactly_one_base"],
            "bag_id_mismatch": bag_id_mismatch,
            "duplicate_records_within_bags": duplicate_base_records,
            "duplicate_exact_base_segments": repeated_exact_segments,
            "empty_or_reversed_base_segments": empty_segments,
            "out_of_range_base_segments": out_of_range,
            "base_start_end_mismatch": base_field_mismatch,
            "adjacent_overlaps_all_segments": overlaps_all,
            "videos_with_any_adjacent_overlap": overlap_videos_all,
            "adjacent_overlaps_nonblank_segments": overlaps_nonblank,
            "videos_with_nonblank_overlap": overlap_videos_nonblank,
            "reference_mismatch_videos": len(reference_mismatches),
            "reference_mismatch_examples": reference_mismatches[:10],
        },
        "leakage": {
            "dev_unique_videos_identity_check_only": len(dev_names),
            "exact_train_dev_video_name_overlap": len(train_names & dev_names),
            "train_dev_video_id_overlap_after_removing_split_prefix": len(train_ids & dev_ids),
            "overlapping_video_ids": sorted(train_ids & dev_ids),
            "train_dev_bag_id_overlap": len(train_bags & dev_bags),
            "bag_id_interpretation": "bag ids are split-local integers and must be namespaced by split; numeric overlap is not video leakage",
        },
        "label_summary": {
            "unique_labels_including_blank": len(label_counts),
            "top_10": label_counts.most_common(10),
        },
        "known_train_keypoint_nonfinite": {
            "source": "code_agent_logs/2026-09-07/phoenix_data_integrity_audit.md",
            "independently_rechecked_here": False,
            "reason_not_rechecked": "the only local keypoint artifact is a monolithic train/dev/test pickle; opening it would violate the no-test-read policy",
            "affected_videos": len(KNOWN_NONFINITE_TRAIN),
            "nonfinite_xy_scalars": sum(KNOWN_NONFINITE_TRAIN.values()),
            "affected_video_detail": affected,
            "total_candidate_frame_rows_if_excluding_affected_videos": sum(value["num_frames"] for value in affected.values()),
            "maximum_candidate_rows_influenced_by_13_separate_anomalies_with_48_frame_history": min(total_frame_rows, 13 * 49),
            "impact_bound_note": "one invalid coordinate can affect validity/bbox at its frame, adjacent motion, and past-only rolling features through at most frame+48; exact frames/landmarks cannot be rechecked without opening the mixed-split pickle",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bags", type=Path, required=True)
    parser.add_argument("--train-meta", type=Path, required=True)
    parser.add_argument("--dev-center", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.train_bags, args.train_meta, args.dev_center)
    result["inputs"] = {
        str(path.resolve()): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (args.train_bags, args.train_meta, args.dev_center)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
