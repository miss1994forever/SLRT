#!/usr/bin/env python3
"""Audit H4 reward stability across the two already frozen future continuations."""
import argparse
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
LABELS = BASE / "p5_h4_schedule_label_dataset_v1_49faacc3"
OUTPUT = BASE / "p5_h4_target_stability_audit_v1_49faacc3"
CONFIG = ROOT / "configs/p5_h4_target_stability_audit_v1_preregistered.json"
CENTER = 40


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare():
    config = json.loads(CONFIG.read_text())
    assets = {"label_metrics_sha256": LABELS / "metrics.json",
              "label_manifest_sha256": LABELS / "manifest.json"}
    for key, path in assets.items():
        if sha256_file(path) != config["fixed_assets"][key]:
            raise RuntimeError(f"asset mismatch: {key}")
    identity_path = OUTPUT / "preregistered_identity.json"
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"nonempty output lacks identity: {OUTPUT}")
        identity = json.loads(identity_path.read_text())
        if identity["config_sha256"] != sha256_file(CONFIG):
            raise RuntimeError("config changed after prepare")
        return identity
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity = {"created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
                "script_sha256_at_prepare": sha256_file(Path(__file__)), "diagnostic_only": True}
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def load_h4():
    center, late, sources = [], [], []
    for path in sorted((LABELS / "shards").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for horizon in row["horizons"]:
                    if horizon["horizon_size"] != 4:
                        continue
                    center.append(horizon["reward_center_future"])
                    late.append(horizon["reward_late_future"])
                    sources.append(row["source_video_id"])
    return np.asarray(center, np.int16), np.asarray(late, np.int16), np.asarray(sources)


def run():
    prepare()
    if (OUTPUT / "metrics.json").exists():
        raise FileExistsError("non-overwriting stability audit exists")
    center, late, sources = load_h4()
    if center.shape != (43437, 81) or late.shape != center.shape:
        raise RuntimeError(f"scope mismatch: {center.shape}, {late.shape}")
    center_max, late_max = center.max(1), late.max(1)
    center_opp, late_opp = center_max > 0, late_max > 0
    robust = np.minimum(center, late)
    robust_opp = robust.max(1) > 0
    union = center_opp | late_opp
    intersection = center_opp & late_opp
    center_choice = np.where(center_opp, center.argmax(1), CENTER)
    late_choice = np.where(late_opp, late.argmax(1), CENTER)
    active = (center != 0) | (late != 0)
    c_active, l_active = center[active].astype(np.float64), late[active].astype(np.float64)
    sign_equal = np.sign(c_active) == np.sign(l_active)
    both_choice_noncenter = center_opp & late_opp
    sources_with_robust = len(set(sources[robust_opp].tolist()))
    metrics = {
        "status": "complete", "created_utc": utcnow(), "gpu_used": False,
        "diagnostic_only": True, "calibration_dev_test_heldout_or_closed_loop_read": False,
        "horizons": len(center), "candidate_schedules": int(center.size),
        "center_future": {"opportunities": int(center_opp.sum()),
                          "prevalence": float(center_opp.mean()),
                          "summed_best_reward": int(np.maximum(center_max, 0).sum())},
        "late_future": {"opportunities": int(late_opp.sum()),
                        "prevalence": float(late_opp.mean()),
                        "summed_best_reward": int(np.maximum(late_max, 0).sum())},
        "opportunity_overlap": {
            "union": int(union.sum()), "intersection": int(intersection.sum()),
            "jaccard": float(intersection.sum() / max(1, union.sum())),
            "center_only": int((center_opp & ~late_opp).sum()),
            "late_only": int((late_opp & ~center_opp).sum()),
        },
        "best_schedule_agreement": {
            "among_both_opportunity_horizons": float(np.mean(center_choice[both_choice_noncenter] == late_choice[both_choice_noncenter])),
            "exact_agreement_count": int(np.sum((center_choice == late_choice) & both_choice_noncenter)),
            "both_opportunity_horizons": int(both_choice_noncenter.sum()),
        },
        "candidate_reward_stability": {
            "active_candidate_pairs": int(active.sum()),
            "pearson": float(np.corrcoef(c_active, l_active)[0, 1]),
            "sign_agreement": float(sign_equal.mean()),
            "both_positive": int(np.sum((c_active > 0) & (l_active > 0))),
            "opposite_sign": int(np.sum(c_active * l_active < 0)),
        },
        "robust_target": {
            "opportunities": int(robust_opp.sum()),
            "prevalence": float(robust_opp.mean()),
            "summed_best_reward": int(np.maximum(robust.max(1), 0).sum()),
            "fraction_of_union_retained": float(robust_opp.sum() / max(1, union.sum())),
            "fraction_of_intersection_retained": float(robust_opp.sum() / max(1, intersection.sum())),
            "sources_with_opportunity": sources_with_robust,
        },
        "decision": "target-health diagnostic only; no policy unlock",
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {"status": "complete", "config_sha256": sha256_file(CONFIG),
                "identity_sha256": sha256_file(OUTPUT / "preregistered_identity.json"),
                "metrics_sha256": sha256_file(OUTPUT / "metrics.json"), "diagnostic_only": True}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = prepare() if args.prepare else run()
    print(json.dumps(result, indent=2))
