#!/usr/bin/env python3
"""Train-only HRNet positive-mass heatmap pilot, reusing the frozen v1 sample."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import run_phoenix_hrnet_heatmap_uncertainty_pilot as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p4_hrnet_positive_mass_uncertainty_pilot_v1_preregistered.json"
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_uncertainty_pilot_v1_49faacc3"
SELECTION = OUTPUT / "selection.json"
PRIOR_SELECTION = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_heatmap_uncertainty_pilot_v1_49faacc3/selection.json"
XX = np.arange(72, dtype=np.float32)
YY = np.arange(96, dtype=np.float32)


def configure_base():
    base.CONFIG = CONFIG
    base.OUTPUT = OUTPUT
    base.SELECTION = SELECTION
    base.heatmap_stats = heatmap_stats


def heatmap_stats(heatmap):
    value = np.asarray(heatmap, np.float32)
    if value.shape != (133, 96, 72):
        raise RuntimeError(f"unexpected heatmap shape: {value.shape}")
    finite = np.isfinite(value).all(axis=(1, 2))
    positive = np.maximum(value, 0.0)
    mass = positive.sum(axis=(1, 2), dtype=np.float64)
    valid = finite & (mass > 1e-12)
    safe_mass = np.where(valid, mass, 1.0)
    xmass = positive.sum(axis=1, dtype=np.float64)
    ymass = positive.sum(axis=2, dtype=np.float64)
    mux = (xmass @ XX.astype(np.float64)) / safe_mass
    muy = (ymass @ YY.astype(np.float64)) / safe_mass
    ex2 = (xmass @ (XX.astype(np.float64) ** 2)) / safe_mass
    ey2 = (ymass @ (YY.astype(np.float64) ** 2)) / safe_mass
    exy = np.einsum("kyx,y,x->k", positive, YY, XX, dtype=np.float64) / safe_mass
    varx = np.maximum(ex2 - mux * mux, 0.0)
    vary = np.maximum(ey2 - muy * muy, 0.0)
    cov = exy - mux * muy
    raw_peak = value.max(axis=(1, 2))
    probability_peak = positive.max(axis=(1, 2)) / safe_mass
    positive_log = np.zeros_like(positive)
    np.log(positive, out=positive_log, where=positive > 0)
    entropy = np.log(safe_mass) - (positive * positive_log).sum(axis=(1, 2), dtype=np.float64) / safe_mass
    result = np.stack([
        mux / 72.0, muy / 96.0, varx / (72.0 ** 2), vary / (96.0 ** 2),
        cov / (72.0 * 96.0), raw_peak, probability_peak, entropy, mass, mux, muy,
    ], axis=1).astype(np.float32)
    result[~valid, :8] = np.nan
    return result


def prepare():
    configure_base()
    base.prepare()
    current = json.loads(SELECTION.read_text())
    prior = json.loads(PRIOR_SELECTION.read_text())
    if current["rows"] != prior["rows"] or current["selected_frames"] != prior["selected_frames"]:
        raise RuntimeError("positive-mass sample is not identical to the frozen T=1 pilot")
    identity_path = OUTPUT / "preregistered_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["prior_selection_sha256"] = base.sha256_file(PRIOR_SELECTION)
    identity["selection_rows_identical_to_prior"] = True
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")


def launch():
    config = json.loads(CONFIG.read_text())
    workers = config["gpu_policy"]["allowed"]
    processes = []
    logs = []
    for index, gpu in enumerate(workers):
        log_path = OUTPUT / f"worker-{index:02d}.log"
        handle = log_path.open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(index),
                   "--workers", str(len(workers)), "--gpu-uuid", gpu["uuid"]]
        processes.append((index, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env), handle))
        logs.append(str(log_path))
    failures = []
    for index, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"worker": index, "exit_code": code})
    if failures:
        raise RuntimeError({"worker_failures": failures, "logs": logs})
    configure_base()
    base.aggregate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        launch()
    elif args.worker is not None:
        configure_base()
        base.run_worker(args.worker, args.workers, args.gpu_uuid)
    else:
        parser.error("choose --prepare, --launch, or --worker")


if __name__ == "__main__":
    main()
