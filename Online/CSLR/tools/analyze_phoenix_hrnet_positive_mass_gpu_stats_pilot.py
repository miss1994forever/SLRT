#!/usr/bin/env python3
"""Run v1 numeric gates and frozen v1-v2 parity gates for GPU statistics."""
import json
from pathlib import Path

import numpy as np

import analyze_phoenix_hrnet_positive_mass_uncertainty_pilot as audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_gpu_stats_pilot_v2_49faacc3"
V1 = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_uncertainty_pilot_v1_49faacc3"
CONFIG = ROOT / "configs/p4_hrnet_positive_mass_gpu_stats_pilot_v2_preregistered.json"
DESTINATION = OUTPUT / "positive_mass_gpu_diagnostics.json"


def load_stats(directory):
    parts = [np.load(path) for path in sorted(directory.glob("worker-??-of-05.npz"))]
    value = np.concatenate([part["statistics"] for part in parts])
    identity = [(str(name), int(frame)) for part in parts
                for name, frame in zip(part["sample_names"], part["frame_indices"])]
    for part in parts:
        part.close()
    return value, identity


def main():
    audit.OUTPUT = OUTPUT
    audit.CONFIG = CONFIG
    audit.DESTINATION = DESTINATION
    audit.main()
    result = json.loads(DESTINATION.read_text())
    current, current_identity = load_stats(OUTPUT)
    prior, prior_identity = load_stats(V1)
    if current_identity != prior_identity or current.shape != prior.shape:
        raise RuntimeError("v1-v2 row identity mismatch")
    moments = [0, 1, 2, 3, 4, 6, 9, 10]
    moment_difference = float(np.nanmax(np.abs(current[..., moments] - prior[..., moments])))
    entropy_difference = float(np.nanmax(np.abs(current[..., 7] - prior[..., 7])))
    mass_relative = np.abs(current[..., 8] - prior[..., 8]) / np.maximum(np.abs(prior[..., 8]), 1e-12)
    mass_relative_difference = float(np.nanmax(mass_relative))
    gates = json.loads(CONFIG.read_text())["outcome_blind_gates"]
    parity = {
        "v1_max_absolute_difference_moments": moment_difference,
        "v1_max_absolute_difference_entropy": entropy_difference,
        "v1_max_relative_difference_positive_mass": mass_relative_difference,
    }
    result["values"].update(parity)
    result["gates_passed"].update({
        "v1_moment_parity": moment_difference <= gates["v1_max_absolute_difference_moments_at_most"],
        "v1_entropy_parity": entropy_difference <= gates["v1_max_absolute_difference_entropy_at_most"],
        "v1_positive_mass_parity": mass_relative_difference <= gates["v1_max_relative_difference_positive_mass_at_most"],
    })
    result["all_gates_passed"] = bool(all(result["gates_passed"].values()))
    result["decision"] = ("unlock full-train extraction" if result["all_gates_passed"]
                          else "stop before full-train extraction")
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
