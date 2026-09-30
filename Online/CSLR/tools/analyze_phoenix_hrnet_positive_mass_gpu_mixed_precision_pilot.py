#!/usr/bin/env python3
"""Apply the frozen v2 numeric/parity audit to mixed-precision v3."""
from pathlib import Path

import analyze_phoenix_hrnet_positive_mass_gpu_stats_pilot as audit


ROOT = Path(__file__).resolve().parents[1]
audit.OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_gpu_mixed_precision_pilot_v3_49faacc3"
audit.CONFIG = ROOT / "configs/p4_hrnet_positive_mass_gpu_mixed_precision_pilot_v3_preregistered.json"
audit.DESTINATION = audit.OUTPUT / "positive_mass_gpu_mixed_precision_diagnostics.json"


if __name__ == "__main__":
    audit.main()
