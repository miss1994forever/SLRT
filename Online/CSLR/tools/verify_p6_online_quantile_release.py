#!/usr/bin/env python3
"""One-command CPU acceptance gate for the frozen P6 online-quantile release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit_p6_online_quantile_release import audit as audit_frozen
from audit_p6_online_quantile_action_parity import audit as audit_actions

DEFAULT_MODEL = ROOT / "configs/frozen_models/p6_online_quantile_fold_models_v1.json"
CONFIG = ROOT / "configs/p6_k_history_online_quantile_oof_v1_preregistered.json"
SCRIPT = ROOT / "tools/analyze_phoenix_p6_k_online_quantile_oof.py"
FILES = [
    ROOT / "utils/online_quantile_controller.py",
    ROOT / "tools/export_p6_online_quantile_models_portable.py",
    ROOT / "tools/p6_ridge_fit_portable.cpp",
    ROOT / "tools/audit_p6_online_quantile_release.py",
    ROOT / "tools/audit_p6_online_quantile_action_parity.py",
    ROOT / "tests/test_p6_online_quantile_controller.py",
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(model):
    model = Path(model)
    payload = json.loads(model.read_text())
    identity = json.loads((ROOT / "results/phoenix-2014t_ISLR/p6_k_history_online_quantile_oof_v1_49faacc3/preregistered_identity.json").read_text())
    fit_root = ROOT / "results/phoenix-2014t_ISLR/p6_local_js_predictor_oof_v1_49faacc3"
    fit_manifest = json.loads((fit_root / "manifest.json").read_text())
    model_checks = {
        "config": payload["config_sha256"] == sha(CONFIG) == identity["config_sha256"],
        "frozen_script": payload["frozen_script_sha256"] == sha(SCRIPT) == identity["script_sha256_at_prepare"],
        "portable_fitter": payload["portable_fitter_sha256"] == sha(ROOT / "tools/p6_ridge_fit_portable.cpp"),
        "fit_inputs": (payload["fit_dataset_sha256"] == sha(fit_root / "dataset.npz") and
                       payload["fit_oof_predictions_sha256"] == fit_manifest["oof_sha256"] ==
                       sha(fit_root / "oof_predictions.npz") and payload["ridge_alpha"] == 100.0),
        "prediction_parity": (payload["prediction_parity"]["rows"] == 183401 and
                              payload["prediction_parity"]["max_abs_prediction_error"] < 1e-4),
    }
    test = subprocess.run([sys.executable, "-m", "unittest", "tests.test_p6_online_quantile_controller", "-v"],
                          cwd=ROOT, capture_output=True, text=True)
    frozen = audit_frozen()
    actions = audit_actions(model)
    result = {
        "passed": (all(model_checks.values()) and test.returncode == 0 and
                   frozen["passed"] and actions["passed"] and
                   actions["matched_block_actions"] == actions["checked_blocks"]),
        "model_checks": model_checks,
        "unit_tests_passed": test.returncode == 0,
        "unit_test_output": test.stdout + test.stderr,
        "frozen_artifact_audit": frozen,
        "action_parity": actions,
        "model_sha256": sha(model),
        "implementation_sha256": {str(path.relative_to(ROOT)): sha(path) for path in FILES},
        "scope": "train fit only; no GPU, calibration, dev, test, or held-out read",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = main(args.model)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "model_sha256": result["model_sha256"],
                      "model_checks": result["model_checks"],
                      "unit_tests_passed": result["unit_tests_passed"],
                      "frozen_artifact_audit_passed": result["frozen_artifact_audit"]["passed"],
                      "action_parity_passed": result["action_parity"]["passed"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
