#!/usr/bin/env python3
"""Export frozen P6 fold models with Python stdlib and a local C++ compiler.

Uses only train-fit features/targets and already-frozen train-fit OOF predictions.
The latter are a numerical parity check, never a fitting target.
"""
import argparse
import ast
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
FIT = BASE / "p6_local_js_predictor_oof_v1_49faacc3"
QUANTILE = BASE / "p6_k_history_online_quantile_oof_v1_49faacc3"
CONFIG = ROOT / "configs/p6_k_history_online_quantile_oof_v1_preregistered.json"
FROZEN = ROOT / "tools/analyze_phoenix_p6_k_online_quantile_oof.py"
CPP = ROOT / "tools/p6_ridge_fit_portable.cpp"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_array(archive, name, destination, dtype, shape):
    with zipfile.ZipFile(archive) as packed, packed.open(name + ".npy") as source:
        magic = source.read(8)
        if magic[:6] != b"\x93NUMPY":
            raise ValueError("invalid NPY magic")
        length = struct.unpack("<H", source.read(2))[0] if magic[6] == 1 else struct.unpack("<I", source.read(4))[0]
        header = ast.literal_eval(source.read(length).decode("ascii").strip())
        if header != {"descr": dtype, "fortran_order": False, "shape": shape}:
            raise ValueError(f"unexpected {name} array: {header}")
        with Path(destination).open("wb") as target:
            shutil.copyfileobj(source, target)


def main(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    config = json.loads(CONFIG.read_text())
    identity = json.loads((QUANTILE / "preregistered_identity.json").read_text())
    fit_manifest = json.loads((FIT / "manifest.json").read_text())
    data = FIT / "dataset.npz"
    predictions = FIT / "oof_predictions.npz"
    if sha(CONFIG) != identity["config_sha256"] or sha(FROZEN) != identity["script_sha256_at_prepare"]:
        raise RuntimeError("frozen quantile identity changed")
    if sha(data) != config["fixed_assets"]["fit_p6_dataset_sha256"] or sha(predictions) != fit_manifest["oof_sha256"]:
        raise RuntimeError("frozen fit data or OOF predictions changed")
    rows = 183401
    with tempfile.TemporaryDirectory(prefix="p6-quantile-model-") as temporary:
        temp = Path(temporary)
        specs = [
            (data, "K", "<f4", (rows, 153)),
            (data, "target", "<f4", (rows,)),
            (data, "folds", "|i1", (rows,)),
            (predictions, "K", "<f4", (rows,)),
        ]
        raw = []
        for index, (archive, name, dtype, shape) in enumerate(specs):
            path = temp / f"{index}.raw"
            extract_array(archive, name, path, dtype, shape)
            raw.append(path)
        executable = temp / "fit"
        subprocess.run(["g++", "-O3", "-std=c++17", str(CPP), "-o", str(executable)], check=True)
        model_data = temp / "models.bin"
        result = subprocess.run([str(executable), *(str(path) for path in raw), str(model_data), str(rows)],
                                check=True, capture_output=True, text=True)
        parity = json.loads(result.stdout)
        if parity["max_abs_prediction_error"] > 1e-4:
            raise RuntimeError(f"reconstructed model differs from frozen K OOF predictions: {parity}")
        floats = struct.unpack("<" + "d" * (5 * (3 * 153 + 1)), model_data.read_bytes())
    folds = []
    for fold in range(5):
        offset = fold * (3 * 153 + 1)
        folds.append({"mean": list(floats[offset:offset + 153]),
                      "scale": list(floats[offset + 153:offset + 306]),
                      "weights": list(floats[offset + 306:offset + 459]),
                      "intercept": floats[offset + 459]})
    payload = {"identity": config["experiment"], "config_sha256": sha(CONFIG),
               "frozen_script_sha256": sha(FROZEN), "fit_dataset_sha256": sha(data),
               "fit_oof_predictions_sha256": sha(predictions),
               "portable_fitter_sha256": sha(CPP), "ridge_alpha": 100.0,
               "prediction_parity": parity, "folds": folds}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({"model_bundle": str(destination), "sha256": sha(destination),
                      "prediction_parity": parity}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.output)
