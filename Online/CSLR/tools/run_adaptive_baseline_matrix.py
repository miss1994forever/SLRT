#!/usr/bin/env python3
"""Run the pre-registered Phoenix adaptive-stride baseline matrix."""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml


CSLR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = CSLR_ROOT / "configs" / "experiments" / "phoenix_adaptive_baselines_v1.yaml"
FAULTY_GPU_UUIDS = {
    "GPU-dbd35875-dfa5-43f1-0cf0-f88ccb529c8a",
    "GPU-06afe121-c4ce-b981-bb86-399e4a85ae83",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else (CSLR_ROOT / path).resolve()


def load_protocol(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    required = {"protocol_version", "experiment_id", "base_config", "checkpoint", "variants"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol is missing fields: {sorted(missing)}")
    return path, protocol


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(*args):
    return subprocess.check_output(["git", *args], cwd=CSLR_ROOT, text=True).strip()


def build_manifest(protocol_path, protocol, hash_assets=True, include_large_hashes=True):
    files = {}
    for value in protocol.get("provenance_files", []):
        path = resolve_path(value)
        record = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            record["bytes"] = path.stat().st_size
            if not hash_assets:
                record["sha256"] = None
                record["hash_status"] = "skipped_diagnostic_run"
            elif include_large_hashes or path.stat().st_size < 1024 * 1024 * 1024:
                record["sha256"] = sha256_file(path)
            else:
                record["sha256"] = None
                record["hash_status"] = "skipped_large_file"
        files[value] = record
    return {
        "manifest_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "experiment_id": protocol["experiment_id"],
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "status_porcelain": git_value("status", "--porcelain"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch_package": package_version("torch"),
            "numpy_package": package_version("numpy"),
            "pyyaml_package": package_version("PyYAML"),
        },
        "files": files,
    }


def validate_gpu_uuid(value, required=True):
    if not value and not required:
        return None
    if not value or "," in value or not value.startswith("GPU-"):
        raise ValueError("provide exactly one healthy GPU UUID via --gpu-uuid or CUDA_VISIBLE_DEVICES")
    if value in FAULTY_GPU_UUIDS:
        raise ValueError(f"refusing known faulty GPU UUID: {value}")
    return value


def validate_manifest_state(manifest, protocol_path, allow_dirty=False):
    if manifest["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("protocol changed after manifest creation; use a new experiment ID/output root")
    current_commit = git_value("rev-parse", "HEAD")
    if manifest["git"]["commit"] != current_commit:
        raise ValueError(
            f"manifest commit {manifest['git']['commit']} differs from current commit {current_commit}"
        )
    if manifest["git"]["status_porcelain"] and not allow_dirty:
        raise ValueError("manifest was created from a dirty worktree")


def resolved_model_config(protocol, variant):
    base_path = resolve_path(protocol["base_config"])
    with base_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    common = protocol["common_inference"]
    data = config.setdefault("data", {})
    data["win_size"] = int(common["window_frames"])
    data["split_size"] = int(common["split_size"])
    data["stride"] = int(variant.get("sampling", {}).get("fixed_stride", data.get("stride", 1)))
    data["prob_thr"] = [common["probability_threshold"]]
    data["sampling"] = dict(variant["sampling"])
    if data["sampling"]["mode"] == "adaptive_motion":
        adaptive = dict(data.get("adaptive_stride", {}))
        adaptive.update({key: value for key, value in data["sampling"].items() if key != "mode"})
        adaptive["enabled"] = True
        data["adaptive_stride"] = adaptive
    else:
        data.setdefault("adaptive_stride", {})["enabled"] = False
    span = config.setdefault("postprocess", {}).setdefault("span_weighted_voting", {})
    span.update(
        {
            "enabled": bool(variant.get("span_weighted_voting", False)),
            "vote_span_frames": int(common["span_frames"]),
            "kernel": common["span_kernel"],
            "min_weight": float(common["span_min_weight"]),
        }
    )
    training = config.setdefault("training", {})
    training["random_seed"] = int(common["random_seed"])
    training["model_dir"] = str(resolve_path(protocol["checkpoint"]).parent.parent)
    config["runtime_profile"] = {"enabled": bool(common.get("profile_model_forward", False))}
    return config


def runtime_summary(records):
    values = [item["wall_time_seconds"] for item in records if item.get("returncode") == 0]
    if not values:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "successful_repetitions": len(values),
        "mean_seconds": mean,
        "median_seconds": statistics.median(values),
        "std_seconds": std,
        "cv_percent": 100.0 * std / mean if mean else None,
        "values_seconds": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--gpu-uuid", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-large-hashes", action="store_true")
    parser.add_argument("--skip-asset-hashes", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--frozen-manifest-sha256")
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    protocol_path, protocol = load_protocol(args.protocol)
    if not args.dry_run and not args.allow_dirty and git_value("status", "--porcelain"):
        raise ValueError("refusing a dirty worktree; commit changes or pass --allow-dirty for diagnostics")
    output_root = (args.output_root or resolve_path(protocol["output_root"])).resolve()
    manifest_path = output_root / "protocol_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.split == "test":
        if not args.allow_test:
            raise ValueError("test requires --allow-test")
        if not manifest_path.is_file() or not args.frozen_manifest_sha256:
            raise ValueError("test requires an existing manifest and --frozen-manifest-sha256")
        actual = sha256_file(manifest_path)
        if actual != args.frozen_manifest_sha256:
            raise ValueError(f"frozen manifest mismatch: expected {args.frozen_manifest_sha256}, got {actual}")

    if not manifest_path.exists():
        manifest = build_manifest(
            protocol_path,
            protocol,
            hash_assets=not args.skip_asset_hashes,
            include_large_hashes=not args.skip_large_hashes,
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_state(manifest, protocol_path, allow_dirty=args.allow_dirty or args.dry_run)

    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")
    dirty = bool(manifest["git"]["status_porcelain"])
    missing_files = [name for name, record in manifest["files"].items() if not record["exists"]]
    incomplete_hashes = any(record.get("sha256") is None for record in manifest["files"].values())
    if not args.dry_run and missing_files:
        raise FileNotFoundError(f"registered provenance files are missing: {missing_files}")
    if not args.dry_run and not args.allow_dirty and dirty:
        raise ValueError("refusing a dirty worktree; commit changes or pass --allow-dirty for diagnostics")
    if args.split == "test" and incomplete_hashes:
        raise ValueError("test requires a manifest with hashes for every registered provenance file")
    if args.manifest_only:
        return

    gpu_uuid = validate_gpu_uuid(args.gpu_uuid, required=not args.dry_run)
    selected = args.variants or list(protocol["variants"])
    unknown = set(selected) - set(protocol["variants"])
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    runnable = []
    for variant_id in selected:
        variant = protocol["variants"][variant_id]
        if "reuse_forward_from" not in variant:
            runnable.append((variant_id, variant))
        else:
            print(f"reuse {variant_id} <- {variant['reuse_forward_from']}")

    env = os.environ.copy()
    if gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    all_runtime = {}
    for repetition in range(1, args.repetitions + 1):
        for variant_id, variant in runnable:
            run_dir = output_root / args.split / variant_id / f"run_{repetition:02d}"
            complete_path = run_dir / "complete.json"
            if complete_path.exists() and args.resume:
                print(f"skip complete {run_dir}")
                continue
            if run_dir.exists() and any(run_dir.iterdir()) and not args.dry_run:
                raise FileExistsError(f"refusing non-empty run directory: {run_dir}")
            config = resolved_model_config(protocol, variant)
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "resolved_config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [
                sys.executable,
                "prediction_slide.py",
                "--config",
                str(config_path),
                "--output_dir",
                str(run_dir),
                "--ckpt_name",
                resolve_path(protocol["checkpoint"]).name,
                "--split",
                args.split,
                "--pred_src",
                protocol["common_inference"]["prediction_source"],
                "--blank_thr",
                str(protocol["common_inference"]["blank_threshold"]),
                "--save_fea",
                "0",
                "--eval_setting",
                f"matrix_{variant_id}_r{repetition:02d}",
            ]
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            (run_dir / "command.json").write_text(
                json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print("RUN", " ".join(command))
            if args.dry_run:
                continue

            started = dt.datetime.now(dt.timezone.utc)
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.perf_counter()
            with (run_dir / "stdout.log").open("w", encoding="utf-8") as output:
                process = subprocess.run(command, cwd=CSLR_ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - start
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            runtime = {
                "variant": variant_id,
                "repetition": repetition,
                "split": args.split,
                "gpu_uuid": gpu_uuid,
                "started_utc": started.isoformat(),
                "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "wall_time_seconds": elapsed,
                "child_user_seconds": after.ru_utime - before.ru_utime,
                "child_system_seconds": after.ru_stime - before.ru_stime,
                "returncode": process.returncode,
            }
            (run_dir / "runtime.json").write_text(
                json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            all_runtime.setdefault(variant_id, []).append(runtime)
            if process.returncode != 0:
                raise RuntimeError(f"{variant_id} failed; see {run_dir / 'stdout.log'}")
            expected = [run_dir / f"{args.split}_results.pkl", run_dir / f"{args.split}_evaluation_results.pkl"]
            if protocol["common_inference"].get("profile_model_forward", False):
                expected.append(run_dir / f"{args.split}_runtime_profile.json")
            if not all(path.is_file() for path in expected):
                raise RuntimeError(f"{variant_id} completed without expected result files")
            complete_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")

    if all_runtime:
        rendered = {key: runtime_summary(value) for key, value in all_runtime.items()}
        summary_path = output_root / args.split / "runtime_summary.json"
        summary_path.write_text(json.dumps(rendered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"runtime_summary={summary_path}")


if __name__ == "__main__":
    main()
