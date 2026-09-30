#!/usr/bin/env python3
"""Durable, resumable five-GPU full-train positive-mass heatmap extraction."""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import run_phoenix_hrnet_heatmap_uncertainty_pilot as base
import run_phoenix_hrnet_positive_mass_gpu_mixed_precision_pilot as v3


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CONFIG = ROOT / "configs/p4_hrnet_positive_mass_fulltrain_v1_preregistered.json"
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_fulltrain_v1_49faacc3"
ASSIGNMENT = OUTPUT / "assignment.json"
STATUS = OUTPUT / "job_status.json"
KEYPOINTS = REPO / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
VIDEO_ZIP = REPO / "data/phoenix_2014t/PHOENIX2014T_videos.zip"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def sample_file(worker, sample_id):
    token = hashlib.sha256(sample_id.encode()).hexdigest()[:24]
    return OUTPUT / "shards" / f"worker-{worker:02d}" / f"{token}.npz"


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(f"non-overwriting full-train identity already exists: {OUTPUT}")
    config = json.loads(CONFIG.read_text())
    unlock = (ROOT / "results/phoenix-2014t_ISLR/p4_hrnet_positive_mass_gpu_mixed_precision_pilot_v3_49faacc3"
              / "positive_mass_gpu_mixed_precision_diagnostics.json")
    if sha256_file(unlock) != config["unlock_diagnostics_sha256"]:
        raise RuntimeError("v3 unlock identity mismatch")
    if not json.loads(unlock.read_text())["all_gates_passed"]:
        raise RuntimeError("v3 did not unlock full-train extraction")
    with KEYPOINTS.open("rb") as handle:
        keypoints = pickle.load(handle)
    rows = [{"sample_id": str(name), "frames": int(len(value))} for name, value in keypoints.items()]
    if len(rows) != config["scope"]["videos_expected"] or sum(x["frames"] for x in rows) != config["scope"]["frames_expected"]:
        raise RuntimeError("full-train scope identity failure")
    workers = int(config["gpu_policy"]["workers"])
    bins = [[] for _ in range(workers)]
    loads = [0] * workers
    for row in sorted(rows, key=lambda x: (-x["frames"], x["sample_id"])):
        worker = min(range(workers), key=lambda index: (loads[index], index))
        bins[worker].append(row)
        loads[worker] += row["frames"]
    OUTPUT.mkdir(parents=True)
    for worker in range(workers):
        (OUTPUT / "shards" / f"worker-{worker:02d}").mkdir(parents=True)
    assignment = {
        "created_utc": utcnow(), "algorithm": config["scope"]["assignment"],
        "videos": len(rows), "frames": sum(x["frames"] for x in rows),
        "worker_frame_loads": loads, "workers": bins,
    }
    ASSIGNMENT.write_text(json.dumps(assignment, indent=2) + "\n")
    identity = {
        "created_before_fulltrain_outputs": True,
        "config_sha256": sha256_file(CONFIG),
        "assignment_sha256": sha256_file(ASSIGNMENT),
        "unlock_diagnostics_sha256": sha256_file(unlock),
        "video_zip_sha256": config["inputs"]["video_zip_sha256"],
        "keypoints_sha256": config["inputs"]["frozen_train_keypoints_sha256"],
    }
    (OUTPUT / "preregistered_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    write_json_atomic(STATUS, {"status": "prepared", "created_utc": utcnow(),
                               "videos": len(rows), "frames": assignment["frames"],
                               "worker_frame_loads": loads})
    print(json.dumps({"output": str(OUTPUT), "videos": len(rows), "frames": assignment["frames"],
                      "worker_frame_loads": loads}, indent=2))


def validate_saved(path, sample_id, frames):
    try:
        with np.load(path, allow_pickle=False) as packed:
            return (str(packed["sample_name"].item()) == sample_id
                    and int(packed["frame_count"].item()) == frames
                    and packed["statistics"].shape == (frames, 133, 11)
                    and packed["bboxes"].shape == (frames, 5))
    except Exception:
        return False


def run_worker(index, gpu_uuid):
    config = json.loads(CONFIG.read_text())
    allowed = {row["uuid"]: row["pci"] for row in config["gpu_policy"]["allowed"]}
    if gpu_uuid not in allowed or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("GPU UUID visibility policy failure")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker requires exactly one visible CUDA device outside sandbox")
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, init_pose_model

    rows = json.loads(ASSIGNMENT.read_text())["workers"][index]
    detector = init_detector(str(base.DET_CONFIG), str(base.DET_CKPT), "cuda:0")
    pose = init_pose_model(str(base.POSE_CONFIG), str(base.POSE_CKPT), "cuda:0")
    archive = zipfile.ZipFile(VIDEO_ZIP)
    started = time.perf_counter()
    completed_videos = 0
    completed_frames = 0
    skipped_videos = 0
    entries = []
    for row_number, row in enumerate(rows):
        name, frames = row["sample_id"], int(row["frames"])
        destination = sample_file(index, name)
        if destination.exists() and validate_saved(destination, name, frames):
            skipped_videos += 1
        else:
            statistics = np.full((frames, 133, 11), np.nan, np.float32)
            bboxes = np.full((frames, 5), np.nan, np.float32)
            for frame_index in range(frames):
                frame = base.decode(archive, name, frame_index)
                detection = inference_detector(detector, frame)[0]
                detection = detection[detection[:, 4] >= float(config["inference"]["detector_score_threshold"])]
                if len(detection):
                    areas = (detection[:, 2] - detection[:, 0]) * (detection[:, 3] - detection[:, 1])
                    detection = detection[areas >= float(config["inference"]["detector_area_threshold"])]
                if not len(detection):
                    continue
                persons = [{"bbox": box} for box in detection]
                poses, outputs = inference_top_down_pose_model(
                    pose, frame, persons, format="xyxy", dataset="TopDownCocoWholeBodyDataset",
                    return_heatmap=True)
                chosen = int(np.argmax([float(item["bbox"][-1]) for item in poses]))
                statistics[frame_index] = v3.heatmap_stats(np.asarray(outputs[0]["heatmap"][chosen], np.float32))
                bboxes[frame_index] = np.asarray(poses[chosen]["bbox"], np.float32)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, sample_name=np.asarray(name), frame_count=np.asarray(frames, np.int32),
                                    statistics=statistics, bboxes=bboxes)
            os.replace(temporary, destination)
        with np.load(destination, allow_pickle=False) as packed:
            stats = np.asarray(packed["statistics"])
            successful = np.isfinite(stats[..., :8]).all(axis=2).all(axis=1)
            valid_frames = int(successful.sum())
        completed_videos += 1
        completed_frames += frames
        entries.append({"sample_id": name, "frames": frames, "valid_frames": valid_frames,
                        "path": str(destination.relative_to(OUTPUT)), "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination)})
        progress = {
            "status": "running", "updated_utc": utcnow(), "worker": index, "gpu_uuid": gpu_uuid,
            "pci": allowed[gpu_uuid], "videos_complete": completed_videos, "videos_total": len(rows),
            "frames_complete": completed_frames, "frames_total": sum(int(x["frames"]) for x in rows),
            "resumed_videos": skipped_videos, "elapsed_seconds": time.perf_counter() - started,
            "current_sample": name, "current_row": row_number + 1,
        }
        write_json_atomic(OUTPUT / f"worker-{index:02d}-progress.json", progress)
    archive.close()
    manifest = {
        "status": "complete", "created_utc": utcnow(), "worker": index, "gpu_uuid": gpu_uuid,
        "pci": allowed[gpu_uuid], "gpu_name": torch.cuda.get_device_name(0),
        "videos": completed_videos, "frames": completed_frames, "resumed_videos": skipped_videos,
        "valid_frames": sum(x["valid_frames"] for x in entries),
        "elapsed_seconds": time.perf_counter() - started, "entries": entries,
    }
    path = OUTPUT / f"worker-{index:02d}-manifest.json"
    write_json_atomic(path, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))


def aggregate():
    config = json.loads(CONFIG.read_text())
    manifests = [json.loads((OUTPUT / f"worker-{i:02d}-manifest.json").read_text())
                 for i in range(int(config["gpu_policy"]["workers"]))]
    videos = sum(x["videos"] for x in manifests)
    frames = sum(x["frames"] for x in manifests)
    valid = sum(x["valid_frames"] for x in manifests)
    gates = config["completion_gates"]
    checks = {
        "videos_complete": videos == int(gates["videos_complete"]),
        "frames_accounted": frames == int(gates["frames_accounted"]),
        "successful_frame_fraction": valid / frames >= float(gates["successful_frame_fraction_at_least"]),
    }
    metrics = {
        "status": "complete" if all(checks.values()) else "failed_gates", "created_utc": utcnow(),
        "train_only": True, "outcomes_read": False, "videos": videos, "frames": frames,
        "valid_frames": valid, "successful_frame_fraction": valid / frames,
        "total_output_bytes": sum(e["bytes"] for x in manifests for e in x["entries"]),
        "five_gpu_wall_seconds": max(x["elapsed_seconds"] for x in manifests),
        "checks": checks, "gate_passed": bool(all(checks.values())),
    }
    metrics_path = OUTPUT / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "status": metrics["status"], "created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
        "assignment_sha256": sha256_file(ASSIGNMENT), "metrics_sha256": sha256_file(metrics_path),
        "worker_manifest_sha256": {str(i): sha256_file(OUTPUT / f"worker-{i:02d}-manifest.json")
                                   for i in range(len(manifests))},
        "gate_passed": metrics["gate_passed"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


def supervise():
    config = json.loads(CONFIG.read_text())
    workers = config["gpu_policy"]["allowed"]
    write_json_atomic(STATUS, {"status": "running", "started_utc": utcnow(),
                               "supervisor_pid": os.getpid(), "workers": len(workers)})
    processes = []
    try:
        for index, gpu in enumerate(workers):
            log_path = OUTPUT / f"worker-{index:02d}.log"
            handle = log_path.open("a")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(index),
                       "--gpu-uuid", gpu["uuid"]]
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            processes.append((index, process, handle))
        failures = []
        for index, process, handle in processes:
            code = process.wait()
            handle.close()
            if code != 0:
                failures.append({"worker": index, "exit_code": code})
        if failures:
            raise RuntimeError({"worker_failures": failures})
        metrics = aggregate()
        write_json_atomic(STATUS, {"status": metrics["status"], "finished_utc": utcnow(),
                                   "supervisor_pid": os.getpid(), "metrics": metrics})
    except Exception as error:
        for _, process, handle in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
        write_json_atomic(STATUS, {"status": "failed", "failed_utc": utcnow(),
                                   "supervisor_pid": os.getpid(), "error": repr(error),
                                   "traceback": traceback.format_exc()})
        raise


def start_detached():
    if not ASSIGNMENT.exists():
        raise RuntimeError("run --prepare before --start-detached")
    if STATUS.exists():
        status = json.loads(STATUS.read_text())
        pid = int(status.get("supervisor_pid", -1))
        if status.get("status") == "running" and pid > 0:
            try:
                os.kill(pid, 0)
                raise RuntimeError(f"supervisor already running with pid {pid}")
            except ProcessLookupError:
                pass
    log = (OUTPUT / "supervisor.log").open("a")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--supervise"],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True)
    write_json_atomic(STATUS, {"status": "starting", "started_utc": utcnow(),
                               "supervisor_pid": process.pid, "detached": True})
    print(json.dumps({"status": "starting", "supervisor_pid": process.pid,
                      "status_file": str(STATUS), "log": str(OUTPUT / "supervisor.log")}, indent=2))


def show_status():
    value = json.loads(STATUS.read_text()) if STATUS.exists() else {"status": "not_prepared"}
    progress = []
    for path in sorted(OUTPUT.glob("worker-??-progress.json")) if OUTPUT.exists() else []:
        progress.append(json.loads(path.read_text()))
    value["worker_progress"] = progress
    print(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--start-detached", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.start_detached:
        start_detached()
    elif args.supervise:
        supervise()
    elif args.worker is not None:
        run_worker(args.worker, args.gpu_uuid)
    elif args.status:
        show_status()
    else:
        parser.error("choose one operation")


if __name__ == "__main__":
    main()
