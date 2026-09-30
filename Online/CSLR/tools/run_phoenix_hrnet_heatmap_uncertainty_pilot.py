#!/usr/bin/env python3
"""Train-only multi-GPU pilot for raw HRNet heatmap uncertainty statistics."""
import argparse
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
CONFIG = ROOT / "configs/p4_hrnet_heatmap_uncertainty_pilot_v1_preregistered.json"
OUTPUT = BASE / "p4_hrnet_heatmap_uncertainty_pilot_v1_49faacc3"
SELECTION = OUTPUT / "selection.json"
VIDEO_ZIP = REPO / "data/phoenix_2014t/PHOENIX2014T_videos.zip"
KEYPOINTS = REPO / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
ENV = Path("/mnt/workspace/conda_envs/haojun/envs/slrt_legacy")
DET_CONFIG = ENV / "lib/python3.9/site-packages/mmdet/.mim/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
POSE_CONFIG = ENV / "lib/python3.9/site-packages/mmpose/.mim/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
CACHE = Path("/home/haojun/.cache/torch/hub/checkpoints")
DET_CKPT = CACHE / "faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
POSE_CKPT = CACHE / "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(f"non-overwriting pilot output exists: {OUTPUT}")
    config = json.loads(CONFIG.read_text())
    with KEYPOINTS.open("rb") as handle:
        keypoints = pickle.load(handle)
    if len(keypoints) != 7096 or any(not str(name).startswith("train/") for name in keypoints):
        raise RuntimeError("train-only keypoint identity failure")
    seed = config["scope"]["selection_seed"]
    names = sorted(keypoints, key=lambda name: (hashlib.sha256(f"{seed}\0{name}".encode()).hexdigest(), name))
    names = names[:int(config["scope"]["videos"])]
    rows = []
    for name in names:
        length = len(keypoints[name])
        count = min(length, int(config["scope"]["frames_per_video_max"]))
        indices = np.unique(np.linspace(0, length - 1, count, dtype=np.int64)).tolist()
        rows.append({"sample_id": name, "frames": length, "selected_frame_indices": indices})
    value = {
        "created_utc": utcnow(),
        "rule": config["scope"],
        "videos": len(rows),
        "selected_frames": sum(len(row["selected_frame_indices"]) for row in rows),
        "rows": rows,
        "outcome_fields_read": False,
    }
    OUTPUT.mkdir(parents=True)
    SELECTION.write_text(json.dumps(value, indent=2) + "\n")
    (OUTPUT / "preregistered_identity.json").write_text(json.dumps({
        "config_sha256": sha256_file(CONFIG),
        "selection_sha256": sha256_file(SELECTION),
        "created_before_gpu_outputs": True,
    }, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "videos": len(rows),
                      "frames": value["selected_frames"]}, indent=2))


def decode(zf, sample_id, frame_index):
    stem = sample_id.split("/", 1)[1]
    member = f"images/{stem}/images{frame_index + 1:04d}.png"
    with zf.open(member) as handle:
        encoded = np.frombuffer(handle.read(), np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"decode failure: {member}")
    return frame


def heatmap_stats(heatmap):
    value = np.asarray(heatmap, np.float32)
    if value.shape != (133, 96, 72):
        raise RuntimeError(f"unexpected heatmap shape: {value.shape}")
    shifted = value - value.max(axis=(1, 2), keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=(1, 2), keepdims=True)
    yy = np.arange(96, dtype=np.float32)[None, :, None]
    xx = np.arange(72, dtype=np.float32)[None, None, :]
    mux = (probability * xx).sum(axis=(1, 2))
    muy = (probability * yy).sum(axis=(1, 2))
    dx, dy = xx - mux[:, None, None], yy - muy[:, None, None]
    varx = (probability * dx * dx).sum(axis=(1, 2))
    vary = (probability * dy * dy).sum(axis=(1, 2))
    cov = (probability * dx * dy).sum(axis=(1, 2))
    raw_peak = value.max(axis=(1, 2))
    probability_peak = probability.max(axis=(1, 2))
    entropy = -(probability * np.log(np.maximum(probability, 1e-30))).sum(axis=(1, 2))
    return np.stack([mux, muy, varx, vary, cov, raw_peak, probability_peak, entropy], axis=1)


def run_worker(index, workers, gpu_uuid):
    output_path = OUTPUT / f"worker-{index:02d}-of-{workers:02d}.npz"
    manifest_path = OUTPUT / f"worker-{index:02d}-manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"non-overwriting worker output exists: {output_path}")
    config = json.loads(CONFIG.read_text())
    allowed = {row["uuid"]: row["pci"] for row in config["gpu_policy"]["allowed"]}
    if gpu_uuid not in allowed or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("GPU UUID visibility policy failure")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker requires exactly one visible CUDA device outside sandbox")
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, init_pose_model

    selection = json.loads(SELECTION.read_text())
    assigned = selection["rows"][index::workers]
    detector = init_detector(str(DET_CONFIG), str(DET_CKPT), "cuda:0")
    pose = init_pose_model(str(POSE_CONFIG), str(POSE_CKPT), "cuda:0")
    with KEYPOINTS.open("rb") as handle:
        frozen = pickle.load(handle)
    zf = zipfile.ZipFile(VIDEO_ZIP)
    sample_names, frame_indices, bboxes = [], [], []
    statistics, predicted, frozen_rows, timings = [], [], [], []
    raw_hands, raw_hand_video, raw_hand_frame = [], [], []
    failures = 0
    started = time.perf_counter()
    for row in assigned:
        name = row["sample_id"]
        for local_number, frame_index in enumerate(row["selected_frame_indices"]):
            total_tick = time.perf_counter()
            tick = time.perf_counter(); frame = decode(zf, name, frame_index); decode_s = time.perf_counter() - tick
            tick = time.perf_counter(); detection = inference_detector(detector, frame)[0]; detector_s = time.perf_counter() - tick
            detection = detection[detection[:, 4] >= float(config["inference"]["detector_score_threshold"])]
            if len(detection):
                areas = (detection[:, 2] - detection[:, 0]) * (detection[:, 3] - detection[:, 1])
                detection = detection[areas >= float(config["inference"]["detector_area_threshold"])]
            if not len(detection):
                failures += 1
                continue
            persons = [{"bbox": box} for box in detection]
            tick = time.perf_counter()
            poses, outputs = inference_top_down_pose_model(
                pose, frame, persons, format="xyxy", dataset="TopDownCocoWholeBodyDataset",
                return_heatmap=True)
            pose_s = time.perf_counter() - tick
            chosen = int(np.argmax([float(item["bbox"][-1]) for item in poses]))
            heatmap = np.asarray(outputs[0]["heatmap"][chosen], np.float32)
            tick = time.perf_counter(); stats = heatmap_stats(heatmap); stat_s = time.perf_counter() - tick
            sample_names.append(name); frame_indices.append(frame_index)
            bboxes.append(np.asarray(poses[chosen]["bbox"], np.float32))
            statistics.append(stats); predicted.append(np.asarray(poses[chosen]["keypoints"], np.float32))
            frozen_rows.append(np.asarray(frozen[name][frame_index], np.float32))
            timings.append([decode_s, detector_s, pose_s, stat_s, time.perf_counter() - total_tick])
            if local_number < 2:
                raw_hands.append(heatmap[91:133].astype(np.float16))
                raw_hand_video.append(name); raw_hand_frame.append(frame_index)
    zf.close()
    path = output_path
    np.savez_compressed(
        path, sample_names=np.asarray(sample_names), frame_indices=np.asarray(frame_indices, np.int32),
        bboxes=np.asarray(bboxes, np.float32), statistics=np.asarray(statistics, np.float32),
        predicted_keypoints=np.asarray(predicted, np.float32), frozen_keypoints=np.asarray(frozen_rows, np.float32),
        timings_seconds=np.asarray(timings, np.float32), raw_hand_heatmaps=np.asarray(raw_hands, np.float16),
        raw_hand_sample_names=np.asarray(raw_hand_video), raw_hand_frame_indices=np.asarray(raw_hand_frame, np.int32))
    manifest = {
        "status": "complete", "worker": index, "workers": workers, "gpu_uuid": gpu_uuid,
        "pci": allowed[gpu_uuid], "gpu_name": torch.cuda.get_device_name(0),
        "assigned_videos": len(assigned), "requested_frames": sum(len(x["selected_frame_indices"]) for x in assigned),
        "completed_frames": len(statistics), "detection_failures": failures,
        "seconds": time.perf_counter() - started, "output_sha256": sha256_file(path), "output_bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def launch():
    config = json.loads(CONFIG.read_text())
    workers = config["gpu_policy"]["allowed"]
    if len(workers) != int(config["gpu_policy"]["workers"]):
        raise RuntimeError("worker configuration mismatch")
    logs = []
    processes = []
    for index, gpu in enumerate(workers):
        log_path = OUTPUT / f"worker-{index:02d}.log"
        handle = log_path.open("w")
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(index),
                   "--workers", str(len(workers)), "--gpu-uuid", gpu["uuid"]]
        processes.append((index, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env), handle))
        logs.append(str(log_path))
    failures = []
    for index, process, handle in processes:
        code = process.wait(); handle.close()
        if code != 0: failures.append({"worker": index, "exit_code": code})
    if failures:
        raise RuntimeError({"worker_failures": failures, "logs": logs})
    aggregate()


def aggregate():
    config = json.loads(CONFIG.read_text())
    workers = int(config["gpu_policy"]["workers"])
    arrays = [np.load(OUTPUT / f"worker-{index:02d}-of-{workers:02d}.npz") for index in range(workers)]
    timings = np.concatenate([value["timings_seconds"] for value in arrays])
    predicted = np.concatenate([value["predicted_keypoints"] for value in arrays])
    frozen = np.concatenate([value["frozen_keypoints"] for value in arrays])
    stats = np.concatenate([value["statistics"] for value in arrays])
    manifests = [json.loads((OUTPUT / f"worker-{i:02d}-manifest.json").read_text()) for i in range(workers)]
    finite = (np.isfinite(predicted[..., :2]).all(2) & np.isfinite(frozen[..., :2]).all(2)
              & (predicted[..., 2] > 0) & (frozen[..., 2] > 0))
    distance = np.linalg.norm(predicted[..., :2] - frozen[..., :2], axis=2)[finite]
    frame_count = sum(row["completed_frames"] for row in manifests)
    requested = sum(row["requested_frames"] for row in manifests)
    wall = max(row["seconds"] for row in manifests)
    names = ["decode", "detector", "pose_with_heatmap", "statistics", "total"]
    timing_summary = {name: {"median_ms": float(np.median(timings[:, i]) * 1000),
                            "p95_ms": float(np.quantile(timings[:, i], .95) * 1000),
                            "mean_ms": float(np.mean(timings[:, i]) * 1000)} for i, name in enumerate(names)}
    single_gpu_fps = frame_count / sum(row["seconds"] for row in manifests)
    metrics = {
        "status": "complete", "created_utc": utcnow(), "train_only": True,
        "wer_utility_reference_or_heldout_read": False,
        "requested_frames": requested, "completed_frames": frame_count,
        "detection_failures": requested - frame_count, "detection_failure_rate": (requested - frame_count) / requested,
        "workers": manifests, "five_gpu_wall_seconds": wall, "aggregate_fps": frame_count / wall,
        "estimated_single_gpu_fps": single_gpu_fps,
        "timing": timing_summary,
        "coordinate_agreement": {"observations": int(len(distance)), "median_pixel_distance": float(np.median(distance)),
                                 "p95_pixel_distance": float(np.quantile(distance, .95))},
        "statistics": {"shape": list(stats.shape), "raw_peak_median": float(np.median(stats[..., 5])),
                       "probability_peak_median": float(np.median(stats[..., 6])),
                       "entropy_median": float(np.median(stats[..., 7]))},
        "projection_827354_frames": {
            "single_gpu_hours": 827354 / single_gpu_fps / 3600,
            "five_gpu_hours": 827354 / (frame_count / wall) / 3600,
        },
    }
    for value in arrays: value.close()
    path = OUTPUT / "metrics.json"; path.write_text(json.dumps(metrics, indent=2) + "\n")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "complete", "created_utc": utcnow(), "config_sha256": sha256_file(CONFIG),
        "selection_sha256": sha256_file(SELECTION), "metrics_sha256": sha256_file(path),
    }, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.launch: launch()
    elif args.worker is not None: run_worker(args.worker, args.workers, args.gpu_uuid)
    else: parser.error("choose --prepare, --launch, or --worker")


if __name__ == "__main__":
    main()
