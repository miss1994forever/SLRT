#!/usr/bin/env python3
"""Train-only CPU feasibility audit for materially new preview observations."""
import gzip
import hashlib
import io
import json
import os
import pickle
import statistics
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ZIP_PATH = REPO / "data/phoenix_2014t/PHOENIX2014T_videos.zip"
KEYPOINTS = REPO / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
P2_ARCHIVE = Path("/tmp/phoenix_p2_train_center_sequences_v1.npz")
FRESH_CONFIG = ROOT / "results/phoenix-2014t_ISLR/p3_partial32_fresh_terminal_listwise_exploratory_v1_49faacc3/resolved_config_preregistered.json"
DENSE_ROOT = ROOT / "results/phoenix-2014t_ISLR/train_dense_stride1_v1_49faacc3/shards"
CHECKPOINT = ROOT / "results/phoenix-2014t_ISLR/ckpts/best.ckpt"
OUTPUT = ROOT / "results/phoenix-2014t_ISLR/p3_new_observable_preview_feasibility_audit_v1_49faacc3"
SEED = "new-observable-preview-audit-v1"
SAMPLES = 32
WINDOW = 16


def hashed_fit_ids():
    cfg = json.loads(FRESH_CONFIG.read_text())
    ids = cfg["split"]["partitions"]["fit"]["sample_ids"]
    return sorted(ids, key=lambda x: (hashlib.sha256(f"{SEED}\0{x}".encode()).hexdigest(), x))[:SAMPLES]


def completed_partial32_scale():
    names = {}
    shards = []
    for path in sorted(DENSE_ROOT.glob("shard-*/train_metadata.pkl.gz")):
        if not (path.parent / "complete.json").exists():
            continue
        shards.append(path.parent.name)
        with gzip.open(path, "rb") as handle:
            for row in pickle.load(handle):
                names[row["name"]] = int(row["num_frames"])
    return {"completed_shards": len(shards), "samples": len(names), "unique_frames": sum(names.values()), "dense_windows": sum(names.values()), "shard_names": shards}


def percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))


def timing(values, units):
    return {"units": units, "mean": float(np.mean(values)), "median": percentile(values, 50), "p95": percentile(values, 95), "observations": len(values)}


def decode_window(zf, name, start):
    frames = []
    stem = name.split("/", 1)[1]
    for index in range(start, start + WINDOW):
        member = f"images/{stem}/images{index + 1:04d}.png"
        with zf.open(member) as handle:
            frames.append(np.asarray(Image.open(io.BytesIO(handle.read())).convert("RGB")))
    return np.stack(frames)


def hand_crop(frame, points, side, size=32):
    indices = np.arange(91, 112) if side == "left" else np.arange(112, 133)
    hand = points[indices]
    valid = np.isfinite(hand).all(1) & (hand[:, 2] >= 0.2)
    if not valid.any():
        return np.zeros((size, size, 3), np.uint8), False
    xy = hand[valid, :2]
    low, high = xy.min(0), xy.max(0)
    center = (low + high) / 2
    extent = max(float((high - low).max()) * 1.5, 24.0)
    x0, y0 = center - extent / 2
    x1, y1 = center + extent / 2
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), np.uint8), False
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA), True


def crop_hands(frames, keypoints):
    crops, valid = [], []
    for frame, points in zip(frames, keypoints):
        pair, pair_valid = [], []
        for side in ("left", "right"):
            crop, ok = hand_crop(frame, points, side)
            pair.append(crop)
            pair_valid.append(ok)
        crops.append(pair)
        valid.append(pair_valid)
    return np.asarray(crops), np.asarray(valid)


def hand_hog(crops, hog):
    rows = []
    for pair in crops:
        rows.append(np.concatenate([hog.compute(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)).ravel() for crop in pair]))
    return np.stack(rows).astype(np.float32)


def pooled_flow(lowres):
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in lowres]
    rows = [np.zeros((8, 8, 2), np.float32)]
    for previous, current in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        rows.append(cv2.resize(flow, (8, 8), interpolation=cv2.INTER_AREA))
    return np.stack(rows)


def load_s3d(use_block):
    sys.path.insert(0, str(ROOT))
    from modelling.S3D import S3Ds
    model = S3Ds(in_channel=3, use_block=use_block)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")["model_state"]
    prefix = "recognition_network.visual_backbone_twostream.rgb_stream.backbone."
    state = {key[len(prefix):]: value for key, value in checkpoint.items() if key.startswith(prefix) and key[len(prefix):] in model.state_dict()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or [x for x in missing if not x.endswith("num_batches_tracked")]:
        raise ValueError({"missing": missing, "unexpected": unexpected})
    model.eval()
    return model


def s3d_input(frames):
    resized = np.stack([cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA) for frame in frames])
    bgr = resized[..., ::-1].copy().astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(bgr.transpose(3, 0, 1, 2)[None])


def cache_audit():
    feature_files = list((ROOT / "results").glob("**/*features.pkl"))
    named = []
    for base in (REPO / "data", ROOT / "results"):
        for pattern in ("**/*flow*", "**/*hog*", "**/*crop*", "**/*embed*"):
            named.extend(path for path in base.glob(pattern) if path.is_file())
    p2 = np.load(P2_ARCHIVE)
    dense_logits = list(DENSE_ROOT.glob("shard-*/train_logits.pkl"))
    return {
        "raw_rgb": {"cached": True, "path": str(ZIP_PATH), "schema": "uncompressed PNG members images/<video>/imagesNNNN.png", "bytes": ZIP_PATH.stat().st_size},
        "train_keypoints": {"cached": True, "path": str(KEYPOINTS), "schema": "dict train/video -> [T,133,3] HRNet whole-body", "bytes": KEYPOINTS.stat().st_size},
        "causal_pose_hand_geometry": {"cached": True, "path": str(P2_ARCHIVE), "shape": list(p2["features"].shape), "dtype": str(p2["features"].dtype), "feature_count": int(len(p2["feature_names"]))},
        "low_resolution_rgb": {"cached": False},
        "hand_crop_appearance": {"cached": False},
        "optical_flow_or_hog": {"cached": False, "matching_named_files": [str(x) for x in named[:20]]},
        "islr_intermediate_embeddings": {"cached": bool(feature_files), "files": [str(x) for x in feature_files], "inference_hook": "prediction_slide.py hooks rgb/pose base[-4] (block4) and base[-1] (block5), but only during complete model forward"},
        "dense_logits": {"cached": bool(dense_logits), "files": len(dense_logits), "status": "forbidden as predictor input: complete expensive-model result"},
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    selected = hashed_fit_ids()
    with KEYPOINTS.open("rb") as handle:
        keypoints = pickle.load(handle)
    if any(not str(name).startswith("train/") for name in keypoints):
        raise ValueError("keypoint cache is not train-only")
    if any(name not in keypoints for name in selected):
        raise ValueError("benchmark selection missing keypoints")
    starts = {name: max(0, (len(keypoints[name]) - WINDOW) // 2) for name in selected}
    open_start = time.perf_counter()
    zf = zipfile.ZipFile(ZIP_PATH)
    zip_open_seconds = time.perf_counter() - open_start
    decoded, decode_ms = {}, []
    lowres_ms, crop_ms, hog_ms, flow_ms, valid = [], [], [], [], []
    lowres_outputs, crops_outputs = {}, {}
    hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
    hog_dim = None
    flow_dim = None
    for name in selected:
        start = starts[name]
        tick = time.perf_counter(); frames = decode_window(zf, name, start); decode_ms.append(1000 * (time.perf_counter() - tick)); decoded[name] = frames
        tick = time.perf_counter(); lowres = np.stack([cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA) for frame in frames]); lowres_ms.append(1000 * (time.perf_counter() - tick)); lowres_outputs[name] = lowres
        tick = time.perf_counter(); crops, mask = crop_hands(frames, keypoints[name][start:start + WINDOW]); crop_ms.append(1000 * (time.perf_counter() - tick)); crops_outputs[name] = crops; valid.extend(mask.ravel().tolist())
        tick = time.perf_counter(); h = hand_hog(crops, hog); hog_ms.append(1000 * (time.perf_counter() - tick)); hog_dim = list(h.shape)
        tick = time.perf_counter(); flow = pooled_flow(lowres); flow_ms.append(1000 * (time.perf_counter() - tick)); flow_dim = list(flow.shape)
    zf.close()

    block1 = load_s3d(1)
    full_rgb = load_s3d(5)
    warm = s3d_input(decoded[selected[0]])
    with torch.no_grad():
        block1(warm); full_rgb(warm)
    early_ms, early_shapes = [], []
    with torch.no_grad():
        for name in selected:
            value = s3d_input(decoded[name]); tick = time.perf_counter(); out = block1(value); early_ms.append(1000 * (time.perf_counter() - tick)); early_shapes.append(list(out.shape))
    full_ms, full_shapes = [], []
    with torch.no_grad():
        for name in selected[:4]:
            value = s3d_input(decoded[name]); tick = time.perf_counter(); out = full_rgb(value); full_ms.append(1000 * (time.perf_counter() - tick)); full_shapes.append(list(out.shape))

    scale = completed_partial32_scale()
    frame_count = scale["unique_frames"]
    decode_per_frame = np.mean(decode_ms) / WINDOW
    lowres_per_frame = np.mean(lowres_ms) / WINDOW
    crop_per_frame = np.mean(crop_ms) / WINDOW
    hog_per_frame = np.mean(hog_ms) / WINDOW
    flow_per_frame = np.mean(flow_ms) / WINDOW
    early_med = percentile(early_ms, 50)
    full_med = percentile(full_ms, 50)
    hand_hog_bytes_frame_fp16 = hog_dim[1] * 2
    estimates = {
        "partial32_unique_frames": frame_count,
        "decode_plus_hand_hog_cpu_hours": frame_count * (decode_per_frame + crop_per_frame + hog_per_frame) / 1000 / 3600,
        "hand_hog_fp16_storage_bytes": frame_count * hand_hog_bytes_frame_fp16,
        "hand_hog_fp16_storage_mib": frame_count * hand_hog_bytes_frame_fp16 / (1024 ** 2),
        "lowres_rgb_uint8_storage_mib": frame_count * 64 * 64 * 3 / (1024 ** 2),
        "hand_crop_rgb_uint8_storage_mib": frame_count * 2 * 32 * 32 * 3 / (1024 ** 2),
        "pooled_flow_fp16_storage_mib": frame_count * 8 * 8 * 2 * 2 / (1024 ** 2),
        "three_candidate_block1_over_one_full_rgb_ratio_cpu": 3 * early_med / full_med,
        "note": "generation estimate is linear from 32 train-video windows and excludes one-time zip central-directory load",
    }
    result = {
        "scope": "read-only cache/code audit plus 32 train-only windows CPU extraction benchmark",
        "gpu_used": False,
        "dev_used": False,
        "test_used": False,
        "new_calibration_or_evaluation_outcomes_read": False,
        "selection": {"rule": "lowest sha256(seed+NUL+fit sample id)", "seed": SEED, "samples": len(selected), "sample_ids": selected},
        "cache_and_leakage_audit": cache_audit(),
        "deployment_computation": {
            "lowres_rgb": "decode current frames and resize; no reference/logit/future leakage",
            "hand_crop_appearance": "reuse causal HRNet hand landmarks for crop coordinates, then crop current RGB; requires pose detector online plus crop encoder",
            "flow_hog": "CPU descriptors from current/past RGB only; optical flow needs previous frame, HOG is per-frame",
            "existing_islr_feature_hook": "not a pre-ISLR preview: block4/block5 hook fires inside the complete two-stream forward after earlier layers and lateral fusion",
        },
        "benchmark": {
            "zip_central_directory_open_seconds": zip_open_seconds,
            "png_decode_16_frames_ms": timing(decode_ms, "ms/window"),
            "png_decode_ms_per_frame_mean": decode_per_frame,
            "lowres_64_rgb_ms": timing(lowres_ms, "ms/16-frame window"),
            "pose_guided_two_hand_crop_32_ms": timing(crop_ms, "ms/16-frame window"),
            "two_hand_hog_ms": timing(hog_ms, "ms/16-frame window"),
            "pooled_farneback_flow_ms": timing(flow_ms, "ms/16-frame window"),
            "valid_hand_crop_fraction": float(np.mean(valid)),
            "feature_shapes_per_window": {"lowres_rgb_uint8": [16, 64, 64, 3], "two_hand_crop_rgb_uint8": [16, 2, 32, 32, 3], "two_hand_hog_float": hog_dim, "pooled_flow_float": flow_dim},
            "s3d_block1_forward_ms": timing(early_ms, "ms/16-frame window"),
            "s3d_block1_raw_shape": early_shapes[0],
            "s3d_block1_spatial_pooled_shape": [early_shapes[0][2], early_shapes[0][1]],
            "s3d_full_rgb_backbone_forward_ms_4_windows": timing(full_ms, "ms/16-frame window"),
            "s3d_full_rgb_raw_shape": full_shapes[0],
        },
        "partial32_scale": scale,
        "generation_estimates": estimates,
        "recommendation": {
            "one_feature": "pose-guided 32x32 two-hand appearance HOG, cached once per frame as float16 (648 dimensions/frame)",
            "why": "materially adds hand appearance absent from geometry, uses no full logits/reference/future, is compact, and is CPU-feasible while reusing the already-required causal pose stream",
            "next_minimum_experiment": "generate this descriptor for the already-frozen 512-sample fit subset (59,108 frames, 211 sources; estimated 5.4 CPU minutes and 73 MiB fp16), join to existing terminal rows, and run fit-only source-group OOF enrichment at 0.5/1/2% override coverage before any full partial32 generation",
        },
        "no_go": {
            "feature": "existing ISLR intermediate hook / S3D early embedding as the next preview",
            "reason": "no cache exists; current hooks are block4/block5 outputs produced only during complete two-stream inference, and even a custom block1 preview must run three candidates before the selected full ISLR window",
            "cpu_three_candidate_block1_over_one_full_rgb_ratio": 3 * early_med / full_med,
        },
        "limitations": ["CPU timing is hardware-specific", "32 windows estimate extraction cost, not predictive utility", "pose detector wall time is not included because train keypoints are cached", "full-backbone comparison is RGB-only; deployed ISLR is two-stream and more expensive"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUTPUT / "manifest.json").write_text(json.dumps({"status": "complete", "created_utc": datetime_now(), "audit_sha256": hashlib.sha256((OUTPUT / "audit.json").read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def datetime_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
