#!/usr/bin/env python3
"""Preregistered FACE_RGB tiny-CNN robust-utility predictor experiment."""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import pickle
import resource
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASE = ROOT / "results/phoenix-2014t_ISLR"
DATA = BASE / "p3_fulltrain_robust_continuation_fit6378_dataset_v2_49faacc3"
POSE = BASE / "p3_fulltrain_causal_pose_hand_motion_v1_49faacc3/train_causal_pose_hand_motion_sequences.npz"
SKELETON = BASE / "p3_fulltrain_robust_predictor_oof_v1_49faacc3"
OUTPUT = BASE / "p3_fulltrain_face_rgb_tinycnn_robust_predictor_oof_v1_49faacc3"
ZIP_PATH = REPO / "data/phoenix_2014t/PHOENIX2014T_videos.zip"
KEYPOINTS = REPO / "data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.train.pkl"
SPLIT = BASE / "train_dense_stride1_v1_49faacc3/fit_calibration_split.json"
CHECKPOINT = BASE / "ckpts/best.ckpt"
AUDIT = OUTPUT / "visibility_cost_audit.json"
CONFIG = OUTPUT / "resolved_config_preregistered.json"
CROPS = OUTPUT / "train_fit_face_affine_crops_uint8.npy"
CROP_VALID = OUTPUT / "train_fit_face_crop_valid.npy"
CROP_META = OUTPUT / "crop_cache_manifest.json"
CROP_INDEX = OUTPUT / "ssl_crop_index.npz"
ENCODER_MANIFEST = OUTPUT / "encoder_manifest.json"
FEATURE_MANIFEST = OUTPUT / "feature_manifest.json"
OOF = OUTPUT / "oof_scores.npz"
METRICS = OUTPUT / "metrics.json"

FOLDS = 5
CROP_SIZE = 64
SSL_SAMPLES = 200_000
SSL_EPOCHS = 10
SSL_BATCH = 512
SSL_SEED = 261041
TEMPERATURE = 0.2
BINARY_SEEDS = [261026, 261027, 261028]
SIGNED_SEEDS = [261029, 261030, 261031]
TARGET = np.asarray([[20, 22], [44, 22], [32, 45]], np.float32)
FACE_GROUPS = {"right_eye": (36, 42), "left_eye": (42, 48), "outer_lip": (48, 60)}
EXPECTED = {
    "video_sha256": "49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457",
    "keypoints_sha256": "18a045bfe7790064e06a9016d0949d7f2f8243ae62ddad4f08d3fb5ea3d5b763",
    "checkpoint_sha256": "b3390f0dc4b6a826b53c88d3309b1d98fb75e5b58a5ec3f5779628cf3d51767b",
    "pose_sha256": "ab659a6e5f1f9f545fe705318ff9be5d7bc2d843acad8930d446b4f83e4dbcb3",
    "skeleton_oof_sha256": "9665b83e11a8c6c415142820d5428ec37f8ac234b3b02b68a23d91d8a94eb29a",
}
EXPECTED_GPU = {
    "pci_bus_id": "00000000:81:00.0",
    "uuid": "GPU-e1683bce-0e4f-68bc-54cc-4a2f62f55631",
}


def imp(name):
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FULL = imp("analyze_phoenix_full_robust_predictor_oof")
PARTIAL = FULL.PARTIAL


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array):
    digest = hashlib.sha256()
    view = np.asarray(array).view(np.uint8).reshape(-1)
    for left in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[left:left + 8 * 1024 * 1024])
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def member_name(name, frame):
    return f"images/{name.split('/', 1)[1]}/images{frame + 1:04d}.png"


def source_id(name):
    return name.rsplit("-", 1)[0]


def fit_ids_in_pose_order():
    split = json.loads(SPLIT.read_text())
    pose = np.load(POSE)
    names = [str(x) for x in pose["video_names"].tolist()]
    ids = [name for name in names if split["assignments"].get(name) == "fit"]
    if len(ids) != 6378 or len(set(ids)) != len(ids):
        raise RuntimeError("frozen fit sample identity mismatch")
    if len({source_id(name) for name in ids}) != 578:
        raise RuntimeError("frozen fit source identity mismatch")
    return ids


def frame_offsets(ids, keypoints):
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(keypoints[name]) for name in ids])
    if int(offsets[-1]) != 743147:
        raise RuntimeError(f"unexpected fit frame count: {offsets[-1]}")
    return offsets


def affine_crop(frame, points, return_oob=False):
    face = np.asarray(points[23:91], np.float32)
    centers = []
    for begin, end in FACE_GROUPS.values():
        group = face[begin:end]
        valid = np.isfinite(group).all(1) & (group[:, 2] >= 0.2)
        if int(valid.sum()) < (len(group) + 1) // 2:
            empty = np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8)
            return (empty, False, 1.0) if return_oob else (empty, False)
        centers.append(group[valid, :2].mean(0))
    anchors = np.ascontiguousarray(np.asarray(centers, np.float32))
    matrix = cv2.getAffineTransform(anchors, TARGET)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
        empty = np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8)
        return (empty, False, 1.0) if return_oob else (empty, False)
    crop = cv2.warpAffine(frame, matrix, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if not return_oob:
        return crop, True
    mask = np.ones(frame.shape[:2], np.uint8)
    warped = cv2.warpAffine(mask, matrix, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return crop, True, float(1.0 - warped.mean())


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 16, 3, 2, 1, groups=16, bias=False),
            nn.Conv2d(16, 24, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 24, 3, 2, 1, groups=24, bias=False),
            nn.Conv2d(24, 32, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
        )
        self.projector = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32))

    def embedding(self, x):
        return F.normalize(self.stem(x).mean((2, 3)), dim=1)

    def forward(self, x):
        return F.normalize(self.projector(self.embedding(x)), dim=1)


def encoder_identity():
    torch.manual_seed(0)
    model = TinyEncoder()
    value = {
        "architecture": "Conv3x3 s2 3->16 BN SiLU; DW3x3 s2 16 + PW16->24 BN SiLU; DW3x3 s2 24 + PW24->32 BN SiLU; DW3x3 s1 32 + PW32->32 BN SiLU; GAP",
        "projection_head": "Linear(32,64)-SiLU-Linear(64,32), training only",
        "deployment_embedding": "L2-normalized pre-projection 32-D",
        "parameters_encoder_including_bn": sum(p.numel() for p in model.stem.parameters()),
        "parameters_projection_head": sum(p.numel() for p in model.projector.parameters()),
        "macs_per_valid_face_crop": 724480,
        "macs_per_arrived_frame_when_valid": 724480,
    }
    value["identity_sha256"] = canonical_sha(value)
    return value


def verify_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this must be run outside the sandbox")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly the preregistered UUID")
    visible_uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_uuid != EXPECTED_GPU["uuid"]:
        raise RuntimeError({"unexpected_CUDA_VISIBLE_DEVICES": visible_uuid, "expected": EXPECTED_GPU["uuid"]})
    props = torch.cuda.get_device_properties(0)
    return {"visible_uuid": visible_uuid, "pci_bus_id": EXPECTED_GPU["pci_bus_id"],
            "name": props.name, "total_memory_bytes": props.total_memory}


def identities():
    actual = {
        "video_sha256": sha256_file(ZIP_PATH),
        "keypoints_sha256": sha256_file(KEYPOINTS),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "pose_sha256": sha256_file(POSE),
        "skeleton_oof_sha256": sha256_file(SKELETON / "oof_scores.npz"),
    }
    if actual != EXPECTED:
        raise RuntimeError({"identity_mismatch": actual, "expected": EXPECTED})
    return actual


def percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))


def run_audit():
    if OUTPUT.exists():
        raise FileExistsError(f"audit requires fresh directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    actual = identities()
    gpu = verify_cuda_device()
    ids = fit_ids_in_pose_order()
    chosen = sorted(ids, key=lambda x: hashlib.sha256(f"face-rgb-smoke-v1\0{x}".encode()).digest())[:32]
    with KEYPOINTS.open("rb") as handle:
        keypoints = pickle.load(handle)
    crops, valid, oob, decode_ms, affine_ms, selected_frames = [], [], [], [], [], []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in chosen:
            points = keypoints[name]
            start = max(0, (len(points) - 16) // 2)
            for frame_index in range(start, min(start + 16, len(points))):
                tick = time.perf_counter()
                with zf.open(member_name(name, frame_index)) as handle:
                    raw = handle.read()
                frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                decode_ms.append(1000 * (time.perf_counter() - tick))
                tick = time.perf_counter()
                crop, ok, outside = affine_crop(frame, points[frame_index], True)
                crops.append(crop); valid.append(ok)
                if ok:
                    oob.append(outside)
                affine_ms.append(1000 * (time.perf_counter() - tick))
                selected_frames.append((name, frame_index))
    valid_crops = np.asarray(crops, np.uint8)[np.asarray(valid, bool)]
    torch.manual_seed(SSL_SEED)
    model = TinyEncoder().cuda().eval()
    tensor = torch.from_numpy(valid_crops.transpose(0, 3, 1, 2)).float().cuda().div_(127.5).sub_(1.0)
    with torch.no_grad():
        for _ in range(10):
            model.embedding(tensor[:1])
    torch.cuda.synchronize(); inference_ms = []
    with torch.no_grad():
        for left in range(len(tensor)):
            tick = time.perf_counter(); model.embedding(tensor[left:left + 1]); torch.cuda.synchronize()
            inference_ms.append(1000 * (time.perf_counter() - tick))
    valid_fraction = float(np.mean(valid))
    low_validity = valid_fraction < 0.90
    systematic_oob = len(oob) == 0 or percentile(oob, 50) > 0.15 or percentile(oob, 95) > 0.40
    status = "needs-direction-low-anchor-validity" if low_validity else ("stop-systematic-truncation" if systematic_oob else "pass")
    audit = {
        "status": status, "created_utc": utcnow(), "outcomes_read": False,
        "finger_rgb_outcomes_read": False, "calibration_dev_test_read": False,
        "scope": "32 deterministic train-fit 16-frame windows; identity, face-anchor crop, cache reuse, causality, and cost smoke",
        "identities": actual, "fit_samples": len(ids), "fit_sources": 578,
        "selection": "lowest SHA256(face-rgb-smoke-v1+NUL+sample_id)",
        "samples": len(chosen), "frames": len(selected_frames), "face_crops": len(crops),
        "valid_face_crops": int(np.sum(valid)), "valid_face_fraction": valid_fraction,
        "anchor_validity_stop_rule": "request direction before preregistration if valid_face_fraction < 0.90",
        "affine_out_of_bounds_fraction": {"mean": float(np.mean(oob)) if oob else None,
                                           "p50": percentile(oob, 50) if oob else None,
                                           "p95": percentile(oob, 95) if oob else None,
                                           "maximum": float(np.max(oob)) if oob else None},
        "systematic_truncation_rule": "stop if valid-crop OOB p50>0.15 or p95>0.40",
        "systematic_truncation_detected": systematic_oob,
        "timing_ms_per_frame": {"decode_p50": percentile(decode_ms, 50), "decode_p95": percentile(decode_ms, 95),
                                "face_affine_p50": percentile(affine_ms, 50), "face_affine_p95": percentile(affine_ms, 95),
                                "face_encoder_batch1_p50": percentile(inference_ms, 50),
                                "face_encoder_batch1_p95": percentile(inference_ms, 95)},
        "encoder": encoder_identity(), "gpu": gpu,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "cpu_gpu_transfer": "one 64x64x3 uint8 crop/frame CPU->GPU; 12,288 bytes/frame before tensor conversion",
        "pose_detector_cost": "shared with K but non-zero; excluded from measured incremental crop/encoder timing",
        "causality_checks": {"same_frame_rgb_and_keypoints_only": True, "temporal_crop_fill": False,
                             "overlapping_windows_reuse_frame_cache": True,
                             "candidate_window": "candidate_start+[-7,...,+8]",
                             "deadline": "min(last real frame, decision_arrival+8)",
                             "sample_id_and_frame_count_checked": True},
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def frozen_config():
    audit = json.loads(AUDIT.read_text())
    if audit["status"] != "pass":
        raise RuntimeError("smoke audit did not pass")
    return {
        "experiment": "FACE_RGB fold-specific self-supervised tiny-CNN robust predictor OOF",
        "created_before_feature_encoder_or_oof_outcomes": True,
        "common_protocol": "docs/NEXT_RESEARCH_COMMON_PROTOCOL_FINE_RGB_20260921.md",
        "branch_handoff": "docs/NEXT_RESEARCH_HANDOFF_FACE_RGB_ROBUST_PREDICTOR_20260921.md",
        "identities": audit["identities"], "visibility_cost_audit_sha256": sha256_file(AUDIT),
        "rows": 366802, "sources": 578,
        "class_counts": {"beneficial": 769, "harmful": 2015, "neutral": 364018},
        "crop": {"wholebody_face_range": "23:91", "confidence": 0.2,
                 "groups": {name: f"{begin}:{end}" for name, (begin, end) in FACE_GROUPS.items()},
                 "minimum_valid_per_group": "at least half",
                 "mapping": {"right_eye_center": [20, 22], "left_eye_center": [44, 22], "outer_lip_center": [32, 45]},
                 "operation": "cv2.warpAffine 64x64 bilinear constant-zero border",
                 "invalid": "32-D zero embedding plus validity=0; no temporal fill",
                 "normalization": "RGB uint8 -> [0,1] -> [-1,1]"},
        "encoder": encoder_identity(),
        "ssl": {"outer_train_only": True,
                "stable_selection": "first 200000 valid crops by SHA256(sample_id+NUL+frame)",
                "samples_per_fold": SSL_SAMPLES, "objective": "symmetric NT-Xent", "temperature": TEMPERATURE,
                "epochs": SSL_EPOCHS, "batch": SSL_BATCH, "optimizer": "AdamW", "lr": 0.001,
                "weight_decay": 0.0001, "scheduler": None, "early_stopping": False,
                "augmentation": {"translation_pixels_xy": [-2, 2], "scale": [0.90, 1.00],
                                 "brightness": [0.8, 1.2], "contrast": [0.8, 1.2], "saturation": [0.8, 1.2],
                                 "horizontal_flip": False,
                                 "implementation": "grid_sample align_corners=True; fixed brightness-contrast-saturation order"},
                "seed": SSL_SEED, "augmentation_and_order_seed": "261041+fold", "checkpoint": "fixed epoch 10"},
        "candidate_feature": {"window": "candidate_start+[-7,-6,...,+8]", "start_padding": "repeat first real frame",
                              "end_padding": "repeat last real frame only once EOS has arrived by deadline",
                              "deadline": "min(last real frame, decision_arrival+8)",
                              "per_frame": "face32+validity = 33",
                              "pooling": "four contiguous 4-frame segment means", "width": 132},
        "models": {"RF": {"input": "B0(49)+faceRGB(132)", "width": 181, "status": "primary gate-eligible"},
                   "KRF": {"input": "B0(49)+K(68)+faceRGB(132)", "width": 249, "status": "secondary non-gating"}},
        "folds": "exact frozen robust-oof-v1 source-disjoint mapping; row parity asserted",
        "normalization": "outer-train rows only",
        "predictor": {"network": "MLP 32-16", "epochs": 20, "optimizer": "AdamW", "lr": 0.002,
                      "weight_decay": 0.0001, "batch": 8192, "binary_seeds": BINARY_SEEDS,
                      "signed_seeds": SIGNED_SEEDS, "signed_primary": True},
        "top_K": 769, "primary_comparison": "RF minus frozen K", "gate": "RF signed top-769 utility >=249",
        "family_priority": ["FINGER_RGB", "FACE_RGB"],
        "gpu": {**EXPECTED_GPU, "forbidden_pci": ["00000000:25:00.0", "00000000:41:00.0"], "gpu0_forbidden": True},
        "forbidden": ["FINGER_RGB outcomes", "calibration/dev/test", "closed-loop before integrated family decision",
                      "crop/CNN/augmentation/temporal tuning after OOF", "external data or weights", "git commit"],
        "output_directory": str(OUTPUT),
    }


def preregister():
    if CONFIG.exists() or CROPS.exists() or CROP_META.exists() or ENCODER_MANIFEST.exists() or OOF.exists():
        raise FileExistsError("preregistration must precede feature/encoder/outcome artifacts")
    value = frozen_config()
    CONFIG.write_text(json.dumps(value, indent=2) + "\n")
    return value


def load_config():
    value = json.loads(CONFIG.read_text())
    if value != frozen_config():
        raise RuntimeError("preregistered configuration changed")
    return value


def build_crops():
    load_config()
    if any(path.exists() for path in (CROPS, CROP_VALID, CROP_META, CROP_INDEX)):
        raise FileExistsError("crop cache is non-overwriting")
    ids = fit_ids_in_pose_order()
    with KEYPOINTS.open("rb") as handle:
        keypoints = pickle.load(handle)
    offsets = frame_offsets(ids, keypoints)
    crops = np.lib.format.open_memmap(CROPS, mode="w+", dtype=np.uint8,
                                      shape=(int(offsets[-1]), CROP_SIZE, CROP_SIZE, 3))
    valid = np.lib.format.open_memmap(CROP_VALID, mode="w+", dtype=np.bool_, shape=(int(offsets[-1]),))
    invalid = 0; oob_values = []; decode_ms = []; affine_ms = []; decoded_bytes = 0
    started = time.perf_counter()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for number, name in enumerate(ids, 1):
            base = int(offsets[number - 1])
            for frame_index, points in enumerate(keypoints[name]):
                tick = time.perf_counter()
                with zf.open(member_name(name, frame_index)) as handle:
                    raw = handle.read()
                frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                decode_ms.append(1000 * (time.perf_counter() - tick)); decoded_bytes += len(raw)
                tick = time.perf_counter(); crop, ok, outside = affine_crop(frame, points, True)
                crops[base + frame_index] = crop; valid[base + frame_index] = ok; invalid += not ok
                if ok:
                    oob_values.append(outside)
                affine_ms.append(1000 * (time.perf_counter() - tick))
            if number % 100 == 0 or number == len(ids):
                crops.flush(); valid.flush()
                print(json.dumps({"crop_samples": number, "frames": int(offsets[number]),
                                  "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    crops.flush(); valid.flush(); del crops, valid, keypoints
    valid_map = np.load(CROP_VALID, mmap_mode="r")
    digests = np.empty(len(valid_map), dtype="S32"); digests[:] = b"\xff" * 32
    sample_fold = np.empty(len(valid_map), np.int8)
    for sample_index, name in enumerate(ids):
        begin, end = int(offsets[sample_index]), int(offsets[sample_index + 1])
        sample_fold[begin:end] = PARTIAL.source_fold(source_id(name))
        for frame_index in range(end - begin):
            if valid_map[begin + frame_index]:
                digests[begin + frame_index] = hashlib.sha256(f"{name}\0{frame_index}".encode()).digest()
    order = np.argsort(digests, kind="stable"); order = order[digests[order] != b"\xff" * 32]
    selections = {}
    for fold in range(FOLDS):
        chosen = order[sample_fold[order] != fold][:SSL_SAMPLES].astype(np.int64)
        if len(chosen) != SSL_SAMPLES:
            raise RuntimeError(f"insufficient fold {fold} SSL crops")
        selections[f"fold_{fold}"] = chosen
    np.savez_compressed(CROP_INDEX, video_names=np.asarray(ids), video_offsets=offsets,
                        sample_fold=sample_fold, **selections)
    manifest = {
        "status": "complete", "created_utc": utcnow(), "samples": len(ids), "sources": 578,
        "frames": int(offsets[-1]), "crop_shape": [int(offsets[-1]), 64, 64, 3], "dtype": "uint8",
        "valid_face_crops": int(valid_map.sum()), "invalid_face_crops": int(invalid),
        "invalid_face_crop_rate": float(invalid / len(valid_map)),
        "affine_out_of_bounds_fraction": {"mean": float(np.mean(oob_values)), "p50": percentile(oob_values, 50),
                                           "p95": percentile(oob_values, 95), "maximum": float(np.max(oob_values))},
        "timing_ms_per_frame": {"decode_p50": percentile(decode_ms, 50), "decode_p95": percentile(decode_ms, 95),
                                "face_affine_p50": percentile(affine_ms, 50), "face_affine_p95": percentile(affine_ms, 95)},
        "decoded_png_bytes": decoded_bytes, "elapsed_seconds": time.perf_counter() - started,
        "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "files": {"crops": {"path": str(CROPS), "bytes": CROPS.stat().st_size, "sha256": sha256_file(CROPS)},
                  "valid": {"path": str(CROP_VALID), "bytes": CROP_VALID.stat().st_size, "sha256": sha256_file(CROP_VALID)},
                  "index": {"path": str(CROP_INDEX), "bytes": CROP_INDEX.stat().st_size, "sha256": sha256_file(CROP_INDEX)}},
        "causality": "each crop uses only same-frame RGB and same-frame face landmarks; invalid crops remain zero",
    }
    CROP_META.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def augment(x, generator):
    n = len(x)
    def uniform(low, high, shape):
        return torch.rand(shape, generator=generator).mul_(high - low).add_(low).to(x.device)
    scale = uniform(0.90, 1.00, (n,)); tx = uniform(-2.0, 2.0, (n,)) * (2.0 / (CROP_SIZE - 1)); ty = uniform(-2.0, 2.0, (n,)) * (2.0 / (CROP_SIZE - 1))
    theta = torch.zeros((n, 2, 3), device=x.device); theta[:, 0, 0] = scale; theta[:, 1, 1] = scale; theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    value = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    brightness = uniform(0.8, 1.2, (n, 1, 1, 1)); contrast = uniform(0.8, 1.2, (n, 1, 1, 1)); saturation = uniform(0.8, 1.2, (n, 1, 1, 1))
    value = value * brightness; mean = value.mean((2, 3), keepdim=True); value = (value - mean) * contrast + mean
    gray = 0.299 * value[:, 0:1] + 0.587 * value[:, 1:2] + 0.114 * value[:, 2:3]
    return ((value - gray) * saturation + gray).clamp(0, 1).mul(2.0).sub(1.0)


def ntxent(z1, z2):
    n = len(z1); z = torch.cat([z1, z2], 0); logits = z @ z.T / TEMPERATURE; logits.fill_diagonal_(-torch.inf)
    target = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(logits, target)


def train_encoders():
    load_config(); gpu = verify_cuda_device()
    if not CROP_META.is_file() or ENCODER_MANIFEST.exists():
        raise RuntimeError("complete crop cache required; encoder training is non-overwriting")
    crops = np.load(CROPS, mmap_mode="r"); index = np.load(CROP_INDEX); records = []
    for fold in range(FOLDS):
        augmentation_seed = SSL_SEED + fold; torch.manual_seed(SSL_SEED); np.random.seed(SSL_SEED)
        model = TinyEncoder().cuda().train(); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        generator = torch.Generator().manual_seed(augmentation_seed); selected = np.asarray(index[f"fold_{fold}"], np.int64)
        selected_crops = np.asarray(crops[selected]).copy(); losses = []; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for epoch in range(SSL_EPOCHS):
            order = torch.randperm(len(selected), generator=generator).numpy(); total = 0.0
            for left in range(0, len(order), SSL_BATCH):
                positions = order[left:left + SSL_BATCH]; batch = selected_crops[positions]
                x = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).float().cuda().div_(255.0)
                loss = ntxent(model(augment(x, generator)), model(augment(x, generator)))
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); total += float(loss.detach()) * len(positions)
            losses.append(total / len(selected)); print(json.dumps({"fold": fold, "ssl_epoch": epoch + 1, "loss": losses[-1], "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
        path = OUTPUT / f"face_tinycnn_fold{fold}_epoch10.pt"
        torch.save({"fold": fold, "epoch": 10, "model_seed": SSL_SEED, "augmentation_seed": augmentation_seed, "model": model.state_dict()}, path)
        records.append({"fold": fold, "outer_train_only": True, "samples": len(selected), "selection_indices_sha256": sha256_array(selected),
                        "model_seed": SSL_SEED, "augmentation_and_order_seed": augmentation_seed, "epoch_losses": losses,
                        "seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                        "checkpoint": str(path), "checkpoint_bytes": path.stat().st_size, "checkpoint_sha256": sha256_file(path)})
        del model, optimizer, selected_crops; torch.cuda.empty_cache()
    manifest = {"status": "complete", "created_utc": utcnow(), "gpu": gpu, "encoder": encoder_identity(), "folds": records,
                "five_fold_train_seconds": sum(x["seconds"] for x in records),
                "single_deployment_encoder": "one shared encoder chosen only after family integration; one face invocation per arrived frame"}
    ENCODER_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_features():
    load_config(); gpu = verify_cuda_device()
    if not ENCODER_MANIFEST.is_file() or FEATURE_MANIFEST.exists():
        raise RuntimeError("complete encoders required; feature generation is non-overwriting")
    crops = np.load(CROPS, mmap_mode="r"); valid = np.load(CROP_VALID, mmap_mode="r"); indices = np.flatnonzero(valid); records = []
    for fold in range(FOLDS):
        checkpoint = OUTPUT / f"face_tinycnn_fold{fold}_epoch10.pt"; model = TinyEncoder().cuda().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cuda")["model"])
        path = OUTPUT / f"face_frame_embeddings_fold{fold}_float16.npy"
        feature = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(crops), 33)); feature[:] = 0; feature[:, 32] = valid.astype(np.float16)
        started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for left in range(0, len(indices), 4096):
                current = indices[left:left + 4096]; batch = np.asarray(crops[current])
                x = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).float().cuda().div_(127.5).sub_(1.0)
                feature[current, :32] = model.embedding(x).cpu().numpy().astype(np.float16)
        feature.flush(); del feature, model
        record = {"fold": fold, "path": str(path), "shape": [len(crops), 33], "dtype": "float16", "bytes": path.stat().st_size,
                  "sha256": sha256_file(path), "seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())}
        records.append(record); print(json.dumps({"embedded_fold": fold, "seconds": round(record["seconds"], 1)}), flush=True); torch.cuda.empty_cache()
    value = {"status": "complete", "created_utc": utcnow(), "gpu": gpu, "samples": 6378, "sources": 578, "frames": len(crops),
             "per_frame_width": 33, "candidate_width": 132, "fold_features": records,
             "invalid_face_crops": int((~valid).sum()), "validity_in_feature": True,
             "causality": "same-frame embedding cached once and reused by overlapping candidate windows",
             "cpu_gpu_transfer": "offline extraction copies uint8 crops in batches; deployment copies one crop/frame and returns 32 float values",
             "total_extraction_seconds": sum(x["seconds"] for x in records)}
    FEATURE_MANIFEST.write_text(json.dumps(value, indent=2) + "\n")
    return value


def candidate_summary(frames, start, decision_arrival):
    n = len(frames); offsets = np.arange(-7, 9, dtype=np.int64) + int(start); deadline = min(n - 1, int(decision_arrival) + 8)
    if min(int(offsets[-1]), n - 1) > deadline:
        raise RuntimeError("candidate window exceeds frozen decision deadline")
    indices = np.clip(offsets, 0, n - 1)
    return np.asarray(frames[indices], np.float32).reshape(4, 4, 33).mean(1).reshape(-1)


def load_rows_for_fold(fold):
    index = np.load(CROP_INDEX); names = [str(x) for x in index["video_names"].tolist()]; offsets = np.asarray(index["video_offsets"], np.int64)
    slices = {name: slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names)}
    frame_features = np.load(OUTPUT / f"face_frame_embeddings_fold{fold}_float16.npy", mmap_mode="r")
    pose = PARTIAL.OLD.SequenceFeatures(POSE); count = 366802
    base = np.empty((count, 49), np.float32); skeleton = np.empty((count, 4 * pose.width), np.float32); face = np.empty((count, 132), np.float32)
    rewards = np.empty(count, np.int64); folds = np.empty(count, np.int8); source_indices = np.empty(count, np.int16)
    source_to_index = {}; classes = Counter(); cursor = 0
    for path in FULL.label_paths():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("type") != "block":
                    continue
                k = PARTIAL.visual_summary(pose.history(row["sample_id"], row["decision_arrival"])); frames = frame_features[slices[row["sample_id"]]]
                source = row["source_video_id"]; source_index = source_to_index.setdefault(source, len(source_to_index)); row_fold = PARTIAL.source_fold(source)
                for candidate in (0, 2):
                    inputs = row["predictor_inputs"][candidate]
                    base[cursor] = np.concatenate([np.asarray(inputs["bookkeeping"], np.float32), np.asarray(inputs["prefix"], np.float32)])
                    skeleton[cursor] = k; face[cursor] = candidate_summary(frames, row["candidate_starts"][candidate], row["decision_arrival"])
                    reward = int(row["label"]["robust_reward_by_candidate"][candidate]); rewards[cursor] = reward; folds[cursor] = row_fold; source_indices[cursor] = source_index
                    classes["beneficial" if reward > 0 else "harmful" if reward < 0 else "neutral"] += 1; cursor += 1
    if cursor != count or classes != Counter({"neutral": 364018, "harmful": 2015, "beneficial": 769}):
        raise RuntimeError({"rows": cursor, "classes": classes})
    old = np.load(SKELETON / "oof_scores.npz")
    for key, current in (("folds", folds), ("rewards", rewards), ("source_indices", source_indices)):
        if not np.array_equal(old[key], current):
            raise RuntimeError(f"frozen OOF row mismatch: {key}")
    return base, skeleton, face, rewards, folds, source_indices, old


def train_fold(matrix, rewards, train, evaluate, device):
    positive = (rewards > 0).astype(np.float32); signed = np.where(rewards < 0, 0, np.where(rewards > 0, 2, 1)).astype(np.int64)
    train_x, eval_x = FULL.normalized_tensors(matrix, train, evaluate, device)
    binary_y = torch.from_numpy(positive[train]).to(device); signed_y = torch.from_numpy(signed[train]).to(device)
    binary = np.mean([FULL.train_binary(train_x, binary_y, eval_x, seed, device) for seed in BINARY_SEEDS], axis=0)
    signed_score = np.mean([FULL.train_signed(train_x, signed_y, eval_x, seed, device) for seed in SIGNED_SEEDS], axis=0)
    return binary, signed_score


def run_oof():
    cfg = load_config(); gpu = verify_cuda_device()
    if not FEATURE_MANIFEST.is_file() or OOF.exists() or METRICS.exists():
        raise RuntimeError("complete features required and OOF is non-overwriting")
    device = torch.device("cuda"); torch.set_num_threads(min(8, os.cpu_count() or 1)); torch.use_deterministic_algorithms(True)
    scores = {objective: {name: np.full(366802, np.nan, np.float32) for name in ("RF", "KRF")} for objective in ("binary", "signed")}
    fold_summaries = []; canonical = None; started = time.perf_counter()
    for fold in range(FOLDS):
        base, skeleton, face, rewards, folds, source_indices, old = load_rows_for_fold(fold)
        if canonical is None:
            canonical = (rewards.copy(), folds.copy(), source_indices.copy(), old)
        else:
            for a, b in zip(canonical[:3], (rewards, folds, source_indices)):
                if not np.array_equal(a, b):
                    raise RuntimeError("cross-fold row identity mismatch")
        evaluate = folds == fold; train = ~evaluate
        for name, matrix in (("RF", np.concatenate([base, face], 1)), ("KRF", np.concatenate([base, skeleton, face], 1))):
            binary, signed_score = train_fold(matrix, rewards, train, evaluate, device); scores["binary"][name][evaluate] = binary; scores["signed"][name][evaluate] = signed_score
        fold_summaries.append({"fold": fold, "train_rows": int(train.sum()), "eval_rows": int(evaluate.sum()), "train_sources_exclude_eval": True,
                               "encoder_checkpoint_sha256": sha256_file(OUTPUT / f"face_tinycnn_fold{fold}_epoch10.pt")})
        print(json.dumps({"predictor_fold": fold, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True); del base, skeleton, face
    rewards, folds, source_indices, old = canonical
    if any(np.isnan(v).any() for objective in scores.values() for v in objective.values()):
        raise RuntimeError("OOF score coverage failure")
    all_scores = {"binary": {"B0": old["binary_B0"], "K": old["binary_K"], **scores["binary"]},
                  "signed": {"B0": old["signed_B0"], "K": old["signed_K"], **scores["signed"]}}
    positive = (rewards > 0).astype(np.float32)
    metric = {objective: {name: PARTIAL.metrics(positive, rewards, array) for name, array in variants.items()} for objective, variants in all_scores.items()}
    k = int(cfg["top_K"]); ci_rfk = FULL.source_bootstrap_delta(rewards, source_indices, all_scores["signed"]["RF"], all_scores["signed"]["K"], k)
    ci_krfk = FULL.source_bootstrap_delta(rewards, source_indices, all_scores["signed"]["KRF"], all_scores["signed"]["K"], k)
    utility = metric["signed"]["RF"]["top_K_signed_robust_utility"]; delta = utility - metric["signed"]["K"]["top_K_signed_robust_utility"]; gate = utility >= 249
    decision = "branch-strong-go-await-finger-and-integration" if gate else ("ranking-only" if delta > 0 and ci_rfk[0] > 0 else "no-go")
    manifest = json.loads(FEATURE_MANIFEST.read_text()); enc = json.loads(ENCODER_MANIFEST.read_text())
    value = {"created_utc": utcnow(), "device": gpu, "rows": len(rewards), "sources": 578,
             "class_counts": dict(sorted(Counter(map(int, rewards)).items())), "feature_widths": {"B0": 49, "K": 117, "RF": 181, "KRF": 249},
             "folds": fold_summaries, "metrics": metric, "signed_RF_minus_K_top_K_utility": delta,
             "signed_RF_minus_K_top_K_utility_source_bootstrap_95pct": ci_rfk,
             "signed_KRF_minus_K_top_K_utility_source_bootstrap_95pct_secondary": ci_krfk,
             "cost": {"encoder_parameters": enc["encoder"]["parameters_encoder_including_bn"],
                      "encoder_macs_per_arrived_frame": enc["encoder"]["macs_per_arrived_frame_when_valid"],
                      "five_fold_ssl_train_seconds": enc["five_fold_train_seconds"],
                      "five_fold_feature_extraction_seconds": manifest["total_extraction_seconds"],
                      "deployment": "one shared encoder, invoked once for every arrived frame; cached embeddings reused"},
             "gate": {"threshold_errors": 249, "RF_utility": utility, "passed": gate, "decision": decision,
                      "action": "do_not_run_closed_loop_in_this_branch"},
             "limitations": ["fit-only OOF ranking", "bounded +8 candidate lookahead", "pose detector shared but non-zero",
                             "KRF is secondary and cannot unlock closed loop"], "elapsed_seconds": time.perf_counter() - started}
    np.savez_compressed(OOF, folds=folds, rewards=rewards, source_indices=source_indices,
                        binary_B0=all_scores["binary"]["B0"], binary_K=all_scores["binary"]["K"], binary_RF=all_scores["binary"]["RF"], binary_KRF=all_scores["binary"]["KRF"],
                        signed_B0=all_scores["signed"]["B0"], signed_K=all_scores["signed"]["K"], signed_RF=all_scores["signed"]["RF"], signed_KRF=all_scores["signed"]["KRF"])
    value["oof_scores"] = {"path": str(OOF), "bytes": OOF.stat().st_size, "sha256": sha256_file(OOF)}
    METRICS.write_text(json.dumps(value, indent=2) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("audit", "preregister", "build-crops", "train-encoders", "build-features", "run-oof")); args = parser.parse_args()
    functions = {"audit": run_audit, "preregister": preregister, "build-crops": build_crops,
                 "train-encoders": train_encoders, "build-features": build_features, "run-oof": run_oof}
    print(json.dumps(functions[args.command](), indent=2))


if __name__ == "__main__":
    main()
