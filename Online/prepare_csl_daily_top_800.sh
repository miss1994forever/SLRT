#!/usr/bin/env bash

set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="slrt_legacy"

BUILDER="/home/haojun/projects/build_csl_daily_subset.py"
SRC_DIR="/home/haojun/projects/SLRT/TwoStreamNetwork/data/csl-daily"
DST_DIR="/home/haojun/projects/SLRT/data/csl-daily-top-800-all"
TARGET_GLOSS_FILE="/home/haojun/projects/sign2text.app/research/srtp-SLR/outputs/csl_daily_800_vocab/target_glosses_top_800.txt"
SUBSET_PREFIX="csl-daily-top-800"

TRAIN_ASSET_DIR="/home/haojun/projects/SLRT/data/csl-daily"
FRAME_DIR_CANDIDATES=(
  "$TRAIN_ASSET_DIR/sentence_frames-512x512"
  "$TRAIN_ASSET_DIR/frames_512x512"
)
TRAIN_ASSETS=(
  "$TRAIN_ASSET_DIR/csl-daily-frames-512x512.tar.gz"
  "$TRAIN_ASSET_DIR/keypoints_hrnet_dark_coco_wholebody.pkl"
  "$TRAIN_ASSET_DIR/keypoints_hrnet_dark_coco_wholebody_iso.pkl"
)

SUBSET_OUTPUTS=(
  "$DST_DIR/$SUBSET_PREFIX.train"
  "$DST_DIR/$SUBSET_PREFIX.dev"
  "$DST_DIR/$SUBSET_PREFIX.test"
  "$DST_DIR/gloss2ids.pkl"
  "$DST_DIR/csl_s2g_gloss2ids.pkl"
  "$DST_DIR/csl_iso_with_blank.vocab"
  "$DST_DIR/subset_stats.json"
)

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

check_required_file() {
  local path="$1"
  [[ -e "$path" ]] || die "Missing required file: $path"
}

all_subset_outputs_exist() {
  local path
  for path in "${SUBSET_OUTPUTS[@]}"; do
    [[ -e "$path" ]] || return 1
  done
  return 0
}

print_training_asset_status() {
  printf '\nTraining asset status:\n'
  local path
  local missing=0
  for path in "${TRAIN_ASSETS[@]}"; do
    if [[ -e "$path" ]]; then
      printf '  EXISTS  %s\n' "$path"
    else
      printf '  MISSING %s\n' "$path"
      missing=1
    fi
  done

  if [[ "$missing" -eq 1 ]]; then
    printf '\nThese assets are not required to build the 800-word subset, but they are required before training can start.\n'
  else
    printf '\nAll training assets are present.\n'
  fi
}

printf 'Checking subset prerequisites...\n'
check_required_file "$BUILDER"
check_required_file "$TARGET_GLOSS_FILE"
check_required_file "$SRC_DIR/csl-daily.train"
check_required_file "$SRC_DIR/csl-daily.dev"
check_required_file "$SRC_DIR/csl-daily.test"
check_required_file "$CONDA_SH"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

python - <<'PY' >/dev/null
import torch
print(torch.__version__)
PY

mkdir -p "$DST_DIR"

if all_subset_outputs_exist && [[ "$FORCE" -ne 1 ]]; then
  printf 'Subset outputs already exist. Reuse current files in %s\n' "$DST_DIR"
else
  printf 'Building 800-word subset into %s\n' "$DST_DIR"
  python "$BUILDER" \
    --src-dir "$SRC_DIR" \
    --dst-dir "$DST_DIR" \
    --target-gloss-file "$TARGET_GLOSS_FILE" \
    --subset-prefix "$SUBSET_PREFIX" \
    --match-mode all \
    --write-derived-vocabs
fi

printf '\nFiltering orphan samples with missing frame assets...\n'
python - <<'PY' "$DST_DIR" "$SUBSET_PREFIX" "$TRAIN_ASSET_DIR"
import gzip
import json
import os
import pickle
import sys
import tarfile
from collections import Counter

dst_dir, subset_prefix, train_asset_dir = sys.argv[1:4]

split_paths = {
    split: os.path.join(dst_dir, f"{subset_prefix}.{split}")
    for split in ("train", "dev", "test")
}
stats_path = os.path.join(dst_dir, "subset_stats.json")
frame_dir_candidates = [
    os.path.join(train_asset_dir, "sentence_frames-512x512"),
    os.path.join(train_asset_dir, "frames_512x512"),
]
tar_path = os.path.join(train_asset_dir, "csl-daily-frames-512x512.tar.gz")


def load_split(path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def write_split(path, items):
    with gzip.open(path, "wb") as handle:
        pickle.dump(items, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_available_names():
    for frame_dir in frame_dir_candidates:
        if os.path.isdir(frame_dir):
            print(f"  using extracted frame directory: {frame_dir}")
            return {
                name for name in os.listdir(frame_dir)
                if os.path.isdir(os.path.join(frame_dir, name))
            }

    if os.path.isfile(tar_path):
        print(f"  using tar index: {tar_path}")
        available = set()
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf:
                name = member.name.strip("/")
                if not name.startswith("frames_512x512/"):
                    continue
                parts = name.split("/", 2)
                if len(parts) >= 2 and parts[1]:
                    available.add(parts[1])
        return available

    raise FileNotFoundError("No extracted frame directory or frame tar archive found for orphan filtering")


def summarize(items):
    gloss_counter = Counter()
    lengths = []
    signer_counter = Counter()
    for item in items:
        gloss_seq = item.get("gloss", "").split()
        if gloss_seq:
            gloss_counter.update(gloss_seq)
            lengths.append(len(gloss_seq))
        signer = item.get("signer")
        if signer is not None:
            signer_counter[signer] += 1
    return {
        "num_samples": len(items),
        "num_unique_glosses": len(gloss_counter),
        "avg_gloss_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "gloss_frequency": dict(gloss_counter),
        "signer_frequency": dict(signer_counter),
    }


available_names = build_available_names()
with open(stats_path, encoding="utf-8") as handle:
    subset_stats = json.load(handle)

filter_summary = {}
for split, path in split_paths.items():
    items = load_split(path)
    kept = []
    dropped = []
    for item in items:
        name = item.get("name")
        if name in available_names:
            kept.append(item)
        else:
            dropped.append(name)

    write_split(path, kept)
    split_stats = summarize(kept)
    old_stats = subset_stats.get("splits", {}).get(split, {})
    for key in ("top_blocking_oov_glosses",):
        if key in old_stats:
            split_stats[key] = old_stats[key]
    subset_stats.setdefault("splits", {})[split] = split_stats
    filter_summary[split] = {
        "kept": len(kept),
        "dropped_missing_frames": len(dropped),
        "dropped_names_preview": dropped[:20],
    }
    print(f"  {split}: kept {len(kept)}, dropped {len(dropped)} missing-frame samples")
    if dropped:
        print(f"    first dropped: {dropped[0]}")

subset_stats["frame_asset_filter"] = {
    "enabled": True,
    "source": "directory" if any(os.path.isdir(p) for p in frame_dir_candidates) else "tar",
    "summary": filter_summary,
}

with open(stats_path, "w", encoding="utf-8") as handle:
    json.dump(subset_stats, handle, ensure_ascii=False, indent=2)
PY

printf '\nValidating subset outputs...\n'
for path in "${SUBSET_OUTPUTS[@]}"; do
  check_required_file "$path"
  printf '  OK %s\n' "$path"
done

python - <<'PY' "$DST_DIR/subset_stats.json"
import json
import sys

path = sys.argv[1]
with open(path, encoding='utf-8') as handle:
    data = json.load(handle)

print('\nSubset summary:')
for split in ('train', 'dev', 'test'):
    stats = data['splits'][split]
    print(f"  {split}: {stats['num_samples']} samples, {stats['num_unique_glosses']} unique glosses, avg_len={stats['avg_gloss_len']:.2f}")
PY

print_training_asset_status

printf '\nReady commands:\n'
printf '  bash %s\n' "/home/haojun/projects/SLRT/Online/prepare_csl_daily_top_800.sh"
printf '  bash %s --force\n' "/home/haojun/projects/SLRT/Online/prepare_csl_daily_top_800.sh"