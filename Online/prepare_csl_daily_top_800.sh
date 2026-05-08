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