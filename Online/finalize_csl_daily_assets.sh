#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/csl-daily}"
ARCHIVE_BASENAME="csl-daily-frames-512x512.tar.gz"
PARTIAL_ARCHIVE="${ARCHIVE_BASENAME}.partial"
FRAME_DIR="sentence_frames-512x512"
FINAL_ZIP="sentence_frames-512x512.zip"
RAW_KEYPOINT_FILE="$DATA_DIR/csl-daily-keypoints.pkl"
WHOLEBODY_FILE="$DATA_DIR/keypoints_hrnet_dark_coco_wholebody.pkl"
WHOLEBODY_ISO_FILE="$DATA_DIR/keypoints_hrnet_dark_coco_wholebody_iso.pkl"

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

stage() {
  printf '\n[%s] %s\n' "$1" "$2"
}

file_size() {
  stat -c '%s' "$1"
}

human_size() {
  numfmt --to=iec --suffix=B "$1"
}

command -v tar >/dev/null 2>&1 || die "tar is required"
command -v zip >/dev/null 2>&1 || die "zip is required"
command -v numfmt >/dev/null 2>&1 || die "numfmt is required"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

stage "1/6" "Checking downloaded frame archive parts"
expected_bytes=0
for index in 00 01 02 03 04 05 06 07 08 09; do
  [[ -s "${ARCHIVE_BASENAME}_${index}" ]] || die "Missing frame archive part: ${ARCHIVE_BASENAME}_${index}"
  part_size=$(file_size "${ARCHIVE_BASENAME}_${index}")
  expected_bytes=$((expected_bytes + part_size))
  printf '  part %s: %s (%s)\n' "$index" "$part_size" "$(human_size "$part_size")"
done
printf '  expected combined size: %s (%s)\n' "$expected_bytes" "$(human_size "$expected_bytes")"

stage "2/6" "Combining frame archive parts into $ARCHIVE_BASENAME"
if [[ -f "$ARCHIVE_BASENAME" ]]; then
  existing_bytes=$(file_size "$ARCHIVE_BASENAME")
  printf '  existing combined archive: %s (%s)\n' "$existing_bytes" "$(human_size "$existing_bytes")"
else
  existing_bytes=0
  printf '  existing combined archive: missing\n'
fi

if [[ "$existing_bytes" -eq "$expected_bytes" ]]; then
  printf '  existing combined archive already matches source parts; skipping recombine\n'
  actual_bytes="$existing_bytes"
else
  printf '  recombining because current archive size does not match expected size\n'
  rm -f "$PARTIAL_ARCHIVE"
  cat ${ARCHIVE_BASENAME}_* > "$PARTIAL_ARCHIVE"
  actual_bytes=$(file_size "$PARTIAL_ARCHIVE")
  [[ "$actual_bytes" -eq "$expected_bytes" ]] || die "Partial combined archive size mismatch: expected $expected_bytes bytes, got $actual_bytes bytes"
  mv -f "$PARTIAL_ARCHIVE" "$ARCHIVE_BASENAME"
  actual_bytes=$(file_size "$ARCHIVE_BASENAME")
fi

printf '  actual combined size: %s (%s)\n' "$actual_bytes" "$(human_size "$actual_bytes")"
[[ "$actual_bytes" -eq "$expected_bytes" ]] || die "Combined archive size mismatch: expected $expected_bytes bytes, got $actual_bytes bytes"
printf '  combined archive size matches source parts\n'

stage "3/6" "Validating combined tar archive"
tar -tzf "$ARCHIVE_BASENAME" >/dev/null
printf '  tar archive validation passed\n'

if [[ ! -d "$FRAME_DIR" ]]; then
  stage "4/6" "Extracting $ARCHIVE_BASENAME"
  tar -xzf "$ARCHIVE_BASENAME"
else
  stage "4/6" "Skipping extraction because $FRAME_DIR already exists"
fi

[[ -d "$FRAME_DIR" ]] || die "Expected extracted directory: $DATA_DIR/$FRAME_DIR"
printf '  extracted directory ready: %s/%s\n' "$DATA_DIR" "$FRAME_DIR"

stage "5/6" "Packing $FRAME_DIR into $FINAL_ZIP"
rm -f "$FINAL_ZIP"
zip -qr "$FINAL_ZIP" "$FRAME_DIR"
[[ -s "$FINAL_ZIP" ]] || die "Failed to create $DATA_DIR/$FINAL_ZIP"
zip_bytes=$(file_size "$FINAL_ZIP")
printf '  created zip: %s (%s)\n' "$zip_bytes" "$(human_size "$zip_bytes")"

[[ -f "$RAW_KEYPOINT_FILE" ]] || die "Missing raw keypoint file: $RAW_KEYPOINT_FILE"

stage "6/6" "Preparing keypoint files for Online and CTC_fusion"
ln -sfn "$RAW_KEYPOINT_FILE" "$WHOLEBODY_FILE"

CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-slrt_legacy}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python - "$RAW_KEYPOINT_FILE" "$WHOLEBODY_ISO_FILE" <<'PY'
import pickle
import sys

raw_path, iso_path = sys.argv[1:3]

with open(raw_path, 'rb') as handle:
    data = pickle.load(handle)

converted = {}
for name, item in data.items():
    if isinstance(item, dict) and 'keypoints' in item:
        converted[name] = item['keypoints']
    else:
        raise TypeError(f'Unexpected keypoint entry format for {name}: {type(item).__name__}')

with open(iso_path, 'wb') as handle:
    pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('converted_entries', len(converted))
first_key = next(iter(converted))
print('sample_key', first_key)
print('sample_shape', converted[first_key].shape)
PY

printf '\nDone. Generated assets:\n'
printf '  %s/%s\n' "$DATA_DIR" "$FINAL_ZIP"
printf '  %s\n' "$WHOLEBODY_FILE"
printf '  %s\n' "$WHOLEBODY_ISO_FILE"
