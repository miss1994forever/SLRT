#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="/home/haojun/projects/SLRT/data/csl-daily"
LOCAL_SCRIPT="/home/haojun/projects/SLRT/Online/download_csl_daily_frames_from_baidu.local.sh"
ARCHIVE_BASENAME="csl-daily-frames-512x512.tar.gz"
FINAL_ZIP="sentence_frames-512x512.zip"
FRAME_DIR="sentence_frames-512x512"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-4}"

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

download_parts_parallel() {
  local running=0
  local cmd=''

  while IFS= read -r cmd; do
    [[ -n "$cmd" ]] || continue
    bash -lc "$cmd" &
    ((running += 1))
    if (( running >= DOWNLOAD_JOBS )); then
      wait -n
      ((running -= 1))
    fi
  done < <(grep -E '^  curl ' "$LOCAL_SCRIPT" | sed 's/^  //')

  while (( running > 0 )); do
    wait -n
    ((running -= 1))
  done
}

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar >/dev/null 2>&1 || die "tar is required"
command -v zip >/dev/null 2>&1 || die "zip is required"

[[ -f "$LOCAL_SCRIPT" ]] || die "Missing local download script: $LOCAL_SCRIPT"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

source "$LOCAL_SCRIPT"
declare -F download_csl_daily_frames_parts >/dev/null 2>&1 || die "download_csl_daily_frames_parts() is not defined in $LOCAL_SCRIPT"

printf 'Downloading CSL-Daily frame archive parts into %s with DOWNLOAD_JOBS=%s\n' "$DATA_DIR" "$DOWNLOAD_JOBS"
download_parts_parallel

for index in 00 01 02 03 04 05 06 07 08 09; do
  [[ -s "${ARCHIVE_BASENAME}_${index}" ]] || die "Missing downloaded part: ${ARCHIVE_BASENAME}_${index}"
done

printf 'Combining archive parts into %s\n' "$ARCHIVE_BASENAME"
cat ${ARCHIVE_BASENAME}_* > "$ARCHIVE_BASENAME"

printf 'Validating tar archive\n'
tar -tzf "$ARCHIVE_BASENAME" >/dev/null

if [[ ! -d "$FRAME_DIR" ]]; then
  printf 'Extracting %s\n' "$ARCHIVE_BASENAME"
  tar -xzf "$ARCHIVE_BASENAME"
fi

[[ -d "$FRAME_DIR" ]] || die "Expected extracted directory: $DATA_DIR/$FRAME_DIR"

printf 'Packing %s into %s\n' "$FRAME_DIR" "$FINAL_ZIP"
rm -f "$FINAL_ZIP"
zip -qr "$FINAL_ZIP" "$FRAME_DIR"

[[ -s "$FINAL_ZIP" ]] || die "Failed to create $DATA_DIR/$FINAL_ZIP"

printf '\nDone. Generated files:\n'
printf '  %s/%s\n' "$DATA_DIR" "$ARCHIVE_BASENAME"
printf '  %s/%s\n' "$DATA_DIR" "$FINAL_ZIP"
printf '  %s/%s\n' "$DATA_DIR" "$FRAME_DIR"