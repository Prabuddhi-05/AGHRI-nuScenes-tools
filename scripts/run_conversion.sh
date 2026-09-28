#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

RESUME_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
  RESUME_ARGS=(--resume)
  shift
fi
if (( $# != 0 )); then
  echo "Usage: $0 [--resume]" >&2
  exit 2
fi

mkdir -p "$CONVERSION_RUN_ROOT/.matplotlib"

PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR="$CONVERSION_RUN_ROOT/.matplotlib" \
PYTHONPATH="$REPO_ROOT/conversion" \
"$PYTHON_EXECUTABLE" "$REPO_ROOT/conversion/full_conversion.py" \
  --source-root "$AGHRI_SOURCE_ROOT" \
  --split-root "$SPLIT_ROOT" \
  --output-root "$AGHRI_DATASET_ROOT" \
  --work-root "$CONVERSION_WORK_ROOT" \
  --discovery-report "$CONVERSION_RUN_ROOT/discovery/inventory.json" \
  --lidar-config "$REPO_ROOT/conversion/configs/lidar_only.yaml" \
  --camera-config "$REPO_ROOT/conversion/configs/camera_lidar.yaml" \
  --annotation-overrides "$REPO_ROOT/conversion/configs/annotation_overrides.yaml" \
  --execute \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$CONVERSION_RUN_ROOT/conversion.log"
