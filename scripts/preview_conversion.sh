#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO_ROOT/conversion" \
"$PYTHON_EXECUTABLE" "$REPO_ROOT/conversion/full_conversion.py" \
  --source-root "$AGHRI_SOURCE_ROOT" \
  --split-root "$SPLIT_ROOT" \
  --output-root "$AGHRI_DATASET_ROOT" \
  --work-root "$CONVERSION_WORK_ROOT" \
  --discovery-report "$CONVERSION_RUN_ROOT/discovery/inventory.json" \
  --lidar-config "$REPO_ROOT/conversion/configs/lidar_only.yaml" \
  --camera-config "$REPO_ROOT/conversion/configs/camera_lidar.yaml" \
  --annotation-overrides "$REPO_ROOT/conversion/configs/annotation_overrides.yaml"
