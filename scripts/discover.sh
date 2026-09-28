#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DISCOVERY_DIR="$CONVERSION_RUN_ROOT/discovery"
mkdir -p "$DISCOVERY_DIR"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO_ROOT/conversion" \
"$PYTHON_EXECUTABLE" "$REPO_ROOT/conversion/full_discovery.py" \
  --source-root "$AGHRI_SOURCE_ROOT" \
  --split-root "$SPLIT_ROOT" \
  --lidar-config "$REPO_ROOT/conversion/configs/lidar_only.yaml" \
  --camera-config "$REPO_ROOT/conversion/configs/camera_lidar.yaml" \
  --annotation-overrides "$REPO_ROOT/conversion/configs/annotation_overrides.yaml" \
  --output "$DISCOVERY_DIR/inventory.json" \
  --plan "$DISCOVERY_DIR/conversion_plan.md"
