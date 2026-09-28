#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODE="${1:-all}"
HASH_ARGS=()
if [[ "${2:-}" == "--skip-payload-hashes" ]]; then
  HASH_ARGS=(--skip-payload-hashes)
elif [[ -n "${2:-}" ]]; then
  echo "Usage: $0 {all|full|zed_lidar|fisheye_lidar} [--skip-payload-hashes]" >&2
  exit 2
fi
if (( $# > 2 )); then
  echo "Usage: $0 {all|full|zed_lidar|fisheye_lidar} [--skip-payload-hashes]" >&2
  exit 2
fi

case "$MODE" in
  all) VARIANTS=(full zed_lidar fisheye_lidar) ;;
  full|zed_lidar|fisheye_lidar) VARIANTS=("$MODE") ;;
  *)
    echo "Unknown PKL mode: $MODE" >&2
    echo "Use: all, full, zed_lidar, or fisheye_lidar" >&2
    exit 2
    ;;
esac

mkdir -p "$AGHRI_PKL_OUTPUT_ROOT/logs"

SYNCHRONIZATION_ARGS=()
if [[ -n "$AGHRI_SYNCHRONIZATION_DIR" ]]; then
  SYNCHRONIZATION_ARGS=(--synchronization-dir "$AGHRI_SYNCHRONIZATION_DIR")
fi

for PKL_VARIANT in "${VARIANTS[@]}"; do
  OUTPUT_DIR="$AGHRI_PKL_OUTPUT_ROOT/$PKL_VARIANT/pkls"
  REPORT_DIR="$AGHRI_PKL_OUTPUT_ROOT/$PKL_VARIANT/reports"
  MANIFEST_DIR="$AGHRI_PKL_OUTPUT_ROOT/$PKL_VARIANT/manifests"

  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing existing PKL output path: $OUTPUT_DIR" >&2
    exit 2
  fi

  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$REPO_ROOT/pkl_generation" \
  "$PYTHON_EXECUTABLE" "$REPO_ROOT/pkl_generation/pkl_generator.py" \
    --dataset-root "$AGHRI_DATASET_ROOT" \
    "${SYNCHRONIZATION_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR" \
    --manifest-dir "$MANIFEST_DIR" \
    --variant "$PKL_VARIANT" \
    "${HASH_ARGS[@]}" \
    2>&1 | tee "$AGHRI_PKL_OUTPUT_ROOT/logs/${PKL_VARIANT}_generation.log"
done
