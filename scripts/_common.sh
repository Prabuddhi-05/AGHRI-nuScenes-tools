#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PATH_CONFIG="${AGHRI_PATHS_FILE:-$REPO_ROOT/paths.env}"

if [[ ! -f "$PATH_CONFIG" ]]; then
  echo "Missing path configuration: $PATH_CONFIG" >&2
  echo "Create it with: cp '$REPO_ROOT/paths.env.example' '$REPO_ROOT/paths.env'" >&2
  exit 2
fi

# paths.env is a user-owned shell fragment containing path assignments.
# shellcheck disable=SC1090
source "$PATH_CONFIG"

: "${AGHRI_SOURCE_ROOT:?Set AGHRI_SOURCE_ROOT in paths.env}"
: "${CONVERSION_RUN_ROOT:?Set CONVERSION_RUN_ROOT in paths.env}"
: "${AGHRI_DATASET_ROOT:?Set AGHRI_DATASET_ROOT in paths.env}"
: "${CONVERSION_WORK_ROOT:?Set CONVERSION_WORK_ROOT in paths.env}"
: "${AGHRI_PKL_OUTPUT_ROOT:?Set AGHRI_PKL_OUTPUT_ROOT in paths.env}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
SPLIT_ROOT="${SPLIT_ROOT:-$REPO_ROOT/splits}"

export REPO_ROOT PATH_CONFIG PYTHON_EXECUTABLE SPLIT_ROOT
export AGHRI_SOURCE_ROOT CONVERSION_RUN_ROOT AGHRI_DATASET_ROOT
export CONVERSION_WORK_ROOT AGHRI_PKL_OUTPUT_ROOT
export AGHRI_SYNCHRONIZATION_DIR="${AGHRI_SYNCHRONIZATION_DIR:-}"
