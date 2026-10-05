#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root with its active Python environment.
exec "${PYTHON_EXECUTABLE:-python}" -u -m lpm_kernel.L2.mlx_training.train \
  --model "${MODEL_BASE_PATH:?Set MODEL_BASE_PATH to the local HF base model}" \
  --output "${MODEL_PERSONAL_DIR:?Set MODEL_PERSONAL_DIR to the adapter output}" \
  --user-name "${USER_NAME:-user}" "$@"
