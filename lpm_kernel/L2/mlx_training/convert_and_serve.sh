#!/usr/bin/env bash
set -euo pipefail
# Use the same fused HF -> pinned llama.cpp GGUF chain as the Web workflow.
"${PYTHON_EXECUTABLE:-python}" -m mlx_lm fuse --model "${MODEL_BASE_PATH:?}" \
  --adapter-path "${MODEL_PERSONAL_DIR:?}" --save-path "${MODEL_MERGED_DIR:?}"
mkdir -p "${MODEL_GGUF_DIR:?}"
"${PYTHON_EXECUTABLE:-python}" lpm_kernel/L2/convert_hf_to_gguf.py "$MODEL_MERGED_DIR" \
  --outfile "$MODEL_GGUF_DIR/model.gguf" --outtype f16
printf '%s\n' "GGUF ready: $MODEL_GGUF_DIR/model.gguf"
# Serving is performed by the application's local LLM service.
