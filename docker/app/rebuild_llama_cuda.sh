#!/usr/bin/env bash
set -euo pipefail
# The shared build entry verifies the persistent volume's revision/backend.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
command -v nvcc >/dev/null || { echo "CUDA compiler unavailable" >&2; exit 1; }
bash /app/scripts/build_llama.sh cuda
/app/llama.cpp/build/bin/llama-server --version
