#!/usr/bin/env bash
# Converter, gguf-py, and runtime always come from this pinned source revision.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_REVISION=$(tr -d '\r\n' < "$PROJECT_ROOT/dependencies/llama.cpp.version")
LLAMA_ROOT="$PROJECT_ROOT/llama.cpp"
BACKEND=${1:-cpu}
case "$BACKEND" in cpu|metal|cuda) ;; *) echo "Unknown llama backend: $BACKEND" >&2; exit 1 ;; esac
if [[ -d "$LLAMA_ROOT" && ! -d "$LLAMA_ROOT/.git" ]]; then
    echo "Existing archive-based llama.cpp is obsolete; preserving it before replacement."
    mv "$LLAMA_ROOT" "$PROJECT_ROOT/llama.cpp.backup.$(date +%s)"
fi
if [[ ! -d "$LLAMA_ROOT/.git" ]]; then
    git init "$LLAMA_ROOT"
    git -C "$LLAMA_ROOT" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if [[ $(git -C "$LLAMA_ROOT" rev-parse HEAD 2>/dev/null || true) != "$LLAMA_REVISION" ]]; then
    git -C "$LLAMA_ROOT" fetch --depth 1 origin "$LLAMA_REVISION"
    git -C "$LLAMA_ROOT" checkout --detach FETCH_HEAD
fi
# Persistent Docker build volumes must also carry the exact revision/backend.
BUILD_MARKER="$LLAMA_ROOT/build/.secondme-build"
EXPECTED_MARKER="$LLAMA_REVISION $BACKEND"
if [[ -x "$LLAMA_ROOT/build/bin/llama-server" && -f "$BUILD_MARKER" && $(cat "$BUILD_MARKER") == "$EXPECTED_MARKER" ]]; then
    "$LLAMA_ROOT/build/bin/llama-server" --version
    exit 0
fi
# A mounted build directory itself cannot be removed; remove its contents.
mkdir -p "$LLAMA_ROOT/build"
find "$LLAMA_ROOT/build" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF)
if [[ "$BACKEND" == metal ]]; then
    [[ $(uname -s) == Darwin ]] || { echo "Metal requires native macOS" >&2; exit 1; }
    CMAKE_ARGS+=(-DGGML_METAL=ON)
elif [[ "$BACKEND" == cuda ]]; then
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
fi
# Docker images are built without the target GPU/CPU: avoid "native" targets.
if [[ "${LLAMA_PORTABLE_BUILD:-0}" == 1 ]]; then
    CMAKE_ARGS+=(-DGGML_NATIVE=OFF)
    if [[ "$BACKEND" == cuda ]]; then
        # libcuda.so.1 comes from the host driver at runtime (as in upstream's CUDA image).
        CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCHITECTURES:-75;80;86;89;90}"
                     "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined")
    fi
fi
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" "${CMAKE_ARGS[@]}"
cmake --build "$LLAMA_ROOT/build" --config Release --target llama-server llama-quantize -j "${LLAMA_BUILD_JOBS:-4}"
printf '%s\n' "$EXPECTED_MARKER" > "$BUILD_MARKER"
