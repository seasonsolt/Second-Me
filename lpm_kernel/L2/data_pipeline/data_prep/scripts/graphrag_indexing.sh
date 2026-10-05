#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 3 ]]; then
    echo 'Usage: graphrag_indexing.sh CONFIG ROOT METRICS' >&2
    exit 1
fi
exec "${PYTHON_EXECUTABLE:-python}" -u -m lpm_kernel.L2.data_pipeline.graphrag_indexing.run \
    --config "$1" --root "$2" --metrics "$3"
