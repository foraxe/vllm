#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
: "${CUDA_HOME:=/usr/local/cuda}"
flags=()
if [[ ${GG_TEST_BAD_RKEY:-0} == 1 ]]; then flags+=(-DGG_TEST_BAD_RKEY); fi
"$CUDA_HOME/bin/nvcc" "${flags[@]}" -O2 -std=c++17 -Xcompiler -fPIC -shared "$HERE/gather_graph.cu" -o "$HERE/libgather_graph.so" -libverbs -lcuda -lpthread
