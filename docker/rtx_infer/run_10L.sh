#!/usr/bin/env bash
# Run the 10-layer (dim 192, state 32, paper shape) inference benchmark.
#
#   ./run_10L.sh <path/to/10L_checkpoint.ckpt> <path/to/preprocessed_data_dir> [BATCH_SIZE]
#
# Env overrides: IMAGE (default ssm-rtx-infer), MATMUL_PRECISION (highest|high),
# ITERS, WARMUP, PRELOAD_BATCHES. Extra args after the 3 positionals pass through.
set -euo pipefail
CKPT="${1:?usage: run_10L.sh <ckpt.ckpt> <data_dir> [batch]}"
DATA="${2:?usage: run_10L.sh <ckpt.ckpt> <data_dir> [batch]}"
BATCH="${3:-16384}"; shift $(( $# < 3 ? $# : 3 )) || true
docker run --gpus all --ipc=host --rm \
  -v "$(cd "$(dirname "$CKPT")" && pwd)":/ckpts:ro \
  -v "$(cd "$DATA" && pwd)":/data:ro \
  -e MODEL=10L \
  -e CKPT="/ckpts/$(basename "$CKPT")" \
  -e DATA_DIR=/data \
  -e BATCH_SIZE="$BATCH" \
  -e MATMUL_PRECISION="${MATMUL_PRECISION:-high}" \
  "${IMAGE:-ssm-rtx-infer}" "$@"
