#!/usr/bin/env bash
# Run the 4-layer (dim 128, state 16) inference benchmark in the container.
#
#   ./run_4L.sh <path/to/4L_checkpoint.ckpt> <path/to/preprocessed_data_dir> [BATCH_SIZE]
#
# Env overrides: IMAGE (default ssm-rtx-infer), MATMUL_PRECISION (highest|high),
# ITERS, WARMUP, PRELOAD_BATCHES. Extra args after the 3 positionals pass through
# to bench_infer.py.
set -euo pipefail
CKPT="${1:?usage: run_4L.sh <ckpt.ckpt> <data_dir> [batch]}"
DATA="${2:?usage: run_4L.sh <ckpt.ckpt> <data_dir> [batch]}"
BATCH="${3:-16384}"; shift $(( $# < 3 ? $# : 3 )) || true
docker run --gpus all --ipc=host --rm \
  -v "$(cd "$(dirname "$CKPT")" && pwd)":/ckpts:ro \
  -v "$(cd "$DATA" && pwd)":/data:ro \
  -e MODEL=4L \
  -e CKPT="/ckpts/$(basename "$CKPT")" \
  -e DATA_DIR=/data \
  -e BATCH_SIZE="$BATCH" \
  -e MATMUL_PRECISION="${MATMUL_PRECISION:-highest}" \
  "${IMAGE:-ssm-rtx-infer}" "$@"
