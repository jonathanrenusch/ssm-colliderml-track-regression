#!/usr/bin/env bash
# Entrypoint for the SSM inference-profiling image.
#
# Pick a model with MODEL=4L | 10L (uses the bundled preset config), OR give an
# explicit CONFIG=. Always provide CKPT= and DATA_DIR=. Everything else has a
# sensible default and can be overridden by env var; extra CLI args pass through
# to bench_infer.py.
#
#   docker run --gpus all --ipc=host \
#     -v /path/to/checkpoints:/ckpts -v /scratch/colliderml:/data \
#     -e MODEL=4L -e CKPT=/ckpts/4L.ckpt \
#     -e DATA_DIR=/data/arxiv_retraining/p200_core_kf_hits_finetune \
#     -e BATCH_SIZE=16384 ssm-rtx-infer
set -euo pipefail

PRESET_DIR=/workspace/docker/rtx_infer/presets
case "${MODEL:-}" in
  4L)  CONFIG="${CONFIG:-$PRESET_DIR/4L_dim128_state16.yaml}"  ;;
  10L) CONFIG="${CONFIG:-$PRESET_DIR/10L_dim192_state32.yaml}" ;;
  "")  : ;;   # no preset: expect explicit CONFIG
  *)   echo "MODEL must be 4L or 10L (got '$MODEL'); or set CONFIG=" >&2; exit 2 ;;
esac

: "${CONFIG:?set MODEL=4L|10L or CONFIG=/path/to/config.yaml}"
: "${CKPT:?set CKPT=/path/to/checkpoint.ckpt}"
: "${DATA_DIR:?set DATA_DIR=/path/to/preprocessed_data_dir}"

exec python /workspace/docker/rtx_infer/bench_infer.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --data-dir "$DATA_DIR" \
  --batch-size "${BATCH_SIZE:-8192}" \
  --preload-batches "${PRELOAD_BATCHES:-16}" \
  --warmup "${WARMUP:-20}" \
  --iters "${ITERS:-200}" \
  --variant "${VARIANT:-v5pc}" \
  --matmul-precision "${MATMUL_PRECISION:-high}" \
  --loader-workers "${LOADER_WORKERS:-8}" \
  "$@"
