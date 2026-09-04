#!/bin/bash
# H100 throughput sweep for the paper's cross-device comparison, mirroring the
# collaborator's RTX 5000 Ada runs (/eos/user/b/bhuth/jonathan_ssm/bench_logs_v2):
# same bench tool, same log format, same batch grid, same ttbar_bench store,
# plus a gpu<N>_model_<model>_<batch>_metrics.csv (memory MiB, power W) polled
# from nvidia-smi during the timed run -- so one plotting script parses both.
#
#   bash scripts/throughput_sweep_h100.sh <config> <ckpt> <model_tag> <out_dir> [gpu]
#
# Deployment settings throughout: TF32 matmuls + BUCKET16 + compiled front-end
# (bench defaults) and the GPU auto-seed in fp64 (fp32 fails physics, 2026-09-04).
set -euo pipefail
CFG="${1:?config}"; CKPT="${2:?ckpt}"; TAG="${3:?model_tag}"; OUT="$(realpath -m "${4:?out_dir}")"; GPU="${5:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-/scratch/colliderml/rtx_share/ttbar_bench}"
BATCHES="${BATCHES:-256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU" TRITON_CACHE_DIR="/tmp/triton_sweep_$GPU"
for BS in $BATCHES; do
  LOG="$OUT/bench_model_${TAG}_${BS}.log"
  CSV="$OUT/gpu${GPU}_model_${TAG}_${BS}_metrics.csv"
  echo "=== bs=$BS -> $LOG"
  # sed -u: unbuffered, so the CSV survives the kill (a buffered sed loses
  # its whole output when the pipeline is killed).  setsid + negative-PID
  # kill takes down nvidia-smi too, not just the tail of the pipeline.
  setsid bash -c "nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits -l 1 -i $GPU | sed -u 's/, /,/' > '$CSV'" &
  SMI=$!
  pixi run -e default python "$REPO/scripts/bench_infer_flat.py" \
    --config "$CFG" --ckpt "$CKPT" --data-dir "$DATA" \
    --batch-size "$BS" --matmul-precision high --iters 100 > "$LOG" 2>&1 || echo "  FAILED (see log)"
  kill -- -"$SMI" 2>/dev/null || kill "$SMI" 2>/dev/null || true; wait "$SMI" 2>/dev/null || true
  grep -E "throughput|peak VRAM" "$LOG" | sed 's/^/  /'
done
echo "SWEEP-DONE -> $OUT"
