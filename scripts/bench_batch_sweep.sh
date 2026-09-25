#!/bin/bash
# Throughput versus batch size, one log per batch size (input of plot_throughput.py).
#
#   bash scripts/bench_batch_sweep.sh <config> <ckpt> <data_dir> <out_dir> [tag] [mode]
#
# Writes <out_dir>/bench_<tag>_<batchsize>.log.  Run on an otherwise idle GPU
# (select it with CUDA_VISIBLE_DEVICES).
set -eu
CONFIG=$1; CKPT=$2; DATA=$3; OUT=$4; TAG=${5:-minGRU}; MODE=${6:-deployed}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
for BS in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144; do
  L="$OUT/bench_${TAG}_${BS}.log"
  python "$HERE/bench_infer.py" --config "$CONFIG" --ckpt "$CKPT" --data-dir "$DATA" \
    --batch-size "$BS" --mode "$MODE" > "$L" 2>&1 || true
  echo "bs=$BS $(grep -E 'throughput|out of memory' "$L" || echo FAILED)"
done
