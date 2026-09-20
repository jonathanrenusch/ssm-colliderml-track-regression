#!/bin/bash
# Batch-size throughput sweep for the v2 paper figure, one model per call.
# Writes logs named the way scripts/plot_throughput_*.py parse them.
#
#   bash scripts/abl_v2_batch_sweep.sh <model_tag> <run_dir> <ckpt> <dtype> <gpu>
set -eu
TAG="$1"; RUN="$2"; CK="$3"; DT="${4:-float16}"; G="${5:-3}"
REPO=/shared/tracking/ssm-colliderml-track-regression
OUT=$REPO/eval_plots/paper_plots/throughput_h100_v2
mkdir -p "$OUT"
cd "$REPO"
export TRK_SEED_DTYPE=float64 TRK_SSD_BUCKET16=1 TRK_COMPILE_FRONTEND=1
export TRITON_CACHE_DIR=/tmp/tc_sweep_gpu${G}
# Pin the GPU with CUDA_VISIBLE_DEVICES rather than --device cuda:N:
# bench_infer_flat reads torch.cuda.max_memory_allocated() on the CURRENT
# device, so a non-zero --device index reports 0.00 GiB peak VRAM.
export CUDA_VISIBLE_DEVICES="$G"
for BS in 256 1024 4096 16384 32768 65536 131072; do
  L="$OUT/bench_model_${TAG}_${BS}.log"
  if grep -q "throughput  *:" "$L" 2>/dev/null; then echo "$TAG bs=$BS (cached)"; continue; fi
  # bench_infer_flat prints its full report and then hangs on exit (its
  # dataloader threads are never joined).  Run it detached, wait for the
  # report -- "peak VRAM" is the last line -- and then kill it: the numbers
  # are complete at that point and waiting out the hang triples the sweep.
  pixi run -e default python scripts/bench_infer_flat.py \
    --config "$RUN/config.yaml" --ckpt "$RUN/ckpts/$CK" \
    --data-dir /scratch/colliderml/ICLR_eval_v2/ttbar_new_pt1 \
    --batch-size "$BS" --iters 100 --gpu-seed --matmul-precision high \
    --encoder-dtype "$DT" --device cuda:0 > "$L" 2>&1 &
  BPID=$!
  for _ in $(seq 1 420); do
    grep -q "peak VRAM" "$L" 2>/dev/null && break
    kill -0 "$BPID" 2>/dev/null || break
    sleep 1
  done
  pkill -9 -P "$BPID" 2>/dev/null || true
  kill -9 "$BPID" 2>/dev/null || true
  wait "$BPID" 2>/dev/null || true
  for q in $(pgrep -f "[b]ench_infer_flat"); do kill -9 "$q" 2>/dev/null || true; done
  echo "$TAG bs=$BS -> $(grep -o 'throughput *: *[0-9,]*' "$L" | tail -1)"
done
