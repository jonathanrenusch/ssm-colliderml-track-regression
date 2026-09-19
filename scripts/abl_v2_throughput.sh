#!/bin/bash
# Deployed throughput for one ablation arm, using the campaign's canonical tool
# so the numbers are comparable with the paper's.
#
#   scripts/abl_v2_throughput.sh <tag> <run_dir> [gpu] [batch]
#
# Deployment mode = GPU seed in-forward + TF32 matmuls + the kernel switches,
# exactly as scripts/04b_eval_ckpt_deploy.sh uses for physics.  Run it on an
# IDLE GPU: contention makes these numbers meaningless.
set -eu
TAG="$1"; RUN_DIR="$2"; G="${3:-0}"; BS="${4:-131000}"
REPO=/shared/tracking/ssm-colliderml-track-regression
OUT=$REPO/eval_plots/ablations_2026-09/v2_throughput
mkdir -p "$OUT"
BUSY=$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$BUSY" -ne 0 ]; then
  echo "REFUSING: GPU $G has $BUSY compute process(es); throughput under contention is noise." >&2
  exit 2
fi
export TRK_SSD_BUCKET16=1 TRK_COMPILE_FRONTEND=1 TRK_SEED_DTYPE=float64
export TRITON_CACHE_DIR=/tmp/tc_bench_gpu${G}
cd "$REPO"
pixi run -e default python scripts/bench_infer_flat.py \
  --config "$RUN_DIR/config.yaml" --ckpt "$RUN_DIR/ckpts/last.ckpt" \
  --data-dir /scratch/colliderml/ICLR_eval_v2_new/ttbar_new_pt1 \
  --batch-size "$BS" --iters 100 --gpu-seed --matmul-precision high \
  --device "cuda:$G" 2>&1 | tee "$OUT/${TAG}_bs${BS}.log"
echo "-> $OUT/${TAG}_bs${BS}.log"
