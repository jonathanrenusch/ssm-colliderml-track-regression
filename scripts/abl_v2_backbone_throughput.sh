#!/bin/bash
# Backbone throughput comparison on ONE idle GPU, two configurations:
#
#   ref   -- every encoder in its reference PyTorch implementation: no custom
#            Triton kernel (Mamba-2 variant v3c = compiled pure-torch dual,
#            minGRU/diagonal SSM with TRK_MINGRU_KERNEL=off, transformer on
#            stock SDPA), and the inference kernel switches OFF.  This is the
#            "what you get without kernel work" column.
#   opt   -- this campaign's fused path: v5pc for Mamba-2, the packed Triton
#            kernel for minGRU, plus TRK_SSD_BUCKET16 / TRK_COMPILE_FRONTEND.
#            The transformer and the diagonal SSM have no fused kernel, so
#            their 'opt' row only picks up the front-end compile.
#
# Everything else is identical across arms: TF32 matmuls, GPU seed (fp64) in
# the timed loop, same input sample, same batch size, 100 timed iterations.
#
#   bash scripts/abl_v2_backbone_throughput.sh <gpu> [batch] [dtype]
set -eu
G="${1:-3}"; BS="${2:-131000}"; DT="${3:-float32}"
REPO=/shared/tracking/ssm-colliderml-track-regression
EVALS=$REPO/eval_plots/ablations_2026-09/v2_evals
OUT=$REPO/eval_plots/ablations_2026-09/v2_throughput
mkdir -p "$OUT"
BUSY=$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$BUSY" -eq 0 ] || { echo "REFUSING: GPU $G busy ($BUSY procs)" >&2; exit 2; }
LOG="$OUT/backbones_bs${BS}_${DT}.log"
cd "$REPO"
export TRK_SEED_DTYPE=float64 TRITON_CACHE_DIR=/tmp/tc_bb_gpu${G}
# Pin with CUDA_VISIBLE_DEVICES, not --device cuda:N: Triton launches on
# the current device and bench_infer_flat reads the peak-VRAM stat there.
export CUDA_VISIBLE_DEVICES="$G"
# arm : run-dir : mamba-variant-for-ref
for spec in \
  "mamba2_bidir:SSM_baseline_25ep" \
  "transformer:V2_txf_25ep" \
  "mingru:V2_mingru_25ep" \
  "diagssm_nonsel:V2_diagssm_25ep" ; do
  tag="${spec%%:*}"; run="${spec##*:}"
  for mode in ref opt; do
    if [ "$mode" = ref ]; then
      EXTRA=(--no-kernel-switches --variant v3c); export TRK_MINGRU_KERNEL=off
    else
      EXTRA=(--variant v5pc); export TRK_MINGRU_KERNEL=auto
      export TRK_SSD_BUCKET16=1 TRK_COMPILE_FRONTEND=1
    fi
    echo "### $tag mode=$mode bs=$BS dtype=$DT $(date -Iseconds)" | tee -a "$LOG"
    # Detach, wait for the report, kill: bench_infer_flat hangs on exit.
    T=$(mktemp)
    pixi run -e default python scripts/bench_infer_flat.py \
      --config "$EVALS/$run/config.yaml" --ckpt "$EVALS/$run/ckpts/model.ckpt" \
      --data-dir /scratch/colliderml/ICLR_eval_v2/ttbar_new_pt1 \
      --batch-size "$BS" --iters 100 --gpu-seed --matmul-precision high \
      --encoder-dtype "$DT" --device cuda:0 "${EXTRA[@]}" > "$T" 2>&1 &
    BPID=$!
    for _ in $(seq 1 420); do
      grep -q "peak VRAM" "$T" 2>/dev/null && break
      kill -0 "$BPID" 2>/dev/null || break
      sleep 1
    done
    pkill -9 -P "$BPID" 2>/dev/null || true; kill -9 "$BPID" 2>/dev/null || true
    wait "$BPID" 2>/dev/null || true
    for q in $(pgrep -f "[b]ench_infer_flat"); do kill -9 "$q" 2>/dev/null || true; done
    grep -E "throughput|peak VRAM|Error" "$T" | tee -a "$LOG" || echo "  FAILED" | tee -a "$LOG"
    rm -f "$T"
  done
done
echo "-> $LOG"
