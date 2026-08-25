#!/bin/bash
# Evaluate one checkpoint on the five ICLR eval stores and produce the
# bootstrap-free RMS-vs-eta summary (fast_rms_eval).
#
#   bash scripts/04_eval_ckpt_iclr.sh <run_dir> <ckpt_name> <out_dir> [eval_root] [gpu]
#
#   run_dir    a comet_offline run directory (holds config.yaml and ckpts/)
#   ckpt_name  file name inside run_dir/ckpts/, e.g. 'epoch=012-val_total=0.03309.ckpt'
#   out_dir    where predictions (preds/<dataset>.h5) and plots/ go
#   eval_root  default /scratch/colliderml/ICLR_eval_ssort (s-sorted stores);
#              pass /scratch/colliderml/ICLR_eval for the deprecated time-sorted ones
#   gpu        CUDA device index, default 0
#
# The checkpoint and config are COPIED into out_dir first because
# RegressionPredictionWriter writes '<ckpt-stem>__test_predictions.h5' next to
# the checkpoint it is given — evaluating in place would litter a live run
# directory and overwrite itself between datasets.
set -euo pipefail
RUN_DIR="$1"; CKPT_NAME="$2"; OUT="$3"; EVAL_ROOT="${4:-/scratch/colliderml/ICLR_eval_ssort}"; GPU="${5:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT/ckpts" "$OUT/preds"
cp "$RUN_DIR/ckpts/$CKPT_NAME" "$OUT/ckpts/model.ckpt"
cp "$RUN_DIR/config.yaml" "$OUT/config.yaml"
cd "$REPO_ROOT/src/track_regression"
export TRK_MATMUL_PRECISION="${TRK_MATMUL_PRECISION:-highest}" CUDA_VISIBLE_DEVICES="$GPU"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_eval_$$}"
for ds in single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar single_muon_uniform; do
  echo "=== $ds  $(date)"
  pixi run -e default python train.py test --config "$OUT/config.yaml" --ckpt_path "$OUT/ckpts/model.ckpt" \
     --trainer.devices 1 --trainer.logger false --data.preprocessed_dir "$EVAL_ROOT/$ds" \
     --data.batch_size 10000 --data.num_workers 0 2>&1 | grep -v "it/s\|Warning" | tail -8
  mv "$OUT/model__test_predictions.h5" "$OUT/preds/$ds.h5"
done
pixi run -e default python scripts/fast_rms_eval.py --pred-dir "$OUT/preds" --store-root "$EVAL_ROOT" \
   --out-dir "$OUT/plots" --subtitle "$(basename "$RUN_DIR") $CKPT_NAME"
cat "$OUT/plots/rms_summary.txt"
