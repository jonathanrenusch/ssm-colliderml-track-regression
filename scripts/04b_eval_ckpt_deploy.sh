#!/bin/bash
# Evaluate one checkpoint on the ICLR eval stores at TRUE DEPLOYMENT SETTINGS
# (paper physics = deployment physics, user decision 2026-09-04):
#   - TF32 matmuls (TRK_MATMUL_PRECISION=high)
#   - fused v5pc kernels + TRK_SSD_BUCKET16=1 + TRK_COMPILE_FRONTEND=1
#   - seed + residual features + anchors computed ON THE GPU inside the model
#     forward (auto-seed path: the collate hands 12 raw features), in the
#     precision given by TRK_SEED_DTYPE (float64 default; float32 fails physics
#     at high pT -- the rc-R / rho-R cancellation, CLAUDE.md 2026-09-04).
#
#   bash scripts/04b_eval_ckpt_deploy.sh <run_dir> <ckpt_name> <out_dir> [eval_root] [gpu]
#
# Identical layout/outputs to 04_eval_ckpt_iclr.sh (preds/<ds>.h5 + plots/).
set -euo pipefail
RUN_DIR="$(realpath "$1")"; CKPT_NAME="$2"; OUT="$(realpath -m "$3")"; EVAL_ROOT="${4:-/scratch/colliderml/ICLR_eval_v2}"; GPU="${5:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT/ckpts" "$OUT/preds"
cp "$RUN_DIR/ckpts/$CKPT_NAME" "$OUT/ckpts/model.ckpt"
cp "$RUN_DIR/config.yaml" "$OUT/config.yaml"
cd "$REPO_ROOT/src/track_regression"
export TRK_MATMUL_PRECISION="${TRK_MATMUL_PRECISION:-high}" CUDA_VISIBLE_DEVICES="$GPU"
export TRK_SSD_BUCKET16="${TRK_SSD_BUCKET16:-1}" TRK_COMPILE_FRONTEND="${TRK_COMPILE_FRONTEND:-1}"
export TRK_SEED_DTYPE="${TRK_SEED_DTYPE:-float64}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_deploy_$GPU}"
echo "deployment eval: matmul=$TRK_MATMUL_PRECISION bucket16=$TRK_SSD_BUCKET16 compile_frontend=$TRK_COMPILE_FRONTEND seed_dtype=$TRK_SEED_DTYPE"
for ds in ${EVAL_DATASETS:-single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar ttbar_new_pt1 single_muon_uniform}; do
  [ -d "$EVAL_ROOT/$ds/test" ] || { echo "=== $ds: not in $EVAL_ROOT, skipped"; continue; }
  echo "=== $ds  $(date)"
  pixi run -e default python train.py test --config "$OUT/config.yaml" --ckpt_path "$OUT/ckpts/model.ckpt" \
     --trainer.devices 1 --trainer.logger false --data.preprocessed_dir "$EVAL_ROOT/$ds" \
     --data.seed_residual_features false \
     --data.batch_size 10000 --data.num_workers 0 2>&1 | grep -v "it/s\|Warning" | tail -8
  mv "$OUT/model__test_predictions.h5" "$OUT/preds/$ds.h5"
done
pixi run -e default python scripts/fast_rms_eval.py --pred-dir "$OUT/preds" --store-root "$EVAL_ROOT" \
   --out-dir "$OUT/plots" --subtitle "$(basename "$RUN_DIR") $CKPT_NAME [deploy: TF32+switches, GPU seed $TRK_SEED_DTYPE]"
cat "$OUT/plots/rms_summary.txt"
