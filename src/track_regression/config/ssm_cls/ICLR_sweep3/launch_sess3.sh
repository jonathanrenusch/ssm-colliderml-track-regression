#!/bin/bash
# Launch sweep-3 runs H', M', R1 (+ chained R2) on GPUs 0, 1, 3 of sess3, one process per GPU, via nohup.
# GPU 2 is reserved for P' (launched separately once implemented).
set -uo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression
CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep3
STORE=/scratch/colliderml/ICLR_retraining_geom/single_muon_uniform
EOS=/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_geom/single_muon_uniform
if [ ! -f "$STORE/train/manifest.json" ]; then
  echo "store missing on /scratch — copying from /eos (145 GB, ~3-4 min)"; bash $REPO/scripts/copy_dataset.sh "$EOS" "$STORE" 16 || exit 1
fi
mkdir -p $REPO/launch_logs/sweep3
cd $REPO/src/track_regression
export TRK_MATMUL_PRECISION=highest
launch () {  # launch <trial> <gpu>  -> prints the log path; pid in $LAUNCHED_PID
  local t=$1 gpu=$2
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $gpu 2>/dev/null | grep -q .; then
    echo "GPU $gpu is busy (nvidia-smi shows a compute process) — NOT launching $t"; return 1
  fi
  LAUNCHED_LOG=$REPO/launch_logs/sweep3/${t}_$(date +%Y%m%d_%H%M%S).log
  CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_$t nohup pixi run -e default python train.py fit \
      --config $CFG/$t.yaml --trainer.devices 1 > "$LAUNCHED_LOG" 2>&1 &
  LAUNCHED_PID=$!
  echo "launched $t on GPU $gpu (pid $LAUNCHED_PID) -> $LAUNCHED_LOG"; sleep 20
}
launch Hp_4L_ds64_lion_cosine_bs36k_anchor_fourier10_absqop    0
launch Mp_4L_ds64_lion_cosine_bs2048_anchor_fourier10_absqop   1
launch R1_4L_ds64_lion_wsdconst_bs36k_anchor_fourier10_absqop  3 && \
  nohup bash $CFG/chain_R.sh "$LAUNCHED_PID" "$LAUNCHED_LOG" 3 > $REPO/launch_logs/sweep3/chain_R_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "chain_R waiting on R1 (pid $LAUNCHED_PID)"
