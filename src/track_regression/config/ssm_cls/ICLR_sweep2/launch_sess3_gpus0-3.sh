#!/bin/bash
# Launch sweep-2 trials G, H, I, J on GPUs 0-3 of THIS machine (sess3), one process per GPU, via nohup.
set -uo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression
CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep2
STORE=/scratch/colliderml/ICLR_retraining_geom/single_muon_uniform
EOS=/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_geom/single_muon_uniform
if [ ! -f "$STORE/train/manifest.json" ]; then
  echo "store missing on /scratch — copying from /eos (145 GB, ~3-4 min)"; bash $REPO/scripts/copy_dataset.sh "$EOS" "$STORE" 16 || exit 1
fi
mkdir -p $REPO/launch_logs/sweep2
cd $REPO/src/track_regression
export TRK_MATMUL_PRECISION=highest
launch () {  # launch <trial> <gpu>
  local t=$1 gpu=$2
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $gpu 2>/dev/null | grep -q .; then
    echo "GPU $gpu is busy (nvidia-smi shows a compute process) — NOT launching $t"; return 1
  fi
  local log=$REPO/launch_logs/sweep2/${t}_$(date +%Y%m%d_%H%M%S).log
  # One process = one GPU: CUDA_VISIBLE_DEVICES hides the other cards entirely, and
  # --trainer.devices 1 overrides the documentary `devices: [k]` in the yaml.
  CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_$t nohup pixi run -e default python train.py fit \
      --config $CFG/$t.yaml --trainer.devices 1 > "$log" 2>&1 &
  echo "launched $t on GPU $gpu (pid $!) -> $log"; sleep 20
}
launch G_4L_ds64_lion_cosine_bs36k_seedanchor            0
launch H_4L_ds64_lion_cosine_bs36k_seedanchor_fourier10  1
launch I_4L_ds64_lion_cosine_bs18k_fp64                  2
launch J_4L_ds64_lion_cosine_bs36k_fourier10             3
