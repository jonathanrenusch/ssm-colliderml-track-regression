#!/bin/bash
# Launch P' on GPU 2 of sess3 (after H'/M'/R1 took 0/1/3).
set -uo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression; CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep3
T=Pp_4L_ds64_lion_cosine_bs36k_anchor_fourier10_absqop_seedres; GPU=${1:-2}
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $GPU 2>/dev/null | grep -q .; then echo "GPU $GPU busy"; exit 1; fi
cd $REPO/src/track_regression; export TRK_MATMUL_PRECISION=highest; mkdir -p $REPO/launch_logs/sweep3
LOG=$REPO/launch_logs/sweep3/${T}_$(date +%Y%m%d_%H%M%S).log
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_$T nohup pixi run -e default python train.py fit --config $CFG/$T.yaml --trainer.devices 1 > "$LOG" 2>&1 &
echo "launched $T on GPU $GPU (pid $!) -> $LOG"
