#!/bin/bash
# Wait for R1 (pid) to finish, then launch R2 from R1's last.ckpt on the same GPU.
#   bash chain_R.sh <R1_pid> <R1_log> <gpu>
set -uo pipefail
PID=$1; LOG=$2; GPU=$3
REPO=/shared/tracking/ssm-colliderml-track-regression
CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep3
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "R1 (pid $PID) finished $(date)"
CK=$(grep -m1 -o "Saving checkpoints to: .*" "$LOG" | sed 's/Saving checkpoints to: //' | tr -d '\r')
cd $REPO/src/track_regression
[ -f "$CK/last.ckpt" ] || { echo "ERROR: $CK/last.ckpt not found — R2 not launched"; exit 1; }
if ! grep -q "Epoch 14: 100%" "$LOG"; then echo "WARNING: R1 log does not show epoch 14 complete — launching R2 anyway from $CK/last.ckpt"; fi
export TRK_MATMUL_PRECISION=highest
R2LOG=$REPO/launch_logs/sweep3/R2_4L_ds64_lion_cosine_bs2048_anchor_fourier10_absqop_from_R1_$(date +%Y%m%d_%H%M%S).log
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_R2 nohup pixi run -e default python train.py fit \
    --config $CFG/R2_4L_ds64_lion_cosine_bs2048_anchor_fourier10_absqop_from_R1.yaml --trainer.devices 1 \
    --model.pretrained_ckpt_path "$(realpath "$CK/last.ckpt")" > "$R2LOG" 2>&1 &
echo "launched R2 on GPU $GPU (pid $!) from $CK/last.ckpt -> $R2LOG"
