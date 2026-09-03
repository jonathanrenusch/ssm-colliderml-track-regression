#!/bin/bash
# Wait for a stage-1 training (pid) to finish, then launch the stage-2 config from its last.ckpt on the same GPU.
#   bash chain_stage2.sh <stage1_pid> <stage1_log> <gpu> <stage2_config_stem>
set -uo pipefail
PID=$1; LOG=$2; GPU=$3; T2=$4
REPO=/shared/tracking/ssm-colliderml-track-regression; CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep5
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "stage 1 (pid $PID) finished $(date)"
CK=$(grep -m1 -o "Saving checkpoints to: .*" "$LOG" | sed 's/Saving checkpoints to: //' | tr -d '\r')
cd $REPO/src/track_regression
[ -f "$CK/last.ckpt" ] || { echo "ERROR: $CK/last.ckpt not found — stage 2 not launched"; exit 1; }
export TRK_MATMUL_PRECISION=highest
L2=$REPO/launch_logs/sweep5/${T2}_$(date +%Y%m%d_%H%M%S).log
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_$T2 nohup pixi run -e default python train.py fit --config $CFG/$T2.yaml --trainer.devices 1 \
    --model.pretrained_ckpt_path "$(realpath "$CK/last.ckpt")" > "$L2" 2>&1 &
echo "launched $T2 on GPU $GPU (pid $!) from $CK/last.ckpt -> $L2"
