#!/bin/bash
# Launch one sweep-3 run on sess5 (one process, one GPU) via nohup.
#   bash launch_sess5.sh <config_stem> [gpu=0]        (DRY=1 for a 60-step dry run)
#   e.g. bash launch_sess5.sh Qp_4L_ds64_lion_cosine_bs2048_anchor_fourier10_absqop_B3mixed16M_50ep 0
#        bash launch_sess5.sh Qm_4L_ds64_lion_cosine_bs2048_anchor_fourier10_absqop_B3muon_50ep 1
# Prerequisite: scripts/10_stage_B3_on_sess5.sh has run (B3 stores + eval farm + mixed store on /scratch).
set -uo pipefail
T="$1"; GPU="${2:-0}"; REPO=/shared/tracking/ssm-colliderml-track-regression
CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep3
STORE=$(grep -m1 "preprocessed_dir:" $CFG/$T.yaml | awk '{print $2}')
[ -f "$STORE/train/manifest.json" ] || { echo "store $STORE missing on /scratch — run scripts/10_stage_B3_on_sess5.sh"; exit 1; }
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $GPU 2>/dev/null | grep -q .; then echo "GPU $GPU busy — not launching"; exit 1; fi
mkdir -p $REPO/launch_logs/sweep3; cd $REPO/src/track_regression; export TRK_MATMUL_PRECISION=highest
EXTRA=""
if [ "${DRY:-0}" = "1" ]; then EXTRA="--trainer.limit_train_batches 60 --trainer.limit_val_batches 2 --trainer.max_epochs 1 --trainer.logger.init_args.name dryrun-SW3-$T"; fi
PERIODIC=""
if grep -q "max_epochs: 50" $CFG/$T.yaml; then
  PERIODIC="--trainer.callbacks+=track_regression._lib.callbacks.Checkpoint --trainer.callbacks.init_args.monitor=val/total --trainer.callbacks.init_args.save_top_k=-1 --trainer.callbacks.init_args.every_n_epochs=10 --trainer.callbacks.init_args.save_on_train_epoch_end=false"
fi
LOG=$REPO/launch_logs/sweep3/${T}_$(date +%Y%m%d_%H%M%S).log
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_$T nohup pixi run -e default python train.py fit \
   --config $CFG/$T.yaml --trainer.devices 1 $PERIODIC $EXTRA > "$LOG" 2>&1 &
echo "launched $T on GPU $GPU (pid $!) -> $LOG"
