#!/bin/bash
# Launch one sweep-5 run on sess5 (one process, one GPU).   bash launch_sess5.sh <config_stem> <gpu> [--model.pretrained_ckpt_path ...]
# DRY=1 for a 60-step dry run.  50-epoch configs get periodic checkpoints every 10 epochs appended.
set -uo pipefail
T="$1"; GPU="$2"; shift 2; REPO=/shared/tracking/ssm-colliderml-track-regression; CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep5
STORE=$(grep -m1 "preprocessed_dir:" $CFG/$T.yaml | awk '{print $2}')
[ -f "$STORE/train/manifest.json" ] || { echo "store $STORE missing — run scripts/10_stage_v2_on_sess5.sh first"; exit 1; }
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $GPU 2>/dev/null | grep -q .; then echo "GPU $GPU busy"; exit 1; fi
mkdir -p $REPO/launch_logs/sweep5; cd $REPO/src/track_regression; export TRK_MATMUL_PRECISION=highest
EXTRA=""; [ "${DRY:-0}" = "1" ] && EXTRA="--trainer.limit_train_batches 60 --trainer.limit_val_batches 2 --trainer.max_epochs 1 --trainer.logger.init_args.name dryrun-SW5-$T"
PERIODIC=""; grep -q "max_epochs: 50" $CFG/$T.yaml && PERIODIC="--trainer.callbacks+=track_regression._lib.callbacks.Checkpoint --trainer.callbacks.init_args.monitor=val/total --trainer.callbacks.init_args.save_top_k=-1 --trainer.callbacks.init_args.every_n_epochs=10 --trainer.callbacks.init_args.save_on_train_epoch_end=false"
LOG=$REPO/launch_logs/sweep5/${T}_$(date +%Y%m%d_%H%M%S).log
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_$T nohup pixi run -e default python train.py fit --config $CFG/$T.yaml --trainer.devices 1 $PERIODIC $EXTRA "$@" > "$LOG" 2>&1 &
echo "launched $T on GPU $GPU (pid $!) -> $LOG"
