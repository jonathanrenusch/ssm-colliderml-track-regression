#!/bin/bash
# SWEEP 4 overnight launcher (sess3): W (GPU 0), W-noF (1), W-qrel (2) as stage1 -> chained stage2; Z pure bs 2048 (GPU 3).
#   bash launch_sess3.sh            (DRY=1: 60-step dry runs of every stage-1 config + Z, then a stage-2 dry run from W1's dry checkpoint)
set -uo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression; CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep4
[ -f /scratch/colliderml/ICLR_retraining_v2_mixed/train/manifest.json ] || { echo "v2 mixed store missing"; exit 1; }
mkdir -p $REPO/launch_logs/sweep4/dryrun; cd $REPO/src/track_regression; export TRK_MATMUL_PRECISION=highest
dry () {  # dry <stem> <gpu> [extra]
  local t=$1 g=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=/tmp/triton_cache_$t pixi run -e default python train.py fit --config $CFG/$t.yaml --trainer.devices 1 \
     --trainer.limit_train_batches 60 --trainer.limit_val_batches 2 --trainer.max_epochs 1 --trainer.logger.init_args.name dryrun-SW4-$t "$@" > $REPO/launch_logs/sweep4/dryrun/$t.log 2>&1
  echo "dry $t (GPU $g): rc=$? $(grep -c 'val epoch' $REPO/launch_logs/sweep4/dryrun/$t.log) val, errors: $(grep -ci 'Traceback\|Error' $REPO/launch_logs/sweep4/dryrun/$t.log)"
}
if [ "${DRY:-0}" = "1" ]; then
  dry W1_ref_bs36k_const15 0 & dry X1_noFourier_bs36k_const15 1 & dry Y1_qrel_bs36k_const15 2 & dry Z_ref_bs2048_onecycle25 3 & wait
  CK=$(grep -m1 -o "Saving checkpoints to: .*" $REPO/launch_logs/sweep4/dryrun/W1_ref_bs36k_const15.log | sed 's/Saving checkpoints to: //')
  dry W2_ref_bs2048_cos10 0 --model.pretrained_ckpt_path "$(realpath $CK/last.ckpt)"
  exit 0
fi
launch () {  # launch <stem> <gpu> -> LAUNCHED_PID / LAUNCHED_LOG
  local t=$1 gpu=$2
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $gpu 2>/dev/null | grep -q .; then echo "GPU $gpu busy — NOT launching $t"; return 1; fi
  LAUNCHED_LOG=$REPO/launch_logs/sweep4/${t}_$(date +%Y%m%d_%H%M%S).log
  CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_$t nohup pixi run -e default python train.py fit --config $CFG/$t.yaml --trainer.devices 1 > "$LAUNCHED_LOG" 2>&1 &
  LAUNCHED_PID=$!; echo "launched $t on GPU $gpu (pid $LAUNCHED_PID) -> $LAUNCHED_LOG"; sleep 20
}
launch W1_ref_bs36k_const15 0 && nohup bash $CFG/chain_stage2.sh $LAUNCHED_PID $LAUNCHED_LOG 0 W2_ref_bs2048_cos10 > $REPO/launch_logs/sweep4/chain_W.log 2>&1 &
launch X1_noFourier_bs36k_const15 1 && nohup bash $CFG/chain_stage2.sh $LAUNCHED_PID $LAUNCHED_LOG 1 X2_noFourier_bs2048_cos10 > $REPO/launch_logs/sweep4/chain_X.log 2>&1 &
launch Y1_qrel_bs36k_const15 2 && nohup bash $CFG/chain_stage2.sh $LAUNCHED_PID $LAUNCHED_LOG 2 Y2_qrel_bs2048_cos10 > $REPO/launch_logs/sweep4/chain_Y.log 2>&1 &
launch Z_ref_bs2048_onecycle25 3
echo "chains for W/X/Y waiting; all launched $(date)"
