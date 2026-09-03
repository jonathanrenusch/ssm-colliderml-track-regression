#!/bin/bash
# SWEEP 5 launcher (sess3): YZ (GPU 0), Y-FT from Y2 (GPU 1), Y-3L stage1->stage2 (GPU 2), Y-d96 stage1->stage2 on GPU 3 after Z finishes.
#   DRY=1 bash launch_sess3.sh   -> 60-step dry runs of YZ, YFT, Y3L1, Yd961 on GPUs 0-2 (Yd961 after YZ) ; bash launch_sess3.sh -> launch
set -uo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression; CFG=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep5; CFG4=$REPO/src/track_regression/config/ssm_cls/ICLR_sweep4
Y1CK=$(realpath $REPO/src/track_regression/logs/comet_offline/05c3390f96124f71b88777a210adc560/ckpts/last.ckpt)   # Y1 stage-1 end (unused now; kept for Y-L)
Y2CK=$(realpath $(ls -d $REPO/src/track_regression/logs/comet_offline/d99c690b*)/ckpts/last.ckpt)                 # Y2 final (scale-free head)
mkdir -p $REPO/launch_logs/sweep5/dryrun; cd $REPO/src/track_regression; export TRK_MATMUL_PRECISION=highest
dry () { local t=$1 g=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=/tmp/triton_cache_$t pixi run -e default python train.py fit --config $CFG/$t.yaml --trainer.devices 1 \
     --trainer.limit_train_batches 60 --trainer.limit_val_batches 2 --trainer.max_epochs 1 --trainer.logger.init_args.name dryrun-SW5-$t "$@" > $REPO/launch_logs/sweep5/dryrun/$t.log 2>&1
  echo "dry $t (GPU $g): rc=$? $(grep -c 'val epoch' $REPO/launch_logs/sweep5/dryrun/$t.log) val, tracebacks: $(grep -c Traceback $REPO/launch_logs/sweep5/dryrun/$t.log)"; }
if [ "${DRY:-0}" = "1" ]; then
  dry YZ_qrel_bs2048_onecycle25 0 & dry YFT_qrel_muonhybrid_bs36k_wsd12_fromY2 1 --model.pretrained_ckpt_path "$Y2CK" & dry Y3L1_qrel_3L_bs36k_const15 2 & wait
  dry Yd961_qrel_d96_bs36k_const15 0; exit 0
fi
launch () { local t=$1 gpu=$2; shift 2
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader -i $gpu 2>/dev/null | grep -q .; then echo "GPU $gpu busy — NOT launching $t"; return 1; fi
  LAUNCHED_LOG=$REPO/launch_logs/sweep5/${t}_$(date +%Y%m%d_%H%M%S).log
  CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_$t nohup pixi run -e default python train.py fit --config $CFG/$t.yaml --trainer.devices 1 "$@" > "$LAUNCHED_LOG" 2>&1 &
  LAUNCHED_PID=$!; echo "launched $t on GPU $gpu (pid $LAUNCHED_PID) -> $LAUNCHED_LOG"; sleep 20; }
launch YZ_qrel_bs2048_onecycle25 0
launch YFT_qrel_muonhybrid_bs36k_wsd12_fromY2 1 --model.pretrained_ckpt_path "$Y2CK"
launch Y3L1_qrel_3L_bs36k_const15 2 && nohup bash $CFG/chain_stage2.sh $LAUNCHED_PID $LAUNCHED_LOG 2 Y3L2_qrel_3L_bs2048_cos10 > $REPO/launch_logs/sweep5/chain_Y3L.log 2>&1 &
# GPU 3: wait for Z, then Y-d96 stage 1 -> stage 2
ZPID=$(pgrep -f "ICLR_sweep4/Z_ref_bs2048_onecycle25.*--trainer.devices" | head -1)
nohup bash -c "while kill -0 $ZPID 2>/dev/null; do sleep 60; done; sleep 30; cd $REPO/src/track_regression; L=$REPO/launch_logs/sweep5/Yd961_qrel_d96_bs36k_const15_\$(date +%Y%m%d_%H%M%S).log; CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton_cache_Yd961 TRK_MATMUL_PRECISION=highest nohup pixi run -e default python train.py fit --config $CFG/Yd961_qrel_d96_bs36k_const15.yaml --trainer.devices 1 > \$L 2>&1 & P=\$!; echo \"launched Yd961 on GPU 3 (pid \$P) -> \$L\"; bash $CFG/chain_stage2.sh \$P \$L 3 Yd962_qrel_d96_bs2048_cos10" > $REPO/launch_logs/sweep5/chain_Yd96_afterZ.log 2>&1 &
echo "Y-d96 queued behind Z (pid $ZPID); all launched $(date)"
