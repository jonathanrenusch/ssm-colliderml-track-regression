#!/bin/bash
# SWEEP 7 launch, sess3 (2026-09-01): R2L-FT on GPUs 1+2 (DDP), R2L-noconv on GPU 3;
# R2L-d96 is queued on GPU 0 behind YZ-mix3 + its auto-eval (queue_d96_gpu0.sh).
# R1L launches on sess5 after the kernel work (user allocation); fourier8 on hold.
set -u
cd /shared/tracking/ssm-colliderml-track-regression/src/track_regression
LOGS=../../launch_logs/sweep7
mkdir -p "$LOGS"
TS=$(date +%Y%m%d_%H%M)

CKPT_2L=/shared/tracking/ssm-colliderml-track-regression/src/track_regression/logs/comet_offline/ebf7104a350b4fdcb6521d8acb0069a6/ckpts/last.ckpt

# R2L-FT: Muon-hybrid + WSD fine-tune of YZ-2L-mix3, DDP 2 GPUs, bs 2x20k, 50 ep
TRK_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=1,2 nohup pixi run -e default python train.py fit \
  --config config/ssm_cls/ICLR_sweep7/R2LFT_qrel_2L_mix3_muonhybrid_ddp2_bs40k_wsd50.yaml \
  --trainer.devices 2 --model.pretrained_ckpt_path "$CKPT_2L" \
  > "$LOGS/R2LFT_${TS}.log" 2>&1 &
echo "R2LFT GPUs 1+2 pid $!"

# R2L-noconv: d_conv 1, 25 ep bs 2048 on mix3, GPU 3
TRK_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=3 nohup pixi run -e default python train.py fit \
  --config config/ssm_cls/ICLR_sweep7/R2Lnoconv_qrel_2L_dconv1_mix3_bs2048_onecycle25.yaml \
  --trainer.devices 1 \
  > "$LOGS/R2Lnoconv_${TS}.log" 2>&1 &
echo "R2Lnoconv GPU 3 pid $!"
