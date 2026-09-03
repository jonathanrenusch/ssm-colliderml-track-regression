#!/bin/bash
# Queue R2L-d96 on GPU 0: wait until YZ-mix3 training (pid 242667) is gone AND its
# auto-eval has written eval_plots/round6/YZmix3/plots/rms_summary.txt, then launch.
set -u
cd /shared/tracking/ssm-colliderml-track-regression/src/track_regression
LOGS=../../launch_logs/sweep7
mkdir -p "$LOGS"

while kill -0 242667 2>/dev/null; do sleep 300; done
# wait for the auto-eval (armed earlier) to finish; give up waiting after 3 h and launch anyway
SUMMARY=/shared/tracking/ssm-colliderml-track-regression/eval_plots/round6/YZmix3/plots/rms_summary.txt
for _ in $(seq 1 36); do [ -f "$SUMMARY" ] && break; sleep 300; done
sleep 120

TS=$(date +%Y%m%d_%H%M)
TRK_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=0 nohup pixi run -e default python train.py fit \
  --config config/ssm_cls/ICLR_sweep7/R2Ld96_qrel_2L_d96_mix3_bs2048_onecycle25.yaml \
  --trainer.devices 1 \
  > "$LOGS/R2Ld96_${TS}.log" 2>&1 &
echo "R2Ld96 GPU 0 pid $! ($(date))" >> "$LOGS/queue_d96.log"
