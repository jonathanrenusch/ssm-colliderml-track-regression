#!/usr/bin/env bash
# Overnight autonomous TXF launch sweep on MIG slices.
# Tries: cudaMallocAsync allocator (NVML-bypass) + default SDPA dispatch
# (mem_efficient is fp32-compatible and memory-efficient).
# Halves BS on OOM until BS=256.
#
# Stops on first BS where all 3 TXF runs reach a real training step.
set -u
cd /shared/tracking/ssm-colliderml-track-regression
LOGDIR=launch_logs

declare -a JOBS=(
  "MIG-1f292bf8-f63f-570f-a866-efaab3ceb50a:pretrain_transformer_2cls_lr1e-4"
  "MIG-82b2fca4-9873-5fce-b977-4bd92a37f2cb:pretrain_transformer_2cls_layerscale01"
  "MIG-9eee2fd9-6bad-585f-a6bc-ee346d274885:pretrain_transformer_2cls_tuned_lr1e-4"
)

SUCCESS_BS=""
for BS in 2048 1024 512 256; do
  STAMP="20260513_overnight_bs${BS}"
  echo "================================"
  echo "[$(date +%H:%M:%S)] Attempt: BS=$BS, stamp=$STAMP"
  echo "================================"
  declare -a PIDS=()
  for entry in "${JOBS[@]}"; do
    mig="${entry%%:*}"
    stem="${entry##*:}"
    log="$LOGDIR/${stem}_bs${BS}_${STAMP}.log"
    CUDA_VISIBLE_DEVICES="$mig" \
    PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync" \
    nohup bash scripts/01_train.sh "$stem" \
      --trainer.devices 1 \
      --data.batch_size "$BS" \
      > "$log" 2>&1 &
    PIDS+=("$!")
    sleep 0.5
  done
  echo "[$(date +%H:%M:%S)] Launched PIDs: ${PIDS[*]}"

  # Poll up to 6 min for verdict.
  DEADLINE=$(($(date +%s) + 360))
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    NVML=$(grep -l "NVML_SUCCESS == r" $LOGDIR/pretrain_transformer_*_bs${BS}_${STAMP}.log 2>/dev/null | wc -l)
    OOM=$(grep -l "OutOfMemoryError\|out of memory" $LOGDIR/pretrain_transformer_*_bs${BS}_${STAMP}.log 2>/dev/null | wc -l)
    TRAIN=$(grep -lE "Epoch 0:.*train/total=" $LOGDIR/pretrain_transformer_*_bs${BS}_${STAMP}.log 2>/dev/null | wc -l)
    DEAD=0
    for pid in "${PIDS[@]}"; do ps -p $pid > /dev/null 2>&1 || DEAD=$((DEAD+1)); done
    echo "[$(date +%H:%M:%S)]   BS=$BS train=$TRAIN/3 nvml=$NVML oom=$OOM dead=$DEAD/3"
    if [ "$TRAIN" -ge 3 ]; then
      SUCCESS_BS=$BS
      echo "[$(date +%H:%M:%S)] SUCCESS at BS=$BS — all 3 TXF training healthily."
      break 2
    fi
    if [ "$DEAD" -ge 3 ] || [ "$NVML" -ge 3 ] || [ "$OOM" -ge 3 ]; then
      echo "[$(date +%H:%M:%S)] BS=$BS FAILED — killing remnants and trying smaller BS."
      pkill -9 -f "python train.py fit.*pretrain_transformer_2cls" 2>/dev/null
      sleep 2
      break
    fi
    sleep 15
  done
done

if [ -n "$SUCCESS_BS" ]; then
  echo "================================"
  echo "OVERNIGHT_SWEEP_DONE: TXF training at BS=$SUCCESS_BS"
  echo "================================"
else
  echo "================================"
  echo "OVERNIGHT_SWEEP_EXHAUSTED: all BS values 2048→256 failed."
  echo "Likely needs a non-MIG GPU or PyTorch version downgrade to clear the NVML caching-allocator bug."
  echo "================================"
fi
