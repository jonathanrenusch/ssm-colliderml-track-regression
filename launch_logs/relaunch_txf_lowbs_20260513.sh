#!/usr/bin/env bash
# TXF retry: cudaMallocAsync (NVML-bypass) + BS=1024 (fits in MIG).
# All 3 experimental TXF configs run with the same BS so the ablation
# stays internally consistent.
set -u
cd /shared/tracking/ssm-colliderml-track-regression
STAMP="${1:-20260513_032500}"
BS="${2:-1024}"
LOGDIR=launch_logs

declare -a JOBS=(
  "MIG-1f292bf8-f63f-570f-a866-efaab3ceb50a:pretrain_transformer_2cls_lr1e-4"
  "MIG-82b2fca4-9873-5fce-b977-4bd92a37f2cb:pretrain_transformer_2cls_layerscale01"
  "MIG-9eee2fd9-6bad-585f-a6bc-ee346d274885:pretrain_transformer_2cls_tuned_lr1e-4"
)

> "$LOGDIR/relaunch_txf_lowbs_${STAMP}_pids.tsv"
echo -e "pid\tmig\tconfig\tbs\tlog" >> "$LOGDIR/relaunch_txf_lowbs_${STAMP}_pids.tsv"

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
  pid=$!
  echo -e "${pid}\t${mig}\t${stem}\t${BS}\t${log}" | tee -a "$LOGDIR/relaunch_txf_lowbs_${STAMP}_pids.tsv"
  sleep 0.5
done

echo "TXF relaunched at BS=$BS. PID table at $LOGDIR/relaunch_txf_lowbs_${STAMP}_pids.tsv"
