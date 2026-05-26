#!/usr/bin/env bash
# Relaunch ONLY the 3 TXF configs with PYTORCH_CUDA_ALLOC_CONF env var
# to bypass the NVML/MIG bug in the default caching allocator. The 6 SSM
# runs are training fine and are NOT touched.
set -u
cd /shared/tracking/ssm-colliderml-track-regression
STAMP="20260513_032000"
LOGDIR=launch_logs

declare -a JOBS=(
  "MIG-1f292bf8-f63f-570f-a866-efaab3ceb50a:pretrain_transformer_2cls_lr1e-4"
  "MIG-82b2fca4-9873-5fce-b977-4bd92a37f2cb:pretrain_transformer_2cls_layerscale01"
  "MIG-9eee2fd9-6bad-585f-a6bc-ee346d274885:pretrain_transformer_2cls_tuned_lr1e-4"
)

> "$LOGDIR/relaunch_txf_${STAMP}_pids.tsv"
echo -e "pid\tmig\tconfig\tlog" >> "$LOGDIR/relaunch_txf_${STAMP}_pids.tsv"

for entry in "${JOBS[@]}"; do
  mig="${entry%%:*}"
  stem="${entry##*:}"
  log="$LOGDIR/${stem}_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$mig" \
  nohup bash scripts/01_train.sh "$stem" \
    --trainer.devices 1 \
    > "$log" 2>&1 &
  pid=$!
  echo -e "${pid}\t${mig}\t${stem}\t${log}" | tee -a "$LOGDIR/relaunch_txf_${STAMP}_pids.tsv"
  sleep 0.5
done

echo "TXF relaunched. PID table at $LOGDIR/relaunch_txf_${STAMP}_pids.tsv"
