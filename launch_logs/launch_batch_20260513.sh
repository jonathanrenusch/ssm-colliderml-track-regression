#!/usr/bin/env bash
# Launch the 9-config batch on 9 MIG slices. Spare MIG-57d25c0d held back.
# Generated 2026-05-13.
set -u
cd /shared/tracking/ssm-colliderml-track-regression
STAMP="20260513_030000"
LOGDIR=launch_logs

# (mig_uuid, config_stem)
declare -a JOBS=(
  "MIG-724d3d25-7d76-5c20-96f6-bda84f095eb0:pretrain_ssm_cls_2L"
  "MIG-1d238dba-6ae3-5ec5-8ba6-e4d5d24c6241:pretrain_ssm_cls_4L"
  "MIG-c99731c9-f577-50c8-b420-9578c39c028d:pretrain_ssm_cls_6L"
  "MIG-17e27f7b-07f4-589c-b832-4a5ffb89c9fc:pretrain_ssm_cls_8L"
  "MIG-f46813af-2529-577c-b8fa-5f9c86a65fba:pretrain_ssm_cls_10L"
  "MIG-1f292bf8-f63f-570f-a866-efaab3ceb50a:pretrain_transformer_2cls_lr1e-4"
  "MIG-82b2fca4-9873-5fce-b977-4bd92a37f2cb:pretrain_transformer_2cls_layerscale01"
  "MIG-9eee2fd9-6bad-585f-a6bc-ee346d274885:pretrain_transformer_2cls_tuned_lr1e-4"
  "MIG-a3395bd2-8015-5e3b-b476-a4128816e08a:pretrain_ssm_cls_padded"
)

> "$LOGDIR/launch_batch_${STAMP}_pids.tsv"
echo -e "pid\tmig\tconfig\tlog" >> "$LOGDIR/launch_batch_${STAMP}_pids.tsv"

for entry in "${JOBS[@]}"; do
  mig="${entry%%:*}"
  stem="${entry##*:}"
  log="$LOGDIR/${stem}_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$mig" nohup bash scripts/01_train.sh "$stem" \
    --trainer.devices 1 \
    > "$log" 2>&1 &
  pid=$!
  echo -e "${pid}\t${mig}\t${stem}\t${log}" | tee -a "$LOGDIR/launch_batch_${STAMP}_pids.tsv"
  # Tiny stagger so concurrent pixi-env resolutions don't thrash the cache.
  sleep 0.5
done

echo "All 9 launched. PID table at $LOGDIR/launch_batch_${STAMP}_pids.tsv"
