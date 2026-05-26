#!/usr/bin/env bash
# Launch the d0 cross-fix pretrain (with reduced BS for MIG) AND
# relaunch the 3 TXF configs with cudaMallocAsync (bypasses the
# caching allocator's NVML query that crashes on MIG).
set -u
cd /shared/tracking/ssm-colliderml-track-regression
STAMP="20260513_032000"
LOGDIR=launch_logs

# d0 on the previously-spare MIG. Reduce BS from 30000 → 12000 to
# fit comfortably in 47GB MIG (40% of original; bf16-mixed; tiny model).
d0_mig="MIG-57d25c0d-b961-55c1-96bc-b532b1f86aea"
d0_log="$LOGDIR/tiny_d0_8L_dim128_rangesplit_upsample_${STAMP}.log"
CUDA_VISIBLE_DEVICES="$d0_mig" \
nohup bash scripts/01_train.sh "tiny_d0_8L_dim128_rangesplit_upsample" \
  --trainer.devices 1 \
  --data.batch_size 12000 \
  > "$d0_log" 2>&1 &
d0_pid=$!
echo "$d0_pid	$d0_mig	tiny_d0_8L_dim128_rangesplit_upsample (BS=12000)	$d0_log"

# TXF retry: cudaMallocAsync bypasses the buggy caching allocator entirely,
# uses CUDA's native memory pool. No NVML query path. The math-SDPA forcing
# from train.py also stays in effect (already applied).
declare -a TXF_JOBS=(
  "MIG-1f292bf8-f63f-570f-a866-efaab3ceb50a:pretrain_transformer_2cls_lr1e-4"
  "MIG-82b2fca4-9873-5fce-b977-4bd92a37f2cb:pretrain_transformer_2cls_layerscale01"
  "MIG-9eee2fd9-6bad-585f-a6bc-ee346d274885:pretrain_transformer_2cls_tuned_lr1e-4"
)

for entry in "${TXF_JOBS[@]}"; do
  mig="${entry%%:*}"
  stem="${entry##*:}"
  log="$LOGDIR/${stem}_${STAMP}.log"
  CUDA_VISIBLE_DEVICES="$mig" \
  PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync" \
  nohup bash scripts/01_train.sh "$stem" \
    --trainer.devices 1 \
    > "$log" 2>&1 &
  pid=$!
  echo "$pid	$mig	$stem	$log"
  sleep 0.5
done

echo "Done."
