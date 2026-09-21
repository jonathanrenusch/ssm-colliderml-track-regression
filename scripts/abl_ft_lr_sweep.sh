#!/bin/bash
# Fine-tune learning-rate sweep for the h=192 minGRU stage 2.
#
# Each point is a COMPLETE, compressed WSD schedule (warm-up / stable / decay
# are fractions of the total step count, so shortening the run compresses the
# whole shape rather than truncating it) starting from the same stage-1
# checkpoint, on ONE GPU at batch 40,000 -- the same effective batch as the
# production DDP 2x20k run, so the learning rate transfers.
#
# Selection statistic: geometric mean of the five clipped validation RMSEs on
# the identical 1 M-track pooled val subset, from the last validation of the
# run.  The truth-KF constants are common to all points, so ranking on the raw
# geomean ranks identically to ranking on GM5.
#
#   bash scripts/abl_ft_lr_sweep.sh <gpu> [steps_per_epoch] [epochs]
set -u
G="${1:-3}"; SPE="${2:-1000}"; EP="${3:-10}"
REPO=/shared/tracking/ssm-colliderml-track-regression
CKPT=$REPO/eval_plots/ablations_2026-09/v2_runs/e7d5d0ea13b044a7b162086c49cdc5a9/ckpts/last.ckpt
OUT=$REPO/eval_plots/ablations_2026-09/ft_lr_sweep
mkdir -p "$OUT"
cd $REPO/src/track_regression
export CUDA_VISIBLE_DEVICES="$G" TRK_MATMUL_PRECISION=highest
export TRITON_CACHE_DIR=/tmp/triton_cache_ftlr_$G
# (adam_max, muon_max) = the production pair 2e-5 / 6e-5 scaled together.
for M in 0.25 0.5 1.0 2.0; do
  A=$(python3 -c "print(f'{2.0e-5*$M:.8g}')")
  U=$(python3 -c "print(f'{6.0e-5*$M:.8g}')")
  L="$OUT/ftlr_x${M}.log"
  if grep -q "RC=0" "$L" 2>/dev/null; then echo "x$M (cached)"; continue; fi
  echo "=== x$M  adam_max=$A muon_max=$U  $(date -Iseconds)"
  pixi run -e default python train.py fit \
    --config config/ssm_cls/ICLR_sweep7/minGRU_h192_FT_mix3_muonhybrid_ddp2_bs40k_wsd50.yaml \
    --model.pretrained_ckpt_path "$CKPT" \
    --model.lrs_config.max "$A" --model.lrs_config.muon_max "$U" \
    --model.lrs_config.initial "$(python3 -c "print(f'{2.0e-6*$M:.8g}')")" \
    --trainer.devices 1 --trainer.max_epochs "$EP" \
    --trainer.limit_train_batches "$SPE" \
    --data.batch_size 40000 \
    > "$L" 2>&1
  echo "### TAG=ftlr_x$M RC=$? $(date -Iseconds)" >> "$L"
  tr '\r' '\n' < "$L" | grep -o "\[val epoch [0-9]*\] iter-3σ RMSE.*" | tail -1
done
echo ALL_FT_LR_DONE
