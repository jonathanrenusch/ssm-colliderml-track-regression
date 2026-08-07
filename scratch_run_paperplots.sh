#!/usr/bin/env bash
set -euo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression
cd "$REPO/src/track_regression"
export TRITON_CACHE_DIR=/tmp/triton_cache
export CUDA_VISIBLE_DEVICES=0
export COMET_OFFLINE_ROOT="$REPO/src/track_regression/logs/comet_offline"
export PAPER_PLOTS_ROOT="$REPO/logs/paper_plots"
mkdir -p "$TRITON_CACHE_DIR" "$PAPER_PLOTS_ROOT"

pixi run python -m track_regression.paper_plots.cli \
  --run-id 386c6525f6c948bba2ed278f2ff6bf60 \
  --nicename finetune_ssm_cls_4L_muon_kfhits_386c6525 \
  --data-dir /scratch/colliderml/arxiv_retraining/p200_core_kf_hits_finetune \
  --gpu 0
