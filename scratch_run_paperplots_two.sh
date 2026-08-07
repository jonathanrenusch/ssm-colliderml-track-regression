#!/usr/bin/env bash
set -uo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression
cd "$REPO/src/track_regression"
export TRITON_CACHE_DIR=/tmp/triton_cache
export CUDA_VISIBLE_DEVICES=0
export COMET_OFFLINE_ROOT="$REPO/src/track_regression/logs/comet_offline"
export PAPER_PLOTS_ROOT="$REPO/logs/paper_plots"
mkdir -p "$TRITON_CACHE_DIR" "$PAPER_PLOTS_ROOT"

run () {
  local rid="$1" nice="$2"
  echo "############## START $nice ($rid) $(date) ##############"
  pixi run python -m track_regression.paper_plots.cli \
    --run-id "$rid" \
    --nicename "$nice" \
    --data-dir /scratch/colliderml/arxiv_retraining/p200_core_kf_hits_finetune \
    --gpu 0
  echo "############## END   $nice rc=$? $(date) ##############"
}

run eaa7e3a1f47547d5b4ae464c7f6e0132 finetune_ssm_cls_4L_muon_kfhits_eaa7e3a1
run ba96d05fa8ab426e8c87b46f4b8c1223 finetune_ssm_cls_10L_muon_kfhits_ba96d05f
echo "ALL DONE $(date)"
