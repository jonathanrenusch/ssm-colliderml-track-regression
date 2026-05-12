#!/usr/bin/env bash
# Run the paper-plot pipeline for every shipped checkpoint, then aggregate
# the cross-run summary tables.
#
# For each <stem> with a non-TBD checkpoint, this:
#   (1) shadow-mirrors the checkpoint + config into logs/comet_offline/<stem>/
#       (the layout paper_plots/cli.py expects);
#   (2) runs `train.py test` to produce last__test_predictions.h5 (skipped
#       if the file already exists);
#   (3) builds a paper-plot bundle under $PAPER_PLOTS_ROOT/<stem>/ with
#       config + ckpt symlinks + plots + bootstrap stats.
#
# Final aggregator emits LaTeX/CSV tables in $PAPER_PLOTS_ROOT/_summary/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHADOW_ROOT="${COMET_OFFLINE_ROOT:-$REPO_ROOT/logs/comet_offline}"
export COMET_OFFLINE_ROOT="$SHADOW_ROOT"
export PAPER_PLOTS_ROOT="${PAPER_PLOTS_ROOT:-$REPO_ROOT/logs/paper_plots}"
mkdir -p "$SHADOW_ROOT" "$PAPER_PLOTS_ROOT"

STEMS=(
  pretrain_transformer_1cls
  pretrain_transformer_2cls
  pretrain_ssm_state
  pretrain_ssm_cls
  finetune_ssm_cls_adamw
  finetune_ssm_cls_lion
  finetune_ssm_cls_muon
)

cd "$REPO_ROOT/src/track_regression"

for stem in "${STEMS[@]}"; do
  ckpt="$REPO_ROOT/checkpoints/${stem}/best.ckpt"
  cfg=$(find config -name "${stem}.yaml" -type f | head -1)

  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] $stem: checkpoint missing"
    continue
  fi

  # shadow comet_offline run dir for paper_plots/cli.py
  rd="$SHADOW_ROOT/$stem"
  mkdir -p "$rd/ckpts"
  # Copy the leaf config and its sibling base.yaml so Lightning's
  # auto-base-loader (train.py:43-51) finds them next to each other.
  cp -f "$cfg" "$rd/config.yaml"
  cp -f "$(dirname "$cfg")/base.yaml" "$rd/base.yaml"
  ln -sf "$ckpt" "$rd/ckpts/last.ckpt"

  echo "=== $stem ==="
  pixi run -e default python -m track_regression.paper_plots.cli \
    --run-id "$stem" \
    --nicename "$stem" \
    --output-root "$PAPER_PLOTS_ROOT" \
    --data-dir "/scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune" \
    --gpu 0 \
    "$@" || echo "[warn] $stem pipeline exited non-zero; continuing"
done

# Aggregator runs at end of each cli.py call automatically; one final pass
# here flushes the cross-run summary tables.
pixi run -e default python -m track_regression.paper_plots.aggregate || true

echo ""
echo "Done. Summary tables: $PAPER_PLOTS_ROOT/_summary/"
