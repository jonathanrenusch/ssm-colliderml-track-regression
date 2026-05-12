#!/usr/bin/env bash
# Run training (pretrain or fine-tune) for one config.
#
# Usage:
#   bash scripts/01_train.sh <config-stem>
#
# Where <config-stem> is the YAML file name without extension under
# src/track_regression/config/<subdir>/, e.g.
#   pretrain_transformer_1cls
#   pretrain_transformer_2cls
#   pretrain_ssm_state
#   pretrain_ssm_cls
#   finetune_ssm_cls_adamw
#   finetune_ssm_cls_lion
#   finetune_ssm_cls_muon
#
# Pretraining uses 1 GPU; fine-tuning uses 4 GPU DDP. Configs encode this
# via `trainer.devices`. Override on the CLI if your hardware differs:
#   bash scripts/01_train.sh pretrain_ssm_cls --trainer.devices 1
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-stem> [extra Lightning CLI args...]" >&2
  exit 1
fi

STEM="$1"; shift

# Locate the config under config/*/<stem>.yaml.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG=$(find "$REPO_ROOT/src/track_regression/config" -name "${STEM}.yaml" -type f | head -1)

if [[ -z "$CFG" ]]; then
  echo "ERROR: no config matching ${STEM}.yaml under src/track_regression/config/" >&2
  exit 1
fi

cd "$REPO_ROOT/src/track_regression"
exec pixi run -e default python train.py fit --config "$CFG" "$@"
