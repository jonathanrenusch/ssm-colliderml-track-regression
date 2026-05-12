#!/usr/bin/env bash
# Run inference for one config + checkpoint, producing test_predictions.h5
# next to the checkpoint.
#
# Usage:
#   bash scripts/02_evaluate.sh <config-stem>
#
# By default uses checkpoints/<config-stem>/best.ckpt.  Pass --ckpt_path
# to override.
#
# IMPORTANT: num_workers=0 is mandatory for inference (DataLoader worker
# forks corrupt gzip-compressed h5 chunks; symptom is a downstream
# "filter returned failure during read" on the next eval invocation).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-stem> [extra Lightning CLI args...]" >&2
  exit 1
fi

STEM="$1"; shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG=$(find "$REPO_ROOT/src/track_regression/config" -name "${STEM}.yaml" -type f | head -1)
CKPT="$REPO_ROOT/checkpoints/${STEM}/best.ckpt"

if [[ -z "$CFG" ]]; then echo "ERROR: config $STEM.yaml not found." >&2; exit 1; fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint $CKPT not found." >&2
  exit 1
fi

cd "$REPO_ROOT/src/track_regression"
exec pixi run -e default python train.py test \
  --config "$CFG" \
  --ckpt_path "$CKPT" \
  --trainer.devices 1 \
  --data.batch_size 10000 \
  --data.num_workers 0 \
  "$@"
