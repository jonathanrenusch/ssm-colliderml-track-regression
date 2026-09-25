#!/bin/bash
# Predict one checkpoint on the four muon test samples and summarise the
# resolutions against the truth-seeded Kalman filter.
#
#   bash scripts/eval_checkpoint.sh <config> <ckpt> <eval_root> <out_dir>
#
# Writes <out_dir>/preds/<dataset>.h5 and <out_dir>/plots/rms_summary.{txt,json}.
# Extra arguments for predict.py (e.g. --encoder-dtype float32) can be passed
# through the PREDICT_ARGS environment variable.
set -euo pipefail
CONFIG=$1; CKPT=$2; EVAL_ROOT=$3; OUT=$4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for ds in single_muon_2GeV single_muon_10GeV single_muon_50GeV single_muon_uniform; do
  echo "=== $ds"
  python "$HERE/predict.py" --config "$CONFIG" --ckpt "$CKPT" --data-dir "$EVAL_ROOT/$ds" \
    --out "$OUT/preds/$ds.h5" ${PREDICT_ARGS:-}
done
python "$HERE/fast_rms_eval.py" --pred-dir "$OUT/preds" --store-root "$EVAL_ROOT" \
  --out-dir "$OUT/plots" --subtitle "$(basename "$CKPT")"
cat "$OUT/plots/rms_summary.txt"
