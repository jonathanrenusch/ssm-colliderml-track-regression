#!/bin/bash
# Paper figures with the PRODUCTION truth-tracking KF (the truth_tracks shipped
# with the datasets) as the reference, replacing the miscalibrated in-pipeline
# ACTS KF refit -- docs/BUGREPORT_acts_pipeline_kf.md, CLAUDE.md 4.29.
#
#   run_truthkf_paper_plots.sh <pred_dir> <out_root> [datasets...]
#     pred_dir : dir of <dataset>.h5 SSM predictions (strict fp32), e.g.
#                eval_plots/sweep7/R2LFT/preds
#     out_root : writes <out_root>/<dataset>/{matched_residuals.npz,*.pdf}
#
# Designs rendered per dataset (approved campaign designs, unchanged):
#   <ds>_truthkf__rmscurve_vs_eta.pdf        (+ _vs_pt for single_muon_uniform)
#   <ds>_truthkf__residual_hist_liny.pdf
#   <ds>_truthkf__rms_vs_eta_summary{,_logy,_preclip,_postclip}.pdf
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRED_DIR="${1:?pred_dir}"
OUT_ROOT="${2:?out_root}"
shift 2
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(single_muon_2GeV single_muon_10GeV single_muon_100GeV single_muon_uniform ttbar_new_pt1 ttbar)
STORE_ROOT="${STORE_ROOT:-/scratch/colliderml/ICLR_eval_v2}"
REF_LABEL="${REF_LABEL:-truth-KF}"
PY="${PY:-python3}"

mkdir -p "$OUT_ROOT"
$PY "$REPO/scripts/build_truthkf_residuals.py" "$PRED_DIR" "$STORE_ROOT" "$OUT_ROOT" "${DATASETS[@]}"

export TRK_PLOT_TAG=truthkf TRK_REF_LABEL="$REF_LABEL"
for ds in "${DATASETS[@]}"; do
  d="$OUT_ROOT/$ds"
  [ -f "$d/matched_residuals.npz" ] || { echo "[skip] $ds: no residuals"; continue; }
  extra=(); [ "$ds" = single_muon_uniform ] && extra=(--with-pt)
  $PY "$REPO/scripts/acts_rms_curves.py" "$d" "$ds" "${extra[@]}"
  TRK_PLOT_SUBTITLE="reference = truth-tracking KF shipped with the dataset (truth_tracks); \
SSM on the same double-matched tracks of the v2 evaluation store" \
    $PY "$REPO/scripts/acts_legacy_style_plots.py" "$d" "$ds" "$REF_LABEL"
done
echo "[truthkf-plots] done -> $OUT_ROOT"
