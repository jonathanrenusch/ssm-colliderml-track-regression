#!/bin/bash
# All per-sample paper figures and the ratio table from one set of predictions.
#
#   bash scripts/make_paper_figures.sh <pred_dir> <eval_root> <out_dir>
#     pred_dir  : directory of <dataset>.h5 network predictions
#     eval_root : evaluation root holding <dataset>/test stores (with truth-KF side-cars)
#     out_dir   : writes <out_dir>/<dataset>/{matched_residuals.npz, *.pdf}
#
# Paper conventions: |eta| <= 2 on every figure and in the table; the uniform
# sample's vs-pT figure and table row are capped at pT <= 70 GeV.
# Per dataset:
#   <ds>_truthkf__rmscurve_vs_eta.pdf      (+ _vs_pt for single_muon_uniform)
#   <ds>_truthkf__residual_hist_liny.pdf
# and the LaTeX rows of the ratio table (400 bootstrap replicas) on stdout and
# in <out_dir>/table_ratios.tex.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRED_DIR="${1:?usage: make_paper_figures.sh <pred_dir> <eval_root> <out_dir>}"
EVAL_ROOT="${2:?eval_root}"
OUT="${3:?out_dir}"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ETA_MAX=2
PT_MAX=70
DATASETS=(single_muon_2GeV single_muon_10GeV single_muon_50GeV single_muon_uniform)

mkdir -p "$OUT"
"$PY" "$REPO/scripts/build_residuals.py" "$PRED_DIR" "$EVAL_ROOT" "$OUT" "${DATASETS[@]}"

for ds in "${DATASETS[@]}"; do
  d="$OUT/$ds"
  [ -f "$d/matched_residuals.npz" ] || { echo "[skip] $ds: no residuals"; continue; }
  extra=(); [ "$ds" = single_muon_uniform ] && extra=(--with-pt --pt-max "$PT_MAX")
  "$PY" "$REPO/scripts/plot_rms_curves.py" "$d" "$ds" --eta-max "$ETA_MAX" "${extra[@]}"
  "$PY" "$REPO/scripts/plot_residual_hists.py" "$d" "$ds" --eta-max "$ETA_MAX"
done

"$PY" "$REPO/scripts/table_ratios.py" "$OUT" --eta-max "$ETA_MAX" --pt-max "$PT_MAX" \
  --n-boot 400 | tee "$OUT/table_ratios.tex"
echo "[paper-figures] done -> $OUT"
