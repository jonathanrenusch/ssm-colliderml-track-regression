#!/bin/bash
# Build the ICLR_v2 paper figure bundle from a set of fp16-inference
# predictions: the truth-KF reference pages at |eta| <= 2, the vs-pT page
# capped at 70 GeV, the pre-clip supplements, the 100 GeV vs-impact pages and
# the quantile-ladder calibration pages.  The paper's sync_figures.sh pulls
# from the directory this writes.
set -eu
cd /shared/tracking/ssm-colliderml-track-regression
PRED=eval_plots/ablations_2026-09/v2_evals/V2_minGRU_FTfinal_fp16/preds
B=eval_plots/paper_plots/truthkf_minGRU_FT_fp16_eta2
rm -rf "$B"
PY="pixi run -e default python" TRK_ABS_ETA_MAX=2 STORE_ROOT=/scratch/colliderml/ICLR_eval_v2 \
  bash scripts/run_truthkf_paper_plots.sh "$PRED" "$B" \
  single_muon_2GeV single_muon_10GeV single_muon_50GeV single_muon_100GeV single_muon_uniform
export TRK_PLOT_TAG=truthkf TRK_REF_LABEL=truth-KF TRK_ABS_ETA_MAX=2
# vs-pT page capped at 70 GeV (paper default since 2026-09-07)
TRK_PT_MAX=70 pixi run -e default python scripts/acts_rms_curves.py \
  "$B/single_muon_uniform" single_muon_uniform --with-pt
# pre-clip supplements
for ds in single_muon_2GeV single_muon_10GeV single_muon_50GeV; do
  TRK_PRECLIP=1 pixi run -e default python scripts/acts_rms_curves.py "$B/$ds" "$ds"
done
TRK_PRECLIP=1 TRK_PT_MAX=70 pixi run -e default python scripts/acts_rms_curves.py \
  "$B/single_muon_uniform" single_muon_uniform --with-pt
# 100 GeV RMS vs impact parameters (analysis-meeting pages)
pixi run -e default python scripts/rms_vs_impact_npz.py "$B/single_muon_100GeV" single_muon_100GeV
# quantile-ladder calibration
for ds in single_muon_2GeV single_muon_10GeV single_muon_50GeV single_muon_uniform; do
  pixi run -e default python scripts/quantile_calibration.py "$PRED/$ds.h5" "$ds" "$B/$ds"
done
echo BUNDLE_DONE
