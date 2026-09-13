# Pre-clip (tail-inclusive) RMS-vs-eta, best deployment model (R2Lnoconv-FT)

Un-clipped RMS (no iterative-3sigma tail removal) of every perigee parameter,
SSM vs the shipped truth-KF, on the double-matched subset, |eta| <= 2, at
deployment inference settings.  One page per test set + the uniform vs-pT page
(capped at 70 GeV).  Generated 2026-09-11 by
`scripts/acts_rms_curves.py` with `TRK_PRECLIP=1` from the fp32 deploy preds
(`eval_plots/sweep7/R2LnoconvFT_deploy/preds`).  These are the SAME numbers as
the paper's post-clip pages but WITHOUT the tail clip, so the legends show the
full-distribution RMS and the ratio strips carry the tails.

Sets: single_muon_{2,10,50,100}GeV, single_muon_uniform (vs-eta + vs-pt),
ttbar, ttbar_new_pt1.  (100 GeV and ttbar are internal-only; the reference is
miscalibrated above ~95 GeV even inside |eta|<=2, so read those with care.)
