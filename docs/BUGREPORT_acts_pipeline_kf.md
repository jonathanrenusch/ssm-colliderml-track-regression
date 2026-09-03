# BUG REPORT: ACTS-pipeline KF baseline is miscalibrated — switch all paper plots/values to the production `truth_tracks` reference

**Date:** 2026-09-03 · **Repo:** `/shared/tracking/ssm-colliderml-track-regression`, branch `v2_campaign` (pull first).
**Full documentation:** CLAUDE.md §4.29 · **Evidence plot:** `eval_plots/paper_plots/acts_kf_check/uniform_kf_calibration_check.pdf`
**Reproduction:** the three-way check described below (session scratchpad `acts_kf_check.py`; its estimator is quoted in §"What to do", item 4).

## The issue

The official ACTS validation pipeline (`scripts/acts_integration.py`, output under
`eval_plots/paper_plots/acts_official/` and `acts_official_R2LFT/`) does **not** compare against
the ColliderML production truth-tracking KF. It runs its **own ad-hoc KF refit** inside the event
loop (truth-*estimated* seeding + `addKalmanTracks` as configured in-script), and that fit is
measurably miscalibrated. Verified on `single_muon_uniform` (200 k events) by joining the
production `truth_tracks` onto the *same particles* and pushing all three estimators through the
*identical* ACTS scipy-Gaussian estimator (iterative ±3σ range, per-bin adaptive windows):

| integrated σ | SSM | pipeline ad-hoc KF | **production truth-KF** | pipe/prod | SSM/prod |
|---|---|---|---|---|---|
| d0 | 14.0 µm | 18.7 µm | 14.3 µm | 1.31 | 0.98 |
| z0 | 20.4 µm | 42.7 µm | 20.4 µm | **2.10** | 1.00 |
| φ | 0.160 mrad | 0.232 mrad | 0.168 mrad | 1.39 | 0.96 |
| θ | 0.0437 mrad | 0.103 mrad | 0.0455 mrad | **2.27** | 0.96 |
| q/p | 2.13e-4 GeV⁻¹ | 2.85e-4 | 2.17e-4 | 1.31 | 0.98 |

Structure: the θ excess is barrel-peaked (3.3× at η = 0 → ~1.2 forward); z0 carries a 10–11 %
far-outlier population at |η| ≈ 0.35–0.45 (the spikes seen in the official pages).

**Mechanism:** pyacts' `ColliderMLRelease1InputConverter` never ingests the v2 production's
per-cluster measurement variances (`var_loc0`/`var_loc1` are not in its hit schema) — it
re-derives local coordinates and covariances from global positions with internal defaults, so
the KF weights hits wrongly (worst for the strip-v coordinate → z0/θ). It also fits only
72–75 % of tracks (truth-estimated seeding + fit failures). The production `truth_tracks` were
fitted upstream on the real digitized measurements with their real covariances.

## Consequence for the paper

Any ratio quoted against the in-pipeline "ACTS KF" (the 0.83–0.92 σ-ratios, any
"beats the ACTS KF by 10–17 %" framing derived from `acts_official*/resolutions.pdf` or the
band/legacy pages) **overstates the SSM by the baseline's miscalibration**. The correct,
defensible result — consistent across both the `fast_rms_eval` framework and the ACTS
estimator — is:

> **SSM ≈ production truth-KF at 0.96–1.00 on all five parameters**, plus the (still valid)
> efficiency contrast: the SSM fits 100 % of prototracks vs the KF refit's ~72–75 %.

## What to do

1. **Switch the reference in every paper figure and number to the default shipped tracks**:
   the production truth-KF. Sources: the per-track `truth_kf_reco.npy` side-cars in the flat
   stores (`/scratch/colliderml/ICLR_retraining_v2/...` — what `fast_rms_eval.py` already uses
   as "truth-KF"), or the raw `truth_tracks` parquet tables
   (`/scratch/colliderml/drift_beamspot_v2/<ds>/.../truth_tracks`, columns
   `d0, z0, phi, theta, qop`; join by event/particle **values**, never by row — CLAUDE.md §0.4).
2. **Regenerate all paper plots and quoted values with that reference** — every figure/table
   currently derived from `eval_plots/paper_plots/acts_official*/` output. The
   `fast_rms_eval`-based bundles (`eval_plots/paper_plots/fast_rms_R2LFT/`,
   `fast_rms_YZ3LFT/`) already use the production truth-KF and are correct as-is.
3. Where an ACTS-pipeline figure is kept (it remains valuable as an independent-framework
   check), **relabel the KF curve "ACTS KF refit (as configured in the validation pipeline)"**,
   state the calibration caveat in the caption, and quote the SSM-vs-production-truth-KF
   numbers in the text — never the pipeline ratios. Keep the matched-fraction comparison
   (100 % vs 74.6 % on ttbar); that one is genuine.
4. For any resolution recomputed with the ACTS estimator, use the **iterative Gaussian fit**:
   per bin, start window = median ± 8·robust-σ (100 bins), fit with
   `acts.examples.scipy.makeScipyHistogramFitFunction`, refit within mean ± 3σ until converged
   (≤ 5 iterations). Do **not** use the fixed-window single pass: the stock windows
   under-resolve central cores (~15 % σ inflation on z0: 26.7 µm bins vs a 15 µm core) and blow
   up on outlier-contaminated bins (σ spikes of 490–610 µm at |η| ≈ 0.4).
5. Paper-text sanity list after the switch: any "×2 better z0/θ than ACTS" claim; the 100 GeV
   φ/q-p sub-unity ratios (still true vs the production truth-KF: ~0.85–0.86 per the campaign
   tables, concentrated at |η| > 2 and in the tails — see CLAUDE.md §4.26); deployment-section
   throughput numbers are unaffected.
6. Upstream note (acknowledgements / report to the ColliderML & pyacts producers, not the
   paper): the Release-1 converter should ingest the digitized `loc0/loc1` +
   `var_loc0/var_loc1` columns; until then, any in-pipeline KF refit on v2 data is not
   calibration-grade. Related upstream items from the same campaign: the v2 hits-schema break
   (nested `particle_ids`, truth moved to `tracker_simhits` — shimmed by
   `scripts/make_acts_compat_parquet.py`) and 4 wrong-surface hits in 45 M ttbar hits
   (`scripts/filter_acts_hits.py`).

## Addendum (2026-09-03): the miscalibration is NOT caused by our v2 adaptations

Checked explicitly: (a) the SSM consumes the identical shim+converter hit positions and
reproduces its flat-store numbers to ~1 % (d0 14.0 vs 13.9 µm) — the measurements reaching both
fitters are fine; (b) all our script changes are SSM-only or system-symmetric; (c) the one real
loophole, `--hit-bounds-tolerance 25` (vs the original 5), is a measured no-op on the dataset in
question: 0 of 3,150,281 uniform-muon hits project outside sensor bounds even at 5 mm. The
defect is in the original configuration: the converter's default measurement covariances plus
the in-script truth-estimated seeding / bare KF setup.

## Addendum 2 (2026-09-03 evening): hit-ordering A/B — efficiency was ordering; σ is covariances

The shim originally passed the v2 DIGITISED hit time as the Release-1 `time` column; ACTS's
`TruthTrackFinder` sorts prototracks by `SimHit::time()`, so all KF prototracks were scrambled
(strips = time 0 first). A/B rerun of the identical uniform 200 k pipeline with `time :=
tracker_simhits.true_time` (now the shim default; `DIGI_TIME=1` restores the old behaviour):

- **KF match rate 76.0 % → 99.3 %** — the fit losses were entirely the scrambled ordering
  (seeding/fit failures), i.e. an artifact of the v2 schema change, now fixed in the shim.
- **The σ miscalibration persists** (iterative estimator, vs production truth-KF):
  d0 1.35×, z0 2.00×, φ 1.52×, θ 3.18×, q/p 1.43×. (Ratios are not directly comparable to
  Addendum-1's 76 %-survivor sample: the fixed run now includes the recovered ~23 % hardest
  tracks.) Conclusion unchanged: the resolution excess is the converter's default measurement
  covariances — paper references stay on the production `truth_tracks`.
- The 100 %-vs-74.6 % efficiency contrast in older text should be restated as
  100 % vs 99.3 % (with the time fix) — the SSM efficiency point survives only as
  "fits every prototrack including the 0.7 % the refit still loses".
