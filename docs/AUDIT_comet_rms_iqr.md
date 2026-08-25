# Audit: `ssm_rms_dm` vs `ssm_iqr_dm` orders-of-magnitude gap in Comet

Date: 2026-08-25. Scope: training/validation metric path in `src/track_regression/model.py`
vs the offline evaluation path (`scripts/fast_rms_eval.py`, `paper_plots/stats.py`,
`eval_utils.py`). Read-only audit; no source file was modified.

## Verdict (TL;DR)

**There is no bug in the metric computation or logging path.** `ssm_rms_dm`,
`ssm_iqr_dm` and `ssm_precision_dm` are computed from the *same* residual tensor, in
the same physical units, on the same double-matched subset, with the same phi wrap,
and Lightning's mean-over-batches aggregation changes them by < 2.5 %. The logged
values reproduce the offline h5-based evaluation of the same checkpoint to ~1 %.

The gap is real and has two layers:

1. **Estimator mismatch (the "inconsistency" the user saw).** The training path logs
   the **raw, un-clipped RMS**. The offline numbers that "agree with IQR/1.349 within
   1.6x" are the **iterative 3σ-clipped RMSE** (`eval_utils.iterative_rms_convergence`),
   a different estimator that removes the tail. The offline path *also* computes the
   raw RMS (`raw_rms` in `paper_plots/stats.py`, `_pre` in `fast_rms_eval.py`) and it
   shows exactly the same 15x–480x ratio to the IQR. Same residuals, different
   statistic.
2. **Why the ratio is 100x–1900x and not 2x.** On the drift_beamspot models, 4–6.5 %
   of double-matched tracks have catastrophic residuals (theta off by ~0.8 rad RMS,
   phi by ~1.2 rad) that carry 99–100 % of Σr². The IQR ignores them by construction;
   RMS and std are dominated by them. The CKF fits those same tracks to within 2–4x of
   its core resolution, so the *labels are fine* — this is an SSM prediction failure
   on a fixed sub-population (heavily enriched at |eta| < 0.5 for theta, in low-pT
   tracks, and in tracks containing negative-time hits). That is a model/data problem
   for a separate investigation, not a metric bug; evidence is in section 6.

Minimal fix (section 7): log an iterative-3σ-clipped RMS and the clipped fraction next
to the raw RMS, so the training curves carry the same estimator the offline reports
call "RMSE". Verified: per-batch clipped RMS averaged over batches equals the pooled
offline value to 0.2 % on all five parameters.

## 1. The symptom, with the actual logged values

Pulled from Comet via the REST API (`comet_ml.API`, user's own `COMET_API_KEY` from
the environment; the runs are online, `logs/comet_offline/<key>/` holds only
ckpts/config/metadata).

Run `b4e6d4930aea441d8e78c368cd250d08` (TRK-SSMCLS-ICLR-ft10L-from57dabaab-newnorm-bs12k,
10L, padded, single_muon_uniform, last step 39899):

| param | `ssm_iqr_dm` | `ssm_precision_dm` (std) | `ssm_rms_dm` | RMS / IQR-σ |
|---|---|---|---|---|
| d0 [mm] | 0.01588 | 0.50488 | 0.50488 | 31.8x |
| z0 [mm] | 0.04240 | 23.7288 | 23.7289 | 560x |
| phi [mrad] | 0.4526 | 262.307 | 262.308 | 580x |
| theta [mrad] | 0.1109 | 213.674 | 213.673 | 1927x |
| qop [1/GeV] | 2.777e-4 | 0.011547 | 0.011547 | 41.6x |

Two facts in that table already localise the cause before reading any code:

* **std == RMS to 5 significant figures** for every parameter, so RMS² − std² = mean² ≈ 0:
  there is no bias, no anchor added twice, no normalised-vs-physical mix-up. The gap is
  pure spread.
* **RMS/std are frozen across all 10 epochs** (phi 262.307–262.309 mrad; z0
  23.706–23.754 mm) while the IQR improves 2–7x (z0 0.287 → 0.042 mm, d0 0.0238 →
  0.0159 mm). A spread that does not respond to training while the core sharpens is a
  fixed population of catastrophic residuals, not a scaling error (a unit or
  normalisation bug would scale *with* the improving core).

Run `09c54481fbb84e42b9e155ed550fd91f` (4L d128 ds128, packed, bs 32k, last step 74799):
d0 15.6x, z0 122x, phi 259x, theta 478x, qop 11.1x. The all-tracks `val/z0/precision`
(std, not DM-filtered) is 33.3 mm and `train/z0/precision` 31–37 mm — same phenomenon
everywhere, slightly worse outside the DM subset.

## 2. Training-time metric path, traced

Entry: `TrackRegressionWrapper._shared_step` (`src/track_regression/model.py:849`).
`validation_step`/`training_step` (lines 1019–1023) both call it; `test_step` (1025–1048)
inlines the same sequence.

* Line 853: `valid_mask = targets.get("track_valid")` — all-True bool from the flat
  collate (`flat_data.py:166`).
* Line 854: `compute_loss` → `TrackParameterLoss.forward` (`losses.py:989`). Targets are
  normalised *locally* (`t_norm = _linear_normalise(target, …)`, `losses.py:248`);
  the `targets` dict is never mutated, so metrics see physical targets.
* Line 896: `preds = self.model.loss_module.predict_physical(outputs["pred"], targets)`
  → `losses.py:1212`. Calls `predict` (`losses.py:1189`), which slices the 30-dim head by
  `_output_slices` in `parameter_order` order (d0:[0,7], z0:[7,14], phi:[14,16],
  theta:[16,23], qop:[23,30] for this config) and dispatches:
  * `QuantileLoss.predict` (`losses.py:259`): median channel → `_linear_denormalise`
    (`losses.py:68`, `(u+1)/2·(max−min)+min`) → physical mm / 1/GeV.
  * `EtaQuantileLoss.predict` (`losses.py:359`): median eta → denormalise → `_eta_to_theta`
    → physical theta [rad].
  * `CircularPhiLoss.predict` (`losses.py:437`): `atan2(sin, cos)` → phi in (−π, π].
  * No `delta_anchor` in any ICLR/finetune config, so the anchor branch (1230–1236) is a
    no-op; `predict_physical` == `predict`.
* Line 897: `self._log_metrics(preds, targets, valid_mask, stage)` (`model.py:910`).

Inside `_log_metrics`, for each `name in parameter_order`:

| line | what | note |
|---|---|---|
| 941–946 | `p = preds[name][valid_mask]`, `t = targets[name][valid_mask]` | both (N,), physical units |
| 947 | `residual = p - t` | |
| 950–952 | phi wrapped to (−π, π] via `torch.where` ±2π | one wrap suffices: both operands ∈ (−π, π] |
| 961 | `{stage}/{name}/mae` | all valid tracks |
| 964–970 | `{stage}/{name}/precision {unit}` = `residual.std() * scale` | all valid tracks |
| 975–979 | `dm_mask = targets["acts_dm_mask"][valid_mask]` | bool (`flat_data.py:172`), guarded by `acts_reco_{name}` presence only |
| 982–987 | `ssm_residual_dm = p[dm_mask] - t[dm_mask]`, phi re-wrapped | **the one tensor all three DM metrics use** |
| 989–995 | `ssm_precision_dm` = `ssm_residual_dm.std() * scale` | |
| 996–1007 | `ssm_iqr_dm` = `(q75 − q25) / 1.349 * scale`, `torch.quantile` on the residuals | not on predicted quantiles |
| 1008–1014 | `ssm_rms_dm` = `sqrt(mean(ssm_residual_dm²)) * scale` | raw, un-clipped |

`scale` comes from `_PRECISION_UNITS` (`model.py:900–906`): d0/z0 ×1 [mm], phi/theta ×1000
[mrad], qop ×1 [1/GeV] — the same `(unit, scale)` tuple is used for all three DM metrics.

Answers to the checklist (a)–(g), per metric — identical for `ssm_precision_dm`,
`ssm_iqr_dm`, `ssm_rms_dm`:

* (a) physical units, then ×scale — yes for all three (same `scale`).
* (b) phi wrapped — yes, on the shared tensor (985–987).
* (c) subset — valid ∧ `acts_dm_mask`, identical (982–984).
* (d) aggregation — Lightning defaults for `self.log` in `validation_step`: `on_step=False,
  on_epoch=True, reduce_fx="mean"`, weighted by the inferred batch size (B for padded
  batches; 1 for packed, see memory note) — the same weights apply to all metrics of a
  step, and `sync_dist=True` averages ranks. Mean-of-per-batch-RMS ≤ pooled RMS
  (Jensen), measured deviation ≤ 2.4 % (section 5).
* (e) no missing sqrt/square: RMS = `sqrt((r²).mean())`, std = `.std()`, IQR linear.
* (f) no `on_step/on_epoch/reduce_fx` mismatch: none are passed; all three use defaults.
* (g) no indexing/ordering mismatch: `preds` is keyed by name, produced by
  `_output_slices` in `parameter_order`; the IQR is of *residuals of the median*, not of
  the predictive quantiles (those feed only `quantile_calibration_metrics`).

Also checked and ruled out: `acts_dm_mask` dtype (bool, so no integer fancy-indexing),
target mutation by the loss (none), autocast (`encoder_autocast_dtype: float32`,
`precision: 32-true`; heads/loss/metrics are fp32 regardless), shape broadcasting
(all (N,)), NaN poisoning (would hit all three), Comet name mangling (names round-trip
verbatim, including the `[mm]` suffix), and the padded/packed collates (`flat_data.py:150–200`
build `acts_dm_mask` identically).

## 3. Offline paths and where the "divergence" actually is

* `eval_utils.iterative_rms_convergence` (`eval_utils.py:322–368`): up to 5 passes of
  clipping at mean ± 3·np.std, then `rms = sqrt(mean(x²))` of the survivors. Returns
  `frac_kept`.
* `paper_plots/stats.py`: `_raw_rms` (line 24, un-clipped), `_iqr_robust_sigma` (35–37,
  `(q3−q1)/1.349`), `_iter3sigma_rms` (40–43). `stats.txt` prints all three; the
  "agree within 1.6x" statement is between **IQR/1.349 and the iter-3σ RMSE**. In the
  same `stats.txt` (finetune_ssm_cls_muon_kf_hits_f93223a9, legacy data) the *raw* RMSE
  is already 7.4x (d0) and 13.4x (z0) the IQR-σ.
* `scripts/fast_rms_eval.py`: `_raw_rms` (162) and `_iter_rms` (166); `rms_summary.txt`
  reports iter-3σ as the headline and a "pre-clip / iter-3sigma ratio" tail line.
* `eval_utils.write_residual_statistics_report` (line 587): IQR/1.349 with `np.percentile`.

Residual construction is the same as the training path: `preds[p] − targets[p]` from
the `RegressionPredictionWriter` h5 (which stores `predict_physical` output), phi wrapped
with `np.mod`, DM mask from `acts_dm.npy` (`fast_rms_eval.py:114–158`).

So the training path does **not** diverge in residuals, units, subset or wrap. It
diverges only in *which RMS is reported*: training logs the raw RMS; the offline
headline is the clipped RMS. For the same checkpoint (09c54481) on single_muon_uniform:

| param | offline iter-3σ RMSE (test) | Comet `ssm_iqr_dm` (val) | offline pre/iter ratio | Comet RMS/IQR-σ | offline clipped % |
|---|---|---|---|---|---|
| d0 | 32.1 µm | 32.4 µm | 15.7x | 15.6x | 4.23 % |
| z0 | 200 µm | 194.7 µm | 118x | 122x | 4.60 % |
| phi | 1.03 mrad | 1.011 mrad | 253x | 259x | 4.39 % |
| theta | 0.733 mrad | 0.447 mrad | 290x | 478x (= 292x vs iter-3σ) | 6.46 % |
| qop | 1.15e-3 | 1.06e-3 | 10.5x | 11.1x | 5.52 % |

(sources: `eval_plots/09c54481_4L_d128_muon/rms_summary.txt`, `eval_plots/CROSS_MODEL_SUMMARY.txt`.)

## 4. Same-sample reproduction

Script: scratchpad `repro_rms_iqr.py` (not committed). Inputs: cached predictions
`/scratch/colliderml/ICLR_eval/_preds/single_muon_uniform.h5` (checkpoint 09c54481,
4,998,269 test tracks) and `acts_dm.npy` / `acts_reco.npy` / `targets.npy` from
`/scratch/colliderml/ICLR_eval/single_muon_uniform/test`. h5 targets equal the store
targets bit-for-bit (row alignment verified). DM subset: 3,491,385 tracks (69.9 %).

Pooled over all DM tracks, with the exact `_log_metrics` definitions:

| param | unit | mean | std | raw RMS | IQR/1.349 | iter-3σ RMS | raw/IQR-σ | iter-3σ/IQR-σ | clipped |
|---|---|---|---|---|---|---|---|---|---|
| d0 | mm | 0.0009 | 0.5052 | 0.5052 | 0.03239 | 0.03209 | 15.6 | 0.99 | 4.23 % |
| z0 | mm | −0.013 | 23.60 | 23.60 | 0.1942 | 0.1996 | 121.5 | 1.03 | 4.60 % |
| phi | mrad | 0.065 | 260.2 | 260.2 | 1.011 | 1.028 | 257.5 | 1.02 | 4.39 % |
| theta | mrad | −0.070 | 212.8 | 212.8 | 0.4463 | 0.7332 | 476.9 | 1.64 | 6.46 % |
| qop | 1/GeV | −1.3e-4 | 0.01203 | 0.01203 | 0.00106 | 0.00115 | 11.3 | 1.08 | 5.52 % |

CKF on the same DM tracks, for contrast: raw/IQR-σ = 1.7 (d0), 5.2 (z0), 2.5 (phi), 3.2
(theta), 2.3 (qop) — the "factor 1–3 for a roughly Gaussian residual" the user expected.

Per-batch (bs = 12 000, on-disk order — what `validation_step` sees) then mean over the
417 batches (what Lightning logs), vs pooled:

| param | mean(std_b) | mean(RMS_b) | mean(IQR-σ_b) | pooled RMS | pooled IQR-σ |
|---|---|---|---|---|---|
| d0 | 0.5043 | 0.5043 | 0.03239 | 0.5052 | 0.03239 |
| z0 | 23.56 | 23.56 | 0.1942 | 23.60 | 0.1942 |
| phi | 259.9 | 259.9 | 1.011 | 260.2 | 1.011 |
| theta | 212.4 | 212.4 | 0.4465 | 212.8 | 0.4463 |
| qop | 0.01174 | 0.01175 | 0.00106 | 0.01203 | 0.00106 |

Those per-batch means reproduce the Comet values for 09c54481 (val split) to ~1 %
(z0 23.56 vs 23.73; phi 259.9 vs 262.0; IQR-σ identical to 3 digits). The aggregation
is not the problem; the largest Jensen effect is 2.4 % on qop.

## 5. Why 100x–1900x specifically: the outlier population

Define outliers per parameter as tracks outside the converged iter-3σ window
(4.2–6.5 % of DM tracks). Then:

* Outliers carry **99.6–100 % of Σr²**; RMS without them equals the iter-3σ RMS
  (0.032 mm d0, 0.200 mm z0, 1.03 mrad phi, 0.73 mrad theta), i.e. within 1–1.6x of
  IQR-σ. RMS² ≈ f·R_out² with f ≈ 0.05 and R_out ≈ 0.84 rad (theta), 1.24 rad (phi),
  ~110 mm (z0), ~2.4 mm (d0) — that arithmetic *is* the 480x / 260x / 120x / 16x.
* 13.4 % of DM tracks are outliers in at least one parameter, 1.8 % in all five;
  P(outlier in q | outlier in p) ≈ 0.4–0.65: largely the *same tracks* fail across
  parameters → a per-track failure, not a per-head one.
* **Labels are fine**: the CKF residual on SSM-outlier tracks is 0.39 mrad theta / 0.32 mm
  z0 (vs 0.12 / 0.09 on core tracks) — 2–4x worse, not 1000x. The truth perigee is
  consistent with a full Kalman fit on exactly the tracks the SSM gets catastrophically
  wrong.
* Theta-outlier fraction vs eta: 20.6 % at |eta| < 0.5, 9 % at 0.5–1, 3.7 % at 1–1.5,
  1.8 % at |eta| > 2.5 — a central spike; z0 outliers are flat (~4.5 %) with a rise at
  |eta| > 2.5.
* Hit-order features (60k outlier vs 60k core tracks, scratchpad `outlier_features.py`):
  the stored hit order is scrambled for **all** tracks — only ~72 % of adjacent hit pairs
  increase in `s`, and the first token is the innermost hit for just 8 % of core tracks
  (20 % of outliers). The flat store sorts hits by the campaign's `time`
  (`scripts/preprocess_flat.py:239`, `np.lexsort((htime[hk], pk))`), which per
  `BUGREPORT_drift_beamspot_hit_time.md` is 57 % exact zeros (strip volumes) and an
  unreferenced Geant4 timestamp on pixels. Outliers are enriched in tracks with any
  negative-time hit (72 % vs 44 % outlier share in the balanced sample; mean min_time
  −11 ns vs −4 ns) and in low pT (pT q10 2.8 GeV vs 14.5 GeV; median 42.5 vs 57.2 GeV).
* Run b4e6d493 additionally trained on the **padded** path (`packed_batches: false` in
  its saved config) with the default `hit_time` sort key, where zero-filled pads tie with
  the 57 % zero-time hits and negative-time hits sort before the pads (the trap in the
  memory note `padded-hit-time-sort-bug`); 09c54481 is packed (`packed_batches: true`),
  so it is affected only through the store's time ordering. Both show the same tail
  fractions, so the ordering of the *store* is the common factor.

None of this is a metric bug; it is the physical content the raw RMS is faithfully
reporting and the IQR is faithfully ignoring. It is, however, the actionable finding:
~5 % of double-matched single muons get a catastrophically wrong SSM fit, and the
fraction is the quantity to track.

## 6. Root cause, ranked

1. **Estimator mismatch, not a computation error — confidence: very high (numerically
   verified).** `model.py:1009` logs the raw RMS; the offline "RMSE" that agrees with
   IQR/1.349 is `eval_utils.iterative_rms_convergence` (`eval_utils.py:322`), a clipped
   estimator. Both paths compute both statistics from identical residuals; only the
   naming/choice differs. Raw/IQR-σ ratios agree between Comet and the offline
   `rms_summary.txt` to a few percent for all five parameters.
2. **Magnitude of the gap: a fixed ~4–6.5 % catastrophic-residual population on the
   drift_beamspot models — confidence: high for existence and size; medium for the
   specific origin.** Carried 99–100 % of Σr²; labels validated by the CKF; central-eta
   theta spike; enriched in negative-time hits and low pT; store hit order scrambled by
   the broken `time` column (`preprocess_flat.py:239`). Needs its own investigation.
3. Ruled out (each checked in code and/or numerically): units/scale, normalised vs
   physical, phi wrap, DM vs all subset, bias (std == RMS), Lightning aggregation
   (≤ 2.4 %), `reduce_fx`/`on_epoch` mismatch, head-slice/parameter-order mismatch, IQR
   on predicted quantiles, mask dtype, autocast, target mutation, Comet name handling.

## 7. Proposed minimal fix (not applied)

Keep the raw `ssm_rms_dm` (it is the number that exposed the failure population) and
add the estimator the offline reports use, plus the clipped fraction, so the training
curves are directly comparable to `stats.txt` / `rms_summary.txt` and the tail is
visible as a fraction rather than as an unexplained 500x.

Verified on the same sample (scratchpad `verify_fix.py`): per-batch clipped RMS, mean
over 417 batches, vs pooled `iterative_rms_convergence`: d0 0.03209/0.03209,
z0 0.1996/0.1996, phi 1.028/1.028, theta 0.7317/0.7332, qop 0.00115/0.00115 (ratio
0.998–1.000); clipped fraction 4.23/4.60/4.40/6.49/5.52 % vs 4.23/4.60/4.39/6.46/5.52 %.

```diff
--- a/src/track_regression/model.py
+++ b/src/track_regression/model.py
@@ module level, near the other helpers
+def _iter_clipped_rms(r: Tensor, n_sigma: float = 3.0, max_iter: int = 5) -> tuple[Tensor, Tensor]:
+    """torch port of ``eval_utils.iterative_rms_convergence``.
+
+    Up to ``max_iter`` passes of clipping at mean ± n_sigma·std (population std, as
+    np.std), then RMS = sqrt(mean(x²)) of the survivors.  Also returns the clipped
+    fraction.  Val/test only: the ``int(keep.sum())`` forces a host sync per pass.
+    """
+    data, prev_n = r, -1
+    for _ in range(max_iter):
+        mu, sd = data.mean(), data.std(unbiased=False)
+        keep = (data - mu).abs() <= n_sigma * sd
+        n_kept = int(keep.sum())
+        if n_kept == prev_n or n_kept < 2:
+            break
+        prev_n = n_kept
+        data = data[keep]
+    return torch.sqrt((data ** 2).mean()), r.new_tensor(1.0 - data.numel() / r.numel())
+
@@ class TrackRegressionWrapper._log_metrics docstring
         - ``{stage}/{name}/ssm_rms_dm {unit}``: RMS of the residuals on the
-          double-matched subset — val/test.
+          double-matched subset — val/test.  **Un-clipped**: with a few % of
+          catastrophic residuals this sits 10–1000x above ``ssm_iqr_dm``; that
+          is the tail, not a units problem.
+        - ``{stage}/{name}/ssm_rms3s_dm {unit}``: iterative 3σ-clipped RMSE,
+          the estimator ``paper_plots``/``fast_rms_eval`` report as "RMSE";
+          comparable to ``ssm_iqr_dm`` (within ~1.6x when the core is Gaussian).
+        - ``{stage}/{name}/ssm_tailfrac_dm``: fraction of DM tracks removed by
+          that clip — the quantity that drives raw RMS >> IQR.
@@ inside the ``if ssm_residual_dm.numel() > 1:`` block, after the raw RMS log
                             rms = torch.sqrt((ssm_residual_dm ** 2).mean())
                             self.log(
                                 f"{stage}/{name}/ssm_rms_dm {unit}",
                                 rms * scale,
                                 sync_dist=True,
                             )
+                            # Iter-3σ-clipped RMSE + clipped fraction (see docstring)
+                            rms3s, tail_frac = _iter_clipped_rms(ssm_residual_dm)
+                            self.log(
+                                f"{stage}/{name}/ssm_rms3s_dm {unit}",
+                                rms3s * scale,
+                                sync_dist=True,
+                            )
+                            self.log(
+                                f"{stage}/{name}/ssm_tailfrac_dm",
+                                tail_frac,
+                                sync_dist=True,
+                            )
```

Cost: 5 small reductions + 5 host syncs per parameter per validation batch; nothing in
the training step (the DM block is val/test only). Alternative (not recommended):
replace the raw RMS with the clipped one — it would hide the 5 % failure population
that the current metric correctly flags.

## 8. Follow-ups outside this audit's scope

* Investigate the ~5 % catastrophic-residual population directly (section 5 has the
  discriminating features): store hit ordering by the broken campaign `time`
  (`preprocess_flat.py:239`) is the common factor across packed and padded runs;
  sorting by `s` (strictly positive, monotone along the trajectory) or a corrected time
  would remove both that and the padded-path pad-interleave trap in one move.
* `finetune_ssm_cls_muon_kf_hits.yaml`-derived configs on the flat store default to
  `packed_batches: false` (padded) unless `ssm_cls/base.yaml` is picked up; b4e6d493 ran
  padded with `hit_time` sort. Consider making `packed_batches: true` explicit.
* `eval_plots/README.md` says "every other model here is packed": true for 09c54481
  (`packed_batches: true` in its saved config) but the newest finetune (b4e6d493) is padded.

## Appendix: files and lines referenced

* `src/track_regression/model.py`: `_shared_step` 849–898; `_PRECISION_UNITS` 900–906;
  `_log_metrics` 910–1015 (residual 947, phi wrap 950–952, DM mask 975–979, DM residual
  982–987, std 989–995, IQR 996–1007, RMS 1008–1014); `test_step` 1025–1048; sort key
  537–538.
* `src/track_regression/losses.py`: `_linear_normalise` 63, `_linear_denormalise` 68;
  `QuantileLoss.forward/predict` 246–263; `EtaQuantileLoss.predict` 359; `CircularPhiLoss.predict`
  437; `TrackParameterLoss.forward` 989–1092, `predict` 1189, `predict_physical` 1212.
* `src/track_regression/eval_utils.py`: `iterative_rms_convergence` 322–368;
  `write_residual_statistics_report` IQR line 587.
* `src/track_regression/paper_plots/stats.py`: 24, 35–37, 40–43, 109–113.
* `src/track_regression/scripts/fast_rms_eval.py`: 110–158 (residuals), 162–167, 305–313.
* `src/track_regression/flat_data.py`: `_pack` 150–176 (`acts_dm_mask` 172), `FlatTrackDataset` 270–292.
* `src/track_regression/data.py`: `_setup_flat` 938–960, `_flat_eval_dataloader` 1013–1028.
* `src/track_regression/scripts/preprocess_flat.py`: 239 (time-sorted CSR).
* Reference numbers: `eval_plots/09c54481_4L_d128_muon/rms_summary.txt`,
  `eval_plots/CROSS_MODEL_SUMMARY.txt`,
  `/eos/project/e/end-to-end-colliderml/data/arxiv_retraining/finetune_ssm_cls_muon_kf_hits_f93223a9/stats.txt`,
  `BUGREPORT_drift_beamspot_hit_time.md`.
* Reproduction scripts (scratchpad, session-local): `repro_rms_iqr.py`, `outlier_features.py`,
  `verify_fix.py`.
