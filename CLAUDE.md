# CLAUDE.md — SSM track regression, campaign 2 (drift_beamspot / ICLR retraining)

Working notes for the campaign-2 retraining study. Written 2026-08-25 from the
repo, the draft (`/eos/user/j/jorenusc/status_updates/SSM/Arxiv_2026_SSM_Tracking.pdf`)
and the datasets themselves. Every number below was measured from the files
unless marked *(draft)*. Update this file as decisions are made.

Related reports produced the same day:
- `docs/AUDIT_comet_rms_iqr.md` — Comet RMS-vs-IQR audit (no bug; see §5.1).
- `docs/HIT_SORTING_ACTS_vs_radial.md` — ACTS hit-ordering study (see §5.2).
- `BUGREPORT_drift_beamspot_hit_time.md` — why `tracker_hits.time` is unusable.

---------------------------------------------------------------------------

## THE FINAL RECIPE (canonical, 2026-09-04) — everything needed to reproduce or deploy the best model

This section is self-contained; the rest of this file is the experiment log that
established each ingredient (references in brackets). `README.md` carries the
public version of the same recipe.

**Best models (physics identical to ±0.005 GM5; both at or below the truth-KF
on every parameter of every test set, pre-clip included):**

| model | architecture | checkpoint | config (`src/track_regression/config/ssm_cls/`) | deployed throughput (1×H100 NVL) |
|---|---|---|---|---|
| **R2L-FT** — the paper model (user decision 2026-09-02) | 2 layers, dim 128, d_state 64, headdim 32, expand 2, d_conv 4 — 0.65 M params | `eval_plots/sweep7/R2LFT/ckpts/model.ckpt` | `ICLR_sweep7/R2LFT_qrel_2L_mix3_muonhybrid_ddp2_bs40k_wsd50.yaml` | 1.48 M tracks/s |
| **R2Lnoconv-FT** — the deployment twin (finished 2026-09-04, §4.31) | same, but `d_conv: 1` (no depthwise conv) | `eval_plots/sweep7/R2LnoconvFT/ckpts/model.ckpt` | `ICLR_sweep7/R2LnoconvFT_qrel_2L_dconv1_mix3_muonhybrid_ddp2_bs40k_wsd50.yaml` | **1.89 M tracks/s** |

**Representation (the precision comes from here, not from capacity):**
1. **ACTS three-point seed** (`seed.py` / `seed_torch.py`): exact port of
   `Acts::estimateTrackParamsFromSeed`, three pixel space points, largest radial
   lever arm, **Bz = 3.0 T** (§4.8 — 2 T is wrong), transported to the beamline
   perigee. Inputs x, y, z, volume_id only. At inference the seed is computed on
   the GPU **inside `TrackParameterRegressor.forward`** automatically when a
   packed batch arrives with 12 features (`out["seed"]` exposed).
2. **15 hit features** = 12 absolute (x, y, z, r, φ_hit, θ_hit, s, volume_id,
   layer_id, surface_id, detector, η_hit) + 3 residuals to the seed helix:
   asinh(du/0.1 mm), asinh(dv/0.1 mm), s_helix (§4.10 — the single largest
   ingredient). Fourier scales 2⁻¹⁰…2⁵ (16 scales; removing Fourier breaks q/p,
   §4.20 X).
3. **Seed-anchored quantile heads, all loss weights 1.0** (§4.14): d0 ±0.4 mm,
   z0 ±3.5 mm, θ ±0.01 rad, φ ±0.015 rad (Δφ wrapped), and the **scale-free q/p
   head** — target `(q/p − seed_qop)/(|seed_qop| + 0.02)`, range ±2
   (`scale_anchor_eps: 0.02`, §4.20 Y — closes low-pT/hadron q/p; an absolute
   head stalls 5–10 % high). 7-quantile pinball ladders everywhere.
4. **Hit order = detector geometry order** (`hit_sorting.geometry_order`, D.2):
   truth-free, equals the ACTS sim-time order on ≥ 99.7 % of tracks. The packed
   encoder never re-sorts — train and inference must feed the same order.

**Data:** v2 stores (§4.17): 3 T perigee targets, |d0| ≤ 7.1 mm, |z0| ≤ 270 mm,
|η| ≤ 3, 6–20 hits, ttbar 1–110 GeV; tables joined by id **values**, never rows
(§0.4). Training store = `/scratch/colliderml/ICLR_retraining_v2_mix3`
(387,290,426 train tracks = uniform-pT muons + log-pT muons + ttbar ≥ 1 GeV).
Eval farm = `/scratch/colliderml/ICLR_eval_v2` (six test sets, truth-KF
side-cars).

**Training (two stages, strict IEEE fp32 = `TRK_MATMUL_PRECISION=highest`):**
1. **Base**: Lion, **batch size 2048**, OneCycle 1e-5 → 5e-5 → 1e-6, wd 1e-3,
   25 epochs on mix3 (~26 h on one H100). Small batch is load-bearing: the
   precision arrives in the anneal; 36 k-batch runs lose 15–80 % on φ/q/p
   (§4.21). Base configs: `ICLR_sweep5/YZ2Lmix3_qrel_2L_bs2048_onecycle25.yaml`
   (d_conv 4) / `ICLR_sweep7/R2Lnoconv_qrel_2L_dconv1_mix3_bs2048_onecycle25.yaml`.
2. **Fine-tune (tail cleaner)**: Muon-hybrid optimizer + WSD, DDP 2×20 k,
   50 epochs from stage-1 `last.ckpt` (same head, same ranges — never widen
   ranges on a kept head). Clipped RMSE unchanged; pushes the **pre-clip**
   ratios ≤ 1.0 everywhere (§4.24, §4.31).

**Inference/deployment kernel path** (physics-validated, §4.12/4.16/4.27):
v5pc fused Triton kernels (`KernelSwapCallback` swaps automatically at
val/test/predict) + GPU seed in-forward + `TRK_SSD_BUCKET16=1` +
`TRK_COMPILE_FRONTEND=1` + TF32 matmuls (`TRK_MATMUL_PRECISION=high`; validated
≤ 0.3 % on every test set for the 2L models, worst 100 GeV q/p). Batches
≥ 16 k tracks (knee; plateau from 64 k, 13.3 GiB at 120 k). For the small-batch
regime (≤ 4 k) use `--cuda-graph` instead (bit-exact dummy-track padding,
+34–53 % at bs 2048; incompatible with BUCKET16). `scripts/bench_infer_flat.py`
has the switches **on by default** (`--no-kernel-switches` reverts); repo-wide
training defaults are unchanged. `TRK_SSD_MERGED_BIDI=1` is a wash — leave off.

**Evaluation rules:** reference = the **production truth-KF shipped with the
data** (`truth_tracks` parquet / `truth_kf_reco.npy` side-cars) — never the
in-pipeline ad-hoc ACTS KF refit, which is miscalibrated 1.35–3.18×
(§4.29/4.30, `docs/BUGREPORT_acts_pipeline_kf.md`). **Paper numbers since
2026-09-04 (§4.32): |η| ≤ 2 only** (the truth-KF itself is miscalibrated
> ~80 GeV outside that; `TRK_ABS_ETA_MAX=2`) **and at deployment settings**
(`scripts/04b_eval_ckpt_deploy.sh`; = strict fp32 to displayed precision).
The GPU seed stays **float64** (`TRK_SEED_DTYPE`; fp32 injects the rc−R
cancellation → mm-scale anchor noise at high pT, and the seed is only 2.9 %
of the forward — §4.32). Metric = iterative
3σ-clipped RMSE with the pre-clip value and clipped fraction always alongside;
`bash scripts/04_eval_ckpt_iclr.sh <run_dir> last.ckpt <out> /scratch/colliderml/ICLR_eval_v2 <gpu>`.

**Result (post-clip SSM/truth-KF, CKF-DM subset; both best models):** GM5
0.99 / 0.99 / 0.91 / 0.97 / 0.99 on µ 2 GeV / 10 GeV / 100 GeV / uniform /
ttbar_new_pt1; worst single parameter q/p 1.01 (R2L-FT ttbar) / 1.03
(noconv-FT ttbar); pre-clip GM5 0.83–0.95 with **no parameter above 1.0**.
Absolute on uniform muons: d0 13.9 / 14.3 µm, z0 21.5 / 21.6 µm,
φ 0.168 / 0.177 mrad, θ 0.057 / 0.058 mrad, q/p 2.18 / 2.24 × 10⁻⁴ GeV⁻¹
(SSM / truth-KF, noconv-FT numbers).

---------------------------------------------------------------------------

## 0. Read this first — three findings that gate everything else

### 0.1 The flat stores are sorted by the BROKEN time, not by `s` (blocking)

Intent (user, 2026-08-25): "until the sorting question is settled, retraining
uses the legacy radial-magnitude sorting `s = sqrt(x²+y²+z²)`".
Reality on disk (`/scratch/colliderml/ICLR_retraining/*` and the `/eos` copy,
written 2026-08-21):

- `scripts/preprocess_flat.py:239` (before today's fix) sorted each track's hits
  with `np.lexsort((htime[hk], pk))`, i.e. by raw `tracker_hits.time`.
- Measured on `single_muon_uniform/test/part_0000` (20 000 tracks): stored order
  monotonic in `hit_time` for **100 %** of tracks, monotonic in `s` for **0 %**
  (ttbar: 0.6 %). 58 % of hits carry `time == 0` (all strip hits), so every
  track presents its **outer strip hits first, then the pixel hits**.
- The packed collate (`flat_data.py:_pack`) and the packed encoder path
  (`model.py:591`, `x_sort_value=None`) consume the stored order verbatim.
  Nothing re-sorts. `plot_preprocessed.py:119` documents the on-disk order as
  "by `tracker_hits.time`" — so this was known at plotting time but never
  propagated to the training path.

Consequences:
- **Every new-campaign result so far was trained/evaluated on scrambled
  sequences**: the 4L from-scratch run `09c54481` (d0 32 µm vs CKF 15 µm), the
  packed-mode legacy-checkpoint transfers in `eval_plots/CROSS_MODEL_SUMMARY.txt`,
  the warm-start attempt `b4e6d493`, and the 10L/bs2048 Lion pretrain
  `baeedc59` **still running on sess12 (L40S, epoch 13 of 50, ~6.2 h/epoch)**.
  Recommendation: stop `baeedc59` and relaunch on the re-sorted stores — user's
  call, not done here.
- The Comet audit (§5.1) independently found that the catastrophic-failure
  population driving the RMS/IQR gap is enriched at |η| < 0.5 and low pT and in
  tracks with negative-time hits — exactly the scrambled-order signature.
- Only `57dabaab` evaluated in *padded* mode with `TRK_SORT_KEY=hit_s` saw a
  correct (s) order on the new data. Its zero-shot numbers are therefore the only
  clean new-data numbers we have (see §3.5).

Fix applied today (surgical, additive):
- `preprocess_flat.py` gained `--sort-key {s,geometry,time}` (default `s`);
  the key is recorded as `hit_sort_key` in every `manifest.json` and
  `dataset_meta.json`. `geometry` is the detector order from
  `hit_sorting.geometry_keys` (§5.2). Verified on uniform shard 0: 100 % of
  tracks monotonic in the chosen key; `s` and `geometry` differ on 13 % of
  tracks, exactly as the sorting study predicts.
- Two rebuilds launched 2026-08-25 15:20 (each ~40 min, 40 workers), both
  with the join fix of §0.4, same seed/shard split as the original stores and
  the uniform set rebalanced to the identical 192/5/5 parts, each with an eval
  farm, truth-KF side-cars and `kf_baselines`:
  - **`/scratch/colliderml/ICLR_retraining_ssort/` + `ICLR_eval_ssort/`** —
    hits by `s` (the user's standing rule until the sorting decision);
  - **`/scratch/colliderml/ICLR_retraining_geom/` + `ICLR_eval_geom/`** —
    hits in geometry order (the study's recommendation).
  **Status 2026-08-25 16:05: both rebuilds complete and verified** (43 min
  each, run concurrently): uniform 191,532,752 / 4,998,235 / 4,998,269
  train/val/test (identical to the original), 20 000/20 000 tracks in the
  chosen order on train and test parts, mislabel proxy 0.00 %; ttbar 200,865
  tracks (+22 from the join fix); truth-KF side-cars attached (uniform test
  99.97 % matched; fixed-pT `part_0004` = the duplicated event range, dropped
  by design); KF baselines recomputed on the new farms
  (`eval_plots/baselines_KF_rebuilt_{ssort,geom}/`) — identical to the old
  ones to 3 s.f. (truth-KF d0 14.29 µm, z0 21.6 µm, φ 0.179 mrad …) because
  the reco columns were always joined by value; the DM count fell 2 %
  (3,491,385 → 3,419,038) because the purity/efficiency of the 4 % mislabelled
  tracks had been computed against the wrong hit set. ~4 % of the specific
  test tracks differ from the old test split (same shards, same count, but
  the events whose own hits pass `min_hits` are now the ones kept), so old
  and new eval numbers are not track-for-track comparable — they were not
  meaningful anyway. Both roots are on
  `/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_{ssort,geom}`
  (verified copies, see §7); the `/eos` `ICLR_retraining` stores are time-sorted
  AND mislabelled (§0.4) and should be marked deprecated.

### 0.4 ~4 % of muon-gun tracks in the 2026-08-21 stores carry another event's hits (blocking)

Found by the sorting study (§5.2), verified directly: the `particles` and
`tracker_hits` parquet tables of `single_muon_uniform` shard 0 hold the same
1 M events but **4.1 % of rows in a different order** (particles
`[6, 5, 0, …]`, hits `[5, 0, 6, …]`); fixed-pT shards 0.4–9 %, ttbar 0 %.
`preprocess_flat.select_shard` joined the two tables by **row index**
(`h_ev = _rowid(...)`), and with `particle_id == 0` for every gun muon each
differing row produced a track with the **targets of event A and the hits of
event B**. Counted against the raw sim hits: 3.84 / 3.20 / 3.54 % of the
2 / 10 / 100 GeV test tracks, 4.34 % of the uniform test part checked
(≈ 8 M of the 191.5 M training tracks), 0 % of ttbar. The ACTS-reco join was
already value-based, so the CKF/truth-KF columns are aligned with the
*targets* — which is exactly why the Comet audit (§5.1) saw "CKF fine, SSM
catastrophic on a fixed 4–6 % population": that population is these tracks
(plus the scrambled order). Every clipped-RMS number quoted so far on the new
data removed them as "tails".

Fix applied today: `select_shard` maps each hits-table row to the particles
row with the same `event_id` value (`h_row2p`); hits whose event has no
particles row are dropped. Proxy check (fraction of tracks whose innermost hit
is > 0.7 rad in φ from the target φ): old store 3.08 %, rebuilt stores 0.00 %
(uniform shard 0, 50 k tracks, both sort keys). Rule: never join ColliderML
tables by row — always by `event_id` / `particle_id` / `simhit_id` values.

### 0.2 `primary` means two different things in the two campaigns

Checked on the raw `particles` tables (legacy: HF `CERN/ColliderML-Release-1`
`ttbar_pu0`/`ttbar_pu200` shard 0; new: `/scratch/colliderml/drift_beamspot/ttbar/v1/runs/0`).
Per hard-scatter ttbar event, charged particles with pT > 0.5 GeV, |η| < 3:

| | legacy Release-1 | new drift_beamspot |
|---|---:|---:|
| in-acceptance particles/event, all | 666 | 145 |
| flagged `primary == True` | **103** | **46** |
| `primary`: fraction produced > 30 µm from PV | **21 %** | 0 % |
| `primary`: fraction produced > 10 mm from PV | 11.4 % | 0 % |
| `primary`: fraction with truth \|d0\| > 100 µm | 10.5 % | (beamspot-dominated) |
| prompt (< 1 µm) particles flagged **non**-primary | 0.03 % | **36 / event** |
| displaced 0.1–100 mm (B/D/K_S/Λ/τ daughters) flagged non-primary | — | 19 / event |
| non-primary produced > 100 mm (Geant4 material secondaries) | 99.6 % of non-primary | 42 / event |

- **Legacy**: `primary` = generator-level particle *including decay daughters*.
  No Geant4 material secondaries leaked (they are 99.6 % > 100 mm and flagged
  non-primary), but **~16 % of the p0 pretraining tracks and ~5 % of the p200
  fine-tune tracks are non-prompt decay products** produced mm–dm from the PV
  (V0s, conversions pointing back to the beamline). That population *is* the
  d0 tail (10.5 % of "primary" tracks with |d0| > 100 µm against a 12.5 µm
  beamspot) that forced the side branch. The user's suspicion is confirmed in
  substance: the training sets contained "secondaries" in the physics sense,
  admitted by the flag semantics, not by a preprocessing bug.
- **New**: `primary` = `parent_id == -1` only. It excludes **36 prompt
  in-acceptance tracks per event** (daughters of ρ/ω/K*/η/Δ decays at the PV —
  physically indistinguishable from direct particles) and all heavy-flavour /
  strange daughters. The new ttbar `core` sample therefore keeps 26 tracks/event
  (legacy: 71.5) and has a species/kinematics bias. For the muon guns this is
  moot (one gun particle, `parent_id == -1`).
- **Action**: replace `primary: true` in the preprocessing with a
  displacement-based definition (production vertex within ~1 µm of the PV ⇒
  prompt; optionally a separate "displaced" class up to a few mm) computed from
  `vx,vy,vz` relative to the event PV. Not done today — it changes the ttbar
  evaluation sample and should be a team decision. See reminder D.1 in §6.

### 0.3 The uniform-pT muon set has no low-pT coverage

Train-split tracks per pT bin (measured over every part/shard):

| pT [GeV] | legacy pretrain p0 (64.3 M) | legacy fine-tune p200 (118.4 M) | **new muon uniform (191.5 M)** | ratio new / legacy-pretrain |
|---|---:|---:|---:|---:|
| 0.5–1 | 18,648,755 (29.0 %) | 50,056,745 (42.3 %) | **0** | 0 |
| 1–2 | 16,887,850 (26.3 %) | 52,722,458 (44.5 %) | 1,757,020 (0.9 %) | **0.10** |
| 2–3 | 7,979,374 (12.4 %) | 10,157,494 (8.6 %) | 1,756,898 (0.9 %) | 0.22 |
| 3–5 | 7,888,931 (12.3 %) | 3,726,650 (3.1 %) | 3,517,060 (1.8 %) | 0.45 |
| 5–10 | 7,092,023 (11.0 %) | 1,137,765 (1.0 %) | 8,782,820 (4.6 %) | 1.2 |
| 10–20 | 3,684,219 (5.7 %) | 370,426 (0.3 %) | 17,566,110 (9.2 %) | 4.8 |
| 20–50 | 1,731,642 (2.7 %) | 159,781 (0.1 %) | 52,707,288 (27.5 %) | 30 |
| 50–110 | 340,579 (0.5 %) | 31,199 (0.03 %) | 105,445,544 (55.1 %) | 310 |

The gun is uniform in pT over [1, 110] GeV → 1.76 M tracks per GeV, nothing
below 1 GeV. The evaluation lives where the coverage is thinnest: the 2 GeV muon
sample sits in the 1.76 M-track bin, and 56 % of the new ttbar tracks are below
2 GeV (29 % below 1 GeV — no training data at all). The worst new-data results
are exactly there (4L on 2 GeV muons: d0 2.05× CKF, z0 4.3×; on ttbar 2.7× / 8.3×).

---------------------------------------------------------------------------

## 1. Legacy dataset accounting (campaign 1, `arxiv_retraining/`)

Source: `manifest.json`, `split.json` and every shard's
`selected_tracks/{track_targets,track_meta,track_event_idx,track_hit_offsets}.npy`
(4 × 1000 shards). Scripts: `src/track_regression/scripts/dataset_accounting_legacy.py`
and `dataset_accounting_flat.py` (~1 min each); full tables incl. n_hits
histograms and 2-D (pT × |d0|), (pT × η) counts in
`docs/dataset_accounting_2026-08-25/{legacy,current}_accounting.{txt,json}`.

Selection (all four variants): charged, `primary`, pT ≥ 0.5 GeV, |η| ≤ 3,
6 ≤ n_hits ≤ 20, |d0| ≤ 2.5 mm, |z0| ≤ 200 mm; p0 variants additionally
`hard_scatter` (vertex_primary == 1); `kf_*` variants additionally ACTS
double-matched (`kf_hits`: CKF hit set instead of truth hits).

| variant | events | tracks (train / val / test) | hits | trk/evt | hits/trk | used for |
|---|---:|---|---:|---:|---:|---|
| `p0_core_pretrain` | 1,000,000 (999,764 with tracks) | **71,451,676** = 64,309,170 / 3,569,141 / 3,573,365 | 946,304,935 | 71.5 | 13.24 | paper pretrain (all `pretrain_*` configs) |
| `p0_core_kf_hits_pretrain` | 1,000,000 | 56,489,784 = 50,842,922 / 2,821,913 / 2,824,949 | 740,316,006 | 56.5 | 13.11 | short-kernel pretrains 7f35983e (10L), 765feecc (4L) |
| `p200_core_kf_matched_finetune` | 100,000 | **131,529,506** = 118,367,570 / 6,570,184 / 6,591,752 | 1,715,256,954 | 1315 | 13.04 | paper fine-tune + all paper evaluation (DM test = 6,591,752) |
| `p200_core_kf_hits_finetune` | 100,000 | 131,529,466 = 118,367,532 / 6,570,182 / 6,591,752 | 1,686,391,880 | 1315 | 12.82 | kf_hits fine-tunes (f93223a9, ba96d05f, 386c6525, eaa7e3a1) |

Splits are 900/50/50 shards (seed 42). The draft's "~70 M / ~130 M" is right;
"event-level 90/5/5" is right.

Breakdowns (train splits):

| | p0 pretrain | p200 fine-tune |
|---|---:|---:|
| median / mean pT | 1.72 / 4.21 GeV | 1.08 / 1.45 GeV |
| \|d0\| < 30 µm | 84.2 % | **94.85 %** (the draft's "≈95 %") |
| 30–100 µm / 0.1–0.3 / 0.3–1 / 1–2.5 mm | 5.2 / 4.5 / 4.1 / 1.9 % | 2.45 / 0.88 / 1.06 / 0.76 % |
| \|z0\| < 50 / 50–100 / 100–150 / 150–200 mm | 63.2 / 29.7 / 6.5 / 0.7 % | 63.2 / 29.6 / 6.5 / 0.7 % |
| η in [-3,-2] / [-2,-1] / [-1,0] (symmetric) | 9.8 / 17.2 / 23.0 % | 16.1 / 15.9 / 18.2 % |
| `vertex_primary == 1` (hard scatter) | 100 % | **4.05 %** — 96 % of fine-tune tracks are pile-up |
| n_hits mode | 13 | 13 |

Primaries vs secondaries: the preprocessed shards cannot answer this
(`track_particle_ids` are per-event indices, not barcodes; `track_meta` holds
only `[pt, vertex_primary]`). Answered from the raw table in §0.2.

Optimizer-step accounting *(draft + configs)*: pretrain 50 epochs × 64.3 M at
bs 2048 = **3.2 B track presentations, 1.57 M steps, ~90 h on 1 H100**;
fine-tune 50 epochs × 118.4 M at effective bs 44 k (2 × 22 k) = 5.9 B
presentations, ~135 k steps, ~60 h.

Legacy in-domain reference (57dabaab, 10L/192/32, p200 kf_matched DM test,
iter-3σ RMS, SSM / CKF): d0 12.2 / 65.7 µm, z0 171 / 188 µm, φ 0.617 / 2.11 mrad,
θ 0.778 / 0.802 mrad, q/p 3.17e-3 / 3.44e-3 GeV⁻¹.

## 2. Current dataset accounting (campaign 2, `ICLR_retraining/`)

Selection `core` with the d0/z0 windows removed, `primary` (new semantics),
perigee recomputed (`perigee.py`), detector rebuilt from `volume_id`.

| dataset | events | tracks (train / val / test) | hits/trk | pT | notes |
|---|---:|---|---:|---|---|
| `single_muon_uniform` | 201,529,256 (1 µ/event) | **201,529,256** = 191,532,752 / 4,998,235 / 4,998,269 | 13.28 | uniform [1, 110] GeV | only set with `truth_tracks` (truth-KF); 252 GiB raw |
| `single_muon_2GeV` | 110,000 | 109,978 = 98,981 / 5,999 / 4,998 | 13.29 | 2 | eval (union of all parts) |
| `single_muon_10GeV` | 110,000 | 109,970 = 98,970 / 6,000 / 5,000 | 13.28 | 10 | eval |
| `single_muon_100GeV` | 110,000 | 109,958 = 98,962 / 5,998 / 4,998 | 13.28 | 100 | eval |
| `ttbar` | 7,680 | 200,843 = 167,374 / 7,296 / 26,173 | 13.21 | 28.7 % < 1 GeV, 27 % in 1–2 | eval; **26 trk/evt** (legacy 71.5, see §0.2); CKF only |

Target ranges (identical across all five, set by the generator): d0 uniform-ish
to ±7.07 mm (45.8 % beyond the old 2.5 mm window, 0.67 % inside 30 µm), z0 flat
to ±200 mm with 1.84 % beyond, η flat in ±3. The old loss norm ranges were
widened accordingly in the ICLR configs (d0 ±7.1, z0 ±270, q/p ±1.0).

Coverage grid for sufficiency studies (train split, uniform muons): a
(1 GeV × 0.5 η × 1 mm |d0|) cell at 1–2 GeV holds ≈ 21 k tracks → the RMS in
any such cell is measurable to ≈ 0.5 %, so *evaluation* statistics are never the
limit; *training* saturation is (§3.2).

Baselines on the new data (`eval_plots/baselines_KF/kf_baselines.txt`, uniform,
DM, iter-3σ): truth-KF d0 14.29 µm, z0 21.6 µm, φ 0.179 mrad, θ 0.0585 mrad,
q/p 2.24e-4; CKF within 3–9 % of that. The CKF is blind beyond |d0| ≈ 3 mm
(seed window), the truth-KF is not. Truth-KF exists for the uniform set only.

## 3. Analysis answers (A.1–A.7)

### 3.1 Legacy accounting — see §1 (and §0.2 for the primary/secondary split).

### 3.2 Data sufficiency

- Per pT bin the new set is **not** comparable at low energy: 0 tracks below
  1 GeV, 10 % of the legacy pretrain count in 1–2 GeV, 22 % in 2–3 GeV, parity
  at 5 GeV, 30–300× more above 20 GeV (§0.3). Because the sampler is uniform,
  55 % of every epoch is 50–110 GeV tracks whose fit is nearly straight-line —
  the gradient signal from the multiple-scattering regime is ~2 % of the batch.
- (pT, d0, z0, η) coverage otherwise is excellent: d0/z0/η are flat and
  identical across all five sets, so there is no d0/z0 extrapolation risk. The
  only uncovered axis is pT < 1 GeV, and 1–3 GeV is thin.
- Method to answer "is 200 M enough" (proposed, not run):
  1. **Learning curve at fixed steps.** Train the small architecture on nested
     subsets {10, 50, 191} M tracks for the *same* number of optimizer steps
     (bs 2048, e.g. 100 k steps); report iter-3σ RMS per pT bin on the 5 M test
     split. If the 191 M curve is still falling in a bin, data is the limit there;
     if 50 M ≈ 191 M, it is not. Cost ≈ 3 × 1 GPU-day, run concurrently.
  2. **Per-bin performance vs per-bin statistics.** Plot RMS(bin)/KF(bin) against
     N_train(bin) across the pT bins of one run: a monotone trend → statistics;
     flat → physics/architecture.
  3. **Reweighted sampling probe.** Same model, sampler ∝ 1/pT (log-uniform):
     low-pT tracks get 16× more presentations per epoch. Better low-pT RMS at
     equal high-pT RMS → the uniform sampling, not the data volume, is the
     problem; over-fitting on the 1.76 M-track bins shows as train/val
     divergence in that bin.
- **Measured handles (2026-08-27, from sweeps 1–2, `rms_by_pt.txt`):**
  (a) E vs B (25 % of the tracks, same steps, uniform set): the 4× cut costs
  d0 +2 % at 1–2 GeV, +3 % at 2–3, rising to +11 % at 50–110 GeV (the
  *opposite* gradient to statistics starvation); z0/θ/q/p unchanged in every
  bin (< 3 %); φ +19 % at 1–3 GeV vs +12 % above 10 GeV — a diversity /
  regularisation effect, since it appears where 105 M tracks exist too.
  (b) H's ratio to the truth-KF per bin vs training tracks per bin (1.76 M →
  105 M, 60×): d0 flat 1.06–1.15, z0 1.32 → 1.15 → 1.26, θ 1.33 → 1.10 →
  1.19, φ *rises* 1.26 → 1.92 with statistics; q/p 2.8 at 1–2 GeV vs 1.9–2.1
  above 5 GeV (anchored-head defect, G 4.7× there). No parameter tracks the
  per-bin statistics. (c) ttbar 0.5–1 GeV (no training data): H/CKF d0 1.93,
  z0 1.33, φ 1.85, θ 1.39, q/p 8.0 vs 1.15 / 1.27 / 1.31 / 1.33 / 2.8 in the
  1–2 GeV bin → extrapolation below 1 GeV costs ×1.7 on d0/φ and ×3 on q/p:
  the only clear data hole. Caveats: 20 epochs, E/B un-anchored (floor masked
  z0/θ there), the reweighted-sampling probe has not been run, and no
  per-bin train/val divergence is logged.
- **Bottom line**: 200 M is more than enough above ~5 GeV and cannot be made
  enough below 1 GeV by any training trick — that needs a production
  (log-uniform pT muon gun from 0.5 GeV, or a dedicated 0.5–5 GeV sample; muon
  guns are cheap). Request it now; it runs in parallel with everything below.

### 3.3 Pretraining strategy and the switch to large-batch fine-tuning

- **Portion**: the full 191.5 M train split, but sampled log-uniformly in pT
  (or equivalently with per-track weights ∝ 1/pT) so that each pT decade
  contributes equally; the uniform-pT epoch is the ablation. Hold out the 5 M
  val split untouched; freeze a fixed 1 M-track val subset (spread over parts,
  `max_val_tracks`) for the per-epoch physics metrics.
- **Stopping criterion (measurable, not a step count).** The legacy recipe was
  50 OneCycle epochs — its val loss was still improving 0.2 %/epoch at the end
  (0.00118 → 0.00117 over the last 5 epochs) *because* the LR was annealing, so
  the legacy curve cannot tell plateau from schedule. Use a WSD schedule in
  pretraining (warm-up, then constant LR) and two triggers:
  1. **Plateau trigger** on the physics metric: per epoch, on the fixed val
     subset, compute the iter-3σ RMS per parameter (`ssm_rms3s_dm`, the metric
     proposed in `docs/AUDIT_comet_rms_iqr.md`) and its geometric mean G over
     the five parameters. Switch when the 3-epoch relative improvement of G
     falls below 1 % (≈ 3× the epoch-to-epoch noise of G on 1 M tracks).
  2. **Probe trigger** (the honest one): every 5 pretrain epochs fork a 1-epoch
     large-batch Muon fine-tune probe from the current checkpoint (4 GPUs,
     ~1–2 h). Switch when the probe's G stops improving relative to the previous
     probe (< 1 %). This directly measures "does more small-batch pretraining
     still buy fine-tuned precision", which is the only thing we care about.
  Then run the WSD decay (≥ 15 % of the steps so far) and hand over.
- Baseline to argue from: legacy = 1.57 M small-batch steps then 135 k
  large-batch steps (12:1). On the new data the 10L bs2048 run reaches the same
  val loss (0.0331) at 1.1 M steps that the 4L bs128k Muon run reached at 75 k
  steps (0.0330) — different architectures and scrambled input, so not a clean
  comparison, but a warning that the small-batch phase may be much longer than
  necessary here. The probe trigger settles it.

### 3.4 Is single-muon pretraining right?

For: 200 M clean, unambiguous tracks (no hit-assignment noise), full geometry
and material coverage, flat d0/z0/η, truth-KF reference available — exactly
what is needed to learn a beamspot-*free* fit and to run architecture /
optimizer / batch-size studies cleanly.

Against: (a) no pT < 1 GeV and thin 1–3 GeV, where ttbar lives (§0.3);
(b) species: muons neither shower nor interact hadronically, and multiple
scattering goes as 1/(βp) — at p = 0.7 GeV a proton (β ≈ 0.6) scatters 1.7× and
a kaon 1.2× more than a muon at the same momentum, so a muon-only fitter is
mis-calibrated on low-p hadrons by construction; electrons (brems) are absent
entirely. Evidence: the 4L run is 2.1× CKF on uniform muons, 2.05× on 2 GeV
muons, 2.7× on ttbar → the species gap is real but *smaller* than the pT gap.

Verdict: **yes as stage 1**, no as the whole programme. Pretrain on muons;
the hadron/ttbar stage needs data that does not exist yet: the new ttbar has
167 k training tracks (legacy fine-tune used 118 M). Even the ~17.5 k upstream
MadGraph events would give < 0.5 M tracks. The cheapest large hadron sample is
a **pile-up-200 ttbar production**: 1 300 selected tracks/event → 10 M tracks
from ~8 k events (the legacy p200 recipe). Request it alongside the low-pT gun.
Until then ttbar is an evaluation set only.

### 3.5 Checkpoint recycling

Evidence available:
- Zero-shot 57dabaab (legacy 10L) on the new uniform set restricted to the
  legacy cuts (|d0| < 2.5, |z0| < 200; 6.19 M tracks; padded path with
  `TRK_SORT_KEY=hit_s`, i.e. correctly s-ordered): d0 41.6 µm, z0 232 µm,
  φ 2.41 mrad, θ 0.742 mrad, q/p 1.17e-3 vs truth-KF 14.2 / 21.0 / 0.176 /
  0.061 / 2.25e-4. On θ and q/p that is **equal** to the 4L trained 50 epochs
  from scratch on the new (scrambled) data (0.733 / 1.15e-3), and d0/z0 are
  within 1.3× / 1.2×. The legacy encoder transfers a lot; its d0/φ are hurt by
  the learned beamspot prior (d0 collapses to ~0 outside the old range:
  1055 µm on the full range).
- The one warm-start attempt on the new data (`b4e6d493`,
  `ICLR/finetune_10L_from57dabaab_newnorm.yaml`) is **invalid evidence**: it
  widened the loss norm ranges on the old head (the destructive ×38 trap,
  memory `norm-range-is-not-a-hard-ceiling`) *and* ran the padded path with
  the `hit_time` sort key on data where 57 % of hits tie with the zero pads.
  Its val/total 0.0376 at epoch 9 says nothing.
- Structural compatibility is clean: inputs are the same 12 hit features (no
  time as input), `detector` rebuilt with the old map, s-order agrees with the
  legacy time-order on 98 % of legacy tracks.

Recommendation: **partial warm start is worth one controlled run, from scratch
is the default.** Load `input_net`, `encoder` (all Mamba layers, CLS tokens,
norms) and the pool head from 57dabaab's *pretrain* checkpoint (or 57dabaab
itself); **re-initialise `output_head` completely** and use the new norm ranges
(legal because the head is fresh — the norm-range trap only bites when keeping
the old head). Keep `pretrained_ckpt_strict: false` and check the printed
"Missing keys re-initialised" list is exactly the output head. Run it at the
same step budget as a from-scratch run of the same architecture and compare
per-pT RMS at 5 and 20 epochs; kill if it is not ahead by epoch 5. Do not
warm-start the head, do not widen ranges on a kept head, and prefer the 4L/6L
target architecture over reviving the 10L unless the sweep says otherwise.

### 3.6 Architecture

Claim 1 — smaller than 10L/192/32: **supported.**
- Legacy depth sweep (Lion bs 2048, p0 `core`, val/total at epoch 49):
  2L 0.00151, 4L 0.00125, 6L 0.00120, 8L 0.00119, 10L 0.00117 — 4→10 layers
  buys 6 % in loss; 6→10 buys 2.5 %.
- Fine-tuned physical RMS on legacy p200 kf_hits (`logs/paper_plots/_summary/all_runs.csv`):
  4L/128/16 (1.03 M params, 386c6525) vs 10L/192/32 (5.48 M, ba96d05f):
  d0 12.65 vs 12.31 µm, z0 205 vs 197 µm, φ 0.708 vs 0.802 mrad (4L better),
  θ 0.869 vs 0.905 (4L better), q/p 3.72e-3 vs 3.44e-3. Within ±8 % everywhere
  at 5.3× fewer parameters and ~2× the inference throughput.
- KF analogy: a linear-Gaussian KF is *exact* after one forward pass plus one
  smoother pass; an iterated/non-linear KF converges in 2–3 iterations. One
  bidirectional layer already is forward+backward; depth is the iteration
  count. 2–4 bidirectional layers is the physics-motivated depth; the extra
  layers in 10L are buying non-linear residual modelling, which the numbers say
  is worth a few percent.

Claim 2 — d_state 16/32 is too small and widening is nearly free: **half
supported, untested.**
- Cost side is right for inference: `arch_sweep` (memory
  `arch-scaling-study-2026-07-25`) measured 16→32 +8 %, 16→64 +17 %,
  16→128 +68 % time-per-2k-tracks, and d_state adds few parameters
  (4L/128: 16→128 is +240 k on 1.1 M; 4L/192: 32→128 is +300 k on 2.4 M).
  Training cost at L ≤ 22 tokens is likewise small. Constraints: `d_state`
  power of 2, ≥ 16, `ngroups = 1` (fused v5pc kernel).
- Capacity side is not obvious from the KF: the recurrent state per layer is
  `expand·dim · d_state` numbers (10L/192/32 → 12 288 per direction) against
  the KF's 5 + 15 covariance entries, so the state is not small in the KF
  sense. What d_state does control is how many independent decay rates /
  memory horizons each channel can mix — plausibly useful for tracks that mix
  pixel and strip resolutions, but nothing measured yet: the only d_state=128
  run (09c54481) has no d_state=16 twin on the same data.
- Proposed sweep (all Lion, bs 2048, WSD, 100 k steps ≈ 200 M presentations on
  the s-sorted uniform set with log-uniform pT sampling; one run per H100, 6 in
  parallel, ~1 day):

  | run | L | dim | d_state | params (total ≈) | tests |
  |---|---|---|---|---|---|
  | A | 4 | 128 | 16 | 1.10 M | small baseline (legacy 4L) |
  | B | 4 | 128 | 128 | 1.34 M | d_state at fixed width |
  | C | 4 | 192 | 32 | 2.40 M | width at legacy d_state |
  | D | 4 | 192 | 128 | 2.70 M | width + state |
  | E | 6 | 192 | 64 | 3.58 M | balanced mid-size |
  | F | 2 | 192 | 128 | 1.52 M | KF-depth lower bound |
  | ref | 10 | 192 | 32 | 5.48 M | legacy shape (relaunch of baeedc59 on sorted data) |

  Decision metric: per-pT-bin iter-3σ RMS on the 5 M test split (geometric
  mean over parameters, reported per bin), plus t2k inference time. Confirm the
  best two at full pretrain length.

### 3.7 The straight answer

**Invest in retraining now — but not on the current stores, and while two
targeted productions are requested in parallel.** Reasoning:

1. There is no clean baseline to "wait from": every campaign-2 result so far
   was trained on time-scrambled sequences (§0.1) with ~4 % of tracks carrying
   another event's hits (§0.4). A 40-minute rebuild fixes both; until the
   rebuilt 4L baseline exists, nothing about the new data can be concluded,
   including the legacy-checkpoint transfer numbers.
2. The existing 200 M muons fully support the methodological questions the
   paper needs answered — beamspot-free d0/φ (the unified network without the
   side branch), architecture size, the batch-size constraint, the
   pretrain→fine-tune trigger — at pT > 3 GeV. All of that transfers unchanged
   to enriched data.
3. What the data cannot do is the low-pT (< 1 GeV) and hadron/ttbar claims.
   Those need (a) a low-pT-enriched muon gun and (b) a pile-up ttbar sample
   (§3.2, §3.4). Both are cheap to generate relative to the 200 M set already
   made; request them today and they arrive while the sweep runs.

What would change my mind (to "wait"):
- the re-sorted 4L baseline (E1) does not beat the scrambled 09c54481 — then
  hit order is not the limiter and something deeper (targets, features) is
  wrong and must be found before spending GPU-weeks;
- the learning curve (E4) saturates by 50 M tracks in every pT bin — then more
  of the *same* data is worthless and only the two productions matter, so the
  sweep should shrink to E1+E2 and stop.

On the batch-size-2048 constraint: treated as a hard requirement throughout
(§4 gets parallelism from concurrent runs). I do think it is worth **one**
cheap re-test, because the evidence was gathered on a different regime
(ttbar, pT peaked at 1–2 GeV, time-ordered hits, ±2.5 mm d0) and is not
reproducible from this repo (the sweeps live in the R&D repo), and because the
cost of being wrong is 10–50× pretraining throughput: run E3 below. If it
confirms the constraint, it is one GPU-day; if it refutes it on this data, the
whole plan gets cheaper.

## 4. Experiment plan — SWEEP 1 (user-defined 2026-08-25, supersedes the E1–E7 draft)

Six single-GPU pretraining trials, one per GPU (this machine GPUs 0–3, the
other machine GPUs 0–1). Configs, README and launch scripts:
`src/track_regression/config/ssm_cls/ICLR_sweep1/` (own `base.yaml` copy).
Data: `ICLR_retraining_geom/single_muon_uniform` (the launch scripts copy it
from /eos if it is missing on that machine's /scratch). Scrambling is **not**
tested any more (decided); the 6L architecture slot was replaced by a low-pT
data-sufficiency run at the user's request.

| trial | GPU | network | recipe | batch | epochs / steps | question |
|---|---|---|---|---|---|---|
| A | other:0 | 4L/128/**ds64** | Lion + OneCycle (legacy) | 2 048 | 20 / 1.87 M | small-batch reference |
| B | sess3:0 | 4L/128/ds64 | Lion + OneCycle, LR ×4.4 | 36 000 | 20 / 106 k | is the small batch still necessary? (vs A) |
| C | other:1 | 10L/192/**ds128** | Lion + OneCycle | 2 048 | 20 / 1.87 M | what does the 5× network buy? (vs A) |
| D | sess3:1 | 4L/128/ds64 | Lion + WSD (5/70/25 %) | 36 000 | 20 / 106 k | schedule at large batch (vs B) |
| E | sess3:2 | 4L/128/ds64 | as B, `max_train_tracks` = 25 % | 36 000 | 80 / 106 k | low-pT data sufficiency: per-pT-bin RMS vs B |
| F | sess3:3 | 4L/128/ds64 | Muon-hybrid + WSD (09c54481 recipe) | 36 000 | 20 / 106 k | fine-tune optimizer from scratch (vs A/B/D) |

Comparison unit = equal epochs (3.83 B presentations); A/C run ~18× more steps
by construction. Read-out: `scripts/04_eval_ckpt_iclr.sh` → `rms_summary.txt`
+ **`rms_by_pt.txt`** (per-pT-bin iter-3σ RMSE, new) and the per-epoch
`val/<p>/ssm_rms3s` curves in Comet (`SW1-*`). Decision rules:
- B ≈ A (within a few % in every parameter and pT bin) → the bs-2048
  constraint is gone on this data; if F ≈ A as well, F is the whole recipe
  (no pretrain/fine-tune split needed).
- E worse than B only in the 1–3 GeV bins → low-pT is data-limited → the
  low-pT gun production is what moves it; E ≈ B everywhere → more of the same
  data is useless there.
- C vs A → whether the paper needs the 10L at all.

Dry-run facts (2026-08-25, 60 steps each): 40 000 tracks/step peaked at 90.6 of
95.8 GB → batch set to 36 000; 1.1 steps/s → ~75 min/epoch → ~25 h per
large-batch trial; A ~30–40 h; C ~2.7 days on an H100. `max_train_tracks` was
added to the DataModule for E (spread over all parts).

**Launch status.** B, D, E, F launched on sess3 2026-08-25 23:43–23:44 via
`launch_sess3_gpus0-3.sh` (one process per GPU, pinned with
`CUDA_VISIBLE_DEVICES` + `--trainer.devices 1`, `TRK_MATMUL_PRECISION=highest`,
nohup): pids 1094567 (B, GPU 0), 1096239 (D, GPU 1), 1098479 (E, GPU 2),
1100661 (F, GPU 3); logs `launch_logs/sweep1/<trial>_20260825_2343*.log`;
Comet `SW1-*` in `ssm-track-regression-iclr`. Verified 23:48: all four in
epoch 0 at 2.1–2.4 steps/s (the dry run's 1.1 was compile warm-up), 83–84 GB
and 99–100 % utilisation per GPU → ~40 min/epoch, i.e. ~14 h for the 20-epoch
trials and ~13 h for E's 80 short epochs — done by mid-day 2026-08-26. Health
check: `nvidia-smi` (4 compute processes, one per GPU) and the
`[val epoch N] iter-3σ RMSE` lines in the logs once per epoch.

**Other machine = sess5 (2 × H100 NVL).** A and C launched 2026-08-26 01:25
via `launch_other_gpus0-1.sh`: the geometry store was copied /eos → 
`/scratch/colliderml/ICLR_retraining_geom/single_muon_uniform` first (2 433
files, 154.8 GB, 135 s at 1.09 GB/s, per-file verified), then A on GPU 0
(pid 16378, `launch_logs/sweep1/A_4L_ds64_lion_cosine_bs2048_20260826_012508.log`)
and C on GPU 1 (pid 16959, `..._C_10L_ds128_lion_cosine_bs2048_20260826_012528.log`).
Verified 01:30: A at 26.4 steps/s (93,521 steps/epoch → ~59 min/epoch →
~20 h for 20 epochs; GPU 0 only 72 % utilised, 5.8 GB — the price of bs 2048),
C at 5.4 steps/s (→ ~4.8 h/epoch → ~4 days for 20 epochs; GPU 1 91 %, 20 GB).
`baeedc59` was not running on sess5 (both GPUs idle at launch; it ran
elsewhere). Note: sess5's `/scratch` root itself holds the OLD time-sorted
uniform store (`/scratch/{train,val,test,dataset_meta.json}`, 137 GB, no
`hit_sort_key`) that `baeedc59` trained on — deprecated, safe to delete.
The repo lives on `/shared` (NFS, visible from both machines), so the same
scripts/configs apply; only `/scratch` is machine-local.

### 4.1 Sweep-1 first results (2026-08-26 13:30; B/D/E/F finished, A at epoch 13, C at epoch 3)

Eval bundles: `eval_plots/sweep1/<trial>/` (`rms_summary.txt`, `rms_by_pt.txt`).
Uniform muons, iter-3σ RMSE, SSM / truth-KF (all fitted tracks):

| trial (epoch) | d0 µm | z0 µm | φ mrad | θ mrad | q/p GeV⁻¹ | geo-mean ratio |
|---|---|---|---|---|---|---|
| truth-KF | 14.3 | 21.6 | 0.179 | 0.058 | 2.24e-4 | 1 |
| B  Lion cosine bs36k (18) | 25.5 | 121 | 0.757 | 0.535 | 5.00e-4 | 4.0 |
| D  Lion WSD bs36k (18) | 24.1 | 185 | 0.862 | 0.776 | 5.30e-4 | 4.8 |
| E  = B, 25 % tracks (79) | 28.4 | 122 | 0.849 | 0.525 | 5.19e-4 | 4.2 |
| F  Muon WSD bs36k (18) | 27.6 | 117 | 0.851 | 0.518 | 6.67e-4 | 4.4 |
| A  Lion cosine bs2048 (12, val subset) | 20.3 | 184 | 0.876 | 0.683 | 4.39e-4 | 4.2 |
| old 09c54481 (scrambled+mislabelled, 50 ep) | 32.1 | 200 | 1.03 | 0.733 | 1.15e-3 | — |

Per-pT (trial B, uniform, SSM / truth-KF): 1–2 GeV 90/58 µm, 393/158 µm,
3.95/1.84 mrad, 1.86/0.62 mrad, 4.8e-3/2.6e-3 (ratios 1.6 / 2.5 / 2.1 / 3.0 /
1.9); 50–110 GeV 24.6/13.0, 117/19.2, 0.70/0.15, 0.51/0.04, 4.4e-4/1.9e-4
(ratios 1.9 / 6.1 / 4.8 / 12 / 2.4). Fixed-pT sets: 2 GeV 66/46, 296/119,
2.9/1.4, 1.40/0.47, 3.4e-3/1.9e-3; 100 GeV 24.5/12.8, 116/18.8, 0.69/0.14,
0.50/0.04, 4.4e-4/1.8e-4. ttbar (CKF ref) 71/42, 293/76, 3.4/1.25, 1.95/0.51,
4.3e-3/2.1e-3 (was 112/628/4.05/2.43/5.9e-3 for the scrambled 4L).

Readings:
0. **Training-length caveat (user, 2026-08-26).** On the legacy data the SSM
   only matched the CKF after the full 50-epoch bs-2048 pretrain plus 50–100
   fine-tune epochs, and the precision "kicked in" late. The legacy 10L
   pretrain val/total fell 66 / 28 / 19 / 14 % per successive 10-epoch decade
   and 0.8 % over the last 5 epochs; the in-domain controls show the 50-epoch
   *pretrain alone* already at the final precision (10L pretrain-only 2787864
   d0 12.3 / z0 172.6 µm vs fine-tuned 57dabaab 12.2 / 171.2; 4L 1e0f5105
   12.5 / 180.5) — the fine-tune added a few %. Sweep 1 ran 20 epochs, and
   B's late-anneal gains (epochs 12→18, ratio to truth-KF) were d0 2.8→1.8,
   **z0 28→5.8, θ 36→9.3**, φ 12→4.5, q/p 6.1→2.3 — i.e. concentrated in
   exactly the "floor" parameters. So the pT-flat floor has TWO candidate
   causes that the 20-epoch data cannot separate: (a) the absolute-coordinate
   representation (item 1), (b) unfinished annealing — SGD/Lion weight noise
   at the end of a 20-epoch cosine is also pT-independent. Evidence for (b):
   D (25 % cooldown) ends 50 % worse than B on z0/θ. The seed anchor (G/H)
   helps under both (it rescales output noise by 35–80×), so it is the right
   fix either way but not a clean discriminator; the discriminator is a
   **long-anneal / two-stage run** (next runs, §4.4). Also keep the bar in
   view: the legacy target was the CKF at 1–2 GeV (d0 66, z0 188 µm); today's
   truth-KF at 50–110 GeV is 13 / 19 µm — 5–10× tighter, so "beat the CKF on
   legacy" (network at 12 / 170 µm) would not beat this reference on z0/θ.
1. **The SSM has a pT-independent precision floor.** Above ~10 GeV its
   resolution stops improving (B: d0 26→25→25 µm, z0 126→118→117 µm,
   θ 0.55→0.51→0.51 mrad for 10–20 / 20–50 / 50–110 GeV) while the truth-KF
   keeps improving with pT (d0 17→14→13, z0 29→22→19, θ 0.092→0.057→0.042). The
   gap is therefore 1.6–3× at 1–3 GeV (multiple-scattering regime, the physics
   problem) but 5–12× at 50–110 GeV (the network's absolute-precision problem:
   21 µm on a ±270 mm z range is 4e-5 relative; the finest Fourier scale is
   2⁵ over the full range; fp32 pipeline). Same floor in every trial, so it is
   neither optimizer, batch size nor data. The legacy campaign never exposed it
   because at 1–2 GeV the KF itself was at 170 µm z0 — above the floor.
   → **Next experiments should attack the floor**: relative/local hit
   coordinates (hits relative to the first hit or to a 3-hit seed), finer
   Fourier scales, a residual-to-analytic-seed formulation, and an fp64 probe
   run to separate arithmetic from representation.
2. **Data (E vs B):** at equal steps, 4× fewer distinct tracks costs ~10 % on
   d0/φ *uniformly in pT* (1–2 GeV: 91.8 vs 90.3 µm d0; 50–110: 27.4 vs 24.6)
   and nothing on z0/θ. No sign of a low-pT-specific data limitation at this
   training length; the low-pT gun is still needed for coverage below 1 GeV,
   not for statistics in 1–3 GeV.
3. **Batch size (A vs B):** undecided until A finishes (A at epoch 12 is ahead
   on d0/q/p, behind on z0/θ/φ; it still has the whole cosine tail).
4. **Schedule/optimizer:** WSD runs plateaued high until the cooldown; D ends
   50 % worse than B on z0/θ (too-short or too-hot plateau). Muon (F) is
   fastest early, equal at the end except q/p (+33 %).
5. Data fixes alone (order + labels) took the 4L from 32/200/1.03/0.73 µm-mrad
   (50 epochs, d_state 128) to 25/121/0.76/0.54 in 20 epochs.

### 4.2 The ACTS seed, ported (2026-08-26) — and what it says about the floor

`src/track_regression/seed.py` (+ `tests/test_seed.py`, 5 tests): exact port of
`Acts::estimateTrackParamsFromSeed` (conformal-map circle through three space
points, sinc-corrected dz/ds, q/pT = 2·bOverS/|B|) with the
`TruthSeedingAlgorithm` triplet rule (pixel space points, volumes 16–18, largest
radial lever arm; fallback to all hits) and transport to the beamline perigee
via `perigee.truth_perigee`. Inputs: x, y, z, volume_id and the constant 2 T
field — nothing else; numpy vectorised over padded batches (0.5 M tracks/s) and
a torch twin. Synthetic exact helices are reproduced to 1e-6 mm / 1e-8 rad.
The flat collate (`flat_data._pack`) now emits `seed_<p>` per track, usable as
`delta_anchor` for every head (θ must then be a plain `quantile` head: the
anchor is subtracted before the θ→η transform).

**Seed resolution alone** (uniform test part, iter-3σ, SSM sweep-1 B and
truth-KF for comparison):

| pT [GeV] | d0 seed / B / tKF [µm] | z0 seed / B / tKF [µm] | φ seed / B / tKF [mrad] | θ seed / B / tKF [mrad] | q/p seed / B / tKF |
|---|---|---|---|---|---|
| 1–2 | 70 / 90 / 58 | 194 / 393 / 158 | 2.4 / 3.95 / 1.84 | 0.79 / 1.86 / 0.62 | 0.20 / 4.8e-3 / 2.6e-3 |
| 5–10 | 36 / 30 / 22 | 50 / 145 / 43 | 1.00 / 1.15 / 0.47 | 0.18 / 0.64 / 0.15 | 0.042 / 1.0e-3 / 6.1e-4 |
| 50–110 | 33 / 25 / 13 | 26 / 117 / 19 | 0.87 / 0.70 / 0.15 | 0.053 / 0.51 / 0.042 | 7.7e-3 / 4.4e-4 / 1.9e-4 |

Above 10 GeV the three-point seed beats the trained network by 4–10× on z0 and
θ and matches it on d0/φ; only q/p (140 mm lever arm) is where the network adds
value. A network that cannot out-resolve a three-point analytic fit on z0/θ is
not limited by physics or data — it is the absolute-coordinate representation.
Seed residual ranges (|truth − seed|, 99.99 %): d0 0.40 mm, z0 3.4 mm, φ 14
mrad, θ 10 mrad, q/p 0.48 e/GeV → the anchored heads' norm ranges (35–80×
narrower than the absolute ones). ttbar seeds have km-scale outliers on
pT < 1 GeV kinks (max d0 25 mm) — harmless for an unclipped pinball loss.

### 4.3 SWEEP 2 (2026-08-26, this machine, GPUs 0–3) — `config/ssm_cls/ICLR_sweep2/`

All = sweep-1 B (4L/128/ds64, Lion+OneCycle, bs 36 k, 20 ep, geometry store) + one change:

| trial | GPU | change | question |
|---|---|---|---|
| G | 0 | all heads predict target − seed (`delta_anchor: seed_<p>`, residual ranges) | head-side representation (vs B) |
| H | 1 | G + input Fourier scales 2⁻¹⁰…2⁵ (finest period ~37 mm in z vs ~1.2 m default; `fourier_encode` is sin(x/2ⁿ)) | encoder-side on top (vs G, J) |
| I | 2 | float64 end-to-end (64-true, kernel v3c), bs 18 k, LR ×2.96 | fp32 arithmetic? (vs B) |
| J | 3 | Fourier 2⁻¹⁰…2⁵ only | encoder-side alone (vs B, H) |

Decision rule: if G/H collapse the high-pT z0/θ gap toward the seed/KF while
low-pT bins are unchanged, the floor was representation; if I moves the floor,
fp32 arithmetic contributed; if nothing moves, look at the loss/head (pinball
on normalised targets) and the fp32 input quantisation next.

**Launch status.** G (pid 1919515, GPU 0), H (1920465, GPU 1), J (1921777,
GPU 3) launched 2026-08-26 14:10–14:11 on sess3 after 60-step dry runs
(82 GB at bs 36 k, ~2 steps/s → ~14 h); logs `launch_logs/sweep2/`. In the
dry run the anchored G/H sat at d0 0.22 mm / z0 1.8 mm after 60 steps where J
was at 4.4 mm / 173 mm — the heads start from seed quality as intended. I
(fp64) needed two fixes before its dry run passed: `KernelSwapCallback` now
re-casts the swapped Mamba2Short modules to `pl_module.dtype` (they were built
float32 after Lightning's `.double()`), and `TrackRegressionWrapper.
on_after_batch_transfer` casts floating batch tensors to the module dtype
(the flat loader's pre-collated float32 tensors are not converted by the
64-true plugin), and `model.py` no longer hard-casts the pooled encoder
output to float32 before the heads (it casts to the heads' dtype). All three
are no-ops in fp32. I launched 2026-08-26 (pid 1933409, GPU 2, `I_4L_ds64_lion_cosine_bs18k_fp64_20260826_141852.log`):
2.67 steps/s at bs 18 k in the real run (the dry run's 1.25 was compile
warm-up), 81.5 GB, 77 % utilisation → 10,639 steps/epoch ≈ 66 min → ~22 h for
20 epochs. All four sweep-2 GPUs verified training 2026-08-26 14:25.

### 4.4 Next runs (queued, need free GPUs): the training-length axis

- **K — long anneal.** B recipe for 50 epochs (legacy length; ~33 h at 2.3
  steps/s) with checkpoints every 10 epochs (`Checkpoint save_top_k` or a
  periodic callback), per-pT read-out at 20/30/40/50. If the high-pT z0/θ
  keep falling past epoch 20 → cause (b) dominates.
- **L — two-stage (legacy curriculum).** Muon-hybrid + WSD continuation
  ("fine-tune") from B's `last.ckpt` for 20–50 epochs on the same muon set
  (there is no separate fine-tune set yet), bs 36 k, fresh optimizer,
  `pretrained_ckpt_path` + original norm ranges (the head is kept — do NOT
  widen ranges). Compare per-pT vs B at epoch 19.
- Same two continuations from **A** (bs 2048) once it finishes, to keep the
  batch-size question attached to the long-training regime.
GPU availability: sess3 GPUs 0/1/3 free ~03:30 2026-08-27 (G/H/J), GPU 2
~noon (I); sess5 GPU 0 free ~21:30 tonight (A), GPU 1 in ~3 days (C).

### 4.5 Sweep-2 results + A (2026-08-27; full inference, `eval_plots/sweep2/`)

All at epoch 19 (`last.ckpt`, validated). Uniform muons, iter-3σ RMSE, SSM / truth-KF
(3.42 M DM tracks); ratio in brackets:

| run | d0 µm | z0 µm | φ mrad | θ mrad | q/p GeV⁻¹ | geo-mean |
|---|---|---|---|---|---|---|
| seed alone (3 pixel pts) | 34 (2.4) | 27.5 (1.3) | 0.89 (5.0) | 0.067 (1.1) | 1.06e-2 (46) | — |
| B  bs36k (ep 18) | 25.5 (1.8) | 121 (5.6) | 0.76 (4.2) | 0.54 (9.2) | 5.0e-4 (2.2) | 4.0 |
| J  Fourier 2⁻¹⁰ only | 25.3 (1.8) | 110 (5.1) | 0.77 (4.3) | 0.57 (9.9) | 4.7e-4 (2.1) | 3.8 |
| I  fp64 | 21.5 (1.5) | 87 (4.0) | 0.61 (3.4) | 0.45 (7.7) | 4.1e-4 (1.8) | 3.1 |
| G  seed anchor | 23.2 (1.6) | **27.5 (1.27)** | 0.56 (3.1) | **0.067 (1.16)** | 2.0e-3 (8.9) | 2.3 |
| A  bs 2048, un-anchored | 17.9 (1.25) | 46 (2.15) | 0.38 (2.1) | 0.22 (3.7) | **3.2e-4 (1.44)** | 2.0 |
| **H  anchor + Fourier** | **16.1 (1.13)** | **26.4 (1.22)** | **0.32 (1.78)** | **0.067 (1.16)** | 4.6e-4 (2.1) | **1.43** |

Fixed-pT / ttbar (SSM / ref; ref = truth-KF, CKF for ttbar), H then A:
2 GeV H 50/46, 155/119, 1.71/1.38, 0.60/0.47, 4.8e-3/1.9e-3 — A 62, 169, 2.05, 0.72, 2.8e-3;
10 GeV H 20.7/19.2, 41/35.5, 0.53/0.37, 0.133/0.120, 9.1e-4/4.8e-4 — A 21.1, 60, 0.56, 0.25, 6.3e-4;
100 GeV H 14.9/12.8, 23.7/18.8, 0.28/0.14, 0.047/0.040, 3.8e-4/1.8e-4 — A 17.0, 43, 0.34, 0.19, 2.7e-4;
ttbar H 50.5/41.5, 95/76, 1.70/1.25, 0.65/0.51, 6.2e-3/2.1e-3 — A 63, 147, 2.12, 0.91, 3.4e-3
(B was 71/293/3.44/1.95/4.3e-3). Per-pT (H, uniform): d0/z0/θ within 1.1–1.3× of the
truth-KF in EVERY bin (z0 208→24 µm following the KF 158→19), φ 1.3–1.9×, q/p 2.1–2.8×.

**Per-domain geometric means of the ratios** (2026-08-27, user request; full table
`eval_plots/sweep2/geomean_ratios.txt`; GM5 = all five parameters, GM4 = without q/p):

| dataset | H GM5 / GM4 | A GM5 / GM4 | B GM5 / GM4 | H's worst parameters |
|---|---|---|---|---|
| µ 2 GeV | 1.41 / 1.23 | 1.45 / 1.45 | 2.09 / 2.19 | q/p 2.46, z0 1.30, θ 1.28 |
| µ 10 GeV | **1.30 / 1.18** | 1.49 / 1.55 | 2.59 / 2.88 | q/p 1.90, φ 1.42 |
| µ 100 GeV | 1.49 / 1.36 | 2.23 / 2.45 | 4.49 / 5.21 | q/p 2.18, **φ 1.98** |
| µ uniform | 1.42 / 1.30 | 1.98 / 2.15 | 3.87 / 4.44 | q/p 2.07, φ 1.78 |
| ttbar (CKF) | 1.51 / 1.28 | 1.69 / 1.72 | 2.68 / 2.89 | **q/p 2.93**, φ 1.36 |

Reading: H is U-shaped in pT — best at 10 GeV, weakest at 100 GeV (φ 2×, q/p 2.2×:
the transverse plane, where the KF's precision comes from the strips' lever arm)
and at 2 GeV / ttbar only through q/p (2.5× / 2.9×; the anchored head — G shows
4.7× / 5.2× there, A's absolute head 1.44 / 1.58). With q/p removed the 2 GeV and
ttbar geometry parameters are already at 1.23–1.28×. Substituting A's q/p ratios
into H gives GM5 ≈ 1.27 / 1.21 / 1.39 / 1.33 (2 / 10 / 100 GeV / ttbar) — what N
(absolute q/p head) should recover; the remaining 100 GeV deficit is φ and is the
target of O/P. B/A un-anchored degrade monotonically with pT (floor); H does not.

Readings:
1. **The floor was representation, and the head-side anchor removes it**: G/H's z0
   and θ equal the seed (27.5 µm / 0.067 mrad, i.e. the network adds ≤ 4 % on
   top of the seed there) and follow the KF's pT dependence. Fourier alone (J)
   changes nothing; Fourier on top of the anchor (H vs G) buys d0 23→16, φ
   0.56→0.32, q/p 2.0e-3→4.6e-4. fp64 (I) is a uniform 10–25 % — arithmetic is
   a minor contributor.
2. **The anchored q/p head is a mistake**: G's q/p is 4× worse than B's absolute
   head (seed q/p is 46× off the KF; at 1–2 GeV the anchored q/p is 5× the
   KF); H only recovers to the absolute-head level. → absolute q/p head.
3. **Batch size 2048 matters after all (A vs B, both un-anchored, epoch 19):** A
   is 1.4–2.6× better on every parameter and has the best q/p of all runs
   (3.2e-4, 1.44×, flat in pT). Its gains came in the anneal (z0 184→47 µm
   over epochs 12→19) — the user's legacy experience reproduced. A still shows
   a (lower) pT-flat floor on z0/θ (43 µm / 0.196 mrad above 10 GeV).
4. What the network adds over the seed after anchoring: nothing on z0/θ (KF is
   1.2–1.3× better than the seed there — a learnable 20–30 %), 2× on d0, 2.7× on
   φ, 20× on q/p — its job is the transverse plane, from the strips' lever arm.
5. ttbar (species/low pT): H 1.2–1.4× the CKF on d0/z0/φ/θ, 2.9× on q/p; low-pT
   bins are now the *closest* to the KF for the geometry parameters (1.1–1.3×),
   i.e. the low-pT gap is small once the representation is right.

### 4.6 SWEEP 3 — REVISED 2026-08-27 (after the geo-mean and data-sufficiency read-outs), AWAITING USER SIGN-OFF (not launched)

Why revised. (i) q/p is the worst parameter of H in every domain (2.5× at 2 GeV,
2.9× ttbar, 2.2× at 100 GeV) and it is a known head defect (anchored q/p; A's
absolute head reaches 1.44) → the absolute q/p head becomes the **new reference
H′**, and every other run is H′ + one change, instead of five runs inheriting the
defect. (ii) The only remaining geometry gap is φ at high pT (1.98× at 100 GeV,
the transverse plane) → the encoder-side run (P′) is the one aimed at it; a
finer absolute Fourier grid (old O) is superseded — seed-residual inputs are
mm-scale, so the existing scales already resolve them at ~10 µm. (iii) The
low-pT statistics question is answered (§3.2 handles a–c: not statistics-
limited in 1–3 GeV, data hole below 1 GeV) → no data twin. (iv) A vs B (bs 2048
gains arrive in the anneal) + the legacy 50-epoch experience + the user's
"large batch first, small batch for the anneal" idea form one axis that fixes
the cost of the scaling step (10L at bs 2048 = 4 days / 20 epochs) → M′, R, Q′.

| run | GPU | recipe (all: 4L/128/ds64, seed anchors on d0/z0/φ/θ, Fourier 2⁻¹⁰…2⁵, Lion, geometry store) | reads against | question / decision |
|---|---|---|---|---|
| **H′** | sess3:0 | H + **absolute q/p head** (norm ±1.0, no anchor); bs 36 k, OneCycle 20 ep, ~14 h | H | new reference; expect q/p → ≤ 1.5× everywhere, GM5 ≈ 1.27 / 1.21 / 1.39 / 1.33 (2 / 10 / 100 GeV / ttbar) |
| **M′** | sess3:1 | H′ at **bs 2048** (A's LR recipe), OneCycle 20 ep, ~18 h | H′ | does the small batch still pay once anchored? (A vs B said 1.4–2.6× un-anchored) |
| **R** | sess3:3 | H′ **hybrid**: stage 1 bs 36 k, WSD constant LR 2.2e-4, 15 ep; stage 2 `pretrained_ckpt_path`, bs 2048, cosine 1.25e-5 → 1e-6 (LR ÷ 17.6 = batch ratio), 5 ep; ~14.5 h | M′ (and H′) | user's idea: is bs 2048 needed throughout or only in the anneal? R ≈ M′ → 3× cheaper scaling recipe; R ≈ H′ → small-batch noise is needed during the bulk phase |
| **P′** | sess3:2 | H′ + **per-hit residuals to the seed helix** as 2 extra input features (Δ(rφ), Δz at each hit; `input_dim` 12 → 14; clipped to ±20 mm; ~½ day code + tests, launched last) | H′ | the KF's representation inside the encoder; targets φ at 100 GeV (1.98×) and q/p |
| **Q′** | sess5:0 | M′ for **50 epochs** (legacy pretrain length, OneCycle stretched), checkpoints every 10; ~49 h | M′ | training length in the legacy regime (bs 2048): if φ/q/p still gain > 10 % from 20 → 50 ep, long runs are mandatory before scaling |

**Launch status (sess3, 2026-08-27 18:16–18:17, user sign-off "launch the first four";
`config/ssm_cls/ICLR_sweep3/launch_sess3.sh`, one process per GPU, nohup,
`TRK_MATMUL_PRECISION=highest`; 60-step dry runs passed first):**
H′ pid 2074465 GPU 0 (run `4cee4cec…`, 84 GB, 100 %), M′ pid 2075106 GPU 1 (run
`7f37bf9f…`, 29 it/s → 93,520 steps/epoch ≈ 53 min → ~18 h), R1 pid 2076720
GPU 3 (run `fe684542…`); `chain_R.sh` (pid 2077848) waits on R1 and launches R2
from its `last.ckpt` on GPU 3 (`--model.pretrained_ckpt_path`, same head, same
ranges). **P′ launched 2026-08-27 18:40 on GPU 2 (pid 2109283, run `051fa443…`)** after its
60-step dry run: `Pp_…_seedres.yaml` = H′ + `data.seed_residual_features: true`
(`flat_data._pack` appends asinh(Δu/0.1 mm), asinh(Δv/0.1 mm), s_helix per hit,
`seed.seed_residuals`, Bz 3 T), `input_dim` 15, norm ranges ±8 / ±8 / 0–4096 mm.
Logs `launch_logs/sweep3/`, Comet `SW3-*`.
Code changes for this sweep: `model.py` WSD accepts `decay_pct: 0` (warm-up +
constant, no cooldown — R1); `preprocess_flat.py --pt-min` (selection
override, recorded in the manifest); `scripts/07_build_mixed_store.py`
(symlink-mixed flat store, see Q′); `scripts/06_fetch_nersc_ttbar.sh`.
**Q′ is redefined (user, 2026-08-27):** 4L (like every other run), bs 2048, 50
epochs, trained on a MIXED store = uniform muons + up to 8 M ttbar tracks from
the new NERSC ttbar runs (§4.7), launched on sess5 once the ttbar data is
preprocessed (`launch_sess5.sh`, periodic checkpoints every 10 epochs appended
on the CLI). Caveat: Q′ then differs from M′ in BOTH length and data — a
muon-only 50-epoch twin on sess5:1 (after killing C) is the proposed control.

Dropped from the earlier draft: O (Fourier 2⁻¹³; fallback if P′ fails), the
no-Fourier ablation, the 25 %-data twin. Read-out: `04_eval_ckpt_iclr.sh` on
`last.ckpt` → per-domain GM5/GM4 (`eval_plots/sweep2/geomean_ratios.txt`
format) + `rms_by_pt.txt`. C (10L, sess5:1) lands in ~1.5 days and answers the
architecture question separately (vs A). Scaling decision after this round
(~2 days): recipe = {H′ or P′} × {M′ / R / Q′ winner}.

### 4.7 New ttbar data on NERSC (found 2026-08-27) — low-pT / hadron enrichment plan

`https://portal.nersc.gov/cfs/m4958/ColliderML/drift_beamspot/ttbar/v1/runs/`
now holds **runs 0–784** (785 runs × 1,280 events ≈ 1.0 M events; we had runs
0–5 = 7,680 events = 200,865 tracks). Verified on run 6: 1,280 events,
`particles / tracker_hits / tracks` **and `truth_tracks`** (d0, z0, φ, θ, q/p,
majority_particle_id, hit_ids, …) — the new ttbar production carries the
truth-KF, so ttbar gets the same reference as the muons (the CKF-only caveat
of §2 disappears for the new runs). Per run ~153 MB for the four parquet tables
(tracker_simhits and ROOT files skipped); fetched 2026-08-27 18:19 → ~18:45
with `scripts/06_fetch_nersc_ttbar.sh 7 784` (8 streams, ~30 runs/min) into
`/scratch/colliderml/drift_beamspot/ttbar/v1/runs/<N>/` (sess3). Expected
yield with the current `core` selection: ~26 tracks/event → ~26 M tracks,
56 % below 2 GeV, 29 % below 1 GeV — exactly the missing domain of §0.3.

Plan (user decisions 2026-08-27):
- **ttbar testing limit pT ≥ 1 GeV**: the ttbar *evaluation* store is rebuilt
  with `preprocess_flat.py --pt-min 1.0` (no training data below 1 GeV for the
  muon-only runs; the < 1 GeV bin is pure extrapolation, §3.2 handle c). New
  eval store `ttbar_new_pt1` = runs 6–45 (51 k events, ~0.9 M tracks ≥ 1 GeV)
  with truth-KF side-cars; the old `ttbar` store (runs 0–5, pT ≥ 0.5) is kept
  for the extrapolation read-out. Eval runs disjoint from training runs.
- **Training enrichment**: runs 46–784 → `ICLR_retraining_geom/ttbar_new`
  (~24 M tracks, pT ≥ 0.5, standard 90/5/5 by event); Q′ trains on
  `ICLR_retraining_geom_mixed8M` = uniform train parts + the first ttbar
  parts up to 8 M tracks (`scripts/07_build_mixed_store.py`; val/test = muon
  only so the per-epoch metrics stay comparable). Batches are contiguous
  blocks within one part, so ~4 % of the steps are pure-ttbar batches.
- **Trap:** the raw tree `/scratch/colliderml/drift_beamspot/ttbar/v1/runs/` now
  holds all 785 runs, so any preprocessing pointed at `ttbar/v1` ingests the
  whole production. The legacy 6-run `ttbar` sample is the farm
  `/scratch/colliderml/drift_beamspot/ttbar_r0-5/v1`; eval = `ttbar_new_eval`
  (runs 6–45), training = `ttbar_new_train` (46–784). `09_rebuild_stores_bz3.sh`
  hit this on 2026-08-27 (its B3 `ttbar` store is rebuilt from runs 0–5 by
  `09b_fix_B3_ttbar_r0-5.sh`).
- **Reminder (user)**: consider the full new ttbar (24 M tracks, truth-KF) as
  the second pretraining/fine-tuning component for the next round — a mixed
  muon + hadron dataset with real low-pT statistics; and re-check the portal
  for further productions (pile-up ttbar) before each round.

### 4.8 The solenoid field is 3 T, not 2 T (found 2026-08-27 by the P′ research agent, verified)

Two independent measurements on `single_muon_uniform/test/part_0000`: (i) the
ACTS conformal seed's q/pT ÷ truth q/pT = 1.50 in every pT bin; (ii) a plain
Cartesian circle through the three seed pixel hits gives Bz = pT_truth /
(KAPPA·R) = **3.001 / 3.005 / 2.999 / 3.003 / 3.046 T** (medians, 1–2 / 2–5 /
5–20 / 20–50 / 50–110 GeV). The truth-KF's q/pT agrees with the truth to 0.14 %,
so ACTS fitted with the right field — only our code (`seed.py`, `perigee.py`,
`hit_sorting.helix_arc_length`) carried the ODD default `DEFAULT_BZ = 2.0`.

Consequences:
1. **Seed q/p was biased ×1.5** (Bz cancels in the seed's d0/z0/φ/θ, not in
   q/p). That is why the anchored q/p head of G/H was 2–5× worse than an absolute
   head (§4.5 reading 2): it had to learn a 50 % multiplicative correction on a
   heavy-tailed anchor. Fixed: `seed.py: DEFAULT_BZ = 3.0` (seed q/p − truth RMS
   at 1–2 GeV shrinks 16×, 0.21 → 0.013 e/GeV). The running sweep-3 runs use
   the absolute q/p head and are unaffected; R2/P′/Q′ pick up the 3 T seed. The
   anchored q/p head deserves a re-test with the 3 T seed (a cheap 6th run).
2. **Truth targets carry a field-dependent transport error.** The stores'
   targets come from `truth_perigee(vertex, momentum, Bz=2.0)`; the vertex sits
   |v_T| ≈ 4 mm from the beamline (drift beamspot), and over that arc the wrong
   curvature shifts the perigee (ttbar run 6, RMS over tracks, 3 T − 2 T):

   | pT [GeV] | Δd0 | Δz0 | Δφ | Δθ, Δq/p |
   |---|---|---|---|---|
   | 0.5–1 | 2.8 µm | **2.35 mm** | **1.31 mrad** | 0 |
   | 1–2 | 1.4 µm | **154 µm** | **0.65 mrad** | 0 |
   | 2–5 | 0.7 µm | 15 µm | 0.31 mrad | 0 |
   | 5–20 | 0.2 µm | 3 µm | 0.12 mrad | 0 |
   | 20–200 | 0.06 µm | 0.3 µm | 0.03 mrad | 0 |

   At 1–2 GeV this is the size of the truth-KF "resolution" we quote there
   (z0 158 µm, φ 1.84 mrad) — i.e. part of what we called KF resolution at low
   pT is target error shared by the KF and the network alike; above 5 GeV it
   is negligible against everything measured so far. All §4.1–4.6 conclusions
   (high-pT floor, anchors, Fourier, batch size) stand; low-pT absolute numbers
   below 2 GeV must be re-measured on 3 T-target stores.
3. **Action taken:** `preprocess_flat.py --bz` (default 2.0, recorded in
   manifest/dataset_meta); `scripts/09_rebuild_stores_bz3.sh` rebuilds the five
   stores with 3 T targets into `ICLR_retraining_geom_B3` / `ICLR_eval_geom_B3`
   (+ farm, truth-KF side-cars now written by `preprocess_flat` itself via the
   per-shard `truth_tracks` join, KF baselines `eval_plots/baselines_KF_geom_B3/`)
   — launched 2026-08-27 ~19:00 on sess3 (CPU only; the 2 T stores stay for the
   running sweep). The new-ttbar stores are built twice (2 T for evaluating the
   running sweep, 3 T for Q′). **Recommendation: Q′ and everything after run on
   the B3 stores; the user decides whether the 2 T `_geom` stores are retired.**
   The ColliderML producers should confirm the 3 T field (not found in any
   on-disk config).

### 4.9 Mixed muon + ttbar training — final definition (user decisions 2026-08-27 evening)

- **Training ttbar = `ICLR_retraining_geom_B3/ttbar_new_pt1_tr`**: runs 46–784,
  **pT ≥ 1 GeV** (user: "apply a 1 GeV cut … so we do not bother with the lower
  energy particles during training"), 3 T targets, geometry order, 4 runs/part:
  17,607,223 tracks = train 15.9 M / val 0.9 M / test 0.8 M. (`ttbar_new`, pT ≥
  0.5, 24.5 M tracks, 2 T and 3 T copies, stays on disk unused.)
- **Mix = `ICLR_retraining_geom_B3_mixed16M`** (`07_build_mixed_store.py
  --extra-max-tracks 100000000 --extra-val`, user 2026-08-27: "the full 15.8 M"):
  uniform B3 train (191.5 M) + ALL ttbar_new_pt1_tr train parts (15.84 M) =
  207.4 M training tracks; val = muon val (5.0 M) + ttbar val (0.88 M), test =
  muon test. (An earlier 12.8 M variant was built and discarded.) ttbar tracks
  per bin (× = statistics gain over the muons alone, from the 12.8 M build scaled
  ×1.23): 1–2 GeV ~6.0 M (×4.4), 2–3 ~2.8 M (×2.6), 3–5 ~2.7 M (×1.8), 5–10
  ~2.3 M (×1.3), > 10 GeV ~2.0 M (×1.0).
  3 T seed q/p residual (truth − seed) on the uniform test part: RMS 0.016 /
  0.008 / 0.007 / 0.008 e/GeV at 1–2 / 2–5 / 5–20 / 20–110 GeV, 99.99 % 0.09
  overall (0.13 at 1–2 GeV), sign flips 7.4 % above 20 GeV → Hq range ±0.15.
- **Default ttbar test set from now on = `ttbar_new_pt1`** (runs 6–45, pT ≥ 1,
  954,495 tracks, **truth-KF reference**; disjoint from all training runs).
  `04_eval_ckpt_iclr.sh` evaluates it; pass the B3 eval root for B3-trained
  runs. The old `ttbar` (runs 0–5, CKF) stays for continuity only.
- **d0/z0 windows**: NOT applied in any campaign-2 store (uniform included) —
  `core` minus the windows, by design (§2); ttbar and muons are consistent.
- **Staging on another machine**: `scripts/10_stage_B3_on_sess5.sh` copies the
  B3 stores from `/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_geom_B3`
  (uniform 148 GB + small stores; copies started 2026-08-27 ~22:00 from sess3),
  builds the eval farm (`scripts/build_eval_farm.py`) and the mixed store; then
  `config/ssm_cls/ICLR_sweep3/launch_sess5.sh <config> <gpu>` (DRY=1 first).
- **Runs on sess5 — LAUNCHED 2026-08-27 22:36 (user: "launch those last two runs")**:
  GPU 0 → **Q′** `Qp_…_B3mixed16M_50ep` (pid 424510, run `4eabeddb…`); GPU 1 → **Qm** =
  muon-only twin `Qm_…_B3muon_50ep` (pid 425229, run `e13bbd50…`) — same recipe, same 3 T
  uniform store, no ttbar: separates "50 epochs" from "hadron data" in Q′. Both 4L,
  bs 2048, OneCycle 50 ep, checkpoints every 10 epochs (`save_top_k -1`), ~2 days.
  **C (10L, sweep 1) was killed at epoch 14/20 for this** (its `last.ckpt` = epoch 13,
  run `22573b1c…`, kept). Unused alternative: **Hq** = H′ with the q/p head anchored
  to the 3 T seed (`Hq_…_anchqop3T_B3`, 14 h) — G/H's anchored-q/p verdict was made
  with a ×1.5-biased anchor (§4.8) and is void.
- **Seed cost, measured.** CPU (one core, numpy): seed 1.5–1.8 µs/track, seed
  residuals (P′) +1.8–1.9 µs/track → 300 k tracks: 0.54 s / 1.08 s. Training:
  hidden in the loader workers (0 % step-rate change, B 2.36 vs G 2.37 it/s).
  **Inference study (sess5 GPU 0, H checkpoint 4L/128/ds64, v5pc kernels, fp32,
  bs 10 k, 270 k tracks timed; `scripts/bench_seed_inference.py`,
  `eval_plots/seed_inference_bench/`)**: model forward 2.23 µs/track = 0.67 s
  per 300 k; **seed on the GPU (`seed.seed_perigee_torch`) 0.22 µs/track =
  0.066 s per 300 k = +9.9 %**; P′ residuals on the GPU +0.04 µs/track (+1.9 %);
  the same seed on the CPU in the main process 1.45 µs/track = +65 %. GPU and
  CPU seeds agree to 1e-7 mm. **DECIDED (user, 2026-08-27): the ~10 % slower
  inference with the seed computed on the GPU is acceptable — the seed (and
  the anchored heads / P′ residual features that need it) is part of the
  deployed model; the CPU-serial path is not.** The torch path is unoptimised
  (padded scatter + ~15 small kernels; fusable if the 10 % ever matters).

### 4.10 Sweep-3 results, first read-out (2026-08-28 10:30; H′, R2, P′ done, M′ at epoch 19)

All runs finished on schedule (bs 36 k = 37.5 min/epoch → 12.5 h; R1 15 epochs +
R2 5 epochs chained automatically; every epoch validated). Bundles
`eval_plots/sweep3/<run>/plots/`, tables `eval_plots/sweep3/geomean_ratios.txt`;
2 T eval stores (the runs trained on 2 T targets). Uniform muons, ratio SSM /
truth-KF (CKF-DM subset, 3.42 M tracks):

| run | d0 | z0 | φ | θ | q/p | GM5 | GM4 |
|---|---|---|---|---|---|---|---|
| H  (sw2, anchored q/p) | 1.13 | 1.22 | 1.78 | 1.16 | 2.07 | 1.42 | 1.30 |
| H′ absolute q/p head | 1.08 | 1.22 | 2.05 | 1.16 | 1.49 | 1.36 | 1.33 |
| R2 hybrid 15 ep bs 36 k → 5 ep bs 2048 | 1.07 | 1.22 | 1.84 | 1.16 | 1.25 | 1.28 | 1.29 |
| **P′ seed-residual inputs** | **0.99** | **1.00** | **1.37** | **0.98** | **1.25** | **1.11** | **1.07** |
| M′ pure bs 2048, 20 ep (finished 11:50) | 1.04 | 1.21 | **1.16** | 1.14 | **1.12** | 1.13 | 1.14 |

M′ per domain (GM5): 2 GeV 1.18, 10 GeV 1.09, 100 GeV 1.09, ttbar_new_pt1 1.18, old ttbar 1.27 — the
small batch alone takes φ from 2.05 (H′) to 1.16 and q/p to 1.12; R2's 5-epoch anneal recovered only
part of that (φ 1.84). P′ (bs 36 k) and M′ (bs 2048) are complementary: P′ owns d0/z0/θ, M′ owns
φ/q/p → the P′ recipe at bs 2048 (= Q′2 / Qm2, and S2) is the obvious combination.

Per domain (GM5): 2 GeV H′ 1.30 / R2 1.25 / **P′ 1.07**; 10 GeV 1.22 / 1.17 /
**1.07**; 100 GeV 1.38 / 1.27 / **1.08**; ttbar_new_pt1 (truth-KF, 648 k DM
tracks) 1.33 / 1.28 / **1.10**; old ttbar (CKF) 1.42 / 1.33 / 1.13. P′ per pT bin
(uniform): d0 0.98–1.01, z0 0.98–1.00, θ 0.96–0.98 in EVERY bin; φ 1.09 (1–2 GeV)
→ 1.43 (50–110); q/p 1.40 → 1.23. Absolute P′ on uniform: 14.2 µm / 21.5 µm /
0.246 mrad / 0.057 mrad / 2.81e-4 vs truth-KF 14.3 / 21.6 / 0.179 / 0.058 / 2.24e-4.

Readings:
1. **The KF's representation inside the encoder closes the geometry gap.** With
   per-hit residuals to the seed helix, d0/z0/θ are at the truth-KF everywhere
   (the ≤ 2 % below 1.0 is within the shared target error of §4.8). The remaining
   deficit is transverse and pT-dependent: φ 1.1× at 1–2 GeV → 1.4× at high pT,
   q/p 1.2–1.4×. Next levers: bs 2048 / the R schedule on P′ (R2 vs H′ gained
   16 % on q/p, 10 % on φ), longer training (φ/q/p still falling at epoch 19),
   the residuals-only ablation P″.
2. **Absolute q/p head (H′ vs H)**: q/p 2.07 → 1.49 as predicted, d0 1.13 → 1.08,
   but φ 1.78 → 2.05 (0.318 → 0.367 mrad) although only the q/p head changed —
   loss-balance coupling (±1.0 range at weight 1) or run-to-run noise; **assume
   ~15 % on φ as the noise floor until a seed twin exists**. With the 3 T seed the
   anchored q/p head deserves one re-test (Hq).
3. **Hybrid batch (R2 vs H′)**: same wall-clock (14.5 vs 12.5 h), better q/p
   (1.25 vs 1.49), φ (1.84 vs 2.05), d0; z0/θ seed-level in both. A 5-epoch
   bs-2048 anneal buys most of what the small batch gives; M′ (pure bs 2048,
   20 ep, ~18 h; eval auto-queued) says how much is left.
4. ttbar_new_pt1 with the truth-KF reference behaves like the muon sets (P′
   1.10): species is not the problem at pT ≥ 1 GeV.
5. **sess5 recommendation**: run Q′ on the P′ recipe — `Qp2_…_seedres_B3mixed16M_50ep`
   (+ muon-only twin `Qm2_…_seedres_B3muon_50ep`), not on H′ (2 GPU-days on a
   superseded representation). Configs written; user sign-off pending. Note: the
   Q′ definition was changed at 22:19 on 2026-08-27 (outside this session) to
   ALL 15.8 M ttbar tracks (`_B3mixed16M`, `10_stage_B3_on_sess5.sh` MAXX = all).
Caveat: `fast_rms_eval` restricts to the CKF double-matched subset (N column)
even where the truth-KF is the reference — consistent across runs, but 32 % of
the uniform / ttbar_new_pt1 tracks are outside the numbers; to revisit.

### 4.11 Are we converging? (2026-08-28) — and the SWEEP 4 proposal (awaiting sign-off)

Per-epoch val curves (`[val epoch N] iter-3σ RMSE`, ratio to the truth-KF; see the
session log / Comet `SW3-*`): in P′ d0/z0/θ have been flat at 0.99–1.02 since
epoch 14 — done. φ and q/p were still falling steeply when the 20-epoch cosine
ended: P′ φ 6.8 → 4.4 → 2.8 → 2.0 → 1.6 → **1.38** over epochs 14→19 (−13 % in
the last epoch), q/p 5.4 → … → **1.28** (−5 %); H′ −6 % / −10 % in the last epoch;
R2 gained ~3 %/epoch on both during its 5-epoch bs-2048 anneal. The anneal was
too short for the transverse parameters — the legacy "precision arrives late"
signature, now confined to φ/q/p.
Tails: P′'s pre-clip/clipped ratios and clipped fractions on uniform muons (d0
1.24× / 1.9 %, z0 2.8× / 5.5 %, φ 1.8× / 4.5 %, θ 2.4× / 7.2 %, q/p 2.4× / 5.2 %)
equal or beat the truth-KF's own (1.32× / 2.3 %, 3.3× / 6.3 %, 2.2× / 5.4 %,
2.4× / 7.0 %, 2.35× / 5.1 %). On ttbar_new_pt1 they match the KF on d0/z0/φ/θ,
but **q/p has a 10.5× pre/post ratio (9 % clipped) vs the CKF's 2.3×** — the one
real tail left (low-pT hadrons / seed sign flips; to be binned in pT).
Structural candidates for the pT-dependent φ gap (1.09 → 1.43): the 1-D long
strips (Δv is the module centre there — a per-hit "1-D measurement" flag or
masking Δv on volumes 28–30), and seed q/p sign flips (7 % above 20 GeV) that
flip the residual pattern — an iterated seed (re-seed from the network's own
first estimate, the KF/GBL way) is the principled fix.

Sweep 4 (all on the 3 T `_B3` stores, P′ recipe, sess3 GPUs as they free up):

| run | change vs P′ | why | cost |
|---|---|---|---|
| S1 | 50 epochs OneCycle bs 36 k | the length axis on the right representation | ~31 h |
| S2 | hybrid: 15 ep bs 36 k const → **10** ep bs 2048 cosine | R2's win with a longer anneal | ~19 h |
| S3 | **fine-tune probe**: Muon+WSD continuation from P′ `last.ckpt`, 10 ep, same head/ranges | the legacy second stage, directly | ~6 h |
| S4 | seed twin of P′ (different `seed_everything`) | the noise floor we keep assuming (15 % on φ) | ~13 h |
Then P″ (residuals-only inputs) and Hq (anchored q/p, 3 T seed) as the next pair.
sess5: Q′2 / Qm2 (P′ recipe, 50 ep, bs 2048, mixed16M / muon-only).

### 4.12 Inference cost of the seed (measured 2026-08-28, one H100 NVL, `scripts/bench_infer_flat.py`)

`docker/rtx_infer/bench_infer.py` (branch `rtx-infer-docker`) ported to the flat
store (+ `--seed-residuals`, + collate timing) = `scripts/bench_infer_flat.py`;
log `eval_plots/sweep3/bench_infer_H100_seedcost.log`. Strict fp32, v5pc kernels,
batches preloaded in pinned RAM (GPU time = forward only, incl. H2D).

| batch | B (no seed, 12 feat, 11 scales) | H′ (seed anchors, 12 feat) | P′ (seed + residuals, 15 feat) | peak VRAM |
|---|---|---|---|---|
| 32 k | 456 k tr/s (70.1 ms) | 449 k (71.3 ms) | 447 k (71.6 ms) | 3.6 GiB |
| 128 k | 447 k | 440 k | 438 k | 14.2 GiB |
| 300 k | 441 k (681 ms) | 431 k (696 ms) | 428 k (701 ms) | 33 GiB |
| 600 k | 438 k (1.37 s) | 435 k (1.38 s) | 430 k (1.39 s) | 66 GiB |

GPU: throughput is flat at 430–456 k tracks/s (2.2–2.3 µs/track) from 32 k up —
the 4L model saturates the H100 at 32 k; the seed costs **−2 %** GPU throughput
(P′ vs B), purely the 480- vs 264-wide first layer. ~800 k tracks would fit in
94 GB; nothing is gained beyond 32 k.
CPU collate (numpy, one core, `_pack` timed with the seed stubbed / on / on +
residuals): 32 k tracks 11 / 45 / 113 ms, 300 k 138 / 542 / 1147 ms, 600 k
289 / 1056 / 2387 ms → the seed adds ~1.3 µs/track, the residuals ~2.1 µs/track,
the seed-free collate is 0.35–0.5 µs/track. So per 600 k tracks: GPU 1.39 s vs
seed 0.77 s + residuals 1.33 s on one core. Serially on one core the seed
pipeline would add 150 % to the end-to-end time; over 8 loader workers it is
0.26 s per 600 k (hidden behind the GPU, as in training where the step rate did
not change); on the GPU (torch twin `estimate_free_torch` + a torch port of
`seed_residuals`) — DONE 2026-08-28 afternoon: `track_regression/seed_torch.py`
(`gpu_seed_features(hit_features, cu_seqlens)` → seed (B,5) + residual features
(n_hits,3), float64 math, parity test `tests/test_seed_torch.py` on CPU and CUDA)
and `scripts/bench_infer_flat.py --gpu-seed` (deployment mode: seed + residuals
+ anchors inside the timed loop) / `--profile-range` (nsys capture range).
Results (P′, GPU forward incl. seed, tracks/s; logs `eval_plots/sweep3/results_*.log`):

| tracks/batch | 2 048 | 4 096 | 8 192 | 16 384 | 32 000 | 128 k | 300 k | 600 k |
|---|---|---|---|---|---|---|---|---|
| fp32, CPU seed (forward only) | 382 k | 413 k | 425 k | 431 k | 434 k | 438 k | 428 k | 430 k |
| fp32, **GPU seed** | 251 k | 325 k | 377 k | 407 k | **426 k** | 432 k | 420 k | 421 k |
| **TF32** matmuls, GPU seed | 309 k | 443 k | 553 k | 627 k | **663 k** | 691 k | 666 k | 666 k |

- Saturation: smooth knee at 8–16 k tracks, plateau 434 k/s (2.3 µs/track) from
  16 k to 128 k; no spike. nsys (timed loop only): 155 kernel launches per
  forward at every size; GPU busy 89 / 95 / 97 % at 2 k / 8 k / 32 k; kernel time
  per track 2.33 / 2.25 / 2.23 µs → the small-batch loss is ⅔ launch gaps
  (155 × 3.5 µs) and ⅓ kernel efficiency. Kernel mix: selective scan 36 %,
  GEMMs 50 %, RMSNorm 5 %, Fourier `torch.cat` 2–4 %.
- GPU seed: −2 % at ≥ 32 k (1.3 ms per 32 k batch, ~40 small kernels); at 2 k
  the launch cost dominates (−34 %) → deploy with ≥ 16 k tracks per batch.
- TF32: ×1.56 throughput; physics on all six eval sets identical to ≤ 0.2 %
  except 100 GeV q/p +2.3 % and φ +1.7 % (`eval_plots/sweep3/Pp_tf32/`).
  Keep strict fp32 for physics numbers, TF32 for the deployment benchmark.
- Hypothesis for the RTX 5000 Ada spike at ~2 048 (colleague): working set
  (≈ 23 MB activations/layer at 2 k tracks) fits the Ada's 32 MB L2 → memory-
  bound kernels run at L2 speed; above ~4 k they spill to 576 GB/s GDDR6. The
  H100's 3.35 TB/s HBM hides the spill → no spike. Test on the Ada: ncu
  `lts__t_bytes` vs `dram__bytes` at 1 k / 2 k / 4 k with the `--profile-range` recipe.

### 4.13 Data-discipline checks (2026-08-28) — before any re-preprocessing

1. **Hit time is NOT fixed in the new ttbar production.** Runs 6 and 400 look
   exactly like run 0: strips 100 % `time == 0`, pixel time uncorrelated with
   radius (ρ = 0.03), stored-time order = geometry order on 6–7 % of tracks.
   → no switch to time sorting; geometry order stays (D.2). Re-check when the
   300 M-muon production arrives.
2. **No hit of any test store lies outside the model's normalisation ranges**
   (0.00 % on uniform, 2/100 GeV, ttbar old/new) — the input ranges are not an
   OOD source.
3. **OOD tracks do not explain the pre-clip excess.** ttbar_new_pt1 (953 k
   tracks with truth-KF, all — not the DM subset): pre-clip q/p ratio 2.80 all
   → 2.81 without pT > 110 GeV (0.27 % of tracks) → 3.03 restricted to tracks
   whose truth − seed lies inside the heads' ranges (0.98 % outside carry
   5.8 % of the SSM q/p Σr²) → seed sign flips 0.28 % carry 0.8 %. The worst
   0.1 % of tracks carry 44 % of the SSM's q/p Σr² (the truth-KF's own worst
   0.1 % carry 70 % of its); the excess is in the bulk of the hadron sample.
   On uniform muons the same cuts change nothing (q/p 1.20 throughout).
   Also: on the full sample the pre-clip q/p ratio is 2.8, on the CKF-DM
   subset 5.0 — the DM restriction flatters the KF's tails.
4. **Recommendation on re-preprocessing now:** worth doing only for the
   evaluation-domain rule (test where you train: ttbar 1–110 GeV → add the
   pT ≤ 110 cut to `ttbar_new_pt1` and keep it in training) and the 3 T
   targets (B3 stores, done); cutting on input ranges is a no-op, and time
   sorting is not available. Do it before the next long round (Q′2/Qm2, sweep 4)
   so those runs are the last "clean" ones: cost minutes for ttbar (reprocess
   with `--pt-max` — to be added, 5 lines), 45 min for the uniform B3 store if
   anything else changes. The > 110 GeV failure is a training-range issue →
   request the muon gun to ≥ 250–300 GeV.

### 4.14 Loss weights and the φ head (user, 2026-08-28) — fixed for the next iteration

P′ (and every anchored run so far) used weights d0 0.1, z0 1, φ 0.1, θ 1, q/p 1 —
inherited from the legacy absolute-range configs. With the anchored pinball
heads the weights act directly as gradient scales: the pinball gradient w.r.t.
the prediction is ±τ (constant, independent of the residual size) in the
normalised [−1, 1] space, so d0's head received 10× less gradient than z0/θ/q/p
throughout. φ was worse: the circular head is Smooth-L1 (Huber, β = 0.01) on
(sin Δφ, cos Δφ) with weight 0.1 — once the residual is below β the gradient is
x/β, i.e. ~2.5e-4/0.01 × 0.1 ≈ 2.5e-3 at P′'s φ residual (0.25 mrad) against
~0.5 for a pinball head: **φ got ≈ 200× less gradient than z0/θ** exactly in the
regime where it lagged (1.37×, still falling 13 %/epoch at the end). d0 was
under-weighted but had converged anyway (0.99×).
Decision: **all weights 1.0, and φ moves to an anchored quantile head**
(`type: quantile, norm ±0.015 rad, delta_anchor: seed_phi`; |truth − seed_φ|
99.99 % = 14 mrad, §4.2; the anchored Δφ is wrapped to (−π, π] in
`TrackParameterLoss.forward` and the metrics path, `predict_physical` wraps
after adding the seed back). φ then also gets a 7-quantile ladder like the
other four. Applied to the pending configs Q′2, Qm2, Hq and to the new sweep-4
reference **Pw** (`Pw_…_seedres_evenw_B3.yaml` = P′ + even weights + φ quantile
head, 3 T store; 60-step dry run passed, `launch_logs/sweep4/dryrun/`). Sweep-4
runs S1–S4 are to be built on Pw. Caveat for reading Pw vs P′: different target
field (B3 vs 2 T) → compare above 5 GeV, or run a B3 P′ twin.
A further, untested idea for q/p (the other laggard): its absolute head maps
±1 e/GeV to [−1, 1], so the 2.8e-4 residual is 1.4e-4 of the range — Lion's
fixed-size updates (±lr per weight) put output noise of ~lr·√128 ≈ 2e-3 on it at
peak LR, only the anneal brings it below the residual — consistent with q/p
converging last and with bs 2048 (more low-LR steps) helping q/p most. A
scale-free head (predict q/p_true / |q/p_seed| with the 3 T seed, range ±3)
would equalise the precision demand across pT; Hq (anchored q/p, ±0.15) is the
first step in that direction.

### 4.15 SWEEP 4 — the overnight set of 2026-08-28 (configs `config/ssm_cls/ICLR_sweep4/`, dry runs passed, AWAITING GO)

Common recipe **W** = P′ representation (seed-residual hit features, 15 inputs) +
even loss weights + φ quantile head (§4.14) + 3 T targets + **mixed store**
`ICLR_retraining_geom_B3_mixed16M` (191.5 M muons + all 15.8 M ttbar pT ≥ 1;
val = muon val + ttbar val) + **hybrid schedule** (stage 1: bs 36 k, WSD constant
2.2e-4, 15 ep ≈ 10.2 h → stage 2: bs 2048, cosine 1.25e-5 → 1e-6, 10 ep ≈ 9.5 h;
`chain_stage2.sh` launches stage 2 from stage 1's `last.ckpt`). 5,760 steps/epoch
at bs 36 k (40.7 min), 101,259 at bs 2048 (57 min).

| run | GPU | change vs W | question | wall-clock |
|---|---|---|---|---|
| **W** (W1→W2) | 0 | — (reference) | the new baseline: P′ + even weights + mixed data + 10-epoch small-batch anneal | ~20 h |
| **X** (X1→X2) | 1 | `fourier_scales: []` — no Fourier encoding, input net reads the 15 min-max features | is Fourier still needed once inputs are seed residuals? (user) | ~20 h |
| **Y** (Y1→Y2) | 2 | scale-free q/p head: target (q/p − seed)/(|seed| + 0.02), range ±2 (`scale_anchor_eps`) | equal q/p precision demand at 1 and 100 GeV (0.008 vs 0.011; absolute head 9× apart) | ~20 h |
| **Z** | 3 | pure bs 2048, OneCycle 25 ep (no hybrid) | hybrid vs pure small batch at equal epochs (M′ beat R2 by 13 %) — decides the scaling recipe | ~24 h |

Code for this sweep: `model.py` accepts `fourier_scales: []` (identity encoding);
`losses.py` `scale_anchor_eps` (target (t − a)/(|a| + ε), inverse in
`predict_physical`, test `tests/test_scaled_anchor.py`). Why not the raw ratio
q/p_true / |q/p_seed| (user's first formulation): above 20 GeV the seed curvature
is at its noise floor, |ratio| reaches 400–4000 (99.9 %), 8.6 % of muons beyond
3 — the ε-regularised residual keeps 99.99 % of muons inside ±1.1.
Dry runs (60 steps, all four stage-1/Z configs + W2 from W1's dry checkpoint):
passed; Y1's q/p val after 60 steps 0.020 vs W1's 0.148 (starts at seed quality).
Read-out: `04_eval_ckpt_iclr.sh <run> last.ckpt <out> /scratch/colliderml/ICLR_eval_geom_B3`
(3 T eval root!) → GM5/GM4 per domain incl. `ttbar_new_pt1`.
**LAUNCHED 2026-08-29 00:28 on sess3 (night chain `scripts/13_night_chain_2026-08-28.sh`,
after the v2 rebuild and passing dry runs), on `ICLR_retraining_v2_mixed` (DATA v2, §4.17):**
W1 pid 2642758 GPU 0 (run `fac3c3b5…`), X1 pid 2642776 GPU 1 (`f0b4f158…`), Y1 pid
2642766 GPU 2 (`05c3390f…`), Z pid 2642759 GPU 3 (`063276b0…`, 26.7 it/s → 101,237
steps/epoch ≈ 63 min → 25 epochs ≈ 26 h); `chain_stage2.sh` ×3 waiting to start
W2/X2/Y2 at bs 2048 when the stage-1 runs finish (~10.3 h at 5,760 steps/epoch).
Logs `launch_logs/sweep4/`, Comet `SW4-*`. Evaluate with the **`ICLR_eval_v2`** farm
(`04_eval_ckpt_iclr.sh <run> last.ckpt <out> /scratch/colliderml/ICLR_eval_v2`).
**Stage-1 read-out (2026-08-29 13:00, W1 `last.ckpt` = 15 ep constant LR, un-annealed,
`ICLR_eval_v2`, SSM / truth-KF):** 100 GeV d0 12.6/12.8, z0 19.2/18.7, φ **0.140/0.140**,
θ 0.040/0.040, q/p 1.04e-3/1.77e-4; 10 GeV 19.2/19.3, 35.6/35.6, 0.357/0.357,
0.119/0.120, 1.9e-3/4.8e-4; ttbar_new_pt1 (646 k DM) 35.1/35.3, 57.1/57.2,
0.914/0.912, 0.423/0.422, 1.6e-2/1.75e-3 — **d0/z0/φ/θ at the truth-KF in every pT
bin for muons AND hadrons already before the anneal** (P′ had φ 1.37×: the even
weights + φ quantile head did what §4.14 predicted); only q/p is open (4–9×, un-
annealed; Y1's scale-free head 3.5× better than W1's absolute head at this stage,
X1's no-Fourier 4× worse). The Comet `val/*/ssm_rms3s` of these runs are pooled over
the MIXED val set (15 % ttbar with 2.5–8× worse intrinsic resolution) and are not
comparable to the muon-only runs — read per dataset with `04_eval_ckpt_iclr.sh`.
Caveat: `--data.max_test_tracks` must NOT be used with `fast_rms_eval` (the store
trim spreads over parts → reference arrays misaligned; garbage reference column).
Mixing vs muon-only is NOT isolated by this set (all four are mixed); the clean test
is the muon-only twin (Qm2-type) — launch on sess5 with the v2 stores.
Not in the set (next): seed twin of W (noise floor), Hq (anchored q/p ±0.15),
residuals-only inputs P″, kernel follow-ups (§4.16).

### 4.17 DATA v2 (2026-08-28 night, user decisions) — re-download, true-time order, in-range selection

**Timing check on the re-produced portal data** (uniform re-produced 2026-08-24,
ttbar 2026-08-26; checksums differ from our 08-21 copies): the pixel time is now
referenced to the collision (median ≈ 1 mm/c, corr(time, r) = 0.44 vs 0.03
before) but **strip hits still have time = 0** (58 % of hits) → the digitised time
still cannot order a track. Per the user's fallback the stores are sorted by
**`tracker_simhits.true_time`** (`preprocess_flat.py --sort-key true_time`: per
(hit, particle) pair via `simhit_ids`, positional into the event's sim-hit list;
tracker_simhits therefore downloaded for every dataset, ~130 GB each for uniform
and ttbar). On the fresh 2 GeV set the true-time order equals the geometry order
on 100.00 % of tracks — so **at inference the geometry order is the faithful
truth-free stand-in** (§5.2: 99.7 % on ttbar).
**Selection v2** (all datasets): `--bz 3.0`, `--apply-d0z0-windows --d0-window
7.1 --z0-window 270` (= the d0/z0 target normalisation ranges of the configs;
tracks outside are dropped), ttbar additionally `--pt-min 1 --pt-max 110`
(train AND test where the muon gun trains). Hit-feature ranges: nothing was
outside (§4.13), no cut needed.
**Layout**: raw `/scratch/colliderml/drift_beamspot_v2/<ds>/v1/…` (ttbar runs 6–784
tables hard-linked from the 08-27 fetch, which already carried the 08-26 files;
runs 0–5 and all `tracker_simhits` fetched fresh; `scripts/06b_fetch_nersc_flat.sh`,
`06_fetch_nersc_ttbar.sh TABLES=… TAG=…`), stores `/scratch/colliderml/ICLR_retraining_v2/
{single_muon_uniform, single_muon_2GeV, single_muon_10GeV, single_muon_100GeV, ttbar (runs 0–5),
ttbar_new_pt1 (runs 6–45, eval), ttbar_new_pt1_tr (46–784, training)}`, eval farm
`ICLR_eval_v2`, baselines `eval_plots/baselines_KF_v2/`, mixed store
`ICLR_retraining_v2_mixed` (uniform + all ttbar_new_pt1_tr train parts, + its val).
Scripts: `11_rebuild_stores_v2.sh` (build + counts), `12_publish_v2_and_retire_old.sh
publish|retire` (copy stores + raw to `/eos/…/data/{ICLR_retraining_v2, drift_beamspot_v2}`,
then delete the superseded `ICLR_retraining{,_geom,_ssort,_geom_B3}` on /eos and the
old raw + 2 T/B3 stores on /scratch — user authorised 2026-08-28). The 2026-08-21
`ICLR_retraining`/`_ssort`/`ICLR_eval` scratch copies were deleted first (space).
**Counts (script 11, 2026-08-29 00:25; train / val / test):** uniform 191,532,752 /
4,998,235 / 4,998,269 (identical to before: no gun track lies outside |d0| 7.1 mm /
|z0| 270 mm); 2 / 10 / 100 GeV 99,981 / 99,972 / 99,966 (100 k events each, one
shard → all 'train', farm unions them); ttbar (runs 0–5, 1–110 GeV) 95,344 / 23,674 /
23,600; ttbar_new_pt1 (eval) 857,037 / 47,241 / 47,643; ttbar_new_pt1_tr 15,802,737 /
878,303 / 879,454; **mixed train store 207,335,489, val 5,876,538**. KF baselines
(`eval_plots/baselines_KF_v2/`) equal the B3 ones to 3 s.f. (truth-KF uniform d0 14.29 µm,
z0 21.6, φ 0.177 mrad, θ 0.058, q/p 2.24e-4). Uniform build 41 min (40 workers, incl.
the simhits join); ttbar stores 2.5 min.
Sweep-4 configs (`ICLR_sweep4/`) point at `ICLR_retraining_v2_mixed`; evaluate
v2-trained runs with the `ICLR_eval_v2` farm.

### 4.16 Kernel experiments (2026-08-28 night) — opt-in inference variants, defaults unchanged

All variants are behind env vars (default = today's IEEE fp32 single-launch path);
parity vs `_packed_scan_torch_ref` in `tests/test_ssd_variants.py` (6 tests, GPU):
default and BL16 paths match the reference to < 2e-4 relative (max abs), TF32 dots
to 5e-3 relative. Bench = `scripts/bench_infer_flat.py --gpu-seed` on the P′
checkpoint, one H100 NVL, seed + residuals on the GPU inside the timed loop;
raw rows `eval_plots/sweep3/kernel_variants_2026-08-28.tsv`.

| variant (env) | 32 k, strict fp32 | 32 k, TF32 matmuls | 2 048, fp32 | 2 048, TF32 |
|---|---|---|---|---|
| baseline (P′, GPU seed) | 431.6 k tr/s (74.1 ms) | 666.6 k (48.0 ms) | 250 k | 308 k |
| TF32 dots in the scan (`TRK_SSD_DOT_PRECISION=tf32`) | 370 k (86.4 ms) **−14 %** | 523 k (61.2 ms; repeat 521 k) **−22 %** | — | — |
| BL16 bucketing (`TRK_SSD_BUCKET16=1`) | 459 k (69.6 ms) +6 % | 775 k (41.3 ms) +16 % | 266 k +6 % | — |
| compiled front-end (`TRK_COMPILE_FRONTEND=1`) | 448 k (71.4 ms) +4 % | 714 k (44.8 ms) +7 % | 257 k +3 % | — |
| **BL16 + compiled front-end** | **470 k (68.1 ms) +9 %** | **850 k (37.6 ms) +27.5 %** | 273 k +9 % | 345 k +12 % |
| all three incl. TF32 dots | — | 738 k (43.3 ms) | — | — |

Findings:
1. **TF32 inside the scan kernel is slower, not faster**: the two `tl.dot` calls
   are 32×64·64×32 and 32×32·32×32 tiles; with `input_precision="tf32"` Triton
   routes them through the MMA path with operand staging (the same pathology the
   shift-matrix conv hit in the campaign log), which costs more than the FFMA
   IEEE dots save. Physics with TF32 dots + TF32 matmuls is identical to TF32
   matmuls alone (≤ 0.2 %, `eval_plots/sweep3/Pp_tf32dots/`), so the option is
   harmless but useless — leave `ieee`.
2. **Padding is the real waste**: routing the ≤ 16-token tracks (≈ 60 %) through
   a BL=16 launch (index-list indirection, one device sync per forward for the
   split) gives +6 % (fp32) / +16 % (TF32) at 32 k — the fp32 scan kernel is
   only 36 % of the time, so the kernel itself got ~1.2–1.5× faster. A finer
   scheme (two tracks per 32-row tile, no sync) should take the rest.
3. **`torch.compile` of normalise → Fourier → input_net** removes the 32 sin/cos
   kernels + `torch.cat` (+4 % / +7 %); a hand-written Triton kernel (3b) was
   therefore not needed.
4. Combined (BL16 + compiled front-end): **470 k tracks/s in strict fp32, 850 k
   with TF32 matmuls** — 1.09× / 1.28× over the respective baselines, 1.97× over
   today's strict-fp32 production path. Deployment recommendation: TF32 matmuls
   (physics ≤ 2.3 % at 100 GeV, §4.12) + `TRK_SSD_BUCKET16=1` +
   `TRK_COMPILE_FRONTEND=1`, batches ≥ 16 k. Strict fp32 stays the default for
   training and for physics numbers.
5. Not done: merging the two scan directions into one launch (needs stacked
   per-direction weights + runtime REVERSE in the kernel — a refactor of
   `fused_bidi_scan_packed`, gain bounded at a few % at 32 k), the two-tracks-
   per-tile packing, CUDA graphs for the 2 k regime, and the in-kernel GEMM fusion.

### 4.18 The 50-epoch runs on sess5 (Q′ mixed / Qm muon-only, H′ recipe, bs 2048) — read-out at epoch 39 (2026-08-29 14:00)

Launched 2026-08-27 22:36 on sess5 (recorded by the other session): H′ recipe (seed
anchors d0/z0/φ/θ, absolute q/p, Fourier 2⁻¹⁰…2⁵, weights 0.1/1/0.1/1/1, NO seed-
residual inputs), B3 stores (3 T, geometry order), OneCycle 50 ep at bs 2048
(~57 min/epoch, ~48 h; finish ~2026-08-29 22:30). Val curves: Q′'s are on the mixed
val set (yardstick, §4.15); Qm's muon val q/p went 1.64 (ep 19) → 1.29 (29) → 1.15
(39) → 1.12 (43), still ~1 %/epoch. Epoch-39 checkpoints evaluated on the **v2 farm**
(`eval_plots/sweep3/{Qp,Qm}_ep39_v2eval/`), SSM / truth-KF:

| set | Q′ (mixed) d0 z0 φ θ q/p | GM5 | Qm (muon-only) d0 z0 φ θ q/p | GM5 |
|---|---|---|---|---|
| µ 100 GeV | 1.00 1.25 1.24 1.16 **1.05** | 1.14 | 0.99 1.25 1.12 1.16 **1.01** | 1.10 |
| µ 10 GeV | 1.03 1.13 1.09 1.11 1.14 | 1.10 | 1.03 1.14 1.06 1.11 1.14 | 1.09 |
| µ 2 GeV | 1.01 1.30 1.04 1.28 1.16 | 1.15 | 1.01 1.30 1.06 1.28 1.22 | 1.17 |
| µ uniform | 1.05 1.21 1.24 1.14 1.18 | 1.16 | 1.04 1.21 1.16 1.14 1.14 | 1.14 |
| ttbar_new_pt1 | 1.01 1.24 1.05 1.25 **1.17** | 1.14 | 1.03 1.25 1.09 1.26 **1.28** | 1.18 |

**Final (epoch 49) read-out, 2026-08-30 00:10 (`eval_plots/sweep3/{Qp,Qm}_ep49_v2eval/`), q/p ratios:**
Q′ 1.14 / 1.12 / **1.00** / 1.13 / 1.14 and Qm 1.16 / 1.09 / **0.96** / 1.10 / 1.22 for 2 GeV /
10 GeV / 100 GeV / uniform / ttbar_new_pt1 (GM5 Q′ 1.14 / 1.09 / 1.09 / 1.14 / 1.13; Qm 1.15 /
1.08 / 1.08 / 1.12 / 1.16); the z0/θ floor (1.2–1.3) and φ (1.03–1.17) unchanged = the H′
representation. So the absolute q/p head reaches the KF at high pT after 50 bs-2048 epochs
and 1.1–1.2× at low pT; the mix buys 7 % on hadron q/p (1.14 vs 1.22) for 3 % on muon q/p.
Readings: (1) **q/p closes with the long small-batch anneal**: 1.0–1.2× at epoch 39
(vs 1.49 for H′ at 20 ep bs 36 k, 1.12 for M′ at 20 ep bs 2048), still falling →
~1.05–1.1 at epoch 49; the absolute head needs ~40 bs-2048 epochs. (2) **First clean
mixed-vs-muon-only comparison** (same recipe, same epochs): the mix costs the muons
nothing on d0/z0/θ, ~7 % on high-pT φ (1.24 vs 1.16; at the run-to-run noise floor)
and 4 % on uniform q/p, and buys 9 % on hadron q/p (1.17 vs 1.28) and 5 % on 2 GeV
q/p. (3) The z0/θ floor at 1.2–1.3× and φ at 1.1–1.2× are the H′ representation
(no residual inputs) and the 0.1 φ weight — both fixed in sweep 4 (W1 shows z0/θ/φ
at 1.00 before its anneal). Sweep 4's open question is therefore only how long an
anneal q/p needs: 10 epochs (W/X/Y) vs 25 (Z) vs the ~40 seen here.

### 4.19 ROADMAP after parity with the truth-KF (user ideas, 2026-08-29 — a reminder for the next chats)

Trigger: when a run is at ≤ 1.0–1.05× the truth-KF on **every** test set (muon 2 / 10 /
100 GeV, uniform, ttbar_new_pt1) on all five parameters incl. q/p. Status 2026-08-29:
geometry parameters are there (W1 stage 1, P′); q/p needs the long/small-batch anneal
(§4.18) or the scale-free head (Y) — read sweep 4 first.

1. **Beat the KF, not just match it — a fine-tune stage**: Muon(-hybrid) optimiser +
   *larger* batch fine-tune from the best checkpoint (the legacy second stage), few
   epochs, same head/ranges; also try it from the Q′/Qm 50-epoch checkpoints. Goal:
   ratios < 1 where the KF's material model is the limit (low-pT hadrons).
2. **Then simplify the model along several axes** (the recipe is fixed at that point,
   so each is a one-variable ablation against the parity model):
   - fewer layers (4 → 3 → 2; 3L proven free 2026-08-31). **User rule (2026-08-31): if 2L
     is under-parameterised, do NOT drop further layers — instead shrink the 3L config to
     dim 96; and a LARGER d_state is basically free at our sequence lengths (user measured),
     so d_state is the last thing to cut.**
   - smaller embedding dim (128 → 96 → 64) and d_state (64 → 32 → 16) — but see the rule
     above, and the dim-96 divergence of §4.22 (halve the peak LR when narrowing),
   - drop the depthwise causal conv in the Mamba block (d_conv 4 → 1 / remove),
   - fewer hit features: the 12 absolute features are largely redundant (r, φ_hit,
     θ_hit, s, η_hit are functions of x, y, z; layer/surface ids vs volume) — try
     xyz + volume + the 3 seed residuals only (P″ = residuals-only is the extreme),
   - precision: train end-to-end in TF32 (`TRK_MATMUL_PRECISION=high`; inference TF32
     already measured harmless to ≤ 2 %, §4.12), then *try* lower (bf16 autocast in
     the GEMMs / the scan) — the user's standing rule "nothing below TF32" applies
     until parity is established and the physics check is repeated per step,
   - kernel: the bucket-16 / compiled front-end switches (§4.16) become the default
     inference path; merged direction launch and GEMM fusion are the next steps.
   Decision metric for every simplification: per-domain GM5 vs the parity model
   (must stay ≤ +2 %) and inference tracks/s (`bench_infer_flat.py --gpu-seed`).
3. Data side (parallel): the 300 M log-pT muon gun (1–300 GeV, 3 T, truth_tracks,
   unique event ids; check whether the strip time is fixed there), pile-up ttbar.

### 4.20 Sweep-4 results (2026-08-29 23:50) and SWEEP 5 (the night of 2026-08-29)

Stage-2 finals (hybrid 15 ep bs 36 k → 10 ep bs 2048; `eval_plots/sweep4/{W2,X2,Y2}/`,
v2 farm, SSM / truth-KF post-clip; Z at epoch 24/25, evaluated tomorrow):

| run | 2 GeV | 10 GeV | 100 GeV | uniform | ttbar_new_pt1 | what it says |
|---|---|---|---|---|---|---|
| **W** absolute q/p | d0/z0/φ/θ **0.99–1.02**, q/p 8.96 · GM5 1.55 | q/p 3.98 · 1.32 | q/p 5.87 · 1.43 | q/p 5.61 · 1.42 | q/p 9.14 · 1.56 | geometry solved; the absolute q/p head does not converge in a 10-epoch anneal |
| X no Fourier | q/p 13.2 · 1.68 | q/p 78.8 · 2.46 | φ 1.16, q/p 34.6 · 2.17 | q/p 43 · 2.23 | q/p 15.6 · 1.74 | Fourier is still needed (q/p, high-pT φ) |
| **Y scale-free q/p** | **1.04** (q/p 1.26) | **1.03** (q/p 1.18) | **1.03** (q/p 1.13) | **1.04** (q/p 1.19) | **1.04** (q/p 1.27) | **the recipe**: d0/z0/φ/θ 0.99–1.02 everywhere, q/p 1.13–1.27 |

Y's val q/p plateaued inside its anneal (1.27e-3 → 1.23e-3 over 10 epochs, +0.1 % in the
last), so a longer identical anneal is not the lever; Z (pure bs 2048, absolute head) reached
1.21e-3 at epoch 23 vs W2's 1.44e-3 → pure small batch beats the hybrid by ~15 % on q/p.
Q′/Qm epoch-49 evaluations: `eval_plots/sweep3/{Qp,Qm}_ep49_v2eval/` (running at write time).

**Sweep 5** (configs `ICLR_sweep5/`, Y recipe throughout, v2 mixed store; user authorised
"more epochs / small-batch anneal / large-batch optimizer switch", and downsizing once geometry
matched):

| run | GPU | change | question |
|---|---|---|---|
| **YZ** | 0 | pure bs 2048, OneCycle 25 ep (~26 h) | pure small batch on the scale-free head: q/p → 1.0? (Z-analogue) |
| **Y-FT** | 1 | Muon-hybrid + WSD, bs 36 k, 12 ep from Y2 `last.ckpt` (~8.5 h) | the legacy second stage / optimizer switch on q/p |
| **Y-3L** | 2 | 3 layers, hybrid 15+10 (~20 h) | downsizing: depth (≤ +2 % GM5 allowed) |
| **Y-d96** | 3 (after Z) | dim 96 / d_state 32 (3 heads), hybrid 15+10 (~20 h) | downsizing: width + state |
**LAUNCHED 2026-08-30 00:01–00:02 (sess3, `ICLR_sweep5/launch_sess3.sh`, dry runs passed):**
YZ GPU 0 (run `700c30c9…`, ~63 min/epoch → ~02:00 on 08-31), Y-FT GPU 1
(run `57695c71…`, from Y2 `last.ckpt`, ~41 min/epoch → ~08:30), Y-3L stage 1
GPU 2 (run `a6219d6f…`, chain → Y3L2, done ~17:30), Y-d96 queued behind Z on
GPU 3 (starts ~00:30, chain → Yd962, done ~18:00). Z's `last.ckpt` is auto-evaluated on GPU 0
when it finishes (`eval_plots/sweep4/Z/`). Logs `launch_logs/sweep5/`, Comet `SW5-*`.
Q′/Qm epoch-49 final numbers: §4.18.
Read-out: `04_eval_ckpt_iclr.sh <run> last.ckpt <out> /scratch/colliderml/ICLR_eval_v2`; for
the two small models also `bench_infer_flat.py --gpu-seed` tracks/s vs Y. Not run: a
longer identical anneal (Y-L config exists), seed twin, Hq, P″.

### 4.21 Z: PARITY WITH THE TRUTH-KF (2026-08-30 00:35) — and the re-planned sweep 5b

**Z** (W recipe: P′ inputs, even weights, φ quantile head, ABSOLUTE q/p head, v2 mixed store,
**pure bs 2048, OneCycle 25 epochs**, 26 h) evaluated on the v2 farm (`eval_plots/sweep4/Z/`):

| set | d0 | z0 | φ | θ | q/p | GM5 |
|---|---|---|---|---|---|---|
| µ 2 GeV | 0.99 | 0.99 | 0.99 | 0.98 | 1.09 | 1.01 |
| µ 10 GeV | 0.99 | 0.99 | 0.99 | 0.98 | 1.08 | 1.00 |
| µ 100 GeV | **0.92** | 0.99 | **0.85** | 0.97 | **0.92** | **0.93** |
| µ uniform | 0.97 | 0.99 | 0.95 | 0.98 | 1.02 | 0.98 |
| ttbar_new_pt1 | 0.98 | 0.98 | 0.99 | 0.98 | 1.10 | 1.01 |

At or below the truth-KF on every parameter and every set except q/p at low pT / hadrons
(1.08–1.10). Same wall-clock as the hybrid W (26 vs 20 h) but W's q/p was 4–9× → **pure small
batch for 25 epochs is the recipe**; the hybrid is only a cost saver and loses q/p. The φ 0.85
and q/p 0.92 at 100 GeV (below the KF) need the pre-clip / per-pT check before being quoted.

**Sweep 5b (re-launched 2026-08-30 00:50 after killing Y-FT, Y-3L and the queued Y-d96, ~40 min
in):** YZ kept on GPU 0 (pure 2048 + scale-free q/p, the expected low-pT q/p fix); **Z-FT** GPU 1 =
Muon-hybrid + WSD large-batch (36 k) fine-tune from Z's `last.ckpt`, 12 ep (roadmap item 1:
below the KF?); **YZ-3L** GPU 2 and **YZ-d96** GPU 3 = the pure-2048 scale-free recipe with 3
layers / dim 96 + d_state 32 (downsizing on the established recipe; ~26 h each, faster per
epoch). Configs `ICLR_sweep5/{YZ3L,YZd96,ZFT}_*.yaml`; the hybrid Y-3L/Y-d96/Y-FT configs stay
unused. Read-out as before on the v2 farm + `bench_infer_flat.py --gpu-seed` for the small models.

**Z-FT read-out (2026-08-30 10:40, `eval_plots/sweep5/ZFT/`):** the Muon-hybrid + WSD large-batch
fine-tune (12 ep bs 36 k from Z) changes nothing on d0/z0/φ/θ and improves q/p by 1–2 % everywhere:
2 GeV 1.09 → 1.08, 10 GeV 1.08 → 1.06, 100 GeV 0.92 → 0.90, uniform 1.02 → 1.00, ttbar_new_pt1
1.10 → 1.09 (GM5 unchanged to ±0.01). The legacy second stage is a polish, not a lever: the
remaining low-pT/hadron q/p gap (1.06–1.09) is for the scale-free head (YZ, running).

**Zmu launched 2026-08-30 10:59 on sess3 GPU 1** (run `a228d56a…`, pid 3390189, `Zmu_ref_muononly_bs2048_onecycle25.yaml`
= Z on the v2 muon-only store, 93,520 steps/epoch, ~24 h): the mixing control on the final recipe (Z vs Zmu).
**sess5 — LAUNCHED 2026-08-30 11:49 (user "go"), after `10_stage_v2_on_sess5.sh` (v2 stores + farm + mixed
store staged 11:14; superseded B3/2 T/old-uniform stores on sess5 deleted) and passing dry runs:** GPU 0 →
**YZ50** (`YZ50_qrel_bs2048_onecycle50`, run `cc3917b1…`, pid 850674: Z's schedule stretched to 50 epochs with the
scale-free q/p head — chosen because YZ's pooled-val q/p at epoch 10 already equalled Z's final value), GPU 1 →
**ZFT50** (`ZFT50_ref_lion_bs2048_onecycle50_fromZ`, run `d1d3435c…`, pid 850817: 50 more small-batch epochs from
Z's `last.ckpt`, Lion OneCycle 2e-6 → 2e-5 → 1e-6, the user's fine-tune idea). Both 28 it/s → ~60 min/epoch →
~52 h, checkpoints every 10 epochs; finish ≈ 2026-09-01 16:00. Evaluate with the local `ICLR_eval_v2` farm.
`Z50` (absolute head) stays as the unused alternative. Z's checkpoint: `logs/comet_offline/063276b01d4848f2b4eb4eceb07c1833/ckpts/last.ckpt`.

### 4.22 PARITY CLOSED (2026-08-31 morning): YZ ≤ truth-KF on every parameter and every test set

Evals `eval_plots/sweep5/{YZ,YZ3L,YZd96,ZFT,Zmu,YZ50_ep19,ZFT50_ep19}/` (v2 farm, post-clip SSM/truth-KF):

| run | 2 GeV | 10 GeV | 100 GeV | uniform | ttbar_new_pt1 |
|---|---|---|---|---|---|
| **YZ** (scale-free q/p, pure bs 2048, 25 ep) | GM5 **0.99** (q/p 1.02) | **0.99** (1.00) | **0.91** (0.85) | **0.97** (0.97) | **0.99** (1.03) |
| **YZ-3L** (3 layers) | 1.00 (1.02) | 0.99 (1.00) | 0.91 (0.85) | 0.97 (0.97) | 0.99 (1.03) |
| Z / Z-FT (absolute head) | 1.01 (1.09/1.08) | 1.00 (1.08/1.06) | 0.93/0.92 (0.92/0.90) | 0.98 (1.02/1.00) | 1.01/1.00 (1.10/1.09) |
| YZ50 @ epoch 19/50 (mid-cycle) | 1.00 (1.04) | 1.00 (1.03) | 0.94 (0.91) | 0.98 (1.01) | 1.00 (1.06) |
| ZFT50 @ epoch 19/50 (mid-cycle, hot phase) | 1.03 (1.18) | 1.03 (1.20) | 0.98 (1.17) | 1.02 (1.22) | 1.03 (1.20) |
| YZ-d96 (dim 96 / ds 32) | **DIVERGED** (ratios 19–650×) | | | | |

- **The q/p gap is closed**: YZ (and its 3-layer twin) are at or below the truth-KF everywhere;
  worst q/p 1.03 (ttbar), 100 GeV at 0.85–0.92 (below the KF — pre-clip/per-pT check before quoting).
- **3 layers are free** (YZ-3L ≡ YZ to ±0.01) → next depth step (2L) worth trying.
- **YZ-d96 diverged slowly** (val d0 32 µm at epoch 0 → 2.3 mm at epoch 24, no NaN, train loss stuck
  at 0.016): the OneCycle peak (5e-5) is too hot for dim 96/ds 32 → rerun with peak ~2.5e-5.
- Zmu (muon-only twin of Z) finishes ~09:30, auto-evaluated → the mixing verdict on the final recipe.
- **New data**: `single_muon_loguniform` (200 shards, log-pT 0.9–110 GeV, produced 08-26) downloading
  to `drift_beamspot_v2/`; preprocessing (v2 flags) + /eos publish chained. Expected ~29 M tracks in
  1–2 GeV (vs 1.76 M in the uniform set): the low-pT muon statistics fix.

### 4.23 Layer-count throughput (2026-08-31, one H100 NVL, 32 k tracks/batch, GPU seed in the timed loop; `eval_plots/sweep5/bench_layers.txt`)

| model (dim 128, d_state 64) | strict fp32 | TF32 matmuls | TF32 + bucket16 + compiled front-end |
|---|---|---|---|
| 4L (YZ) | 429 k tr/s | 657 k | 730 k |
| **3L (YZ-3L, physics ≡ 4L)** | **540 k** | **816 k** | **1.05 M** |
| 2L (untrained, timing only) | 741 k | 1.06 M | **1.36 M** |

Scaling ≈ proportional to layer count. The 3-layer parity model in deployment mode
(TF32 + kernel switches) fits **>1 M tracks/s per H100**. User rule (§4.19): the 2L keeps
dim 128 / d_state 64 — if it is under-parameterised, shrink the 3L to dim 96 instead of
going below 2 layers; larger d_state is ~free at these sequence lengths.
ROUND 6 (running/queued): YZ-mix3 (GPU 0) and YZ-2L-mix3 (GPU 1) — 25 ep bs 2048 on the
THREE-WAY mix (uniform + log-pT 0.9–110 + ttbar, ~397 M tracks, ~45 h); **YZ-3L-FT** (GPUs 2+3,
DDP, bs 2×20 k, Muon-hybrid WSD 50 ep from YZ-3L `last.ckpt`, user go) — all auto-launching
when the loguniform store + mix3 finish (`round6_chain.sh`, `YZ3LFT_launch_waiter`).

### 4.24 Round-6 results (2026-09-01 evening) — BEST MODEL = YZ-3L-FT; every new run ≤ truth-KF everywhere

Finished 2026-09-01: YZ50 (4L, scale-free q/p, pure bs 2048, 50 ep, v2 2-way mix, run
`cc3917b1`), ZFT50 (Z + 50 more bs-2048 Lion epochs, run `d1d3435c`), YZ-3L-FT (3L, Muon-hybrid
WSD DDP 2×20 k, 50 ep on mix3 from YZ-3L, run `7e50a1db`), YZ-2L-mix3 (2L/128/ds64, 25 ep bs
2048 on mix3, run `ebf7104a`). YZ-mix3 (4L on mix3) still running (ep 19/25, done ~01:00 09-02).
Evals `eval_plots/round6/<run>/plots/` (v2 farm). Post-clip SSM/truth-KF — all four runs are
**at or below the truth-KF on every parameter of every test set**, differences between runs
±0.01–0.02 (saturated at the KF/target-error level): GM5 0.99 / 0.99 / 0.91 / 0.97 / 0.99
(2 GeV / 10 GeV / 100 GeV / uniform / ttbar_new_pt1) for all of YZ50, YZ-3L-FT, YZ-2L-mix3;
worst single q/p 1.01 (YZ50 ttbar). The discriminator is the PRE-clip (tail-inclusive) table:
**YZ-3L-FT is the only run with no parameter above 1.0 anywhere** (GM5 0.95 / 0.94 / 0.82 /
0.92 / 0.89; e.g. ttbar q/p 0.79, 100 GeV z0 0.70), while YZ50 has 2 GeV θ 1.20 / φ 1.03 tails.
YZ-2L-mix3 pre-clip ≡ YZ-3L-FT to ±0.01 → **2 layers suffice on the mix3 data** (the 25-epoch
2L matches the 50-epoch fine-tuned 3L; the 4L is not needed). Best model for the paper:
**YZ-3L-FT** (best/equal physics incl. tails + 1.05 M tracks/s deployment); YZ-2L-mix3 is the
efficiency headline (1.36 M tracks/s, same physics). Table generator:
scratchpad `round6_table.py` (post|pre) over the `rms_summary.json` files.

### 4.25 Paper plots + official-pipeline status (2026-09-01)

- `fast_rms_eval.py` no longer draws the CKF as the extra green (C2) series when the truth-KF
  exists (user decision: truth-KF is available for all v2 test sets). The DM-subset definition
  is UNCHANGED (still CKF-matched ∧ truth-KF-finite) so all campaign tables stay comparable —
  revisiting the DM restriction is still open (§4.10 caveat).
- Paper bundle for the best network: `eval_plots/paper/YZ3LFT/` (6 datasets × 4 figure modes,
  SSM + truth-KF only, same unbinned μ averages in the legends).
- pyacts **47.6.1** (latest, 2026-09) installed into `/shared/tracking/pyacts_env/`
  (venv, python 3.13 — separate from the pixi training env; `import acts` + `acts.examples`
  incl. Eff/Fake/DuplicationPlotTool configs verified). `scripts/acts_integration.py` (runs the
  SSM as an ACTS IAlgorithm so ACTS's own performance writers evaluate it like the KF) needs,
  besides raw parquet `particles`/`tracker_hits` dirs (have: `/scratch/colliderml/
  drift_beamspot_v2/`), the geometry assets `odd.json`, `gen3_material_map_map.json`,
  `geoid_map.csv`, `odd-seeding-config-gen3.json` (referenced as `~/odd-json/`) — NOT on sess3;
  they live wherever the acts_integration work ran. The collaborators' "official plotting
  pipeline" link in the user's request was missing; the ColliderML GitHub repo
  (OpenDataDetector/ColliderML) contains no tracking-performance plotting — awaiting the link.
  **Resolved 2026-09-01 evening (user):** the official script is
  `/eos/user/b/bhuth/jonathan_ssm/acts_integration.py` — synced into `scripts/acts_integration.py`
  and minimally adapted for v2 checkpoints (`--sort-key geometry`, `--seed-residual-features`,
  auto seed anchors from the loss module, `--d0-max/--z0-max`, two path fixes;
  `scripts/perf/common.pyc` = the sourceless helper module it imports). pyacts 47.6.1 venv
  `/shared/tracking/pyacts_env` verified (imports the adapted script with torch 2.13 + CUDA).
  **BLOCKED on the EOS ACL of `/eos/user/b/bhuth/odd-json/`** (odd.json 8.6 MB,
  gen3_material_map_map.json 260 MB, geoid_map.csv, odd-seeding-config-gen3.json — exist there,
  unreadable by us; nowhere else: portal, wheels, colliderml package all checked). Ask bhuth to
  open read or copy into `jonathan_ssm/`; then run per dataset on the raw v2 parquet
  (`--particles-dir/<hits-dir>` per run dir) → `eval_plots/paper/acts_official/<dataset>/`.

### 4.26 Why the SSM beats the truth-KF at 100 GeV (2026-09-01, user question) — forward region + tails, not a bias

Forensics on YZ-3L-FT residuals (scratch `analyze_100gev.py`; 2 GeV as control): both estimators
unbiased at 100 GeV (medians ≲ 0.1 µm / 0.001 mrad, per-η medians noise) → NOT a target-convention
artifact. Per |η| the core-width ratio is 0.94–1.00 in the barrel and collapses forward:
d0 0.79, φ 0.70, q/p 0.58 at |η| 2.5–3.0; q99 ratios reach 0.44–0.65 forward; the truth-KF's
z0 q99.9 is 374 µm vs the SSM's 131 µm, and its core kurtosis is higher on every gaining
parameter. θ is the control: corr(SSM,tKF) = 0.989, ratio 1.00 in the barrel — where information
is shared and Gaussian there is no gain. At 2 GeV (MS-dominated, honestly Gaussian process noise)
ratios are 0.98–0.99 with corr 0.9–0.97 — the KF is at the information floor there, as expected.
Reading: at high pT the resolution is measurement-model dominated, and the KF is only optimal for
Gaussian errors with correct covariances — shallow-incidence endcap strips (1-D measurements),
pitch-scale digitisation and delta-ray outliers violate that; the pinball-trained network is
median-robust. Paper phrasing: "matches the KF in the barrel, more robust forward and in the
tails"; "truth-KF as configured" (no outlier rejection on truth hits) is not the theoretical
bound forward.

### 4.27 SWEEP 7 — the speed round (launched 2026-09-01 evening, user go; configs `config/ssm_cls/ICLR_sweep7/`)

All on the full three-way mix `ICLR_retraining_v2_mix3` (user rule). Untrained-variant throughput
measured first (`eval_plots/round6/bench_2L_variants.txt`, 32 k batch, GPU seed): 2L base
741 k fp32 / 1.36 M deployed; **dim 96 960 k / 1.54 M; d_conv 1 800 k / 1.60 M; Fourier 8 scales
791 k / 1.46 M**. Decision metric per §4.19: per-domain GM5 vs YZ-3L-FT (≤ +2 %, no pre-clip > 1.0)
+ tracks/s.

| run | GPU | change vs YZ-2L-mix3 | question |
|---|---|---|---|
| **R2L-FT** | sess3 1+2 (DDP) | Muon-hybrid+WSD 50 ep from its `last.ckpt` (the YZ-3L-FT recipe) | closes 2L's last q/p 1.005/1.013 → the 2L becomes the paper model (1.36 M tr/s) |
| **R2L-noconv** | sess3 3 | `d_conv: 1`, 25 ep bs 2048 | conv-free block at 1.60 M tr/s deployed |
| **R2L-d96** | sess3 0 (queued behind YZ-mix3 + its auto-eval, `queue_d96_gpu0.sh`) | dim 96, OneCycle peak HALVED to 2.5e-5 (§4.22 divergence) | width cut at 1.54 M tr/s |
| **R1L** | sess5 (after the kernel work) | `num_layers: 1` | where depth finally breaks (user) |
| residuals-only P″ | sess5 | 12 absolute features dropped (needs a small flat_data option — to implement on sess5) | feature redundancy + collate cost |
| fourier8 | ON HOLD | 16 → 8 scales (+8 %) | next iteration (7-GPU budget) |

Kernel work (sess5, before R1L): merged bidirectional launch, two-tracks-per-16-row-tile packing,
CUDA graphs for the ≤ 2 k-batch regime; parity via `tests/test_ssd_variants.py`.

**sess5 execution (2026-09-01 late evening → 09-02 00:30):** loguniform store staged from /eos
(147 GB, 120 s), mix3 rebuilt (387,290,426 train tracks — identical to sess3);
`hit_feature_columns` collate option added for P″ (`flat_data._pack` + DataModule +
`tests/test_flat_data.py::test_pack_hit_feature_columns_subset`); **R1L (GPU 0, 47 it/s) and
R2L-resonly (GPU 1, 42 it/s) launched 21:46 on mix3** — dry runs passed (P″ needed `input_fields`
subset to `[volume_id, du_asinh, dv_asinh, s_helix]`), done ~Sep 3 morning. Kernel results:
1. **Merged bidirectional launch** (`TRK_SSD_MERGED_BIDI=1`, `_ssd_short_fwd_kernel2pm`: one fused
   in-proj GEMM (T, 2·dproj) + ONE scan launch, grid axis 2 = direction, runtime reversal, stacked
   weights; `mamba_short.fused_bidi_scan_packed` branches on the env): parity 9/9
   (`test_ssd_variants.py`, incl. full-layer end-to-end), but **a wash at 32 k** (fp32 702 k vs
   747 k base; deploy 1.40 M vs 1.42 M) and **+3 % at bs 2048** — keep opt-in, not recommended.
2. **CUDA graphs** (`bench_infer_flat.py --cuda-graph`): static shapes via **dummy-track padding**
   (pad tokens grouped into K ≤ 20-token dummy segments appended after the real tracks, outputs
   discarded — provably inert: graph-vs-eager max |Δpred| = **0.0**, checked per run). At bs 2048
   (measured UNDER training contention on the same GPU, both sides equally): strict fp32
   **515 k vs 336 k eager (+53 %)**, deploy (GPU seed + TF32) **543 k vs 405 k (+34 %)** — the
   launch-gap hypothesis of §4.12 confirmed; clean uncontended numbers pending a free GPU.
   Incompatible with `TRK_SSD_BUCKET16` (device-sync split).
3. **Five graph-blocking host syncs removed** (all numerically identical, default-path safe):
   `mamba_cls` aug_cu scalar write → `F.pad(cumsum)`; `seed_torch` `repeat_interleave(lens)` →
   `bucketize` (static shape), `valid[..] = True` → tensor RHS, `torch.tensor(inf)`/scalar puts in
   `select_triplet_torch` → scalar-overload where/scatter_; `losses` median-quantile argmin →
   cached Python int. `gpu_seed_features(..., max_len=20)` = static-shape (graph-safe) mode.
4. **Real robustness bug found & fixed**: the packed path's DDP unused-parameter tie
   (`cls_concat + 0.0 * x_hits.float().sum()`) is a GLOBAL sum — one NaN token anywhere poisoned
   every track's pooled output in the batch at inference. Now applied in training mode only.
   Affected tests all pass (38 passed, 2 host-specific skips).
Not done: two-tracks-per-16-row-tile packing; uncontended graph benches; graphs + bucket16.

### 4.28 Sweep-7 results (2026-09-03 morning; d96/noconv evals pending) + the official plot factory

Post-clip SSM/truth-KF GM5 (2 GeV / 10 GeV / 100 GeV / uniform / ttbar_new_pt1), v2 farm:
- **R2L-FT** (2L Muon-FT 50 ep, mix3): 0.988 / 0.986 / 0.917 / 0.972 / 0.987 — ≡ YZ-3L-FT to ±0.005,
  pre-clip all ≤ 1.0; **paper model switched to the 2-layer fine-tuned network (user 2026-09-02)**;
  our-style bundle `eval_plots/paper/R2LFT/`.
- **R1L** (ONE layer, 25 ep mix3): geometry 0.95–1.00 everywhere; only q/p degrades (1.04–1.10 post,
  ≤ 1.03 pre-clip except none) — depth "breaks" only in q/p, by 4–10 %. One bidirectional pass
  almost suffices.
- **R2L-resonly (P″)**: exactly the §"can residuals-only work" prediction — d0/z0/φ/θ at the KF
  (post 0.99–1.11, pre ≤ 1.03), q/p catastrophic with the predicted pT pattern (post 1.41 → 2.72 →
  2.88 at 2/10/100 GeV; pre 8.7× at 10 GeV): the scale-free q/p head cannot see the seed-curvature
  denominator without absolute scale. Next: **P‴ = residual features + the 5 seed params as global
  inputs** (the KF's exact information set, still no absolute hit coordinates).

**Official-pipeline plot factory (2026-09-02/03)**: `run_acts_official_plots.sh` per dataset+model →
`eval_plots/paper_plots/{acts_official,acts_official_R2LFT}/<ds>/`: official `resolutions.pdf`;
legacy-design pages from the pipeline's own tracks (`<ds>_acts__rms_vs_eta_summary*`,
`__residual_hist_{liny,logy}` — unbinned µ + kept-after-clipping counts in legends; data via
`--dump-residuals` → `matched_residuals.npz`: SSM associated by prototrack order (hard-checked),
KF by unique ΔR < 0.05); band pages (`__res_vs_{eta,pt}_bands`) with the writers' fit-σ errors as
legacy-style shaded bands (NOT bootstrap), vs-pT + ratio panel ONLY for `single_muon_uniform`
(200 k-event slice; fixed-pT vs-pT pages are single-bin nonsense). Fixes en route: per-dataset
`--residual-window` (stock qop ±0.1 window ≫ core → σ=0/spiked fits), per-GPU `TRITON_CACHE_DIR`
(parallel runs raced on AFS ~/.triton → PermissionError), `--hit-bounds-tolerance 25` +
`filter_acts_hits.py` (4 wrong-surface hits in 45 M ttbar aborted runs — report to producers along
with the v2 schema break: `make_acts_compat_parquet.py` shims nested particle_ids/simhit truth back
to the Release-1 layout the pyacts converter expects). Official ttbar numbers (both models, 237.7 k
tracks): SSM matched 100.0 % vs ACTS-KF 74.6 %, Gaussian σ ratios 0.83–0.92 on all five params.

### 4.28b Gradient-cosine probe rerun at 450x2048 — the (phi,q/p) block does NOT exist (2026-09-03, user request)

`scripts/grad_cos_probe.py` on the paper checkpoint (`eval_plots/sweep7/R2LFT/ckpts/model.ckpt`,
verified byte-identical to `logs/comet_offline/ae2dc434.../ckpts/last.ckpt` = the FINAL R2L-FT
checkpoint), 450 x 2048 = **921,600 ttbar_new_pt1 tracks** (was 20 x 2048), out in
`eval_plots/paper_plots_extras/grad_cos_R2LFT_450x2048/`.

**The old figure's error bars were the wrong quantity.** `paper_matrices.py` annotated
`mean +- std`, where `std` is the scatter of SINGLE-BATCH cosines — a per-batch gradient-noise
floor set by the batch size, which does **not** shrink with more batches. The uncertainty on the
plotted mean is `std/sqrt(N)`; at N=20 that was +-0.076, not the +-0.33 printed. Both are now
written out (plus the full per-batch stack, so anything can be re-derived without a rerun) and the
figure annotates the s.e.m.

New numbers (mean +- s.e.m.): the matrix splits into a geometry block and curvature.
`(d0,phi) +0.797+-0.008` (101 sigma), `(z0,theta) +0.361+-0.011` (32 sigma),
`(phi,theta) +0.232+-0.014`, `(d0,theta) +0.228+-0.015`, `(z0,phi) +0.198+-0.014`,
`(d0,z0) +0.190+-0.015`; q/p nearly orthogonal to all four (`|C| <= 0.065`,
`(theta,qop) -0.000+-0.003`). **The paper's third headline pairing, "(phi,qop) bending at +0.11",
is +0.065+-0.005 and is not a leading entry** — `sections/results.tex` + caption rewritten to the
geometry-block / q/p-orthogonal framing, `sections/interpretability.tex` (dormant) flagged stale.
`(d0,theta)` moved +0.095 -> +0.228 and `(d0,z0)` +0.129 -> +0.190: both ~2 sigma of the old
s.e.m., i.e. the 20-batch numbers were simply noise-limited.

Estimator checks done: batches are iid (lag-1 autocorr <= 0.08; chunk-to-chunk variance = the iid
prediction to x0.93-1.00, so no part-level correlation inflating the s.e.m.); the eval sample comes
from `ttbar_new_eval/v1`, disjoint from the `ttbar_new_train/v1` source in mix3 — no leakage.

**A `pooled` estimator was added and is CONFOUNDED — do not quote it.** Cosine of gradients summed
over all 921,600 tracks looks far stronger (geometry block +0.44..+0.95, `(d0,phi) +0.952`), but
`systematic_frac` is only 0.12-0.19 (1.0 = a fixed direction every batch, 0.047 = pure noise at
N=450) and projecting out the direction common to all five tasks collapses it:
`(d0,z0) +0.67 -> -0.12`, `(z0,theta) +0.44 -> -0.29`, `(phi,qop) +0.09 -> -0.55`. The pooled block
structure is almost entirely ONE shared direction — expected, since the model was fine-tuned on
mix3 and probed on held-out ttbar under a plain-sum objective, so residual gradients roughly cancel.
`mean` measures the correlation of per-track gradient FLUCTUATIONS (the GradNorm/PCGrad
multi-task-interference quantity the paper cites) and is immune to a common offset. Only
`(d0,phi)` survives all three views (+0.797 / +0.952 / +0.734). Kept in the npz as a diagnostic.

Probe fixes alongside: one forward + five `autograd.grad` instead of 5 redundant forwards (and all
five gradients now come from one identical forward); `std` via `np.std(ddof=1)` instead of the
ill-conditioned `E[x^2]-E[x]^2`; `valid_mask=track_valid` passed as in training (all-True here, so
a no-op); dead `head_prefixes` removed — it implied `pool_head` was excluded from the trunk when it
never was (trunk = 632,928 params, unchanged). Refactor validated: reproduces the published 20-batch
mean matrix to all three printed decimals.

### 4.29 The pipeline's ad-hoc ACTS KF is MISCALIBRATED (2026-09-03, user suspicion confirmed)

Three-way check on uniform muons (200 k events, common subset 143 k tracks, identical residuals
+ identical ACTS scipy-Gaussian estimator + identical windows; offline estimator validated
against the writer's own reswidth curves — overlay exact):
`eval_plots/paper_plots/acts_kf_check/uniform_kf_calibration_check.pdf` + scratch `acts_kf_check.py`.

| σ (integrated) | SSM | pipeline KF | production truth-KF | pipe/prod | SSM/prod |
|---|---|---|---|---|---|
| d0 [µm] | 15.0 | 19.4 | 15.3 | 1.27 | 0.98 |
| z0 [µm] | 23.4 | 44.7 | 23.4 | **1.91** | 1.00 |
| φ [mrad] | 0.294 | 0.356 | 0.299 | 1.19 | 0.99 |
| θ [mrad] | 0.115 | 0.158 | 0.115 | 1.37 | 1.00 |
| q/p [GeV⁻¹] | 2.71e-4 | 3.28e-4 | 2.71e-4 | 1.21 | 1.00 |

Structure: θ excess is barrel-peaked (3.2× at η=0 → 1.2 forward), z0 has two degenerate-fit
spikes at |η|≈0.4 (20–30×) — the longitudinal/strip-v measurement model; d0/φ/q/p flat ~1.2×.
Mechanism: the Release-1 converter never receives the v2 per-cluster variances
(`var_loc0/var_loc1` are not in its hitSchema) and re-derives local coords + covariances from
the shimmed GLOBAL positions with internal defaults, while the production `truth_tracks` were
fitted on the real digitized measurements. Its truth-ESTIMATED seeding also only fits 72–75 %.
**Paper rule: quote SSM vs the PRODUCTION truth-KF (0.98–1.00 under the ACTS estimator —
consistent with all campaign tables); do NOT sell the 0.83–0.92 ratios vs the in-pipeline KF
as "beats the ACTS KF" — label that baseline "ACTS KF refit as configured in-pipeline" and
report the calibration caveat to the ColliderML/pyacts producers (converter should ingest the
digitized loc/var columns).**

### 4.30 Paper switched to the shipped truth-KF reference (2026-09-03 evening) — plots + text done

Executed the final instruction of `docs/BUGREPORT_acts_pipeline_kf.md` (drop the in-pipeline
ad-hoc KF as a baseline everywhere):
- **Plots.** `scripts/build_truthkf_residuals.py` (SSM h5 preds + `truth_kf_reco.npy` side-cars
  → `matched_residuals.npz` on the same double-matched subset `fast_rms_eval` uses) driven by
  the new `scripts/run_truthkf_paper_plots.sh <pred_dir> <out_root> [datasets]`; both plot
  scripts are now reference-agnostic (`TRK_PLOT_TAG`, `TRK_REF_LABEL`, `TRK_PLOT_SUBTITLE`;
  filenames `<ds>_truthkf__*`). Bundle for the paper model:
  `eval_plots/paper_plots/truthkf_R2LFT/{single_muon_2GeV,10GeV,100GeV,uniform,ttbar,ttbar_new_pt1}/`
  (rmscurve vs η for all, vs pT for uniform, residual-hist liny, fast_rms-design summaries).
  Matched counts reproduce `eval_plots/sweep7/R2LFT/plots/rms_summary.json` `n_dm` exactly
  (69,957 / 70,819 / 70,406 / 3,418,865 / 96,965 / 646,382) → figures and the paper's ratio
  table are the same numbers.
- **Paper** (`/shared/tracking/NeurIPS_2026_SSM_Tracking`, uncommitted): `material/iclr/sync_figures.sh`
  now pulls from `truthkf_R2LFT` and the \ttbar pages come from `ttbar_new_pt1` (runs 6–45,
  646,382 matched tracks — the sample the results table's \ttbar row already used, replacing the
  ACTS-pipeline runs 6–15 / 238 k pages). Text: reference reframed as "the truth-seeded KF fits
  shipped with the benchmark" in abstract comment, intro (×3), data, results, discussion,
  conclusion; the ACTS pipeline is now an *independent framework check* only, with the
  miscalibration caveat written into appendix §F (1.4–3.2×, z0 2.0× / θ 3.2×, converter
  covariance mechanism). Efficiency restated **100.0 % vs 99.3 % of 199,760 uniform-muon
  prototracks** (`eval_plots/paper_plots/acts_timefix/single_muon_uniform/report.txt`), the
  74.6 %/25.4 % ttbar claim is gone. Integrated ttbar σ updated to 34.7 µm / 56.2 µm /
  0.90 mrad / 0.415 mrad / 1.76e-3. Forensics numbers re-measured on the R2LFT truth-KF npz:
  barrel 0.92–1.00, |η| 2.5–3 core 21/29/41 % tighter on d0/φ/q/p, q99.9 ratios 0.34–0.48,
  θ barrel corr 0.996 at ratio 1.00. Build: `PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2025/bin/x86_64-linux:$PATH bash build.sh`
  (the system texlive lacks fancyhdr) → 22 pages, no warnings.

### 4.31 R2Lnoconv-FT done (2026-09-04 ~03:00) — the conv-free 2L matches the paper model; recipe canonicalised

The Muon-hybrid + WSD 50-epoch DDP fine-tune of R2Lnoconv (d_conv 1, from
`logs/comet_offline/8f7e4ac91f444d8ea19c6ea82f627adc/ckpts/last.ckpt`) finished
on schedule; auto-eval `eval_plots/sweep7/R2LnoconvFT/` (v2 farm). Post-clip
SSM/truth-KF GM5: 0.991 / 0.988 / 0.913 / 0.973 / 0.992 (µ 2 / 10 / 100 GeV /
uniform / ttbar_new_pt1) — equal to R2L-FT to ±0.005; worst single parameter
ttbar q/p 1.02–1.03 (R2L-FT: 1.01); **pre-clip: no parameter above 1.0 anywhere**
(GM5 0.83–0.95, e.g. 100 GeV z0 0.70), the same tail profile as R2L-FT. So
dropping the depthwise conv costs ~1–2 % on hadron q/p post-clip and nothing
else → **R2Lnoconv-FT is the deployment model** (measured bs sweep, TF32 +
switches, `eval_plots/sweep7/noconv_bs_sweep.txt`: 1.74 M tracks/s at bs 16 k,
1.84 M at 32 k, plateau **1.89 M** from 64 k, 13.3 GiB VRAM at 120 k). The paper
model stays R2L-FT per the user's 2026-09-02 decision; whether the paper's
throughput headline quotes the noconv twin is the user's call. The canonical
recipe summary was written to the top of this file ("THE FINAL RECIPE") and
`README.md` was updated to match (both models, kernel path, truth-KF-only
reference rule) — 2026-09-04.

### 4.32 Stakeholder revision (2026-09-04): |η| ≤ 2 fiducial cut, deployment-precision physics, seed stays fp64, ttbar out of the paper

Stakeholder meeting: **the shipped truth-KF is miscalibrated above ~80 GeV outside |η| < 2**
(upstream, unfixable before the ICLR deadline). User decisions executed the same day:
paper model = **R2Lnoconv-FT**; every paper figure/number restricted to |η| ≤ 2 (truth η)
and produced at **deployment settings** (TF32 matmuls + BUCKET16 + compiled front-end +
GPU seed in-forward); **ttbar removed from the paper's physics results entirely** (stays as
training data); grad-cos probe moved to the appendix (my recommendation, user approved);
main text = 10 GeV RMS-vs-η + throughput; residuals (2/10/100 GeV) appendix-only at 240
bins over ±8·rms3 (was 120/±4).

- **Machinery**: `TRK_ABS_ETA_MAX` env (fast_rms_eval / acts_rms_curves /
  acts_legacy_style_plots — cuts tracks AND axes, bin width kept);
  `scripts/04b_eval_ckpt_deploy.sh` (deployment-settings eval: TF32+switches,
  `--data.seed_residual_features false` → model auto-seed, and the new
  `_deployment_anchors` override in `model.py` anchors predict_physical with
  `out["seed"]` so features AND anchors come from the GPU seed);
  `TRK_SEED_DTYPE` env in `seed_torch.py` (float64 default).
- **Deployment eval = strict-fp32 eval to displayed precision on every set**
  (`eval_plots/sweep7/R2LnoconvFT_deploy/`); |η|≤2 numbers in `plots_eta2/`:
  post-clip SSM/tKF all ≤ 1.00 (worst 2 GeV q/p +0.3 %), pre-clip all ≤ 1.00;
  100 GeV φ 0.92 / q/p 0.93 / d0 0.96 (the old 0.85s were forward-dominated —
  consistent with §4.26 and with the experts' finding). Paper bundle
  `eval_plots/paper_plots/truthkf_R2LnoconvFT_eta2/`.
- **Seed precision (user asked for fp32/TF32 default): REJECTED by measurement.**
  fp32 seed noise = the rc−R / rho−R cancellation (R ≈ 111 m at 100 GeV):
  Δd0 635 µm RMS at 50–110 GeV (19× seed resolution), z0 2.3 mm, s_helix 1e7 mm
  outliers; full eval at 100 GeV: z0 54 vs 18.6 µm (2.9×), θ 2.5×, pre-clip 100–570×
  (receipt `scratchpad deploy_smoke_100gev_seedfp32`). And the motivation was void:
  measured seed cost = **0.015 µs/track in-forward = 2.9 % of the 0.53 µs deployed
  noconv forward** (0.020 µs standalone; fp32 would save 0.55 % end-to-end). TF32
  is a no-op (seed has no matmuls). **fp64 stays the default**; documented in
  paper app:seed with the cancellation argument.
- **Grad-cos probe re-run on the noconv ckpt** (450×2048 ttbar, GPU seed features,
  `--variant v3c` — stock kernels can't run d_conv=1, probe got a variant arg):
  (d0,φ) +0.807±0.008, (z0,θ) +0.315±0.011, geometry block +0.20–0.25, |C(·,q/p)| ≤ 0.048
  — reproduces R2L-FT (checkpoint-robust). `eval_plots/paper_plots_extras/grad_cos_R2LnoconvFT_450x2048/`.
- **Cross-device throughput** (`eval_plots/paper_plots/throughput_h100/` + colleague's
  Ada `bench_logs_v2`; combined single-panel figure + summary in
  `eval_plots/paper_plots/throughput_cross_device/`): H100 NVL noconv deployed
  **1.91 M tracks/s** (131k batch, uncontended; 395 W peak), RTX 5000 Ada **0.504 M**
  (65k; 251 W; their 262k run died = 32 GB card), CPU KF 30 k. Kernel ablation on
  noconv: fused fp32 820k → +TF32 1.35M → +switches 1.78M @32k; CUDA-graph 913k vs
  eager 484k @2048; stock v0 CANNOT run d_conv=1 (causal_conv1d width ≥ 2) — stock
  rows stay from the d_conv=4 twin (2.5×). **Throughput per device-$ (list prices:
  Ada $4,000 MSRP, TR 3990X $3,990 MSRP, H100 NVL ~$30k street): 126 / 7.5 / 64
  tracks/s/$ → Ada = 16.7× the CPU (the paper's new abstract number, replacing
  "10–20×"), H100 = 8.5×.**
- **Internal (non-paper) diagnostics** `eval_plots/internal_datascience/R2LnoconvFT/`
  (`scripts/internal_tail_plots.py`): far-tail residual pages (±40·rms3 log-y,
  >3σ/10σ/30σ fractions, d0-tail η map) + iter-3σ RMS vs truth |d0| and z0
  (flat to |z0| < 200 mm, SSM ≤ tKF everywhere, both degrade at the acceptance edge).
- **Paper** (`/shared/tracking/NeurIPS_2026_SSM_Tracking`, uncommitted as before):
  results/data/method/abstract/intro/conclusion/appendix rewritten to the above;
  seed text now states "exactly three hits from the innermost (pixel) subdetector";
  seed timing paragraph in results; tab:seedquality recomputed |η|≤2; TF32-physics
  statement flipped to "physics = deployment path, strict-fp32 unchanged <0.3 %";
  main body exactly 9 pages (Conclusions on p9), no ttbar anywhere in results,
  sync_figures.sh → `truthkf_R2LnoconvFT_eta2` + throughput figure + noconv heatmap.

### 5.1 Comet RMS-vs-IQR audit (`docs/AUDIT_comet_rms_iqr.md`)

Verdict: **no logging bug.** `ssm_rms_dm`, `ssm_iqr_dm`, `ssm_precision_dm`
share residuals, units, DM subset and φ wrap (`model.py:900–987`); the
mean-over-batches changes them ≤ 2.4 %. The gap is an **estimator mismatch**:
training logs the *raw* RMS, the offline reports quote the *iterative-3σ-clipped*
RMSE (`eval_utils.py:322`, `paper_plots/stats.py:40`, `fast_rms_eval.py:166`).
The raw/clipped ratios logged by Comet for 09c54481 (15.6× / 122× / 259× /
478× / 11× for d0/z0/φ/θ/q/p) match `eval_plots/09c54481_4L_d128_muon/rms_summary.txt`
to a few percent. The size of the gap is real: 4–6.5 % of DM tracks have
catastrophic residuals (θ ≈ 0.84 rad RMS) carrying ~100 % of Σr²; 20.6 % of
|η| < 0.5 tracks vs 1.8 % at |η| > 2.5, enriched at low pT and in tracks with
negative-time hits — the fingerprint of the scrambled order (§0.1). Proposed
(not applied): log `ssm_rms3s_dm` (torch port of `iterative_rms_convergence`,
verified equal to the pooled offline value within 0.2 %) and `ssm_tailfrac_dm`
next to the raw RMS.

### 5.2 Time-independent hit sorting (ACTS) — `docs/HIT_SORTING_ACTS_vs_radial.md`

New code: `src/track_regression/hit_sorting.py` (pure numpy: `geometry_keys` /
`geometry_order`, `z_direction`, `distance_from_origin`, `cylindrical_radius`,
`distance_from_perigee`, `helix_arc_length`), `tests/test_hit_sorting.py`
(23 tests, pass), `scripts/hit_sorting_study.py`, displays in
`dataset_plots/event_displays_acts_sorted/` (per-dataset ACTS-ordered
displays on the same events/axes as the existing figures, three-row
side-by-sides legacy-`s` / ACTS / geometry, low-pT zooms,
`summary_disagreement.pdf` vs pT, |η|, |d0|, |z0|).

What ACTS does (repo at `a267dbb2`): truth tracks are built by
`std::sort` on `SimHit::time()` (`TruthTrackFinder.cpp` L106–116,
`TruthSeedingAlgorithm.cpp` L131–137) — never by path length, radius or
geometry id; the KF/CKF need no order at all (measurements keyed by surface).
That key exists in our raw data as `tracker_simhits.true_time` via
`tracker_hits.simhit_ids` (100 % resolvable) but not at inference.
**Side finding: all ColliderML time columns are in ACTS native units, mm/c
(1 ns = 299.792458 mm)** — the "slope 298" of the bug report's Finding 2 is
the unit, so pixel times *are* event-referenced; Finding 1 (strips carry no
time) stands and the digitised pixel time is smeared by tens of mm, so the
column still cannot order a track.

Exact per-track agreement with the ACTS truth-time order (test stores,
mislabelled tracks excluded):

| method | µ 2 GeV | µ 10 GeV | µ 100 GeV | µ uniform | ttbar |
|---|---:|---:|---:|---:|---:|
| stored on disk (digitised time) | 0.0 % | 0.0 % | 0.0 % | 0.005 % | 0.6 % |
| legacy `s` from origin | 87.2 % | 86.8 % | 87.5 % | 87.2 % | 84.9 % |
| `r` / transverse helix arc (truth) | 70.5 / 71.0 % | 75.7 / 75.8 % | 76.9 / 76.9 % | 76.6 / 76.6 % | 73.6 / 75.5 % |
| **geometry order (truth-free)** | **100.0 %** | **100.0 %** | **100.0 %** | **99.995 %** | **99.73 %** |
| \|X − P\| from truth perigee | 100.0 % | 99.96 % | 100.0 % | 99.995 % | 99.95 % |
| mixed helix arc (truth) | 99.96 % | 99.98 % | 99.92 % | 99.92 % | 99.91 % |

- The `s` failure is **the moved beamspot, not low pT**: flat at 13–15 % from
  0.5 GeV to > 100 GeV; 0 % of tracks at |z0| < 40 mm, ~31 % at
  |z0| = 160–200 mm, peaking at 0.5 < |η| < 1.5. Anchoring `s` at the truth
  perigee removes it. The legacy campaign (|z0| ≲ 60 mm) never saw this —
  there `s` agreed with time on 98 %.
- Geometry order = pixel → short strip → long strip; within a group barrel
  before endcap; barrel layers by r (quantised 0.1 mm), discs by z along the
  track's own flight direction (sign of z at max r − z at min r); ties by the
  other coordinate. Uses only x, y, z, volume_id — inputs the model already
  gets, so train and inference see the same sequence. The 54 ttbar misses are
  pT < 1 GeV kinks/loopers that also defeat every truth key.
- **Recommendation (study):** switch training + inference to the geometry
  order; do not keep `s` (13–15 % scrambled, precisely the large-|z0| tracks
  this campaign exists for); do not train on truth-time/perigee keys
  (unavailable at inference). Implemented as `--sort-key geometry`; the
  `ICLR_retraining_geom` store is being built alongside the `s` store so the
  decision costs no time (§0.1). **User decision pending** — the displays are
  in `dataset_plots/event_displays_acts_sorted/`.

## 6. Reminders (recorded 2026-08-25)

- **D.1 — secondaries in the training data.** Checked; see §0.2. Legacy sets
  contained ~16 % (p0) / ~5 % (p200) non-prompt decay daughters under
  `primary == True`; the new campaign's `primary` instead *excludes* prompt
  resonance daughters and all heavy-flavour daughters. Preprocessing should
  select on displacement from the PV, not on the flag. Decision pending.
- **D.2 — sorting: DECIDED 2026-08-25 (user) — geometry order.** Training
  and inference use `hit_sorting.geometry_order` (the `ICLR_retraining_geom`
  stores; `preprocess_flat.py --sort-key` now defaults to `geometry`; all ICLR
  configs, the E1 config and `04_eval_ckpt_iclr.sh` point at the `_geom`
  stores/farm). `ICLR_retraining_ssort` stays as the ablation twin (E1 can
  quantify the cost of the 13–15 % `s` mis-orderings). The original
  `ICLR_retraining` stores are time-sorted *and* 4 % mislabelled (§0.4) and
  must not be used for training or evaluation.
- **D.3 — Comet logging (user, 2026-08-25).** No double-matched metrics any
  more: the val/test estimators `ssm_precision / ssm_iqr / ssm_rms` are on
  all valid tracks (the `_dm` suffix and the `acts_dm_mask` dependence are
  gone), and the iterative-3σ-clipped RMSE + clipped fraction
  (`val/<p>/ssm_rms3s`, `val/<p>/ssm_tailfrac`) are logged once per
  validation epoch, unbinned over the whole val set pooled across ranks
  (`model.py: on_validation_epoch_end`), never during training steps. The
  training-step diagnostics are unchanged (mae, precision). Smoke-tested
  2026-08-25 on 2 GPUs (DDP, 2 × 120 steps, 81,920 pooled val tracks): metrics
  logged, one console line per epoch, no NCCL issues.

## 7. Open questions / decision log

- [ ] Stop `baeedc59` (10L bs2048, sess12) and relaunch on s-sorted stores?
- [x] **Seed definition: DECIDED 2026-08-26 (user) — keep the ACTS pixel-only
      three-point seed as implemented in `seed.py`; no longer-lever-arm variants.**
      Its q/p is at the Gluckstern limit for 3 pixel points over ~131 mm
      (0.0075 GeV⁻¹ at 50–110 GeV, 10.6 % charge-sign flips there); long-arm
      triplets gain only 1.6–2.6× on q/pT (stereo-strip r-φ leakage) and
      destroy z0/θ. The network's job is the q/p refinement from all hits.
- [ ] Adopt displacement-based `primary` for ttbar (changes the eval sample).
- [x] **Data re-preprocessing — DONE 2026-08-28 night as DATA v2 (§4.17), superseding the 08-29 reminder:** (i) add `--pt-max 110` to `preprocess_flat.py` and rebuild the ttbar
      training store `ttbar_new_pt1_tr` and the eval store `ttbar_new_pt1` with 1 ≤ pT ≤ 110 GeV
      (test where you train; §4.13 item 4), then rebuild the mixed store(s); (ii) keep
      geometry order (hit time is NOT fixed in the new runs, §4.13 item 1); (iii) everything
      on 3 T targets (B3). Not before the overnight runs of 2026-08-28 finish.**
- [x] 2026-08-27 ttbar testing limit pT ≥ 1 GeV (user) — `--pt-min 1.0` eval store
      `ttbar_new_pt1` from the new NERSC runs (§4.7).
- [x] **C (10L/192/32 un-anchored, sess5 GPU 1) killed 2026-08-27 22:29 at epoch 14/20 (user); last.ckpt kept.**
      Judgement 2026-08-27: yes — its read-out (un-anchored 10L vs un-anchored 4L
      at 20 epochs) no longer informs the design, all future runs are anchored;
      the per-epoch val curves already give the data point (C 10–30 % ahead of A
      at equal epochs, §4.1); the depth question is re-asked on the anchored
      recipe at scaling time. Use sess5:1 for the muon-only 50-epoch twin of Q′.
      User's call; not done.
- [x] 2026-08-27 NERSC ttbar runs 0–784 fetched (§4.7); truth-KF exists for ttbar now.
- [x] 2026-08-27 **Seed at inference: DECIDED (user) — GPU seed (`seed_perigee_torch`),
      +10 % forward time (0.22 µs/track on an H100 for the 4L) is acceptable.** The
      inference/deployment benchmarks quote model + GPU seed from now on (§4.9).
- [ ] **Back up the raw NERSC ttbar runs 6–784 (119 GB, parquet tables only) from sess3
      `/scratch/colliderml/drift_beamspot/ttbar/v1/runs/` to /eos** (`data/drift_beamspot/…`) —
      they exist nowhere else on our side; needs a shell on sess3 (`copy_dataset.sh`).
      The derived 3 T stores ARE on /eos (`ICLR_retraining_geom_B3`, 7 stores, verified).
- [ ] Request productions: log-uniform-pT muon gun ≥ 0.5 GeV (≥ 50 M);
      pile-up-200 ttbar (≥ 8 k events → ≥ 10 M tracks). ttbar truth-KF re-reco
      is a separate request (currently CKF-only).
- [x] Sorting: **geometry order chosen (user, 2026-08-25)**; `s` store kept as
      the ablation twin. At inference the same `hit_sorting.geometry_order` must
      be applied to the hits before packing (it is what `preprocess_flat.py`
      now does for every store; no model change).
- [ ] All campaign-2 numbers before 2026-08-25 (09c54481, baeedc59, transfers,
      the `eval_plots/` bundles, §3.5 zero-shot numbers on the padded path —
      those had correct order but still ~4 % mislabelled tracks) need redoing
      on the rebuilt stores.
- [x] 2026-08-25 logging: DM restriction removed, `val/<p>/ssm_rms3s` +
      `ssm_tailfrac` (unbinned, pooled, once per val epoch) added — reminder D.3.
      Comet metric names changed (`ssm_*_dm` → `ssm_*`), so curves are not
      continuous with pre-2026-08-25 runs.
- [x] 2026-08-25 `fast_rms_eval.py` writes `rms_by_pt.txt` (per-pT-bin iter-3σ RMSE, SSM vs reference) — the read-out for trial E.
- [x] 2026-08-25 `preprocess_flat.py`: `--sort-key {s,geometry,time}`
      (default `s`) + event_id-value join (§0.4). Stores rebuilt and verified:
      `ICLR_retraining_ssort` / `ICLR_eval_ssort` and `ICLR_retraining_geom` /
      `ICLR_eval_geom` (145 GB each, /scratch on this host; 737 GB free).
      **Both store roots copied to `/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_{ssort,geom}`
      2026-08-25 21:00 and verified per file (6 883 files, 155.2 GB apparent each,
      0 missing / 0 size mismatches).** The eval farms are symlinks only and are
      not on /eos — recreate them with `scripts/05_rebuild_iclr_stores.sh` (its
      "eval farm" block) after restoring a store. The /eos `ICLR_retraining`
      (time-sorted, mislabelled) is still there and should be removed or renamed
      `_DEPRECATED` — not done, user's call.
- [x] 2026-08-25 visual checks for the rebuilt stores:
      `src/track_regression/scripts/kf_baseline_plots.py` (RMS-vs-η of the
      reference fits alone, 2×3 design) → `eval_plots/baselines_KF_rebuilt_{geom,ssort}/`;
      `src/track_regression/scripts/sorting_visual_check.py` (per-track s-order vs
      geometry-order displays with a |X − perigee|-vs-sequence-position
      monotonicity panel, disagreement vs |z0| / pT, full-detector overlays) →
      `eval_plots/baselines_KF_rebuilt_geom/sorting_check/`. Measured there:
      s ≠ geometry on 12.9 % (uniform), 13.2 / 14.4 / 13.2 % (2 / 10 / 100 GeV),
      15.5 % (ttbar) of tracks; 0 % at |z0| < 50 mm rising to 33 % at
      180–200 mm; flat in pT. READMEs in both directories.
- [x] 2026-08-25 tooling for the plan: `scripts/04_eval_ckpt_iclr.sh <run_dir>
      <ckpt> <out_dir> [eval_root] [gpu]` evaluates one checkpoint on all five
      eval stores + `fast_rms_eval` summary (copies the ckpt first, see §8);
      `config/ssm_cls/ICLR/pretrain_ssm_cls_4L_d128_muon_geom_E1.yaml` was the
      draft E1; superseded by `config/ssm_cls/ICLR_sweep1/` (§4).
- [x] Eval of `baeedc59` (10L/192/32, Lion bs 2048) epoch-12 on the five
      time-sorted eval stores (same scrambled input as 09c54481, so the two are
      comparable). Table: `docs/dataset_accounting_2026-08-25/eval_baeedc59_10L_bs2048_ep12_rms_summary.txt`.
      Uniform muons, iter-3σ, SSM vs 4L-bs128k@ep49 vs reference fit
      (`fast_rms_eval` uses the truth-KF wherever a part carries
      `truth_kf_reco.npy` — uniform and the three fixed-pT sets — and the CKF
      only for ttbar; its txt header still says "CKF"):
      d0 **24.4** / 32.1 / 14.3 µm, z0 **362** / 200 / 21.6 µm, φ **1.50** / 1.03 /
      0.179 mrad, θ **1.47** / 0.733 / 0.059 mrad, q/p **6.4e-4** / 1.15e-3 /
      2.24e-4. Equal val/total (0.0331 vs 0.0330) but the two runs trade
      parameters completely differently — the loss is not a proxy for the
      per-parameter physics metric, which is another reason for the
      `ssm_rms3s_dm` trigger of §3.3. Neither run is anywhere near the
      reference; both are on scrambled input, so no conclusion about
      batch size or depth is drawn from this pair (E1–E3 will).

## 8. Repo facts that bite (carried from memory + verified today)

- **θ vs η (user question 2026-08-31):** the regressed and reported parameter is the perigee
  polar angle θ (mrad residuals). Legacy/un-anchored heads used `quantile_eta` (loss in η
  space, predictions returned in θ); every anchored head since sweep 2 is θ-native
  (Δθ = θ − seed_θ, plain quantile). "RMSE vs η" plots bin in η but plot θ residuals.
  σ_η ≈ σ_θ/sin θ if pseudorapidity resolution is ever quoted.

- `train.py` auto-loads `base.yaml` from the *config's own directory*; a leaf
  `callbacks:` list **replaces** the base list (re-add `KernelSwapCallback`).
- Flat stores need `trainer.use_distributed_sampler: false` under DDP
  (`BlockBatchSampler` shards itself; the DataModule raises otherwise).
- Padded path + `hit_time` sort key = pads interleaved with real hits whenever
  time ≤ 0 (memory `padded-hit-time-sort-bug`); evaluate padded checkpoints
  with `TRK_SORT_KEY=hit_s`. Packed path never re-sorts — the on-disk order is
  the model input.
- Never widen loss norm ranges on a kept head; fresh head → new ranges.
- `RegressionPredictionWriter` writes `<ckpt-stem>__test_predictions.h5` next
  to the checkpoint — copy the checkpoint elsewhere before evaluating several
  datasets, and move the h5 between runs.
- `ICLR_eval/<fixed-pT>/test` unions train+val+test parts (the model never saw
  them); the reused-`event_id` defect in the fixed-pT drops breaks post-hoc
  joins (memory `kf-baselines-reference`).
- Legacy runs used `TRK_MATMUL_PRECISION=highest`; `train.py` now defaults to
  `high` (TF32) — set it explicitly for precision-comparable numbers.
- **Skipped validations (found 2026-08-26, fixed).** `BlockBatchSampler`'s
  per-epoch jitter changed the block count by ±1 while `__len__` stayed at
  epoch 0's value; Lightning validates only at `batch_idx + 1 == len`, so a
  one-short epoch ran no validation (no `val/*` point, no checkpoint
  candidate). Sweep-1 impact: B/D/F validated at epochs 0,5,6,7,12,15,18 only
  (their `best.ckpt` = epoch 18, `last.ckpt` = epoch 19 unvalidated); A/C
  skip ~15 % of epochs (3 and 10 so far in A); E none. Fix: the sampler now
  yields the worst-case block count in every epoch (`tests/test_flat_data.py::
  test_block_sampler_yields_len_every_epoch`). A and C keep running on the old
  code (a restart would cost 12 h / 3 days for a few val points); evaluate
  their `last.ckpt` offline with `04_eval_ckpt_iclr.sh`.
