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
  meaningful anyway. Copy the chosen one to
  `/eos/.../ICLR_retraining_<ssort|geom>` with `scripts/copy_dataset.sh`; the
  `/eos` `ICLR_retraining` stores are time-sorted AND mislabelled (§0.4) and
  should be marked deprecated.

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

## 4. Experiment plan (first series, 6 H100s, bs 2048 per pretraining run)

Ordered by information per GPU-hour. All pretraining Lion + WSD at bs 2048,
packed, `TRK_MATMUL_PRECISION=highest`, s-sorted stores
(`/scratch/colliderml/ICLR_retraining_ssort`), eval with `fast_rms_eval.py`
on `/scratch/colliderml/ICLR_eval_ssort` (per-pT-bin variant to be added).

| # | experiment | hypothesis | setup | confirm / kill | cost |
|---|---|---|---|---|---|
| E0 | data fixes | — | rebuilt stores with the event_id join, `s` and geometry variants (running); decide the sort key; displacement-based `primary`; request low-pT gun + pile-up ttbar | — | CPU only |
| E1 | ordering effect | scrambled order capped 09c54481 | re-run the 09c54481 recipe (4L/128/128, Muon, 4×32 k) on s-sorted stores, 10 epochs; compare per-pT RMS to 09c54481 at equal epochs | confirm: ≥ 20 % better in every parameter at 10 epochs; kill: no gain → §3.7 "change my mind" | 4 GPUs × ~6 h |
| E2 | architecture screen | 4–6 layers suffice; d_state widening helps | the 6-run sweep of §3.6 at 100 k steps, one GPU each, concurrent | rank by per-bin G; keep any run within 3 % of the best that is ≥ 2× faster | 6 GPU-days, 1 calendar day |
| E3 | batch-size re-test | bs 2048 is still necessary on this data | 4L/128/128, Lion, bs 2048 vs 16 384 (LR × √8), same track budget (1 B presentations), per-bin RMS | confirm constraint: bs 2048 better by > 5 % in any core parameter; refute: equal or better at 16 k | 2 GPU-days |
| E4 | data scaling | 200 M is enough above 5 GeV, not below | 4L on {10, 50, 191} M subsets at 100 k steps; RMS vs N per pT bin | falling at 191 M → data-limited bin | 3 GPU-days |
| E5 | warm start | legacy encoder transfers | 57dabaab-pretrain encoder + fresh head vs scratch twin, same budget | ahead at epoch 5 and 20 → adopt; else drop | 2 GPU-days |
| E6 | full pretrain + trigger | probe trigger switches earlier than 50 epochs | winner of E2, log-uniform pT sampling, WSD, probe fine-tune every 5 epochs (§3.3) | switch point, then Muon fine-tune on the same set (no separate fine-tune set exists yet) | ~10 GPU-days |
| E7 | sampling ablation | log-uniform pT sampling buys low-pT RMS at no high-pT cost | E6 twin with uniform sampling | per-bin comparison | ~10 GPU-days |

Total ≈ 33 GPU-days ≈ 6 calendar days on 6 H100s with E1–E5 concurrent in
the first two days. E6/E7 wait for E2's winner.

## 5. Subagent results

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
- **D.2 — sorting.** Until the user decides on the ACTS study (§5.2),
  retraining uses the legacy radial-magnitude order `s` —
  `ICLR_retraining_ssort` (§0.1). The study says `s` mis-orders 13–15 % of
  new-data tracks (large |z0|) and recommends the geometry order
  (`ICLR_retraining_geom`, built in parallel). The original `ICLR_retraining`
  stores are time-sorted *and* 4 % mislabelled (§0.4) and must not be used for
  training or evaluation.

## 7. Open questions / decision log

- [ ] Stop `baeedc59` (10L bs2048, sess12) and relaunch on s-sorted stores?
- [ ] Adopt displacement-based `primary` for ttbar (changes the eval sample).
- [ ] Request productions: log-uniform-pT muon gun ≥ 0.5 GeV (≥ 50 M);
      pile-up-200 ttbar (≥ 8 k events → ≥ 10 M tracks). ttbar truth-KF re-reco
      is a separate request (currently CKF-only).
- [ ] Sorting: `s` (standing rule) vs geometry order (study recommendation,
      §5.2). Both stores exist once the rebuild finishes; E1/E2 can start on
      either. If geometry is chosen, inference must apply
      `hit_sorting.geometry_order` to the hits before packing (one line in the
      collate / preprocessing, no model change).
- [ ] All campaign-2 numbers before 2026-08-25 (09c54481, baeedc59, transfers,
      the `eval_plots/` bundles, §3.5 zero-shot numbers on the padded path —
      those had correct order but still ~4 % mislabelled tracks) need redoing
      on the rebuilt stores.
- [ ] Add `ssm_rms3s_dm` / `ssm_tailfrac_dm` to `_log_metrics` (§5.1 diff).
- [ ] Add a per-pT-bin mode to `fast_rms_eval.py` (needed by E2–E4).
- [x] 2026-08-25 `preprocess_flat.py`: `--sort-key {s,geometry,time}`
      (default `s`) + event_id-value join (§0.4). Stores rebuilt and verified:
      `ICLR_retraining_ssort` / `ICLR_eval_ssort` and `ICLR_retraining_geom` /
      `ICLR_eval_geom` (145 GB each, /scratch on this host; 737 GB free).
      Not yet copied to /eos.
- [x] 2026-08-25 tooling for the plan: `scripts/04_eval_ckpt_iclr.sh <run_dir>
      <ckpt> <out_dir> [eval_root] [gpu]` evaluates one checkpoint on all five
      eval stores + `fast_rms_eval` summary (copies the ckpt first, see §8);
      `config/ssm_cls/ICLR/pretrain_ssm_cls_4L_d128_muon_ssort_E1.yaml` is E1
      (the 09c54481 recipe on the s-sorted store, 10 epochs). Not launched.
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
