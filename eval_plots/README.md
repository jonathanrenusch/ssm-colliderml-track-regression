# ICLR evaluation — RMSE vs η, bootstrap-free

> **WARNING (2026-08-25).** The flat stores under `ICLR_retraining/` (and every
> eval built from them) store each track's hits ordered by the *broken*
> `tracker_hits.time` — all strip hits (time == 0) first, then the pixels — and
> the packed training/eval path consumes that order verbatim. Every new-campaign
> SSM number produced from these stores (09c54481, baeedc59, packed-mode legacy
> transfers) is on scrambled sequences. Re-sorted stores: `ICLR_retraining_ssort/`
> and `ICLR_eval_ssort/`. See `CLAUDE.md` §0.1. Only padded evaluations run with
> `TRK_SORT_KEY=hit_s` (57dabaab) were correctly ordered. In addition ~4 % of the
> muon-gun tracks in these stores carry the hits of a *different event* (row-index
> join in `preprocess_flat.select_shard`, fixed 2026-08-25; `CLAUDE.md` §0.4) — the
> catastrophic-residual population that the iter-3σ clip removes.

Six models × six datasets. `CROSS_MODEL_SUMMARY.txt` is the one-page comparison.

## Models

| directory | arch | trained on | kind |
|---|---|---|---|
| `09c54481_4L_d128_muon` | 4L d192→128 d_state **128** | **new** drift_beamspot muon (core) | from scratch, 50 ep |
| `57dabaab_10L_truthhit_FT` | 10L dim192 d_state32 | legacy p200 **truth hits** (kf_matched) | finetune, ep49 |
| `2787864_10L_truthhit_PT` | 10L dim192 d_state32 | legacy p0_core **truth hits** | pretrain, ep48 |
| `1e0f5105_4L_truthhit_PT` | 4L dim128 d_state16 | legacy p0_core **truth hits** | pretrain, ep49 |
| `ba96d05f_10L_kfhits_FT` | 10L dim192 d_state32 | legacy p200 **KF hits** | finetune, ep16 |
| `eaa7e3a1_4L_kfhits_FT` | 4L dim128 d_state16 | legacy p200 **KF hits** | finetune, ep49 |

Each model was run with **its own** `config.yaml`, so its architecture and — critically —
its own d0/z0/qop denormalisation ranges apply. All at full IEEE fp32
(`float32_matmul_precision = 'highest'`, TF32 off), confirmed in every job log.

## READ THIS BEFORE COMPARING

Every legacy checkpoint carries the **old loss norm ranges**: d0 ±2.5 mm, z0 ±200 mm.
The new drift_beamspot targets span **d0 ±7.07 mm** and **z0 ±268.8 mm**. Those models
were never trained to emit most of that range, so their new-data d0/z0 numbers
(~2000 µm d0) measure the output ceiling, not model quality.

The in-domain control at the bottom of `CROSS_MODEL_SUMMARY.txt` proves the point:
on their own legacy data all five reach **d0 12.3–12.8 µm against the CKF's 65.7 µm** —
a 5× win. Only `09c54481` was trained on the new target range, so it is the only
model whose new-data numbers mean anything.

## The central-η spike (fixed 2026-08-22)

The first pass showed a large SSM-only RMS spike at |η| < 1 on the legacy
in-domain plots (φ 0.65 → 7.7 mrad at η = 0) that is absent from the archived
April figures. Cause:

* the deprecated **padded** encoder path argsorts the *padded* sequence by
  `x_sort_value`, and pad slots are zero-filled;
* `model.py` now passes **`hit_time`** as that key (it replaced `s`, which
  underestimates on-helix arc length for forward tracks);
* truth time is **signed** — vertex time smearing pushes it to −0.6 ns on the
  legacy p200 data — so real hits with `time ≤ 0` sort at or behind the pads and
  the scan sees padding interleaved into the hit sequence;
* central tracks have the shortest flight path and therefore the smallest times:
  **26.9 % of |η| < 0.25 tracks are affected, falling to 0 % beyond |η| > 2.5** —
  exactly the shape of the spike. `s` is strictly positive and never had this.

Only `57dabaab` ran padded (its config predates `packed_batches`); every other
model here is packed, which does not sort and has no pads, so they are
unaffected. `57dabaab` was re-run with `TRK_SORT_KEY=hit_s` (a new env override
in `model.py`) and now reproduces its archived April numbers: d0 12.19 vs 12.19,
z0 171.2 vs 174.1, φ 0.617 vs 0.624, θ 0.778 vs 0.787.

**If you evaluate any padded-mode checkpoint, set `TRK_SORT_KEY=hit_s`.** The new
drift_beamspot data also has vertex time smearing (±0.32 ns), so the same trap
applies there.

## Clipping was verified, not assumed

`fast_rms_eval`'s iter-3σ is **bit-identical** to `paper_plots`
(`iterative_rms_convergence`), per-bin and unbinned. The pre-clip RMS differs by
1.2e-5 absolute because this one accumulates in float64 where the legacy one uses
the input dtype. The CKF curves match the April archive to 4 significant figures
on the same 6,591,752 tracks, which is what isolated the problem to the SSM
inference rather than the plotting.

## Baseline caveat

The red curve is the **CKF**, not the truth-tracking KF. Only `single_muon_uniform`
ships a `truth_tracks` table and the preprocessor does not ingest it yet.

## Files

Per dataset: `_postclip` (cleanest), `_preclip`, `_logy` (readable before/after),
and plain linear. Blue = SSM, red = CKF; solid = iter-3σ, dashed = pre-clip.
`OWN_*` = the legacy dataset that model was trained on (the control).
