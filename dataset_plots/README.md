# drift_beamspot — dataset plots

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

Everything here is produced from the **preprocessed flat stores** at
`/scratch/colliderml/ICLR_retraining/<dataset>/test` by
`src/track_regression/scripts/plot_preprocessed.py`.

That means: **no selection variants and no extra cuts**. What you see is exactly
what the model is trained on — selection variant `core`, with the d0/z0 window
cuts removed. There is no `loose` / `core_kf_matched` / `core_kf_hits` overlay
any more; those definitions are still recorded in `target_quantiles.txt` for
reference.

PDF only. Y axes are **absolute counts**, never a density; the log/linear choice
is made from the data's own dynamic range, so a falling spectrum (ttbar pT, q/p)
gets log and a flat one (uniform d0, muon-gun pT) stays linear.

```
distributions/<dataset>/
    targets_and_kinematics.pdf   d0 z0 phi theta qop + pT, eta, hits-per-track
    hit_features.pdf             all 12 hit input features
event_displays/<dataset>/
    overlay_10events.pdf         muon sets: 10 events overlaid, one colour each
    event_NN_id<eid>.pdf         ttbar: one figure per event (~16-30 tracks each)
distribution_stats.txt           quantiles, mean+-sd and cut-retention tables
                                 + the full selection-variant definitions
```

Event displays are x–y, z–x and z–y (no 3-D). Axes are **fixed across every
figure** to the ODD envelope, |r| < 1100 mm and |z| < 3100 mm, with equal aspect,
so events and datasets are directly comparable by eye. Only hits belonging to
*selected* tracks are drawn — the same hits the model sees, not the full event.

To regenerate:

```bash
python src/track_regression/scripts/plot_preprocessed.py          # defaults to the test split
python src/track_regression/scripts/plot_preprocessed.py --split train --n-events 20
```
