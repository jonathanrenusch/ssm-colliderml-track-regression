# RMS evaluation — run `09c54481fbb84e42b9e155ed550fd91f`

4L · dim 128 · d_state 128 · MuonHybrid · 50 epochs · batch 32 k/device × 4 GPUs
Trained on `drift_beamspot / single_muon_uniform`, selection `core`, no d0/z0 windows.
Checkpoint: `epoch=049-val_total=0.03301.ckpt`.

## Which file to open

Four figures per dataset. All are 2×3 grids: the five perigee parameters plus the
η distribution of the double-matched tracks in the sixth cell.
Blue = SSM, red = CKF; solid = iter-3σ, dashed = pre-clip.

| suffix | what it shows |
|---|---|
| `_postclip` | iter-3σ core only — **the cleanest read** |
| `_preclip`  | pre-clip only, tails included |
| `_logy`     | both, log y — **the honest before/after in one figure** |
| *(none)*    | both, linear y — pre-clip dominates, post-clip curves flatten onto the axis |

`rms_summary.txt` / `.json` carry the unbinned numbers and the tail statistics.

## Datasets

`single_muon_uniform` is the training dataset, so it is evaluated on its **held-out
test split** (5.0 M tracks). The other four were never trained on at any split, so
**all their parts are unioned** — 110 k instead of ~5 k for the fixed-pT sets, which
otherwise give only ~170 tracks per η bin.

Every plot is restricted to the ACTS double-matched subset, since that is the only
regime where a CKF comparison exists.

## What is not here

No bootstrap, so **no confidence bands** — that is the whole speedup (34 s for all
five datasets against tens of minutes for the full `paper_plots` pipeline). Use
`track_regression.paper_plots.cli` when you need error bars for the paper.

## Regenerating

Predictions are cached at `/scratch/colliderml/ICLR_eval/_preds/<dataset>.h5`, so
replotting does not re-run inference:

```bash
python src/track_regression/scripts/fast_rms_eval.py \
  --pred-dir /scratch/colliderml/ICLR_eval/_preds \
  --store-root /scratch/colliderml/ICLR_eval \
  --out-dir <somewhere>
```
