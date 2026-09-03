# Seed-guided bidirectional state space models for charged-particle track fitting

Code release for a Mamba-2-based regressor of the five perigee track parameters
`(d0, z0, φ, θ, q/p)` from raw silicon-detector hits. Each track's hits are read
as **residuals to an analytic three-point helix seed** (the ACTS conformal-map
seed estimator, ported), and the network predicts **corrections to that seed**.
On the ColliderML *drift-beamspot* datasets the paper model — a **2-layer,
0.65 M-parameter** encoder, Muon-hybrid fine-tuned — is **at or below the ACTS
truth-seeded Kalman filter on all five parameters on every test set** (single
muons at 2, 10 and 100 GeV, a uniform 1–110 GeV spectrum, and ttbar hadrons),
including the un-clipped residual tails, evaluated through ACTS's own
performance pipeline. Inference is embarrassingly parallel across tracks; the
GPU kernels are rewritten for the short-sequence regime of a track, and the
headline throughput comparison — at hardware and energy cost matched to a
multi-core CPU running the classical fit — is on a workstation-class GPU
(RTX 5000 Ada, benchmark in progress), targeting an order-of-magnitude gain.

The repository contains the model and training code (Lightning CLI + YAML), the
seed estimator (numpy + torch twins), the data preprocessing pipeline for the
ColliderML parquet datasets, fused short-sequence Triton kernels, the evaluation
and plotting pipeline against the truth-tracking KF, and an ACTS integration that
runs the trained model as an `IAlgorithm` inside a real ACTS event loop so it can
be scored by ACTS's own performance writers.

**No checkpoints are shipped.** Train your own with the recipe below, or wire an
existing checkpoint into a config via `pretrained_ckpt_path`.

## The track-fitting problem

![Track-fitting at HL-LHC](media/figure_overview.png)

**(a)** Perigee parameters of a charged-particle track in a solenoidal magnetic
field. The trajectory (orange) is a helix; **r** is the point of closest approach
to the beamline. `d0` is the perpendicular distance from **r** to the *z*-axis;
`z0` is the *z*-coordinate of **r**; `φ` and `θ` parametrise the momentum **p**
at the perigee; `q/p` is the signed inverse momentum. *Schematic — not to
scale.* Adapted from the ACTS tracking documentation
([acts-project.github.io/tracking.html](https://acts-project.github.io/tracking.html)).

**(b)** A single proton–proton bunch crossing at the High-Luminosity LHC produces
thousands of charged-particle trajectories that must be reconstructed from the
sparse silicon-detector hit collection. *Track finding* assigns hits to candidate
trajectories and *track fitting* estimates the five perigee parameters of each
candidate; this work targets the latter, given the hit-to-track assignment.
Panel (b) is reproduced from the ColliderML dataset paper
([arXiv:2512.15230](https://arxiv.org/abs/2512.15230); Elitez, Gessinger,
Murnane, Raaholt, Salzburger, Skov, Stefl, Zaborowska, 2025). The datasets used
here are the *drift-beamspot* ColliderML production (moving beamspot, 3 T
solenoid, truth-tracking-KF reference fits included).

## Architecture

<p align="center">
  <img src="media/architecture.png" alt="Bidirectional Mamba-2 architecture" width="66%">
</p>

Bidirectional Mamba-2 encoder with CLS-token readout, packed (padding-free)
batching, and a quantile regression head per parameter. The final models are
**2 layers** (the paper model, Muon-hybrid fine-tuned; a 3-layer variant is
identical in physics), `dim 128`, `d_state 64`, `headdim 32`, `expand 2`,
`d_conv 4` — 0.65 M / 0.9 M parameters.

## The final recipe

Everything below is the outcome of the campaign-2 ablation program; each
ingredient was established against controls (see `CLAUDE.md` for the full
experiment log).

**1. The seed.** For every track, an exact port of ACTS's
`estimateTrackParamsFromSeed` (`src/track_regression/seed.py`) fits a helix
through three pixel space points (largest radial lever arm, conformal-map
circle + sinc-corrected dz/ds) and transports it to the beamline perigee.
Inputs: hit `x, y, z`, `volume_id`, and the constant 3 T field — reconstructed
quantities only, no truth. A torch twin (`seed_torch.py`) runs the seed on the
GPU at inference (+2 % forward time).

**2. Seed-residual input features.** Each hit enters the encoder as the 12
absolute features (below) **plus 3 residuals to the seed helix**:
`asinh(du/0.1 mm)`, `asinh(dv/0.1 mm)`, `s_helix` — the transverse and
longitudinal distances from the helix and the path length along it
(`input_dim: 15`). The three seed hits are self-identifying (residual 0). This
is the single largest precision ingredient: it puts the Kalman filter's
representation (residuals to a reference trajectory) inside the encoder.

**3. Seed-anchored heads.** Every head predicts a *correction* to the seed,
added back at prediction time (a target-side skip connection, no gradient
through the seed):
- `d0, z0, θ`: `delta_anchor: seed_<p>` with narrow norm ranges (±0.4 mm,
  ±3.5 mm, ±0.01 rad),
- `φ`: anchored quantile head, Δφ wrapped to (−π, π], range ±0.015 rad,
- `q/p`: **scale-free head** — the target is
  `(q/p − seed_qop) / (|seed_qop| + 0.02)`, range ±2
  (`scale_anchor_eps: 0.02`). This equalises the precision demand across the
  momentum spectrum and is what closes q/p at low pT and on hadrons; a plain
  absolute q/p head stalls 5–10 % above the KF.
- All five loss weights **1.0**, 7-quantile pinball ladders (monotone via
  softplus + cumsum). Uneven weights and a Smooth-L1 circular φ head cost up to
  2× in φ resolution — do not reintroduce them.

**4. Hit ordering.** Hits are sorted in **detector geometry order**
(`hit_sorting.geometry_order`: pixel → short strip → long strip; barrel by
radius, endcaps by z along the flight direction) — a truth-free key that
reproduces ACTS's simulation-time ordering on ≥99.7 % of tracks. The packed
encoder consumes the stored order verbatim; train and inference must use the
same key. (The digitised hit time cannot order a track: strip hits carry
`time = 0`.)

**5. Training.** Strict IEEE fp32 (`TRK_MATMUL_PRECISION=highest`), Lion,
**batch size 2048**, OneCycle (1e-5 → 5e-5 → 1e-6), weight decay 1e-3,
**25 epochs** over a mixed store of ~200–400 M tracks (uniform-pT muons +
log-pT muons + ttbar hadrons with pT ≥ 1 GeV). Small batches matter: the
precision arrives during the anneal, and large-batch runs (36 k) with short
anneals lose 15–80 % on φ/q/p. An optional second stage — **Muon-hybrid
optimizer + WSD schedule, large batch (2×20 k DDP), 50 epochs** — leaves the
clipped RMSE unchanged and cleans the residual tails (the paper model's
pre-clip ratios are ≤ 1.0 everywhere only after this stage).

**6. Data selection (v2).** 3 T-field perigee targets, `|d0| ≤ 7.1 mm`,
`|z0| ≤ 270 mm` (= the head norm ranges), 6–20 hits, `|η| ≤ 3`; ttbar
restricted to 1–110 GeV (test where you train). No beamspot constraint
anywhere — the fit is beamspot-free by construction.

## Results

Iterative 3σ-clipped RMSE of (prediction − truth), ratio **SSM ÷ truth-tracking
Kalman filter** (the strongest classical reference: KF on the true hit set;
the CKF tracks it to within 3–9 %). Evaluated on the CKF-double-matched subset
of each test set; `GM5` = geometric mean over the five parameters.

**R2L-FT — the paper model** (2 layers, Muon-hybrid fine-tuned; the 3-layer twin is identical):

| dataset | d0 | z0 | φ | θ | q/p | GM5 | tracks |
|---|---|---|---|---|---|---|---|
| µ 2 GeV | 0.985 | 0.989 | 0.991 | 0.978 | 0.990 | **0.987** | 70 k |
| µ 10 GeV | 0.985 | 0.986 | 0.984 | 0.981 | 0.990 | **0.985** | 71 k |
| µ 100 GeV | 0.914 | 0.987 | 0.845 | 0.970 | 0.855 | **0.912** | 70 k |
| µ uniform 1–110 GeV | 0.970 | 0.989 | 0.945 | 0.979 | 0.964 | **0.969** | 3.42 M |
| ttbar 1–110 GeV | 0.983 | 0.982 | 0.984 | 0.982 | 1.001 | **0.986** | 646 k |

**YZ-2L-mix3** (2 layers, 25 epochs, no fine-tune — the throughput model):

| dataset | d0 | z0 | φ | θ | q/p | GM5 |
|---|---|---|---|---|---|---|
| µ 2 GeV | 0.984 | 0.988 | 0.990 | 0.977 | 1.005 | 0.989 |
| µ 10 GeV | 0.986 | 0.985 | 0.983 | 0.980 | 0.997 | 0.986 |
| µ 100 GeV | 0.920 | 0.989 | 0.857 | 0.977 | 0.854 | 0.917 |
| µ uniform 1–110 GeV | 0.973 | 0.991 | 0.949 | 0.980 | 0.975 | 0.973 |
| ttbar 1–110 GeV | 0.983 | 0.983 | 0.984 | 0.982 | 1.013 | 0.989 |

Pre-clip (tail-inclusive, no outlier removal) the paper model is **also at or
below the KF on every parameter of every set** (GM5 0.82–0.95; e.g. ttbar q/p
0.79, 100 GeV z0 0.70). Absolute values on uniform muons (SSM / truth-KF,
post-clip): d0 13.9 / 14.3 µm, z0 21.4 / 21.6 µm, φ 0.167 / 0.177 mrad,
θ 0.057 / 0.058 mrad, q/p 2.16 / 2.24 × 10⁻⁴ GeV⁻¹.

## Inference throughput

The trigger-relevant comparison is against a multi-core CPU running the ACTS
Kalman fit (~30 k tracks/s on a 64-core Threadripper) at matched hardware and
energy cost — a workstation-class RTX 5000 Ada, benchmarked by a collaborator
(in progress), where we expect an order-of-magnitude (up to ~20×) advantage.
The H100 numbers below (32 k tracks/batch, GPU seed included) are shown only to
isolate the kernel adaptation, not as the headline:

| model | strict fp32 | TF32 matmuls | TF32 + kernel switches |
|---|---|---|---|
| 4L | 429 k tracks/s | 657 k | 730 k |
| stock `mamba_ssm` kernels | 589 k | — | — |
| **2L paper model, fused kernels** | 759 k | — | — |
| **2L paper model, deployment path** | — | — | **1.48 M** |
| 3L twin, deployment path | 540 k | 816 k | 1.05 M |

Kernel switches = `TRK_SSD_BUCKET16=1` (16-token bucketing for the ~60 % of
tracks with ≤16 hits) and `TRK_COMPILE_FRONTEND=1` (compiled
normalise → Fourier → input-net front end). TF32 matmuls change the physics by
≤ 2.3 % (100 GeV φ/q/p only); strict fp32 stays the default for training and
physics numbers. Throughput saturates from ~16 k tracks/batch; benchmark with
`scripts/bench_infer_flat.py --gpu-seed`.

## Fast short-sequence SSM kernels (on by default)

Tracks are at most **22 tokens** (≤20 hits + 2 CLS). The stock `mamba_ssm`
chunked selective scan is built for sequences of thousands of tokens, so at this
length it is almost pure launch/bookkeeping overhead. This repo replaces the
*evaluation* of the Mamba-2 update — never the math — with the algebraically
identical **single-chunk SSD quadratic dual**: one dense `L×L` (≤22×22)
decay-weighted product per track.

| Impl | File | Use |
|------|------|-----|
| Pure-PyTorch quadratic dual (`torch.compile`-friendly, exact autograd) | `src/track_regression/mamba_short.py` (`Mamba2Short`) | training **and** inference |
| Fused portable Triton kernels | `src/track_regression/ops/ssd_short_triton.py` | inference |

Every SSM-CLS config ships a `KernelSwapCallback(variant: auto)`: training runs
the compiled PyTorch dual (`v3c`, exact gradients), validation/test/predict run
the fused Triton packed kernels (`v5pc`, batch ceiling ≥ 600 k tracks per call).
`Mamba2Short` also has a **native build path** — inference needs neither
`mamba-ssm` nor `causal-conv1d` installed. Correctness is gated by an
fp64-anchored oracle chain (`tests/test_mamba2short.py`,
`tests/test_ssd_variants.py`).

```python
from track_regression.mamba_short import apply_variant
apply_variant(model, "v5pc")   # fused Triton, packed — fastest inference
apply_variant(model, "v3c")    # compiled pure-PyTorch — training / portable
apply_variant(model, "v0")     # stock chunked kernel (reference)
```

## Layout

```
.
├── README.md / LICENSE / CLAUDE.md         ← this file, GPL-3.0-or-later, experiment log
├── media/architecture.{pdf,png}
├── src/track_regression/
│   ├── train.py                            ← Lightning CLI entry
│   ├── model.py                            ← LightningModule, Fourier front end, WSD/OneCycle
│   ├── losses.py                           ← quantile ladders, delta anchors, scale_anchor_eps
│   ├── seed.py / seed_torch.py             ← ACTS three-point seed + residuals (numpy / GPU)
│   ├── hit_sorting.py                      ← truth-free detector-geometry hit ordering
│   ├── perigee.py                          ← truth perigee transport (3 T)
│   ├── flat_data.py / data.py              ← flat block-sampled store + packed collate
│   ├── mamba_cls.py / mamba_short.py       ← bidirectional CLS encoder + short-seq kernels
│   ├── ops/ssd_short_triton.py             ← fused Triton kernels (+ BL16 bucketing)
│   ├── muon.py                             ← Muon (2-D) + AdamW (1-D) hybrid optimizer
│   ├── scripts/preprocess_flat.py          ← parquet → flat store (sorting, seeds, truth-KF)
│   ├── scripts/fast_rms_eval.py            ← RMSE-vs-η bundles + per-pT tables (truth-KF ref)
│   ├── scripts/build_eval_farm.py, create_split.py, kf_baselines.py, ...
│   └── config/ssm_cls/ICLR_sweep5/         ← the final-recipe configs (see Configs)
└── scripts/
    ├── 06b_fetch_nersc_flat.sh / 06_fetch_nersc_ttbar.sh   ← raw parquet download
    ├── 11_rebuild_stores_v2.sh              ← preprocess all datasets with the v2 selection
    ├── 07_build_mixed_store.py              ← muon + ttbar mixed training store
    ├── 04_eval_ckpt_iclr.sh                 ← one checkpoint → all test sets → plots/tables
    ├── bench_infer_flat.py                  ← throughput benchmark (GPU seed, TF32, nsys)
    └── acts_integration.py                  ← the SSM as an ACTS IAlgorithm (pyacts ≥ 47.5)
```

## Quickstart

```bash
# 1. pixi env (CUDA 12.x + PyTorch pinned by the lockfile)
pixi install --locked

# 2. raw data: ColliderML drift-beamspot parquet from the NERSC portal
#    (particles, tracker_hits, tracker_simhits, tracks, truth_tracks)
bash scripts/06b_fetch_nersc_flat.sh          # muon guns
bash scripts/06_fetch_nersc_ttbar.sh 0 784    # ttbar runs

# 3. preprocess → flat stores + eval farm + KF baselines (v2 selection:
#    true-time hit order, 3 T targets, |d0|<=7.1mm, |z0|<=270mm, ttbar 1-110 GeV)
bash scripts/11_rebuild_stores_v2.sh

# 4. mixed training store (muons + ttbar; val = muon val + ttbar val)
pixi run -e default python scripts/07_build_mixed_store.py --extra-val ...

# 5. train the final recipe (single H100, ~45 h for 25 epochs on the full mix)
cd src/track_regression
pixi run -e default python train.py fit \
  --config config/ssm_cls/ICLR_sweep5/YZ3L_qrel_3L_bs2048_onecycle25.yaml

# 6. optional tail-cleaning fine-tune (2x H100 DDP, Muon-hybrid + WSD)
pixi run -e default python train.py fit \
  --config config/ssm_cls/ICLR_sweep5/YZ3LFT_qrel_3L_mix3_muonhybrid_ddp2_bs40k_wsd50.yaml

# 7. evaluate one checkpoint on every test set (plots + per-pT tables)
bash scripts/04_eval_ckpt_iclr.sh <run_dir> last.ckpt <out_dir> <eval_farm_root> <gpu>
```

Dataset paths in the shipped configs are absolute under `/scratch/colliderml/`;
edit them in lock-step with the scripts if your scratch lives elsewhere.

## Dataset

Source: the ColliderML **drift-beamspot** production
([portal.nersc.gov/cfs/m4958/ColliderML/drift_beamspot](https://portal.nersc.gov/cfs/m4958/ColliderML/drift_beamspot/);
same simulation chain as [arXiv:2512.15230](https://arxiv.org/abs/2512.15230) —
ODD detector, Geant4 via DD4hep, ACTS reconstruction — with a drifting beamspot
and a 3 T solenoid). Per event, four parquet tables are used: `particles`
(truth), `tracker_hits` (measurements = network inputs), `tracker_simhits`
(Geant4 truth, used only for the true-time sort key at preprocessing),
`truth_tracks` (the truth-tracking-KF reference fit) and `tracks` (CKF).

| dataset | tracks (train split) | role |
|---|---|---|
| `single_muon_uniform` (1–110 GeV) | 191.5 M | training + eval |
| `single_muon_loguniform` (0.9–110 GeV) | ~190 M | training (low-pT statistics) |
| `single_muon_{2,10,100}GeV` | 100 k each | eval only |
| `ttbar` runs 46–784, pT 1–110 GeV | 15.8 M | training (hadron component) |
| `ttbar` runs 6–45, pT 1–110 GeV | — | eval only (`ttbar_new_pt1`, 954 k tracks) |

`preprocess_flat.py` applies the selection, sorts each track's hits (default
`--sort-key geometry`; the v2 stores use `true_time`, which equals geometry
order on ~100 % of tracks and is truth-free at inference), recomputes the
perigee targets at 3 T (`--bz 3.0`), joins tables **by id values, never by row
order**, and writes a memory-mapped flat CSR store per split
(`hits.npy, offsets.npy, lengths.npy, targets.npy, acts_reco.npy, acts_dm.npy,
truth_kf_reco.npy, ...` per part) with the write-time shuffle the block sampler
relies on. Every manifest records `hit_sort_key`, `bz` and the selection.

The packed collate (`flat_data._pack`) emits per track: `hit_features (L, 15)`
(columns `x, y, z, r, φ_hit, θ_hit, s, volume_id, layer_id, surface_id,
detector, η_hit` + the 3 seed residuals), the five targets, and `seed_<p>`
anchor values — the seed is computed inside the collate (or on the GPU at
inference via `seed_torch.gpu_seed_features`).

## Evaluation

- Reference: the **truth-tracking Kalman filter** (`truth_tracks`), the
  strongest classical fit available; the CKF is no longer drawn.
- Metric: iterative 3σ-clipped RMSE per parameter (plus the pre-clip RMSE and
  clipped fraction, always reported next to it), unbinned and binned in η / pT.
- `scripts/04_eval_ckpt_iclr.sh` runs one checkpoint over all test stores and
  produces `rms_summary.{txt,json}`, `rms_by_pt.txt` and RMSE-vs-η figure
  bundles (`fast_rms_eval.py`, bootstrap-free, ~35 s per store).
- θ is the regressed and reported parameter; "vs η" plots bin in η but show θ
  residuals (σ_η ≈ σ_θ/sin θ).
- **ACTS-side validation**: `scripts/acts_integration.py` (pyacts ≥ 47.5) runs
  the trained model as an ACTS `IAlgorithm` on the raw parquet — digitized
  information only — so ACTS's `TrackTruthMatcher` + performance writers score
  it exactly like the ACTS KF baseline (scipy Gaussian-fit resolution profiles,
  `resolutions.pdf`). v2 checkpoints need `--sort-key geometry
  --seed-residual-features --d0-max 7.1 --z0-max 270 --pt-min 1 --pt-max 110`;
  the gen3 ODD geometry JSONs and the geoid map CSV are required inputs.

## Official ACTS-pipeline evaluation (paper figures)

The paper's resolution figures are produced by embedding the model in the ACTS
event loop and scoring it with ACTS's own performance writers — the same harness
the classical fitters are qualified with. The pipeline is driven by
`scripts/run_acts_official_plots.sh` (per dataset) and needs, besides the model
checkpoint, a **pyacts** install (a self-contained venv at
`/shared/tracking/pyacts_env`, pyacts 47.6.1) and the gen3 ODD geometry assets
(`odd.json`, `gen3_material_map_map.json`, `geoid_map.csv`,
`odd-seeding-config-gen3.json`).

The v2 (2026 re-produced) ColliderML parquet changed schema in a way that
silently breaks the stock pyacts converter (nested `particle_ids`, truth
positions moved to `tracker_simhits`), so `scripts/make_acts_compat_parquet.py`
shims each dataset back to the Release-1 layout the converter expects (joining
truth by `event_id` value), and `scripts/filter_acts_hits.py` drops the handful
of hits whose mapped surface projects out of bounds (a v2 producer bug: ~4 in
45 M). Downstream plotting:

- `scripts/acts_integration.py --dump-residuals` — runs the SSM + truth-seeded
  KF in one loop, writes `resolutions.pdf` (native ACTS σ-vs-{pT,η} with ratio
  panels) and `matched_residuals.npz` (per-particle truth/SSM/KF perigee params).
- `scripts/acts_band_plots.py` — σ-vs-{η,pT} with the ACTS fit-σ uncertainty
  bands and SSM/KF ratio subpanels (no bootstrap).
- `scripts/acts_legacy_style_plots.py` — campaign-design RMS-vs-η and residual
  histograms (linear + log), iterative-3σ RMS and surviving-track counts in the
  legends.
- `scripts/paper_matrices.py` — the gradient-cosine matrix (diagonal saturates
  the scale, ±std per cell) and the SSM prediction-vs-truth confusion matrix.

Paper model = the 2-layer Muon-hybrid fine-tune (`R2L-FT`). The manuscript lives
in `/shared/tracking/NeurIPS_2026_SSM_Tracking` (ICLR branch); its figures are
assembled under `material/iclr/` via `material/iclr/sync_figures.sh`.

## Configs — the final recipe family (`config/ssm_cls/ICLR_sweep5/`, `ICLR_sweep7/`)

| config | what it is |
|---|---|
| `YZ3L_qrel_3L_bs2048_onecycle25.yaml` | **the recipe**: 3L, seed residuals + anchors, scale-free q/p, even weights, bs 2048, OneCycle 25 ep |
| `YZ2Lmix3_qrel_2L_bs2048_onecycle25.yaml` | 2-layer model on the 3-way mix (the paper model before its fine-tune) |
| `ICLR_sweep7/R2LFT_qrel_2L_mix3_muonhybrid_ddp2_bs40k_wsd50.yaml` | **the paper model**: Muon-hybrid fine-tune of the 2-layer, 1.48 M tracks/s deployed |
| `YZ3LFT_qrel_3L_mix3_muonhybrid_ddp2_bs40k_wsd50.yaml` | 3-layer fine-tuned twin (identical physics) |
| `YZ3L_qrel_3L_bs2048_onecycle25.yaml` | the 3L recipe pre-fine-tune |

Head configuration that defines the recipe (from the YAML):

```yaml
d0:    {type: quantile, weight: 1.0, delta_anchor: seed_d0,  norm: ±0.4 mm}
z0:    {type: quantile, weight: 1.0, delta_anchor: seed_z0,  norm: ±3.5 mm}
phi:   {type: quantile, weight: 1.0, delta_anchor: seed_phi, norm: ±0.015 rad}  # wrapped
theta: {type: quantile, weight: 1.0, delta_anchor: seed_theta, norm: ±0.01 rad}
qop:   {type: quantile, weight: 1.0, delta_anchor: seed_qop, scale_anchor_eps: 0.02, norm: ±2}
```

The `train.py` CLI auto-loads `base.yaml` from the config's own directory; a
leaf `callbacks:` list **replaces** the base list (re-add `KernelSwapCallback`).
Earlier campaign configs (`ssm_cls/ICLR*`, `transformer/`, `ssm/`,
`experimental/`) are retained as the ablation record; the transformer and
state-pool variants are parameter-matched baselines from campaign 1.

## Caveats and known issues

- **Strict fp32 for training and physics numbers** (`TRK_MATMUL_PRECISION=highest`).
  TF32 matmuls are a validated inference-only speed option (≤ 2.3 % physics
  change); nothing below TF32 has been validated.
- **Packed batching is required** for the seed-residual features and is the only
  path the flat store exercises; under DDP set
  `trainer.use_distributed_sampler: false` (the block sampler shards itself).
- **Never widen loss norm ranges on a kept head** when warm-starting; a fresh
  head may use new ranges.
- **The hit order is part of the model.** The packed encoder never re-sorts;
  feed hits in the same geometry order at inference that the store used in
  training (`hit_sorting.geometry_order`, or the GPU seed path which asserts it).
- Evaluation restricts to the CKF-double-matched subset (~68 % of tracks) so
  numbers are comparable across models and to the CKF; the truth-KF covers
  ~100 % and the unrestricted ratios are equal or better.
- `RegressionPredictionWriter` writes `<ckpt-stem>__test_predictions.h5` next to
  the checkpoint — `04_eval_ckpt_iclr.sh` copies the checkpoint first for this
  reason.
- Comet logging is offline by default (`logs/comet_offline/`); set
  `COMET_API_KEY` to mirror.

## Acknowledgments

Parts of this codebase — most notably the vendored bits under
`src/track_regression/_lib/` (transformer encoder, dense / attention /
activation / norm blocks, FlexAttention score-mods, padding helpers, and the
Lightning callback skeletons) and the overall Lightning-CLI + YAML configuration
layout — are derived from or inspired by
[samvanstroud/hepattn](https://github.com/samvanstroud/hepattn)
(upstream `HEAD` at the time of the vendoring drop:
[`1df05ccb00c1a4e5a22a7b76f6182c955c2d9def`](https://github.com/samvanstroud/hepattn/commit/1df05ccb00c1a4e5a22a7b76f6182c955c2d9def)).
hepattn is licensed under GPL-3.0; this repository is distributed under the
compatible GPL-3.0-or-later. We thank Sam van Stroud and the hepattn
contributors. The seed estimator is a port of
`Acts::estimateTrackParamsFromSeed`; the truth-tracking-KF reference fits and
the datasets are produced by the ColliderML team.

## Citation

```bibtex
% TODO: paper citation placeholder — fill in once the accompanying
% publication is released.

@article{Elitez2025ColliderML,
  title         = {{ColliderML}: The First Release of an
                   {OpenDataDetector} High-Luminosity Physics
                   Benchmark Dataset},
  author        = {Elitez, Do{\u{g}}a and Gessinger, Paul and
                   Murnane, Daniel and Raaholt, Marcus Selchou and
                   Salzburger, Andreas and Skov, Stine Kofoed and
                   Stefl, Andreas and Zaborowska, Anna},
  journal       = {arXiv preprint arXiv:2512.15230},
  year          = {2025},
  eprint        = {2512.15230},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ex},
  url           = {https://arxiv.org/abs/2512.15230}
}
```

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
