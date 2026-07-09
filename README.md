# Bidirectional state space models for charged-particle trajectory regression

Code release for a Mamba-2-based regressor of the five perigee track parameters
`(d0, z0, φ, θ, q/p)` from raw silicon-detector hits, evaluated against the
ACTS combinatorial Kalman filter (CKF) on the
[CERN/ColliderML-Release-1](https://huggingface.co/datasets/CERN/ColliderML-Release-1)
ttbar dataset.

The repository contains the model, three encoder variants — two
bidirectional **Mamba-2** (Dao & Gu, 2024) backbones (state-pool and
CLS-pool readout) and a parameter-matched **multi-head self-attention**
transformer baseline — so the architecture comparison is Mamba-2 vs
self-attention at matched parameter count, the loss family (quantile,
circular, Gaussian-NLL), the data preprocessing pipeline, and a
publication-quality plotting pipeline that emits per-parameter
residual plots, RMS-vs-η panels, and bootstrap-error tables.

The headline-numbers transformer baseline runs PyTorch SDPA in fp32 so
the encoder precision matches the SSM variants. The repository also
ships **FlashAttention-2** (Dao, 2023) support via the `flash-varlen`
backend — packed, padding-free attention with `cu_seqlens` boundaries
— as an opt-in bf16 speed path for users who can trade
precision-matched comparability for throughput.

**No checkpoints are shipped.** Train your own, or wire your existing
checkpoint into the fine-tune YAML via `pretrained_ckpt_path`.

## The track-fitting problem

![Track-fitting at HL-LHC](media/figure_overview.png)

**(a)** Perigee parameters of a charged-particle track in a solenoidal
magnetic field. The trajectory (orange) is a helix; **r** is the point
of closest approach to the beamline. `d0` is the perpendicular distance
from **r** to the *z*-axis (within the transverse plane shown faintly in
blue at *z = z0*); `z0` is the *z*-coordinate of **r**; `φ` and `θ`
parametrise the momentum **p** at the perigee; `q/p` is the signed
inverse momentum. *Schematic — not to scale.* Adapted from the ACTS
tracking documentation
([acts-project.github.io/tracking.html](https://acts-project.github.io/tracking.html)).

**(b)** A single proton–proton bunch crossing at the High-Luminosity
LHC produces thousands of charged-particle trajectories that must be
reconstructed from the sparse silicon-detector hit collection.
*Track finding* assigns hits to candidate trajectories and *track
fitting* estimates the five perigee parameters of each candidate; this
work targets the latter, given the hit-to-track assignment. Panel (b)
is reproduced from the ColliderML dataset paper *ColliderML: The First
Release of an OpenDataDetector High-Luminosity Physics Benchmark Dataset*
([arXiv:2512.15230](https://arxiv.org/abs/2512.15230); Elitez,
Gessinger, Murnane, Raaholt, Salzburger, Skov, Stefl, Zaborowska,
2025), which also provides the ttbar simulation used to train and
evaluate the models in this repository.

## Architecture

<p align="center">
  <img src="media/architecture.png" alt="Bidirectional Mamba-2 architecture" width="66%">
</p>

## Hardware

- Pretraining one config: ~60–100 h on **1× H100 80 GB** (batch size kept
  small for regularisation; extra GPUs do not help).
- Fine-tuning one config: ~2 days on **2× H100 80 GB DDP**
  (set via `trainer.devices: [0,1]`).
- Inference + plots: <2 h on a single H100 (bootstrap stats add ~3
  min/run).

Minimum to run anything: 1× GPU with ≥40 GB VRAM, 16 GB system RAM,
~150 GB scratch for the preprocessed dataset.

## Layout

```
.
├── README.md
├── LICENSE                          ← GPL-3.0-or-later
├── media/architecture.{pdf,png}     ← model diagram (also figure_overview.png)
├── pyproject.toml                   ← pixi/hatchling project config
├── src/track_regression/
│   ├── train.py                     ← Lightning CLI entry
│   ├── data.py                      ← memory-mapped dataset
│   ├── model.py                     ← TrackParameterRegressor LightningModule
│   ├── losses.py                    ← quantile / circular / Gaussian losses
│   ├── mamba_state.py               ← BidirectionalMambaEncoder (state pool)
│   ├── mamba_cls.py                 ← BidirectionalMambaCLSEncoder (CLS pool)
│   ├── transformer_encoder.py       ← parameter-matched transformer baseline
│   ├── muon.py                      ← Muon (2-D) + AdamW (1-D) hybrid optimizer
│   ├── eval_utils.py                ← residual / DM / bootstrap helpers
│   ├── callbacks.py                 ← RegressionPredictionWriter, MinimalGpuMonitor
│   ├── selection_utils.py           ← named-variant selection loader
│   ├── selection_p200_datasets.yaml ← preprocessing track-selection variants
│   ├── paper_plots/                 ← publication-quality plot pipeline
│   ├── scripts/                     ← preprocess + create_split
│   ├── config/<arch>/<stem>.yaml
│   └── _lib/                        ← minimal vendored library bits
│       ├── encoder.py / dense.py / attention.py / activation.py / norm.py
│       ├── cuda_timer.py / comet_logger.py
│       ├── flex/                    ← FlexAttention score-mods + masks
│       └── callbacks/               ← Checkpoint / GradientLogger / InferenceTimer / SaveConfig
└── scripts/
    ├── 00_download_data.sh
    ├── 01_train.sh CONFIG
    ├── 02_evaluate.sh CONFIG
    └── 03_paper_plots.sh
```


## Fast short-sequence kernels (GPU-optimization campaign)

The stock `mamba_ssm` chunked scan is built for sequences of thousands of
tokens; our tracks have at most 22 (20 hits + 2 CLS). The campaign on branch
`opt_kernel` replaces its evaluation with the algebraically identical
single-chunk SSD quadratic dual — one dense 22x22 lower-triangular product
per track, embarrassingly parallel, no chunk machinery and no batch ceiling.
Full record: `docs/perf/OPTIMIZATION_LOG.md`; campaign context: `CLAUDE.md`.

**Defaults (no action needed):** every SSM-CLS config now ships a
`KernelSwapCallback(variant="auto")` in its base callbacks —
- while **training**, the encoder runs the compiled pure-PyTorch quadratic
  dual (`v3c`): exact autograd, ~1.6-2.5x faster training steps;
- in **validation / test / predict**, it runs the fused Triton packed
  kernels (`v5pc`): ~5x inference throughput (0.18 -> 0.91 M tracks/s on
  H100, t2k 11.1 -> 2.2 ms) and a batch ceiling of 600K+ tracks per call.
Remove the callback entry from the config to run the stock kernels (`v0`).

Every variant is checkpoint-compatible (identical parameter names) and
passes a correctness-oracle chain (`tests/test_mamba2short.py`: fp64
algebra, stock-kernel parity, end-to-end golden vs the trained checkpoint,
<=1% physics-drift gate on 131k tracks, gradient parity). Notable: by fp64
referee the new path is ~50x closer to the exact math than the stock
kernel, whose internal TF32 matmuls it replaces with strict IEEE fp32.

**Precision (important):** training scripts historically set
`torch.set_float32_matmul_precision("high")` = TF32 GEMMs (10 mantissa
bits). `train.py` now honours `TRK_MATMUL_PRECISION`; **new trainings must
run with `TRK_MATMUL_PRECISION=highest`** (full IEEE fp32 everywhere —
fp64-referee showed TF32 gradients carry ~50% relative error on this
model). The default stays "high" only for numerics-compatibility when
fine-tuning historical checkpoints.

Ad-hoc benchmarking / variant selection:

```bash
# throughput of any variant (v0|v2p|v3|v3c|v4|v5|v5p|v5pc|auto)
pixi run -e default python scripts/perf/bench_variant.py \
    --config src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml \
    --ckpt <ckpt.ckpt> --variant v5pc --mode staged --batch-size 32768

# physics-drift gate (<=1% clipped-RMS per parameter, 131k-track subset)
pixi run -e default python scripts/perf/physics_drift.py --variant v5pc ...

# in code:
from track_regression.mamba_short import apply_variant
apply_variant(model, "v5pc")   # or "auto", "v3c", ...
```

The kernels are plain portable Triton (no Hopper-only features) and
re-autotune on RTX-class GPUs (deployment target: RTX 5000 Ada / 3090).

## Quickstart

```bash
# 1. install pixi env (5–10 min on a fresh machine; <1 min thereafter
#    because pixi caches the compiled mamba-ssm + causal-conv1d wheels).
#    Requires nvcc + a CUDA-12.x-compatible GCC.
#    Always use --locked: plain `pixi install` re-solves the lockfile
#    on every `pixi run` and triggers partial reinstalls, which is much
#    slower than locking once and reusing the cached env thereafter.
pixi install --locked

# 2. dataset — print the download / preprocess / split commands and run
#    them. Raw download ~1.8 TB, preprocessed ~140 GB. Output paths are
#    hard-coded to /scratch/colliderml/arxiv_retraining/... in the shipped
#    configs and scripts; edit those if your scratch lives elsewhere.
bash scripts/00_download_data.sh

# 3. train one or more configs
bash scripts/01_train.sh pretrain_ssm_cls
bash scripts/01_train.sh finetune_ssm_cls_muon  # set pretrained_ckpt_path first

# 4. evaluate (writes test_predictions.h5)
bash scripts/02_evaluate.sh finetune_ssm_cls_muon

# 5. publication plots + cross-run summary tables
bash scripts/03_paper_plots.sh
```

Outputs:
- `logs/paper_plots/<stem>/plots/*.{pdf,png}` — per-run figures
- `logs/paper_plots/_summary/{all_runs,ablation_*}.{csv,tex}` —
  cross-run tables with bootstrap 2σ confidence intervals, ready to
  `\input` into a LaTeX document
- `logs/paper_plots/<stem>/stats.{txt,json}` — per-parameter raw std,
  IQR/1.349, iterative 3σ-clipped RMS, all with bootstrap 2σ CI

## Environment

Pixi pins the full CUDA + Python + PyTorch toolchain. The first install
compiles `mamba-ssm` (2.3.0) and `causal-conv1d` (1.6.0) from source — no
prebuilt wheels exist on PyPI for these versions. Pixi caches the build
outputs so subsequent installs are fast.

Default environment ships PyTorch 2.9.1 + CUDA 12.8, Lightning 2.5.2,
flash-attn 2.8.3, lion-pytorch, the HuggingFace `datasets` package, and
the `colliderml` data-fetching CLI.

```bash
pixi install --locked                         # build .pixi/ from the lockfile;
                                              # plain `pixi install` re-solves
                                              # the lockfile on every `pixi run`
                                              # afterwards — much slower.
# optional: export COMET_API_KEY=…            # mirror logs to Comet (offline by default)
```

All dataset paths are hard-coded as absolute paths under
`/scratch/colliderml/arxiv_retraining/...` in the shipped configs and
scripts (see the `Dataset` section below). If your scratch directory is
elsewhere, edit the paths in `src/track_regression/config/**/*.yaml`,
`scripts/00_download_data.sh`, and the per-dataset constants at the top
of `src/track_regression/scripts/parallel_copy_to_scratch.sh`.

## Dataset

Source: [CERN/ColliderML-Release-1](https://huggingface.co/datasets/CERN/ColliderML-Release-1)
([Elitez et al., arXiv:2512.15230](https://arxiv.org/abs/2512.15230)).
Channel: ttbar 14 TeV pp. Detector: Open Data Detector (ODD), simulated
with Geant4 via DD4hep, reconstructed with ACTS.

Two preprocessed variants are needed:

| Variant | Pileup | Tracks | Size | Used by |
|---|---|---|---|---|
| `/scratch/colliderml/arxiv_retraining/p0_core_pretrain` | 0 | 71.5 M | ~47 GB | all pretrain configs |
| `/scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune` | 200 | 131 M (DM-only) | ~92 GB | all fine-tune configs **and** evaluation |

Selection: `pT ≥ 0.5 GeV`, `|η| ≤ 3`, `|d0| ≤ 2.5 mm`, `min_hits = 6`,
`max_hits = 20`, `|z0| ≤ 200 mm`, charged primary tracks. The
`core_kf_matched` variant additionally requires ACTS double-matching
(hit-purity > 75 % AND hit-efficiency > 75 %).

### Per-track sample layout

After preprocessing, each variant lives at the path in the table above as a
set of `shard_XXXX/` subdirectories. Hits are packed into a
memory-mappable CSR layout (`hits.npy`, `hit_indices.npy`, `offsets.npy`,
`targets.npy`, plus the sort-key sidecar `hit_times.npy` and the ACTS
sidecars `acts_reco.npy` / `acts_dm_mask.npy`). Only hits belonging to
selected tracks are stored — ~30–250× smaller than the raw parquet.

The dataloader yields one dict per track:

| key | shape | dtype | notes |
|---|---|---|---|
| `hit_features` | `(L, 12)` | float32 | per-hit features (column layout below) |
| `hit_s` | `(L,)` | float32 | column 6 of `hit_features`, surfaced separately for posenc / loss anchors |
| `hit_time` | `(L,)` | float32 | truth time of each hit (ns, unsmeared) — encoder sort key |
| `targets` | `(5,)` | float32 | `[d0, z0, phi, theta, qop]`, units mm / mm / rad / rad / 1·GeV⁻¹ |
| `length` | int | — | actual number of hits, `6 ≤ L ≤ 20` |
| `acts_reco` | `(5,)` | float32 | ACTS-CKF parameters for the same track (optional) |
| `acts_dm` | bool | — | ACTS double-matched flag (optional) |

Derived quantities used in evaluation: `pT = sin(θ) / |qop|`,
`η = -ln tan(θ/2)`.

`hit_features` columns (in order): `x, y, z, r, phi_hit, theta_hit, s,
volume_id, layer_id, surface_id, detector, eta_hit`. Positions are in mm,
angles in rad; `volume_id / layer_id / surface_id / detector` are integer
detector identifiers cast to float32. **`s = √(x² + y² + z²)` is the 3-D
straight-line distance from the IP** — it is kept as an input feature, but
the encoder sort key is the separate `hit_time` sidecar, *not* `s`. Truth
time is monotonic in on-helix arc length and gives the physically correct
inward → outward ordering on every track; `s` loses monotonicity on
low-`pT` curling tracks whose radial coordinate decreases (the helix
spirals back inward) while `|z|` still grows, so sorting by `s` permutes
hits across the helix arc on those tracks. Empirically (shard 0 of the
`p0_core_pretrain` variant restricted to the double-matched subset,
56,187 tracks) `s`-order and `hit_time`-order agree exactly on 98.04 %
of tracks; the 1.96 % that disagree are concentrated at low `pT`
(≲ 5 GeV) and central `|η|` (≲ 1.5), peaking at 4.28 % in
`|η| ∈ [0.5, 1.0)` and dropping to ≤0.05 % for `|η| > 2` (forward tracks
are fine under `s` because `|z|` dominates). CLS tokens,
when used, are inserted *after* sorting.

Track length distribution (`core` selection): min = 6, max = 20, mean ≈
12–13, std ≈ 4. The 20-hit cap motivates `chunk_size: 16` in the SSM
configs.

### Batching modes

Both backbones support two batching modes, selected by `packed_batches` on
the LightningDataModule:

- **Padded (default).** Variable-length tracks are dynamically padded to
  batch-max length, with a kv-mask carried through the encoder. Compatible
  with every encoder variant.
- **Packed (opt-in, `packed_batches=True`).** Tracks are concatenated into a
  single `(1, total_L, D)` token stream accompanied by `seq_idx (1, total_L)`
  int32 segment IDs. Wastes no compute on padding positions. Implemented
  for the **SSM-CLS** encoder (`mamba_cls.py`).

### Sequence readout — CLS vs state

The repository ships two SSM readout variants. **CLS pooling is the
recommended default.** A learned CLS token at each scan terminus is
concatenated into a `(B, 2·dim)` vector that feeds the regression head; at
matched total parameters it consistently outperforms the state-pool readout
(which projects the final recurrent hidden state via `fwd_head` /
`bwd_head` Dense stacks), with two practical advantages: (i) the readout
dimension is independent of `d_state`, so the encoder scales depth-first
without bloating the head, and (ii) only the CLS path supports packed
batching. The state-pool variant is retained as an ablation reference.


### Simulation

Events are generated with MadGraph + Pythia8 (`ttbar`, 14 TeV pp),
simulated in Geant4 via DD4hep on the Open Data Detector, and digitised +
pattern-reconstructed with ACTS (CKF). See
[Elitez et al., arXiv:2512.15230](https://arxiv.org/abs/2512.15230) for the
full simulation chain.

Building each variant takes three steps. Run `bash scripts/00_download_data.sh`
to print the exact commands, or follow the recipe directly. All target
paths are absolute (`/scratch/colliderml/arxiv_retraining/...`); change
them in lock-step with the configs if your scratch lives elsewhere.

1. **Download raw parquet shards** from HuggingFace via the `colliderml`
   CLI (installed by `pixi install --locked`). Three physics objects per pileup
   — `particles` (truth), `tracker_hits` (Geant4 silicon measurements),
   `tracks` (ACTS CKF reconstruction) — totalling ≈ 1.8 TB. The exact
   per-config commands are printed by `scripts/00_download_data.sh`.

2. **Preprocess** — apply the selection in
   `src/track_regression/selection_p200_datasets.yaml`, pack a compact CSR layout that
   stores only the hits belonging to selected tracks (~30–250× smaller
   than raw), and augment ACTS-CKF reconstructed parameters + the
   double-matched mask onto every track:

   ```bash
   cd src/track_regression
   # pretrain variant (hard scatter only)
   pixi run -e default python -m track_regression.scripts.preprocess_colliderml_compact \
       --data-dir   /scratch/colliderml/arxiv_retraining/raw/p0 \
       --output-dir /scratch/colliderml/arxiv_retraining/p0_core_pretrain \
       --selection-file selection_p200_datasets.yaml \
       --selection-variant core --selection '{"hard_scatter": true}' \
       --num-workers 8 --augment-acts
   # fine-tune variant (pileup-200, ACTS-DM-required)
   pixi run -e default python -m track_regression.scripts.preprocess_colliderml_compact \
       --data-dir   /scratch/colliderml/arxiv_retraining/raw/p200 \
       --output-dir /scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune \
       --selection-file selection_p200_datasets.yaml \
       --selection-variant core_kf_matched --selection '{"hard_scatter": false}' \
       --num-workers 8 --augment-acts
   ```

3. **Split** is automatic — the preprocessor writes
   `split.json` (default 90 / 5 / 5 train/val/test) next to the shards
   at the end of every run. Pass `--no-split` if you want to defer it,
   or override the ratios with `--train-frac / --val-frac / --test-frac`
   on the same `preprocess_colliderml_compact` call. To rebuild the split
   later (e.g. after manually adding shards), call `create_split` directly:

   ```bash
   pixi run -e default python -m track_regression.scripts.create_split \
       --preprocessed-dir /scratch/colliderml/arxiv_retraining/p0_core_pretrain
   ```

The raw parquets at `/scratch/colliderml/arxiv_retraining/raw/` are no longer needed after step 2
and can be deleted to reclaim ~1.8 TB.

## Pretraining (1× H100)

Four backbone variants. Each runs on hard-scatter data only
(`p0_core_pretrain`) for ~50 epochs under Lion + OneCycleLR.

```bash
bash scripts/01_train.sh pretrain_transformer_1cls
bash scripts/01_train.sh pretrain_transformer_2cls
bash scripts/01_train.sh pretrain_ssm_state
bash scripts/01_train.sh pretrain_ssm_cls
```

To resume from your own checkpoint or to use it for evaluation, pass
`--ckpt_path`:

```bash
bash scripts/01_train.sh pretrain_ssm_cls --ckpt_path /path/to/your.ckpt
```

## Fine-tuning (2-4× H100 DDP)

Three SSM-CLS optimizer ablations, each warm-starting from a
user-supplied SSM-CLS pretrain checkpoint via `pretrained_ckpt_path`
(set inside the YAML). The repo ships these pointing at `null` —
update them before launching:

```yaml
# in src/track_regression/config/ssm_cls/finetune_ssm_cls_*.yaml
model:
  pretrained_ckpt_path: /path/to/your/pretrain_ssm_cls/best.ckpt
```

Then:

```bash
bash scripts/01_train.sh finetune_ssm_cls_adamw     # AdamW + WSD
bash scripts/01_train.sh finetune_ssm_cls_lion      # Lion-continuation + WSD
bash scripts/01_train.sh finetune_ssm_cls_muon      # Muon (2-D) + AdamW (1-D) hybrid
```

## Evaluation

`scripts/02_evaluate.sh <stem>` runs `train.py test` and writes a
per-track HDF5 with predicted quantiles, point estimates, and matched
ACTS-CKF parameters for the double-matched (DM) subset.

By default `02_evaluate.sh` looks for the checkpoint at
`checkpoints/<stem>/best.ckpt`. Override with `--ckpt_path`.

```bash
bash scripts/02_evaluate.sh finetune_ssm_cls_muon \
    --ckpt_path /path/to/your.ckpt
```

Output: `logs/comet_offline/<run-id>/<ckpt-stem>__test_predictions.h5`.

## Publication plots

`scripts/03_paper_plots.sh` produces a self-contained bundle per run
under `logs/paper_plots/<stem>/`:

- `plots/target_vs_pred_summary.{pdf,png}` — 5-parameter grids
- `plots/rms_vs_eta_summary.{pdf,png}` — RMS vs η with bootstrap ±2σ band
- `plots/heatmap_pred_vs_truth_summary_{ssm,ckf}.{pdf,png}`
- `plots/residual_vs_pt_summary_{ssm,ckf}.{pdf,png}`
- `plots/residual_hist_summary_{linear,logy}_{preclip,postclip}.{pdf,png}`
- `stats.{txt,json}` — per-parameter raw std, IQR/1.349, and iterative
  3σ-clipped RMS, each with bootstrap 2σ CI
- `plots/individuals/` — per-parameter singles

The cross-run aggregator runs at the end and writes
`logs/paper_plots/_summary/{all_runs,ablation_*}.{csv,tex}` — LaTeX
tables ready for inclusion in a paper.

The script edits a list of run names at the top — drop in any subset of
the configs above whose checkpoints you want to plot.

## Configs

```
src/track_regression/config/
├── transformer/
│   ├── base.yaml                            (auto-loaded by train.py)
│   ├── pretrain_transformer_1cls.yaml       (1 register token, fp32 SDPA)
│   └── pretrain_transformer_2cls.yaml       (2 register tokens, parameter-matched to SSM-CLS)
├── ssm/
│   ├── base.yaml
│   └── pretrain_ssm_state.yaml              (state-pool readout, fp32 backbone)
├── ssm_cls/
│   ├── base.yaml
│   ├── pretrain_ssm_cls.yaml                (CLS-pool readout — headline pretrain)
│   ├── finetune_ssm_cls_adamw.yaml          (AdamW + WSD — headline fine-tune)
│   ├── finetune_ssm_cls_lion.yaml           (Lion-continuation + WSD)
│   ├── finetune_ssm_cls_muon.yaml           (Muon-hybrid + WSD)
│   ├── pretrain_ssm_cls_packed.yaml         (research variant — packed-batch path)
│   ├── pretrain_ssm_cls_kf_hits.yaml        (research variant — restrict hits to KF-recovered)
│   ├── finetune_ssm_cls_{adamw,lion,muon}_packed.yaml  (research variants — packed-batch fine-tune)
│   └── finetune_ssm_cls_muon_kf_hits.yaml   (research variant — Muon + KF-hits selection)
└── experimental/                             (research-variant subtree)
    ├── scaling/                              (depth-sweep ablations: 2/4/6/8/10 L)
    └── d0_cross_fix/                         (single-parameter d0-only ablation)
```

The headline numbers reported in the paper use only the seven non-`experimental/`
configs without `_packed` / `_kf_hits` suffixes: the three pretrain configs
(`pretrain_transformer_{1,2}cls.yaml`, `pretrain_ssm_state.yaml`,
`pretrain_ssm_cls.yaml`) plus the three fine-tune optimizer ablations
(`finetune_ssm_cls_{adamw,lion,muon}.yaml`). The `experimental/`
subtree and the suffixed SSM-CLS variants are reproducible research
variants that are not exercised by the README quickstart; the suffixed
SSM-CLS variants are kept alongside their parents because they inherit
from the same `ssm_cls/base.yaml`.

Naming convention: **`<role>_<backbone>[_<axis>].yaml`** where role ∈
{pretrain, finetune}, backbone ∈ {transformer_1cls, transformer_2cls,
ssm_state, ssm_cls}, axis is the optimizer for the fine-tune triplet.

The `base.yaml` in each subdir is auto-loaded by `train.py` (Lightning
CLI's `default_config_files` mechanism); the leaf YAML only overrides
the architecture-specific fields.

## Caveats and known issues

- **`num_workers=0` for inference.** DataLoader worker forks corrupt
  gzip-compressed h5 chunks. Pretrain/finetune may use workers freely.
- **fp32 backbone for headline numbers.** The Mamba-2 selective-scan
  kernel has a bf16 round-off ceiling on long-range integrated parameters
  (`z0`, `θ`, `q/p`). All shipped configs — pretrain and fine-tune,
  SSM and transformer — set `encoder_autocast_dtype: float32` so the
  SSM-vs-transformer architecture comparison is precision-matched. The
  transformer baseline therefore uses `attn_type: torch` (PyTorch SDPA)
  rather than `flash-varlen`, since the flash-attn-2 CUDA kernel only
  supports bf16/fp16. Users who want the bf16 flash-varlen speed at the
  cost of comparability can set `encoder_autocast_dtype: bfloat16` and
  `attn_type: flash-varlen` on the transformer configs at their own
  discretion.
- **`chunk_size: 16` for Mamba-2.** Smaller values crash the installed
  `mamba_ssm` Triton kernel; larger values waste compute on padding for
  ≤20-hit tracks.
- **Comet logging is offline by default.** `MyCometLogger` writes to
  `logs/comet_offline/<experiment-key>/`; `COMET_API_KEY` is read from
  the environment if you want to mirror to a Comet workspace.
- **`losses.py` registers Gaussian-NLL losses (`gaussian`,
  `gaussian_eta`)** that are not referenced from any shipped config.
  They are kept available for future studies.

## Acknowledgments

Parts of this codebase — most notably the vendored bits under
`src/track_regression/_lib/` (transformer encoder, dense / attention /
activation / norm blocks, FlexAttention score-mods, padding helpers,
and the Lightning callback skeletons) and the overall Lightning-CLI +
YAML configuration layout — are derived from or inspired by
[samvanstroud/hepattn](https://github.com/samvanstroud/hepattn)
(upstream `HEAD` at the time of the vendoring drop:
[`1df05ccb00c1a4e5a22a7b76f6182c955c2d9def`](https://github.com/samvanstroud/hepattn/commit/1df05ccb00c1a4e5a22a7b76f6182c955c2d9def)).
hepattn is licensed under GPL-3.0; this repository is distributed under
the compatible GPL-3.0-or-later. We thank Sam van Stroud and the
hepattn contributors — reusing that work let us focus on the SSM
architecture, loss design, and ablations that are the contribution of
this repository.

## Citation

```bibtex
% TODO: paper citation placeholder — fill in once the accompanying
% publication is released (BibTeX entry for this repository / paper
% goes here).

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
