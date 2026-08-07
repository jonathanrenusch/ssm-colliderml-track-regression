# CLAUDE.md — ssm-colliderml-track-regression

Code release of *"Bidirectional State-Space Models for High-Precision Track Fitting at the LHC"*
(Renusch et al.). This repo is currently hosting the **GPU kernel-optimization campaign**
(branch `opt_kernel`) — the follow-up publication line. Results must be publication-grade:
reproducible, logged, methodology documented.

## Environment

- CERN k8s pod, **2× NVIDIA H100 NVL 95 GB** (400 W cap, max boost 1785 MHz). Likely final
  deployment target is an RTX/Ada (sm_89) box → mainline optimizations must stay portable
  (no Hopper-only TMA/wgmma paths).
- Run everything through pixi from the repo root: `pixi run -e default python …`
  Set `TRITON_CACHE_DIR=/tmp/triton_cache`.
- default env: python 3.12.0, torch 2.9.1+cu128, triton 3.5.1, **mamba_ssm==2.3.0 (pinned)**,
  causal_conv1d 1.6.0, comet_ml, pandas/matplotlib/h5py (NO rich/pynvml).
  A second env `mamba232` (mamba_ssm==2.3.2) exists only for baseline comparison — never
  change the default pin.
- Profilers: `ncu`, `nsys`, `compute-sanitizer` at `/usr/local/cuda-13.1/bin` (ncu also in env).
- **Filesystems:** `/shared` = NFS, survives restarts, home of everything overnight-critical
  (this repo lives there). `~/.claude` is on **AFS → dies when the Kerberos token expires**;
  before overnight work: `kinit && aklog && krenew -b -t -K 60`, run in `tmux`. `/tmp` does NOT
  survive pod restart (Triton cache only). `/scratch` = node NVMe with the datasets.
  Full protocol: `OVERNIGHT_CHECKLIST.md`.

## Data & checkpoints

- **⚠ Data-path trap:** configs reference `/scratch/colliderml/arxiv_retraining/p200_core_kf_hits_finetune`
  which does NOT exist. Real dir: `/scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune`
  (1000 shards, manifest max_hits=20, 131.5 M tracks). `scripts/perf/common.py` applies the
  override; anything else needs `--data.preprocessed_dir` override.
- Optimization-target config: `src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml`
  (4L BidirectionalMambaCLSEncoder, dim=128, d_state=16, headdim=32→8 heads, ngroups=1,
  chunk_size=16, RMSNorm, pool=ssm_cls, fp32, trained **packed**).
- Checkpoints (all verified on disk under `logs/src/track_regression/logs/comet_offline/`):
  - **4L target:** `1e0f5105c86d4bdd98a0cd3fa780f7dc/ckpts/epoch=049-val_total=0.00125.ckpt`
  - 10L dim192 d_state32 (paper shape): `76304d6ec483428b806456c3a6b9bbf7/ckpts/epoch=048-val_total=0.00110.ckpt`
  - 10L dim128 d_state16 (depth scaling): `e149d7efa6b847a689525e61ab7560cb/ckpts/epoch=047-val_total=0.00117.ckpt`

## Architecture map

`train.py` (LightningCLI) → `model.py::TrackParameterRegressor` (forward ~495-616; pool
selection ~331-399; `predict_step:816` → `losses.py::predict_physical:1212`) →
`mamba_cls.py::BidirectionalMambaCLSEncoder` (padded forward + `_forward_packed`; CLS layout
`[cls_bwd, h_0..h_{L-1}, cls_fwd]`) → `mamba_state.py` (`BidirectionalMambaLayer`,
`Mamba2WithState`, ssm_state encoder) → `mamba_short.py` (**campaign: chunk-free quadratic-dual
Mamba2 for L≤22**, identical param names to stock `Mamba2`). Eval:
`eval_utils.py::iterative_rms_convergence` (~322; iter-3σ-clip, ≤5 passes, returns `rms` =
sqrt(mean(x²)) — **the headline physics metric**), `compute_residuals` (:142). Prediction dump:
`callbacks.py::RegressionPredictionWriter`. Packed collate: `data.py::collate_tracks_packed:526`.
Tests / correctness oracles: `tests/test_packed_equivalence.py`, `tests/test_mamba2short.py`.

## The kernel campaign (context for any session picking this up)

- **Physics prior:** sequences are ≤22 tokens (≤20 hits + 2 CLS). The stock mamba_ssm chunked
  scan is built for L≫8000 → pure overhead here, plus a batch ceiling (~60K packed tokens;
  grid-axis-1 65535 limit for padded). At one chunk the SSD math collapses to a dense 22×22
  quadratic form per track — embarrassingly parallel, no ceiling.
- **Variant ladder:** V0 stock packed (reference) → V2′ corrected padded-static layout on the
  stock kernel (parity bridge; the OLD padded path is knowingly wrong for variable lengths:
  pad leakage + CLS placement + hit_time=0 sort bug) → V3 `Mamba2Short` pure-torch + compile
  (correctness reference + fallback) → **V4 fused Triton kernel (the centerpiece)**.
- **Hard constraints:** the Mamba2 math is NEVER altered (algebraic re-expression only).
  Accept gates for every variant: golden match vs stock on the trained checkpoint
  (atol/rtol 1e-3) AND ≤1 % per-parameter clipped-RMS drift on the fixed 1-shard
  (~131K-track) physics subset. V0 numbers always reported.
- **Precision policy (user, 2026-07-07): fp32 end-to-end; NO bf16/fp16 experiments.**
  Trap: `train.py:26` sets `torch.set_float32_matmul_precision("high")` → production numerics
  include TF32 *linears* (scan internals fp32). Golden reference is captured under production
  flags; benches must pin flags explicitly and record them. cuDNN conv defaults to TF32 →
  `Mamba2Short` uses shifted-MAC conv, not cuDNN. Stock kernel's `tl.dot` is TF32 internally
  → fp64 reference calibrates match tolerances (O1).
- **Parity traps** (all verified in installed source): `norm_before_gate=False` → gated BEFORE
  RMSNorm; in_proj split [z:256, xBC:288, dt:8]; conv over joint xBC then split; D-skip on raw
  post-conv x; dt softplus **thresholded at 20**; A=−exp(A_log.float()); B/C shared over heads.
- **KPIs:** tracks/s and usable batch/VRAM co-primary; headline `t_2K = 2000 × per-track-time`
  at the throughput-optimal batch, **target ≤ 0.5 ms (≥4 M tracks/s), 4L config**. Power/SM-util
  are diagnostics. Measured numbers ALWAYS outrank analytical estimates.
- **🚫 NEVER launch a retraining run without the user's explicit morning approval.** Retrain
  candidates (ssm_state pooling, Mamba-3 trapezoidal / conv-removal / MIMO) are *prepared* as
  configs + cost estimates only.

## Perf framework (campaign tooling)

- Queue runner: `pixi run -e default python scripts/perf/runner.py --queue scripts/perf/queues/nightN.yaml`
  (2 GPU lanes, subprocess-isolated jobs, resumable via `docs/perf/results/nightN/queue_state.json`;
  re-run the same command to resume). Profiler jobs run exclusive.
- One-off bench: `scripts/perf/bench_variant.py --variant {v0,v2p,v3,v3c,v4} --mode {staged,e2e,sweep,confirm}`.
- Physics gate: `scripts/perf/physics_drift.py --variant vX` (PASS = all 5 params ≤1 % drift).
- Golden capture: `scripts/capture_golden.py` → `docs/perf/results/golden/`.
- Profiles: `scripts/perf/parse_profiles.py` → tables + PNGs under `docs/perf/plots/nightN/`.
- **Central log: `docs/perf/OPTIMIZATION_LOG.md`** (append-only, dated nightly sections,
  morning feedback recorded there). Results: `docs/perf/results/nightN/results.{jsonl,csv}`,
  live tail: `docs/perf/results/night_run.log`. Comet: project `ssm-track-perf`, one experiment
  per night (offline-zip fallback under logs/ if comet is unreachable).

## Campaign status (update me — newest first)

- **2026-07-24 (streaming-loader val-skip FIXED, 4L kf_hits fine-tune relaunched):**
  With `num_workers>1` the streaming loader yielded fewer batches than the
  DataLoader length estimate (per-worker `drop_last` remainders), so Lightning's
  modulo-based end-of-epoch check never fired → **validation silently skipped**
  (the 2026-07-21 100-epoch 4L run had no val metrics at all); same arithmetic
  explains the multi-GPU epoch-end NCCL hang. Fix in `data.py`:
  `ColliderMLStreamingDataset.batches_per_epoch()` computes the exact per-epoch
  count (min across ranks) and `train_dataloader()` caps
  `trainer.limit_train_batches` to it each reload. Verified exact at 1/5/12
  workers vs a real DataLoader + 2-epoch smoke run (val logged, ckpts saved).
  Relaunched 50-epoch 4L kf_hits fine-tune on GPU0 via nohup (12 workers, log
  `logs/finetune_kfhits/finetune_4L_kfhits_shortkernel_valfix.log`). Multi-GPU
  fix is by-construction, untested (GPU1 busy with the 10L run).

- **2026-07-09 ~14:30 (day 3 wrap — FINAL pretrain launches v4):** tmux
  pretrain_4L (~33 it/s) + pretrain_10L (~12.8 it/s), full IEEE fp32, `auto`
  kernel (v3c train / v5pc eval) now the CONFIG DEFAULT via KernelSwapCallback
  in the base configs. ⚠ Leaf-level `trainer.callbacks:` lists REPLACE the base
  list — repeat the KernelSwapCallback in any leaf that defines callbacks (this
  silently reverted the 10L to the stock kernel once; found by py-spy). Fused
  Triton Lion default; crossing metrics val-only; diagnostics @500. R&D dead
  code removed; suite 52/52; commit 1347e37; README documents the kernels.

- **2026-07-09 ~12:40 (precision audit → pretrainings RELAUNCHED as v2):**
  fp64 referee proved the first launches trained with TF32 gradients (train.py's
  "high"): grad rel-err median **53%** vs 0.3% under IEEE. Fixes: (1) train.py now
  reads `TRK_MATMUL_PRECISION` (default "high" for old-ckpt compatibility; **new
  trainings MUST set =highest**); (2) `train_metrics_every_n_steps` init arg gates
  per-step diagnostics (launches use 50; loss/grads identical). Loader exonerated
  (144.7K tracks/s standalone ≫ need). Net: 4L now **24.8 it/s IEEE** (was 19.6
  TF32), 10L 8.1 it/s (was 10.6 — pays the fp32-GEMM cost; precision per user
  directive). tmux pretrain_4L / pretrain_10L, logs `*_shortkernel_v2.log`,
  epochs ≈17 min / ≈51 min. **DO NOT KILL. DO NOT LAUNCH NEW TRAININGS WITHOUT
  TRK_MATMUL_PRECISION=highest.**

- **2026-07-09 ~02:15 (night 3 COMPLETE — TWO PRETRAININGS RUNNING, user-approved):**
  tmux `pretrain_4L` (GPU0, ~19.6 it/s, epoch ≈21 min) and `pretrain_10L` (GPU1,
  ~10.6 it/s, epoch ≈40 min), both batch 2048 fp32 with **v3c** via
  KernelSwapCallback, dataset `p0_core_kf_hits_pretrain` (copied to /scratch, 56.5M
  tracks), configs `*_shortkernel.yaml`, comet TRK-SSMCLS-pretrain-{4L,10L}-shortkernel.
  **DO NOT KILL THESE SESSIONS.** Training speedups measured in real train.py runs:
  pretrain@2048 **1.56×**, finetune 2×H100 **1.50×** (v3c @40960/rank = 85 GiB/GPU,
  1.86× the config batch). v5p/v5pc made trainable (exact recompute backward, O11
  grads ≈4e-7) but v5p trains 3.4× slower than v3c and v5pc-compiled training is
  blocked by an Inductor dynamic-shapes device assert → **v3c = training variant,
  v5pc = inference variant**. Research check: cuBLAS FP32-emulation (3xTF32/BF16x9)
  is the one remaining kernel-level avenue at fp32 accuracy — needs cuBLAS ≥12.9
  (we ship 12.8); dual-stream direction overlap is the other candidate.

- **2026-07-09 ~05:30 (night-2 extension):** State-eject encoder now runs the fast
  path (`apply_variant("v3c")` works on `BidirectionalMambaEncoder`; closed-form
  terminal state via `Mamba2ShortWithState`): **0.274 M tracks/s = 2.38× its stock,
  ceiling 31,744 → ≥262K tracks**, architecture untouched, timing-only (no trained
  ckpt). Kernel occupancy split (HPP) measured: no gain — kernel2p is at its
  structural plateau (~0.89±0.02 M for v5pc, autotune noise ±4%). Suite 51/51.

- **2026-07-09 ~03:30 (night 2 COMPLETE):** **v5pc is the new best and the
  recommended production variant: 0.907 M tracks/s (t2k 2.205 ms) = 5.06× over v0;
  max batch 606,208 tracks/forward; physics PASS, golden PASS, suite 51/51; 100%
  RTX-portable Triton (no Hopper code anywhere).** v5pc = PACKED stream (production
  collate, no pads anywhere) + kernel2p (packed row addressing) + torch.compile
  dynamic glue (`enable_packed_compile`) + maxnreg autotune. Packed beat padded
  23% head-to-head at equal kernels → adopted as standard. Measured-dead tonight:
  v6 in-kernel GEMM (72 ms/launch, SMEM re-materialization), shift-matrix conv
  (4× regression). Also: **training via v3c works incl. backward (2.45× step at
  BS 22K)**; state-eject encoder benched (0.115 M, own grid ceiling, needs a hook
  for kernel-swap — night-3 candidate); v5 maxes the 400 W cap at optimal batch;
  10L snapshot v5pc: dim192 3.6×, dim128 5.4× (+ fixed non-pow2 d_ssm bug found
  by that run). H100-custom stretch not attempted (portable-first). See
  OPTIMIZATION_LOG Night 2 for decisions & night-3 options.

- **2026-07-08 ~01:30 (night 1 COMPLETE):** Full ladder shipped and gated, ending at
  **v5 = 0.742 M tracks/s (t2k 2.70 ms) vs v0 0.18 M (11.1 ms) = 4.15×; usable batch
  524,288 tracks in one forward (77 GiB) vs stock's 69.5K launch ceiling (7.5×). All
  gates green: physics ≤0.2% drift (limit 1%) for v2p/v3/v3c/v4/v5 on 131K tracks;
  golden truth-anchored PASS; test suite 48/48.** Variant flags: v0/v2p/v3/v3c/v4/v4t/v5
  via `mamba_short.apply_variant`; **v5 (kernel 2, per-track head-loop) is the current
  best**. Stock ceiling measured exactly (909,386→909,699 tokens,
  `_chunk_cumsum_fwd_kernel` launch failure; 2.3.2 identical). 10L scaling: dim192
  2.4×, dim128 3.4×. Rejected on evidence: mamba_ssm upgrade, TF32-in-scan (v4t),
  fused dual-direction in_proj. Remaining 5.4× to the 0.5 ms target likely needs
  persistent-kernel GEMM fusion (projections dominate; see OPTIMIZATION_LOG Night 1
  Findings #12/#15 + morning questions). Runner idle; queue `night1.yaml` drained;
  repo uncommitted by design (commit only on user request).
- **2026-07-07 (night 1 start):** plan approved
  (`~/.claude/plans/read-replan-prompt-md-and-carry-reactive-toucan.md`). REPLAN_PROMPT.md
  deleted (superseded). `mamba232` pixi env added (mamba_ssm 2.3.2.post1; default env
  untouched). Perf harness landed under `scripts/perf/` (runner/bench/physics/profilers).
