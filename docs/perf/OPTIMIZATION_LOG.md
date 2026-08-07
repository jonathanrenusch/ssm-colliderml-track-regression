# OPTIMIZATION_LOG — GPU kernel campaign (branch `opt_kernel`)

Append-only nightly log of the kernel-optimization campaign. One section per
night; morning feedback is recorded in place. Conventions:

- **KPIs:** tracks/s and usable batch/VRAM co-primary; headline
  `t2k = 2000 × per-track-time` at the throughput-optimal batch,
  **target ≤ 0.5 ms (≥ 4 M tracks/s), 4L config**. Measured numbers ALWAYS
  outrank analytical estimates.
- **Gates per variant:** golden match vs stock on the trained checkpoint
  (atol/rtol 1e-3) AND ≤ 1 % per-parameter clipped-RMS drift on the fixed
  ~131k-track physics subset (`scripts/perf/physics_drift.py`). V0 numbers
  always reported alongside.
- **Precision:** fp32 end-to-end, `torch.set_float32_matmul_precision("high")`
  (production numerics, TF32 linears). Flags recorded in every result row.
- Results: `docs/perf/results/nightN/results.{jsonl,csv}`; queue state:
  `docs/perf/results/nightN/queue_state.json`; live tail:
  `docs/perf/results/night_run.log`; plots: `docs/perf/plots/nightN/`;
  profiles: `docs/perf/profiles/nightN/`. Comet project: `ssm-track-perf`.

---

## Night 1 — 2026-07-07

### Goal

Establish V0 (stock packed) baselines on the 4L target checkpoint: staged /
e2e / sweep + chunk_size=32 context point; diagnose the ~60K packed-token
ceiling; build the fixed physics subset and capture the V0 physics reference.
V2′/V3 (and V4) jobs are queued as placeholders pending
`track_regression.mamba_short`.

### Headline KPIs

_(pending — filled by `scripts/perf/report.py --results-dir
docs/perf/results/night1 --night 1`)_

| variant | best tracks/s | t2k [ms] | best batch | max batch | VRAM [GiB] | power mean [W] | physics gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | ref |

### 10L snapshot (depth-scaling curiosity metric; also the shape-generality check)

| ckpt | shape | v0 [M tracks/s] | v3c [M tracks/s] | speedup | t2k v0→v3c [ms] |
|---|---|---:|---:|---:|---|
| 76304d6e (paper) | 10L dim192 N32 H12 | 0.060 | 0.147 | 2.4× | 33.1 → 13.6 |
| e149d7ef | 10L dim128 N16 H8 | 0.073 | 0.248 | 3.4× | 27.3 → 8.1 |

`Mamba2Short` + the fused kernel run the dim192/H12/N32 production shape
unmodified (O3/O9 cover both shapes; old saved configs needed a stale-kwarg
filter in `scripts/perf/common.py`).

### Jobs

Queue: `scripts/perf/queues/night1.yaml` → `docs/perf/results/night1/`.

- `v0_staged` — staged forward, BS 22000, 200 iters
- `v0_e2e` — real dataloader incl. H2D, 220 batches
- `v0_sweep` — doubling from 1024 tracks to the ceiling, boundary bisected
- `v0_chunk32_staged` — stock kernel, `chunk_size=32` override (context point)
- `diagnose_packed60k` — token-ceiling binary search + compute-sanitizer
- `physics_subset_build` — fixed ~131k-track subset (heavy_cpu)
- `v0_physics_ref` — V0 physics reference npz + gate row

### Findings

**Interim (19:00, updated as the night progresses — final KPI table appended by report.py):**

1. **Oracle chain O1–O10 fully green** (`tests/test_mamba2short.py`). Key numerics
   discovered on the way:
   - O1: the stock kernel's own TF32-`tl.dot` noise floor vs an fp64 reference is
     **2.3e-5 (dim128) / 2.0e-4 (dim192)** at unit scale.
   - O2: `Mamba2Short`'s quadratic dual is algebraically exact (1.2e-15 vs an
     independent fp64 einsum reference).
   - **fp64 ground-truth experiment (golden batch, 512 tracks): |V0−truth| =
     5.3e-2 max / 2.3e-3 mean; |V3−truth| = 1.1e-3 max / 2.9e-5 mean → V3 is
     ~50× MORE accurate than the stock kernel.** Under production flags
     ("high", TF32 linears) V0's distance to truth grows to 0.39. The
     contractual "golden match at raw atol 1e-3" is unachievable against a
     TF32-noisy reference for ANY re-layouted evaluation (including the stock
     kernel itself, V2′) once trained activation scales (|x|≈40–130) amplify
     the per-layer ~5e-5 relative noise. **Gate amended** (evidence above):
     scale-normalized diff ≤1e-3 OR distance-to-fp64-truth ≤ 1.5× V0's own —
     plus the unchanged ≤1% physics gate as the binding arbiter.
     `golden_small{,_ieee}.pt` embed the fp64 truth (`pred_truth64`).
2. **Physics gates (131,072 tracks): v2p PASS (max |drift| 0.037%), v3 PASS
   (max 0.196%, φ), v3c/v4 pending** — far under the 1% gate.
3. **Batch ceiling reproduced and measured exactly**: stock packed V0 fails at
   **909,386 OK → 909,699 tokens FAIL** (≈69.5K tracks) with
   `Triton Error [CUDA]: invalid argument` raised at the
   **`_chunk_cumsum_fwd_kernel` launch** (verbatim traceback in
   `packed_ceiling_diagnosis.md`; bisected to ≤512 tokens, fresh subprocess
   per probe). Its grid is `(batch, nchunks, headgroups)` — nchunks≈56.9K is
   *under* the 65,535 axis limit, so the exact driver-level trigger is
   unresolved (logged honestly); the practical fact stands: the stock
   chunked machinery cannot launch past ~0.91M packed tokens on this stack
   (torch 2.9.1 / triton 3.5.1 / mamba_ssm 2.3.0 AND 2.3.2 — both measured),
   and PR #708 was closed unmerged upstream. **V3/V4 have none of this
   machinery** (batch on grid axis 0, limit 2³¹): v3 eager reaches ~270K
   tracks / ~70 GiB before clean OOM; v4 does 131K tracks in 19.3 GiB.
4. **Throughput ladder so far (staged, 4L, ~22K–131K tracks):**
   - v0 packed eager: **0.18 M tracks/s** (t2k ≈ 11 ms) — flat vs batch, and
     v3 eager is the same speed: profile shows ~70% of GPU time in unfused
     eager elementwise kernels (memory traffic), GEMMs only ~11%.
   - v3c (compiled static core): **~0.60 M tracks/s** (t2k ≈ 3.35 ms), 3.3×.
   - v4.0 (fused Triton scan kernel + compiled glue): **0.61 M tracks/s**,
     equal to v3c but ~35% less VRAM. Profile: the fused kernel itself is
     3.55 ms/launch at 32K tracks (~20–30% of its bandwidth roofline — poor
     coalescing, needs tuning) yet already 54% of core time; remaining time =
     projection GEMMs (~10 ms/iter) + gated-norm reduction (~5 ms/iter).
   - **Bottleneck analysis for the 4M-tracks/s target**: projections are
     ~40 MFLOP/track → need TF32 GEMMs at ~40% peak + near-zero everything
     else. Planned V4.1: in-kernel REVERSE (kills 4 flip-gathers/layer),
     fused fwd+bwd in_proj (one 128→1104 GEMM), kernel coalescing pass.
5. torch.compile of the FULL model (Fourier pipeline included) hits an
   Inductor bug (`NotImplementedError: SliceView`) — pipeline stays eager
   (only ~2.9 ms/iter at 33K tracks; negligible for now).
7. **v4 physics gate: PASS (max |drift| 0.097%)**; v3c PASS (0.178%). All four
   optimized variants pass on 131,072 tracks.
8. **mamba_ssm 2.3.2 verdict (night-1 deliverable): NO change** — 0.181 M
   tracks/s and the same 67,584-track sweep ceiling as 2.3.0 (grid-axis
   limit unfixed upstream). Upgrading the pin buys nothing; the custom line
   is the only path. `mamba232` env kept for Mamba-3 kernel style references.
9. **e2e ≈ staged for V0** (0.181 vs 0.177 M tracks/s at 22K batch) — the
   dataloader is not the current bottleneck.
10. **v4t (TF32 tl.dot inside the scan, = stock-kernel numerics) measured and
    REJECTED**: 0.56 M tracks/s vs 0.60 IEEE — 32×32 tiles are too small for
    tensor cores to pay; block-level diff 1.2e-4 (≈ stock noise floor) for
    negative gain. The IEEE kernel is both more accurate AND faster.
11. V4.1 structural experiments measured: fused dual-direction in_proj (one
    128→1104 GEMM) REGRESSED to 0.50 M — Inductor materialises a copy for
    each sliced opaque-op input; per-direction GEMMs + in-kernel REVERSE
    (no flip gathers) restored 0.60 M with VRAM down to 19.3 GiB at 131K
    tracks (v4 sweep). Custom one-pass gated-RMSNorm kernel added (replaces
    Inductor's 0.58 TB/s reduction). Remaining lever per profile: the scan
    kernel itself (3.55 ms/launch at 32K tracks ≈ 20–30% of roofline,
    latency/occupancy-bound — ncu WarpStateStats queued as
    `profile_ncu_v4_r2`); runner hang-detector fixed to exempt profiler
    jobs (ncu kernel-replay is silent for minutes → false 600s-stale kill).
6. Implementation note: packed→padded-static conversion happens INSIDE the
   encoder (arithmetic from cu_seqlens, no argsort), so dataloader, model.py
   and all eval plumbing are untouched; `apply_variant(model, vX)` is the
   only switch. CLS insertion is a grad-safe one-hot blend; backward
   direction uses a per-row valid-prefix flip (self-inverse).

### Artifacts

- results: `docs/perf/results/night1/results.{jsonl,csv}`
- job logs: `docs/perf/results/night1/jobs/*.log`
- diagnosis: `docs/perf/results/night1/packed60k_diagnosis.md`
- physics: `docs/perf/results/night1/physics/`, gate rows in
  `docs/perf/results/physics_gate.csv`
- plots: `docs/perf/plots/night1/`

### Late-night additions (post-KPI-table)

12. **ncu steady-state on the fused scan kernel** (standalone harness —
    profiling the full compiled app is intractable, ncu serialisation blew two
    timeouts; `v4_kernel_steady.ncu-rep`): **L1/TEX throughput 95% (the
    bottleneck), DRAM only 13%, occupancy 31% (register-capped, 5 blocks/SM)**.
    The 4-tap conv loads ×3 tensors ×8 head-programs (B/C re-read 8×) hammer
    L1. Night-2 kernel plan: (a) `(B,)`-grid head-loop variant — load B/C and
    compute G once per track (kills the 8× redundancy), per-head y written to
    disjoint out columns (no cross-head SSA issues); (b) beyond that, the
    endgame for 4M tracks/s is fusing the projections into a persistent
    kernel (weights stationary in SMEM) — GEMMs are now ~10 ms/iter vs the
    kernel's 3.5 ms.
13. **v4 ceiling confirmed in a fresh process: 524,288 tracks in ONE forward
    (6.84M tokens, 77 GiB, 0.551 M tracks/s)** — 7.5× the stock kernel's
    69.5K-track launch ceiling. Throughput-optimal batch is much smaller
    (~16–131K tracks, 0.585–0.611 M tracks/s).
14. Test-hygiene bug found by the final full-suite run and fixed: the O7
    golden check restores production "high" (TF32) matmul precision globally,
    inflating later IEEE-vs-IEEE tests (in-suite-only failures). Autouse
    fixture now pins/restores the flag per test. **Final suite: 47/47 green.**

15. **v5 (kernel 2: one program per track, heads looped in-kernel, B/C conv +
    Gram matrix computed ONCE) — built, gated, shipped the same night: 0.742 M
    tracks/s, t2k 2.696 ms, max batch 524,288, physics gate PASS (≤0.18%),
    golden truth-anchored PASS, suite 48/48.** This directly cashed in the ncu
    finding (#12): the 8× B/C L1 re-reads were worth +22% over v4. The 4L
    ladder ends the night at **4.15× V0** with t2k 11.1 → 2.70 ms.

### Decisions & questions for the morning

**Decisions taken tonight (all evidence-based, see Findings):**
- Golden-gate definition amended to scale-normalized / fp64-truth-anchored
  (Finding 1) — needs your sign-off as the contractual criterion going forward.
- mamba_ssm upgrade: rejected (no effect). TF32-in-scan (v4t): rejected
  (slower AND noisier). Fused dual-direction in_proj: rejected (Inductor copy).
- v4 ships as: per-direction cuBLAS projections + fused IEEE scan kernel with
  in-kernel backward flip + custom gated-norm kernel + compiled glue.

**Questions:**
1. The 0.5 ms t2k target: measured best tonight is **2.70 ms (v5)** — 5.4×
   away. The safe half of the night-2 plan (the (B,)-grid kernel rework) was
   already executed tonight as v5 (+22%). Roofline says ~40 MFLOP/track of
   TF32 projections bound the ceiling near ~0.9–1.5 ms t2k for this
   decomposition; reaching 0.5 ms needs the persistent-kernel GEMM fusion
   (weights stationary in SMEM, projections inside the kernel — high effort,
   genuinely hard, uncertain). OK to spend night 2 on it? Alternative night-2
   focus: dual-GPU aggregate throughput (2 cards ≈ 1.5 M tracks/s today) +
   kernel micro-tuning + retrain-track prep.
2. Physics-gate margin is huge (worst 0.2% vs 1%) — happy to keep 1 shard?
3. Retrain-track prep (night 2–3, per your selection): ssm_state pooling,
   Mamba-3 trapezoidal, conv-removal/MIMO — configs + cost estimates will be
   drafted night 2; NO launches without your explicit approval.
4. The old padded path is knowingly broken (pad leakage + CLS placement +
   hit_time-0 sort bug) — want a deprecation warning added to it?

---

## Night 2 — 2026-07-08

### Goal

Fused-projection kernel (V6) per the approved plan: portable-first
(deployment RTX 5000 Ada / 3090), packed-vs-padded head-to-head with the
fastest kernel, secondary track (training micro-bench, state-eject encoder,
e2e at scale) on GPU 1.

### Findings (running)

1. **V6 (in-kernel in_proj, one program per track) — built, CORRECT
   (O9 green first run), and measured DEAD: 72 ms/launch standalone
   (~25× slower than the whole v5 chain).** ncu: zero register spills but
   SMEM-limited to 2 blocks/SM (25% occupancy) and **DRAM at 66% for
   83 ms ≈ 185 GB moved** — with ~45 unrolled tl.dot ops in one program,
   Triton's codegen stages/re-materialises operands from global memory
   instead of registers. Not a tuning problem; the
   one-program-does-the-GEMM design is wrong for Triton at (22×128)@(128×552)
   granularity. cuBLAS keeps the projections. Code kept as variant `v6`
   (correct, documented-rejected). Consistent with the literature: cuBLAS
   wins pure GEMM; Triton wins fusing AROUND GEMMs, not replacing them at
   tiny shapes.
2. Revised route to the same traffic goal: **v5p — packed-stream execution
   of the v5 generation**: cuBLAS GEMMs run on the packed rows (~31% fewer
   rows, pad slots never exist), the scan kernel gets packed row addressing
   from cu_seqlens, glue compiles with a bucketed static stream length.
   This is also exactly the quantitative packed-vs-padded head-to-head the
   user required.
3. **Packed vs padded head-to-head (user-required, measured at 32,768 tracks,
   equal kernel treatment):** v5 padded 0.739 M (t2k 2.71 ms) vs **v5pc
   packed 0.910 M (t2k 2.20 ms) — packed wins by 23%** and uses less VRAM
   (3.4 vs 4.9 GiB). Sequence: v5p eager glue +4.6% → `torch.compile(
   dynamic=True)` over the packed layer stack (`_packed_core` /
   `enable_packed_compile`, dynamic shapes work incl. the opaque custom
   ops) +18% → maxnreg autotune configs +5%. **PACKED (v5pc) is adopted as
   the standard.** No static conversion, no pad slots anywhere; the
   production packed collate feeds it unchanged.
4. **Shift-matrix conv inside kernel2p: measured and REVERTED** — 4×
   regression (kernel3's SMEM-staging pathology in miniature). Strided
   masked loads stay. Second confirmation tonight that at these tile sizes
   extra tl.dot ops cost more than the L1 traffic they save.
5. **v5pc gates: physics PASS (131K tracks), golden truth-anchored PASS,
   suite 51/51.** Official rows: staged 0.886 M @32K (t2k 2.257, 3.4 GiB);
   sweep best **0.907 M (t2k 2.205 ms)**; **max batch 606,208 tracks in one
   forward** (> padded's 524K — no pad slots). 10L snapshot with v5pc:
   dim192 **0.217 M (3.6× its v0)** — also caught a non-pow2 d_ssm=384 bug
   in the gated-norm kernel (fixed, masked block); dim128 **0.398 M (5.4×)**.
6. Night-2 4L ladder summary: v0 0.18 → night-1 best v5 0.74 → **night-2
   best v5pc 0.91 M tracks/s (t2k 11.1 → 2.20 ms, 5.06×)**. Portable Triton
   only — no Hopper-specific code exists; the optional H100-custom stretch
   was NOT attempted (the v6 detour consumed the margin; portable-first per
   user directive).
7. Secondary-track results (GPU 1, see agent table below): **training with
   v3c works incl. backward — 2.05× step at BS 2048, 2.45× at BS 22000**
   (CPU collate now the next training bottleneck; padded-static autograd
   costs ~2.5× training VRAM); **state-eject encoder (9L dim128 N32,
   padded-native) 0.115 M tracks/s with its own 65,535 grid ceiling at
   32,768 tracks** (kernel-swap needs a padded-static/packed hook in
   `BidirectionalMambaEncoder` — night-3 candidate, architecture unchanged);
   **v5 e2e @131K = 744K tracks/s, +2% over staged** (dataloader fully
   overlapped); **v5 at optimal batch runs AT the 400 W cap (355.7 W mean /
   400.6 peak, SM 86%, mild clock throttle)** — the "power not maximised"
   question is closed: the fast variants do max the card at their optimal
   batch; tracks/s/W is up 4× vs v0.

### Night-2 extension (post-reset continuation, user go-ahead)

8. **Kernel2p occupancy split (HEADS_PER_PROG in the autotune space):
   measured, NO GAIN** — the autotuner keeps all-heads-per-program; the
   B/C+G recompute cost eats the occupancy benefit. Configs kept (harmless;
   they matter on other shapes/GPUs). Run-to-run autotune noise is ~4%
   (0.874–0.910 M for identical code) — quote v5pc as ~0.89 ± 0.02 M.
   The kernel is at its structural plateau; further gains need a different
   decomposition, not tuning.
9. **State-ejection encoder on the fast path (user request) — DONE,
   timing-only, architecture untouched**: intermediate layers take
   `Mamba2Short` directly; the terminal state comes from the closed form
   (`Mamba2ShortWithState`: h_T = Σ_s exp(cumA_T−cumA_s)·dt_s·x_s⊗B_s),
   verified vs the stock scan at 1.2e-4 on random weights.
   **As configured (9L dim128 N32, padded, random weights): 0.274 M
   tracks/s (t2k 7.30 ms) vs stock 0.115 M (17.40) = 2.38×; ceiling
   31,744 → ≥262,144 tracks.** `apply_variant(model, "v3c")` now handles
   the state encoder transparently. Suite stays 51/51.

### Decisions & questions for the morning (night 2)

- **v5pc (packed, compiled, portable) is the recommended production
  variant**: 0.91 M tracks/s, t2k 2.20 ms, all gates green, zero
  Hopper-specific code. The 0.5 ms target stands at 4.4× away.
- Both in-Triton-GEMM routes are now measured-dead (v6, shift-conv); the
  projections (~7.6 ms/iter cuBLAS TF32) + scan kernel (~20 ms/iter at 32K)
  bound this decomposition near ~1.3–1.5 M tracks/s with further kernel
  occupancy work. Honest options for night 3: (a) deep kernel-2p occupancy
  rework ((B,2)-grid half-register split, measured stepwise), (b) accept
  ~1M-class throughput and shift focus to the state-eject encoder hook +
  training-path polish + RTX portability validation package, (c) revisit
  invasive/architectural levers (needs retraining, your call entirely).
- 10L192 t2k is 9.2 ms — if the paper-shape encoder matters for deployment,
  night-3 could tune the dim192 shapes specifically (autotune keys exist).

- Walkthrough given; **golden-gate amendment APPROVED**.
- Night-2 focus approved: fused-projection kernel (V6), **portable-first**
  (deployment target = RTX 5000 Ada / RTX 3090 at CERN; H100-custom max-out
  variant strictly optional at the end).
- **bf16-retrain idea struck**: the 5th-decimal regression floor + atan2
  heads need fp32 — not a campaign item (intern project someday).
- Packed vs padded: must be a quantitative head-to-head with the fastest
  kernel before packed becomes the standard.
- Dual-GPU scaling: no engineering content (2 independent processes = 2×);
  dropped as a work item.
- Training: interested — fwd+bwd micro-bench queued (v3c is trainable today;
  v5/v6 backward is a night-3 recompute item).
- State-ejection encoder: bench as configured (no arch changes, no physics
  gate — no trained ckpt), secondary track tonight.
- Deprecation warning on the legacy padded path: yes (added).
- Batch-not-maximized and power questions answered (throughput flat vs batch;
  fewer W = less data motion; tracks/s/W ↑4×) — see conversation log.

### KPI table (auto, 2026-07-07T16:46:16+00:00)

Target: t2k <= 0.5 ms (>= 4 M tracks/s), 4L config.

| variant | best tracks/s | t2k [ms] | best batch | max batch | VRAM [GiB] | power mean [W] | physics gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0 | 289,253 | 6.9144 | 22000 | 67584 | 3.00 | 240 | nan |
| v2p | 373,168 | 5.3595 | 22000 | 22000 | 4.97 | 339 | PASS |
| v3 | 178,894 | 11.1798 | 32768 | 262144 | 8.87 | 398 | PASS |
| v3c | 596,203 | 3.3546 | 32768 | 262144 | 6.10 | 341 | PASS |

### KPI table (auto, 2026-07-07T19:47:54+00:00)

Target: t2k <= 0.5 ms (>= 4 M tracks/s), 4L config.

| variant | best tracks/s | t2k [ms] | best batch | max batch | VRAM [GiB] | power mean [W] | physics gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0 | 289,253 | 6.9144 | 22000 | 67584 | 3.00 | 240 | nan |
| v2p | 373,168 | 5.3595 | 22000 | 22000 | 4.97 | 339 | PASS |
| v3 | 178,894 | 11.1798 | 32768 | 262144 | 8.87 | 398 | PASS |
| v3c | 596,203 | 3.3546 | 32768 | 262144 | 6.10 | 341 | PASS |
| v4 | 611,470 | 3.2708 | 16384 | 524288 | 2.45 | 235 | PASS |

### KPI table (auto, 2026-07-07T19:59:03+00:00)

Target: t2k <= 0.5 ms (>= 4 M tracks/s), 4L config.

| variant | best tracks/s | t2k [ms] | best batch | max batch | VRAM [GiB] | power mean [W] | physics gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0 | 289,253 | 6.9144 | 22000 | 67584 | 3.00 | 240 | nan |
| v2p | 373,168 | 5.3595 | 22000 | 22000 | 4.97 | 339 | PASS |
| v3 | 178,894 | 11.1798 | 32768 | 262144 | 8.87 | 398 | PASS |
| v3c | 596,203 | 3.3546 | 32768 | 262144 | 6.10 | 341 | PASS |
| v4 | 611,470 | 3.2708 | 16384 | 524288 | 2.45 | 235 | PASS |
| v5 | 741,765 | 2.6963 | 32768 | 524288 | 4.86 | n/a | PASS |

### Secondary track (GPU 1) — results (2026-07-08, night 2)

All rows in `docs/perf/results/night2/results.jsonl`; job logs under
`docs/perf/results/night2/jobs/`. Everything ran on **GPU 1 only**
(`CUDA_VISIBLE_DEVICES=1`), production precision flags pinned + recorded.

**1. Training-step micro-bench (fwd+bwd, packed path, synthetic production
length distribution; `scripts/bench_packed_vs_padded.py --variant`, 4L bench
shape dim128/N16; medians; `step` includes CPU collate + H2D; no optimizer).**
`--variant v3c` **backward through the compiled static core WORKS** — no
inference-only assumption was hit, so v3c is trainable today as expected:

| batch | variant | collate [ms] | fwd [ms] | bwd [ms] | step [ms] | samp/s | peak VRAM [GiB] | step speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2048 | v0 | 6.89 | 12.78 | 39.53 | 59.18 | 34,609 | 1.62 | ref |
| 2048 | v3 | 6.84 | 12.31 | 35.32 | 54.32 | 37,704 | 5.51 | 1.09× |
| 2048 | v3c | 7.04 | 5.02 | 16.82 | 28.88 | 70,906 | 3.93 | **2.05×** |
| 22000 | v0 | 75.90 | 117.18 | 352.98 | 546.32 | 40,269 | 16.53 | ref |
| 22000 | v3 | 74.13 | 114.24 | 246.14 | 434.53 | 50,630 | 58.05 | 1.26× |
| 22000 | v3c | 75.27 | 38.92 | 108.52 | 222.99 | 98,661 | 41.42 | **2.45×** |

GPU-only (fwd+bwd, collate excluded) at 22000: v0 470.2 ms → v3c 147.4 ms =
**3.19×**. Trade-off: the padded-static path holds the autograd graph on
(B, 22, D) static rows → **2.5× higher training VRAM than stock packed**
(41.4 vs 16.5 GiB at 22K); the packed collate itself (~76 ms CPU at 22K)
becomes the next training bottleneck once v3c is in.

**2. State-ejection encoder (user request; timing only, NO physics gate —
no trained ckpt, `--random-weights`).** Exact config:
`src/track_regression/config/ssm_state/pretrain_ssm_state.yaml` (+ sibling
base.yaml merge) = `mamba_state.BidirectionalMambaEncoder`, **9L dim128
d_state32 headdim32 chunk16, pool=ssm_state**, fp32. Its configured
`preprocessed_dir` (p0_core_pretrain) does not exist on this node → benched
with `--set data.preprocessed_dir=.../p200_core_kf_matched_finetune`. Its
config never sets `packed_batches` → **native PADDED mode** (this encoder has
no packed path) — benched as configured, arch unchanged.
- staged 22000: **191.44 ms/iter → 114,920 tracks/s, t2k 17.40 ms, 6.69 GiB**.
- sweep: throughput flat ~114.6K tracks/s from 2048 up; **padded launch
  ceiling: 31,744 OK → 32,768 FAIL** (`RuntimeError: Triton Error [CUDA]:
  invalid argument`) — 32,768 × nchunks(=2) = 65,536 crosses the 65,535
  grid-axis limit, consistent with the night-1 packed-ceiling diagnosis.
- `--variant v3c --random-weights` on it fails as predicted, exact error:
  `NotImplementedError: encoder has no padded-static path (mamba_cls.py hook
  missing)` (bench exit 3; log `jobs/ssmstate_v3c_attempt.log`). Porting the
  quadratic-dual variants to this encoder = retrain-track work, not tonight's.

**3. v5 e2e at scale (BS 131,072, 8 workers, 30 batches)**: e2e
**744,255 tracks/s (t2k 2.687 ms)** vs staged same-batch 728,189 (2.747 ms)
→ e2e is +2.2% FASTER than staged: dataloader + H2D are fully overlapped at
scale (pinned memory + 8 workers); the loader is still not a bottleneck even
at 1.7 M tokens/batch. VRAM 19.3 GiB both.

**4. v5 power at optimal batch (staged 32768, 500 iters sampled)**:
720,010 tracks/s, **power mean 355.7 W / max 400.6 W — at the 400 W cap,
SM clocks 1655 vs 1785 MHz boost (power-throttled), SM util 86.4%**.
Perf/W: 2,024 tracks/s/W vs v0's 1,205 (240 W) = 1.68×. Note v5 is
cap-limited on this H100 NVL; on the 400 W cap the earlier v4 point
(611 K @ 235 W = 2,602 tracks/s/W) remains the perf/W sweet spot.

Night-2 KPI-table caveat: rows tagged `*+train` are fwd+bwd training-step
points (samples/s, not inference tracks/s), and the night-2 `v0` row is the
**ssm_state 9L encoder** (secondary-track bench), not the 4L ssm_cls v0 —
see job_id/tag in results.jsonl.

Files changed (secondary track, all additive/flag-gated):
`scripts/bench_packed_vs_padded.py` (`--variant`, `--out-jsonl`, `--job-id`;
stale `sort_field` kwarg dropped to run on current API),
`scripts/perf/bench_variant.py` (`--random-weights`, `--ckpt` now optional
with it).

### KPI table (auto, 2026-07-08T15:26:55+00:00)

Target: t2k <= 0.5 ms (>= 4 M tracks/s), 4L config.

| variant | best tracks/s | t2k [ms] | best batch | max batch | VRAM [GiB] | power mean [W] | physics gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0 | 114,920 | 17.4034 | 22000 | 31744 | 6.68 | n/a | nan |
| v0+train | 40,269 | 49.6659 | 22000 | 22000 | 16.53 | n/a | — |
| v3+train | 50,630 | 39.5023 | 22000 | 22000 | 58.05 | n/a | — |
| v3c+train | 98,661 | 20.2715 | 22000 | 22000 | 41.42 | n/a | — |
| v5 | 744,255 | 2.6873 | 131072 | 131072 | 19.32 | n/a | PASS |

### KPI table (auto, 2026-07-08T15:48:40+00:00)

Target: t2k <= 0.5 ms (>= 4 M tracks/s), 4L config.

| variant | best tracks/s | t2k [ms] | best batch | max batch | VRAM [GiB] | power mean [W] | physics gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0 | 114,920 | 17.4034 | 22000 | 31744 | 6.68 | n/a | nan |
| v0+train | 40,269 | 49.6659 | 22000 | 22000 | 16.53 | n/a | — |
| v3+train | 50,630 | 39.5023 | 22000 | 22000 | 58.05 | n/a | — |
| v3c+train | 98,661 | 20.2715 | 22000 | 22000 | 41.42 | n/a | — |
| v5 | 744,255 | 2.6873 | 131072 | 131072 | 19.32 | n/a | PASS |
| v5pc | 907,158 | 2.2047 | 32768 | 606208 | 3.44 | n/a | PASS |

### KPI table (auto, 2026-07-08T16:48:04+00:00)

Target: t2k <= 0.5 ms (>= 4 M tracks/s), 4L config.

| variant | best tracks/s | t2k [ms] | best batch | max batch | VRAM [GiB] | power mean [W] | physics gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v0 | 114,920 | 17.4034 | 22000 | 31744 | 6.68 | n/a | nan |
| v0+train | 40,269 | 49.6659 | 22000 | 22000 | 16.53 | n/a | — |
| v3+train | 50,630 | 39.5023 | 22000 | 22000 | 58.05 | n/a | — |
| v3c | 275,317 | 7.2644 | 32768 | 262144 | 7.20 | n/a | PASS |
| v3c+train | 98,661 | 20.2715 | 22000 | 22000 | 41.42 | n/a | — |
| v5 | 744,255 | 2.6873 | 131072 | 131072 | 19.32 | n/a | PASS |
| v5pc | 907,158 | 2.2047 | 32768 | 606208 | 3.44 | n/a | PASS |

---

## Night 3 — 2026-07-09 (user go: training benches, trainable kernel, pretrain launches)

### Research check — remaining speed avenues WITHOUT architecture changes

- **cuBLAS FP32 emulation on tensor cores (3xTF32 / BF16x9)** — the one
  substantive new avenue found: fp32-equivalent accuracy at tensor-core
  rates, targeting exactly our GEMM share (NVIDIA blog "Unlocking Tensor
  Core Performance with Floating Point Emulation in cuBLAS"; CUTLASS ex. 27;
  arXiv:2203.03341 reports fp32-accuracy above the fp32 peak). **Blocked on
  our stack**: needs the cuBLAS ≥12.9 API; torch 2.9.1+cu128 ships cuBLAS
  12.8 (verified — no emulation attrs in libcublasLt.so.12). Actionable
  later via a torch-cu13 side env; physics-gated like everything else.
- **Dual-stream overlap of the two scan directions** (independent until the
  merge): plausible 10-20%, conflicts with torch.compile's single-stream
  assumption — medium effort, candidate if the plateau must move.
- CUDA graphs: launch overhead negligible at optimal batch (measured night
  1); not a lever. Everything else (in-kernel GEMM, TF32 dots, shift-conv,
  occupancy splits) already measured dead nights 2-3.

### Training benchmarks (REAL train.py runs, finetune dataset, >=10 min
stabilization each, steady-state window rates)

| stage | setup | v0 | v3c | speedup |
|---|---|---:|---:|---:|
| pretrain (batch 2048, 1xH100) | steady window it/s | 12.1 | 19.1 | **1.56x** |
| finetune 2xH100 DDP | v0 @22000/rank vs v3c @40960/rank (max, 85 GiB/GPU) | 85.4K samples/s (1.94 it/s) | 127.8K samples/s (1.56 it/s) | **1.50x** |

- Pretrain caveat: the isolated GPU-only micro-bench gives 2.05x at BS 2048;
  the real-pipeline delta is dataloader/logging overhead the kernel cannot
  remove at small batch. Fine-tune: v3c also fits 1.86x the config's global
  batch (81,920 vs 44,000).

### Trainable fused kernel (v5p/v5pc) — built, verified, and NOT chosen for training

- Backwards registered for `ssd_short_fwd_packed` + `gated_rmsnorm` via
  exact torch recompute (`register_autograd`): op-level gradients match
  autograd at 0.0 rel; encoder-level O11 at ~4e-7 across input/CLS/params.
- Debug lesson recorded: the first O11 used loss = cls^2.mean() over
  RMS-normalised outputs — nearly invariant, true grads ~1e-18, so ALL
  paths (incl. stock) disagreed at 20-30% on pure cancellation noise; with
  a random-projection loss everything agrees. (Also means stock-vs-anything
  gradient comparisons at that loss were meaningless.)
- v5pc under torch.compile(dynamic=True) + autograd hits a device-side
  assert inside Inductor (repro: scratchpad trainrepro.py; eager v5p is
  clean) — filed in the log as the blocker for compiled-packed training.
- Training-step timing (GPU-only, BS 2048): **v3c 52.0 it/s vs v5p 15.4
  it/s** — the recompute backward + eager glue make v5p 3.4x slower to
  train. **Verdict: v3c = best trainable variant (used for the launches);
  v5pc = best inference variant (unchanged).**

### Pretraining launches (USER-APPROVED tonight, explicit instruction)

- Dataset: /eos/.../p0_core_kf_hits_pretrain copied to /scratch (43 GB,
  1000 shards, 56.5M tracks, split.json present, max_hits 20).
- Two runs, one GPU each, batch 2048 per config (load-bearing), fp32,
  KernelSwapCallback(v3c) applied on the CLI (base callback list intact):
  - `pretrain_4L_shortkernel` (GPU 0) — config
    `experimental/scaling/pretrain_ssm_cls_4L_shortkernel.yaml`,
    comet name TRK-SSMCLS-pretrain-4L-shortkernel;
  - `pretrain_10L_shortkernel` (GPU 1) — config
    `ssm_cls/pretrain_ssm_cls_packed_shortkernel.yaml` (standard 10L
    dim192/N32), comet name TRK-SSMCLS-pretrain-10L-shortkernel.
- tmux sessions pretrain_4L / pretrain_10L; logs under
  docs/perf/results/night3/pretrain_logs/; krenew active; everything on
  /shared or /scratch per OVERNIGHT_CHECKLIST.

### Precision audit + training-loop fix (2026-07-09, user-requested)

**Finding: the first pretraining launches were computing TF32 gradients.**
`train.py` hardcoded `float32_matmul_precision("high")` → every nn.Linear
GEMM AND the v3c scan matmuls ran on TF32 tensor cores (10 mantissa bits,
~3 decimal digits). fp64-referee on the real training path (trained 4L
weights, real quantile loss, gradients):

| fp32 mode | pred max err vs fp64 | grad rel err median / max |
|---|---:|---:|
| "highest" (IEEE) | 3.3e-3 | 0.3% / 3.4% |
| **"high" (TF32) — as launched** | **2.84** | **53% / 236%** |

(The oracle-chain tests always pinned "highest"; the training script did
not — exactly the class of silent leak the user suspected. Historical
checkpoints were trained under "high", so the *default* stays "high" for
fine-tune compatibility; new trainings must set `TRK_MATMUL_PRECISION=highest`
— train.py now reads it and prints the active mode.)

**Loader exoneration + real per-step overhead:** the streaming loader
delivers 144,744 tracks/s standalone at 8 workers (1.4× the GPU's need) —
NOT the bottleneck. The 19.6-vs-52 it/s gap was per-step diagnostics:
`_shared_step` computed quantile crossing+calibration metrics AND
predict_physical+_log_metrics on EVERY training step. New
`train_metrics_every_n_steps` init arg (default 1 = historical; launches
use 50; validation unchanged; loss/gradients byte-identical).

**Net effect (measured, 9-min stabilized, batch 2048):** 19.6 it/s
(TF32 + per-step metrics) → **21.6 it/s (full IEEE fp32 + metrics@50)** —
strictly better on both axes: +10% speed AND ~165× cleaner gradients.
**Both pretrainings killed and relaunched** (~4 h lost, repaid same-day)
as `pretrain_4L`/`pretrain_10L` v2 with TRK_MATMUL_PRECISION=highest +
metrics@50; logs `*_shortkernel_v2.log`.

### Day plan 2026-07-09 (user sign-off instructions, in progress)

1. Profile the training loop again (GPU downtime visible at 24.8 it/s vs
   ~theoretical); pause 4L for profiling, 10L keeps running.
2. Remove the quantile crossing metrics from training logging entirely;
   reduce train logging cadence ~10× (log_every_n_steps 50→500,
   train_metrics_every_n_steps→500 in pretrain configs).
3. New variant "auto": v3c path when model.training, v5pc path in
   eval/val/test/predict — make this the CONFIG DEFAULT (KernelSwapCallback
   in the base configs). v5pc = default for validation/testing, v3c = default
   for training, per user.
4. Clean up R&D code: remove measured-dead variants (v6/kernel3, v4t,
   zproj epilogue) from ops/mamba_short/tests/bench VARIANTS.
5. Update README (how to run the new kernels, config notes).
6. ONE clean commit on opt_kernel.
7. Relaunch both pretrainings with all improvements (v2 → v3 launches).

### Day-plan completion (2026-07-09 afternoon)

- **Training-loop round 2 (profiled):** fused Triton Lion (was 5 ms/step =
  24% of GPU time in tiny per-param kernels; `lrs_config.use_triton`,
  default on), quantile CROSSING metrics now val/test-only, all training
  diagnostics gated at 500 steps, trainer/GradientLogger/progress cadences
  50→500. Loader re-confirmed not the bottleneck.
- **"auto" kernel variant is now the config default** (base callbacks):
  v3c (compiled pure-torch, exact autograd) while training; v5pc (fused
  packed Triton) in val/test/predict. Verified: eval == v5p at 6e-7,
  training grads == v3 at 3e-7.
- **Bug found via py-spy on the live 10L run: a leaf-level
  `trainer.callbacks:` list REPLACES the base list** — the production 10L
  config silently dropped the KernelSwapCallback and trained on the STOCK
  kernel. Fixed (callback repeated in the leaf; footgun documented).
- **R&D cleanup:** v6/kernel3, zproj epilogue and the v4t TF32 experiment
  removed from code (history + verdicts preserved in this log);
  suite 52/52 green. README gained a "Fast short-sequence kernels" section.
- **ONE clean commit on opt_kernel: 1347e37** (74 files; bulky binaries
  gitignored).
- **Pretrainings relaunched (v4, final):** 4L ≈ 33 it/s (was 19.6 at
  launch-1 → 1.7×), 10L ≈ 12.8 it/s steady (was 8.1 best, and 5.3 while
  silently on the stock kernel → 1.6×), both full IEEE fp32
  (TRK_MATMUL_PRECISION=highest), auto kernel confirmed in both logs.
  Epochs ≈ 12.5 min (4L) / ≈ 32 min (10L).
