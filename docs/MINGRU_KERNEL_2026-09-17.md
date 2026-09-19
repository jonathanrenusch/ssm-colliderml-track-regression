# minGRU — a linear-recurrent GRU and its fused short-sequence kernel

2026-09-17. Follows the conventions of the Mamba-2 kernel campaign
(`docs/perf/OPTIMIZATION_LOG.md`, branch `opt_kernel`): measured numbers
outrank estimates, every variant carries a parity gate, strict IEEE fp32 is
the default and TF32 is a reported deployment setting.

## 1. Why

The 2026-09 architecture ablation found a bidirectional **GRU** matching the
paper's bidirectional Mamba-2 on physics (geometry identical to <0.005 GM5;
q/p 4 % better at 2 GeV and on hadrons).  What a classical GRU cannot match is
the kernel: its reset gate makes `h_t` nonlinear in `h_{t-1}`, so each of the
L timesteps is a dependent H x H GEMM and the sequential chain survives every
optimisation.

**minGRU** (Feng et al. 2024) drops the recurrent-state dependence inside the
gates:

    z_t = sigmoid(W_z x_t),  n_t = W_n x_t          (input only)
    h_t = (1 - z_t) * h_{t-1} + z_t * n_t
        = a_t * h_{t-1} + b_t                        with a_t in (0,1)

so the recurrence becomes **linear and elementwise**.  Two consequences:

* `a_t, b_t` for the whole packed stream come from ONE GEMM — no sequential
  dependency in the expensive part;
* the scan itself is elementwise, so a fused kernel keeps `h` in registers and
  the L steps are L fused multiply-adds per channel — **no `tl.dot`, no L x L
  decay matrix, no Gram matrix**, unlike the SSD kernel.

Measured FLOP split per track (H = 194, 20 slots):

| stage | MFLOP/track | share |
|---|---:|---:|
| layer-1 in-projection (388 -> 776) | 12.0 | 63 % |
| layer-0 in-projection (128 -> 776) | 4.0 | 21 % |
| input net (Fourier 480 -> 128) | 3.1 | 16 % |
| **the recurrence itself** | **0.047** | **0.25 %** |

The scan is free; everything is projections.  That is what makes the kernel
problem tractable — and it is why the two wins below are the ones that matter.

## 2. Kernel design

`src/track_regression/ops/mingru_short_triton.py`, two variants:

* `mingru_bidi_fused` — padded-static `(B, S, 4H)`, the analogue of the SSD
  `v3c/v5pc` padded path;
* `mingru_bidi_packed` — **packed stream `(T, 4H)` with `cu_seqlens`, no pad
  rows anywhere**, the analogue of `v5p`.  This is the deployed path.

Conventions inherited from the SSD kernel:

* **in-kernel REVERSE**: the backward direction reads the segment from its end
  and stores at the physical row, so the output needs no un-flip and the
  caller never gathers;
* **direction by channel offset** into one shared in-projection output rather
  than a Python slice — night 1 measured that Inductor materialises a copy for
  every sliced opaque-op input (the fused dual-direction in_proj regressed
  0.60 -> 0.50 M there);
* `@torch.library.custom_op` so the op stays opaque to Inductor;
* **strict IEEE fp32 inside the kernel** — night 2 measured TF32 `tl.dot` in
  the scan as both slower and noisier.  (Here there is no dot at all.)
* grid `(segment, channel-block, direction)`; `BD` autotuned 64–256.

Because every packed row is written exactly once per direction, the output is
allocated with `empty` — no zero-fill pass.

Training uses a **Hillis-Steele prefix scan** instead (5 rounds at L = 20
rather than 20 sequential steps): affine maps compose, so after k rounds the
pair at position t is the map from `h_{t-2^k}` to `h_t`.  Same result, a 4x
shorter autograd graph.

## 3. Measured throughput — one H100 NVL, uncontended

`scripts/bench_infer_flat.py`, ttbar_new_pt1 store, GPU seed (fp64) inside the
timed loop, 100 iters, batches preloaded to pinned RAM.

| model / path | 2 k | 8 k | 32 k | 131 k | VRAM @131k |
|---|---:|---:|---:|---:|---:|
| minGRU packed, strict fp32 | 0.56 M | 1.12 M | 1.43 M | 1.45 M | — |
| minGRU packed, TF32 | 0.60 M | 1.75 M | 2.55 M | 2.84 M | — |
| **minGRU packed, deployed (TF32 + compiled front-end)** | 0.64 M | 1.95 M | **3.44 M** | **3.96 M** | 11.1 GiB |
| minGRU *padded* path, strict fp32 (ablation) | 0.47 M | 0.85 M | 0.98 M | 0.98 M | — |
| classical GRU, cuDNN fp32 | 0.42 M | 0.90 M | 1.43 M | 1.43 M | — |
| classical GRU, cuDNN TF32 | 0.43 M | 0.92 M | 1.51 M | 1.61 M | — |
| SSM (paper model) fused strict fp32 | 0.41 M | 0.75 M | 0.93 M | 0.93 M | — |
| SSM (paper model) full deployment path | 0.43 M | 1.15 M | 1.64 M | 1.76 M | — |
| transformer SDPA fp32 / TF32 | — | — | 0.49 / 0.66 M | 0.48 / 0.64 M | — |
| forward-only Mamba-2, fp32 | — | — | 0.71 M | 0.70 M | — |

**minGRU deployed = 2.25x the SSM's deployed path on the same node and script**
(3.96 vs 1.76 M), and 2.07x the paper's quoted 1.91 M.  It also clears the
original kernel campaign's stretch target of 4 M tracks/s (t2k <= 0.5 ms).

Where the speed comes from, in order:

1. **packed vs padded: 1.48x** (0.98 -> 1.45 M, strict fp32).  Tracks average
   13.3 hits padded to 20, so a third of the projection FLOPs were pure waste.
   Predicted 1.50x from the FLOP count above — measured 1.48x.
2. **TF32 on the projections: 1.96x** (1.45 -> 2.84 M).  Compare the cuDNN GRU,
   where TF32 buys only 1.13x: that model is launch-bound, minGRU is
   FLOP-bound, so the tensor cores actually pay.
3. **compiled front-end: 1.39x** (2.84 -> 3.96 M) — the normalise -> Fourier ->
   input-net stack, same switch the SSM deployment uses.

Cross-device projection: scaling by the measured H100:Ada ratio for the SSM
(1.91 M : 0.504 M = 3.79x) puts minGRU at **~1.04 M tracks/s on an RTX 5000
Ada** — to be confirmed by direct measurement, not claimed.

### 3a. CUDA graphs — the small-batch regime

Measured on GPU 0 **under training contention, both sides equally** (the
methodology §4.27 used for the SSM graph numbers); the ratio is the
measurement, the absolute values are depressed:

| batch | eager | CUDA graph | gain |
|---|---:|---:|---:|
| 2 048 | 378.6 k | **909.6 k** | **+140 %** |
| 4 096 | 759.1 k | **1.468 M** | **+93 %** |

`graph-vs-eager max |dpred| = 0.000e+00` at both sizes — the dummy-track
padding is provably inert, exactly as for the SSM.

Note a structural advantage over the SSM here: its CUDA-graph path is
**incompatible with `TRK_SSD_BUCKET16`** (a device-sync split), so it has to
choose between the two.  The packed minGRU kernel has no bucketing and no
device sync, so graphs and the packed path compose — the small-batch regime
gets both wins at once.  This matters for online/HLT-style use where batches
are small.

### 3b. Portability to the RTX 5000 Ada (sm_89)

The deployment target is not only Hopper.  Verified by **ahead-of-time
compilation for sm_89 on this machine** (no Ada device required), for every
autotune config of both kernels:

* PTX emitted with `.target sm_89`;
* **0 bytes of shared memory** (state is in registers) vs Ada's 101,376 B/SM;
* no Hopper-only PTX (`wgmma`, `tma`, `cp.async.bulk`, `mbarrier`,
  `setmaxnreg`, cluster launch);
* `num_warps` <= 8, 3-D grid — all within sm_89 limits.

Locked in as `tests/test_mingru.py::test_kernels_compile_for_ada_sm89`, so a
Hopper-only construct cannot slip in unnoticed.  The kernel uses no `tl.dot`
at all, which is also why it needs no tensor cores to be correct — Triton
re-autotunes `BD`/`num_warps` per architecture on first launch.

VRAM headroom: 11.1 GiB at a 131 k batch, so the Ada's 32 GB holds the
throughput-optimal batch comfortably (the SSM's 262 k run died on that card,
§4.32).

## 4. Correctness gates (`tests/test_mingru.py`, 13 tests)

* scan vs the explicit recurrence, and the parallel scan vs the sequential one
  (1e-11, float64);
* **fused kernel vs the torch reference on a realistic batch: < 1e-5 relative**
  (97 tracks, random lengths 6–20, both directions);
* **packed path vs padded eager path end-to-end: < 1e-5 relative**;
* packed <-> padded equivalence, segment/permutation independence, order
  sensitivity, and pad-safety (a track's output cannot depend on the longest
  track in its batch);
* prefix-flip index is self-inverse.

## 5. Cost

Training throughput is the one place minGRU loses: **20.4 steps/s vs the
classical GRU's 30.0** at batch 2048 (the parallel scan materialises more
intermediate traffic than cuDNN's fused GRU).  That is training wall-clock
only; it does not touch deployment.

## 6. Not done

Direct Ada measurement; uncontended CUDA-graph numbers (only the contended
ratio is measured so far); bf16/fp16 (the standing rule is nothing below TF32 until parity is
established); per-layer fusion of the two in-projections.
