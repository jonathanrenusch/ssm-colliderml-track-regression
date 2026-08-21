# RTX inference container — overview & reading guide

**Start here.** This directory is a self-contained way to benchmark (and profile)
the short-sequence SSM track-fitter's *inference* on a workstation GPU
(RTX 5000 Ada / L40S / any card), using the best inference kernel (`v5pc`, fused
portable Triton) in full IEEE fp32, with the data preloaded into RAM so the
timed number is **raw GPU compute**.

- To **use** it → `README.md`
- To **build & ship** it to a colleague → `BUILD_AND_SHARE.md`
- To **understand** it → this file + the reading guide below.

---

## What it does, in one paragraph

Builds the model from a config's `model:` node, loads a checkpoint, swaps the
encoder onto the fused Triton inference kernel (`v5pc`), preloads a few
pre-collated packed batches into pinned CPU RAM, stages them onto the GPU, and
times `model(...)` on the resident data. Disk, collate, and the host→device copy
are **outside** the timer (the H2D cost is measured and printed as a ~1%
reference). Two model presets are bundled (4L / 10L); it's otherwise generic.

## Reading guide — which files, in what order

| # | file | what to learn from it |
|---|------|-----------------------|
| 1 | `README.md` | how to run it; env knobs; batch sizing; precision; Nsight |
| 2 | `bench_infer.py` | **the core** — model build, kernel swap, RAM preload, the timed loop (compute-only), H2D measurement, and the printed report. Self-contained; only depends on torch + `track_regression`. |
| 3 | `entrypoint.sh` + `run_4L.sh` / `run_10L.sh` | how env vars (`MODEL`, `CKPT`, `DATA_DIR`, `BATCH_SIZE`, `MATMUL_PRECISION`, …) map to a run |
| 4 | `presets/4L_dim128_state16.yaml`, `presets/10L_dim192_state32.yaml` | the two model architectures (model node only, extracted from the trained run configs) |
| 5 | `Dockerfile` | the environment (pinned to the exact validated versions) |
| 6 | `BUILD_AND_SHARE.md` | build, self-test, and two ways to hand it to a colleague |

To understand the **kernel** the container exercises (one level deeper, in the
main source tree), read in this order:
| file | what it is |
|------|------------|
| `src/track_regression/mamba_short.py` | `Mamba2Short` (pure-torch quadratic dual) + `apply_variant()` (the `v0/v3c/v5pc` swap machinery) |
| `src/track_regression/ops/ssd_short_triton.py` | the fused Triton kernels — conv + selective scan + D-skip + gated-RMSNorm in one launch, IEEE fp32 throughout (`input_precision="ieee"`) |
| `src/track_regression/mamba_cls.py` / `mamba_state.py` | the bidirectional CLS encoder and the hooks `apply_variant` needs |

## Measured results (L40S — Ada sm_89, same architecture as the RTX 5000 Ada)

Raw GPU compute, batch 32768, warm-up 20, CV ~0.15%. Cross-checked against the
legacy `scripts/perf/bench_variant.py --mode staged` to **<0.2%**.

**Our kernel vs the stock Mamba-2 kernel (same card, same batch):**

| model | stock `v0` | ours `v5pc` | speedup |
|-------|-----------:|------------:|--------:|
| 4L  | t2k 16.3 ms | t2k 5.74 ms | **2.84×** |
| 10L | t2k 62.2 ms | t2k 21.5 ms | **2.90×** |

**Full fp32 (`highest`) vs TF32 linears (`high`) — the precision cost:**

| model | full fp32 t2k | TF32 t2k | absolute cost of full fp32 |
|-------|--------------:|---------:|---------------------------:|
| 4L  | 5.75 ms | 5.74 ms | **+0.02 ms / 2k tracks (+0.3%) — free** |
| 10L | 24.70 ms | 21.44 ms | **+3.25 ms / 2k tracks (+15%)** |

H2D copy (measured): ~1.0 ms/batch = ~1% (4L) / ~0.25% (10L) of compute.

## Precision audit — is the *whole* network full fp32?

Under `--matmul-precision highest` (the default), **yes — the entire `v5pc`
inference path is full IEEE fp32**, not just the scan:

| component | op | precision under `highest` |
|-----------|----|---------------------------|
| input pipeline (Fourier + MLP) | `nn.Linear` | fp32 |
| encoder `in_proj` / `out_proj` | `nn.Linear` (GEMM) | fp32 |
| depthwise causal conv (width 4) | inside the Triton kernel | **fp32 (IEEE), always** — not cuDNN |
| SSD selective scan | Triton kernel (`tl.dot` `input_precision="ieee"`) | **fp32, always** |
| gated RMSNorm | Triton kernel | **fp32, always** |
| pool head / output head | `nn.Linear` | fp32 |

Key points:
- The conv, scan, and gated-norm are done **inside the fused kernel in IEEE fp32
  regardless of any flag** — there is no cuDNN convolution in the `v5pc` path, so
  there is **no TF32 leak** there. (cuDNN conv only appears in the *stock* `v0`
  path via `causal_conv1d`, which `v5pc` replaces.)
- The **only** thing the precision flag changes is the `nn.Linear` GEMMs
  (`in_proj`/`out_proj`, pipeline, head): `high` runs them in TF32, `highest`
  runs them in full fp32. So `highest` = uniformly full precision, with no
  mixed-precision bottleneck between the linears and the fp32 scan.
- For certainty, `bench_infer` also sets `cuda.matmul.allow_tf32=False` and
  `cudnn.allow_tf32=False` under `highest`, so even a fallback path can't
  silently use TF32.

**Recommendation:** `highest` is the default and is the right choice for the 4L
(free) and anywhere numerics matter; on the 10L it costs ~15%, so only drop to
`high` there if you need the throughput and have confirmed the physics is
unaffected. Note the published physics numbers were validated under `high` (TF32
linears) and passed the ≤1% gate, so both are physics-safe for inference; `highest`
is simply the stricter, no-mixed-precision option.

## Environment notes
- Data format assumed unchanged ("compact" v2: per-shard `hits.npy` /
  `hit_times.npy` / `selected_tracks/`, plus `manifest.json` + `split.json`).
- Numbers above are the L40S (142 SMs, 48 GB). A full H100 runs the 4L at
  ~2.2 ms t2k; the gap is the card (HBM3 bandwidth + more SMs), not the kernel.
