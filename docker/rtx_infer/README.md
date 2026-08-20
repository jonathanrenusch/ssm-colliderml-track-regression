# RTX inference-profiling container

A self-contained Docker image to profile the short-sequence SSM track-fitter's
**inference** on a workstation GPU (built for **RTX 5000 Ada / sm_89**, but the
Triton kernel JIT-autotunes for whatever card is present). It runs the **best
inference kernel** (`v5pc` — fused, portable Triton) in **strict IEEE fp32**, and
**preloads the data into pinned CPU RAM** so disk + collate are out of the hot
loop and the measurement is compute-bound. It prints throughput / latency / VRAM.

Two models are bundled as presets (best checkpoints):

| preset | architecture | checkpoint (copy it in) |
|--------|--------------|--------------------------|
| `MODEL=4L`  | 4 layers, dim 128, d_state 16 | `epoch=049-val_total=0.00125.ckpt` |
| `MODEL=10L` | 10 layers, dim 192, d_state 32 (paper shape) | `epoch=048-val_total=0.00110.ckpt` |

It's also generic: give `CONFIG=` (any resolved config with a `model:` node) +
`CKPT=` instead of `MODEL=`.

## Build

From the **repo root** (the build context needs `src/` and `docker/`):

```bash
docker build -f docker/rtx_infer/Dockerfile -t ssm-rtx-infer .
```

First build compiles `mamba-ssm` + `causal-conv1d` from source for sm_89 (a few
minutes). No flash-attn / colliderml / datasets are pulled — they aren't needed
for SSM-CLS inference.

## Run

Mount the checkpoints and the data dir, pick a model, set a batch size:

```bash
docker run --gpus all --ipc=host \
  -v /path/to/checkpoints:/ckpts:ro \
  -v /scratch/colliderml:/data:ro \
  -e MODEL=4L \
  -e CKPT=/ckpts/epoch=049-val_total=0.00125.ckpt \
  -e DATA_DIR=/data/arxiv_retraining/p200_core_kf_hits_finetune \
  -e BATCH_SIZE=16384 \
  ssm-rtx-infer
```

`--ipc=host` (or `--shm-size=2g`) is **required** — the DataLoader workers use
shared memory, and Docker's default 64 MB `/dev/shm` will otherwise crash the
preload.

Knobs (all env vars, with defaults): `BATCH_SIZE=8192`, `PRELOAD_BATCHES=16`,
`WARMUP=20`, `ITERS=200`, `VARIANT=v5pc`, `MATMUL_PRECISION=highest`,
`LOADER_WORKERS=8`. Extra CLI args pass straight through, e.g.
`docker run ... ssm-rtx-infer --iters 500`.

### Precision

Default is **`highest` = strict IEEE fp32 everywhere** (honours "fully fp32").
For the faster TF32-in-the-linear-GEMMs mode (still 32-bit storage; this is what
the H100 campaign's headline numbers used), set `-e MATMUL_PRECISION=high`. The
scan kernel itself is always IEEE fp32.

### Batch sizing (32 GB RTX 5000 Ada)

VRAM is small — strict fp32, measured on-device: **4L ≈ 0.1 GiB per 1k tracks**
(3.4 GiB at batch 32 768), **10L ≈ 0.18 GiB per 1k**. So a 32 GB card has lots of
headroom; start around `BATCH_SIZE=16384–32768` and raise it. If you hit an OOM
the script exits with a clear message — just lower `BATCH_SIZE`.

## What it times

The headline throughput is **raw GPU compute only**: the preloaded batches are
staged onto the GPU once, then the timed loop runs `model(...)` on the resident
data. Disk, collate, and the host→device (H2D) copy are **outside** the timer —
the data pipeline is a separate deployment concern. The H2D copy is *measured*
and printed as a reference line (typically ~1% of compute for the 4L, well under
1% for the 10L), so you can see it's negligible without it inflating the number.
This matches the `bench_variant.py --mode staged` methodology used elsewhere in
the repo (cross-checked to <0.2%).

## Example output

```
==============================================================
 INFERENCE BENCHMARK RESULT
==============================================================
  GPU                   : NVIDIA RTX 5000 Ada Generation (sm_89)
  config                : 4L_dim128_state16.yaml
  kernel variant        : v5pc
  matmul precision      : highest (strict fp32)
  batch size            : 16384
--------------------------------------------------------------
  per-batch ms  mean    : ...   (RAW GPU COMPUTE, H2D excluded)
  throughput            : ... tracks/s  (compute only)
  t2k (2000 x per-track): ... ms
  peak VRAM             : ... GiB
  -- reference: H2D copy/batch ... ms (...% of compute)
==============================================================
```

## Profiling with Nsight

The bench is a plain process, so Nsight wraps it directly. Two options:

1. **Host Nsight around the container** (no image change):
   ```bash
   nsys profile -o rtx4L docker run --gpus all --ipc=host ... ssm-rtx-infer --iters 40
   ```
2. **Nsight inside the image**: uncomment the `nsight-systems-cli` line in the
   Dockerfile, rebuild, then:
   ```bash
   docker run --gpus all --ipc=host ... --entrypoint nsys ssm-rtx-infer \
     profile -o /data/rtx4L python /workspace/docker/rtx_infer/bench_infer.py \
     --config docker/rtx_infer/presets/4L_dim128_state16.yaml \
     --ckpt /ckpts/... --data-dir /data/... --iters 40
   ```

What a healthy profile looks like (we checked on our H100 via `nsys stats
--report cuda_gpu_kern_sum`): the fused kernel `_ssd_short_fwd_kernel2p`
dominates (~73% of GPU time), the fp32 projection GEMMs (`..._f32f32_f32_...`)
are ~11%, gated-RMSNorm ~7%, and **no host→device memcpy shows up** in the timed
region (the RAM preload worked). One minor item to keep an eye on: a batch of
small `int` fill kernels (~7%) from buffer setup — not a bottleneck, but a
candidate if you want to squeeze the last few percent.

## Notes

- Assumes the preprocessed data format currently on `/scratch` ("compact"
  format v2: per-shard `hits.npy` / `hit_times.npy` / `selected_tracks/*.npy`
  + `manifest.json`). Point `DATA_DIR` at any dir in that format.
- Checkpoints are **not** baked into the image — mount them (they load
  strict; only a stale `sort_field` arg is dropped, no missing params).

## Debug reference (our numbers — NOT an RTX)

Measured while validating this on an **H100 NVL MIG 3g.47gb slice** (≈3/7 of a
full H100) in **strict fp32** — shown only to prove the routine works and give a
rough shape; your full RTX 5000 Ada will differ:

| model | batch | tracks/s | t2k | peak VRAM |
|-------|------:|---------:|----:|----------:|
| 4L    |  8192 | 270,000  | 7.40 ms | 1.07 GiB |
| 4L    | 32768 | 277,000  | 7.21 ms | 3.40 GiB |
| 10L   |  8192 |  57,000  | 34.85 ms | 1.49 GiB |

(CV < 0.3% across 100 timed batches. On a full card and/or `MATMUL_PRECISION=high`
these rise substantially.)

### L40S reference (Ada sm_89 — same architecture as the RTX 5000 Ada)

Compute-only (H2D excluded), batch 32768, warmup 20, CV ~0.15%. The L40S is a
bigger Ada chip than the RTX 5000 Ada (142 SMs / 48 GB vs ~100 SMs / 32 GB), so
treat these as an upper bound for the RTX; the **fp32-vs-TF32 ratio transfers
directly**:

| model | precision | tracks/s | t2k | H2D (measured) |
|-------|-----------|---------:|----:|---------------:|
| 4L  | fp32 (highest) | 347,800 | 5.75 ms | 0.98 ms (1.0% of compute) |
| 4L  | TF32 (high)    | 348,700 | 5.74 ms | — |
| 10L | fp32 (highest) |  81,000 | 24.70 ms | 0.98 ms (0.24%) |
| 10L | TF32 (high)    |  93,300 | 21.44 ms | — |

Takeaways: strict fp32 is **free on the 4L** (~0.3% vs TF32) and costs **~15% on
the 10L** (bigger projections); H2D is ~1% either way. Cross-checked against
`bench_variant.py --mode staged` (agrees to <0.2%). For context, a full H100 NVL
runs the 4L at ~2.2 ms t2k — the ~2.6× gap is the card (HBM3 bandwidth + more
SMs), not the kernel.
