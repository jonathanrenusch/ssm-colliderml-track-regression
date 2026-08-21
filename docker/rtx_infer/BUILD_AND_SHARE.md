# Build, test, and share the RTX inference container

A step-by-step runbook for building the image on a Docker-capable machine,
testing it yourself, and handing a self-contained bundle to a colleague — one
for the **4-layer** network and one for the **10-layer** network.

> Nothing here was run inside the CERN pod (it has no Docker). The **code and
> pinned dependency set are validated** (they produce the measured numbers via
> pixi), but the **Docker build itself is untested end-to-end** — see
> *Confidence* at the bottom.

---

## 0. Prerequisites (build machine)

- Docker (Engine 20.10+).
- **NVIDIA Container Toolkit** so `--gpus all` works. Quick check:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
  ```
  If that prints your GPU, you're good. (Building does not need a GPU; *running* does.)
- ~15 GB free disk and ~8 GB RAM (the CUDA extensions compile with `ninja`).

## 1. Get the code

```bash
git clone git@github.com:jonathanrenusch/ssm-colliderml-track-regression.git
cd ssm-colliderml-track-regression
git checkout rtx-infer-docker      # branch that carries the kernels + this container
```

## 2. Build the image

From the **repo root** (the build context needs `src/` and `docker/`):

```bash
docker build -f docker/rtx_infer/Dockerfile -t ssm-rtx-infer .
```

First build is ~10–20 min (it compiles `mamba-ssm` + `causal-conv1d` from source
for sm_89/9.0). Rebuilds are cached.

## 3. Test it yourself

You need two things the image does **not** contain: a **checkpoint** and a
**data directory** (compact format: `manifest.json` + `split.json` +
`shard_XXXX/`).

```bash
# 4-layer:
./docker/rtx_infer/run_4L.sh  /path/to/epoch=049-val_total=0.00125.ckpt  /path/to/data_dir  16384
# 10-layer:
./docker/rtx_infer/run_10L.sh /path/to/epoch=048-val_total=0.00110.ckpt /path/to/data_dir  16384
```

Expect the `INFERENCE BENCHMARK RESULT` table (raw GPU compute, H2D excluded,
with a measured H2D reference line). If you hit an OOM, lower the batch (last
arg). `--ipc=host` is already set in the scripts (required — DataLoader shm).

To sanity-check numerics vs speed, add `-e MATMUL_PRECISION=high` for the
TF32-linears mode (see the precision note in `README.md`).

## 4. Share with a colleague

Two options.

### Option A — they build it (they have git + Docker)
Send them: the repo (or just point them at the `rtx-infer-docker` branch) + the
checkpoint + a data dir. They run steps 1–3. Simplest if they're set up for it.

### Option B — prebuilt image bundle (recommended; no build on their side)
Export the built image to a tarball and hand it over with the checkpoint and a
run script. Same image works for both networks — only the checkpoint + script
differ.

```bash
# once: export the image (~6–8 GB compressed)
docker save ssm-rtx-infer | gzip > ssm-rtx-infer.tar.gz

# assemble a 4-layer bundle
mkdir -p bundle-4L
cp ssm-rtx-infer.tar.gz                                   bundle-4L/
cp /path/to/epoch=049-val_total=0.00125.ckpt             bundle-4L/model_4L.ckpt
cp docker/rtx_infer/run_4L.sh                             bundle-4L/
printf 'docker load < ssm-rtx-infer.tar.gz\n./run_4L.sh model_4L.ckpt <YOUR_DATA_DIR> 16384\n' > bundle-4L/HOWTO.txt
tar czf ssm-infer-4L.tar.gz bundle-4L

# assemble a 10-layer bundle
mkdir -p bundle-10L
cp ssm-rtx-infer.tar.gz                                   bundle-10L/
cp /path/to/epoch=048-val_total=0.00110.ckpt            bundle-10L/model_10L.ckpt
cp docker/rtx_infer/run_10L.sh                            bundle-10L/
printf 'docker load < ssm-rtx-infer.tar.gz\n./run_10L.sh model_10L.ckpt <YOUR_DATA_DIR> 16384\n' > bundle-10L/HOWTO.txt
tar czf ssm-infer-10L.tar.gz bundle-10L
```

Your colleague then:
```bash
tar xzf ssm-infer-4L.tar.gz && cd bundle-4L
docker load < ssm-rtx-infer.tar.gz
./run_4L.sh model_4L.ckpt /their/data_dir 16384
```

**Data:** if they don't already have a compact-format dataset, include a small
subset in the bundle — the benchmark only preloads ~16 batches, so ~10–20
`shard_XXXX/` dirs plus `manifest.json` and `split.json` are plenty. (Or, if the
`.ckpt` + image are the only sensitive parts, tell them which `/scratch` path to
point at.)

## 5. Confidence & troubleshooting

**Confidence the build succeeds first try: ~80%.** The dependency set is pinned
to the exact versions validated on the pod, and python is pinned to 3.12.0 to
match — so version drift is unlikely. It is, however, **not built/tested here**
(no Docker in the pod). Most likely failure points, with fixes:

| symptom | cause | fix |
|---|---|---|
| `mamba-ssm`/`causal-conv1d` compile error or hang | not enough RAM, or too many parallel nvcc jobs | give Docker ≥8 GB RAM; add `ENV MAX_JOBS=4` before those `pip install` lines |
| `CUDA arch` / `unsupported gpu` at build | odd toolkit | archs are pinned to `8.9 9.0`; widen `TORCH_CUDA_ARCH_LIST` if needed |
| pip can't find a pinned version | index hiccup | retry; versions are all on PyPI / the pytorch cu128 index |
| `--gpus all` unknown flag at run | NVIDIA Container Toolkit missing | install `nvidia-container-toolkit`, restart Docker |
| `bus error` / shm crash at run | default 64 MB `/dev/shm` | keep `--ipc=host` (the run scripts set it) or add `--shm-size=2g` |
| OOM at run | batch too large for the card | lower the batch arg |

If the source compile is the sticking point, the fallback is the pixi
environment used on the pod (fully reproducible via `pixi.lock`) — ask and I'll
add a pixi-based Dockerfile variant.

## 6. What the number means (and alignment with the legacy tool)

The headline is **raw GPU forward compute**: batches are staged onto the GPU
once, then `model(...)` is timed on the resident data — disk, collate, and the
host→device copy are **outside** the timer (the data pipeline is a separate
deployment concern; the H2D cost is measured and printed for reference, ~1%).

This is the **same quantity** the repo's legacy `scripts/perf/bench_variant.py
--mode staged` measures (pure forward on a resident batch, CUDA-event timed,
after warm-up). They were cross-checked on the L40S and agree to **<0.2%**
(4L: 348.7k vs 348.1k tracks/s; 10L: 93.3k vs 93.1k). The only differences are
knobs, not definition: `bench_variant` reuses one batch and pins TF32 (`high`);
`bench_infer` cycles several preloaded batches and lets you pick the precision
(`--matmul-precision`). Match them by running `bench_infer` with
`MATMUL_PRECISION=high`.
