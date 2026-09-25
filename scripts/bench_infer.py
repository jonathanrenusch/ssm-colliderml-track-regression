#!/usr/bin/env python3
"""Inference throughput benchmark (tracks per second on one GPU).

    python scripts/bench_infer.py --config configs/minGRU_stage1.yaml --ckpt <ckpt> \
        --data-dir data/eval/ttbar_bench --batch-size 131072 --mode deployed

A set of packed batches (12 measured hit features per hit) is preloaded into
pinned host memory; the timed loop copies each batch to the GPU and runs the
full forward pass: the analytic seed and the seed-residual features on the GPU
(float64), the Fourier front end, the encoder, the quantile heads and the
decoding to physical parameters.

``--mode deployed`` (the "optimized" column of the paper): fused kernels,
TF32 GEMMs, encoder under float16 autocast.  ``--mode reference`` (the
"default kernel" column): the exact code path the encoder trains with --
padded layout, its training-path kernels, strict IEEE fp32.

Select the GPU with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

PARAMS = ("d0", "z0", "phi", "theta", "qop")


def preload_batches(data_dir: str, batch_size: int, n_batches: int, workers: int):
    import torch
    from track_regression.data import ColliderMLRegrDataModule

    dm = ColliderMLRegrDataModule(preprocessed_dir=data_dir, batch_size=batch_size,
                                  num_workers=workers, pin_memory=False)
    dm.setup("test")
    batches, n_tracks, n_hits = [], 0, 0
    for inputs, _ in dm.test_dataloader():
        batches.append({k: v.pin_memory() for k, v in inputs.items()})
        n_tracks += int(inputs["track_lengths"].numel())
        n_hits += int(inputs["track_lengths"].sum())
        if len(batches) >= n_batches:
            break
    if not batches:
        raise RuntimeError(f"no batches loaded from {data_dir}")
    print(f"[bench] preloaded {len(batches)} batches, {n_tracks:,} tracks, "
          f"{n_hits / n_tracks:.1f} hits/track", flush=True)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="store root with a test/ split")
    ap.add_argument("--batch-size", type=int, default=131072)
    ap.add_argument("--mode", default="deployed", choices=["deployed", "reference"])
    ap.add_argument("--preload-batches", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--loader-workers", type=int, default=8)
    args = ap.parse_args()

    deployed = args.mode == "deployed"
    os.environ["TRK_REFERENCE_KERNELS"] = "0" if deployed else "1"
    import torch
    from track_regression.inference import load_model

    torch.set_float32_matmul_precision("high" if deployed else "highest")
    if not torch.cuda.is_available():
        sys.exit("[bench] no CUDA device visible")
    model = load_model(args.config, args.ckpt, "cuda", "float16" if deployed else "float32")
    batches = preload_batches(args.data_dir, args.batch_size, args.preload_batches, args.loader_workers)

    def step(b):
        b = {k: v.to("cuda", non_blocking=True) for k, v in b.items()}
        out = model(b)
        anchors = {f"seed_{p}": out["seed"][:, i] for i, p in enumerate(PARAMS)}
        return model.loss_module.predict_physical(out["pred"], anchors)

    tracks = [int(b["track_lengths"].numel()) for b in batches]
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for i in range(args.warmup):              # Triton autotuning, torch.compile
                step(batches[i % len(batches)])
            torch.cuda.synchronize()
            n = 0
            t0 = time.perf_counter()
            for i in range(args.iters):
                step(batches[i % len(batches)])
                n += tracks[i % len(batches)]
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        sys.exit(f"[bench] out of memory at batch size {args.batch_size}")

    print("\n" + "=" * 62)
    print(" INFERENCE BENCHMARK")
    print("=" * 62)
    print(f"  GPU                   : {torch.cuda.get_device_name()}")
    print(f"  encoder               : {type(model.encoder).__name__}")
    print(f"  mode                  : {args.mode}")
    print(f"  batch size            : {args.batch_size}")
    print(f"  per-batch ms          : {1e3 * wall / args.iters:.3f}")
    print(f"  throughput            : {n / wall:,.0f} tracks/s")
    print(f"  peak VRAM             : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("=" * 62, flush=True)
    os._exit(0)                                    # do not wait on loader threads


if __name__ == "__main__":
    main()
