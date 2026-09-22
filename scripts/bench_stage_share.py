#!/usr/bin/env python3
"""Stage-wise share of the deployed forward pass: seed | front end | encoder | heads.

Reports CUDA kernel count and GPU kernel time per stage (torch.profiler, real
tracks), the method behind the paper's "seed = 9 % of the forward" number.
Runs the same deployment path as ``bench_infer_flat.py --gpu-seed``:
on-GPU float64 seed, min-max -> Fourier -> input net (compiled when
TRK_COMPILE_FRONTEND=1), the encoder under the chosen autocast dtype, heads
and quantile decoding in fp32.

    python scripts/bench_stage_share.py --config <cfg> --ckpt <ckpt> \
        --data-dir <store> --batch-size 65536 --encoder-dtype float16 \
        --matmul-precision high

Select the GPU with CUDA_VISIBLE_DEVICES (Triton launches on the current device).
The script exits with os._exit(0) so the data-loader threads cannot hang it.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench_infer_flat next to this file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--encoder-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--matmul-precision", default="high", choices=["highest", "high"])
    ap.add_argument("--variant", default="v5pc", choices=["v5pc", "v5p", "v3c", "v0"],
                    help="Mamba-2 kernel variant (ignored for other encoders)")
    ap.add_argument("--no-kernel-switches", action="store_true")
    ap.add_argument("--reps", type=int, default=5, help="profiled forwards per stage")
    ap.add_argument("--loader-workers", type=int, default=4)
    args = ap.parse_args()

    if not args.no_kernel_switches:
        os.environ.setdefault("TRK_SSD_BUCKET16", "1")
        os.environ.setdefault("TRK_COMPILE_FRONTEND", "1")
    os.environ.setdefault("TRK_SEED_DTYPE", "float64")
    os.environ.setdefault("TRK_MATMUL_PRECISION", args.matmul_precision)

    import torch
    torch.set_float32_matmul_precision(args.matmul_precision)
    import bench_infer_flat as bif
    from torch.profiler import ProfilerActivity, profile
    from track_regression.seed_torch import gpu_seed_features

    dev = "cuda:0"
    model = bif.build_model(args.config, args.ckpt, dev)
    model.encoder_autocast_dtype = {"float32": torch.float32, "float16": torch.float16,
                                    "bfloat16": torch.bfloat16}[args.encoder_dtype]
    model, _ = bif.apply_kernel(model, args.variant)
    batches = bif.preload_batches(args.data_dir, args.batch_size, 1, args.loader_workers)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batches[0].items()}
    hf = b["hit_features"][0]
    n_tracks = int(b["track_lengths"].numel())
    names = ("d0", "z0", "phi", "theta", "qop")

    def stage_seed():
        return gpu_seed_features(hf, b["cu_seqlens"])

    def stage_frontend(res):
        x = torch.cat([hf, res], 1).unsqueeze(0)
        if os.environ.get("TRK_COMPILE_FRONTEND", "0") == "1":
            fe = getattr(model, "_compiled_frontend", None)
            if fe is None:
                fe = torch.compile(model._frontend_eager, dynamic=True)
                model._compiled_frontend = fe
            return fe(x)
        return model._frontend_eager(x)

    def stage_encoder(xe):
        with torch.amp.autocast("cuda", dtype=model.encoder_autocast_dtype,
                                enabled=model.encoder_autocast_dtype != torch.float32):
            if model.pool == "register_token":
                _, pooled = model.encoder(xe, x_sort_value=b.get("hit_time"), seq_idx=b["seq_idx"],
                                          cu_seqlens=b["cu_seqlens"])
            else:
                _, pooled = model.encoder(xe, x_sort_value=None, seq_idx=b["seq_idx"],
                                          cu_seqlens=b["cu_seqlens"])
        return pooled.to(next(model.output_head.parameters()).dtype)

    def stage_heads(pooled, seed):
        pred = model.output_head(model.pool_head(pooled))
        anchors = {f"seed_{n}": seed[:, i] for i, n in enumerate(names)}
        return model.loss_module.predict_physical(pred, anchors)

    def prof(fn, *a):
        with torch.inference_mode():
            for _ in range(10):                      # warm-up: autotune, compile
                out = fn(*a)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(args.reps):
                    out = fn(*a)
                torch.cuda.synchronize()
        ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
              and not e.name.startswith("##") and not e.name.startswith("Memcpy")
              and not e.name.startswith("Memset")]
        agg = defaultdict(lambda: [0, 0.0])
        for e in ev:
            agg[e.name][0] += 1
            agg[e.name][1] += e.device_time
        top = sorted(agg.items(), key=lambda kv: -kv[1][1])
        return out, len(ev) / args.reps, sum(e.device_time for e in ev) / args.reps / 1e3, top

    with torch.inference_mode():
        seed, res = stage_seed()
        xe = stage_frontend(res)
        pooled = stage_encoder(xe)

    rows = []
    for name, fn, a in [("seed", stage_seed, ()), ("front end", stage_frontend, (res,)),
                        ("encoder", stage_encoder, (xe,)), ("heads+predict", stage_heads, (pooled, seed))]:
        _, n, ms, top = prof(fn, *a)
        rows.append((name, n, ms, top))
    tot_ms = sum(r[2] for r in rows)
    tot_n = sum(r[1] for r in rows)
    print("\n" + "=" * 66)
    print(f" STAGE SHARE  {torch.cuda.get_device_name(dev)}  |  {n_tracks:,} tracks/batch  |  "
          f"encoder {args.encoder_dtype}, matmul {args.matmul_precision}, seed {os.environ['TRK_SEED_DTYPE']}")
    print("=" * 66)
    print(f"  {'stage':<14}{'kernels':>9}{'ms/forward':>13}{'share':>9}{'us/track':>11}")
    for name, n, ms, _ in rows:
        print(f"  {name:<14}{n:>9.0f}{ms:>13.3f}{100 * ms / tot_ms:>8.1f}%{1e3 * ms / n_tracks:>11.4f}")
    print(f"  {'TOTAL':<14}{tot_n:>9.0f}{tot_ms:>13.3f}{100:>8.0f}%{1e3 * tot_ms / n_tracks:>11.4f}"
          f"   -> {n_tracks / tot_ms * 1e3 / 1e6:.2f} M tracks/s (kernel time only)")
    print("-" * 66)
    print("  largest encoder kernels:")
    for kname, (cnt, t) in rows[2][3][:6]:
        print(f"   {t / args.reps / 1e3:8.3f} ms  x{cnt // args.reps:<3d} {kname[:90]}")
    print("=" * 66, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
