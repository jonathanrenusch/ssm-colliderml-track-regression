#!/usr/bin/env python3
"""GPU kernel time per stage of the deployed forward pass: seed | front end | encoder | heads.

    python scripts/bench_stage_share.py --config configs/minGRU_stage1.yaml --ckpt <ckpt> \
        --data-dir data/eval/ttbar_bench --batch-size 131072

Same path as ``bench_infer.py --mode deployed`` (float64 GPU seed, compiled
front end, fp16 encoder, fused kernels, TF32 GEMMs); every stage is profiled
separately with ``torch.profiler`` and its CUDA kernel count and kernel time
are reported.  Select the GPU with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=131072)
    ap.add_argument("--reps", type=int, default=5, help="profiled forwards per stage")
    ap.add_argument("--loader-workers", type=int, default=4)
    args = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile

    from bench_infer import PARAMS, preload_batches
    from track_regression.inference import load_model
    from track_regression.seed_torch import gpu_seed_features

    torch.set_float32_matmul_precision("high")
    model = load_model(args.config, args.ckpt, "cuda", "float16")
    b = {k: v.cuda() for k, v in preload_batches(args.data_dir, args.batch_size, 1, args.loader_workers)[0].items()}
    hf, cu = b["hit_features"][0], b["cu_seqlens"]
    n_tracks = int(b["track_lengths"].numel())
    frontend = torch.compile(model._frontend, dynamic=True)

    def stage_seed():
        return gpu_seed_features(hf, cu, max_len=20)

    def stage_frontend(res):
        return frontend(torch.cat([hf, res], 1).unsqueeze(0))

    def stage_encoder(xe):
        with torch.autocast("cuda", dtype=model.encoder_autocast_dtype):
            return model.encoder(xe, cu_seqlens=cu, seq_idx=b["seq_idx"])[1].float()

    def stage_heads(pooled, seed):
        pred = model.output_head(model.pool_head(pooled))
        return model.loss_module.predict_physical(pred, {f"seed_{p}": seed[:, i] for i, p in enumerate(PARAMS)})

    def prof(fn, *a):
        with torch.inference_mode():
            for _ in range(10):                      # warm-up: autotune, compile
                fn(*a)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(args.reps):
                    fn(*a)
                torch.cuda.synchronize()
        ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
              and not e.name.startswith(("##", "Memcpy", "Memset"))]
        agg = defaultdict(lambda: [0, 0.0])
        for e in ev:
            agg[e.name][0] += 1
            agg[e.name][1] += e.device_time
        top = sorted(agg.items(), key=lambda kv: -kv[1][1])
        return len(ev) / args.reps, sum(e.device_time for e in ev) / args.reps / 1e3, top

    with torch.inference_mode():
        seed, res = stage_seed()
        xe = stage_frontend(res)
        pooled = stage_encoder(xe)
    rows = [(name, *prof(fn, *a)) for name, fn, a in [
        ("seed", stage_seed, ()), ("front end", stage_frontend, (res,)),
        ("encoder", stage_encoder, (xe,)), ("heads+predict", stage_heads, (pooled, seed))]]
    tot_ms, tot_n = sum(r[2] for r in rows), sum(r[1] for r in rows)
    print("\n" + "=" * 66)
    print(f" STAGE SHARE  {torch.cuda.get_device_name()}  |  {n_tracks:,} tracks/batch")
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
