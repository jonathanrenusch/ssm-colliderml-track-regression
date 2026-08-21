"""Self-contained inference benchmark for the short-sequence SSM track-fitter.

Designed to run inside the RTX-profiling Docker image (see Dockerfile), but also
runs natively via `pixi run -e default python docker/rtx_infer/bench_infer.py ...`.

What it does
------------
* Builds a TrackParameterRegressor from a (fully-resolved) config's ``model`` node,
  loads a checkpoint (strict), and swaps the encoder onto the fused inference
  kernel (``v5pc``; falls back to ``v5p``) -- the portable Triton path with no
  Hopper-only features, so it JIT-autotunes on Ada (sm_89) unchanged.
* Runs strict IEEE fp32 end to end (no TF32, no bf16) by default.
* PRELOADS a set of pre-collated packed batches into pinned CPU RAM once, then
  times the forward loop over them -- so disk + collate are out of the hot loop
  and the measurement is compute-bound (host->device copy overlaps and is ~1-3%).
* Prints throughput / latency / VRAM so the numbers are visible without a profiler
  (and is trivially wrappable in nsys/ncu -- see README).

Only depends on torch + the `track_regression` package (+ pyyaml, h5py via the
datamodule). No dependency on the repo's perf harness.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import statistics
import sys
import time
from pathlib import Path

import yaml


# --------------------------------------------------------------------------- #
# config instantiation (minimal, self-contained)
# --------------------------------------------------------------------------- #
def _import(dotted: str):
    mod, _, cls = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), cls)


def _instantiate(node):
    """Turn class_path/init_args dicts into objects; drop init_args a class no
    longer accepts (saved configs from older runs may carry removed kwargs)."""
    if isinstance(node, dict):
        if "class_path" in node:
            cls = _import(node["class_path"])
            init_args = {k: _instantiate(v) for k, v in (node.get("init_args") or {}).items()}
            try:
                params = inspect.signature(cls.__init__).parameters
                if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    stale = [k for k in init_args if k not in params]
                    for k in stale:
                        print(f"[bench] {cls.__name__}: dropping stale init_arg {k!r}", flush=True)
                        init_args.pop(k)
            except (TypeError, ValueError):
                pass
            return cls(**init_args)
        return {k: _instantiate(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_instantiate(v) for v in node]
    return node


def build_model(config_path: Path, ckpt_path: Path, device: str):
    import torch

    cfg = yaml.safe_load(open(config_path))
    model_node = cfg["model"]["model"] if "model" in cfg.get("model", {}) else cfg["model"]
    model = _instantiate(model_node)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Lightning prefixes wrapper params with "model."
    state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")} or state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[bench] state_dict load: {len(missing)} missing, {len(unexpected)} unexpected "
              f"(first missing: {missing[:3]})", flush=True)
    n = sum(p.numel() for p in model.parameters())
    enc = model.encoder
    enc_desc = (f"{type(enc).__name__} L={getattr(enc,'num_layers','?')} "
                f"dim={getattr(enc,'dim','?')} d_state={getattr(getattr(enc,'final_layer',None),'d_state','?')}")
    print(f"[bench] params: {n/1e6:.3f} M | {enc_desc}", flush=True)
    return model.eval().to(device)


def apply_kernel(model, variant: str):
    from track_regression.mamba_short import apply_variant
    try:
        return (apply_variant(model, variant) or model), variant
    except Exception as e:  # noqa: BLE001
        print(f"[bench] {variant} failed ({type(e).__name__}: {e}); falling back to v5p", flush=True)
        import torch
        torch._dynamo.reset()
        return (apply_variant(model, "v5p") or model), "v5p"


# --------------------------------------------------------------------------- #
# data preloading into pinned CPU RAM
# --------------------------------------------------------------------------- #
def preload_batches(data_dir: str, batch_size: int, n_batches: int, workers: int):
    import torch
    from track_regression.data import ColliderMLRegrDataModule

    dm = ColliderMLRegrDataModule(
        preprocessed_dir=data_dir, batch_size=batch_size, num_workers=workers,
        pin_memory=False, packed_batches=True, load_acts=False,
        streaming=True, shard_buffer_size=8,
    )
    dm.setup("test")
    loader = dm.test_dataloader()
    batches, total_tracks, total_tokens = [], 0, 0
    for inputs, _ in loader:
        b = {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        batches.append(b)
        total_tracks += int(inputs["track_lengths"].numel())
        total_tokens += int(inputs["track_lengths"].sum())
        if len(batches) >= n_batches:
            break
    if not batches:
        raise RuntimeError(f"no batches loaded from {data_dir}")
    mib = sum(v.element_size() * v.nelement() for b in batches for v in b.values()
              if torch.is_tensor(v)) / 2**20
    print(f"[bench] preloaded {len(batches)} batches -> {total_tracks:,} tracks "
          f"({mib:.0f} MiB pinned RAM), mean len "
          f"{total_tokens/total_tracks:.1f}", flush=True)
    return batches


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def run(model, batches, warmup, iters, device):
    import torch

    def to_dev(b):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in b.items()}

    tracks_per_batch = [int(b["track_lengths"].numel()) for b in batches]
    # Stage the batches onto the GPU ONCE (resident). The timed loop then measures
    # RAW GPU COMPUTE only -- host->device copies are outside the timer. The data
    # pipeline (disk->collate->H2D) is a separate deployment concern; we report
    # the H2D cost separately below for reference, but it is NOT in the headline.
    gpu_batches = [to_dev(b) for b in batches]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(warmup):                      # covers Triton autotune/compile
            model(gpu_batches[i % len(gpu_batches)])
            if (i + 1) % 10 == 0:
                print(f"[bench] warmup {i+1}/{warmup}", flush=True)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        n_tracks = 0
        t0 = time.perf_counter()
        for i in range(iters):
            starts[i].record()
            model(gpu_batches[i % len(gpu_batches)])   # compute only -- data resident
            ends[i].record()
            n_tracks += tracks_per_batch[i % len(gpu_batches)]
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    per = [s.elapsed_time(e) for s, e in zip(starts, ends)]   # ms, COMPUTE ONLY
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    # Separately MEASURE the host->device copy cost (pinned RAM -> GPU) for reference.
    h2d = []
    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    for i in range(min(40, 5 * len(batches))):
        b = batches[i % len(batches)]
        torch.cuda.synchronize(); se.record()
        _ = to_dev(b)
        ee.record(); torch.cuda.synchronize()
        h2d.append(se.elapsed_time(ee))
    import numpy as np
    h2d_ms = float(np.median(h2d))
    return per, wall, n_tracks, peak_gib, h2d_ms


def report(per, wall, n_tracks, peak_gib, h2d_ms, meta):
    import numpy as np
    a = np.asarray(per)
    tps = n_tracks / wall
    print("\n" + "=" * 62)
    print(" INFERENCE BENCHMARK RESULT")
    print("=" * 62)
    for k, v in meta.items():
        print(f"  {k:<22}: {v}")
    print("-" * 62)
    print(f"  batches timed         : {len(per)}")
    print(f"  per-batch ms  mean    : {a.mean():.3f}   (RAW GPU COMPUTE, H2D excluded)")
    print(f"                std     : {a.std():.3f}  (CV {100*a.std()/a.mean():.2f}%)")
    print(f"                p50/p90 : {np.percentile(a,50):.3f} / {np.percentile(a,90):.3f}")
    print(f"  throughput            : {tps:,.0f} tracks/s  (compute only)")
    print(f"  t2k (2000 x per-track): {2000.0*1000.0/tps:.3f} ms")
    print(f"  peak VRAM             : {peak_gib:.2f} GiB")
    print(f"  -- reference: H2D copy/batch {h2d_ms:.3f} ms "
          f"({100*h2d_ms/a.mean():.2f}% of compute; separate pipeline concern)")
    print("=" * 62 + "\n", flush=True)
    return {"tracks_per_s": round(tps, 1), "t2k_ms": round(2000e3 / tps, 3),
            "per_batch_ms_mean": round(float(a.mean()), 4), "peak_vram_gib": round(peak_gib, 3),
            "h2d_ms": round(h2d_ms, 4), "h2d_pct": round(100 * h2d_ms / float(a.mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path, help="resolved config with a model node")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, help="preprocessed data dir (e.g. /scratch/.../p200_...)")
    ap.add_argument("--batch-size", type=int, default=8192, help="tracks per batch (lower for smaller VRAM)")
    ap.add_argument("--preload-batches", type=int, default=16, help="# batches to hold in pinned RAM")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--variant", default="v5pc", choices=["v5pc", "v5p", "v3c", "v0"])
    ap.add_argument("--matmul-precision", default="highest", choices=["highest", "high"],
                    help="highest = strict IEEE fp32 (default); high = TF32 in linear GEMMs")
    ap.add_argument("--loader-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.environ.setdefault("TRK_MATMUL_PRECISION", args.matmul_precision)
    import torch
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.matmul_precision == "highest":
        # Full IEEE fp32 EVERYWHERE: also forbid TF32 in cuBLAS matmul and cuDNN
        # convolutions (separate switches from set_float32_matmul_precision). The
        # v5pc path does conv/scan/norm inside the fp32 Triton kernel already, so
        # this only matters for any fallback (e.g. F.conv1d/cuDNN in v3/stock) --
        # belt-and-suspenders so nothing in the graph silently runs TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        sys.exit("[bench] no CUDA device visible")
    dev = args.device
    name = torch.cuda.get_device_name(dev)
    cap = "".join(map(str, torch.cuda.get_device_capability(dev)))

    model = build_model(args.config, args.ckpt, dev)
    model, used = apply_kernel(model, args.variant)
    batches = preload_batches(args.data_dir, args.batch_size, args.preload_batches, args.loader_workers)

    try:
        per, wall, n_tracks, peak, h2d_ms = run(model, batches, args.warmup, args.iters, dev)
    except torch.cuda.OutOfMemoryError:
        sys.exit(f"[bench] OOM at batch_size={args.batch_size} on {name} "
                 f"({torch.cuda.get_device_properties(dev).total_memory/2**30:.0f} GiB) "
                 f"-- retry with a smaller --batch-size.")

    report(per, wall, n_tracks, peak, h2d_ms, {
        "GPU": f"{name} (sm_{cap})",
        "config": args.config.name,
        "checkpoint": args.ckpt.name,
        "kernel variant": used,
        "matmul precision": args.matmul_precision + (" (strict fp32)" if args.matmul_precision == "highest" else " (TF32 linears)"),
        "batch size": args.batch_size,
    })


if __name__ == "__main__":
    main()
