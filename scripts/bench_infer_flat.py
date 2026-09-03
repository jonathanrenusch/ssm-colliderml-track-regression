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
SEED_RESIDUALS = False
GPU_SEED = False
PROFILE_RANGE = False


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

    kw = dict(preprocessed_dir=data_dir, batch_size=batch_size, num_workers=workers,
              pin_memory=False, packed_batches=True, load_acts=False)
    if SEED_RESIDUALS:
        kw["seed_residual_features"] = True
    dm = ColliderMLRegrDataModule(**kw)
    dm.setup("test")
    loader = dm.test_dataloader()
    batches, total_tracks, total_tokens = [], 0, 0
    t_prev = time.perf_counter(); collate_s = []
    for inputs, _ in loader:
        collate_s.append(time.perf_counter() - t_prev)
        b = {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        batches.append(b)
        total_tracks += int(inputs["track_lengths"].numel())
        total_tokens += int(inputs["track_lengths"].sum())
        if len(batches) >= n_batches:
            break
        t_prev = time.perf_counter()
    if collate_s:
        # loader wall time per batch with num_workers=0 == collate cost (incl. seed [+ residuals]) on one core
        print(f"[bench] loader/collate per batch: mean {1e3*sum(collate_s)/len(collate_s):.0f} ms "
              f"({1e6*sum(collate_s)/max(total_tracks,1):.2f} us/track, workers={workers})", flush=True)
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
def _pad_batches_static(batches, bs):
    """Pad every preloaded packed batch to STATIC shapes for CUDA-graph replay.

    Pad tokens are grouped into K dummy tracks appended after the real ones
    (segment lengths 1..20, so the BL=32 scan constraint holds); their outputs
    are discarded.  Returns (padded pinned batches, B_static, T_cap).
    """
    import torch

    Ts = [int(b["hit_features"].shape[1]) for b in batches]
    max_t, min_t = max(Ts), min(Ts)
    K = (max_t - min_t) // 19 + 2                 # dummy tracks; each carries 1..20 pad tokens
    T_cap = max_t + K                             # every batch pads by >= K tokens
    B_static = bs + K
    out = []
    for b in batches:
        T = int(b["hit_features"].shape[1])
        pad = T_cap - T
        # dummy segment lengths: pad split over K segments, each in [1, 20]
        base_len, rem = divmod(pad, K)
        dlens = [base_len + (1 if i < rem else 0) for i in range(K)]
        assert all(1 <= l <= 20 for l in dlens), (pad, K, dlens)
        F = b["hit_features"].shape[2]
        hf = torch.zeros(1, T_cap, F, dtype=b["hit_features"].dtype)
        hf[0, :T] = b["hit_features"][0]
        cu_old = b["cu_seqlens"]
        cu = torch.empty(B_static + 1, dtype=cu_old.dtype)
        cu[: bs + 1] = cu_old
        run_ = int(cu_old[-1])
        for i, l in enumerate(dlens):
            run_ += l
            cu[bs + 1 + i] = run_
        assert run_ == T_cap
        seq = torch.empty(1, T_cap, dtype=b["seq_idx"].dtype)
        seq[0, :T] = b["seq_idx"][0]
        seq[0, T:] = torch.repeat_interleave(
            torch.arange(bs, B_static, dtype=b["seq_idx"].dtype),
            torch.tensor(dlens))
        tl_ = torch.empty(B_static, dtype=b["track_lengths"].dtype)
        tl_[:bs] = b["track_lengths"]
        tl_[bs:] = torch.tensor(dlens, dtype=b["track_lengths"].dtype)
        nb = {"hit_features": hf, "cu_seqlens": cu, "seq_idx": seq, "track_lengths": tl_}
        for k, v in b.items():
            if k in nb or not torch.is_tensor(v):
                continue
            if v.dim() >= 2 and v.shape[1] == T:      # per-token extras (hit_s, hit_time)
                pv = torch.zeros(v.shape[0], T_cap, *v.shape[2:], dtype=v.dtype)
                pv[:, :T] = v
                nb[k] = pv
            else:
                nb[k] = v
        out.append({k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in nb.items()})
    return out, B_static, T_cap


def run_graphed(model, batches, warmup, iters, device, bs):
    """CUDA-graph replay of the forward: static buffers, one graph, per-iter copy_ + replay.

    Returns (per_ms, wall_s, n_tracks, peak_gib, max_dev): max_dev is the worst
    |graph - eager| over the real tracks of batch 0 (padding-inertness check).
    """
    import torch

    padded, B_static, T_cap = _pad_batches_static(batches, bs)
    print(f"[bench] cuda-graph: B_static={B_static} (bs={bs} + {B_static-bs} dummies), T_cap={T_cap}", flush=True)
    static = {k: v.to(device) for k, v in padded[0].items() if torch.is_tensor(v)}

    with torch.inference_mode():
        # warmup eagerly on the padded shapes (autotune/compile), then capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(max(warmup, 3)):
                out = model(static)
                if (i + 1) % 10 == 0:
                    print(f"[bench] warmup {i+1}/{warmup}", flush=True)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = model(static)

        # padding-inertness check: replay(batch0) vs eager(batch0 unpadded)
        g.replay(); torch.cuda.synchronize()
        graph_pred = _pred_tensor(static_out)[:bs].clone()
        eager_out = model({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batches[0].items()})
        eager_pred = _pred_tensor(eager_out)
        max_dev = float((graph_pred - eager_pred[:bs]).abs().max())

        tracks_per_batch = [int(b["track_lengths"].numel()) for b in batches]
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        n_tracks = 0
        t0 = time.perf_counter()
        for i in range(iters):
            b = padded[i % len(padded)]
            starts[i].record()
            for k, v in static.items():
                v.copy_(b[k], non_blocking=True)
            g.replay()
            ends[i].record()
            n_tracks += tracks_per_batch[i % len(batches)]
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    per = [st.elapsed_time(e) for st, e in zip(starts, ends)]
    peak = torch.cuda.max_memory_allocated() / 2**30
    return per, wall, n_tracks, peak, max_dev


def _pred_tensor(out):
    import torch
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        if "pred" in out:
            return out["pred"]
        return torch.cat([v.reshape(v.shape[0], -1) for v in out.values()], dim=1)
    return out[0]


def run(model, batches, warmup, iters, device):
    import torch

    def to_dev(b):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in b.items()}

    if GPU_SEED:
        from track_regression.seed_torch import gpu_seed_features
        names = ("d0", "z0", "phi", "theta", "qop")
        raw_model = model
        def model(b):                                   # noqa: F811  -- deployment path
            hf = b["hit_features"][0]
            seed, res = gpu_seed_features(hf, b["cu_seqlens"])
            b2 = dict(b); b2["hit_features"] = torch.cat([hf, res], 1).unsqueeze(0)
            out = raw_model(b2)
            anchors = {f"seed_{n}": seed[:, i] for i, n in enumerate(names)}
            return raw_model.loss_module.predict_physical(out["pred"], anchors)

    tracks_per_batch = [int(b["track_lengths"].numel()) for b in batches]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(warmup):                      # covers Triton autotune/compile
            model(to_dev(batches[i % len(batches)]))
            if (i + 1) % 10 == 0:
                print(f"[bench] warmup {i+1}/{warmup}", flush=True)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        n_tracks = 0
        if PROFILE_RANGE:
            torch.cuda.cudart().cudaProfilerStart()
        t0 = time.perf_counter()
        for i in range(iters):
            b = batches[i % len(batches)]
            starts[i].record()
            model(to_dev(b))
            ends[i].record()
            n_tracks += tracks_per_batch[i % len(batches)]
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if PROFILE_RANGE:
            torch.cuda.cudart().cudaProfilerStop()

    per = [s.elapsed_time(e) for s, e in zip(starts, ends)]  # ms, incl. H2D
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return per, wall, n_tracks, peak_gib


def report(per, wall, n_tracks, peak_gib, meta):
    import numpy as np
    a = np.asarray(per)
    tps_wall = n_tracks / wall
    print("\n" + "=" * 62)
    print(" INFERENCE BENCHMARK RESULT")
    print("=" * 62)
    for k, v in meta.items():
        print(f"  {k:<22}: {v}")
    print("-" * 62)
    print(f"  batches timed         : {len(per)}")
    print(f"  per-batch ms  mean    : {a.mean():.3f}")
    print(f"                std     : {a.std():.3f}  (CV {100*a.std()/a.mean():.2f}%)")
    print(f"                p50/p90 : {np.percentile(a,50):.3f} / {np.percentile(a,90):.3f}")
    print(f"  throughput            : {tps_wall:,.0f} tracks/s  (wall-clock, RAM-preloaded)")
    print(f"  t2k (2000 x per-track): {2000.0*1000.0/tps_wall:.3f} ms")
    print(f"  peak VRAM             : {peak_gib:.2f} GiB")
    print("=" * 62 + "\n", flush=True)
    return {"tracks_per_s": round(tps_wall, 1), "t2k_ms": round(2000e3 / tps_wall, 3),
            "per_batch_ms_mean": round(float(a.mean()), 4), "peak_vram_gib": round(peak_gib, 3)}


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
    ap.add_argument("--seed-residuals", action="store_true", help="collate appends the 3 seed-residual hit features (P')")
    ap.add_argument("--profile-range", action="store_true", help="wrap the timed loop in cudaProfilerStart/Stop (for nsys --capture-range=cudaProfilerApi)")
    ap.add_argument("--gpu-seed", action="store_true", help="deployment mode: seed + residual features computed on the GPU inside the timed loop (collate gives 12 features)")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the forward in a CUDA graph (static shapes via dummy-track "
                         "padding) and time buffer-copy + replay — the small-batch launch-gap killer. "
                         "Incompatible with TRK_SSD_BUCKET16 (data-dependent split).")
    args = ap.parse_args()
    if args.cuda_graph and os.environ.get("TRK_SSD_BUCKET16") == "1":
        sys.exit("[bench] --cuda-graph is incompatible with TRK_SSD_BUCKET16=1 (device-sync split)")

    os.environ.setdefault("TRK_MATMUL_PRECISION", args.matmul_precision)
    import torch
    torch.set_float32_matmul_precision(args.matmul_precision)
    if not torch.cuda.is_available():
        sys.exit("[bench] no CUDA device visible")
    dev = args.device
    name = torch.cuda.get_device_name(dev)
    cap = "".join(map(str, torch.cuda.get_device_capability(dev)))

    global SEED_RESIDUALS, GPU_SEED, PROFILE_RANGE
    SEED_RESIDUALS = bool(args.seed_residuals)
    GPU_SEED = bool(args.gpu_seed)
    PROFILE_RANGE = bool(args.profile_range)
    model = build_model(args.config, args.ckpt, dev)
    model, used = apply_kernel(model, args.variant)
    batches = preload_batches(args.data_dir, args.batch_size, args.preload_batches, args.loader_workers)

    graph_note = None
    try:
        if args.cuda_graph:
            call = model
            if args.gpu_seed:
                from track_regression.seed_torch import gpu_seed_features
                names = ("d0", "z0", "phi", "theta", "qop")
                raw_model = model
                def call(b):
                    hf = b["hit_features"][0]
                    seed, res = gpu_seed_features(hf, b["cu_seqlens"], max_len=20)
                    b2 = dict(b); b2["hit_features"] = torch.cat([hf, res], 1).unsqueeze(0)
                    out = raw_model(b2)
                    anchors = {f"seed_{n}": seed[:, i] for i, n in enumerate(names)}
                    return raw_model.loss_module.predict_physical(out["pred"], anchors)
            per, wall, n_tracks, peak, max_dev = run_graphed(
                call, batches, args.warmup, args.iters, dev, args.batch_size)
            graph_note = f"CUDA graph (replayed); graph-vs-eager max |dpred| = {max_dev:.3e}"
            print(f"[bench] {graph_note}", flush=True)
        else:
            per, wall, n_tracks, peak = run(model, batches, args.warmup, args.iters, dev)
    except torch.cuda.OutOfMemoryError:
        sys.exit(f"[bench] OOM at batch_size={args.batch_size} on {name} "
                 f"({torch.cuda.get_device_properties(dev).total_memory/2**30:.0f} GiB) "
                 f"-- retry with a smaller --batch-size.")

    report(per, wall, n_tracks, peak, {
        "GPU": f"{name} (sm_{cap})",
        "config": args.config.name,
        "checkpoint": args.ckpt.name,
        "kernel variant": used,
        "matmul precision": args.matmul_precision + (" (strict fp32)" if args.matmul_precision == "highest" else " (TF32 linears)"),
        "batch size": args.batch_size,
        "seed mode": ("GPU (in timed loop)" if args.gpu_seed
                      else "CPU collate (+residuals)" if args.seed_residuals
                      else "GPU (auto, in model forward)"
                      if batches[0]["hit_features"].shape[-1] < getattr(model, "input_dim", 0)
                      else "none (features complete)"),
    })


if __name__ == "__main__":
    main()
