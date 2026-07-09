"""Benchmark one encoder variant of the track-regression model.

Modes
-----
staged
    Build the datamodule (test split), capture ONE real batch, move it to
    GPU once, then time ``model(inputs)`` for ``--timed-iters`` iterations
    (cuda-event pairs, single final sync) after ``--warmup`` warmup iters.
    Pure device-side forward cost — no dataloader in the timed path.
e2e
    Iterate the real test dataloader for ``--max-batches`` batches, timing
    each batch *including* the H2D copy (cuda_timer-style sync per batch).
confirm
    staged at exactly ``--batch-size`` — meant to be launched in a fresh
    process to confirm a sweep point.
sweep
    Repeated staged at doubling batch sizes starting from ``--batch-size``.
    Batch sizes are synthesized WITHOUT rebuilding the dataloader by
    replicating/slicing the one captured batch (packed: token tensors are
    concatenated and cu_seqlens/seq_idx rebuilt; padded: repeat along the
    batch dim). CUDA OOM / launch failures are recorded verbatim and the
    last-good/first-fail boundary is bisected to <=5 %.

Variant plumbing: ``v0`` runs the stock model as configured (packed). Any
other variant calls ``track_regression.mamba_short.apply_variant(model, v)``;
if that module/function is missing or raises NotImplementedError the script
exits with code 3 (the kernel module lands separately — the harness must not
block on it).

Usage::

    pixi run -e default python scripts/perf/bench_variant.py \\
        --config src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml \\
        --ckpt logs/.../epoch=049-val_total=0.00125.ckpt \\
        --variant v0 --mode staged --batch-size 22000 --timed-iters 200 \\
        --out-jsonl docs/perf/results/night1/results.jsonl --job-id v0_staged
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

common.ensure_src_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpu_sampler import GpuSampler  # noqa: E402

VARIANTS = ("v0", "v2p", "v3", "v3c", "v4", "v5", "v5p", "v5pc", "auto")  # keep in sync with mamba_short._VARIANTS
MODES = ["staged", "e2e", "sweep", "confirm"]
INT32_MAX = 2**31 - 1

RESULT_FIELDS = [
    "ts", "job_id", "tag", "variant", "mode",
    "batch_tracks", "batch_tokens",
    "t_iter_ms_mean", "t_iter_ms_std", "t_iter_ms_p10", "t_iter_ms_p50", "t_iter_ms_p90",
    "tracks_per_s", "tokens_per_s", "t2k_ms",
    "vram_gib_torch_peak",
    "power_w_mean", "power_w_max", "sm_util_mean", "vram_mib_max",
    "clocks_sm_mean", "n_samples", "throttled",
    "precision_flags", "env", "status", "error",
    "len_mean", "len_min", "len_max", "timed_iters", "warmup",
]


# ---------------------------------------------------------------------------
# Model / variant plumbing
# ---------------------------------------------------------------------------

def build_model(
    cfg: dict, ckpt_path: Path | None, random_weights: bool = False
) -> torch.nn.Module:
    """Instantiate TrackParameterRegressor and load checkpoint weights (strict).

    ``random_weights=True`` skips the checkpoint load entirely (model exactly
    as configured, default random init) — timing-only benches for configs
    without a trained checkpoint. The timed path is identical either way.
    """
    model = common._instantiate(cfg["model"]["model"])
    if random_weights:
        print("[bench] --random-weights: skipping checkpoint load "
              "(random init; timing-only, NOT physics-valid)", flush=True)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Lightning checkpoints prefix all keys with "model." (wrapper's self.model).
        state = {
            k[len("model."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state, strict=True)
    model = model.eval().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    src = "RANDOM INIT" if random_weights else str(ckpt_path)
    print(f"[bench] params: {n_params / 1e6:.3f} M   weights: {src}", flush=True)
    return model


def apply_variant_or_exit(model: torch.nn.Module, variant: str) -> torch.nn.Module:
    """v0 is a no-op; other variants require track_regression.mamba_short."""
    if variant == "v0":
        return model
    try:
        from track_regression.mamba_short import apply_variant
    except ImportError as e:
        print(
            f"[bench] EXIT 3: variant {variant!r} requires "
            f"track_regression.mamba_short.apply_variant, which is not "
            f"importable yet ({e}). The kernel module lands separately — "
            f"re-run this job once it exists.",
            file=sys.stderr, flush=True,
        )
        sys.exit(3)
    try:
        return apply_variant(model, variant) or model
    except NotImplementedError as e:
        print(
            f"[bench] EXIT 3: apply_variant(model, {variant!r}) raised "
            f"NotImplementedError: {e}",
            file=sys.stderr, flush=True,
        )
        sys.exit(3)


# ---------------------------------------------------------------------------
# Batch capture + synthetic resizing
# ---------------------------------------------------------------------------

def build_datamodule(cfg: dict, batch_size: int, num_workers: int):
    from track_regression.data import ColliderMLRegrDataModule

    data_cfg = dict(cfg["data"])
    data_cfg["batch_size"] = batch_size
    data_cfg["num_workers"] = num_workers
    dm = ColliderMLRegrDataModule(**data_cfg)
    dm.setup("test")
    return dm


def batch_stats(inputs: dict[str, torch.Tensor]) -> dict:
    """Track/token counts + length stats for a packed or padded batch."""
    if "cu_seqlens" in inputs:
        lengths = inputs["track_lengths"].to("cpu", torch.int64)
        n_tracks = int(lengths.numel())
        n_tokens = int(lengths.sum())
    else:
        hv = inputs.get("hit_valid")
        if hv is not None:
            lengths = hv.sum(dim=1).to("cpu", torch.int64)
        else:
            b, l = inputs["hit_features"].shape[:2]
            lengths = torch.full((b,), l, dtype=torch.int64)
        n_tracks = int(lengths.numel())
        n_tokens = int(lengths.sum())
    return {
        "batch_tracks": n_tracks,
        "batch_tokens": n_tokens,
        "len_mean": round(float(lengths.float().mean()), 3),
        "len_min": int(lengths.min()),
        "len_max": int(lengths.max()),
    }


def resize_packed_batch(inputs: dict[str, torch.Tensor], target_tracks: int) -> dict[str, torch.Tensor]:
    """Synthesize a packed batch with ``target_tracks`` tracks from a captured one.

    Token tensors (shape ``(1, T, ...)``) are tiled along the token dim and
    sliced at a track boundary; ``cu_seqlens``/``seq_idx`` are rebuilt from the
    tiled ``track_lengths``. cu_seqlens is accumulated in int64 and cast back
    to int32 only after an overflow check (the kernels require int32).
    """
    base_lengths = inputs["track_lengths"].to("cpu", torch.int64)
    b0 = int(base_lengths.numel())
    t0 = int(inputs["hit_features"].shape[1])
    reps = math.ceil(target_tracks / b0)

    lengths = base_lengths.repeat(reps)[:target_tracks]
    cu = torch.zeros(target_tracks + 1, dtype=torch.int64)
    torch.cumsum(lengths, dim=0, out=cu[1:])
    total_tokens = int(cu[-1])
    if total_tokens > INT32_MAX:
        raise OverflowError(
            f"synthetic packed batch of {target_tracks} tracks needs "
            f"{total_tokens} tokens > int32 max — cu_seqlens cannot be int32"
        )

    out: dict[str, torch.Tensor] = {}
    for key, t in inputs.items():
        if key == "cu_seqlens":
            out[key] = cu.to(torch.int32).to(t.device)
        elif key == "track_lengths":
            out[key] = lengths.to(t.dtype).to(t.device)
        elif key == "seq_idx":
            seq = torch.repeat_interleave(
                torch.arange(target_tracks, dtype=torch.int64), lengths,
            )
            out[key] = seq.to(torch.int32).unsqueeze(0).to(t.device)
        elif t.ndim >= 2 and t.shape[0] == 1 and t.shape[1] == t0:
            tiled = t.repeat((1, reps) + (1,) * (t.ndim - 2))
            out[key] = tiled[:, :total_tokens].contiguous()
        else:
            out[key] = t
    return out


def resize_padded_batch(inputs: dict[str, torch.Tensor], target_tracks: int) -> dict[str, torch.Tensor]:
    """Repeat a padded ``(B, L, ...)`` batch along the batch dim, then slice."""
    b0 = int(inputs["hit_features"].shape[0])
    reps = math.ceil(target_tracks / b0)
    out: dict[str, torch.Tensor] = {}
    for key, t in inputs.items():
        if t.ndim >= 1 and t.shape[0] == b0:
            tiled = t.repeat((reps,) + (1,) * (t.ndim - 1))
            out[key] = tiled[:target_tracks].contiguous()
        else:
            out[key] = t
    return out


def resize_batch(inputs: dict[str, torch.Tensor], target_tracks: int) -> dict[str, torch.Tensor]:
    if "cu_seqlens" in inputs:
        return resize_packed_batch(inputs, target_tracks)
    return resize_padded_batch(inputs, target_tracks)


def to_cuda(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.cuda(non_blocking=True) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timed_forward_loop(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    warmup: int,
    timed_iters: int,
) -> list[float]:
    """Warmup + timed loop with per-iter cuda-event pairs and one final sync."""
    with torch.inference_mode():
        for i in range(warmup):
            model(inputs)
            if (i + 1) % 25 == 0:
                print(f"[bench] warmup {i + 1}/{warmup}", flush=True)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
        torch.cuda.nvtx.range_push("timed")  # parse_profiles.py keys on this
        for i in range(timed_iters):
            starts[i].record()
            model(inputs)
            ends[i].record()
            if (i + 1) % 25 == 0:
                print(f"[bench] iter {i + 1}/{timed_iters}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def times_to_row(times_ms: list[float], n_tracks: int, n_tokens: int) -> dict:
    t = np.asarray(times_ms, dtype=np.float64)
    mean = float(t.mean())
    tracks_per_s = n_tracks / (mean / 1e3)
    return {
        "t_iter_ms_mean": round(mean, 4),
        "t_iter_ms_std": round(float(t.std()), 4),
        "t_iter_ms_p10": round(float(np.percentile(t, 10)), 4),
        "t_iter_ms_p50": round(float(np.percentile(t, 50)), 4),
        "t_iter_ms_p90": round(float(np.percentile(t, 90)), 4),
        "tracks_per_s": round(tracks_per_s, 1),
        "tokens_per_s": round(n_tokens / (mean / 1e3), 1),
        "t2k_ms": round(2000.0 * 1000.0 / tracks_per_s, 4),
    }


def sampler_gpu_index() -> int:
    """Physical GPU index for nvidia-smi: first entry of CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = cvd.split(",")[0].strip()
    return int(first) if first.isdigit() else 0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _emit(args, row: dict) -> None:
    row.setdefault("ts", common.utc_ts())
    row.setdefault("job_id", args.job_id)
    row.setdefault("tag", args.tag)
    row.setdefault("variant", args.variant)
    if args.out_jsonl:
        common.append_jsonl(args.out_jsonl, row)
        common.append_csv(Path(args.out_jsonl).with_suffix(".csv"), row, RESULT_FIELDS)
    line = {k: row.get(k) for k in (
        "mode", "batch_tracks", "batch_tokens", "t_iter_ms_mean",
        "tracks_per_s", "t2k_ms", "vram_gib_torch_peak", "status",
    )}
    print(f"[bench] RESULT {line}", flush=True)


def run_point(
    args,
    model: torch.nn.Module,
    gpu_inputs: dict[str, torch.Tensor],
    mode: str,
    precision_flags: dict,
    env: dict,
) -> dict:
    """One staged measurement of an on-GPU batch → emitted result row."""
    stats = batch_stats(gpu_inputs)
    torch.cuda.reset_peak_memory_stats()
    sampler = None
    if args.gpu_samples_csv:
        sampler = GpuSampler(sampler_gpu_index(), args.gpu_samples_csv).start()
    row: dict = {
        "mode": mode, **stats,
        "precision_flags": precision_flags, "env": env,
        "timed_iters": args.timed_iters, "warmup": args.warmup,
    }
    try:
        times = timed_forward_loop(model, gpu_inputs, args.warmup, args.timed_iters)
        row.update(times_to_row(times, stats["batch_tracks"], stats["batch_tokens"]))
        row["status"] = "ok"
        row["error"] = ""
    finally:
        row["vram_gib_torch_peak"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if sampler is not None:
            sampler.stop()
            row.update(sampler.summary())
    _emit(args, row)
    return row


def run_e2e(args, model, cfg, precision_flags, env) -> None:
    from track_regression._lib.cuda_timer import cuda_timer

    dm = build_datamodule(cfg, args.batch_size, args.num_workers)
    loader = dm.test_dataloader()
    max_batches = args.max_batches or (args.warmup + args.timed_iters)

    sampler = None
    if args.gpu_samples_csv:
        sampler = GpuSampler(sampler_gpu_index(), args.gpu_samples_csv).start()
    torch.cuda.reset_peak_memory_stats()

    times_ms: list[float] = []
    n_tracks_seen = 0
    n_tokens_seen = 0
    stats_first: dict | None = None
    try:
        with torch.inference_mode():
            for batch_idx, (inputs, _targets) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                stats = batch_stats(inputs)
                if stats_first is None:
                    stats_first = stats
                with cuda_timer(times_ms):  # syncs per batch; includes H2D
                    gpu_inputs = to_cuda(inputs)
                    model(gpu_inputs)
                if batch_idx >= args.warmup:
                    n_tracks_seen += stats["batch_tracks"]
                    n_tokens_seen += stats["batch_tokens"]
                if (batch_idx + 1) % 25 == 0:
                    print(f"[bench] e2e batch {batch_idx + 1}/{max_batches}", flush=True)
        torch.cuda.synchronize()
    finally:
        if sampler is not None:
            sampler.stop()

    timed = times_ms[args.warmup:]
    if not timed:
        print(f"[bench] ERROR: only {len(times_ms)} batches, need > {args.warmup}", flush=True)
        sys.exit(1)
    total_s = sum(timed) / 1e3
    row = {
        "mode": "e2e", **(stats_first or {}),
        **times_to_row(timed, n_tracks_seen // len(timed), n_tokens_seen // len(timed)),
        "tracks_per_s": round(n_tracks_seen / total_s, 1),
        "tokens_per_s": round(n_tokens_seen / total_s, 1),
        "t2k_ms": round(2000.0 * 1000.0 / (n_tracks_seen / total_s), 4),
        "vram_gib_torch_peak": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "precision_flags": precision_flags, "env": env,
        "timed_iters": len(timed), "warmup": args.warmup,
        "status": "ok", "error": "",
    }
    if sampler is not None:
        row.update(sampler.summary())
    _emit(args, row)


def _try_sweep_point(args, model, cpu_inputs, n_tracks, precision_flags, env) -> tuple[bool, dict]:
    """One sweep point; returns (ok, row). CUDA failures are recorded verbatim."""
    gpu_inputs = None
    try:
        gpu_inputs = to_cuda(resize_batch(cpu_inputs, n_tracks))
        row = run_point(args, model, gpu_inputs, "sweep", precision_flags, env)
        return True, row
    except (torch.cuda.OutOfMemoryError, OverflowError, RuntimeError) as e:
        is_cuda_fail = (
            isinstance(e, (torch.cuda.OutOfMemoryError, OverflowError))
            or "CUDA" in str(e)
        )
        if not is_cuda_fail:
            raise
        err = f"{type(e).__name__}: {e}"
        print(f"[bench] sweep point {n_tracks} tracks FAILED: {err}", flush=True)
        row = {
            "mode": "sweep", "batch_tracks": n_tracks, "batch_tokens": None,
            "precision_flags": precision_flags, "env": env,
            "status": "fail", "error": err,
            "vram_gib_torch_peak": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }
        _emit(args, row)
        del gpu_inputs
        torch.cuda.empty_cache()
        return False, row


def run_sweep(args, model, cpu_inputs, precision_flags, env) -> None:
    curve: list[dict] = []
    last_good: int | None = None
    first_fail: int | None = None

    n = args.batch_size
    while True:
        ok, row = _try_sweep_point(args, model, cpu_inputs, n, precision_flags, env)
        curve.append(row)
        if ok:
            last_good = n
            n *= 2
            if args.sweep_max_tracks and n > args.sweep_max_tracks:
                print(f"[bench] sweep stopped at --sweep-max-tracks={args.sweep_max_tracks}", flush=True)
                break
        else:
            first_fail = n
            break

    # Bisect the boundary to <=5 %.
    if first_fail is not None and last_good is not None:
        while (first_fail - last_good) / last_good > 0.05:
            mid = (first_fail + last_good) // 2
            ok, row = _try_sweep_point(args, model, cpu_inputs, mid, precision_flags, env)
            curve.append(row)
            if ok:
                last_good = mid
            else:
                first_fail = mid
        print(f"[bench] boundary: last_good={last_good} first_fail={first_fail}", flush=True)

    good = [r for r in curve if r.get("status") == "ok"]
    if not good:
        print("[bench] sweep: no successful points", flush=True)
        sys.exit(1)
    best = max(good, key=lambda r: r["tracks_per_s"])
    print(
        f"[bench] sweep best: {best['batch_tracks']} tracks -> "
        f"{best['tracks_per_s']:.0f} tracks/s (t2k {best['t2k_ms']:.3f} ms); "
        f"max working batch: {last_good}",
        flush=True,
    )
    print(
        "[bench] recommended confirm:\n"
        f"  pixi run -e default python scripts/perf/bench_variant.py "
        f"--config {args.config} --ckpt {args.ckpt} --variant {args.variant} "
        f"--mode confirm --batch-size {best['batch_tracks']} "
        f"--timed-iters {args.timed_iters} --warmup {args.warmup} "
        f"--out-jsonl {args.out_jsonl} --job-id {args.job_id}_confirm --tag {args.tag}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, type=Path, help="leaf YAML config")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Lightning checkpoint (required unless --random-weights)")
    ap.add_argument("--random-weights", action="store_true",
                    help="skip the checkpoint load: bench the model exactly as "
                         "configured with default random init (timing-only; "
                         "for configs without a trained checkpoint)")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--batch-size", type=int, default=22000, help="tracks per batch")
    ap.add_argument("--timed-iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="e2e: batches to run (0 = warmup+timed)")
    ap.add_argument("--sweep-max-tracks", type=int, default=0, help="sweep: stop doubling past this (0 = until failure)")
    ap.add_argument("--gpu-samples-csv", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--job-id", default="adhoc")
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VAL",
                    help="dotted config override, e.g. "
                         "model.model.init_args.encoder.init_args.chunk_size=32 "
                         "(repeatable; value parsed as YAML)")
    args = ap.parse_args()
    if not args.random_weights and args.ckpt is None:
        ap.error("--ckpt is required unless --random-weights is given")

    precision_flags = common.pin_precision_flags()
    env = common.env_fingerprint()
    print(f"[bench] env: {env}", flush=True)
    print(f"[bench] precision: {precision_flags}", flush=True)

    cfg = common.load_config(args.config)
    for ov in args.overrides:
        key, _, val = ov.partition("=")
        if not _:
            ap.error(f"--set expects KEY=VAL, got {ov!r}")
        common.apply_dotted_override(cfg, key.strip(), val)

    try:
        model = build_model(cfg, args.ckpt, random_weights=args.random_weights)
        model = apply_variant_or_exit(model, args.variant)

        if args.mode == "e2e":
            run_e2e(args, model, cfg, precision_flags, env)
            return

        # staged / confirm / sweep all start from one captured real batch.
        dm = build_datamodule(cfg, args.batch_size, args.num_workers)
        loader = dm.test_dataloader()
        inputs, _targets = next(iter(loader))
        stats = batch_stats(inputs)
        print(f"[bench] captured batch: {stats}", flush=True)

        if args.mode == "sweep":
            run_sweep(args, model, inputs, precision_flags, env)
        else:
            if stats["batch_tracks"] != args.batch_size:
                # Loader may return fewer tracks than requested (shard tail):
                # synthesize the exact requested size for staged/confirm.
                inputs = resize_batch(inputs, args.batch_size)
            run_point(args, model, to_cuda(inputs), args.mode, precision_flags, env)
    except SystemExit:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        _emit(args, {
            "mode": args.mode, "status": "error", "error": f"{type(e).__name__}: {e}",
            "precision_flags": precision_flags, "env": env,
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
