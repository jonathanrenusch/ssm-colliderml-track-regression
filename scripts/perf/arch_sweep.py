"""Architecture-scaling inference sweep for the short-sequence kernel (v5pc).

Question (user, 2026-07-25): with the NEW fused kernel, which axis of added
model complexity — depth (num_layers), width (dim), or SSM state (d_state) —
costs the most inference time?  We want an empirical, variance-aware answer to
guide where physics capacity can be bought cheaply.

Method
------
* One real packed batch is captured once, resized to a FIXED track count, and
  reused for every architecture (the raw hit inputs are architecture-independent),
  so per-track time is directly comparable across points.
* For each grid point we build the model with RANDOM weights (timing only; we do
  NOT retrain and inference speed is weight-value-independent), apply the
  production inference kernel v5pc (fall back to v5p if v5pc's torch.compile glue
  fails for an exotic shape), then warm up and time ``--timed-iters`` forward
  passes with per-iter CUDA events.  Rich per-iter statistics (mean, std, 95% CI
  of the mean, median, p10/p90) let variance be accounted for.
* Results are appended to JSONL+CSV after every point, so a crash/kill loses at
  most the in-flight point.

This reuses ``bench_variant`` wholesale for build/apply/resize/timing so the
numbers are on the exact same footing as the campaign's headline benches.
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.ensure_src_on_path()
import bench_variant as bv  # noqa: E402


BASELINE = dict(L=4, D=128, N=16, HD=32, G=1)


def build_grid() -> list[dict]:
    """One-axis-at-a-time from the 4L/dim128/state16 baseline + cross points."""
    pts: list[tuple] = []

    def add(L, D, N, HD=32, G=1):
        pts.append((L, D, N, HD, G))

    add(**BASELINE)  # baseline
    # depth axis (num_layers)
    for L in (1, 2, 3, 6, 8, 10, 12, 16, 20):
        add(L, 128, 16)
    # width axis (dim); dim must be divisible by 16 so 2*dim/headdim is integer
    for D in (64, 96, 160, 192, 256, 320, 384, 512):
        add(4, D, 16)
    # SSM state axis (d_state)
    for N in (8, 24, 32, 48, 64, 96, 128):
        add(4, 128, N)
    # head dimension (fixed dim -> changes nheads)
    for HD in (16, 64, 128):
        add(4, 128, 16, HD=HD)
    # ngroups (B/C sharing) — must divide nheads (= 2*128/32 = 8)
    for G in (2, 4, 8):
        add(4, 128, 16, G=G)
    # cross points (interaction between axes; incl. paper 10L shape)
    for c in [(10, 192, 32), (10, 128, 16), (10, 192, 16), (10, 192, 64),
              (8, 256, 32), (6, 192, 64), (12, 128, 64), (4, 384, 64), (8, 192, 32)]:
        add(*c)

    seen, grid = set(), []
    for (L, D, N, HD, G) in pts:
        key = (L, D, N, HD, G)
        if key in seen:
            continue
        seen.add(key)
        d_inner = 2 * D
        if d_inner % HD != 0:
            continue                      # nheads must be integer
        nheads = d_inner // HD
        if nheads % G != 0:
            continue                      # ngroups must divide nheads
        grid.append(dict(L=L, D=D, N=N, HD=HD, G=G, nheads=nheads, d_inner=d_inner))
    return grid


def override_cfg(base_cfg: dict, pt: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    mi = cfg["model"]["model"]["init_args"]
    mi["dim"] = pt["D"]                    # top-level trunk width
    enc = mi["encoder"]["init_args"]
    enc["num_layers"] = pt["L"]
    enc["dim"] = pt["D"]
    enc["d_state"] = pt["N"]
    enc["headdim"] = pt["HD"]
    enc["ngroups"] = pt["G"]
    return cfg


def rich_stats(times_ms: list[float], n_tracks: int) -> dict:
    t = np.asarray(times_ms, dtype=np.float64)
    n = t.size
    mean = float(t.mean())
    std = float(t.std(ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    tps = n_tracks / (mean / 1e3)
    return {
        "n_timed": int(n),
        "t_iter_ms_mean": round(mean, 5),
        "t_iter_ms_std": round(std, 5),
        "t_iter_ms_ci95": round(ci95, 5),
        "t_iter_ms_cv_pct": round(100.0 * std / mean, 3) if mean else 0.0,
        "t_iter_ms_median": round(float(np.median(t)), 5),
        "t_iter_ms_p10": round(float(np.percentile(t, 10)), 5),
        "t_iter_ms_p90": round(float(np.percentile(t, 90)), 5),
        "t_iter_ms_min": round(float(t.min()), 5),
        "tracks_per_s": round(tps, 1),
        "us_per_track": round(1e6 * (mean / 1e3) / n_tracks, 4),
        "t2k_ms": round(2000.0 * 1000.0 / tps, 5),
    }


def apply_kernel(model, primary: str):
    """Apply v5pc; on failure fall back to v5p. Returns (model, variant_used)."""
    from track_regression.mamba_short import apply_variant
    try:
        return apply_variant(model, primary) or model, primary
    except Exception as e:  # noqa: BLE001
        print(f"[arch] {primary} failed ({type(e).__name__}: {e}); trying v5p", flush=True)
        torch._dynamo.reset()
        return apply_variant(model, "v5p") or model, "v5p"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path(
        "src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml"))
    ap.add_argument("--variant", default="v5pc")
    ap.add_argument("--target-tracks", type=int, default=16384)
    ap.add_argument("--capture-batch", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed-iters", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("docs/perf/results/night_arch/arch_sweep.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="cap number of grid points (0 = all; for smoke tests)")
    args = ap.parse_args()

    precision = common.pin_precision_flags()
    env = common.env_fingerprint()
    print(f"[arch] env: {env}", flush=True)
    print(f"[arch] precision: {precision}", flush=True)

    base_cfg = common.load_config(args.config)

    # Capture one real batch, resize once to the fixed track count, move to GPU.
    dm = bv.build_datamodule(base_cfg, args.capture_batch, num_workers=1)
    inputs, _ = next(iter(dm.test_dataloader()))
    inputs = bv.resize_batch(inputs, args.target_tracks)
    gpu_inputs = bv.to_cuda(inputs)
    stats = bv.batch_stats(gpu_inputs)
    print(f"[arch] fixed batch: {stats}", flush=True)

    grid = build_grid()
    if args.limit > 0:
        grid = grid[: args.limit]
    print(f"[arch] {len(grid)} architectures to sweep (target {args.target_tracks} tracks, "
          f"{args.warmup} warmup + {args.timed_iters} timed iters each)", flush=True)

    for i, pt in enumerate(grid):
        tag = f"L{pt['L']}_D{pt['D']}_N{pt['N']}_HD{pt['HD']}_G{pt['G']}"
        row = {
            "ts": common.utc_ts(), "idx": i, "tag": tag,
            "L": pt["L"], "dim": pt["D"], "d_state": pt["N"], "headdim": pt["HD"],
            "ngroups": pt["G"], "nheads": pt["nheads"], "d_inner": pt["d_inner"],
            "target_tracks": stats["batch_tracks"], "batch_tokens": stats["batch_tokens"],
            "len_mean": stats["len_mean"], "warmup": args.warmup,
            "requested_variant": args.variant, "env": env, "precision_flags": precision,
        }
        print(f"\n[arch] === {i+1}/{len(grid)}  {tag} "
              f"(nheads={pt['nheads']}, d_inner={pt['d_inner']}) ===", flush=True)
        model = None
        try:
            torch.cuda.reset_peak_memory_stats()
            cfg = override_cfg(base_cfg, pt)
            model = bv.build_model(cfg, None, random_weights=True)
            row["n_params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 4)
            model, used = apply_kernel(model, args.variant)
            row["variant_used"] = used
            times = bv.timed_forward_loop(model, gpu_inputs, args.warmup, args.timed_iters)
            row.update(rich_stats(times, stats["batch_tracks"]))
            row["status"] = "ok"
            row["error"] = ""
            row["vram_gib_peak"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            print(f"[arch] RESULT {tag}: t2k={row['t2k_ms']}ms "
                  f"tracks/s={row['tracks_per_s']:.0f} "
                  f"CV={row['t_iter_ms_cv_pct']}% params={row['n_params_m']}M "
                  f"[{used}]", flush=True)
        except Exception as e:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            row["vram_gib_peak"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            print(f"[arch] FAILED {tag}: {row['error']}", flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        finally:
            common.append_jsonl(args.out, row)
            common.append_csv(Path(args.out).with_suffix(".csv"), row)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch._dynamo.reset()
            except Exception:  # noqa: BLE001
                pass

    print(f"\n[arch] sweep complete -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
