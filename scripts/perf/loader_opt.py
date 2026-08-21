"""Optimise the input pipeline until it stops being the bottleneck.

Part 1  sweep the loader knobs (batch, workers, prefetch, pin_memory) the
        datamodule already exposes, maximising SUSTAINED tracks/s.
Part 2  TRUE end-to-end: iterate the winning loader, copy each batch H2D and run
        the v5pc forward, real wall-clock (loader stalls INCLUDED, overlapped
        with compute via prefetch + async copies). Compare to compute-only.

Measurement is TIME-BOUNDED with a timed warm-up: the DataLoader prefetch queue
(workers x prefetch_factor batches) is pre-filled during spin-up, so a short
fixed-batch window just drains it at memory speed and reports a fantasy rate.
We instead pull-and-discard for ``warmup_s`` (drains the initial queue and
reaches steady state) then count tracks for ``measure_s`` (worker-production
limited). Iterator is recreated on exhaustion so the window always fills.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
common.ensure_src_on_path()
import bench_variant as bv  # noqa: E402
from track_regression.data import ColliderMLRegrDataModule  # noqa: E402


def make_loader(cfg, batch_size, num_workers, pin_memory, prefetch_factor):
    dc = dict(cfg["data"])
    dc.update(batch_size=batch_size, num_workers=num_workers,
              pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    dm = ColliderMLRegrDataModule(**dc)
    dm.setup("test")
    return dm, dm.test_dataloader()


class Feeder:
    """Endless batch source (recreates the iterator on exhaustion)."""
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def sustained_tps(loader, warmup_s, measure_s, consume=None):
    """Steady-state tracks/s. ``consume`` optionally runs per batch (e.g. forward)."""
    f = Feeder(loader)
    t_end = time.perf_counter() + warmup_s
    while time.perf_counter() < t_end:          # drain prefill + reach steady state
        b, _ = f.next()
        if consume:
            consume(b)
    if consume:
        torch.cuda.synchronize()
    ntr = 0
    t0 = time.perf_counter()
    t_end = t0 + measure_s
    while time.perf_counter() < t_end:
        b, _ = f.next()
        if consume:
            consume(b)
        ntr += int(b["track_lengths"].numel())
    if consume:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return ntr / wall, ntr, wall


def sweep(cfg, warmup_s, measure_s):
    print("\n===== PART 1 · loader knob sweep (SUSTAINED wall-clock tracks/s) =====")
    best = None
    print("-- stage 1: batch x workers (pin=True, prefetch=2) --")
    print(f"{'batch':>7} {'workers':>7} {'tracks/s':>11}")
    for bs in (8192, 32768):
        for nw in (8, 16, 32, 48, 64):
            try:
                dm, loader = make_loader(cfg, bs, nw, True, 2)
                tps, _, _ = sustained_tps(loader, warmup_s, measure_s)
                print(f"{bs:>7} {nw:>7} {tps:>11,.0f}", flush=True)
                if best is None or tps > best[0]:
                    best = (tps, bs, nw, True, 2)
                del dm, loader
            except Exception as ex:  # noqa: BLE001
                print(f"{bs:>7} {nw:>7}  FAILED: {type(ex).__name__}: {ex}", flush=True)
    _, bbs, bnw, _, _ = best
    print(f"-- stage 2: prefetch x pin at batch={bbs}, workers={bnw} --")
    print(f"{'prefetch':>8} {'pin':>6} {'tracks/s':>11}")
    for pf in (2, 4, 8):
        for pin in (True, False):
            try:
                dm, loader = make_loader(cfg, bbs, bnw, pin, pf)
                tps, _, _ = sustained_tps(loader, warmup_s, measure_s)
                print(f"{pf:>8} {str(pin):>6} {tps:>11,.0f}", flush=True)
                if tps > best[0]:
                    best = (tps, bbs, bnw, pin, pf)
                del dm, loader
            except Exception as ex:  # noqa: BLE001
                print(f"{pf:>8} {str(pin):>6}  FAILED: {type(ex).__name__}: {ex}", flush=True)
    print(f"\n>>> best loader: batch={best[1]} workers={best[2]} pin={best[3]} "
          f"prefetch={best[4]}  ->  {best[0]:,.0f} tracks/s", flush=True)
    return best


def end_to_end(cfg, best, ckpt, warmup_s, measure_s):
    tps_l, bs, nw, pin, pf = best
    print("\n===== PART 2 · true end-to-end wall-clock (loader + H2D + v5pc) =====")
    model = bv.build_model(cfg, ckpt, random_weights=(ckpt is None))
    from track_regression.mamba_short import apply_variant
    apply_variant(model, "v5pc")

    dm, loader = make_loader(cfg, bs, nw, pin, pf)
    ref, _ = next(iter(loader))
    ref_gpu = bv.to_cuda(ref)
    comp_ms = float(np.median(bv.timed_forward_loop(model, ref_gpu, 40, 200)))
    comp_tps = int(ref["track_lengths"].numel()) / (comp_ms / 1e3)
    print(f"compute-only @batch{bs}: {comp_ms:.3f} ms/batch -> {comp_tps:,.0f} tracks/s", flush=True)

    def consume(b):
        with torch.inference_mode():
            model({k: v.to("cuda", non_blocking=pin) for k, v in b.items() if torch.is_tensor(v)})

    e2e_tps, ntr, wall = sustained_tps(loader, warmup_s, measure_s, consume=consume)
    frac = 100.0 * e2e_tps / comp_tps
    print(f"end-to-end wall-clock: {ntr:,} tracks in {wall:.1f}s -> {e2e_tps:,.0f} tracks/s", flush=True)
    print(f"loader standalone (best): {tps_l:,.0f} tracks/s", flush=True)
    print(f"\n>>> end-to-end = {frac:.0f}% of compute-only "
          f"({'COMPUTE-BOUND' if frac >= 90 else 'still input-bound'})", flush=True)
    return dict(batch=bs, workers=nw, pin=bool(pin), prefetch=pf,
                loader_tps=round(tps_l), compute_tps=round(comp_tps),
                e2e_tps=round(e2e_tps), pct_of_compute=round(frac, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path(
        "src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml"))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--warmup-s", type=float, default=5.0)
    ap.add_argument("--measure-s", type=float, default=8.0)
    ap.add_argument("--e2e-warmup-s", type=float, default=8.0)
    ap.add_argument("--e2e-measure-s", type=float, default=15.0)
    ap.add_argument("--only-e2e", action="store_true",
                    help="skip the sweep; run a small e2e grid of overlap-friendly configs")
    args = ap.parse_args()

    common.pin_precision_flags()
    print(f"[loader-opt] env: {common.env_fingerprint()}", flush=True)
    cfg = common.load_config(args.config)

    if args.only_e2e:
        # e2e wants async H2D overlap -> pin=True + non_blocking=True. Standalone
        # loader may be a touch slower pinned, but it decides whether the GPU
        # ever waits. Grid a few worker/prefetch combos; loader tps re-measured.
        print("\n===== e2e grid (pin=True, async overlap) · batch sweep @16 workers =====")
        results = []
        for bs in (32768, 65536, 131072):
            for nw in (16,):
                pf = 8
                dm, loader = make_loader(cfg, bs, nw, True, pf)
                ltps, _, _ = sustained_tps(loader, 4, 6)
                del dm, loader
                s = end_to_end(cfg, (ltps, bs, nw, True, pf), args.ckpt,
                               args.e2e_warmup_s, args.e2e_measure_s)
                results.append(s)
                print(f"[grid] batch={bs} workers={nw}: e2e {s['e2e_tps']:,} "
                      f"({s['pct_of_compute']}% of compute)\n", flush=True)
        best = max(results, key=lambda r: r["e2e_tps"])
        print(f"\n[loader-opt] BEST E2E: {best}", flush=True)
        return

    best = sweep(cfg, args.warmup_s, args.measure_s)
    summary = end_to_end(cfg, best, args.ckpt, args.e2e_warmup_s, args.e2e_measure_s)
    print(f"\n[loader-opt] SUMMARY: {summary}", flush=True)


if __name__ == "__main__":
    main()
