"""Measure the two pipeline stages the throughput benches EXCLUDE:

  A. host->device (H2D) copy  — pageable vs pinned, across batch sizes
  B. data loading (disk->decode->collate) — wall-clock tracks/s vs num_workers

Puts a rough number (and an optimisation headroom) on the "I/O" that the
GPU-forward-compute figures (e.g. 0.91 M tracks/s) leave out.
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


def batch_bytes(inputs: dict) -> int:
    return sum(v.element_size() * v.nelement() for v in inputs.values() if torch.is_tensor(v))


def time_h2d(inputs_cpu: dict, pinned: bool, reps: int = 50) -> float:
    """Median ms to copy the whole batch dict host->device."""
    src = {k: (v.pin_memory() if pinned and torch.is_tensor(v) else v) for k, v in inputs_cpu.items()}
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps + 5):
        torch.cuda.synchronize()
        s.record()
        _ = {k: (v.to("cuda", non_blocking=pinned) if torch.is_tensor(v) else v) for k, v in src.items()}
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts[5:]))


def bench_h2d(base_inputs: dict) -> None:
    print("\n===== A. HOST -> DEVICE COPY =====")
    print(f"{'tracks':>8} {'tokens':>9} {'MiB':>8} | {'pageable ms':>12} {'GB/s':>7} | "
          f"{'pinned ms':>10} {'GB/s':>7} | {'speedup':>7}")
    for tracks in (2048, 8192, 16384, 32768, 65536, 131072):
        try:
            inp = bv.resize_batch(base_inputs, tracks)
        except Exception as ex:  # noqa: BLE001
            print(f"{tracks:>8}  resize failed: {ex}")
            continue
        nb = batch_bytes(inp)
        toks = bv.batch_stats(inp)["batch_tokens"]
        pg = time_h2d(inp, pinned=False)
        pn = time_h2d(inp, pinned=True)
        gbs_pg = nb / (pg / 1e3) / 1e9
        gbs_pn = nb / (pn / 1e3) / 1e9
        print(f"{tracks:>8} {toks:>9} {nb/2**20:>8.2f} | {pg:>12.4f} {gbs_pg:>7.1f} | "
              f"{pn:>10.4f} {gbs_pn:>7.1f} | {pg/pn:>6.2f}x")


def bench_loader(cfg: dict, batch_size: int, workers_list: list[int], n_batches: int) -> None:
    print("\n===== B. DATA LOADER (disk -> decode -> collate), wall-clock =====")
    print(f"batch_size={batch_size}, timing {n_batches} batches after 5 warmup, pin_memory as configured")
    print(f"{'workers':>7} {'batches':>7} {'wall s':>8} {'tracks/s':>11} {'ms/batch':>9} {'p90 ms':>8}")
    for nw in workers_list:
        try:
            dm = bv.build_datamodule(cfg, batch_size=batch_size, num_workers=nw)
            loader = dm.test_dataloader()
            it = iter(loader)
            # warmup (worker spin-up + first shard open)
            for _ in range(5):
                next(it)
            per, ntr = [], 0
            t0 = time.perf_counter()
            for _ in range(n_batches):
                tb = time.perf_counter()
                inputs, _ = next(it)
                per.append((time.perf_counter() - tb) * 1e3)
                ntr += bv.batch_stats(inputs)["batch_tracks"]
            wall = time.perf_counter() - t0
            print(f"{nw:>7} {n_batches:>7} {wall:>8.2f} {ntr/wall:>11.0f} "
                  f"{np.mean(per):>9.2f} {np.percentile(per,90):>8.2f}")
            del dm, loader, it
        except StopIteration:
            print(f"{nw:>7}  loader exhausted before {n_batches} batches")
        except Exception as ex:  # noqa: BLE001
            print(f"{nw:>7}  FAILED: {type(ex).__name__}: {ex}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path(
        "src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml"))
    ap.add_argument("--capture-batch", type=int, default=8192)
    ap.add_argument("--loader-batch", type=int, default=8192)
    ap.add_argument("--loader-batches", type=int, default=40)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = ap.parse_args()

    common.pin_precision_flags()
    print(f"[io] env: {common.env_fingerprint()}")
    cfg = common.load_config(args.config)

    dm = bv.build_datamodule(cfg, args.capture_batch, num_workers=2)
    base_inputs, _ = next(iter(dm.test_dataloader()))
    st = bv.batch_stats(base_inputs)
    print(f"[io] captured batch for H2D: {st}")
    del dm

    bench_h2d(base_inputs)
    bench_loader(cfg, args.loader_batch, args.workers, args.loader_batches)
    print("\n[io] done")


if __name__ == "__main__":
    main()
