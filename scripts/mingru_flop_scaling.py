#!/usr/bin/env python3
"""Does deployed minGRU throughput actually scale with projection FLOPs?

The claim underpinning every "cheaper backbone" idea is that ~all of the cost
is the dense in-projections.  If that is true, throughput should track
1 / FLOPs as the hidden width changes.  If it is flat, the kernel is launch- or
bandwidth-bound and cutting FLOPs buys nothing -- which would kill the
structured-projection idea before it costs a GPU-day.

Measured with the packed inference kernel on random packed batches (physics is
irrelevant here; only shapes are).  Run it with both GPUs busy if need be --
every variant is timed under the same contention, back to back, interleaved.
"""
from __future__ import annotations

import os
import sys
import time

import torch

from track_regression.mingru import MinGRUCLSEncoder

D = 128
MEAN_HITS = 13.3


def make_batch(n_tracks: int, dev, gen):
    lens = torch.randint(6, 21, (n_tracks,), device=dev, generator=gen)
    cu = torch.zeros(n_tracks + 1, dtype=torch.int32, device=dev)
    cu[1:] = lens.cumsum(0).to(torch.int32)
    x = torch.randn(1, int(cu[-1]), D, device=dev, generator=gen)
    return x, cu


def bench(enc, x, cu, iters=40):
    with torch.no_grad():
        for _ in range(8):
            enc(x, cu_seqlens=cu)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            enc(x, cu_seqlens=cu)
        torch.cuda.synchronize()
    return (time.time() - t) / iters


def main(n_tracks: int = 32_000, reps: int = 3) -> int:
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    x, cu = make_batch(n_tracks, dev, gen)
    widths = [int(w) for w in (os.environ.get('WIDTHS') or '97,137,194,274,388').split(',')]
    encs, flops = {}, {}
    for h in widths:
        e = MinGRUCLSEncoder(dim=D, hidden_size=h, num_layers=2,
                             compile_core=False).to(dev).eval()
        encs[h] = e
        p = sum(q.numel() for n, q in e.named_parameters() if "pool" not in n)
        flops[h] = 2 * p * MEAN_HITS
    best = {h: 1e9 for h in widths}
    for _ in range(reps):                       # interleave, keep the best
        for h in widths:
            best[h] = min(best[h], bench(encs[h], x, cu))

    print(f"packed inference, {n_tracks:,} tracks/batch, TF32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"{'hidden':>7}{'enc params':>12}{'MFLOP/trk':>11}{'ms':>9}"
          f"{'M tracks/s':>12}{'FLOP ratio':>12}{'speed ratio':>13}")
    ref_h = widths[len(widths) // 2]
    for h in widths:
        ms = best[h] * 1e3
        tps = n_tracks / best[h] / 1e6
        p = sum(q.numel() for n, q in encs[h].named_parameters() if "pool" not in n)
        print(f"{h:>7}{p:>12,}{flops[h]/1e6:>11.2f}{ms:>9.2f}{tps:>12.3f}"
              f"{flops[ref_h]/flops[h]:>12.2f}{best[ref_h]/best[h]:>13.2f}")
    print("\nIf 'speed ratio' tracks 'FLOP ratio', the model is projection-bound and")
    print("cheaper projections pay.  If it is much flatter, it is launch/bandwidth")
    print("bound and only fewer TOKENS or fewer LAUNCHES help.")
    return 0


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 32_000))
