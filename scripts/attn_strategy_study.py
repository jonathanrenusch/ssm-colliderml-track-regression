#!/usr/bin/env python3
"""Is FlashAttention the right tool at L ~ 20?  (referee-facing study)

FlashAttention exists to avoid materialising the L x L score matrix for LONG
sequences.  A track has <= 20 hits (+2 CLS), so that matrix is 22 x 22 per head
-- 484 numbers, register-resident.  The tiling, online softmax and varlen
index plumbing are then pure overhead.  This measures the attention step alone,
at the trained transformer's exact shapes, across the strategies available:

  dense-padded-sdpa   (B, Lmax, h, dh) + key mask, torch SDPA  [what we run]
  dense-padded-math   same, but the explicit QK^T / softmax / AV -- "simple
                      full attention", no kernel cleverness at all
  flash-sdpa-padded   torch SDPA forced onto its flash backend
  flash-varlen        flash_attn_varlen_func on the packed stream
  dense-packed-mask   one dense block-diagonal mask over the packed stream
                      (the O((B*L)^2) trap; only run at small B)

Reported per token so batch sizes are comparable.
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

H, DH, LMAX = 4, 32, 22          # heads, head dim, 20 hits + 2 CLS
MEAN_L = 13.2 + 2


def _lens(B, dev, gen):
    return torch.randint(6, LMAX - 1, (B,), device=dev, generator=gen) + 2


def _timeit(fn, n=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


def main(dtype_s: str = "float16") -> int:
    dev = "cuda"
    dt = {"float16": torch.float16, "float32": torch.float32}[dtype_s]
    gen = torch.Generator(device=dev).manual_seed(0)
    print(f"attention step only, heads={H} head_dim={DH} Lmax={LMAX} dtype={dtype_s}")
    print(f"{'strategy':<22}" + "".join(f"{f'B={b//1000}k':>12}" for b in (8192, 32768, 131072)))
    rows: dict[str, list] = {}

    for B in (8192, 32768, 131072):
        lens = _lens(B, dev, gen)
        T = int(lens.sum())
        cu = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        cu[1:] = lens.cumsum(0).to(torch.int32)
        q = torch.randn(B, LMAX, H, DH, device=dev, dtype=dt, generator=gen)
        k = torch.randn_like(q); v = torch.randn_like(q)
        pos = torch.arange(LMAX, device=dev)
        keep = pos.unsqueeze(0) < lens.unsqueeze(1)                      # (B, Lmax)
        bias = torch.zeros(B, 1, 1, LMAX, device=dev, dtype=dt)
        bias.masked_fill_(~keep[:, None, None, :], float("-inf"))
        qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))              # (B,H,L,dh)

        qp = torch.randn(T, H, DH, device=dev, dtype=dt, generator=gen)
        kp, vp = torch.randn_like(qp), torch.randn_like(qp)

        cand = {}
        cand["dense-padded-sdpa"] = lambda: F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=bias)

        def dense_math():
            s = (qh @ kh.transpose(-1, -2)) * (DH ** -0.5) + bias
            return torch.softmax(s, -1) @ vh
        cand["dense-padded-math"] = dense_math

        def flash_sdpa():
            with torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    qh, kh, vh, attn_mask=None)
        cand["flash-sdpa-padded"] = flash_sdpa

        if dt is torch.float16:
            try:
                from flash_attn import flash_attn_varlen_func
                ml = int(lens.max())
                cand["flash-varlen"] = lambda: flash_attn_varlen_func(
                    qp, kp, vp, cu, cu, ml, ml)
            except Exception as e:  # noqa: BLE001
                print(f"  (flash-varlen unavailable: {type(e).__name__})")

        for name, fn in cand.items():
            try:
                ms = _timeit(fn) * 1e3
                rows.setdefault(name, []).append(ms / (T / 1e6))   # us per 1e6 tok -> ns/token
            except Exception as e:  # noqa: BLE001
                rows.setdefault(name, []).append(float("nan"))
                print(f"  {name} @B={B}: {type(e).__name__}: {str(e)[:70]}")
        del q, k, v, qh, kh, vh, qp, kp, vp, bias
        torch.cuda.empty_cache()

    for name, vals in rows.items():
        print(f"{name:<22}" + "".join(f"{x:>11.1f}µ" for x in vals))
    print("\nunits: microseconds per 1e6 tokens (lower is better)")
    best = min((v[-1], n) for n, v in rows.items() if v[-1] == v[-1])
    print(f"fastest at the largest batch: {best[1]} ({best[0]:.1f} µs/1e6 tok)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "float16"))
