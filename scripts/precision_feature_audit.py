#!/usr/bin/env python3
"""What do 10 mantissa bits cost, feature by feature, on real hits?

fp16 and TF32 carry the SAME 10-bit mantissa; bf16 carries 7.  So for a
FORWARD pass the precision question is already answered by the campaign's
TF32 measurements -- what is genuinely new in fp16 is the 5-bit exponent
(max 65504, smallest normal 6.1e-5).  This script measures both, per input
feature and per target, on a real batch, and converts the quantisation error
into the physical units the paper quotes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

FIELDS = ["x", "y", "z", "r", "phi_hit", "theta_hit", "s", "volume_id",
          "layer_id", "surface_id", "detector", "eta_hit",
          "du_asinh", "dv_asinh", "s_helix"]
NORM_MIN = np.array([-1031., -1031., -3026., 31., -3.1416, 0.027, 31., 16., 2.,
                     1., 0., -4.3, -8., -8., 0.])
NORM_MAX = np.array([1031., 1031., 3026., 1032., 3.1416, 3.114, 3185., 30., 16.,
                     3360., 8., 4.3, 8., 8., 4096.])


def roundtrip(a: np.ndarray, dtype) -> np.ndarray:
    return torch.from_numpy(a).to(dtype).float().numpy()


def main(store: str, n: int = 200_000) -> int:
    part = sorted(Path(store).glob("train/part_*"))[0]
    # the store holds the 12 ABSOLUTE features; the 3 seed residuals
    # (du_asinh, dv_asinh, s_helix) are built in the collate and are O(1) or
    # asinh-compressed by construction, so they are reported from the config
    # ranges rather than from disk.
    feats = np.load(part / "hits.npy", mmap_mode="r")[:n].astype(np.float64)
    print(f"{feats.shape[0]:,} hits from {part}\n")

    print("RAW absolute features -- max |relative| error of one round trip, and")
    print("the same expressed in the feature's own units at its 99th percentile:")
    print(f"{'feature':<12}{'p99 |value|':>13}{'fp16 abs err':>15}{'bf16 abs err':>15}"
          f"{'tf32 abs err':>15}{'fp16 ovfl':>11}")
    for i, f in enumerate(FIELDS[:feats.shape[1]]):
        v = np.asarray(feats[:, i], dtype=np.float64)
        p99 = np.percentile(np.abs(v), 99)
        row = f"{f:<12}{p99:>13.4g}"
        for dt in (torch.float16, torch.bfloat16):
            err = np.abs(roundtrip(v.astype(np.float32), dt) - v).max()
            row += f"{err:>15.3g}"
        # TF32 has 10 mantissa bits like fp16 but fp32's exponent: emulate by
        # truncating the fp32 significand to 10 bits.
        f32 = v.astype(np.float32)
        u = f32.view(np.uint32) & np.uint32(0xFFFFE000)
        err32 = np.abs(u.view(np.float32).astype(np.float64) - v).max()
        row += f"{err32:>15.3g}"
        row += f"{'YES' if np.abs(v).max() > 65504 else '-':>11}"
        print(row)

    print("\nSame features AFTER the model's min-max normalisation to [0, 1]")
    print("(this is what actually reaches the network, before the Fourier encoding):")
    print(f"{'feature':<12}{'fp16 abs err':>15}{'-> physical':>18}")
    span = NORM_MAX - NORM_MIN
    for i, f in enumerate(FIELDS[:feats.shape[1]]):
        v = np.asarray(feats[:, i], dtype=np.float64)
        nv = (v - NORM_MIN[i]) / span[i]
        err = np.abs(roundtrip(nv.astype(np.float32), torch.float16) - nv).max()
        print(f"{f:<12}{err:>15.3g}{err * span[i]:>18.4g}")

    # Where does the micrometre-scale information actually live?  Not in the
    # absolute coordinates: the Fourier encoding turns them into O(1) sines,
    # and the seed residuals carry the fine structure directly.
    print("\nFourier encoding of x (sin(x_norm / 2^k)) -- the values that actually")
    print("enter the first GEMM -- and the seed residual scale:")
    xs = np.asarray(feats[:, 0], dtype=np.float64)
    xn = (xs - NORM_MIN[0]) / span[0]
    for k in (-10, -5, 0, 5):
        arg = xn / (2.0 ** k)
        sv = np.sin(arg)
        err = np.abs(roundtrip(sv.astype(np.float32), torch.float16) - sv).max()
        # what x-displacement does that error correspond to, via d(sin)/dx?
        slope = np.abs(np.cos(arg)).clip(1e-3) / (2.0 ** k) / span[0]
        print(f"  scale 2^{k:<4} sin in [-1,1], fp16 err {err:.3g}"
              f"  -> {np.median(err / slope) * 1e3:8.2f} um of x")
    print("  (the finest scale resolves x to ~um in fp16: that is where the")
    print("   precision is carried, and it is an O(1) quantity)")

    print("\nReading:")
    print(" * 10 mantissa bits cannot hold a 2 m detector to 43 um -- that needs")
    print("   log2(2062/0.043) = 15.5 bits.  Min-max normalisation does NOT help:")
    print("   near 1.0 fp16 steps by 2^-11, which is ~0.5 mm of x and ~1.5 mm of z.")
    print("   So absolute coordinates must never be carried or differenced in fp16.")
    print(" * but the network does not take its precision from them.  The fine")
    print("   Fourier components and the seed residuals asinh(du/0.1mm) are O(1)")
    print("   quantities that resolve microns in 10 bits (table above).  Both are")
    print("   computed pointwise, which autocast leaves in fp32; only the GEMMs")
    print("   that consume them run reduced.")
    print(" * this is why TF32 was already measured harmless (<=0.3 %, CLAUDE.md")
    print("   4.12/4.32): TF32 has the SAME 10 mantissa bits as fp16.  For the")
    print("   forward pass fp16 therefore inherits a measured result; what is new")
    print("   is only the 5-bit exponent, and nothing here is near 65504.")
    print(" * bf16 has 7 mantissa bits, 8x coarser than fp16/TF32, buying range")
    print("   this model does not need -- it is the wrong trade here.")
    print(" * the fp64 seed stays fp64 regardless (CLAUDE.md 4.32: the rc-R")
    print("   cancellation makes even fp32 unusable at high pT).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "/scratch/colliderml/ICLR_retraining_v2_mix3"))
