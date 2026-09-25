#!/usr/bin/env python3
"""LaTeX rows of the encoder-ablation table, with bootstrap uncertainties.

Each encoder arm is given as a residual bundle root (build_residuals.py output,
holding <dataset>/matched_residuals.npz).  Rows are clipped-core RMS ratios to
the truth-seeded KF for d0, z0, phi, theta, q/p and their geometric mean, on
|truth eta| <= 2 (and pT <= 70 GeV on the uniform sample).  Uncertainties are
the paired bootstrap of table_ratios.py.

    main      the uniform-momentum sample only, one row per encoder;
    appendix  the same for every muon test sample.

Usage:
  table_encoder_ablation.py --mingru DIR --mamba2 DIR --transformer DIR --diagssm DIR
                            [--mode main|appendix] [--n-boot 400]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_ratios import SETS, _fmt, load_residuals, rms3  # noqa: E402

ARMS = [
    ("mingru", r"minGRU"),
    ("mamba2", r"Mamba-2, bidirectional"),
    ("transformer", r"Transformer"),
    ("diagssm", r"diagonal SSM, non-selective"),
]
ETA_MAX, PT_MAX = 2.0, 70.0


def load(bundle: Path, ds: str):
    """-> list of (SSM residual, KF residual) per parameter, and the track count."""
    res_s, res_k, n = load_residuals(bundle / ds / "matched_residuals.npz", ds, ETA_MAX, PT_MAX)
    return list(zip(res_s, res_k)), n


def row(pairs, n, n_boot):
    """Point ratios + their geometric mean, and the bootstrap errors."""
    def once(idx=None):
        r = [rms3(a if idx is None else a[idx]) / rms3(b if idx is None else b[idx])
             for a, b in pairs]
        return r + [math.exp(sum(math.log(v) for v in r) / 5)]
    val = once()
    err = [None] * len(val)
    if n_boot:
        rng = np.random.default_rng(12345)
        reps = np.array([once(rng.integers(0, n, n)) for _ in range(n_boot)])
        err = list(reps.std(axis=0, ddof=1))
    return " & ".join(_fmt(v, e) for v, e in zip(val, err))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, label in ARMS:
        ap.add_argument(f"--{key}", type=Path, required=True, help=f"residual bundle root, {label}")
    ap.add_argument("--mode", choices=["main", "appendix"], default="main")
    ap.add_argument("--n-boot", type=int, default=400, help="bootstrap replicas (0 = no errors)")
    a = ap.parse_args()
    arms = [(getattr(a, key), label) for key, label in ARMS]
    if a.mode == "main":
        for bundle, label in arms:
            pairs, n = load(bundle, "single_muon_uniform")
            print(f"    {label:<30s} & {row(pairs, n, a.n_boot)} \\\\   % N = {n:,}", flush=True)
        return 0
    for ds, dlabel in SETS:
        print(f"    \\multicolumn{{7}}{{l}}{{\\itshape {dlabel}}} \\\\")
        for bundle, label in arms:
            pairs, n = load(bundle, ds)
            print(f"    \\quad {label:<26s} & {row(pairs, n, a.n_boot)} \\\\", flush=True)
        print(r"    \addlinespace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
