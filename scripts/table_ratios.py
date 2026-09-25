#!/usr/bin/env python3
"""LaTeX rows of the resolution-ratio table (network / truth-seeded KF).

Reads ``<bundle_dir>/<dataset>/matched_residuals.npz`` (from build_residuals.py)
for the four muon samples and prints one row per sample: the clipped-core RMS
ratios of d0, z0, phi, theta, q/p, then the un-clipped RMS ratios.  Tracks are
restricted to |truth eta| <= --eta-max, and the uniform sample additionally to
pT <= --pt-max.

Uncertainties: paired bootstrap -- each replica draws one set of track indices
and reuses it for the network, the reference and all five parameters, with the
clip recomputed inside the replica; the quoted error is the standard deviation
over replicas, printed as one significant digit.

Usage: table_ratios.py <bundle_dir> [--eta-max 2] [--pt-max 70] [--n-boot 400]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

SETS = [("single_muon_2GeV", r"$\mu$, 2\,GeV"),
        ("single_muon_10GeV", r"$\mu$, 10\,GeV"),
        ("single_muon_50GeV", r"$\mu$, 50\,GeV"),
        ("single_muon_uniform", r"$\mu$, 1--70\,GeV")]


def rms3(x, iters=10, tol=1e-4):
    """Clipped RMS: iterate r <- RMS of the entries with |x| <= 3 r until r
    changes by less than tol (relative), at most `iters` times."""
    keep = np.isfinite(x)
    r = np.sqrt(np.mean(x[keep] ** 2))
    for _ in range(iters):
        keep = np.isfinite(x) & (np.abs(x) <= 3.0 * r)
        if keep.sum() < 10:
            break
        new = np.sqrt(np.mean(x[keep] ** 2))
        if abs(new - r) <= tol * r:
            r = new
            break
        r = new
    return r


def _fmt(val: float, err: float | None) -> str:
    """Value with a one-digit uncertainty in brackets, e.g. 0.992(2): the value
    is quoted to the decimal place of that digit (0.0016 -> 0.992(2),
    0.016 -> 0.96(2))."""
    if err is None or not math.isfinite(err) or err <= 0:
        return f"{val:.2f}"
    exp = math.floor(math.log10(err))
    lead = round(err / 10 ** exp)
    if lead == 10:                       # 9.6e-3 -> 1e-2, one place coarser
        lead, exp = 1, exp + 1
    dec = max(0, -exp)
    return f"{val:.{dec}f}({lead})"


def load_residuals(npz: Path, ds: str, eta_max: float, pt_max: float):
    """-> (SSM residuals, KF residuals) as lists over the five parameters, after the cuts."""
    d = np.load(npz)
    truth, ssm, kf = d["truth"], d["ssm"], d["kf"]
    theta = truth[:, 3]
    eta = -np.log(np.tan(np.clip(theta, 1e-9, np.pi - 1e-9) / 2.0))
    m = np.abs(eta) <= eta_max
    if ds == "single_muon_uniform" and np.isfinite(pt_max):
        pt = np.sin(theta) / np.maximum(np.abs(truth[:, 4]), 1e-12)
        m &= pt <= pt_max
    res_s, res_k = [], []
    for j in range(5):
        rs, rk = ssm[m, j] - truth[m, j], kf[m, j] - truth[m, j]
        if j == 2:                                    # phi: wrap
            rs = (rs + np.pi) % (2 * np.pi) - np.pi
            rk = (rk + np.pi) % (2 * np.pi) - np.pi
        res_s.append(np.ascontiguousarray(rs))
        res_k.append(np.ascontiguousarray(rk))
    return res_s, res_k, int(m.sum())


def _ratios(res_s, res_k, idx=None):
    """(clipped, un-clipped) ratio lists for one (re)sample; idx=None uses everything."""
    post, pre = [], []
    for rs, rk in zip(res_s, res_k):
        a, b = (rs, rk) if idx is None else (rs[idx], rk[idx])
        post.append(rms3(a) / rms3(b))
        pre.append(math.sqrt(float(np.mean(a ** 2))) / math.sqrt(float(np.mean(b ** 2))))
    return post, pre


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir", type=Path, help="root holding <dataset>/matched_residuals.npz")
    ap.add_argument("--eta-max", type=float, default=2.0)
    ap.add_argument("--pt-max", type=float, default=70.0, help="pT cap [GeV] of the uniform sample")
    ap.add_argument("--n-boot", type=int, default=400, help="bootstrap replicas (0 = no errors)")
    a = ap.parse_args()
    for ds, label in SETS:
        z = a.bundle_dir / ds / "matched_residuals.npz"
        if not z.exists():
            print(f"% MISSING {ds}")
            continue
        res_s, res_k, n = load_residuals(z, ds, a.eta_max, a.pt_max)
        post, pre = _ratios(res_s, res_k)
        e_post = e_pre = [None] * 5
        if a.n_boot:
            rng = np.random.default_rng(12345)
            bp = np.empty((a.n_boot, 5)); bq = np.empty((a.n_boot, 5))
            for b in range(a.n_boot):
                idx = rng.integers(0, n, n)
                bp[b], bq[b] = _ratios(res_s, res_k, idx)
            e_post, e_pre = bp.std(axis=0, ddof=1), bq.std(axis=0, ddof=1)
        cells = " & ".join(_fmt(v, e) for v, e in zip(post, e_post)) + " & & " + \
                " & ".join(_fmt(v, e) for v, e in zip(pre, e_pre))
        print(f"    {label:<20s} & {cells} \\\\   % N = {n:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
