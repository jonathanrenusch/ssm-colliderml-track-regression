#!/usr/bin/env python3
"""LaTeX rows for the paper's tab:ratios, from a truth-KF residual bundle.

    paper_tab_ratios.py <bundle_dir> [eta_max] [pt_max_uniform] [n_boot]

Reads <bundle>/<dataset>/matched_residuals.npz (keys truth/ssm/kf, each
(N,5) = d0, z0, phi, theta, q/p) and prints one row per sample with the
clipped-core and un-clipped RMS ratios SSM / truth-KF, i.e. exactly the two
halves of the table.  The fiducial |eta| <= 2 comes from the truth theta and
the uniform row additionally takes pT <= 70 GeV (CLAUDE.md 4.35), both derived
from the truth columns so no extra side-car is needed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SETS = [("single_muon_2GeV", r"$\mu$, 2\,GeV"),
        ("single_muon_10GeV", r"$\mu$, 10\,GeV"),
        ("single_muon_50GeV", r"$\mu$, 50\,GeV"),
        ("single_muon_uniform", r"$\mu$, 1--70\,GeV")]


def rms3(x, iters=10, tol=1e-4):
    """iterative 3-sigma clipped RMS (the campaign's estimator)."""
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
    """0.992(3): value to three decimals, 1-sigma error on the last digit."""
    if err is None:
        return f"{val:.2f}"
    d = max(1, min(99, int(math.ceil(err * 1000 - 1e-9))))
    return f"{val:.3f}({d})" if d < 10 else f"{val:.3f}({d})"


def _ratios(res_s, res_k, idx=None):
    """(post, pre) ratio lists for one (re)sample; idx=None uses everything."""
    post, pre = [], []
    for rs, rk in zip(res_s, res_k):
        a, b = (rs, rk) if idx is None else (rs[idx], rk[idx])
        post.append(rms3(a) / rms3(b))
        pre.append(math.sqrt(float(np.mean(a ** 2))) / math.sqrt(float(np.mean(b ** 2))))
    return post, pre


def main(bundle: Path, eta_max: float, pt_max: float, n_boot: int = 0) -> int:
    for ds, label in SETS:
        z = bundle / ds / "matched_residuals.npz"
        if not z.exists():
            print(f"% MISSING {ds}")
            continue
        d = np.load(z)
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
        post, pre = _ratios(res_s, res_k)
        e_post = e_pre = [None] * 5
        if n_boot:
            n = int(m.sum())
            rng = np.random.default_rng(12345)
            bp = np.empty((n_boot, 5)); bq = np.empty((n_boot, 5))
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                bp[b], bq[b] = _ratios(res_s, res_k, idx)
            e_post, e_pre = bp.std(axis=0, ddof=1), bq.std(axis=0, ddof=1)
        cells = " & ".join(_fmt(v, e) for v, e in zip(post, e_post)) + " & & " + \
                " & ".join(_fmt(v, e) for v, e in zip(pre, e_pre))
        print(f"    {label:<20s} & {cells} \\\\   % N = {int(m.sum()):,}")
    return 0


if __name__ == "__main__":
    a = sys.argv
    sys.exit(main(Path(a[1]), float(a[2]) if len(a) > 2 else 2.0,
                  float(a[3]) if len(a) > 3 else 70.0,
                  int(a[4]) if len(a) > 4 else 0))
