#!/usr/bin/env python3
"""LaTeX rows for the paper's tab:ratios, from a truth-KF residual bundle.

    paper_tab_ratios.py <bundle_dir> [eta_max] [pt_max_uniform]

Reads <bundle>/<dataset>/matched_residuals.npz (keys truth/ssm/kf, each
(N,5) = d0, z0, phi, theta, q/p) and prints one row per sample with the
clipped-core and un-clipped RMS ratios SSM / truth-KF, i.e. exactly the two
halves of the table.  The fiducial |eta| <= 2 comes from the truth theta and
the uniform row additionally takes pT <= 70 GeV (CLAUDE.md 4.35), both derived
from the truth columns so no extra side-car is needed.
"""
from __future__ import annotations

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


def main(bundle: Path, eta_max: float, pt_max: float) -> int:
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
        post, pre = [], []
        for j in range(5):
            rs, rk = ssm[m, j] - truth[m, j], kf[m, j] - truth[m, j]
            if j == 2:                                    # phi: wrap
                rs = (rs + np.pi) % (2 * np.pi) - np.pi
                rk = (rk + np.pi) % (2 * np.pi) - np.pi
            post.append(rms3(rs) / rms3(rk))
            pre.append(np.sqrt(np.mean(rs ** 2)) / np.sqrt(np.mean(rk ** 2)))
        cells = " & ".join(f"{v:.2f}" for v in post) + " & & " + \
                " & ".join(f"{v:.2f}" for v in pre)
        print(f"    {label:<20s} & {cells} \\\\   % N = {int(m.sum()):,}")
    return 0


if __name__ == "__main__":
    a = sys.argv
    sys.exit(main(Path(a[1]), float(a[2]) if len(a) > 2 else 2.0,
                  float(a[3]) if len(a) > 3 else 70.0))
