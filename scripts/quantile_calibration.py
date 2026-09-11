#!/usr/bin/env python3
"""Quantile-calibration page for one prediction file (reviewer request 2026-09-11).

The eval h5 stores every head's ORDERED, DENORMALISED 7-quantile ladder in
DELTA space (`/quantiles/<p>` (N, 7) = `predict_quantiles`, i.e. the physical
delta to the seed anchor, monotone-reconstructed; for the scale-free q/p head
the delta is still missing its per-track scale (|seed|+eps)).  The anchors are
not stored, but coverage needs none of them:

- linear heads (d0, z0, theta): the physical quantile is q_j + anchor and the
  physical median prediction is q_med + anchor, so
  P(truth <= q_phys_j) = P(truth - pred <= q_j - q_med) exactly.
- phi: same, on the wrapped difference.
- q/p (scale-free, eps = 0.02): pred = q_med * (|a| + eps) + a determines the
  anchor a uniquely per track (sign from the numerator pred - eps*q_med, since
  the two sign branches are mutually exclusive); then
  q_phys_j = q_j * (|a| + eps) + a.

Calibration = empirical coverage of each nominal level tau vs tau, shown as
(empirical - nominal) in percentage points with binomial errors; 6th panel =
summary.  Honors TRK_ABS_ETA_MAX (paper: 2) and TRK_PT_MAX (paper: 70 for the
uniform set), both from the truth kinematics stored in the same file.  Note
the stored ladders are monotone-reconstructed, so raw-output quantile
crossings are not measurable here (they are logged at training time).

Usage: quantile_calibration.py <pred.h5> <dataset_label> <out_dir>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the recipe's pinball ladder (all five heads; see THE FINAL RECIPE)
TAUS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
PARAMS = ["d0", "z0", "phi", "theta", "qop"]
MATH = {"d0": r"$d_0$", "z0": r"$z_0$", "phi": r"$\varphi$", "theta": r"$\theta$", "qop": r"$q/p$"}
ETA_MAX = float(os.environ.get("TRK_ABS_ETA_MAX", "3.0"))
PT_MAX = float(os.environ.get("TRK_PT_MAX", "inf"))


def _wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


SCALE_EPS_QOP = 0.02        # scale_anchor_eps of the recipe's q/p head


def main():
    h5_path, ds, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    f = h5py.File(h5_path)
    tgt = {p: f[f"targets/{p}"][:].astype(np.float64) for p in PARAMS}
    prd = {p: f[f"preds/{p}"][:].astype(np.float64) for p in PARAMS}
    qs = {p: f[f"quantiles/{p}"][:].astype(np.float64) for p in PARAMS}
    th = np.clip(tgt["theta"], 1e-8, np.pi - 1e-8)
    eta = -np.log(np.tan(th / 2.0))
    keep = np.isfinite(qs["d0"][:, 0])
    if ETA_MAX < 3.0:
        keep &= np.abs(eta) <= ETA_MAX
    cut_note = f"$|\\eta| \\leq {ETA_MAX:g}$" if ETA_MAX < 3.0 else "full acceptance"
    if np.isfinite(PT_MAX):
        pt = np.sin(th) / np.maximum(np.abs(tgt["qop"]), 1e-12)
        keep &= pt <= PT_MAX
        cut_note += f"; $p_\\mathrm{{T}} \\leq {PT_MAX:g}$ GeV"
    n = int(keep.sum())

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    axes = axes.ravel()
    med_idx = int(np.argmin(np.abs(TAUS - 0.5)))
    summary = []
    for i, p in enumerate(PARAMS):
        t, q, pr = tgt[p][keep], qs[p][keep], prd[p][keep]
        qm = q[:, med_idx]
        if p == "phi":
            cov = np.array([(_wrap(t - pr - (q[:, j] - qm)) <= 0).mean()
                            for j in range(len(TAUS))])
        elif p == "qop":
            num = pr - SCALE_EPS_QOP * qm
            a = np.where(num > 0, num / (1.0 + qm), num / (1.0 - qm))
            # exactness check of the anchor recovery (float32 storage noise)
            recon = qm * (np.abs(a) + SCALE_EPS_QOP) + a
            assert np.nanmax(np.abs(recon - pr)) < 1e-5, "q/p anchor recovery failed"
            cov = np.array([(t <= q[:, j] * (np.abs(a) + SCALE_EPS_QOP) + a).mean()
                            for j in range(len(TAUS))])
        else:
            cov = np.array([((t - pr) <= (q[:, j] - qm)).mean()
                            for j in range(len(TAUS))])
        err = np.sqrt(TAUS * (1 - TAUS) / n)
        dev = (cov - TAUS) * 100.0
        ax = axes[i]
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
        ax.errorbar(TAUS, dev, yerr=err * 100.0, fmt="o-", color="C0", ms=4.5,
                    lw=1.6, capsize=2.5, label="SSM")
        ax.fill_between(TAUS, -err * 100.0, err * 100.0, color="C0", alpha=0.25, lw=0)
        lim = max(1.3 * np.abs(dev).max(), 1.3 * 100.0 * err.max(), 0.5)
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(MATH[p])
        ax.set_xlabel(r"nominal quantile level $\tau$")
        ax.set_ylabel("empirical $-$ nominal coverage [pp]", fontsize=9)
        summary.append((p, np.abs(dev).max()))
    ax6 = axes[5]
    ax6.axis("off")
    lines = [f"{ds}", f"$N={n:,}$ tracks ({cut_note})", "",
             "worst |coverage deviation| over the ladder:"]
    for p, d in summary:
        lines.append(f"{MATH[p]}:  {d:.2f} pp")
    lines += ["", "band = binomial error on the", "empirical coverage at this $N$"]
    ax6.text(0.05, 0.95, "\n".join(lines), transform=ax6.transAxes,
             va="top", fontsize=11)
    fig.suptitle(f"{ds} --- calibration of the 7-quantile ladders "
                 f"(deployment path); $N={n:,}$ tracks; {cut_note}",
                 y=0.995, fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{ds}__quantile_calibration.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"[qcal] {out}", flush=True)


if __name__ == "__main__":
    main()
