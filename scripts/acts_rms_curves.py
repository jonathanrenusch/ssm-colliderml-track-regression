#!/usr/bin/env python3
"""Iterative-3-sigma-clipped RMS resolution curves from the ACTS-matched
residuals (matched_residuals.npz written by acts_integration.py --dump-residuals).

Why not the ACTS per-bin Gaussian fit: that fit degenerates in sparse bins
(single-bin high-pT spikes; the SSM's ultra-narrow forward theta core collapses
the fit and the curve stops short of |eta|=3).  The iterative-3-sigma RMS is
always defined where tracks exist, so coverage is full and high-pT bins are
made stable by equal-count (quantile) binning.

Per figure:
  * one panel per perigee parameter + a 6th panel with the binning-variable
    distribution; a SSM/KF ratio strip under each parameter panel.
  * bands = analytic clipped-RMS standard error  RMS / sqrt(2 N_kept)  (NOT
    bootstrap; the ACTS pipeline supplies the matched track sample).
  * every legend line: iter-3sigma RMS + "clipped N of M (x%)".
  * headline: total tracks entering the plot (fitted by both estimators).

Usage: acts_rms_curves.py <out_dir> <dataset_label> [--with-pt]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from track_regression.eval_utils import iterative_rms_convergence  # noqa: E402

PARAMS = ["d0", "z0", "phi", "theta", "qop"]
MATH = {"d0": r"$d_0$", "z0": r"$z_0$", "phi": r"$\varphi$", "theta": r"$\theta$", "qop": r"$q/p$"}
UNIT = {"d0": "µm", "z0": "µm", "phi": "mrad", "theta": "mrad", "qop": "$10^{-3}$/GeV"}
SCALE = {"d0": 1e3, "z0": 1e3, "phi": 1e3, "theta": 1e3, "qop": 1e3}
COL = {"SSM": "C0", "KF": "C3"}
ALPHA = 0.25


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _clip_rms(x):
    """iterative-3sigma RMS, kept count, total."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, 0, 0
    r = iterative_rms_convergence(x)
    return float(r["rms"]), int(r["n_kept"]), int(x.size)


def _bins(v, var, nb=24):
    if var == "eta":
        return np.linspace(-3.0, 3.0, nb + 1)
    # pT: uniform (linear) bins -- the uniform-pT muon sample is flat in pT, so
    # equal-width bins hold comparable track counts and are stable across the
    # full range; bins with < min tracks are dropped downstream.
    lo = max(float(v.min()), 1.0)
    hi = float(np.percentile(v, 99.9))
    return np.linspace(lo, hi, nb + 1)


def _curve(resid, xvar, edges):
    cen, val, err, nk, nt = [], [], [], [], []
    idx = np.digitize(xvar, edges) - 1
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 60:
            continue
        rms, k, n = _clip_rms(resid[m])
        if not np.isfinite(rms) or rms <= 0:
            continue
        cen.append(0.5 * (edges[b] + edges[b + 1]) if xvar is not None else 0)
        # use the median x in the bin for a faithful position
        cen[-1] = float(np.median(xvar[m]))
        val.append(rms); err.append(rms / np.sqrt(2 * max(k, 1)))
        nk.append(k); nt.append(n)
    return (np.array(cen), np.array(val), np.array(err))


def draw(out_dir: Path, ds: str, with_pt: bool):
    z = np.load(out_dir / "matched_residuals.npz")
    truth, ssm, kf = z["truth"], z["ssm"], z["kf"]
    both = np.isfinite(ssm[:, 0]) & np.isfinite(kf[:, 0])
    truth, ssm, kf = truth[both], ssm[both], kf[both]
    th = truth[:, 3]
    eta = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    pt = np.sin(th) / np.maximum(np.abs(truth[:, 4]), 1e-12)
    N = len(truth)

    variants = [("eta", eta, r"truth $\eta$", "rmscurve_vs_eta", 24)]
    if with_pt:
        variants.append(("pT", pt, r"$p_{\mathrm{T}}$ [GeV]", "rmscurve_vs_pt", 18))

    for vname, xv, xlabel, stem, nb in variants:
        edges = _bins(xv, vname, nb)
        fig = plt.figure(figsize=(15, 9.4))
        gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27)
        for i, p in enumerate(PARAMS):
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i],
                                          height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(sub[0]); axr = fig.add_subplot(sub[1], sharex=ax)
            sc = SCALE[p]
            rs = (_wrap(ssm[:, i] - truth[:, i]) if p == "phi" else ssm[:, i] - truth[:, i])
            rk = (_wrap(kf[:, i] - truth[:, i]) if p == "phi" else kf[:, i] - truth[:, i])
            curves = {}
            for lab, resid in (("SSM", rs), ("KF", rk)):
                c, v, e = _curve(resid, xv, edges)
                curves[lab] = (c, v * sc, e * sc)
                urms, uk, un = _clip_rms(resid)
                clipped = un - uk
                ax.plot(c, v * sc, "-", color=COL[lab], lw=1.8,
                        label=f"{lab}: {urms*sc:.3g} {UNIT[p]}\n"
                              f"({100*clipped/max(un,1):.1f}% clipped)")
                ax.fill_between(c, (v - e) * sc, (v + e) * sc, color=COL[lab], alpha=ALPHA, lw=0)
            ax.set_ylabel(f"iter-3$\\sigma$ RMS({MATH[p]}) [{UNIT[p]}]", fontsize=9)
            ax.set_title(MATH[p]); ax.set_ylim(bottom=0)
            ax.legend(loc="best", fontsize=6.6, framealpha=0.9,
                      handlelength=1.2, borderpad=0.25, labelspacing=0.2)
            cs, vs, es = curves["SSM"]; ck, vk, ek = curves["KF"]
            common, a, bxi = np.intersect1d(cs, ck, return_indices=True)
            if common.size:
                r = vs[a] / vk[bxi]
                re = r * np.sqrt((es[a] / vs[a]) ** 2 + (ek[bxi] / vk[bxi]) ** 2)
                axr.axhline(1.0, color="0.4", lw=0.8, ls=":")
                axr.plot(common, r, "-", color="C0", lw=1.4)
                axr.fill_between(common, r - re, r + re, color="C0", alpha=ALPHA, lw=0)
            axr.set_ylabel("SSM/KF", fontsize=8); axr.set_xlabel(xlabel)
            if vname == "eta":
                ax.set_xlim(-3, 3); axr.set_xlim(-3, 3)
            plt.setp(ax.get_xticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[5])
        ax6.hist(xv, bins=edges, histtype="step", color="0.3", lw=1.4)
        ax6.set_xlabel(xlabel); ax6.set_ylabel("tracks / bin")
        ax6.set_title("track distribution")
        if vname == "eta":
            ax6.set_xlim(-3, 3)
        fig.suptitle(f"{ds} --- iterative-3$\\sigma$-clipped RMS vs {xlabel}; "
                     f"total $N={N:,}$ tracks (fitted by both); "
                     f"bands = analytic RMS error", y=0.995, fontsize=11)
        fig.savefig(out_dir / f"{ds}_acts__{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[rmscurve] {out_dir / f'{ds}_acts__{stem}.pdf'}", flush=True)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]); ds = sys.argv[2]
    draw(out_dir, ds, "--with-pt" in sys.argv[3:])
