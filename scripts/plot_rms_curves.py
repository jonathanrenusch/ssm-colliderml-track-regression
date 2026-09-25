#!/usr/bin/env python3
"""Resolution curves: iterative-3-sigma-clipped RMS vs truth eta (and vs pT).

Reads ``<bundle_dir>/matched_residuals.npz`` (from build_residuals.py) and
writes, next to it,
  * ``<dataset>_truthkf__rmscurve_vs_eta.pdf``, and with ``--with-pt``
  * ``<dataset>_truthkf__rmscurve_vs_pt.pdf``.

Per figure: one panel per perigee parameter with an SSM / truth-KF ratio strip
underneath, and a 6th panel with the distribution of the binning variable.
Bands are the analytic standard error of the clipped RMS, RMS / sqrt(2 N_kept).
Each legend entry quotes the unbinned clipped RMS and the fraction of tracks
the clip removed.  A per-bin RMS (rather than a per-bin Gaussian fit) is
defined wherever tracks exist, so sparse bins stay stable.

Paper conventions (defaults): |eta| <= 2 on every page, pT <= 70 GeV on the
vs-pT page only.

Usage: plot_rms_curves.py <bundle_dir> <dataset> [--with-pt] [--eta-max 2] [--pt-max 70]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from track_regression.eval_utils import PARAMS, iterative_rms_convergence  # noqa: E402

TAG = "truthkf"          # file-stem tag
REF = "truth-KF"         # legend label of the reference fit
MATH = {"d0": r"$d_0$", "z0": r"$z_0$", "phi": r"$\varphi$", "theta": r"$\theta$", "qop": r"$q/p$"}
UNIT = {"d0": "µm", "z0": "µm", "phi": "mrad", "theta": "mrad", "qop": "$10^{-3}$/GeV"}
SCALE = {"d0": 1e3, "z0": 1e3, "phi": 1e3, "theta": 1e3, "qop": 1e3}
COL = {"SSM": "C0", REF: "C3"}
ALPHA = 0.25


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _clip_rms(x):
    """(iterative-3-sigma RMS, kept count, total count) of the finite entries."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, 0, 0
    r = iterative_rms_convergence(x)
    return float(r["rms"]), int(r["n_kept"]), int(x.size)


def _bins(v, var, eta_max, nb=24):
    if var == "eta":
        # keep the bin width of 24 bins over -3..3 (0.25) whatever the cut
        nb_cut = max(4, int(round(nb * eta_max / 3.0)))
        return np.linspace(-eta_max, eta_max, nb_cut + 1)
    # pT: equal-width bins -- the uniform-pT sample is flat in pT, so the bins
    # hold comparable track counts.
    lo = max(float(v.min()), 1.0)
    hi = float(np.percentile(v, 99.9))
    return np.linspace(lo, hi, nb + 1)


def _curve(resid, xvar, edges):
    """Per-bin clipped RMS at the median x of each bin (bins with < 60 tracks dropped)."""
    cen, val, err = [], [], []
    idx = np.digitize(xvar, edges) - 1
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 60:
            continue
        rms, k, _ = _clip_rms(resid[m])
        if not np.isfinite(rms) or rms <= 0:
            continue
        cen.append(float(np.median(xvar[m])))
        val.append(rms)
        err.append(rms / np.sqrt(2 * max(k, 1)))
    return np.array(cen), np.array(val), np.array(err)


def draw(bundle: Path, ds: str, with_pt: bool, eta_max: float, pt_max: float):
    z = np.load(bundle / "matched_residuals.npz")
    truth, ssm, kf = z["truth"], z["ssm"], z["kf"]
    both = np.isfinite(ssm[:, 0]) & np.isfinite(kf[:, 0])
    truth, ssm, kf = truth[both], ssm[both], kf[both]
    th = truth[:, 3]
    eta = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    if eta_max < 3.0:                      # the samples extend to |eta| = 3
        keep = np.abs(eta) <= eta_max
        truth, ssm, kf, eta, th = truth[keep], ssm[keep], kf[keep], eta[keep], th[keep]
    pt = np.sin(th) / np.maximum(np.abs(truth[:, 4]), 1e-12)
    N = len(truth)

    all_keep = np.ones(N, bool)
    variants = [("eta", eta, r"truth $\eta$", "rmscurve_vs_eta", 24, all_keep)]
    if with_pt:
        variants.append(("pT", pt, r"$p_{\mathrm{T}}$ [GeV]", "rmscurve_vs_pt", 18,
                         pt <= pt_max if np.isfinite(pt_max) else all_keep))

    for vname, xv_all, xlabel, stem, nb, keep in variants:
        xv = xv_all[keep]
        truth_v, ssm_v, kf_v, n_v = truth[keep], ssm[keep], kf[keep], int(keep.sum())
        edges = _bins(xv, vname, eta_max, nb)
        fig = plt.figure(figsize=(15, 9.4))
        gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27)
        for i, p in enumerate(PARAMS):
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i],
                                          height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(sub[0]); axr = fig.add_subplot(sub[1], sharex=ax)
            sc = SCALE[p]
            rs = ssm_v[:, i] - truth_v[:, i]
            rk = kf_v[:, i] - truth_v[:, i]
            if p == "phi":
                rs, rk = _wrap(rs), _wrap(rk)
            curves = {}
            for lab, resid in (("SSM", rs), (REF, rk)):
                c, v, e = _curve(resid, xv, edges)
                curves[lab] = (c, v * sc, e * sc)
                urms, uk, un = _clip_rms(resid)
                note = f"({100 * (un - uk) / max(un, 1):.1f}% clipped)"
                ax.plot(c, v * sc, "-", color=COL[lab], lw=1.8,
                        label=f"{lab}: {urms*sc:.3g} {UNIT[p]}\n{note}")
                ax.fill_between(c, (v - e) * sc, (v + e) * sc, color=COL[lab], alpha=ALPHA, lw=0)
            ax.set_ylabel(f"iter-3$\\sigma$ RMS({MATH[p]}) [{UNIT[p]}]", fontsize=9)
            ax.set_title(MATH[p])
            # y range: anchor at 0 only when the curves span a wide range; a
            # flat curve is zoomed to the data (extra padding on top for the legend).
            vals = [(v, e) for _, v, e in curves.values() if v.size]
            lo = min(float(np.nanmin(v - e)) for v, e in vals) if vals else 0.0
            hi = max(float(np.nanmax(v + e)) for v, e in vals) if vals else 1.0
            if lo > 0.4 * hi:
                span = max(hi - lo, 1e-12 * max(hi, 1e-30))
                ax.set_ylim(lo - 0.15 * span, hi + 0.45 * span)
            else:
                ax.set_ylim(bottom=0)
            ax.legend(loc="best", fontsize=6.6, framealpha=0.9,
                      handlelength=1.2, borderpad=0.25, labelspacing=0.2)
            cs, vs, es = curves["SSM"]; ck, vk, ek = curves[REF]
            common, a, bxi = np.intersect1d(cs, ck, return_indices=True)
            if common.size:
                r = vs[a] / vk[bxi]
                re = r * np.sqrt((es[a] / vs[a]) ** 2 + (ek[bxi] / vk[bxi]) ** 2)
                axr.axhline(1.0, color="0.4", lw=0.8, ls=":")
                axr.plot(common, r, "-", color="C0", lw=1.4)
                axr.fill_between(common, r - re, r + re, color="C0", alpha=ALPHA, lw=0)
            axr.set_ylabel(f"SSM/{REF}", fontsize=8); axr.set_xlabel(xlabel)
            # no tick label at the strip's top: it would collide with the main panel's
            axr.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
            if vname == "eta":
                ax.set_xlim(-eta_max, eta_max); axr.set_xlim(-eta_max, eta_max)
            plt.setp(ax.get_xticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[5])
        ax6.hist(xv, bins=edges, histtype="step", color="0.3", lw=1.4)
        ax6.set_xlabel(xlabel); ax6.set_ylabel("tracks / bin")
        ax6.set_title("track distribution")
        if vname == "eta":
            ax6.set_xlim(-eta_max, eta_max)
        cut_note = f"; $|\\eta| \\leq {eta_max:g}$" if eta_max < 3.0 else ""
        if vname == "pT" and np.isfinite(pt_max):
            cut_note += f"; $p_\\mathrm{{T}} \\leq {pt_max:g}$ GeV"
        fig.suptitle(f"{ds} --- iterative-3$\\sigma$-clipped RMS vs {xlabel}; "
                     f"total $N={n_v:,}$ tracks (fitted by both){cut_note}; "
                     f"bands = analytic RMS error", y=0.995, fontsize=11)
        out = bundle / f"{ds}_{TAG}__{stem}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[rmscurve] {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir", type=Path, help="directory holding matched_residuals.npz")
    ap.add_argument("dataset", help="dataset name (title and file stem)")
    ap.add_argument("--with-pt", action="store_true", help="also draw the vs-pT page")
    ap.add_argument("--eta-max", type=float, default=2.0, help="|truth eta| cut (default 2)")
    ap.add_argument("--pt-max", type=float, default=70.0,
                    help="pT cap [GeV] of the vs-pT page (default 70; 'inf' for none)")
    a = ap.parse_args()
    draw(a.bundle_dir, a.dataset, a.with_pt, a.eta_max, a.pt_max)
