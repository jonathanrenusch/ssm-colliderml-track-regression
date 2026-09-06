#!/usr/bin/env python3
"""Iter-3sigma RMS of every perigee parameter binned in truth |d0| and truth
z0, SSM vs the shipped truth-KF, from a matched_residuals.npz (house 6-panel
design with ratio strips).  Requested by the analysis meeting for the 100 GeV
sample (2026-09-06).  Honors TRK_ABS_ETA_MAX (paper: 2) and the
TRK_PLOT_TAG/TRK_REF_LABEL conventions of the other bundle scripts.

Usage: rms_vs_impact_npz.py <npz_dir> <dataset_label>
Writes <npz_dir>/<ds>_<tag>__rms_vs_absd0.pdf and ..._rms_vs_z0.pdf
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

from track_regression.eval_utils import iterative_rms_convergence  # noqa: E402

TAG = os.environ.get("TRK_PLOT_TAG", "truthkf")
REF = os.environ.get("TRK_REF_LABEL", "truth-KF")
ETA_MAX = float(os.environ.get("TRK_ABS_ETA_MAX", "3.0"))
PARAMS = ["d0", "z0", "phi", "theta", "qop"]
MATH = {"d0": r"$d_0$", "z0": r"$z_0$", "phi": r"$\varphi$", "theta": r"$\theta$", "qop": r"$q/p$"}
UNIT = {"d0": "µm", "z0": "µm", "phi": "mrad", "theta": "mrad", "qop": "$10^{-3}$/GeV"}
SCALE = {"d0": 1e3, "z0": 1e3, "phi": 1e3, "theta": 1e3, "qop": 1e3}
COL = {"SSM": "C0", REF: "C3"}


def _wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def it3(x):
    r = iterative_rms_convergence(x)
    return float(r["rms"]), int(r["n_kept"]), int(len(x))


def main():
    out_dir, ds = Path(sys.argv[1]), sys.argv[2]
    z = np.load(out_dir / "matched_residuals.npz")
    truth, ssm, kf = z["truth"], z["ssm"], z["kf"]
    both = np.isfinite(ssm[:, 0]) & np.isfinite(kf[:, 0])
    truth, ssm, kf = truth[both], ssm[both], kf[both]
    eta = -np.log(np.tan(np.clip(truth[:, 3], 1e-8, np.pi - 1e-8) / 2.0))
    if ETA_MAX < 3.0:
        keep = np.abs(eta) <= ETA_MAX
        truth, ssm, kf, eta = truth[keep], ssm[keep], kf[keep], eta[keep]
    N = len(truth)
    res = {}
    for i, p in enumerate(PARAMS):
        s, k = ssm[:, i] - truth[:, i], kf[:, i] - truth[:, i]
        if p == "phi":
            s, k = _wrap(s), _wrap(k)
        res[("SSM", p)], res[(REF, p)] = s, k

    variants = [
        (np.abs(truth[:, 0]), r"truth $|d_0|$ [mm]", "rms_vs_absd0",
         np.linspace(0.0, np.quantile(np.abs(truth[:, 0]), 0.999), 21)),
        (truth[:, 1], r"truth $z_0$ [mm]", "rms_vs_z0", np.linspace(-270, 270, 25)),
    ]
    for vals, xlabel, stem, edges in variants:
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(vals, edges) - 1, 0, len(centers) - 1)
        fig = plt.figure(figsize=(15, 9.4))
        gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27)
        for i, p in enumerate(PARAMS):
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i],
                                          height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(sub[0]); axr = fig.add_subplot(sub[1], sharex=ax)
            sc = SCALE[p]
            curves = {}
            for lab in ("SSM", REF):
                arr = res[(lab, p)]
                y = np.full(len(centers), np.nan); ye = np.full(len(centers), np.nan)
                for b in range(len(centers)):
                    m = idx == b
                    if m.sum() > 200:
                        r, kkept, _ = it3(arr[m])
                        y[b] = r; ye[b] = r / np.sqrt(2 * max(kkept, 1))
                curves[lab] = (y, ye)
                ok = np.isfinite(y)
                u, uk, un = it3(arr)
                ax.plot(centers[ok], y[ok] * sc, "-", color=COL[lab], lw=1.8,
                        label=f"{lab}: {u*sc:.3g} {UNIT[p]}\n({100*(1-uk/max(un,1)):.1f}% clipped)")
                ax.fill_between(centers[ok], (y - ye)[ok] * sc, (y + ye)[ok] * sc,
                                color=COL[lab], alpha=0.25, lw=0)
            ax.set_ylabel(f"iter-3$\\sigma$ RMS({MATH[p]}) [{UNIT[p]}]", fontsize=9)
            ax.set_title(MATH[p]); ax.set_ylim(bottom=0)
            ax.legend(loc="best", fontsize=6.6, framealpha=0.9,
                      handlelength=1.2, borderpad=0.25, labelspacing=0.2)
            ys, yse = curves["SSM"]; yk, yke = curves[REF]
            r = ys / yk
            re = r * np.sqrt((yse / ys) ** 2 + (yke / yk) ** 2)
            axr.axhline(1.0, color="0.4", lw=0.8, ls=":")
            axr.plot(centers, r, "-", color="C0", lw=1.4)
            axr.fill_between(centers, r - re, r + re, color="C0", alpha=0.25, lw=0)
            axr.set_ylabel(f"SSM/{REF}", fontsize=8); axr.set_xlabel(xlabel)
            plt.setp(ax.get_xticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[5])
        ax6.hist(vals, bins=edges, histtype="step", color="0.3", lw=1.4)
        ax6.set_xlabel(xlabel); ax6.set_ylabel("tracks / bin"); ax6.set_title("track distribution")
        cut_note = f"; $|\\eta| \\leq {ETA_MAX:g}$" if ETA_MAX < 3.0 else ""
        fig.suptitle(f"{ds} --- iterative-3$\\sigma$-clipped RMS vs {xlabel}; "
                     f"total $N={N:,}$ tracks (fitted by both){cut_note}; "
                     f"bands = analytic RMS error", y=0.995, fontsize=11)
        fig.savefig(out_dir / f"{ds}_{TAG}__{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[rms_vs_impact] {out_dir / f'{ds}_{TAG}__{stem}.pdf'}", flush=True)


if __name__ == "__main__":
    main()
