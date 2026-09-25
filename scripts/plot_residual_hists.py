#!/usr/bin/env python3
"""Residual histograms, network vs truth-seeded KF.

Reads ``<bundle_dir>/matched_residuals.npz`` (from build_residuals.py) and
writes ``<dataset>_truthkf__residual_hist_liny.pdf`` next to it: a 2x3 grid
with one density-normalised histogram per perigee parameter (240 bins over
+-8 clipped RMS; entries outside are put into the edge bins) and the truth-eta
distribution in the 6th cell.  Every legend entry quotes the unbinned
iterative-3-sigma RMS and the fraction of tracks the clip removed.

Usage: plot_residual_hists.py <bundle_dir> <dataset> [--eta-max 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from track_regression.eval_utils import (  # noqa: E402
    DISPLAY_SCALE, DISPLAY_UNIT, PARAMS, fill_eta_stephist, iterative_rms_convergence, make_grid,
)

REF = "truth-KF"
# q/p in 1e-3/GeV so the axis labels stay short
SCALE = {**DISPLAY_SCALE, "qop": 1e3}
UNIT = {**DISPLAY_UNIT, "qop": r"$10^{-3}$/GeV"}


def _wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def load(bundle: Path, eta_max: float) -> dict:
    """Residuals of the tracks fitted by both, inside |truth eta| <= eta_max."""
    z = np.load(bundle / "matched_residuals.npz")
    truth, ssm, kf = z["truth"], z["ssm"], z["kf"]
    eta_all = -np.log(np.tan(np.clip(truth[:, 3], 1e-8, np.pi - 1e-8) / 2.0))
    if eta_max < 3.0:
        keep = np.abs(eta_all) <= eta_max
        truth, ssm, kf = truth[keep], ssm[keep], kf[keep]
    both = np.isfinite(ssm[:, 0]) & np.isfinite(kf[:, 0])
    res = {"count": int(both.sum())}
    for i, p in enumerate(PARAMS):
        s = ssm[both, i] - truth[both, i]
        k = kf[both, i] - truth[both, i]
        if p == "phi":
            s, k = _wrap(s), _wrap(k)
        res[f"ssm_{p}"] = s
        res[f"ref_{p}"] = k
    th = truth[both, 3]
    res["eta"] = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    return res


def residual_hist_page(res: dict, out_dir: Path, dataset: str, label: str = "minGRU") -> Path:
    fig, axes = make_grid()
    for i, p in enumerate(PARAMS):
        ax = axes[i]
        scale, unit = SCALE[p], UNIT[p]
        for arr, colour, tag in ((res[f"ssm_{p}"], "C0", label),
                                 (res[f"ref_{p}"], "C3", REF)):
            cut = iterative_rms_convergence(arr)
            rms3, kept = cut["rms"], cut["n_kept"]
            clip_pct = 100.0 * (1.0 - kept / max(len(arr), 1))
            lo, hi = -8.0 * rms3, 8.0 * rms3
            ax.hist(np.clip(arr, lo, hi) * scale, bins=240,
                    range=(lo * scale, hi * scale), histtype="step",
                    color=colour, lw=1.6, density=True,
                    label=f"{tag}  iter-3σ = {rms3 * scale:.3g} {unit}\n"
                          f"({clip_pct:.1f} % clipped)")
        ax.set_xlabel(f"residual({p}) [{unit}]")
        ax.set_ylabel("density")
        ax.set_title(p)
        ax.legend(loc="best", fontsize=6.4, framealpha=0.9,
                  handlelength=1.2, borderpad=0.25, labelspacing=0.2)
    fill_eta_stephist(axes[5], res["eta"])
    # The title sits above the saved canvas (y=1.05, no tight bbox) and is not
    # visible in the PDF, but tight_layout reserves room for it, so its text
    # fixes the panel geometry of the paper figure -- keep it unchanged.
    fig.suptitle(f"{dataset}_truthkf — residuals (iterative-3σ clip in the legends) — "
                 f"total $N={res['count']:,}$ tracks fitted by both\n"
                 f"reference = truth-tracking KF shipped with the dataset (truth_tracks); "
                 f"{label} on the same double-matched tracks of the v2 evaluation store", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"{dataset}_truthkf__residual_hist_liny.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir", type=Path, help="directory holding matched_residuals.npz")
    ap.add_argument("dataset", help="dataset name (title and file stem)")
    ap.add_argument("--eta-max", type=float, default=2.0, help="|truth eta| cut (default 2)")
    ap.add_argument("--label", default="minGRU", help="legend label of the network (default minGRU)")
    a = ap.parse_args()
    res = load(a.bundle_dir, a.eta_max)
    out = residual_hist_page(res, a.bundle_dir, a.dataset, a.label)
    print(f"[residual-hist] {out}", flush=True)


if __name__ == "__main__":
    main()
