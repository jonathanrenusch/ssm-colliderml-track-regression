#!/usr/bin/env python3
"""One page: RMS vs pT on the uniform-muon sample, ALL ablation arms overlaid.

Same sample, same estimator, same binning and the same reference as the paper's
`single_muon_uniform_rmscurve_vs_pt` page (iterative-3-sigma clipped RMS,
|eta| <= TRK_ABS_ETA_MAX, pT <= TRK_PT_MAX) -- but instead of one model against
the truth-KF, every architecture from the v2 ablation is drawn as its own line,
with the truth-KF in black as the common reference and a ratio strip beneath
each panel.

Usage: abl_compare_rms_vs_pt.py [out.pdf]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/track_regression"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from acts_rms_curves import (  # noqa: E402
    ETA_MAX, MATH, PARAMS, PT_MAX, SCALE, UNIT, _clip_rms, _curve, _wrap,
)

ROOT = Path(__file__).resolve().parents[1] / "eval_plots/ablations_2026-09/arm_residuals"
# (directory, legend label, colour, linestyle) -- order = legend order
ARMS = [
    ("SSM_baseline_25ep",  "Mamba-2, bidirectional (paper)", "C0", "-"),
    ("V2_mamba1dir_25ep",  "Mamba-2, one-directional",       "C2", "-"),
    ("V2_diagssm_25ep",    "diagonal SSM, non-selective",    "C1", "--"),
    ("V2_txf_25ep",        "Transformer",                    "C4", "-."),
    ("V2_mingru_25ep",     "minGRU",                         "C5", "-"),
]
REF_LABEL = "truth-KF"


def load(arm: str, ds: str = "single_muon_uniform"):
    z = np.load(ROOT / arm / ds / "matched_residuals.npz")
    truth, ssm, kf = z["truth"], z["ssm"], z["kf"]
    both = np.isfinite(ssm).all(1) & np.isfinite(kf).all(1) & np.isfinite(truth).all(1)
    truth, ssm, kf = truth[both], ssm[both], kf[both]
    th = truth[:, 3]
    eta = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    pt = np.sin(th) / np.maximum(np.abs(truth[:, 4]), 1e-12)
    keep = (np.abs(eta) <= ETA_MAX) & (pt <= PT_MAX)
    return truth[keep], ssm[keep], kf[keep], pt[keep]


def resid(a, truth, i, p):
    return _wrap(a[:, i] - truth[:, i]) if p == "phi" else a[:, i] - truth[:, i]


def main(out: str) -> int:
    data = {}
    for arm, *_ in ARMS:
        if (ROOT / arm / "single_muon_uniform" / "matched_residuals.npz").exists():
            data[arm] = load(arm)
    if not data:
        print(f"no arm residuals under {ROOT}")
        return 1
    ref_arm = next(iter(data))
    truth0, _, kf0, pt0 = data[ref_arm]
    edges = np.linspace(max(float(pt0.min()), 1.0), float(min(np.percentile(pt0, 99.9), PT_MAX)), 25)

    fig = plt.figure(figsize=(15.5, 8.6))
    outer = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26)
    for i, p in enumerate(PARAMS):
        # the arms are so close that the top panel is degenerate -- give the
        # ratio strip real estate, since that is where the comparison lives
        sub = outer[i // 3, i % 3].subgridspec(2, 1, height_ratios=[2, 1.4], hspace=0.06)
        ax, axr = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        # reference: truth-KF (identical across arms -- same matched tracks)
        rmin, rmax = [], []
        ck, vk, _ = _curve(resid(kf0, truth0, i, p), pt0, edges)
        ax.plot(ck, np.array(vk) * SCALE[p], color="k", lw=2.0, ls=":", label=REF_LABEL, zorder=5)
        for arm, lab, col, ls in ARMS:
            if arm not in data:
                continue
            truth, ssm, _, pt = data[arm]
            c, v, e = _curve(resid(ssm, truth, i, p), pt, edges)
            c, v, e = np.array(c), np.array(v), np.array(e)
            ax.plot(c, v * SCALE[p], color=col, ls=ls, lw=1.6, label=lab)
            ax.fill_between(c, (v - e) * SCALE[p], (v + e) * SCALE[p], color=col, alpha=0.18, lw=0)
            vki = np.interp(c, ck, vk)
            r = v / vki
            axr.plot(c, r, color=col, ls=ls, lw=1.4)
            rmin.append(float(r.min())); rmax.append(float(r.max()))
        axr.axhline(1.0, color="k", lw=1.0, ls=":")
        rmin.append(1.0); rmax.append(1.0)
        ax.set_ylabel(f"{MATH[p]} RMS [{UNIT[p]}]")
        ax.set_xticklabels([])
        ax.grid(alpha=0.25)
        axr.grid(alpha=0.25)
        axr.set_ylabel("/ KF", fontsize=9)
        axr.set_xlabel(r"truth $p_\mathrm{T}$ [GeV]")
        # zoom the ratio to the arms actually present in this panel
        lo, hi = min(rmin), max(rmax)
        pad = max(0.012, 0.12 * (hi - lo))
        axr.set_ylim(lo - pad, hi + pad)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lax = fig.add_subplot(outer[1, 2]); lax.axis("off")
    lax.legend(handles, labels, loc="center", fontsize=11, frameon=False,
               title="encoder (identical recipe, data and heads)", title_fontsize=11)
    n = len(truth0)
    fig.suptitle(
        f"Encoder ablation, identical recipe and data — iterative-3σ RMS vs $p_T$, "
        f"uniform-$p_T$ muons ({n:,} tracks, $|\\eta|\\leq{ETA_MAX:g}$"
        + (f", $p_T\\leq{PT_MAX:g}$ GeV" if np.isfinite(PT_MAX) else "") + ")",
        fontsize=12)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.985)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[abl-compare] {out}  ({len(data)} arms, {n:,} tracks)")

    print(f"\nintegrated (unbinned) iter-3sigma RMS, arm / truth-KF, |eta|<={ETA_MAX:g}"
          + (f", pT<={PT_MAX:g}" if np.isfinite(PT_MAX) else ""))
    print(f"{'arm':<34}" + "".join(f"{p:>9}" for p in PARAMS))
    for arm, lab, *_ in ARMS:
        if arm not in data:
            continue
        truth, ssm, kf, _ = data[arm]
        row = f"{lab:<34}"
        for i, p in enumerate(PARAMS):
            row += f"{_clip_rms(resid(ssm, truth, i, p))[0] / _clip_rms(resid(kf, truth, i, p))[0]:>9.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "eval_plots/paper_plots/ablation_compare/"
                       "ablation_rmscurve_vs_pt_uniform.pdf"))
