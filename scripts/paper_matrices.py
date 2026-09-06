#!/usr/bin/env python3
"""Two paper matrices, both as vector PDF:

1. Gradient-cosine similarity (re-rendered from grad_cosines.npz): the color
   scale maxes out on the unit diagonal (vmax=1), off-diagonal alignments read
   against it, and each cell shows mean +- the STANDARD ERROR ON THE MEAN.
   (The npz also carries `std`, the single-batch scatter -- that is the
   gradient-noise floor of one minibatch, not the uncertainty on the plotted
   mean, and it does not shrink as more minibatches are averaged.)

2. Confusion matrix (predicted vs truth) for the SSM, built from the ACTS
   matched_residuals.npz (truth + fitted params), 2x3 per-parameter grid,
   log-density viridis with the y=x reference, theta labelled as theta.

Usage: paper_matrices.py <grad_cosines.npz> <matched_residuals.npz> <out_dir> [dataset_label]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

PARAMS = ["d0", "z0", "phi", "theta", "qop"]
MATH = {"d0": r"$d_0$", "z0": r"$z_0$", "phi": r"$\varphi$", "theta": r"$\theta$", "qop": r"$q/p$"}
UNIT = {"d0": "mm", "z0": "mm", "phi": "rad", "theta": "rad", "qop": "1/GeV"}
# v2 impact-parameter windows (drift-beamspot); angles/qop by data percentile
HARD_RANGE = {"d0": (-7.1, 7.1), "z0": (-270.0, 270.0)}


def cosine_matrix(npz_path: Path, out: Path):
    z = np.load(npz_path)
    mean = z["mean"]
    # Error bar = uncertainty on the plotted MEAN, not the single-batch scatter.
    # Older npz files carry only `std` (the per-batch noise floor); derive the
    # SEM from it when `n_batches` is recorded, and only fall back to `std`
    # itself if neither is available.
    if "sem" in z:
        err = z["sem"]
    elif "n_batches" in z:
        err = z["std"] / np.sqrt(float(z["n_batches"]))
    else:
        print("[matrices] WARNING: npz has no sem/n_batches; annotating per-batch std", flush=True)
        err = z["std"]
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    # 0..1 sequential scale (user, 2026-09-06): every mean entry of the probe
    # is >= 0, and a symmetric +-1 scale wasted half the range while the probe
    # script's own +-0.2 render clipped the large entries.  Blues matches the
    # paper's C0 performance-plot palette.
    im = ax.imshow(np.clip(mean, 0.0, 1.0), vmin=0.0, vmax=1.0, cmap="Blues")
    for i in range(5):
        for j in range(5):
            m = mean[i, j]
            txt = "1" if i == j else f"{m:+.3f}\n$\\pm${err[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if m > 0.6 else "black")
    ax.set_xticks(range(5), [MATH[p] for p in PARAMS])
    ax.set_yticks(range(5), [MATH[p] for p in PARAMS])
    ax.set_title("Trunk-gradient cosine similarity")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("mean cosine")
    fig.tight_layout()
    fig.savefig(out / "cos_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[matrices] wrote {out/'cos_heatmap.pdf'}", flush=True)


def confusion_matrix(npz_path: Path, out: Path, ds: str):
    z = np.load(npz_path)
    truth, ssm = z["truth"], z["ssm"]
    ok = np.isfinite(ssm[:, 0])
    truth, ssm = truth[ok], ssm[ok]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    axes = axes.ravel()
    for i, p in enumerate(PARAMS):
        ax = axes[i]
        t, s = truth[:, i], ssm[:, i]
        if p in HARD_RANGE:
            lo, hi = HARD_RANGE[p]
        else:
            lo = float(np.percentile(t, 0.5)); hi = float(np.percentile(t, 99.5))
        bins = np.linspace(lo, hi, 121)
        H, xe, ye = np.histogram2d(t, np.clip(s, lo, hi), bins=[bins, bins])
        pc = ax.pcolormesh(xe, ye, H.T, norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
                           cmap="viridis", shading="auto")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, alpha=0.7)
        ax.set_xlabel(f"truth {MATH[p]} [{UNIT[p]}]")
        ax.set_ylabel(f"SSM {MATH[p]} [{UNIT[p]}]")
        ax.set_aspect("equal"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(MATH[p])
        fig.colorbar(pc, ax=ax, shrink=0.85)
    axes[5].axis("off")
    axes[5].text(0.5, 0.5, f"{ds}\nSSM predicted vs. truth\n{len(truth):,} tracks\n"
                 "(log density)", ha="center", va="center", fontsize=11,
                 transform=axes[5].transAxes)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix_ssm.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[matrices] wrote {out/'confusion_matrix_ssm.pdf'}", flush=True)


if __name__ == "__main__":
    cos_npz, res_npz, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ds = sys.argv[4] if len(sys.argv) > 4 else "ttbar"
    out_dir.mkdir(parents=True, exist_ok=True)
    cosine_matrix(cos_npz, out_dir)
    if str(res_npz) != "-":          # "-" = cosine matrix only
        confusion_matrix(res_npz, out_dir, ds)
