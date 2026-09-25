#!/usr/bin/env python3
"""Gradient-cosine heatmap from the npz written by grad_cos_probe.py.

Each off-diagonal cell is the mean cosine similarity, over the probed
minibatches, between the trunk gradients of two per-parameter losses
(npz key ``mean``); the diagonal is 1 by construction.  Colour scale 0..1.
The standard errors of the means are in the npz (``sem``) and the probe's
summary.txt.

Usage: plot_grad_cos.py <grad_cosines.npz> <out_dir>   -> <out_dir>/cos_heatmap.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = ["$d_0$", "$z_0$", r"$\varphi$", r"$\theta$", "$q/p$"]


def cosine_matrix(npz_path: Path, out: Path) -> Path:
    mean = np.load(npz_path)["mean"]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(np.clip(mean, 0.0, 1.0), vmin=0.0, vmax=1.0, cmap="Blues")
    disp = np.where(np.eye(5, dtype=bool), 1.0, mean)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{disp[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9, color="white" if disp[i, j] > 0.6 else "black")
    ax.set_xticks(range(5), LABELS); ax.set_yticks(range(5), LABELS)
    ax.set_title("Trunk-gradient cosine similarity")
    fig.colorbar(im, ax=ax, shrink=0.85, label="mean cosine")
    fig.tight_layout()
    path = out / "cos_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[grad-cos] {cosine_matrix(Path(sys.argv[1]), out_dir)}")
