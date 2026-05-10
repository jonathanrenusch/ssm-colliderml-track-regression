"""Shared 3×2 panel utilities + the η step-hist that fills the 6th cell."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def make_grid(figsize=(13.5, 8.0)):
    """Return (fig, axes-flat) with a 2-row × 3-col grid (= 6 cells)."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    return fig, axes.flatten()


def fill_eta_stephist(ax, eta: np.ndarray, *, bins=None) -> None:
    """Step histogram of pseudorapidity over the DM regime — fills the 6th panel."""
    if bins is None:
        bins = np.linspace(-3.0, 3.0, 61)
    ax.hist(eta, bins=bins, histtype="step", linewidth=1.6, color="0.25",
            label=f"DM tracks (N={len(eta):,})")
    ax.set_xlabel(r"truth $\eta$")
    ax.set_ylabel("tracks / bin")
    ax.set_title(r"DM track $\eta$ distribution")
    ax.legend(loc="lower center", fontsize=8.5)
    ax.grid(alpha=0.25)
