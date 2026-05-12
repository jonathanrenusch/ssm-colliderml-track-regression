"""Single-panel pT distribution of double-matched test tracks (appendix figure)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import save_fig


def make(res: dict, plots_dir: Path) -> None:
    pt = res["pt"]
    n = len(pt)

    edges = np.linspace(0.5, 30.0, 60)  # 0.5 GeV core cut → 30 GeV cap
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(pt, bins=edges, histtype="step", color="C2", linewidth=1.6,
            label=f"DM tracks  N = {n:,}")
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("tracks / bin (log)")
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Test set $p_T$ — double-matched tracks")
    save_fig(fig, plots_dir, "pt_distribution_dm")
