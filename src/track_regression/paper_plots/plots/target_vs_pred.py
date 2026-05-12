"""Target distribution overlaid with SSM prediction.

Two outputs:
- `individuals/target_vs_pred_<p>.{pdf,png}` — per-parameter
- `target_vs_pred_summary.{pdf,png}` — 2×3 panel (5 params + η step hist)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from track_regression.eval_utils import (
    PARAMS,
    PARAM_VALUE_LABELS,
)

from .. import save_fig
from ._panels import fill_eta_stephist, make_grid

# Same selection-cut bounds as the heatmaps (loss norm_min/max).
RANGES = {
    "d0":    (-2.5, 2.5),
    "z0":    (-200.0, 200.0),
    "phi":   (-np.pi, np.pi),
    "theta": (0.0, np.pi),
    "qop":   (-2.0, 2.0),
}


def _draw_one(ax, truth, pred, p):
    lo, hi = RANGES[p]
    bins = np.linspace(lo, hi, 121)
    ax.hist(truth, bins=bins, histtype="stepfilled", alpha=0.35, color="0.4",
            label="Truth")
    ax.hist(pred, bins=bins, histtype="step", linewidth=1.6, color="C0",
            label="SSM prediction")
    ax.set_xlabel(PARAM_VALUE_LABELS[p])
    ax.set_ylabel("tracks / bin")
    if p == "d0":
        ax.set_yscale("log")
    ax.set_title(p)


def make(res: dict, plots_dir: Path) -> None:
    individuals = plots_dir / "individuals"

    # Per-param singles
    for p in PARAMS:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        _draw_one(ax, res[f"truth_{p}"], res[f"pred_ssm_{p}"], p)
        ax.legend(loc="best")
        save_fig(fig, individuals, f"target_vs_pred_{p}")

    # Summary 2×3
    fig, axes = make_grid()
    for i, p in enumerate(PARAMS):
        _draw_one(axes[i], res[f"truth_{p}"], res[f"pred_ssm_{p}"], p)
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8.5)
    fill_eta_stephist(axes[5], res["eta"])
    fig.suptitle(f"Target vs SSM prediction — DM, N={res['count']:,}", y=0.995)
    fig.tight_layout()
    save_fig(fig, plots_dir, "target_vs_pred_summary")
