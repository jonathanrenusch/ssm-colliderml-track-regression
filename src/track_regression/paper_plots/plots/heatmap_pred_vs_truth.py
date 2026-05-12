"""Heatmap pred vs truth, per-param + 2×3 summary (one figure each for SSM, CKF)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from track_regression.eval_utils import (
    FULL_RANGE_PARAMS,
    HEATMAP_RANGE as EVAL_HEATMAP_RANGE,
    PARAMS,
    PARAM_VALUE_LABELS,
)

from .. import save_fig
from ._panels import fill_eta_stephist, make_grid

# Limits follow the original `evaluate_tail_diagnostics.py` convention:
#   * d0 → hard ±2.5 mm (FULL_RANGE_PARAMS / HEATMAP_RANGE in eval_utils.py)
#   * everything else → 0.5–99.5 percentile of pooled (truth, pred), so
#     ~99 % of the joint distribution sets the axis and the figure
#     "comfortably fills the full range" (matches the original eval plots).


def _resolve_range(truth: np.ndarray, pred: np.ndarray, p: str) -> tuple[float, float]:
    """Limits:

      * d0 in EVAL_HEATMAP_RANGE → hard ±2.5 mm (matches eval_utils convention).
      * other params → 0.5–99 percentile of **TRUTH only** so the data fills
        the plot.  Earlier attempts that pooled (truth, pred) get pulled wide
        by sparse model outliers (e.g. qop pred reaches ±2.3 with O(100) tracks)
        leaving the dense ridge looking shrunken.  Truth-only honours the
        physical envelope.
    """
    if p in EVAL_HEATMAP_RANGE:
        return EVAL_HEATMAP_RANGE[p]
    if p in FULL_RANGE_PARAMS:
        return (float(min(truth.min(), pred.min())),
                float(max(truth.max(), pred.max())))
    lo = float(np.percentile(truth, 0.5))
    hi = float(np.percentile(truth, 99.5))
    return lo, hi


def _draw_one(ax, truth, pred, p, *, with_colorbar=True, fig=None):
    lo, hi = _resolve_range(truth, pred, p)
    bins = np.linspace(lo, hi, 121)
    H, xe, ye = np.histogram2d(truth, pred, bins=[bins, bins])
    pc = ax.pcolormesh(xe, ye, H.T,
                       norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
                       cmap="viridis", shading="auto")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, alpha=0.7)
    ax.set_xlabel(f"truth {PARAM_VALUE_LABELS[p]}")
    ax.set_ylabel(f"pred {PARAM_VALUE_LABELS[p]}")
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(p)
    if with_colorbar and fig is not None:
        fig.colorbar(pc, ax=ax, shrink=0.85)


def _summary(res: dict, model_key: str, label: str, plots_dir: Path,
             stem: str) -> None:
    fig, axes = make_grid(figsize=(14.5, 9.0))
    for i, p in enumerate(PARAMS):
        _draw_one(axes[i], res[f"truth_{p}"], res[f"pred_{model_key}_{p}"], p,
                  with_colorbar=True, fig=fig)
        # heatmap aspect=equal can clip subplot — relax for grid:
        axes[i].set_aspect("auto")
    fill_eta_stephist(axes[5], res["eta"])
    fig.suptitle(f"{label}: prediction vs truth — DM, N={res['count']:,}", y=0.995)
    fig.tight_layout()
    save_fig(fig, plots_dir, stem)


def make(res: dict, plots_dir: Path) -> None:
    individuals = plots_dir / "individuals"

    # Per-param singles
    for p in PARAMS:
        for who, key in (("ssm", "ssm"), ("ckf", "ckf")):
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            _draw_one(ax, res[f"truth_{p}"], res[f"pred_{key}_{p}"], p,
                      with_colorbar=True, fig=fig)
            ax.set_title(f"{who.upper()}: {p}")
            save_fig(fig, individuals, f"heatmap_pred_vs_truth_{p}_{who}")

    # Summary 2×3 panels
    _summary(res, "ssm", "SSM", plots_dir, "heatmap_pred_vs_truth_summary_ssm")
    _summary(res, "ckf", "CKF", plots_dir, "heatmap_pred_vs_truth_summary_ckf")
