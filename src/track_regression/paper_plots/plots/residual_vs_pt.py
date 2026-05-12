"""Residual vs pT — 2×3 summary panels (5 params + η step hist).

Two pT axis variants per model: log-pT and linear-pT.
Residual axis is **always the inner 95 %** of the residual distribution
(2.5 %–97.5 % percentiles), so structure is visible even pre-clip.
Cell colour = absolute counts on a log scale (no density).

Per-param singles in `individuals/`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from track_regression.eval_utils import PARAMS

from .. import save_fig
from ..stats import DISPLAY_SCALE, DISPLAY_UNIT
from ._panels import fill_eta_stephist, make_grid

PT_LOG_EDGES = np.logspace(np.log10(0.5), np.log10(30.0), 81)
PT_LIN_EDGES = np.linspace(0.5, 30.0, 81)


def _resid_label(p: str) -> str:
    return rf"$\Delta {p}$ [{DISPLAY_UNIT[p]}]"


def _draw_one(ax, pt, r, p, *, pt_edges, log_pt, fig=None, ylim=None):
    """If ``ylim`` is None, the y-window is the inner 95 % of the supplied
    residual (per-panel auto-fit).  Pass an explicit ``(lo, hi)`` to
    enforce a shared window — required for paired SSM/CKF singles that
    will be placed side-by-side in the paper, so both panels share scale.
    """
    scale = DISPLAY_SCALE[p]
    r_scaled = r * scale
    if ylim is None:
        lo, hi = np.percentile(r_scaled, [2.5, 97.5])
    else:
        lo, hi = ylim
    res_edges = np.linspace(lo, hi, 121)
    H, xe, ye = np.histogram2d(pt, r_scaled, bins=[pt_edges, res_edges])
    pc = ax.pcolormesh(xe, ye, H.T,
                       norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
                       cmap="viridis", shading="auto")
    ax.axhline(0.0, color="r", ls="--", lw=0.8)
    if log_pt:
        ax.set_xscale("log")
    ax.set_xlim(pt_edges[0], pt_edges[-1])
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(_resid_label(p))
    ax.set_title(p)
    if fig is not None:
        fig.colorbar(pc, ax=ax, shrink=0.8, label="tracks / bin (log)")


def _summary(res, model_key, label, plots_dir, stem, *, log_pt):
    fig, axes = make_grid(figsize=(14.5, 8.5))
    pt = res["pt"]
    edges = PT_LOG_EDGES if log_pt else PT_LIN_EDGES
    for i, p in enumerate(PARAMS):
        _draw_one(axes[i], pt, res[f"{model_key}_{p}"], p,
                  pt_edges=edges, log_pt=log_pt, fig=fig)
    fill_eta_stephist(axes[5], res["eta"])
    axis_label = "log $p_T$" if log_pt else "linear $p_T$"
    fig.suptitle(f"{label} residual vs $p_T$ ({axis_label}, inner 95 %) — DM, "
                 f"N={res['count']:,}", y=0.995)
    fig.tight_layout()
    save_fig(fig, plots_dir, stem)


def make(res: dict, plots_dir: Path) -> None:
    individuals = plots_dir / "individuals"
    pt = res["pt"]

    for log_pt in (True, False):
        suffix = "logpt" if log_pt else "linpt"
        edges = PT_LOG_EDGES if log_pt else PT_LIN_EDGES

        # Per-param singles for SSM and CKF.  Y-window is locked to the
        # SSM inner-95 % so the two files share scale and can be placed
        # side-by-side in the paper — CKF tails that exceed the SSM
        # window fall off the figure (that's the intended visual: SSM's
        # tail suppression is what drives the headline claim).
        for p in PARAMS:
            ssm_scaled = res[f"ssm_{p}"] * DISPLAY_SCALE[p]
            shared_ylim = tuple(np.percentile(ssm_scaled, [2.5, 97.5]))
            for who in ("ssm", "ckf"):
                fig, ax = plt.subplots(figsize=(6.4, 4.6))
                _draw_one(ax, pt, res[f"{who}_{p}"], p,
                          pt_edges=edges, log_pt=log_pt, fig=fig,
                          ylim=shared_ylim)
                ax.set_title(f"{who.upper()}: {p}")
                save_fig(fig, individuals,
                         f"residual_vs_pt_{suffix}_{p}_{who}")

        # Summary 2×3 panels (one per model)
        _summary(res, "ssm", "SSM", plots_dir,
                 f"residual_vs_pt_summary_{suffix}_ssm", log_pt=log_pt)
        _summary(res, "ckf", "CKF", plots_dir,
                 f"residual_vs_pt_summary_{suffix}_ckf", log_pt=log_pt)
