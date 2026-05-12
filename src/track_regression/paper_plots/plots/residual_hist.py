"""Residual histograms — 2×3 summary panel (5 params + η step hist).

Always **absolute counts** (no density).  Variants:
  linear y, pre-clip   — windowed to inner 95 % so the core is visible
  linear y, post iter-3σ
  log y, pre-clip      — full residual range, log absolute counts (tail comparison)
  log y, post iter-3σ  — full kept range, log absolute counts

Per-param singles in `individuals/`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from track_regression.eval_utils import (
    PARAMS,
    iterative_rms_convergence,
)


def _fmt_n(n: int) -> str:
    """Compact track-count for legend, e.g. 6.59M / 142k / 8200."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{n}"

from .. import save_fig
from ..stats import DISPLAY_SCALE, DISPLAY_UNIT
from ._panels import fill_eta_stephist, make_grid


def _resid_label(p: str) -> str:
    return rf"$\Delta {p}$ [{DISPLAY_UNIT[p]}]"


def _prep(ssm, ckf, p, *, post_clip, log_y):
    """Return (ssm_scaled, ckf_scaled, bin_edges).

    Range logic:
      * linear y → inner 95 % of pooled (SSM+CKF) — needed for the core to
        be visible at all.
      * log y    → inner 99.9 % of pooled — drops the literal-handful of
        extreme outliers, but keeps the full sub-percent tail visible so
        the SSM-vs-CKF tail-suppression argument can actually be seen.
    """
    scale = DISPLAY_SCALE[p]
    if post_clip:
        cs = iterative_rms_convergence(ssm)
        cc = iterative_rms_convergence(ckf)
        ssm = ssm[(ssm >= cs["cut_lo"]) & (ssm <= cs["cut_hi"])]
        ckf = ckf[(ckf >= cc["cut_lo"]) & (ckf <= cc["cut_hi"])]

    s = ssm * scale
    c = ckf * scale
    pooled = np.concatenate([s, c])
    if log_y:
        lo, hi = np.percentile(pooled, [0.05, 99.95])
    else:
        lo, hi = np.percentile(pooled, [2.5, 97.5])
    # 200 bins ≈ 33 k tracks/bin at full DM, enough resolution to see both
    # the core shape and the tail asymmetry against CKF on log y.
    edges = np.linspace(lo, hi, 201)
    return s, c, edges


def _draw_one(ax, ssm, ckf, bins, p, *, log_y, with_legend=False):
    ax.hist(ckf, bins=bins, histtype="step", color="C3", lw=1.4,
            label=f"CKF  N={_fmt_n(len(ckf))}")
    ax.hist(ssm, bins=bins, histtype="step", color="C0", lw=1.4,
            label=f"SSM  N={_fmt_n(len(ssm))}")
    ax.axvline(0, color="0.4", ls=":", lw=0.8)
    ax.set_xlabel(_resid_label(p))
    ax.set_ylabel("tracks / bin" + (" (log)" if log_y else ""))
    if log_y:
        ax.set_yscale("log")
        # Show down to single-count bins so the SSM vs CKF tail
        # suppression is visible, not auto-clipped to top 2 decades.
        ax.set_ylim(bottom=0.7)
    # Cap x-tick density — qop on linear-postclip in particular gets crammed.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_title(p)
    if with_legend:
        ax.legend(loc="upper right", fontsize=8.5)


def _summary(res, *, post_clip, log_y, plots_dir, stem, title):
    fig, axes = make_grid()
    for i, p in enumerate(PARAMS):
        ssm, ckf, bins = _prep(res[f"ssm_{p}"], res[f"ckf_{p}"], p,
                               post_clip=post_clip, log_y=log_y)
        _draw_one(axes[i], ssm, ckf, bins, p, log_y=log_y, with_legend=(i == 0))
    fill_eta_stephist(axes[5], res["eta"])
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    save_fig(fig, plots_dir, stem)


def _individual(res, *, post_clip, log_y, plots_dir, stem_fmt):
    for p in PARAMS:
        ssm, ckf, bins = _prep(res[f"ssm_{p}"], res[f"ckf_{p}"], p,
                               post_clip=post_clip, log_y=log_y)
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        _draw_one(ax, ssm, ckf, bins, p, log_y=log_y, with_legend=True)
        save_fig(fig, plots_dir, stem_fmt.format(p=p))


def make(res: dict, plots_dir: Path) -> None:
    individuals = plots_dir / "individuals"
    n = res["count"]
    base = f"DM, N={n:,}"

    variants = [
        # (post_clip, log_y, stem, title)
        (False, False, "residual_hist_summary_linear_preclip",
         f"Residuals (linear y, pre-clip, inner 95 %) — {base}"),
        (True, False, "residual_hist_summary_linear_postclip",
         f"Residuals (linear y, post iter-3σ) — {base}"),
        (False, True, "residual_hist_summary_logy_preclip",
         f"Residuals (log absolute counts, pre-clip, full range — tails) — {base}"),
        (True, True, "residual_hist_summary_logy_postclip",
         f"Residuals (log absolute counts, post iter-3σ) — {base}"),
    ]
    for post, logy, stem, title in variants:
        _summary(res, post_clip=post, log_y=logy, plots_dir=plots_dir,
                 stem=stem, title=title)
        single_stem = stem.replace("residual_hist_summary",
                                   "residual_hist") + "_{p}"
        _individual(res, post_clip=post, log_y=logy, plots_dir=individuals,
                    stem_fmt=single_stem)
