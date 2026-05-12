"""RMS vs η, SSM vs CKF, pre-clip + iter-3σ, with bootstrap 2σ band.

- `individuals/rms_vs_eta_<p>.{pdf,png}` — per-param
- `rms_vs_eta_summary.{pdf,png}`        — 2×3 (5 params + η step hist)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.colliderml_regr.eval_utils import (
    PARAMS,
    iterative_rms_convergence,
)

from matplotlib.lines import Line2D

from .. import save_fig
from ..bootstrap import bootstrap_metric
from ..stats import DISPLAY_SCALE, DISPLAY_UNIT
from ._panels import fill_eta_stephist, make_grid


def _style_legend_handles(mode: str) -> tuple[list, list]:
    """Single shared legend describing line style/colour conventions."""
    if mode == "both":
        return (
            [Line2D([0], [0], color="C0", lw=1.8),
             Line2D([0], [0], color="C0", lw=1.0, ls="--", alpha=0.7),
             Line2D([0], [0], color="C3", lw=1.8),
             Line2D([0], [0], color="C3", lw=1.0, ls="--", alpha=0.7)],
            ["SSM (iter-3σ)", "SSM (pre-clip)",
             "CKF (iter-3σ)", "CKF (pre-clip)"],
        )
    label = "iter-3σ" if mode == "postclip" else "pre-clip"
    return (
        [Line2D([0], [0], color="C0", lw=1.8),
         Line2D([0], [0], color="C3", lw=1.8)],
        [f"SSM ({label})", f"CKF ({label})"],
    )


def _binwise(eta, res, edges, fn, n_boot, seed, min_n=30):
    rng = np.random.default_rng(seed)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.full(len(centers), np.nan)
    sg = np.full(len(centers), np.nan)
    bin_idx = np.clip(np.digitize(eta, edges) - 1, 0, len(centers) - 1)
    for b in range(len(centers)):
        sel = res[bin_idx == b]
        if len(sel) < min_n:
            continue
        mu[b] = fn(sel)
        if n_boot > 0:
            samps = np.empty(n_boot)
            for k in range(n_boot):
                idx = rng.integers(0, len(sel), len(sel))
                samps[k] = fn(sel[idx])
            sg[b] = np.std(samps, ddof=1)
        else:
            sg[b] = 0.0
    return mu, sg


def _draw_one(ax, eta, ssm, ckf, p, *, n_boot, scale, unit, mode="both", compact=False):
    """mode: 'both' (pre + iter-3σ), 'preclip' (pre only), 'postclip' (iter-3σ only).

    compact=True (used in summary panels with a shared figure-level legend):
        labels show only ``μ = value ± err`` so the per-panel legend is small.
    compact=False (per-param singles):
        labels include model + clipping prefix so the legend is self-contained.
    """
    edges = np.linspace(-3.0, 3.0, 31)
    centers = 0.5 * (edges[:-1] + edges[1:])
    raw_rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    iter_rms = lambda x: iterative_rms_convergence(x)["rms"]

    ssm_post, ssm_post_s = _binwise(eta, ssm, edges, iter_rms, n_boot, seed=12)
    ssm_pre,  ssm_pre_s  = _binwise(eta, ssm, edges, raw_rms,  n_boot, seed=11)
    ckf_post, ckf_post_s = _binwise(eta, ckf, edges, iter_rms, n_boot, seed=14)
    ckf_pre,  ckf_pre_s  = _binwise(eta, ckf, edges, raw_rms,  n_boot, seed=13)

    # Unbinned (full-DM) values + their bootstrap 2σ shown in the legend.
    # n=80 is plenty given full-DM N=6.6 M.
    ssm_post_m, ssm_post_su = bootstrap_metric(ssm, iter_rms, n=80, seed=21)
    ssm_pre_m,  ssm_pre_su  = bootstrap_metric(ssm, raw_rms,  n=80, seed=22)
    ckf_post_m, ckf_post_su = bootstrap_metric(ckf, iter_rms, n=80, seed=23)
    ckf_pre_m,  ckf_pre_su  = bootstrap_metric(ckf, raw_rms,  n=80, seed=24)
    ssm_post_ub, ssm_post_e = ssm_post_m * scale, 2 * ssm_post_su * scale
    ssm_pre_ub,  ssm_pre_e  = ssm_pre_m  * scale, 2 * ssm_pre_su  * scale
    ckf_post_ub, ckf_post_e = ckf_post_m * scale, 2 * ckf_post_su * scale
    ckf_pre_ub,  ckf_pre_e  = ckf_pre_m  * scale, 2 * ckf_pre_su  * scale
    def f(v: float) -> str:
        """3 sig figs, scientific for the extreme small (qop)."""
        if v == 0 or not np.isfinite(v):
            return f"{v:.3g}"
        a = abs(v)
        if a >= 100:
            return f"{v:.0f}"        # 830 µm
        if a >= 10:
            return f"{v:.1f}"        # 12.3 µm
        if a >= 1:
            return f"{v:.2f}"        # 2.11 mrad
        if a >= 0.01:
            return f"{v:.3f}"        # 0.643 mrad
        return f"{v:.2e}"            # 3.40e-03 1/GeV

    def _fmt_n(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return f"{n}"

    def lbl(name, ub, e, n):
        e_str = f(e)
        try:
            e_zero = float(e_str) == 0.0
        except ValueError:
            e_zero = False
        prefix = "" if compact else f"{name}  "
        body = f"μ = {f(ub)}" + ("" if e_zero else f" ± {e_str}")
        # Compact mode (panel grid) drops unit + N — both are in the ylabel
        # and the figure-level suptitle. Saves ~40 % of label width so the
        # 3-line legend fits inside the axes without bleeding into the
        # neighbouring panel.
        if compact:
            return prefix + body
        return prefix + body + f" {unit}  (N={_fmt_n(n)})"

    # Sample sizes per line.  Pre-clip uses full DM; iter-3σ keeps ~99.7 %.
    def _n_after_iter3sigma(x):
        cut = iterative_rms_convergence(x)
        return cut["n_kept"]

    n_full = len(ssm)  # SSM and CKF DM masks are the same length (paired)
    n_ssm_post = _n_after_iter3sigma(ssm)
    n_ckf_post = _n_after_iter3sigma(ckf)

    # 2σ bands (95% CI under bootstrap normal approx).  At full-DM N=6.6 M
    # the per-bin bootstrap σ is sub-pixel; the band is mathematically present
    # but visually collapses onto the line — see legend numerics.
    show_post = mode in ("both", "postclip")
    show_pre  = mode in ("both", "preclip")
    pre_style = "--" if mode == "both" else "-"
    pre_lw    = 1.0  if mode == "both" else 1.8
    pre_alpha = 0.7  if mode == "both" else 1.0

    if show_post:
        ax.plot(centers, ssm_post * scale, "-", color="C0", lw=1.8,
                label=lbl("SSM iter-3σ", ssm_post_ub, ssm_post_e, n_ssm_post))
        ax.fill_between(centers, (ssm_post - 2 * ssm_post_s) * scale,
                        (ssm_post + 2 * ssm_post_s) * scale, color="C0", alpha=0.20)
    if show_pre:
        ax.plot(centers, ssm_pre * scale, pre_style, color="C0",
                lw=pre_lw, alpha=pre_alpha,
                label=lbl("SSM pre-clip", ssm_pre_ub, ssm_pre_e, n_full))
        if mode == "preclip":
            ax.fill_between(centers, (ssm_pre - 2 * ssm_pre_s) * scale,
                            (ssm_pre + 2 * ssm_pre_s) * scale,
                            color="C0", alpha=0.20)
    if show_post:
        ax.plot(centers, ckf_post * scale, "-", color="C3", lw=1.8,
                label=lbl("CKF iter-3σ", ckf_post_ub, ckf_post_e, n_ckf_post))
        ax.fill_between(centers, (ckf_post - 2 * ckf_post_s) * scale,
                        (ckf_post + 2 * ckf_post_s) * scale, color="C3", alpha=0.20)
    if show_pre:
        ax.plot(centers, ckf_pre * scale, pre_style, color="C3",
                lw=pre_lw, alpha=pre_alpha,
                label=lbl("CKF pre-clip", ckf_pre_ub, ckf_pre_e, n_full))
        if mode == "preclip":
            ax.fill_between(centers, (ckf_pre - 2 * ckf_pre_s) * scale,
                            (ckf_pre + 2 * ckf_pre_s) * scale,
                            color="C3", alpha=0.20)
    ax.set_xlabel(r"truth $\eta$")
    ax.set_ylabel(f"RMS({p}) [{DISPLAY_UNIT[p]}]")
    ax.set_xlim(-3, 3)
    ax.set_ylim(bottom=0)
    ax.set_title(p)


def make(res: dict, plots_dir: Path, *, n_boot: int = 50) -> None:
    individuals = plots_dir / "individuals"
    eta = res["eta"]

    # Three modes: combined (both lines), pre-clip only, post-clip only.
    MODES = [
        ("both",     "rms_vs_eta_summary",
         "RMS vs η — pre-clip + iter-3σ"),
        ("preclip",  "rms_vs_eta_summary_preclip",
         "RMS vs η — pre-clip only (tail-inclusive)"),
        ("postclip", "rms_vs_eta_summary_postclip",
         "RMS vs η — iter-3σ-clipped core only"),
    ]

    # Per-param singles, one file per (param, mode)
    for p in PARAMS:
        scale = DISPLAY_SCALE[p]
        unit = DISPLAY_UNIT[p]
        for mode, _stem, _title in MODES:
            fig, ax = plt.subplots(figsize=(7.0, 4.6))
            _draw_one(ax, eta, res[f"ssm_{p}"], res[f"ckf_{p}"], p,
                      n_boot=n_boot, scale=scale, unit=unit, mode=mode)
            ax.legend(loc="best", fontsize=8.5)
            suffix = "" if mode == "both" else f"_{mode}"
            save_fig(fig, individuals, f"rms_vs_eta_{p}{suffix}")

    # Summary 2×3, one file per mode.  Per-panel legend = compact μ values
    # only (no model/clip wording).  One figure-level legend at the top
    # explains line colour + style → model / clipping.
    for mode, stem, title in MODES:
        fig, axes = make_grid()
        for i, p in enumerate(PARAMS):
            scale = DISPLAY_SCALE[p]
            unit = DISPLAY_UNIT[p]
            _draw_one(axes[i], eta, res[f"ssm_{p}"], res[f"ckf_{p}"], p,
                      n_boot=n_boot, scale=scale, unit=unit, mode=mode,
                      compact=True)
            # Lock the legend inside the upper-left corner of the axes
            # (with a small inset) so it never extends past the panel
            # boundary into the row below. Tight font / padding so the
            # 3 curves fit cleanly.
            axes[i].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
                           fontsize=7.0, handlelength=1.3, handletextpad=0.4,
                           framealpha=0.85, borderpad=0.25,
                           labelspacing=0.2)
            # Give the legend a bit of vertical room above the curve.
            ymin, ymax = axes[i].get_ylim()
            axes[i].set_ylim(bottom=ymin, top=ymax * 1.20)
        fill_eta_stephist(axes[5], eta)
        handles, labels = _style_legend_handles(mode)
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.985))
        fig.suptitle(f"{title} — bands = bootstrap ±2σ — DM, "
                     f"N={res['count']:,}", y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, plots_dir, stem)
