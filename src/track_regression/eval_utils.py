"""Shared constants, the iterative-3-sigma RMS estimator and plot helpers used
by the evaluation scripts (and by the model's validation metrics)."""

from __future__ import annotations

import numpy as np


PARAMS = ["d0", "z0", "phi", "theta", "qop"]

# Display units for tables and figures: mm -> um, rad -> mrad, q/p in 1/GeV.
DISPLAY_SCALE = {"d0": 1e3, "z0": 1e3, "phi": 1e3, "theta": 1e3, "qop": 1.0}
DISPLAY_UNIT = {"d0": "µm", "z0": "µm", "phi": "mrad", "theta": "mrad", "qop": "1/GeV"}


def iterative_rms_convergence(
    residuals: np.ndarray,
    n_sigma: float = 3.0,
    max_iter: int = 5,
) -> dict:
    """Iteratively clip residuals to mean +- n_sigma*sigma for at most ``max_iter`` passes.

    The cut window uses sigma = np.std (mean-centred spread). The returned
    ``"rms"`` is the RMSE of the surviving set, ``sqrt(mean(x**2))``, so it is
    sensitive to both spread and bias: for an unbiased estimator RMSE = sigma,
    for a biased one RMSE = sqrt(sigma**2 + mean**2). Iteration stops early
    once a pass removes no further entries.
    """
    data = np.asarray(residuals, dtype=np.float64)
    prev_n = -1
    cut_lo = float(np.min(data))
    cut_hi = float(np.max(data))
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        mean = float(np.mean(data))
        sigma = float(np.std(data))
        cut_lo = mean - n_sigma * sigma
        cut_hi = mean + n_sigma * sigma
        mask = (data >= cut_lo) & (data <= cut_hi)
        n_kept = int(np.sum(mask))
        if n_kept == prev_n:
            break
        prev_n = n_kept
        data = data[mask]

    return {
        "mean": float(np.mean(data)),
        "rms": float(np.sqrt(np.mean(data ** 2))),
        "sigma": float(np.std(data)),
        "n_kept": len(data),
        "n_total": len(residuals),
        "cut_lo": cut_lo,
        "cut_hi": cut_hi,
        "n_iterations": n_iter,
        "frac_kept": len(data) / max(len(residuals), 1),
    }


# ---------------------------------------------------------------------------
# plot helpers (matplotlib is imported lazily so the model does not need it)
# ---------------------------------------------------------------------------

def apply_paper_style() -> None:
    """rcParams for the summary figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # editable text in vector PDF
        "ps.fonttype": 42,
    })


def make_grid(figsize=(13.5, 8.0)):
    """Return (fig, flat axes) of a 2 x 3 grid: five parameters + one extra cell."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    return fig, axes.flatten()


def fill_eta_stephist(ax, eta: np.ndarray, *, bins=None) -> None:
    """Step histogram of the truth pseudorapidity of the evaluated tracks (6th cell)."""
    if bins is None:
        bins = np.linspace(-3.0, 3.0, 61)
    ax.hist(eta, bins=bins, histtype="step", linewidth=1.6, color="0.25",
            label=f"DM tracks (N={len(eta):,})")
    ax.set_xlabel(r"truth $\eta$")
    ax.set_ylabel("tracks / bin")
    ax.set_title(r"DM track $\eta$ distribution")
    ax.legend(loc="lower center", fontsize=8.5)
    ax.grid(alpha=0.25)
