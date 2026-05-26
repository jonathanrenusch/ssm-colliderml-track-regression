"""Shared constants, data loading, and residual computation for evaluation scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


# ============================================================================
# Constants
# ============================================================================

PARAMS = ["d0", "z0", "phi", "theta", "qop"]

UNIT_SCALE = {
    "d0": 1.0,       # mm
    "z0": 1.0,       # mm
    "phi": 1e3,      # rad → mrad
    "theta": 1e3,    # rad → mrad
    "qop": 1.0,      # 1/GeV
}

PARAM_LABELS = {
    "d0": r"$\sigma(d_0)$ [mm]",
    "z0": r"$\sigma(z_0)$ [mm]",
    "phi": r"$\sigma(\phi)$ [mrad]",
    "theta": r"$\sigma(\theta)$ [mrad]",
    "qop": r"$\sigma(q/p)$ [1/GeV]",
}

PARAM_VALUE_LABELS = {
    "d0": r"$d_0$ [mm]",
    "z0": r"$z_0$ [mm]",
    "phi": r"$\phi$ [rad]",
    "theta": r"$\theta$ [rad]",
    "qop": r"$q/p$ [1/GeV]",
}

RESID_LABELS = {
    "d0": r"$\Delta d_0$ [mm]",
    "z0": r"$\Delta z_0$ [mm]",
    "phi": r"$\Delta \phi$ [mrad]",
    "theta": r"$\Delta \theta$ [mrad]",
    "qop": r"$\Delta (q/p)$ [1/GeV]",
}

# Parameters whose full data range should be shown (no percentile clipping)
FULL_RANGE_PARAMS = {"d0"}

# Hard axis limits for specific parameters in heatmaps (keeps SSM/ACTS aligned)
HEATMAP_RANGE = {
    "d0": (-2.5, 2.5),
}


# ============================================================================
# Data loading
# ============================================================================

def load_predictions(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load predictions, targets, and (optionally) quantiles from HDF5 file.

    Returns ``{"preds": {name: array}, "targets": {name: array},
    "quantiles": {name: array}, "quantile_levels": {name: array}}``.

    The ``quantiles`` group is only populated when the HDF5 file contains
    quantile predictions (shape ``(N, Q)`` per parameter).
    ``quantile_levels`` maps parameter names to their ``(Q,)`` tau values.
    """
    data: dict[str, dict[str, np.ndarray]] = {
        "preds": {},
        "targets": {},
        "quantiles": {},
        "quantile_levels": {},
    }
    with h5py.File(path, "r") as f:
        for group in ("preds", "targets"):
            for name in f[group]:
                data[group][name] = f[group][name][:]
        # Load quantile predictions if present
        if "quantiles" in f:
            for name in f["quantiles"]:
                data["quantiles"][name] = f["quantiles"][name][:]
                if "levels" in f["quantiles"][name].attrs:
                    data["quantile_levels"][name] = f["quantiles"][name].attrs["levels"]
    return data


def load_acts_augmentation(
    data_dir: Path,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load ACTS reco params, DM mask, and per-track hit counts from shards.

    Returns ``(acts_reco, acts_dm_mask, nhits)`` or ``None`` if not available.
    ``acts_reco`` has shape (N, 5) with columns [d0, z0, phi, theta, qop],
    NaN where ACTS had no matching track.
    ``acts_dm_mask`` has shape (N,) bool.
    ``nhits`` has shape (N,) int32 — number of truth hits per selected track.
    """
    split_file = data_dir / "split.json"
    if not split_file.exists():
        return None

    with open(split_file) as f:
        splits = json.load(f)

    shard_indices = sorted(splits.get(split, []))
    if not shard_indices:
        return None

    all_reco, all_dm, all_nhits = [], [], []
    for idx in tqdm(shard_indices, desc="Loading ACTS augmentation", file=sys.stderr):
        sel_dir = data_dir / f"shard_{idx:04d}" / "selected_tracks"
        reco_file = sel_dir / "acts_reco.npy"
        dm_file = sel_dir / "acts_dm_mask.npy"
        offsets_file = sel_dir / "track_hit_offsets.npy"
        if not reco_file.exists():
            return None  # Augmentation not available for this shard
        all_reco.append(np.load(reco_file))
        all_dm.append(np.load(dm_file))
        offsets = np.load(offsets_file)
        all_nhits.append(np.diff(offsets).astype(np.int32))

    return (
        np.concatenate(all_reco, axis=0),
        np.concatenate(all_dm, axis=0),
        np.concatenate(all_nhits, axis=0),
    )


# ============================================================================
# Residuals
# ============================================================================

def compute_residuals(data: dict) -> dict[str, np.ndarray]:
    """Compute pred − truth residuals and truth eta for binning."""
    residuals: dict[str, np.ndarray] = {}
    for name in PARAMS:
        residuals[name] = data["preds"][name] - data["targets"][name]

    # Wrap phi residual to [-π, π]
    residuals["phi"] = (residuals["phi"] + np.pi) % (2 * np.pi) - np.pi

    # Truth eta from truth theta binning (this is not a residual but needed for precision vs η)
    theta_truth = data["targets"]["theta"]
    residuals["eta"] = -np.log(np.tan(np.clip(theta_truth, 1e-8, np.pi - 1e-8) / 2.0))

    return residuals


def compute_acts_residuals(
    acts_reco: np.ndarray,
    targets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute ACTS residuals from (N, 5) reco array and target dict.

    Only includes tracks where ACTS had a match (non-NaN).
    Returns residuals dict with 'eta' and only the matched entries.
    """
    has_match = ~np.any(np.isnan(acts_reco), axis=1)
    reco = acts_reco[has_match]
    param_order = PARAMS  # [d0, z0, phi, theta, qop]

    residuals: dict[str, np.ndarray] = {}
    for i, name in enumerate(param_order):
        residuals[name] = reco[:, i] - targets[name][has_match]

    residuals["phi"] = (residuals["phi"] + np.pi) % (2 * np.pi) - np.pi

    theta_truth = targets["theta"][has_match]
    residuals["eta"] = -np.log(np.tan(np.clip(theta_truth, 1e-8, np.pi - 1e-8) / 2.0))

    return residuals


def filter_residuals(
    residuals: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Filter residuals dict to a boolean or index mask."""
    return {k: v[mask] for k, v in residuals.items()}


def compute_precision_vs_eta(
    residuals: dict[str, np.ndarray],
    eta_range: tuple[float, float] = (-3.0, 3.0),
    n_eta_bins: int = 30,
    use_rms: bool = False,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Compute binned precision (std or RMS of residual) vs η.

    Parameters
    ----------
    use_rms : bool
        If True, compute RMS = sqrt(mean(x²)) instead of std.

    Returns ``(eta_bins, {param: {eta_centers, std, std_err, count, unbinned_std}})``.
    """
    eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
    eta = residuals["eta"]
    result: dict[str, dict[str, np.ndarray]] = {}

    def _metric(x):
        if use_rms:
            return float(np.sqrt(np.mean(x ** 2)))
        return float(np.std(x))

    for name in PARAMS:
        res = residuals[name]

        assert len(eta) == len(res), (
            f"Length mismatch: eta has {len(eta)} elements, but {name} has {len(res)}"
        )
        e, r = eta, res

        unbinned_val = _metric(r)

        centers, vals, counts = [], [], []
        for i in range(len(eta_bins) - 1):
            mask = (e >= eta_bins[i]) & (e < eta_bins[i + 1])
            n = int(np.sum(mask))

            centers.append((eta_bins[i] + eta_bins[i + 1]) / 2)
            if n > 2:
                vals.append(_metric(r[mask]))
                counts.append(n)
            else:
                vals.append(np.nan)
                counts.append(0)

        vals_arr = np.array(vals)
        counts_arr = np.array(counts, dtype=float)

        result[name] = {
            "eta_centers": np.array(centers),
            "std": vals_arr,
            "std_err": vals_arr / np.sqrt(2 * counts_arr),
            "count": counts_arr,
            "unbinned_std": unbinned_val,
        }

    return eta_bins, result


def compute_core_metrics_vs_eta(
    residuals: dict[str, np.ndarray],
    eta_range: tuple[float, float] = (-3.0, 3.0),
    n_eta_bins: int = 30,
    core_fraction: float = 0.95,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Compute RMS, std and IQR of the inner `core_fraction` of residuals vs η.

    No iterative clipping: per bin, drop the outer (1-core_fraction) symmetrically
    via percentiles and compute three metrics on the retained core.
    """
    eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
    eta = residuals["eta"]
    result: dict[str, dict[str, np.ndarray]] = {}

    tail = (1.0 - core_fraction) / 2.0
    lo_pct, hi_pct = 100.0 * tail, 100.0 * (1.0 - tail)

    def _metrics(x: np.ndarray) -> tuple[float, float, float]:
        lo, hi = np.percentile(x, [lo_pct, hi_pct])
        core = x[(x >= lo) & (x <= hi)]
        if core.size < 2:
            return float("nan"), float("nan"), float("nan")
        rms = float(np.sqrt(np.mean(core ** 2)))
        std = float(np.std(core))
        q1, q3 = np.percentile(core, [25.0, 75.0])
        iqr = float(q3 - q1)
        return rms, std, iqr

    for name in PARAMS:
        res = residuals[name]
        assert len(eta) == len(res)

        ub_rms, ub_std, ub_iqr = _metrics(res)

        centers, rms_vals, std_vals, iqr_vals, counts = [], [], [], [], []
        for i in range(len(eta_bins) - 1):
            mask = (eta >= eta_bins[i]) & (eta < eta_bins[i + 1])
            n = int(np.sum(mask))
            centers.append((eta_bins[i] + eta_bins[i + 1]) / 2)
            if n > 10:
                r, s, q = _metrics(res[mask])
                rms_vals.append(r)
                std_vals.append(s)
                iqr_vals.append(q)
                counts.append(int(round(n * core_fraction)))
            else:
                rms_vals.append(np.nan)
                std_vals.append(np.nan)
                iqr_vals.append(np.nan)
                counts.append(0)

        result[name] = {
            "eta_centers": np.array(centers),
            "rms": np.array(rms_vals),
            "std": np.array(std_vals),
            "iqr": np.array(iqr_vals),
            "count": np.array(counts, dtype=float),
            "unbinned_rms": ub_rms,
            "unbinned_std": ub_std,
            "unbinned_iqr": ub_iqr,
        }

    return eta_bins, result


# ============================================================================
# Iterative convergence methods (ATLAS-style precision)
# ============================================================================

def iterative_rms_convergence(
    residuals: np.ndarray,
    n_sigma: float = 3.0,
    max_iter: int = 5,
) -> dict:
    """Iteratively clip residuals to mean ± n_sigma*sigma for a fixed number of passes.

    The cut window uses sigma = np.std (mean-centred spread), which is the
    natural meaning of "3 sigma from the mean" for the clipping step. The
    returned ``"rms"`` is the true RMSE of the converged set,
    ``sqrt(mean(x**2))`` — sensitive to both spread *and* bias, matching the
    ML-community RMSE convention. For an unbiased estimator RMSE = sigma; for
    a biased one RMSE = sqrt(sigma**2 + mean**2), so a bias collapse like the
    d0 cross surfaces in this metric instead of being hidden.

    Capped at ``max_iter=5`` passes (ATLAS-style but limited): deep iteration
    collapses heavy-tailed distributions to their detector core and erases
    η-dependent structure, which isn't what we want here.
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


def _gauss(x, A, mu, sigma):
    """Gaussian function for curve fitting."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def iterative_gaussfit_convergence(
    residuals: np.ndarray,
    n_sigma: float = 2.0,
    max_iter: int = 3,
    tol: float = 0.0005,
    n_bins: int = 200,
) -> dict:
    """Iteratively fit Gaussian to histogram, refine range until stable.

    Two initial fits (full range + restricted) plus ``max_iter=3`` refinement
    passes, so 5 total — capped to avoid the tail-collapse pathology that makes
    the precision curves look artificially flat for heavy-tailed parameters.
    """
    res = np.asarray(residuals, dtype=np.float64)
    n_bins = min(n_bins, max(50, len(res) // 50))

    # --- First fit: full range ---
    lo, hi = float(np.min(res)), float(np.max(res))
    counts, edges = np.histogram(res, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0

    try:
        p0 = [float(np.max(counts)), float(np.mean(res)), float(np.std(res))]
        popt, _ = curve_fit(_gauss, centers[valid], counts[valid].astype(float),
                            p0=p0, maxfev=5000)
        mean, sigma = float(popt[1]), float(abs(popt[2]))
    except Exception:
        mean, sigma = float(np.mean(res)), float(np.std(res))

    # --- Second fit: restricted range ---
    lo = mean - n_sigma * sigma
    hi = mean + n_sigma * sigma
    counts, edges = np.histogram(res, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0

    try:
        popt, _ = curve_fit(_gauss, centers[valid], counts[valid].astype(float),
                            p0=[float(np.max(counts)), mean, sigma], maxfev=5000)
        mean, sigma = float(popt[1]), float(abs(popt[2]))
    except Exception:
        pass

    # --- Iterative convergence ---
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        lo = mean - n_sigma * sigma
        hi = mean + n_sigma * sigma
        counts, edges = np.histogram(res, bins=n_bins, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        valid = counts > 0

        try:
            popt, _ = curve_fit(_gauss, centers[valid], counts[valid].astype(float),
                                p0=[float(np.max(counts)), mean, sigma], maxfev=5000)
            mean_new, sigma_new = float(popt[1]), float(abs(popt[2]))
        except Exception:
            break

        if abs(mean - mean_new) < tol and abs(sigma - sigma_new) < tol:
            mean, sigma = mean_new, sigma_new
            break
        mean, sigma = mean_new, sigma_new

    mask = (res >= lo) & (res <= hi)
    return {
        "mean": mean,
        "sigma": sigma,
        "n_kept": int(np.sum(mask)),
        "n_total": len(res),
        "cut_lo": float(lo),
        "cut_hi": float(hi),
        "n_iterations": n_iter,
        "frac_kept": float(np.sum(mask)) / max(len(res), 1),
    }


def compute_precision_vs_eta_iterative(
    residuals: dict[str, np.ndarray],
    eta_range: tuple[float, float] = (-3.0, 3.0),
    n_eta_bins: int = 30,
    method: str = "iterative_rms",
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, dict]]:
    """Compute binned precision using iterative convergence methods.

    Parameters
    ----------
    method : str
        ``"iterative_rms"`` or ``"iterative_gaussfit"``.

    Returns
    -------
    eta_bins : np.ndarray
    precision_data : dict  (same structure as ``compute_precision_vs_eta``)
    unbinned_cuts : dict   {param: convergence result dict with cut boundaries}
    """
    eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
    eta = residuals["eta"]
    result: dict[str, dict[str, np.ndarray]] = {}
    unbinned_cuts: dict[str, dict] = {}

    conv_fn = iterative_rms_convergence if method == "iterative_rms" else iterative_gaussfit_convergence
    # "iterative_rms" returns RMSE = sqrt(mean(x²)) (bias-sensitive,
    # ML-community convention); "iterative_gaussfit" returns σ from the
    # Gaussian fit. Both are reported as the per-bin "precision".
    metric_key = "rms" if method == "iterative_rms" else "sigma"
    min_tracks = 30  # need enough statistics for iterative methods

    for name in PARAMS:
        res = residuals[name]

        # Unbinned convergence
        ub = conv_fn(res)
        unbinned_sigma = ub[metric_key]
        unbinned_cuts[name] = ub

        centers, sigmas, counts = [], [], []
        for i in range(len(eta_bins) - 1):
            mask = (eta >= eta_bins[i]) & (eta < eta_bins[i + 1])
            n = int(np.sum(mask))
            centers.append((eta_bins[i] + eta_bins[i + 1]) / 2)

            if n >= min_tracks:
                conv = conv_fn(res[mask])
                sigmas.append(conv[metric_key])
                counts.append(conv["n_kept"])
            else:
                sigmas.append(np.nan)
                counts.append(0)

        sigmas_arr = np.array(sigmas)
        counts_arr = np.array(counts, dtype=float)

        result[name] = {
            "eta_centers": np.array(centers),
            "std": sigmas_arr,
            "std_err": sigmas_arr / np.sqrt(2 * np.maximum(counts_arr, 1)),
            "count": counts_arr,
            "unbinned_std": unbinned_sigma,
        }

    return eta_bins, result, unbinned_cuts


# ============================================================================
# Residual statistics report (shared by both scripts)
# ============================================================================

def write_residual_statistics_report(
    ml_residuals: dict[str, np.ndarray],
    output_dir: Path,
    ml_label: str = "SSM",
    acts_residuals: dict[str, np.ndarray] | None = None,
    acts_label: str = "ACTS CKF",
    filename: str = "residual_statistics.txt",
    regime_subtitle: str | None = None,
    eta_range: tuple[float, float] = (-3.0, 3.0),
    n_eta_bins: int = 30,
) -> None:
    """Write detailed residual statistics to a text file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    percentiles = [0.1, 0.5, 1, 2.5, 5, 10, 16, 25, 50, 75, 84, 90, 95, 97.5, 99, 99.5, 99.9]

    n_ml = len(ml_residuals.get("eta", []))
    n_acts = len(acts_residuals.get("eta", [])) if acts_residuals is not None else 0

    lines: list[str] = []
    lines.append(f"Residual Statistics Report")
    lines.append(f"{'=' * 90}")
    if regime_subtitle:
        lines.append(f"Selection : {regime_subtitle}")
    lines.append(f"ML label  : {ml_label} ({n_ml:,} tracks)")
    if acts_residuals is not None:
        lines.append(f"ACTS label: {acts_label} ({n_acts:,} tracks)")
    lines.append("")

    for name in PARAMS:
        if name not in ml_residuals:
            continue
        scale = UNIT_SCALE.get(name, 1.0)
        unit = {1.0: "", 1e3: " mrad"}.get(scale, "")
        ml_res = ml_residuals[name] * scale
        has_acts = acts_residuals is not None and name in acts_residuals

        lines.append(f"{'─' * 90}")
        lines.append(f"  {name.upper()}{unit}")
        lines.append(f"{'─' * 90}")

        lines.append(f"  {'Statistic':<25s}  {'ML':>14s}")
        if has_acts:
            lines[-1] += f"  {'ACTS':>14s}  {'ML/ACTS ratio':>14s}"
        lines.append(f"  {'-' * 25}  {'-' * 14}")
        if has_acts:
            lines[-1] += f"  {'-' * 14}  {'-' * 14}"

        acts_res = acts_residuals[name] * scale if has_acts else None

        stats = [
            ("N tracks", f"{len(ml_res):>14,d}",
             f"{len(acts_res):>14,d}" if has_acts else None, None),
            ("Mean", f"{np.mean(ml_res):>14.6f}",
             f"{np.mean(acts_res):>14.6f}" if has_acts else None, None),
            ("Std (np.std)", f"{np.std(ml_res):>14.6f}",
             f"{np.std(acts_res):>14.6f}" if has_acts else None,
             f"{np.std(ml_res) / np.std(acts_res):>14.4f}" if has_acts else None),
            ("MAD (median abs dev)", f"{np.median(np.abs(ml_res - np.median(ml_res))):>14.6f}",
             f"{np.median(np.abs(acts_res - np.median(acts_res))):>14.6f}" if has_acts else None,
             None),
            ("IQR / 1.349 (robust σ)", f"{(np.percentile(ml_res, 75) - np.percentile(ml_res, 25)) / 1.349:>14.6f}",
             f"{(np.percentile(acts_res, 75) - np.percentile(acts_res, 25)) / 1.349:>14.6f}" if has_acts else None,
             None),
            ("68.3% width / 2 (1σ core)", f"{(np.percentile(ml_res, 84.13) - np.percentile(ml_res, 15.87)) / 2:>14.6f}",
             f"{(np.percentile(acts_res, 84.13) - np.percentile(acts_res, 15.87)) / 2:>14.6f}" if has_acts else None,
             None),
            ("95.4% width / 4 (2σ core)", f"{(np.percentile(ml_res, 97.72) - np.percentile(ml_res, 2.28)) / 4:>14.6f}",
             f"{(np.percentile(acts_res, 97.72) - np.percentile(acts_res, 2.28)) / 4:>14.6f}" if has_acts else None,
             None),
            ("99.7% width / 6 (3σ core)", f"{(np.percentile(ml_res, 99.87) - np.percentile(ml_res, 0.13)) / 6:>14.6f}",
             f"{(np.percentile(acts_res, 99.87) - np.percentile(acts_res, 0.13)) / 6:>14.6f}" if has_acts else None,
             None),
        ]

        # Iterative convergence metrics (applied on scaled residuals)
        ml_irms = iterative_rms_convergence(ml_res)
        ml_igf = iterative_gaussfit_convergence(ml_res)
        acts_irms = iterative_rms_convergence(acts_res) if has_acts else None
        acts_igf = iterative_gaussfit_convergence(acts_res) if has_acts else None

        stats.append(("", "", None, None))  # blank separator
        stats.append((
            f"Iter. RMSE (3σ clip)",
            f"{ml_irms['rms']:>14.6f}",
            f"{acts_irms['rms']:>14.6f}" if has_acts else None,
            None,
        ))
        stats.append((
            f"  └ mean after clip",
            f"{ml_irms['mean']:>14.6f}",
            f"{acts_irms['mean']:>14.6f}" if has_acts else None,
            None,
        ))
        stats.append((
            f"  └ kept / iters",
            f"  {ml_irms['frac_kept']:.1%} / {ml_irms['n_iterations']}it",
            f"  {acts_irms['frac_kept']:.1%} / {acts_irms['n_iterations']}it" if has_acts else None,
            None,
        ))
        stats.append((
            f"  └ cut range",
            f"[{ml_irms['cut_lo']:+.4f}, {ml_irms['cut_hi']:+.4f}]",
            f"[{acts_irms['cut_lo']:+.4f}, {acts_irms['cut_hi']:+.4f}]" if has_acts else None,
            None,
        ))
        stats.append((
            f"Gauss fit σ (2σ clip)",
            f"{ml_igf['sigma']:>14.6f}",
            f"{acts_igf['sigma']:>14.6f}" if has_acts else None,
            None,
        ))
        stats.append((
            f"  └ mean after clip",
            f"{ml_igf['mean']:>14.6f}",
            f"{acts_igf['mean']:>14.6f}" if has_acts else None,
            None,
        ))
        stats.append((
            f"  └ kept / iters",
            f"  {ml_igf['frac_kept']:.1%} / {ml_igf['n_iterations']}it",
            f"  {acts_igf['frac_kept']:.1%} / {acts_igf['n_iterations']}it" if has_acts else None,
            None,
        ))
        stats.append((
            f"  └ cut range",
            f"[{ml_igf['cut_lo']:+.4f}, {ml_igf['cut_hi']:+.4f}]",
            f"[{acts_igf['cut_lo']:+.4f}, {acts_igf['cut_hi']:+.4f}]" if has_acts else None,
            None,
        ))

        for j, (lbl, ml_s, acts_s, ratio_s) in enumerate(stats):
            if ratio_s is None and has_acts and j >= 2:
                try:
                    ml_val = float(ml_s.strip().replace(",", ""))
                    acts_val = float(acts_s.strip().replace(",", ""))
                    if acts_val != 0:
                        stats[j] = (lbl, ml_s, acts_s, f"{ml_val / acts_val:>14.4f}")
                except ValueError:
                    pass

        for lbl, ml_s, acts_s, ratio_s in stats:
            line = f"  {lbl:<25s}  {ml_s}"
            if has_acts:
                line += f"  {acts_s}"
                if ratio_s:
                    line += f"  {ratio_s}"
            lines.append(line)

        lines.append("")
        lines.append(f"  {'Percentile':>12s}  {'ML':>14s}")
        if has_acts:
            lines[-1] += f"  {'ACTS':>14s}  {'ML/ACTS':>10s}"
        lines.append(f"  {'-' * 12}  {'-' * 14}")
        if has_acts:
            lines[-1] += f"  {'-' * 14}  {'-' * 10}"

        ml_pcts = np.percentile(ml_res, percentiles)
        acts_pcts = np.percentile(acts_res, percentiles) if has_acts else None

        for k, pct in enumerate(percentiles):
            line = f"  {pct:>11.1f}%  {ml_pcts[k]:>14.6f}"
            if has_acts:
                line += f"  {acts_pcts[k]:>14.6f}"
                if acts_pcts[k] != 0:
                    line += f"  {ml_pcts[k] / acts_pcts[k]:>10.4f}"
            lines.append(line)

        # --- Mean residual vs η table ---
        lines.append("")
        eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
        ml_eta = ml_residuals.get("eta")
        acts_eta = acts_residuals.get("eta") if has_acts else None

        if ml_eta is not None:
            hdr = f"  {'η bin':>12s}  {'ML mean':>14s}  {'ML RMS':>14s}  {'ML n':>10s}"
            sep = f"  {'-' * 12}  {'-' * 14}  {'-' * 14}  {'-' * 10}"
            if has_acts:
                hdr += f"  {'ACTS mean':>14s}  {'ACTS RMS':>14s}  {'ACTS n':>10s}"
                sep += f"  {'-' * 14}  {'-' * 14}  {'-' * 10}"
            lines.append(hdr)
            lines.append(sep)

            for bi in range(len(eta_bins) - 1):
                lo_e, hi_e = eta_bins[bi], eta_bins[bi + 1]
                ml_mask = (ml_eta >= lo_e) & (ml_eta < hi_e)
                ml_bin = ml_res[ml_mask]
                n_ml_bin = len(ml_bin)
                if n_ml_bin > 0:
                    ml_m = float(np.mean(ml_bin))
                    ml_r = float(np.std(ml_bin))
                else:
                    ml_m, ml_r = float("nan"), float("nan")

                line = f"  [{lo_e:+5.1f},{hi_e:+5.1f}]  {ml_m:>14.6f}  {ml_r:>14.6f}  {n_ml_bin:>10,d}"

                if has_acts and acts_eta is not None:
                    acts_mask_b = (acts_eta >= lo_e) & (acts_eta < hi_e)
                    acts_bin = acts_res[acts_mask_b]
                    n_a_bin = len(acts_bin)
                    if n_a_bin > 0:
                        a_m = float(np.mean(acts_bin))
                        a_r = float(np.std(acts_bin))
                    else:
                        a_m, a_r = float("nan"), float("nan")
                    line += f"  {a_m:>14.6f}  {a_r:>14.6f}  {n_a_bin:>10,d}"

                lines.append(line)

        lines.append("")

    report = "\n".join(lines)
    (output_dir / filename).write_text(report)
    print(f"    → {output_dir / filename}")


# ============================================================================
# Common CLI arguments and data preparation
# ============================================================================

def add_common_args(parser):
    """Add CLI arguments shared by both evaluation scripts."""
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to test_predictions.h5 written by RegressionPredictionWriter",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output plots (default: same dir as predictions file)",
    )
    parser.add_argument("--eta-min", type=float, default=-3.0)
    parser.add_argument("--eta-max", type=float, default=3.0)
    parser.add_argument("--n-eta-bins", type=int, default=30)
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to preprocessed data root for ACTS comparison",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Which split to load ACTS data from (default: test)",
    )
    parser.add_argument(
        "--regime-dm-tight-selection", action="store_true",
        help="Add an extra regime applying tight kinematic cuts (pT >= 0.5 GeV, |eta| <= 3, ...) + DM",
    )


def load_all_data(args):
    """Load predictions, ACTS augmentation, and compute residuals.

    Returns a dict with all the data needed by both scripts.
    """
    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent / "eval_plots"

    print(f"Loading predictions from {pred_path}")
    data = load_predictions(pred_path)
    n_tracks = len(data["preds"]["d0"])
    print(f"  Loaded {n_tracks:,} tracks")

    acts_reco, acts_dm_mask, track_nhits = None, None, None
    if args.data_dir:
        result = load_acts_augmentation(Path(args.data_dir), split=args.split)
        if result is not None:
            acts_reco, acts_dm_mask, track_nhits = result
            n_matched = int(np.sum(~np.any(np.isnan(acts_reco), axis=1)))
            n_dm = int(np.sum(acts_dm_mask))
            print(f"  Loaded ACTS augmentation: {n_matched:,} ACTS-matched, "
                  f"{n_dm:,} double-matched out of {len(acts_reco):,} selected tracks")
            if len(acts_reco) != n_tracks:
                print(f"  WARNING: ACTS data has {len(acts_reco):,} tracks but HDF5 has "
                      f"{n_tracks:,}. Truncating to shorter.")
                min_n = min(len(acts_reco), n_tracks)
                acts_reco = acts_reco[:min_n]
                acts_dm_mask = acts_dm_mask[:min_n]
                track_nhits = track_nhits[:min_n]
                for group in ("preds", "targets"):
                    for name in data[group]:
                        data[group][name] = data[group][name][:min_n]
        else:
            print(f"  ACTS augmentation not found in {args.data_dir}")

    ml_residuals = compute_residuals(data)

    return {
        "data": data,
        "n_tracks": n_tracks,
        "ml_residuals": ml_residuals,
        "acts_reco": acts_reco,
        "acts_dm_mask": acts_dm_mask,
        "track_nhits": track_nhits,
        "output_dir": output_dir,
        "eta_range": (args.eta_min, args.eta_max),
        "n_eta_bins": args.n_eta_bins,
        "regime_dm_tight_selection": getattr(args, "regime_dm_tight_selection", False),
    }


def build_regime_data(ctx):
    """Build regime-specific data dicts from loaded context.

    Returns a list of (regime_name, kwargs_dict) tuples for run_evaluation_suite.
    """
    data = ctx["data"]
    n_tracks = ctx["n_tracks"]
    ml_residuals = ctx["ml_residuals"]
    acts_reco = ctx["acts_reco"]
    acts_dm_mask = ctx["acts_dm_mask"]
    track_nhits = ctx["track_nhits"]
    output_dir = ctx["output_dir"]
    eta_range = ctx["eta_range"]
    n_eta_bins = ctx["n_eta_bins"]

    regimes = []

    if acts_reco is not None:
        # ── Regime 1: all_selected ──
        all_acts_residuals = compute_acts_residuals(acts_reco, data["targets"])
        n_acts = len(all_acts_residuals["eta"])
        has_match = ~np.any(np.isnan(acts_reco), axis=1)
        r1_acts_reco_vals = {name: acts_reco[has_match, i] for i, name in enumerate(PARAMS)}
        r1_acts_targets = {name: data["targets"][name][has_match] for name in PARAMS}
        r1_ml_preds_matched = {name: data["preds"][name][has_match] for name in PARAMS}

        regimes.append(("all_selected", {
            "ml_residuals": ml_residuals,
            "output_dir": output_dir / "all_selected",
            "eta_range": eta_range,
            "n_eta_bins": n_eta_bins,
            "ml_label": "SSM",
            "acts_residuals": all_acts_residuals,
            "acts_label": "ACTS CKF",
            "regime_subtitle": f"{n_tracks:,} selected tracks, {n_acts:,} ACTS-matched",
            "ml_preds": data["preds"],
            "ml_targets": data["targets"],
            "acts_reco_values": r1_acts_reco_vals,
            "acts_reco_targets": r1_acts_targets,
            "ml_preds_matched": r1_ml_preds_matched,
            "nhits": track_nhits,
            "acts_nhits": track_nhits[has_match] if track_nhits is not None else None,
        }))

        # ── Regime 2: double_matched ──
        dm_mask = acts_dm_mask & ~np.any(np.isnan(acts_reco), axis=1)
        n_dm = int(np.sum(dm_mask))

        if n_dm > 0:
            dm_ml_residuals = filter_residuals(ml_residuals, dm_mask)
            dm_acts_reco = acts_reco[dm_mask]
            dm_targets = {name: data["targets"][name][dm_mask] for name in PARAMS}
            dm_acts_residuals: dict[str, np.ndarray] = {}
            for i, name in enumerate(PARAMS):
                dm_acts_residuals[name] = dm_acts_reco[:, i] - dm_targets[name]
            dm_acts_residuals["phi"] = (
                (dm_acts_residuals["phi"] + np.pi) % (2 * np.pi) - np.pi
            )
            dm_acts_residuals["eta"] = -np.log(
                np.tan(dm_targets["theta"] / 2.0 + 1e-12)
            )
            dm_preds = {name: data["preds"][name][dm_mask] for name in PARAMS}
            dm_acts_reco_vals = {name: dm_acts_reco[:, i] for i, name in enumerate(PARAMS)}

            regimes.append(("double_matched", {
                "ml_residuals": dm_ml_residuals,
                "output_dir": output_dir / "double_matched",
                "eta_range": eta_range,
                "n_eta_bins": n_eta_bins,
                "ml_label": "SSM",
                "acts_residuals": dm_acts_residuals,
                "acts_label": "ACTS CKF",
                "regime_subtitle": f"{n_dm:,}/{n_tracks:,} double-matched tracks",
                "ml_preds": dm_preds,
                "ml_targets": dm_targets,
                "acts_reco_values": dm_acts_reco_vals,
                "acts_reco_targets": dm_targets,
                "nhits": track_nhits[dm_mask] if track_nhits is not None else None,
                "acts_nhits": track_nhits[dm_mask] if track_nhits is not None else None,
            }))

        # ── Regime 3: double_matched_primary_d0 ──
        # Same as double_matched but restricted to |d0_truth| < 0.25 mm
        # to exclude secondary vertices (d0 distribution is a narrow peak).
        if n_dm > 0:
            d0_truth = data["targets"]["d0"][dm_mask]
            primary_d0_mask_local = np.abs(d0_truth) < 0.25
            n_primary = int(np.sum(primary_d0_mask_local))

            if n_primary > 0:
                pd0_ml_residuals = filter_residuals(dm_ml_residuals, primary_d0_mask_local)
                pd0_acts_reco = dm_acts_reco[primary_d0_mask_local]
                pd0_targets = {name: dm_targets[name][primary_d0_mask_local] for name in PARAMS}
                pd0_acts_residuals: dict[str, np.ndarray] = {}
                for i, name in enumerate(PARAMS):
                    pd0_acts_residuals[name] = pd0_acts_reco[:, i] - pd0_targets[name]
                pd0_acts_residuals["phi"] = (
                    (pd0_acts_residuals["phi"] + np.pi) % (2 * np.pi) - np.pi
                )
                pd0_acts_residuals["eta"] = -np.log(
                    np.tan(pd0_targets["theta"] / 2.0 + 1e-12)
                )
                pd0_preds = {name: dm_preds[name][primary_d0_mask_local] for name in PARAMS}
                pd0_acts_reco_vals = {name: pd0_acts_reco[:, i] for i, name in enumerate(PARAMS)}
                dm_nhits = track_nhits[dm_mask] if track_nhits is not None else None

                regimes.append(("double_matched_primary_d0", {
                    "ml_residuals": pd0_ml_residuals,
                    "output_dir": output_dir / "double_matched_primary_d0",
                    "eta_range": eta_range,
                    "n_eta_bins": n_eta_bins,
                    "ml_label": "SSM",
                    "acts_residuals": pd0_acts_residuals,
                    "acts_label": "ACTS CKF",
                    "regime_subtitle": f"{n_primary:,}/{n_dm:,} DM tracks, |d0_truth| < 0.25 mm",
                    "ml_preds": pd0_preds,
                    "ml_targets": pd0_targets,
                    "acts_reco_values": pd0_acts_reco_vals,
                    "acts_reco_targets": pd0_targets,
                    "nhits": dm_nhits[primary_d0_mask_local] if dm_nhits is not None else None,
                    "acts_nhits": dm_nhits[primary_d0_mask_local] if dm_nhits is not None else None,
                }))

        # ── Regime 4: double_matched_secondary_d0 ──
        # Tracks with |d0_truth| >= 0.25 mm AND |d0_pred_SSM| >= 0.25 mm
        # Excludes tracks the model pulled into the primary peak.
        if n_dm > 0:
            d0_pred = dm_preds["d0"]
            secondary_d0_mask_local = (np.abs(d0_truth) >= 0.25) & (np.abs(d0_pred) >= 0.25)
            n_secondary = int(np.sum(secondary_d0_mask_local))

            if n_secondary > 0:
                sd0_ml_residuals = filter_residuals(dm_ml_residuals, secondary_d0_mask_local)
                sd0_acts_reco = dm_acts_reco[secondary_d0_mask_local]
                sd0_targets = {name: dm_targets[name][secondary_d0_mask_local] for name in PARAMS}
                sd0_acts_residuals: dict[str, np.ndarray] = {}
                for i, name in enumerate(PARAMS):
                    sd0_acts_residuals[name] = sd0_acts_reco[:, i] - sd0_targets[name]
                sd0_acts_residuals["phi"] = (
                    (sd0_acts_residuals["phi"] + np.pi) % (2 * np.pi) - np.pi
                )
                sd0_acts_residuals["eta"] = -np.log(
                    np.tan(sd0_targets["theta"] / 2.0 + 1e-12)
                )
                sd0_preds = {name: dm_preds[name][secondary_d0_mask_local] for name in PARAMS}
                sd0_acts_reco_vals = {name: sd0_acts_reco[:, i] for i, name in enumerate(PARAMS)}
                dm_nhits_sec = track_nhits[dm_mask] if track_nhits is not None else None

                regimes.append(("double_matched_secondary_d0", {
                    "ml_residuals": sd0_ml_residuals,
                    "output_dir": output_dir / "double_matched_secondary_d0",
                    "eta_range": eta_range,
                    "n_eta_bins": n_eta_bins,
                    "ml_label": "SSM",
                    "acts_residuals": sd0_acts_residuals,
                    "acts_label": "ACTS CKF",
                    "regime_subtitle": f"{n_secondary:,}/{n_dm:,} DM tracks, |d0_truth| & |d0_pred| >= 0.25 mm",
                    "ml_preds": sd0_preds,
                    "ml_targets": sd0_targets,
                    "acts_reco_values": sd0_acts_reco_vals,
                    "acts_reco_targets": sd0_targets,
                    "nhits": dm_nhits_sec[secondary_d0_mask_local] if dm_nhits_sec is not None else None,
                    "acts_nhits": dm_nhits_sec[secondary_d0_mask_local] if dm_nhits_sec is not None else None,
                }))

        # ── Regime 5: acts_baseline_comparison ──
        theta_truth = data["targets"]["theta"]
        qop_truth = data["targets"]["qop"]
        pt_truth = np.abs(np.sin(theta_truth) / (qop_truth + 1e-12))
        eta_truth = -np.log(np.tan(np.clip(theta_truth, 1e-8, np.pi - 1e-8) / 2.0))

        base_mask = acts_dm_mask & ~np.any(np.isnan(acts_reco), axis=1)
        eta_mask = base_mask & (eta_truth >= -3.0) & (eta_truth <= 3.0)
        pt_mask = eta_mask & (pt_truth >= 0.5)
        comp_mask = pt_mask & (track_nhits >= 6)
        n_comp = int(np.sum(comp_mask))

        if n_comp > 0:
            comp_ml_residuals = filter_residuals(ml_residuals, comp_mask)
            comp_acts_reco = acts_reco[comp_mask]
            comp_targets = {name: data["targets"][name][comp_mask] for name in PARAMS}
            comp_acts_residuals: dict[str, np.ndarray] = {}
            for i, name in enumerate(PARAMS):
                comp_acts_residuals[name] = comp_acts_reco[:, i] - comp_targets[name]
            comp_acts_residuals["phi"] = (
                (comp_acts_residuals["phi"] + np.pi) % (2 * np.pi) - np.pi
            )
            comp_acts_residuals["eta"] = -np.log(
                np.tan(comp_targets["theta"] / 2.0 + 1e-12)
            )
            comp_preds = {name: data["preds"][name][comp_mask] for name in PARAMS}
            comp_acts_reco_vals = {name: comp_acts_reco[:, i] for i, name in enumerate(PARAMS)}

            regimes.append(("acts_baseline_comparison", {
                "ml_residuals": comp_ml_residuals,
                "output_dir": output_dir / "acts_baseline_comparison",
                "eta_range": eta_range,
                "n_eta_bins": n_eta_bins,
                "ml_label": "SSM",
                "acts_residuals": comp_acts_residuals,
                "acts_label": "ACTS CKF",
                "regime_subtitle": f"{n_comp:,}/{n_tracks:,} comparison tracks (DM, pT>0.5, nhits>=6)",
                "ml_preds": comp_preds,
                "ml_targets": comp_targets,
                "acts_reco_values": comp_acts_reco_vals,
                "acts_reco_targets": comp_targets,
                "nhits": track_nhits[comp_mask] if track_nhits is not None else None,
                "acts_nhits": track_nhits[comp_mask] if track_nhits is not None else None,
            }))

        # ── Regime 6 (opt-in): DM + tight kinematic cuts (no hard_scatter) ──
        if ctx.get("regime_dm_tight_selection", False):
            d0_truth_all = data["targets"]["d0"]
            z0_truth_all = data["targets"]["z0"]
            dm_base = acts_dm_mask & ~np.any(np.isnan(acts_reco), axis=1)

            cuts = [
                ("DM + not-NaN",          dm_base),
                ("pt >= 0.5 GeV",         pt_truth >= 0.5),
                ("|eta| <= 3",            (eta_truth >= -3.0) & (eta_truth <= 3.0)),
                ("nhits in [6, 20]",      (track_nhits >= 6) & (track_nhits <= 20)),
                ("|d0_truth| <= 1 mm",    np.abs(d0_truth_all) <= 1.0),
                ("|z0_truth| <= 150 mm",  np.abs(z0_truth_all) <= 150.0),
            ]
            tight_mask = np.ones(n_tracks, dtype=bool)
            print("\n  DM tight selection — cut breakdown:")
            for label, mcut in cuts:
                before = int(np.sum(tight_mask))
                tight_mask &= mcut
                after = int(np.sum(tight_mask))
                dropped = before - after
                pct = 100.0 * dropped / max(before, 1)
                print(f"    {label:<26} kept {after:>10,} / {before:>10,}  (dropped {dropped:>9,}, {pct:5.2f}%)")
            n_tight = int(np.sum(tight_mask))
            frac_total = 100.0 * n_tight / max(n_tracks, 1)
            print(f"    {'FINAL':<26} kept {n_tight:>10,} / {n_tracks:>10,}  ({frac_total:5.2f}% of all tracks)\n")

            if n_tight > 0:
                ts_ml_residuals = filter_residuals(ml_residuals, tight_mask)
                ts_acts_reco = acts_reco[tight_mask]
                ts_targets = {name: data["targets"][name][tight_mask] for name in PARAMS}
                ts_acts_residuals: dict[str, np.ndarray] = {}
                for i, name in enumerate(PARAMS):
                    ts_acts_residuals[name] = ts_acts_reco[:, i] - ts_targets[name]
                ts_acts_residuals["phi"] = (
                    (ts_acts_residuals["phi"] + np.pi) % (2 * np.pi) - np.pi
                )
                ts_acts_residuals["eta"] = -np.log(
                    np.tan(ts_targets["theta"] / 2.0 + 1e-12)
                )
                ts_preds = {name: data["preds"][name][tight_mask] for name in PARAMS}
                ts_acts_reco_vals = {name: ts_acts_reco[:, i] for i, name in enumerate(PARAMS)}

                regimes.append(("dm_tight_selection", {
                    "ml_residuals": ts_ml_residuals,
                    "output_dir": output_dir / "dm_tight_selection",
                    "eta_range": eta_range,
                    "n_eta_bins": n_eta_bins,
                    "ml_label": "SSM",
                    "acts_residuals": ts_acts_residuals,
                    "acts_label": "ACTS CKF",
                    "regime_subtitle": (
                        f"{n_tight:,}/{n_tracks:,} tracks "
                        f"(DM, pT>0.5, nhits∈[6,20], |d0|<1, |z0|<150, no hard_scatter cut)"
                    ),
                    "ml_preds": ts_preds,
                    "ml_targets": ts_targets,
                    "acts_reco_values": ts_acts_reco_vals,
                    "acts_reco_targets": ts_targets,
                    "nhits": track_nhits[tight_mask] if track_nhits is not None else None,
                    "acts_nhits": track_nhits[tight_mask] if track_nhits is not None else None,
                }))

    else:
        # Legacy mode — no ACTS augmentation
        regimes.append(("legacy", {
            "ml_residuals": ml_residuals,
            "output_dir": output_dir,
            "eta_range": eta_range,
            "n_eta_bins": n_eta_bins,
            "ml_preds": data["preds"],
            "ml_targets": data["targets"],
        }))

    return regimes


def print_precision_summary(
    precision: dict[str, dict],
    label: str,
    acts_precision: dict | None = None,
    acts_label: str = "ACTS CKF",
) -> None:
    """Print table of unbinned σ values."""
    print(f"\n  {label} — parameter precisions (unbinned σ):")
    for name in PARAMS:
        if name not in precision:
            continue
        scale = UNIT_SCALE.get(name, 1.0)
        unit = {1.0: "", 1e3: " mrad"}.get(scale, "")
        ml_str = f"{precision[name]['unbinned_std'] * scale:.5f}{unit}"
        acts_str = ""
        if acts_precision and name in acts_precision:
            acts_str = f"  |  {acts_label}: {acts_precision[name]['unbinned_std'] * scale:.5f}{unit}"
        print(f"    {name:6s}: SSM σ = {ml_str}{acts_str}")
