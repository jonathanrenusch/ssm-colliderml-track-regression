"""Compute per-run summary stats with bootstrap σ; emit stats.txt + stats.json."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from track_regression.eval_utils import (
    PARAMS,
    UNIT_SCALE,
    iterative_rms_convergence,
)

from .bootstrap import bootstrap_paired


# Display unit conversion (rad → mrad, mm → µm) — purely cosmetic for stats.txt
DISPLAY_SCALE = {"d0": 1e3, "z0": 1e3, "phi": 1e3, "theta": 1e3, "qop": 1.0}
DISPLAY_UNIT = {"d0": "µm", "z0": "µm", "phi": "mrad", "theta": "mrad", "qop": "1/GeV"}


def _raw_std(x: np.ndarray) -> float:
    return float(np.std(x))


def _iqr_robust_sigma(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25.0, 75.0])
    return float((q3 - q1) / 1.349)


def _iter3sigma_rms(x: np.ndarray) -> float:
    return iterative_rms_convergence(x)["rms"]


METRICS = {
    "raw_std": _raw_std,
    "iqr_robust_sigma": _iqr_robust_sigma,
    "iter3sigma_rms": _iter3sigma_rms,
}


def compute_stats(
    res: dict,
    *,
    n_boot: int = 200,
    seed: int = 0,
    n_workers: int = 5,
) -> dict:
    """Compute per-(param, metric) bootstrap stats for SSM and CKF + ratio.

    Returns a nested dict::

        {param: {metric: {"ssm": (mean, std), "ckf": (mean, std), "ratio": (mean, std)}}}

    Plus a ``count`` field with the DM track count.
    """
    out: dict = {"count": int(res.get("count", 0))}

    def _one(param: str) -> tuple[str, dict]:
        ssm = res[f"ssm_{param}"]
        ckf = res[f"ckf_{param}"]
        per_metric = {}
        for mname, fn in METRICS.items():
            r = bootstrap_paired(ssm, ckf, fn, n=n_boot, seed=seed)
            per_metric[mname] = {"ssm": r["a"], "ckf": r["b"], "ratio": r["ratio"]}
        return param, per_metric

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for param, per_metric in ex.map(_one, PARAMS):
            out[param] = per_metric

    return out


def _fmt(val: float, sig: float, scale: float, sci: bool = False) -> str:
    """Format mean ± 2σ (95% CI under normal-bootstrap assumption)."""
    if not np.isfinite(val):
        return "    nan      "
    s = 2.0 * sig
    if sci:
        return f"{val * scale:.4e} ± {s * scale:.1e}"
    return f"{val * scale:.4f} ± {s * scale:.4f}"


def write_stats(stats: dict, bundle_dir: Path) -> None:
    """Write stats.json and a human-readable stats.txt."""
    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    lines: list[str] = []
    lines.append(f"Double-matched track count: {stats['count']:,}")
    lines.append("All ± uncertainties below are bootstrap **2σ** (95% CI).")
    lines.append("")

    # Three blocks: pre-clip RMS (= raw std), IQR/1.349, iter-3σ RMS
    blocks = [
        ("raw_std",          "Raw standard deviation (pre-clip RMS, tail-inclusive)"),
        ("iqr_robust_sigma", "IQR / 1.349 (robust core σ, no clipping)"),
        ("iter3sigma_rms",   "Iterative 3σ-clipped RMS (post-clip core)"),
    ]
    for metric, header in blocks:
        lines.append("=" * 92)
        lines.append(header)
        lines.append("=" * 92)
        lines.append(f"  {'param':<7} {'unit':<7} {'SSM':>22} {'CKF':>22} {'ratio SSM/CKF':>20}")
        for p in PARAMS:
            scale = DISPLAY_SCALE[p]
            unit = DISPLAY_UNIT[p]
            r = stats[p][metric]
            sci = (p == "qop")
            lines.append(
                f"  {p:<7} {unit:<7} "
                f"{_fmt(r['ssm'][0], r['ssm'][1], scale, sci):>22} "
                f"{_fmt(r['ckf'][0], r['ckf'][1], scale, sci):>22} "
                f"{r['ratio'][0]:>10.3f} ± {2 * r['ratio'][1]:.3f}"
            )
        lines.append("")

    # Ratio summary table
    lines.append("=" * 92)
    lines.append("Summary: SSM/CKF ratios (pre- and post-clip)")
    lines.append("=" * 92)
    lines.append(f"  {'param':<7} {'pre-clip ratio (raw)':>26} {'IQR ratio':>20} {'post-clip ratio (iter-3σ)':>30}")
    for p in PARAMS:
        a = stats[p]["raw_std"]["ratio"]
        b = stats[p]["iqr_robust_sigma"]["ratio"]
        c = stats[p]["iter3sigma_rms"]["ratio"]
        lines.append(
            f"  {p:<7} "
            f"{a[0]:>13.3f} ± {2 * a[1]:.3f}    "
            f"{b[0]:>10.3f} ± {2 * b[1]:.3f}    "
            f"{c[0]:>16.3f} ± {2 * c[1]:.3f}"
        )
    lines.append("")

    with open(bundle_dir / "stats.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
