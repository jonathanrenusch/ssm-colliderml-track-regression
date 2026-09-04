#!/usr/bin/env python3
"""Legacy-style plots from the OFFICIAL ACTS pipeline's own tracks.

Reads the ``matched_residuals.npz`` that ``acts_integration.py --dump-residuals``
writes (per-particle truth + SSM + ACTS-KF perigee parameters, associated inside
the ACTS event loop) and renders, NEXT to the official ``resolutions.pdf``:

- ``<ds>_acts__rms_vs_eta_summary{,_logy,_preclip,_postclip}.pdf`` — the exact
  campaign design (fast_rms_eval): 2x3 grid, solid iter-3sigma / dashed pre-clip,
  unbinned mu + clipped stats in the legends;
- ``<ds>_acts__residual_hist_{liny,logy}.pdf`` — residual histograms in the same
  design, with the iter-3sigma RMSE and clipped fraction in every legend.

The comparable subset is tracks fitted by BOTH systems (the ACTS KF loses ~25%
to seeding/fit failures; per-system match counts go into the suptitle).

Usage: acts_legacy_style_plots.py <official_out_dir> <dataset_label> [kf_label]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "track_regression" / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fast_rms_eval import PARAMS, _iter_rms, _raw_rms, _save_pdf, _wrap, make_plots  # noqa: E402
from track_regression.eval_utils import iterative_rms_convergence  # noqa: E402
from track_regression.paper_plots.plots._panels import fill_eta_stephist, make_grid  # noqa: E402
from track_regression.paper_plots.stats import DISPLAY_SCALE, DISPLAY_UNIT  # noqa: E402


# TRK_ABS_ETA_MAX: fiducial |eta| cut on every page (paper default 2.0 since
# 2026-09-04: the shipped truth-KF is miscalibrated above ~80 GeV, |eta| > 2).
ETA_MAX = float(os.environ.get("TRK_ABS_ETA_MAX", "3.0"))


def load(out_dir: Path, kf_label: str):
    z = np.load(out_dir / "matched_residuals.npz")
    truth, ssm, kf = z["truth"], z["ssm"], z["kf"]
    eta_all = -np.log(np.tan(np.clip(truth[:, 3], 1e-8, np.pi - 1e-8) / 2.0))
    if ETA_MAX < 3.0:
        keep = np.abs(eta_all) <= ETA_MAX
        truth, ssm, kf = truth[keep], ssm[keep], kf[keep]
    has_ssm = np.isfinite(ssm[:, 0])
    has_kf = np.isfinite(kf[:, 0])
    both = has_ssm & has_kf
    res = {"count": int(both.sum()), "n_total": len(truth),
           "ref_name": kf_label, "second_name": None}
    for i, p in enumerate(PARAMS):
        s = ssm[both, i] - truth[both, i]
        k = kf[both, i] - truth[both, i]
        if p == "phi":
            s, k = _wrap(s), _wrap(k)
        res[f"ssm_{p}"] = s
        res[f"ckf_{p}"] = k
    th = truth[both, 3]
    res["eta"] = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    res["pt"] = np.sin(th) / np.maximum(np.abs(truth[both, 4]), 1e-12)
    return res, int(has_ssm.sum()), int(has_kf.sum())


def residual_hist_pages(res: dict, out_dir: Path, dataset: str, subtitle: str, kf_label: str):
    """Residual histograms, campaign design: legend = iter-3sigma RMSE + clipped %."""
    for logy in (False,):  # linear-y only
        fig, axes = make_grid()
        for i, p in enumerate(PARAMS):
            ax = axes[i]
            scale, unit = DISPLAY_SCALE[p], DISPLAY_UNIT[p]
            for arr, colour, tag in ((res[f"ssm_{p}"], "C0", "SSM"),
                                     (res[f"ckf_{p}"], "C3", kf_label)):
                cut = iterative_rms_convergence(arr)
                rms3, kept = cut["rms"], cut["n_kept"]
                clip_pct = 100.0 * (1.0 - kept / max(len(arr), 1))
                # +-8 rms3 at 240 bins (was +-4 / 120, 2026-09-04): the wider
                # window shrinks the out-of-range towers at the edge bins while
                # keeping the same bin width relative to the core.
                lo, hi = -8.0 * rms3, 8.0 * rms3
                ax.hist(np.clip(arr, lo, hi) * scale, bins=240,
                        range=(lo * scale, hi * scale), histtype="step",
                        color=colour, lw=1.6, density=True,
                        label=f"{tag}  iter-3σ = {rms3 * scale:.3g} {unit}\n"
                              f"({clip_pct:.1f} % clipped)")
            ax.set_xlabel(f"residual({p}) [{unit}]")
            ax.set_ylabel("density")
            ax.set_title(p)
            if logy:
                ax.set_yscale("log")
            ax.legend(loc="best", fontsize=6.4, framealpha=0.9,
                      handlelength=1.2, borderpad=0.25, labelspacing=0.2)
        fill_eta_stephist(axes[5], res["eta"])
        fig.suptitle(f"{dataset} — residuals (iterative-3σ clip in the legends) — "
                     f"total $N={res['count']:,}$ tracks fitted by both\n{subtitle}", y=1.05)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save_pdf(fig, out_dir, f"{dataset}__residual_hist_{'logy' if logy else 'liny'}")


def main():
    out_dir = Path(sys.argv[1])
    dataset = sys.argv[2]
    kf_label = sys.argv[3] if len(sys.argv) > 3 else "ACTS KF"
    # TRK_PLOT_TAG: file-stem tag ("acts" = in-pipeline refit pages, "truthkf" =
    # production truth-tracking-KF reference).  TRK_PLOT_SUBTITLE overrides the
    # provenance line under the title (the default describes the ACTS event loop).
    tag = os.environ.get("TRK_PLOT_TAG", "acts")
    res, n_ssm, n_kf = load(out_dir, kf_label)
    subtitle = os.environ.get(
        "TRK_PLOT_SUBTITLE",
        f"association inside the ACTS event loop; subset fitted by both systems "
        f"(SSM matched {n_ssm:,}, {kf_label} {n_kf:,} of {res['n_total']:,})")
    make_plots(res, out_dir, f"{dataset}_{tag}", subtitle)
    residual_hist_pages(res, out_dir, f"{dataset}_{tag}", subtitle, kf_label)
    print(f"[legacy-style] wrote {dataset}_{tag}__* to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
