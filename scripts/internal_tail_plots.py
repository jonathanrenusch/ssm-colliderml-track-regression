#!/usr/bin/env python3
"""INTERNAL data-science / debugging pages (not for the paper), 2026-09-04:

1. <ds>__tails_logy.pdf     — residual histograms on a log-y axis over ±40
   iter-3σ, so the far-tail bins the RMS clip removes are actually visible;
   legend: pre-clip RMS, iter-3σ RMS, max|res|, and the fraction of tracks
   beyond 3σ / 10σ / 30σ.
2. <ds>__rms_vs_d0z0.pdf    — iter-3σ RMSE of every parameter binned in truth
   |d0| and truth z0 (SSM vs truth-KF + ratio strip): does precision depend on
   the impact parameters?

Legacy (fast_rms_eval) pipeline, i.e. the flat-store h5 predictions + the
truth_kf_reco side-cars — not the ACTS pipeline.  No |eta| cut (do NOT export
TRK_ABS_ETA_MAX when running this).

Usage: internal_tail_plots.py <pred_dir> <store_root> <out_dir> [datasets...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "track_regression" / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

from fast_rms_eval import load_flat_acts, load_truth_kf, _wrap  # noqa: E402
from track_regression.eval_utils import PARAMS, iterative_rms_convergence  # noqa: E402
from track_regression.paper_plots.stats import DISPLAY_SCALE, DISPLAY_UNIT  # noqa: E402

COL = {"SSM": "C0", "truth-KF": "C3"}


def load(h5_path: Path, store_dir: Path):
    with h5py.File(h5_path, "r") as f:
        preds = {p: f["preds"][p][:] for p in PARAMS}
        targets = {p: f["targets"][p][:] for p in PARAMS}
    acts, dm_mask = load_flat_acts(store_dir)
    tkf = load_truth_kf(store_dir)
    n = len(targets["d0"])
    acts, dm_mask = acts[:n], dm_mask[:n]
    if tkf is None:
        raise SystemExit(f"{store_dir}: no truth_kf_reco side-car")
    tkf = np.asarray(tkf)[:n]
    dm = np.asarray(dm_mask, bool) & np.isfinite(acts[:, 0]) & np.isfinite(tkf[:, 0])
    res = {}
    for i, p in enumerate(PARAMS):
        s = preds[p][dm] - targets[p][dm]
        k = tkf[dm, i] - targets[p][dm]
        if p == "phi":
            s, k = _wrap(s), _wrap(k)
        res[("SSM", p)] = s
        res[("truth-KF", p)] = k
    tr = {p: targets[p][dm] for p in PARAMS}
    return res, tr, int(dm.sum())


def tails_page(res, tr, n, ds, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    axes = axes.ravel()
    for i, p in enumerate(PARAMS):
        ax = axes[i]
        sc, unit = DISPLAY_SCALE[p], DISPLAY_UNIT[p]
        for tag in ("SSM", "truth-KF"):
            arr = res[(tag, p)]
            c = iterative_rms_convergence(arr)
            rms3 = c["rms"]
            pre = float(np.sqrt(np.mean(np.asarray(arr, np.float64) ** 2)))
            f3 = 100.0 * np.mean(np.abs(arr) > 3 * rms3)
            f10 = 100.0 * np.mean(np.abs(arr) > 10 * rms3)
            f30 = 100.0 * np.mean(np.abs(arr) > 30 * rms3)
            lo, hi = -40 * rms3, 40 * rms3
            ax.hist(np.clip(arr, lo, hi) * sc, bins=400, range=(lo * sc, hi * sc),
                    histtype="step", color=COL[tag], lw=1.2,
                    label=(f"{tag}: iter-3σ {rms3*sc:.3g} {unit}, pre-clip {pre*sc:.3g}\n"
                           f">3σ {f3:.2f}%  >10σ {f10:.3f}%  >30σ {f30:.4f}%  "
                           f"max {np.max(np.abs(arr))*sc:.3g}"))
        ax.set_yscale("log")
        ax.set_xlabel(f"residual({p}) [{unit}]"); ax.set_ylabel("tracks / bin")
        ax.set_title(f"{p} — tails (axis = ±40 iter-3σ, overflow in edge bins)")
        ax.legend(fontsize=6.0, loc="upper right", framealpha=0.9, labelspacing=0.3)
    eta = -np.log(np.tan(np.clip(tr["theta"], 1e-8, np.pi - 1e-8) / 2.0))
    # 6th panel: where do the >10σ d0 tails live in (eta, pT)?
    bad = np.abs(res[("SSM", "d0")]) > 10 * iterative_rms_convergence(res[("SSM", "d0")])["rms"]
    axes[5].hist(eta, bins=60, histtype="step", color="0.4", density=True, label="all tracks")
    if bad.sum() > 10:
        axes[5].hist(eta[bad], bins=60, histtype="step", color="C1", density=True,
                     label=f"SSM |d0 res| > 10σ (N={bad.sum():,})")
    axes[5].set_xlabel(r"truth $\eta$"); axes[5].set_ylabel("density")
    axes[5].set_title("where the d0 far-tails live"); axes[5].legend(fontsize=7)
    fig.suptitle(f"{ds} — far-tail diagnostics (INTERNAL, no |η| cut) — N={n:,} DM tracks", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_dir / f"{ds}__tails_logy.pdf", bbox_inches="tight")
    plt.close(fig)


def rms_vs_impact_page(res, tr, n, ds, out_dir):
    """iter-3σ RMS of each parameter vs truth |d0| and truth z0 + ratio strips."""
    for var, vals, xlabel, stem in (
        ("absd0", np.abs(tr["d0"]), r"truth $|d_0|$ [mm]", "rms_vs_absd0"),
        ("z0", tr["z0"], r"truth $z_0$ [mm]", "rms_vs_z0"),
    ):
        edges = (np.quantile(vals, np.linspace(0, 1, 21)) if var == "absd0"
                 else np.linspace(-270, 270, 25))
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(vals, edges) - 1, 0, len(centers) - 1)
        fig = plt.figure(figsize=(15, 9.4))
        gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27)
        for i, p in enumerate(PARAMS):
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i],
                                          height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(sub[0]); axr = fig.add_subplot(sub[1], sharex=ax)
            sc, unit = DISPLAY_SCALE[p], DISPLAY_UNIT[p]
            curves = {}
            for tag in ("SSM", "truth-KF"):
                arr = res[(tag, p)]
                y = np.full(len(centers), np.nan)
                for b in range(len(centers)):
                    m = idx == b
                    if m.sum() > 100:
                        y[b] = iterative_rms_convergence(arr[m])["rms"]
                curves[tag] = y
                ok = np.isfinite(y)
                ax.plot(centers[ok], y[ok] * sc, "-", color=COL[tag], lw=1.6, label=tag)
            ax.set_ylabel(f"iter-3σ RMS({p}) [{unit}]", fontsize=9)
            ax.set_title(p); ax.set_ylim(bottom=0); ax.legend(fontsize=7)
            r = curves["SSM"] / curves["truth-KF"]
            axr.axhline(1.0, color="0.4", lw=0.8, ls=":")
            axr.plot(centers, r, "-", color="C0", lw=1.4)
            axr.set_ylabel("SSM/tKF", fontsize=8); axr.set_xlabel(xlabel)
            plt.setp(ax.get_xticklabels(), visible=False)
        ax6 = fig.add_subplot(gs[5])
        ax6.hist(vals, bins=edges, histtype="step", color="0.3", lw=1.4)
        ax6.set_xlabel(xlabel); ax6.set_ylabel("tracks / bin"); ax6.set_title("track distribution")
        fig.suptitle(f"{ds} — iter-3σ RMS vs {xlabel} (INTERNAL, no |η| cut) — N={n:,} DM tracks",
                     y=0.995, fontsize=11)
        fig.savefig(out_dir / f"{ds}__{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    pred_dir, root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    datasets = sys.argv[4:] or ["single_muon_uniform", "ttbar_new_pt1"]
    out.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        h5 = pred_dir / f"{ds}.h5"
        store = root / ds / "test"
        if not h5.exists() or not store.exists():
            print(f"[skip] {ds}")
            continue
        res, tr, n = load(h5, store)
        tails_page(res, tr, n, ds, out)
        rms_vs_impact_page(res, tr, n, ds, out)
        print(f"[internal] {ds}: tails + rms_vs_d0/z0 -> {out}")


if __name__ == "__main__":
    main()
