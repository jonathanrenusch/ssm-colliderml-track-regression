#!/usr/bin/env python3
"""Paper version of the targets_and_kinematics page (user request 2026-09-06).

Same content as plot_preprocessed.plot_targets -- the five regression targets
plus pT / eta / sequence length of a preprocessed flat store -- but in the
paper's performance-plot design: C0-blue step histograms with the translucent
C0 fill used for the RMS-band uncertainties, read from the v2 training store
(the dataset_plots/distributions page of 2026-08-23 was made from the
deprecated pre-v2 store).

Usage: targets_kinematics_paper.py [store_split_dir] [out_pdf] [max_tracks]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from track_regression.scripts.plot_preprocessed import Store, style  # noqa: E402

PANELS = [  # (label, log_y)
    ("$d_0$ [mm]", False), ("$z_0$ [mm]", False), (r"$\varphi$ [rad]", False),
    (r"$\theta$ [rad]", False), ("$q/p$ [1/GeV]", True),
    (r"$p_{\mathrm{T}}$ [GeV]", False), (r"$\eta$", False), ("hits per track", False),
]


def hist(ax, v, *, bins=120, rng=None, log=False, discrete=False):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if discrete:
        u = np.unique(v)
        b = np.arange(u.min() - 0.5, u.max() + 1.5)
    else:
        lo, hi = np.percentile(v, [0.02, 99.98]) if rng is None else rng
        b = np.linspace(lo, hi, bins + 1)
    h, edges = np.histogram(v, bins=b)
    # paper design: C0 step outline + the translucent C0 fill of the RMS bands
    ax.stairs(h, edges, fill=True, color="C0", alpha=0.25, lw=0)
    ax.stairs(h, edges, color="C0", lw=1.6)
    if log:
        ax.set_yscale("log")
        nz = h[h > 0]
        if nz.size:
            ax.set_ylim(bottom=max(0.5, nz.min() * 0.7))
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel("tracks (log)" if log else "tracks")


def main():
    split = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/scratch/colliderml/ICLR_retraining_v2/single_muon_uniform/train")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        REPO / "eval_plots/paper_plots/targets_kinematics/single_muon_uniform_targets_kinematics.pdf"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
    style()
    st = Store(split)
    T, PT, ETA, NH = st.targets(n)
    cols = [T[:, 0], T[:, 1], T[:, 2], T[:, 3], T[:, 4], PT, ETA, NH]
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.2))
    for a, v, (lab, lg) in zip(ax.flatten(), cols, PANELS):
        hist(a, v, log=lg, discrete=(lab == "hits per track"),
             rng=(-3.05, 3.05) if lab == r"$\eta$" else None)
        a.set_xlabel(lab)
    fig.suptitle(f"single_muon_uniform --- regression targets and kinematics, "
                 f"training store ({len(T):,} tracks sampled)", y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[targets] {out}", flush=True)


if __name__ == "__main__":
    main()
