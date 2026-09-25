#!/usr/bin/env python3
"""Encoder-ablation dot plot: resolution ratio to the truth-seeded KF, every arm.

Each arm is given as a fast_rms_eval.py output directory (holding
rms_summary.json).  One panel per perigee parameter, one row per muon test
sample, arms staggered within a row.  Filled marker = iterative-3-sigma-clipped
RMS ratio, open marker = un-clipped (tail-inclusive) RMS ratio, joined by a line.

A dot plot rather than overlaid resolution curves: the arms agree so closely
that overlaid curves hide one another, while here a tight cluster reads as
parity and a displaced colour singles out the arm that differs.

Usage:
  plot_encoder_ablation.py --mamba2 DIR --transformer DIR --mingru DIR --diagssm DIR
                           [--out figures/encoder_ablation_dotplot.pdf]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# (CLI key, legend label, colour, marker), drawn top to bottom within a row
ARMS = [
    ("mamba2", "Mamba-2, bidirectional", "C0", "o"),
    ("transformer", "Transformer", "C4", "^"),
    ("mingru", "minGRU", "C5", "D"),
    ("diagssm", "diagonal SSM, non-selective", "C1", "v"),
]
PARAMS = ["d0", "z0", "phi", "theta", "qop"]
MATH = {"d0": r"$d_0$", "z0": r"$z_0$", "phi": r"$\varphi$",
        "theta": r"$\theta$", "qop": r"$q/p$"}
SETS = [("single_muon_2GeV", r"$\mu$, 2 GeV"), ("single_muon_10GeV", r"$\mu$, 10 GeV"),
        ("single_muon_50GeV", r"$\mu$, 50 GeV"), ("single_muon_uniform", r"$\mu$, uniform")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, label, *_ in ARMS:
        ap.add_argument(f"--{key}", type=Path, required=True,
                        help=f"fast_rms_eval output dir, {label}")
    ap.add_argument("--out", type=Path, default=Path("figures/encoder_ablation_dotplot.pdf"))
    a = ap.parse_args()
    data = {key: json.loads((getattr(a, key) / "rms_summary.json").read_text())
            for key, *_ in ARMS}

    fig, axes = plt.subplots(1, 5, figsize=(15.0, 4.1), sharey=True)
    ny = len(SETS)
    off = np.linspace(0.30, -0.30, len(ARMS))       # stagger arms within a row

    for j, prm in enumerate(PARAMS):
        ax = axes[j]
        lo, hi = 1.0, 1.0
        for i, (s, _) in enumerate(SETS):
            y0 = ny - 1 - i
            ax.axhspan(y0 - 0.5, y0 + 0.5, color="0.95" if i % 2 else "white", lw=0)
            for k, (key, lab, col, mk) in enumerate(ARMS):
                e = data[key].get(s)
                if not e:
                    continue
                post = e[f"{prm}_ssm_post"] / e[f"{prm}_ref_post"]
                pre = e[f"{prm}_ssm_pre"] / e[f"{prm}_ref_pre"]
                y = y0 + off[k]
                ax.plot([pre, post], [y, y], color=col, lw=1.0, alpha=0.45, zorder=2)
                ax.plot(pre, y, mk, mfc="none", mec=col, ms=5, mew=1.2, zorder=3)
                ax.plot(post, y, mk, color=col, ms=5, zorder=4,
                        label=lab if (i == 0 and j == 0) else None)
                lo, hi = min(lo, pre, post), max(hi, pre, post)
        ax.axvline(1.0, color="k", lw=1.0, ls=":", zorder=1)
        pad = 0.05 * (hi - lo) + 0.004
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, ny - 0.4)
        ax.set_title(MATH[prm], fontsize=13)
        ax.set_xlabel("ratio to truth-KF")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(range(ny))
    axes[0].set_yticklabels([lbl for _, lbl in SETS][::-1], fontsize=10)

    # two legends: the architectures, and what the marker fill means
    h, l = axes[0].get_legend_handles_labels()
    leg_arms = fig.legend(h, l, loc="lower center", ncol=len(ARMS), frameon=False,
                          fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.add_artist(leg_arms)
    fill_keys = [
        plt.Line2D([], [], marker="o", color="0.25", ls="-", lw=1.0, ms=5,
                   label=r"3$\sigma$-clipped RMS"),
        plt.Line2D([], [], marker="o", mfc="none", mec="0.25", color="0.25",
                   ls="-", lw=1.0, ms=5, mew=1.2, label="un-clipped RMS"),
    ]
    fig.legend(handles=fill_keys, loc="lower center", ncol=2, frameon=True,
               framealpha=0.9, fontsize=9.5, bbox_to_anchor=(0.5, 0.055),
               title="marker fill", title_fontsize=9.5)
    fig.suptitle("Encoder ablation — identical recipe, data, schedule and seed; "
                 r"$|\eta|\leq2$", fontsize=11.5, y=0.99)
    fig.subplots_adjust(top=0.80, bottom=0.30, left=0.085, right=0.99, wspace=0.16)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[dotplot] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
