#!/usr/bin/env python3
"""One figure for the encoder ablation: ratio to the truth-KF, every arm.

Why a dot plot and not overlaid resolution curves: the arms agree so closely
that overlaid RMS-vs-pT curves are degenerate -- four of five lines are hidden
under the fifth and the eye sees nothing.  Here parity IS the signal, so the
figure has to make agreement legible: a tight cluster means the encoders are
interchangeable, and a single displaced colour in a single panel is the one
architecture that is not.

Filled marker = post-clip (iterative 3-sigma).  Open marker = pre-clip
(tail-inclusive).  The pair reads as the range a model spans once tails count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "eval_plots/ablations_2026-09/v2_evals"
ARMS = [
    ("SSM_baseline_25ep", "Mamba-2, bidirectional", "C0", "o"),
    ("V2_txf_25ep", "Transformer", "C4", "^"),
    ("V2_mingru_25ep", "minGRU", "C5", "D"),
    ("V2_diagssm_25ep", "diagonal SSM, non-selective", "C1", "v"),
]
# The one-directional Mamba arm is held back from the paper (user decision
# 2026-09-20): it ties the bidirectional reference, which would force
# rewriting the bidirectionality motivation in intro/method and is not
# supported by the current abstract.  Pass --with-1dir for the internal page.
ARM_1DIR = ("V2_mamba1dir_25ep", "Mamba-2, one-directional", "C2", "s")
PARAMS = ["d0", "z0", "phi", "theta", "qop"]
MATH = {"d0": r"$d_0$", "z0": r"$z_0$", "phi": r"$\varphi$",
        "theta": r"$\theta$", "qop": r"$q/p$"}
SETS = [("single_muon_2GeV", r"$\mu$, 2 GeV"), ("single_muon_10GeV", r"$\mu$, 10 GeV"),
        ("single_muon_50GeV", r"$\mu$, 50 GeV"), ("single_muon_uniform", r"$\mu$, uniform")]


def main(out: str, with_1dir: bool = False) -> int:
    if with_1dir:
        ARMS.insert(1, ARM_1DIR)
    data = {}
    for arm, *_ in ARMS:
        p = ROOT / arm / "plots" / "rms_summary.json"
        if p.exists():
            data[arm] = json.loads(p.read_text())
    if len(data) < 2:
        print(f"need >=2 evaluated arms under {ROOT}")
        return 1

    fig, axes = plt.subplots(1, 5, figsize=(15.0, 4.1), sharey=True)
    ny = len(SETS)
    off = np.linspace(0.30, -0.30, len(ARMS))       # stagger arms within a row

    for j, prm in enumerate(PARAMS):
        ax = axes[j]
        lo, hi = 1.0, 1.0
        for i, (s, _) in enumerate(SETS):
            y0 = ny - 1 - i
            ax.axhspan(y0 - 0.5, y0 + 0.5, color="0.95" if i % 2 else "white", lw=0)
            for k, (arm, lab, col, mk) in enumerate(ARMS):
                e = data.get(arm, {}).get(s)
                if not e:
                    continue
                post = e[f"{prm}_ssm_post"] / e[f"{prm}_ckf_post"]
                pre = e[f"{prm}_ssm_pre"] / e[f"{prm}_ckf_pre"]
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

    # two legends: the architectures, and separately what the marker fill means
    # (the fill convention is data, not a caption aside -- it gets its own key)
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
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[dotplot] {out}  ({len(data)} arms)")
    return 0


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--with-1dir"]
    sys.exit(main(argv[0] if argv
                  else "eval_plots/paper_plots/ablation_compare/ablation_dotplot.pdf",
                  with_1dir="--with-1dir" in sys.argv))
