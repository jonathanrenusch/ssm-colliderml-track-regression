#!/usr/bin/env python3
"""LaTeX bodies for the encoder-ablation tables, with bootstrap uncertainties.

Two views of the same measurement, both as ratios to the truth-seeded KF on
the double-matched tracks, |eta| <= 2 (and pT <= 70 GeV on the uniform sample,
the paper's convention):

    main      one table, the uniform-momentum sample only -- the broadest
              test set -- with all five perigee parameters and their
              geometric mean, one row per backbone.
    appendix  the same, for every muon test sample.

Uncertainties come from the paired bootstrap of paper_tab_ratios.py: one set
of resampled track indices per replica, reused for the model, the reference
and all five parameters, with the iterative clip recomputed inside the
replica.

    abl_v2_arch_table.py {main|appendix} [n_boot] [residual_root]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_tab_ratios import _fmt, rms3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "eval_plots/ablations_2026-09/v2_residuals"
FT_BUNDLE = (Path(__file__).resolve().parents[1]
             / "eval_plots/paper_plots/truthkf_minGRU_FT_fp16_eta2")

SETS = [("single_muon_2GeV", r"$\mu$, 2\,GeV"),
        ("single_muon_10GeV", r"$\mu$, 10\,GeV"),
        ("single_muon_50GeV", r"$\mu$, 50\,GeV"),
        ("single_muon_uniform", r"$\mu$, 1--70\,GeV")]
# The one-directional Mamba-2 arm is held back from the paper (2026-09-20).
ARMS = [
    ("V2_mingru_25ep", r"minGRU"),
    ("SSM_baseline_25ep", r"Mamba-2, bidirectional"),
    ("V2_txf_25ep", r"Transformer"),
    ("V2_diagssm_25ep", r"diagonal SSM, non-selective"),
]
ETA_MAX, PT_MAX = 2.0, 70.0


def load(bundle: Path, ds: str):
    """-> list of (ssm_residual, kf_residual) per parameter, after the cuts."""
    z = np.load(bundle / ds / "matched_residuals.npz")
    truth, ssm, kf = z["truth"], z["ssm"], z["kf"]
    theta = truth[:, 3]
    eta = -np.log(np.tan(np.clip(theta, 1e-9, np.pi - 1e-9) / 2.0))
    m = np.abs(eta) <= ETA_MAX
    if ds == "single_muon_uniform":
        pt = np.sin(theta) / np.maximum(np.abs(truth[:, 4]), 1e-12)
        m &= pt <= PT_MAX
    out = []
    for j in range(5):
        rs, rk = ssm[m, j] - truth[m, j], kf[m, j] - truth[m, j]
        if j == 2:
            rs = (rs + np.pi) % (2 * np.pi) - np.pi
            rk = (rk + np.pi) % (2 * np.pi) - np.pi
        out.append((np.ascontiguousarray(rs), np.ascontiguousarray(rk)))
    return out, int(m.sum())


def row(pairs, n, n_boot, with_gm):
    """Point ratios (+ geometric mean) and their bootstrap errors."""
    def once(idx=None):
        r = [rms3(a if idx is None else a[idx]) / rms3(b if idx is None else b[idx])
             for a, b in pairs]
        return r + [math.exp(sum(math.log(v) for v in r) / 5)] if with_gm else r
    val = once()
    err = [None] * len(val)
    if n_boot:
        rng = np.random.default_rng(12345)
        reps = np.array([once(rng.integers(0, n, n)) for _ in range(n_boot)])
        err = list(reps.std(axis=0, ddof=1))
    return " & ".join(_fmt(v, e) for v, e in zip(val, err))


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    root = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT
    arms = list(ARMS)
    if mode == "main":
        # The shipped model is the fine-tuned minGRU; show it next to the
        # stage-1 arms so the reader sees what the second stage adds.
        arms = [(FT_BUNDLE, r"minGRU, fine-tuned (this work)")] + \
               [(root / a, lab) for a, lab in arms]
        for bundle, label in arms:
            pairs, n = load(Path(bundle), "single_muon_uniform")
            print(f"    {label:<30s} & {row(pairs, n, n_boot, True)} \\\\   % N = {n:,}")
        return 0
    for ds, dlabel in SETS:
        print(f"    \\multicolumn{{7}}{{l}}{{\\itshape {dlabel}}} \\\\")
        for a, label in ARMS:
            pairs, n = load(root / a, ds)
            print(f"    \\quad {label:<26s} & {row(pairs, n, n_boot, True)} \\\\")
        print(r"    \addlinespace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
