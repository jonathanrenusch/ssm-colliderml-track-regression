#!/usr/bin/env python3
"""Cross-architecture comparison table for the v2 ablation arms.

Reads every ``rms_summary.json`` under ``eval_plots/ablations_2026-09/v2_evals``
and prints, per test set, the ratio of each arm's iterative-3-sigma-clipped
RMSE to the truth-KF shipped with the data, plus the geometric mean over the
five parameters (GM5).  Pre-clip ratios come with it, because the clipped
number alone hides the tails (campaign rule, CLAUDE.md 4.24).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "eval_plots/ablations_2026-09/v2_evals"
PARAMS = ["d0", "z0", "phi", "theta", "qop"]
SETS = ["single_muon_2GeV", "single_muon_10GeV", "single_muon_50GeV",
        "single_muon_100GeV", "single_muon_uniform", "ttbar_new_pt1"]


def _get(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None


def load(run_dir: Path):
    js = run_dir / "plots" / "rms_summary.json"
    if not js.exists():
        js = run_dir / "plots_eta2" / "rms_summary.json"
    if not js.exists():
        return None
    return json.loads(js.read_text())


def ratios(entry, preclip=False):
    """-> {param: model/reference} for one dataset entry of rms_summary.json.

    The ``ckf_`` prefix is historical: ``fast_rms_eval`` writes the reference
    column under that name even when the reference is the truth-seeded KF
    shipped with the data, which is what these eval farms carry.
    """
    sfx = "pre" if preclip else "post"
    out = {}
    for p in PARAMS:
        ssm, ref = entry.get(f"{p}_ssm_{sfx}"), entry.get(f"{p}_ckf_{sfx}")
        if ssm is None or not ref:
            return None
        out[p] = ssm / ref
    return out


def gm(r):
    return math.exp(sum(math.log(v) for v in r.values()) / len(r))


def main():
    arms = sorted(d for d in ROOT.iterdir() if d.is_dir()) if ROOT.exists() else []
    data = {}
    for a in arms:
        js = load(a)
        if js:
            data[a.name] = js
    if not data:
        print(f"no evaluated arms under {ROOT}")
        return 1

    for preclip in (False, True):
        tag = "PRE-clip (tail-inclusive)" if preclip else "POST-clip (iter-3sigma)"
        print(f"\n=== {tag}: arm / truth-KF, GM5 over d0 z0 phi theta qop ===")
        print(f"{'arm':<26}" + "".join(f"{s.replace("single_muon_",""):>14}" for s in SETS))
        for name, js in sorted(data.items()):
            row = f"{name:<26}"
            for s in SETS:
                e = js.get(s)
                r = ratios(e, preclip) if e else None
                row += f"{gm(r):>14.3f}" if r else f"{'-':>13}"
            print(row)

    print("\n=== POST-clip per parameter ===")
    for s in SETS:
        print(f"\n-- {s}")
        print(f"{'arm':<26}" + "".join(f"{p:>9}" for p in PARAMS) + f"{'GM5':>9}")
        for name, js in sorted(data.items()):
            e = js.get(s)
            r = ratios(e) if e else None
            if not r:
                continue
            print(f"{name:<26}" + "".join(f"{r[p]:>9.3f}" for p in PARAMS)
                  + f"{gm(r):>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
