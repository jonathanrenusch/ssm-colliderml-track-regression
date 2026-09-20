#!/usr/bin/env python3
"""Per-parameter detail for ONE ablation arm: absolute values and ratios.

Usage: abl_v2_arm_detail.py <arm> [post|pre]

The geometric mean hides which parameter is carrying a difference, so this
prints every parameter separately: the arm's iterative-3-sigma-clipped RMSE,
the truth-KF's on the same double-matched tracks, and their ratio.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "eval_plots/ablations_2026-09/v2_evals"
PARAMS = [("d0", "um", 1.0), ("z0", "um", 1.0), ("phi", "mrad", 1.0),
          ("theta", "mrad", 1.0), ("qop", "1e-4/GeV", 1e4)]
SETS = ["single_muon_2GeV", "single_muon_10GeV", "single_muon_50GeV",
        "single_muon_100GeV", "single_muon_uniform", "ttbar_new_pt1"]


def main(arm: str, which: str = "post") -> int:
    js_path = ROOT / arm / "plots" / "rms_summary.json"
    if not js_path.exists():
        print(f"no evaluation at {js_path}")
        return 1
    js = json.loads(js_path.read_text())
    sfx = "post" if which == "post" else "pre"
    label = "iter-3sigma clipped" if sfx == "post" else "pre-clip (tail-inclusive)"

    print(f"\n{arm} — {label} RMSE, arm / truth-KF (|eta| <= 2, deployment settings)\n")
    head = f"{'dataset':<20}{'N':>9}"
    for p, u, _ in PARAMS:
        head += f"{p + ' [' + u + ']':>22}"
    head += f"{'GM5':>8}"
    print(head)
    print("-" * len(head))
    for s in SETS:
        e = js.get(s)
        if not e:
            continue
        row = f"{s.replace('single_muon_', 'mu '):<20}{e.get('n_dm', 0):>9,}"
        ratios = []
        for p, _, scale in PARAMS:
            a, r = e.get(f"{p}_ssm_{sfx}"), e.get(f"{p}_ckf_{sfx}")
            if a is None or not r:
                row += f"{'-':>22}"
                continue
            ratios.append(a / r)
            row += f"{a * scale:8.3f}/{r * scale:<7.3f}{a / r:5.2f}"
        row += f"{math.exp(sum(map(math.log, ratios)) / len(ratios)):>8.3f}" if ratios else ""
        print(row)
    print("\nEach cell: arm / truth-KF, then their ratio. The reference is the")
    print("truth-seeded KF shipped with the data, on the double-matched subset.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "post"))
