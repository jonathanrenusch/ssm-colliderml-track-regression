#!/usr/bin/env python3
"""Pre- vs post-clip ratio tables for one or more ablation arms.

The campaign's discriminator between models that already sit at the truth-KF is
the PRE-clip (tail-inclusive) number, not the clipped one (CLAUDE.md 4.24): the
clipped RMSE saturates at the reference once the core is right, while the
un-clipped value still separates the tails.  This prints both, per parameter,
per test set, for every arm given -- so a fine-tune can be read against its own
stage-1 checkpoint.

Usage: abl_v2_preclip_table.py <arm> [<arm> ...]
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


def load(arm: str):
    for sub in ("plots", "plots_eta2"):
        p = ROOT / arm / sub / "rms_summary.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def ratios(e, sfx):
    out = {}
    for p in PARAMS:
        a, r = e.get(f"{p}_ssm_{sfx}"), e.get(f"{p}_ckf_{sfx}")
        if a is None or not r:
            return None
        out[p] = a / r
    return out


def gm(r):
    return math.exp(sum(map(math.log, r.values())) / len(r))


def main(arms: list[str]) -> int:
    data = {a: load(a) for a in arms}
    missing = [a for a, d in data.items() if d is None]
    if missing:
        print(f"no evaluation found for: {', '.join(missing)}")
        return 1

    for sfx, label in (("post", "POST-clip (iterative 3-sigma)"),
                       ("pre", "PRE-clip (tail-inclusive)")):
        print(f"\n{'='*104}\n{label} — arm / truth-KF, |eta| <= 2, deployment settings\n{'='*104}")
        print(f"{'dataset':<20}{'arm':<24}" + "".join(f"{p:>10}" for p in PARAMS)
              + f"{'GM5':>9}{'N':>11}")
        for s in SETS:
            first = True
            for a in arms:
                e = data[a].get(s)
                r = ratios(e, sfx) if e else None
                if not r:
                    continue
                print(f"{(s.replace('single_muon_', 'mu ') if first else ''):<20}"
                      f"{a:<24}" + "".join(f"{r[p]:>10.3f}" for p in PARAMS)
                      + f"{gm(r):>9.3f}{e.get('n_dm', 0):>11,}")
                first = False
            print()

    if len(arms) == 2:
        a, b = arms
        print(f"{'='*104}\nCHANGE from {a} to {b}  (negative = better after)\n{'='*104}")
        print(f"{'dataset':<22}{'clip':<7}" + "".join(f"{p:>10}" for p in PARAMS) + f"{'GM5':>9}")
        for s in SETS:
            for sfx in ("post", "pre"):
                ea, eb = data[a].get(s), data[b].get(s)
                ra, rb = (ratios(ea, sfx) if ea else None), (ratios(eb, sfx) if eb else None)
                if not ra or not rb:
                    continue
                print(f"{(s.replace('single_muon_', 'mu ') if sfx == 'post' else ''):<22}"
                      f"{sfx:<7}"
                      + "".join(f"{100*(rb[p]/ra[p]-1):>9.1f}%" for p in PARAMS)
                      + f"{100*(gm(rb)/gm(ra)-1):>8.1f}%")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["V2_mingru_25ep", "V2_minGRU_FT50"]))
