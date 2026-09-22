"""Compare two fast_rms_eval bundles cell by cell: SSM/reference ratios (post- and
pre-clip, five parameters, every test set), N identical or not, worst |delta|.

    python scripts/compare_rms_summary.py <ref_dir_or_json> <new_dir_or_json> [--tol 0.0005]

Used as the physics gate of the inference-precision variants
(docs/PRECISION_STUDY_2026-09-21.md): the bar is agreement to the third decimal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PARAMS = ("d0", "z0", "phi", "theta", "qop")


def load(p: str) -> dict:
    path = Path(p)
    if path.is_dir():
        path = path / "plots" / "rms_summary.json" if (path / "plots").is_dir() else path / "rms_summary.json"
    return json.loads(path.read_text())


def ratios(e: dict, sfx: str) -> list[float]:
    # ``ckf_`` is fast_rms_eval's historical prefix for the reference fit (truth-KF here).
    return [e[f"{p}_ssm_{sfx}"] / e[f"{p}_ckf_{sfx}"] for p in PARAMS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=0.0005, help="|delta ratio| above which a cell is flagged")
    a = ap.parse_args()
    ref, new = load(a.ref), load(a.new)
    worst, flagged, n_ok, cells, same3 = 0.0, [], True, 0, 0
    print(f"{'set':22s} {'clip':4s} " + " ".join(f"{p:>16s}" for p in PARAMS) + "   N_ref / N_new")
    for ds in sorted(set(ref) & set(new)):
        r, n = ref[ds], new[ds]
        same_n = r["n_dm"] == n["n_dm"] and r["n_total"] == n["n_total"]
        n_ok &= same_n
        for sfx in ("post", "pre"):
            rr, nn = ratios(r, sfx), ratios(n, sfx)
            row = []
            for p, x, y in zip(PARAMS, rr, nn):
                d = y - x; worst = max(worst, abs(d)); cells += 1
                same3 += round(x, 3) == round(y, 3)      # the paper prints three decimals
                if abs(d) > a.tol:
                    flagged.append((ds, sfx, p, x, y))
                row.append(f"{x:.4f}->{y:.4f}")
            print(f"{ds:22s} {sfx:4s} " + " ".join(f"{c:>16s}" for c in row)
                  + f"   {r['n_dm']} / {n['n_dm']}" + ("" if same_n else "  N DIFFERS"))
    missing = (set(ref) ^ set(new))
    print(f"\ncells compared: {cells}   worst |delta ratio| = {worst:.5f}   "
          f"cells above {a.tol}: {len(flagged)}   identical at 3 decimals: {same3}/{cells}   "
          f"N identical on all sets: {n_ok}"
          + (f"   sets only on one side: {sorted(missing)}" if missing else ""))
    for ds, sfx, p, x, y in flagged:
        print(f"   {ds} {sfx} {p}: {x:.4f} -> {y:.4f} (delta {y - x:+.4f})")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
